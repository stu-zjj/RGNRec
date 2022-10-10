import numpy as np
from util import timer
import tensorflow as tf
from model.AbstractRecommender import UIOAbstractRecommender
from util.tool import l2_loss
import scipy.sparse as sp


class RGNRec(UIOAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(RGNRec, self).__init__(dataset, conf)
        self.embedding_size = int(conf["embedding_size"])
        self.r_alpha1 = float(conf["r_alpha1"])
        self.r_alpha2 = float(conf["r_alpha2"])
        self.num_epochs = int(conf["epochs"])
        self.reg = float(conf["reg"])
        self.reg_w = float(conf["reg_w"])
        self.beta = float(conf["beta"])
        self.lr = float(conf["learning_rate"])
        self.layer_size = int(conf["layer_size"])
        self.layeruo_size = int(conf["layeruo_size"])
        self.verbose = int(conf["verbose"])
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.sess = sess

    def _create_recsys_adj_mat(self):
        user_item_idx = [[u, i] for (u, i), r in self.dataset.train_matrix.todok().items()]
        user_list, item_list = list(zip(*user_item_idx))

        self.user_idx = tf.constant(user_list, dtype=tf.int32, shape=None, name="user_idx")
        self.item_idx = tf.constant(item_list, dtype=tf.int32, shape=None, name="item_idx")

        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(n_nodes, n_nodes))  # (m+n)*(m+n)
        adj_mat = tmp_adj + tmp_adj.T

        return self._normalize_spmat(adj_mat)

    def _create_owner_adj_mat(self):
        user_owner_idx = [[u, o] for (u, o), r in self.social_matrix.todok().items()]
        user_list, owner_list = list(zip(*user_owner_idx))

        self.user_idx = tf.constant(user_list, dtype=tf.int32, shape=None, name="user_idx")
        self.owner_idx = tf.constant(owner_list, dtype=tf.int32, shape=None, name="owner_idx")

        user_np = np.array(user_list, dtype=np.int32)
        owner_np = np.array(owner_list, dtype=np.int32)
        values = np.ones_like(user_np, dtype=np.float32)
        n_nodes = 2*self.num_users
        tmp_adj = sp.csr_matrix((values, (user_np, owner_np+self.num_users)), shape=(n_nodes, n_nodes))  # (m+n)*(m+n)
        adj_mat = tmp_adj + tmp_adj.T
        return self._normalize_spmat(adj_mat)

    def _normalize_spmat(self, adj_mat):
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        print('use the pre adjcency matrix')
        return adj_matrix

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_placeholder(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")

    def _create_variables(self):
        initializer = tf.contrib.layers.xavier_initializer()
        self.user_embeddings = tf.Variable(initializer([self.num_users, self.embedding_size]), name='user_embeddings')
        self.item_embeddings = tf.Variable(initializer([self.num_items, self.embedding_size]), name='item_embeddings')
        weight_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        # uu_weight project user embeddings into social space
        self.uu_weight = tf.Variable(weight_initializer([self.embedding_size, self.embedding_size]), name='uu_weight')
        # ui_weight project user embeddings into recommendation space
        self.ui_weight = tf.Variable(weight_initializer([self.embedding_size, self.embedding_size]), name='ui_weight')

    def _gcn(self, norm_adj, init_embeddings, layer):
        ego_embeddings = init_embeddings
        all_embeddings = [ego_embeddings]
        for k in range(layer):
            ego_embeddings = tf.sparse_tensor_dense_matmul(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        return all_embeddings

    def _owner_gcn(self):
        norm_adj = self._create_owner_adj_mat()
        norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj)
        owner_embeddings = tf.matmul(self.user_embeddings, self.uu_weight)
        user_embeddings = tf.matmul(self.user_embeddings, self.ui_weight)

        ego_embeddings = tf.concat([user_embeddings, owner_embeddings], axis=0)
        all_embeddings = self._gcn(norm_adj, ego_embeddings,self.layeruo_size)
        user_embeddings, owner_embeddings = tf.split(all_embeddings, [self.num_users, self.num_users], 0)
        return user_embeddings, owner_embeddings

    def _recsys_gcn(self):
        norm_adj = self._create_recsys_adj_mat()
        norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj)
        user_embeddings = tf.matmul(self.user_embeddings, self.ui_weight)

        ego_embeddings = tf.concat([user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = self._gcn(norm_adj, ego_embeddings,self.layer_size)
        user_embeddings, item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)
        return user_embeddings, item_embeddings

    def _fast_lossui(self, user_embeddings, item_embeddings, user_idx, item_idx, alpha):
        term1 = tf.matmul(user_embeddings, user_embeddings, transpose_a=True)
        term2 = tf.matmul(item_embeddings, item_embeddings, transpose_a=True)
        lossui = tf.reduce_sum(term1*term2, axis=-1)

        embed_a = tf.nn.embedding_lookup(user_embeddings, user_idx)
        embed_b = tf.nn.embedding_lookup(item_embeddings, item_idx)
        ui_ratings = tf.reduce_sum(embed_a*embed_b, axis=-1)

        loss1 = (alpha-1)*tf.reduce_sum(tf.square(ui_ratings)) - 2.0*alpha*tf.reduce_sum(ui_ratings)

        return loss1 + lossui

    def _fast_lossuo(self, user_embeddings, owner_embeddings, user_idx, item_idx, alpha):
        term1 = tf.matmul(user_embeddings, user_embeddings, transpose_a=True)
        term2 = tf.matmul(owner_embeddings, owner_embeddings, transpose_a=True)
        lossuo = tf.reduce_sum(term1 * term2, axis=-1)

        embed_a = tf.nn.embedding_lookup(user_embeddings, user_idx)
        embed_b = tf.nn.embedding_lookup(owner_embeddings, item_idx)
        uo_ratings = tf.reduce_sum(embed_a * embed_b, axis=-1)

        loss2 = (alpha - 1) * tf.reduce_sum(tf.square(uo_ratings)) - 2.0 * alpha * tf.reduce_sum(uo_ratings)

        return loss2 + lossuo

    def build_graph(self):
        # ---------- matrix factorization -------
        self._create_placeholder()
        self._create_variables()
        social_user_embeddings, self.owner_embeddings = self._owner_gcn()
        item_user_embeddings, self.final_item_embeddings = self._recsys_gcn()

        self.final_user_embeddings = tf.divide(item_user_embeddings + social_user_embeddings, 2)

        self.item_owner_embeddings = tf.gather(self.owner_embeddings, self.all_ownerIDs[:], axis=0)
        recsys_loss = self._fast_lossui(self.final_user_embeddings, self.final_item_embeddings, self.user_idx, self.item_idx, self.r_alpha1) + \
                      self.beta * self._fast_lossuo(self.final_user_embeddings, self.item_owner_embeddings, self.user_idx, self.item_idx, self.r_alpha2)

        self.obj_loss = recsys_loss + self.reg * l2_loss(self.user_embeddings, self.item_embeddings) + \
                        self.reg_w * l2_loss(self.ui_weight, self.uu_weight)

        self.update_opt = tf.train.AdagradOptimizer(self.lr).minimize(self.obj_loss)

        # for the testing phase
        self.item_embeddings_final = tf.Variable(tf.zeros([self.num_items, self.embedding_size]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.user_embeddings_final = tf.Variable(tf.zeros([self.num_users, self.embedding_size]),
                                                 dtype=tf.float32, name="user_embeddings_final", trainable=False)
        self.owner_embeddings_final = tf.Variable(tf.zeros([self.num_items, self.embedding_size]),
                                                 dtype=tf.float32, name="owner_embeddings_final", trainable=False)

        self.assign_opt = [tf.assign(self.user_embeddings_final, self.final_user_embeddings),
                           tf.assign(self.item_embeddings_final, self.final_item_embeddings),
                           tf.assign(self.owner_embeddings_final, self.item_owner_embeddings)]

        u_embed = tf.nn.embedding_lookup(self.user_embeddings_final, self.user_ph)
        self.batch_ratings = tf.matmul(u_embed, self.item_embeddings_final, transpose_a=False, transpose_b=True)+ \
                             tf.matmul(u_embed, self.owner_embeddings_final, transpose_a=False, transpose_b=True)



    # ---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.num_epochs):
            _, _ = self.sess.run([self.update_opt, self.obj_loss])
            if epoch >=1:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        self.sess.run(self.assign_opt)
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items=None):
        if candidate_items is None:
            ratings = self.sess.run(self.batch_ratings, feed_dict={self.user_ph: user_ids})

        return ratings
