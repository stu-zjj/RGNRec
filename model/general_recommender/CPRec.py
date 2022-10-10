"""
Reference: Steffen Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
    GMF: Xiangnan He et al., "Neural Collaborative Filtering." in WWW 2017.
@author: wubin
"""
import tensorflow as tf
import numpy as np
from time import time
from util import tool
from model.AbstractRecommender import OwnerAbstractRecommender
from util import timer
from util import l2_loss
from util.tool import csr_to_user_dict
from util.cython.random_choice import batch_randint_choice
from util.data_iterator import DataIterator


class cp2(OwnerAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(cp2, self).__init__(dataset, conf)
        self.learning_rate = conf["learning_rate"]
        self.embedding_size = conf["embedding_size"]
        self.num_epochs = conf["epochs"]
        self.reg_mf = conf["reg_mf"]
        self.batch_size = conf["batch_size"]
        self.verbose = conf["verbose"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_dict = csr_to_user_dict(dataset.train_matrix)
        self.sess = sess

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None], name="item_input")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None], name="item_input_neg")
            self.owner_input = tf.placeholder(tf.int32, shape=[None], name="owner_input")
            self.owner_input_neg = tf.placeholder(tf.int32, shape=[None], name="owner_input_neg")
            self.user_ph = tf.placeholder(tf.int32, [None], name="user")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            initializer = tool.get_initializer(self.init_method, self.stddev)

            self.user_embeddings = tf.Variable(initializer([self.num_users, self.embedding_size]),
                                               name='user_embeddings', dtype=tf.float32)  # (users, embedding_size)
            self.item_embeddings = tf.Variable(initializer([self.num_items, self.embedding_size]),
                                               name='item_embeddings', dtype=tf.float32)  # (items, embedding_size)

            self.weight_W_c = tf.Variable(initializer([self.embedding_size, self.embedding_size]),
                                          name='wc', dtype=tf.float32)

            self.weight_W_p = tf.Variable(initializer([self.embedding_size, self.embedding_size]),
                                          name='wp', dtype=tf.float32)

            self.user_embeddings_c = tf.matmul(self.user_embeddings, self.weight_W_c)

            self.user_embeddings_p = tf.matmul(self.user_embeddings, self.weight_W_p)

    def _create_inference(self, item_input, owner_input):
        with tf.name_scope("inference"):
            # embedding look up
            user_embedding_c = tf.nn.embedding_lookup(self.user_embeddings_c, self.user_input)
            item_embedding = tf.nn.embedding_lookup(self.item_embeddings, item_input)
            user_embedding_p = tf.nn.embedding_lookup(self.user_embeddings_p, owner_input)
            predict = tf.reduce_sum(
                tf.multiply(user_embedding_c, item_embedding) + tf.multiply(user_embedding_c, user_embedding_p), 1)
            return user_embedding_c, item_embedding, user_embedding_p, predict

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            uc, i_p, up_p, self.output = self._create_inference(self.item_input, self.owner_input)
            uc, i_n, up_n, self.output_neg = self._create_inference(self.item_input_neg, self.owner_input_neg)
            self.loss = tf.reduce_mean(tf.nn.softplus(-self.output + self.output_neg)) + self.reg_mf * l2_loss(uc, i_p,
                                                                                                               up_p,
                                                                                                               i_n,
                                                                                                               up_n)

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        # for the testing phase
        self.item_embeddings_final = tf.Variable(tf.zeros([self.num_items, self.embedding_size]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.user_embeddings_final = tf.Variable(tf.zeros([self.num_users, self.embedding_size]),
                                                 dtype=tf.float32, name="user_embeddings_final", trainable=False)
        self.owner_embeddings_final = tf.Variable(tf.zeros([self.num_users, self.embedding_size]),
                                                 dtype=tf.float32, name="owner_embeddings_final", trainable=False)

        self.assign_opt = [tf.assign(self.user_embeddings_final, self.user_embeddings_c),
                           tf.assign(self.item_embeddings_final, self.item_embeddings),
                           tf.assign(self.owner_embeddings_final, self.user_embeddings_p)]

        u_embed = tf.nn.embedding_lookup(self.user_embeddings_final, self.user_ph)
        o_embed = tf.nn.embedding_lookup(self.owner_embeddings_final, self.all_ownerIDs[:])
        self.batch_ratings = tf.matmul(u_embed, self.item_embeddings_final, transpose_a=False, transpose_b=True) + tf.matmul(u_embed,o_embed, transpose_a=False, transpose_b=True)

    def _generate_sequences(self):
        users_list, item_pos_list, owner_pos_list = [], [], []
        for user_id in range(self.num_users):
            items_by_userid = self.dataset.train_matrix[user_id].indices
            for item_id in items_by_userid:
                users_list.append(user_id)
                item_pos_list.append(item_id)
                owner_id = self.all_ownerIDs[item_id]
                owner_pos_list.append(owner_id)
        return users_list, item_pos_list, owner_pos_list

    def _sample_negative(self, users_list):
        neg_items_list, neg_owners_list = [], []
        user_neg_items_dict, user_neg_owners_dict = {}, {}
        all_uni_user, all_counts = np.unique(users_list, return_counts=True)
        user_count = DataIterator(all_uni_user, all_counts, batch_size=1024, shuffle=False)
        for bat_users, bat_counts in user_count:
            exclusion = [self.train_dict[u] for u in bat_users]
            bat_neg = batch_randint_choice(self.num_items, bat_counts, replace=True, exclusion=exclusion)
            for userid, neg_itemids in zip(bat_users, bat_neg):
                user_neg_items_dict[userid] = neg_itemids
                user_neg_owners_dict[userid] = self.all_ownerIDs[neg_itemids]
        for u, c in zip(all_uni_user, all_counts):
            neg_items = np.reshape(user_neg_items_dict[u], newshape=[c])
            neg_owners = np.reshape(user_neg_owners_dict[u], newshape=[c])
            neg_items_list.extend(neg_items)
            neg_owners_list.extend(neg_owners)
        return neg_items_list, neg_owners_list

    # ---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        users_list, item_pos_list, owner_pos_list = self._generate_sequences()
        for epoch in range(1, self.num_epochs + 1):
            total_loss = 0.0
            training_start_time = time()
            item_neg_list, owner_neg_list = self._sample_negative(users_list)
            data_iter = DataIterator(users_list, item_pos_list, item_neg_list, owner_pos_list, owner_neg_list,
                                     batch_size=self.batch_size, shuffle=True)
            for bat_users, bat_items_pos, bat_items_neg, bat_owner_pos, bat_owner_neg in data_iter:
                feed_dict = {self.user_input: bat_users,
                             self.item_input: bat_items_pos,
                             self.item_input_neg: bat_items_neg,
                             self.owner_input: bat_owner_pos,
                             self.owner_input_neg: bat_owner_neg,
                             }
                loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                total_loss += loss
            self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss / len(data_iter),
                                                                  time() - training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))

    @timer
    def evaluate(self):
        self.sess.run(self.assign_opt)
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items=None):
        if candidate_items is None:
            ratings = self.sess.run(self.batch_ratings, feed_dict={self.user_ph: user_ids})

        return ratings
