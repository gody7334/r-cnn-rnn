from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim
import ipdb

from utils.config import global_config
from graph.ops import image_embedding
from graph.ops import image_processing
from graph.ops import inputs as input_ops
from graph.ops.debug import _debug_func

class IForward(object):
    
    def __init__(self, mode, data):
        assert mode in ["train", "eval", "inference"], "mode should be 'train', 'eval', or 'inference"
        self.config = global_config.global_config
        self.mode = mode
        
        self.train_inception = global_config.parse_args.train_inception
        
        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.images = data.images

        # An int32 Tensor with shape [batch_size, padded_length].
        self.input_seqs = data.input_seqs

        # An int32 Tensor with shape [batch_size, padded_length].
        self.target_seqs = data.target_seqs

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = data.input_mask

        # A float32 Tensor with shape [batch_size, embedding_size].
        self.image_embeddings = None

        # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
        self.seq_embeddings = None

        # initializer
        self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)
            
        self.random_normal_initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0)

        # 
        self.global_step = None
        
        self.setup_global_step()
        self.build_seq_embeddings()
    
    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step
    
    def build_seq_embeddings(self):
        """Builds the input sequence embeddings.
           pass words (sentence) and get word embeddings (words features)
        Inputs:
          self.input_seqs

        Outputs:
          self.seq_embeddings
          input_seqs N x T (batch size x sentence(words))
          seq_embedding N x T x D (expand each word to D dimension word vector)
          !!!!!!! in tensorflow 1.1 lstm_cell, dimension D (input's dimension) 
          !!!!!!! must equal to lstm 'num_units' 
          (potential issue cause by initial basic lstm cell's weight dimension)
        """
        # Therefore, using linear transformation to expand input dimension.
        # Although lstm will do it again with _linear() function........
        self.input_seqs.set_shape([None, None, 4]) # N T D
        with tf.variable_scope("seq_embedding") as scope:
            w_h = tf.get_variable('w_h', [4, self.config.num_lstm_units], initializer=self.initializer)
            b_h = tf.get_variable('b_h', [self.config.num_lstm_units], initializer=tf.constant_initializer(0.0))
            input_seqs = tf.reshape(self.input_seqs,[-1,4])
            seq_embeddings = tf.matmul(input_seqs, w_h) + b_h
            seq_embeddings = tf.reshape(seq_embeddings, [self.config.batch_size,-1,self.config.num_lstm_units])
            
        # with tf.variable_scope("seq_embedding"):
        #   embedding_map = tf.get_variable(
        #       name="map",
        #       shape=[self.config.vocab_size, self.config.embedding_size],
        #       initializer=self.initializer)
        #   seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

        self.seq_embeddings = seq_embeddings
        
    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights=1.0, bbox_outside_weights=1.0, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        # stop_gradient used to not compute gradient in back prap,
        # as its only used to compute smoothL1 sign, not in the computation graph for optimization
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        
        # loss_box = tf.reduce_mean(tf.reduce_sum(
        #   out_loss_box,
        #   axis=dim
        # ))
        # return loss_box
        
        # donnot sum loss yet
        return out_loss_box