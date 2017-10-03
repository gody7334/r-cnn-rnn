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

class Forward(object):

    def __init__(self, mode, data):

        assert mode in ["train", "eval", "inference"], "mode should be 'train', 'eval', or 'inference"
        self.config = global_config.global_config
        self.mode = mode
        self.train_inception = global_config.parse_args.train_inception

        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)
            
        self.random_normal_initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0)
      
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

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_smooth_l1_losses = None  # Used in evaluation.

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_smooth_l1_losses_weights = None  # Used in evaluation.
        
        #
        self.global_step = None

        self.inception_output = None
        self.inception_end_points = None
        self.initial_state = None
        self.sequence_length = None
        self.lstm_outputs = None
        self.logits = None
        self.targets = None
        self.weights = None
        
        self.setup_global_step()
        self.build_image_embeddings()
        self.build_seq_embeddings()
        self.build_model()

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def build_image_embeddings(self):
        """Builds the image model subgraph and generates image embeddings.
           pass image into inceptionV3 and get image features map (add full connected layer at the end)
        Inputs:
          self.images

        Outputs:
          self.image_embeddings
        """
        inception_output, inception_end_points = image_embedding.inception_v3(
            self.images,
            trainable=self.train_inception,
            is_training=self.is_training())
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
            
        self.inception_output = inception_output
        self.inception_end_points = inception_end_points
        
        # TOOD experiement different layer
        # post pocessing different layer output before fc layer
        # inception_mixed_6e_conv = slim.conv2d(inception_end_points['Mixed_6e'], 
        #     768, [3, 3], trainable=True, weights_initializer=self.initializer,
        #     scope="inception/mixed_6e/conv/3x3")
        
        # use conv2d 1*1 to adjust to embeddeding_size
        # image_embeddings = slim.conv2d(inception_end_points['Mixed_6e'], 
        #     self.config.image_embedding_depth, [1, 1], trainable=True, weights_initializer=self.initializer,
        #     scope="image_embedding")
        
        # reduce dimension using fc, later lstm layer 
        # also apply linear transformation to get initial 'h' and 'c'
        # Map inception output into embedding space.
        inception_output = tf.reshape(inception_end_points['MaxPool_3a_3x3'], [self.config.batch_size, -1])
        with tf.variable_scope("image_embedding") as scope:
          image_embeddings = tf.contrib.layers.fully_connected(
              inputs=inception_output,
              num_outputs=self.config.embedding_size,
              activation_fn=None,
              weights_initializer=self.initializer,
              biases_initializer=None,
              scope=scope)

        # Save the embedding size in the graph.
        tf.constant(self.config.embedding_size, name="embedding_size")

        self.image_embeddings = image_embeddings

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

    def build_model(self):
        """Builds the model.

        Inputs:
          self.image_embeddings
          self.seq_embeddings
          self.target_seqs (training and eval only)
          self.input_mask (training and eval only)

        Outputs:
          self.total_loss (training and eval only)
          self.target_cross_entropy_losses (training and eval only)
          self.target_cross_entropy_loss_weights (training and eval only)
        """
        # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
        # modified LSTM in the "Show and Tell" paper has no biases and outputs
        # new_c * sigmoid(o).
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_lstm_units, state_is_tuple=True)
        # ipdb.set_trace()

        if self.mode == "train":
          lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
              input_keep_prob=self.config.lstm_dropout_keep_prob,
              output_keep_prob=self.config.lstm_dropout_keep_prob)
        # ipdb.set_trace()

        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
          # Feed the image embeddings to set the initial LSTM state.
          # Initial LSTM variables using image features map
          zero_state = lstm_cell.zero_state(
              batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
          _, initial_state = lstm_cell(self.image_embeddings, zero_state)
          
          self.initial_state = initial_state

          # Allow the LSTM variables to be reused.
          # then reused them for rest of LSTM operation as variables will update in each recursive step
          lstm_scope.reuse_variables()

          if self.mode == "inference":
            # In inference mode, use concatenated states for convenient feeding and
            # fetching.
            tf.concat(axis=1, values=initial_state, name="initial_state")

            # Placeholder for feeding a batch of concatenated states.
            state_feed = tf.placeholder(dtype=tf.float32,
                                        shape=[None, sum(lstm_cell.state_size)],
                                        name="state_feed")
            state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

            # Run a single LSTM step.
            lstm_outputs, state_tuple = lstm_cell(
                inputs=tf.squeeze(self.seq_embeddings, axis=[1]),
                state=state_tuple)

            # Concatentate the resulting state.
            tf.concat(axis=1, values=state_tuple, name="state")
          else:
            # Run the batch of sequence embeddings through the LSTM.
            sequence_length = tf.reduce_mean(tf.reduce_sum(self.input_mask, 1),1)
            self.sequence_length = sequence_length
            
            lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                inputs=self.seq_embeddings,
                                                sequence_length=sequence_length,
                                                initial_state=initial_state,
                                                dtype=tf.float32,
                                                scope=lstm_scope)
            
        # lstm_outputs=_debug_func(lstm_outputs,'lstm_outputs',break_point=False)
        self.lstm_outputs = lstm_outputs
        
        # get lstm output shape (batch_size, length of bboxes per image, lstm output dim)
        lstm_outputs_shape = tf.shape(lstm_outputs)

        # Stack batches vertically.
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])
        # lstm_outputs=_debug_func(lstm_outputs,'lstm_outputs',break_point=False)

        # reduce dimension into 4
        with tf.variable_scope("logits") as logits_scope:
          logits = tf.contrib.layers.fully_connected(
              inputs=lstm_outputs,
              num_outputs=4,
              activation_fn=None,
              weights_initializer=self.random_normal_initializer,
              scope=logits_scope)

        # keep first 2 dim and assign fully connected dim - 4 to the last dim
        logits_origin_shape = tf.concat(axis=0, values=[tf.cast(tf.slice(lstm_outputs_shape,[0],[2]),tf.int32), tf.constant(4,tf.int32,[1])])
        # reshape to origin shape for drawing bbox function
        logits_origin = tf.reshape(logits, logits_origin_shape)
        image_processing.draw_tf_bounding_boxes(self.images, logits_origin, name="logits")
        image_processing.draw_tf_bounding_boxes(self.images, self.target_seqs, name="targets")

        if self.mode == "inference":
          tf.nn.softmax(logits, name="softmax")
        else:
          logits = tf.reshape(logits, [-1])
          targets = tf.reshape(self.target_seqs, [-1])
          
          self.logits = logits
          self.targets = self.target_seqs
          
          # as different sentence length, using mask to filter out non sentence loss
          weights = tf.to_float(tf.reshape(self.input_mask, [-1]))
          self.weights = weights
          
          # Compute losses.
          # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
          losses = self._smooth_l1_loss(bbox_pred=logits, bbox_targets=targets)
          
          batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                              tf.reduce_sum(weights), name="batch_loss")
          tf.losses.add_loss(batch_loss)
          total_loss = tf.losses.get_total_loss()

          # Add summaries.
          tf.summary.scalar("losses/batch_loss", batch_loss)
          tf.summary.scalar("losses/total_loss", total_loss)
          for var in tf.trainable_variables():
            tf.summary.histogram("parameters/" + var.op.name, var)

          self.total_loss = total_loss
          self.target_smooth_l1_losses = losses  # Used in evaluation.
          self.target_smooth_l1_losses_weights = weights  # Used in evaluation.
          
    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights=1.0, bbox_outside_weights=1.0, sigma=1.0, dim=[1]):
        '''
        TODO bbox_targets need to resize too
        to align diff pics resized in each batch 
        '''
        
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
        
    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step