from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from config import global_config
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
        self.target_cross_entropy_losses = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_loss_weights = None

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
        inception_output = image_embedding.inception_v3(
            self.images,
            trainable=self.train_inception,
            is_training=self.is_training())
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

        # Map inception output into embedding space.
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
        """
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
          embedding_map = tf.get_variable(
              name="map",
              shape=[self.config.vocab_size, self.config.embedding_size],
              initializer=self.initializer)
          seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

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
        if self.mode == "train":
          lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
              input_keep_prob=self.config.lstm_dropout_keep_prob,
              output_keep_prob=self.config.lstm_dropout_keep_prob)

        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
          # Feed the image embeddings to set the initial LSTM state.
          # Initial LSTM variables using image features map
          zero_state = lstm_cell.zero_state(
              batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
          _, initial_state = lstm_cell(self.image_embeddings, zero_state)

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
            sequence_length = tf.reduce_sum(self.input_mask, 1)
            lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                inputs=self.seq_embeddings,
                                                sequence_length=sequence_length,
                                                initial_state=initial_state,
                                                dtype=tf.float32,
                                                scope=lstm_scope)

        # Stack batches vertically.
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

        with tf.variable_scope("logits") as logits_scope:
          logits = tf.contrib.layers.fully_connected(
              inputs=lstm_outputs,
              num_outputs=self.config.vocab_size,
              activation_fn=None,
              weights_initializer=self.initializer,
              scope=logits_scope)

        if self.mode == "inference":
          tf.nn.softmax(logits, name="softmax")
        else:

          targets = tf.reshape(self.target_seqs, [-1])
          # as different sentence length, using mask to filter out non sentence loss
          weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

          # Compute losses.
          losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                  logits=logits)
          batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                              tf.reduce_sum(weights),
                              name="batch_loss")
          tf.losses.add_loss(batch_loss)
          total_loss = tf.losses.get_total_loss()

          # Add summaries.
          tf.summary.scalar("losses/batch_loss", batch_loss)
          tf.summary.scalar("losses/total_loss", total_loss)
          for var in tf.trainable_variables():
            tf.summary.histogram("parameters/" + var.op.name, var)

          self.total_loss = total_loss
          self.target_cross_entropy_losses = losses  # Used in evaluation.
          self.target_cross_entropy_loss_weights = weights  # Used in evaluation.
