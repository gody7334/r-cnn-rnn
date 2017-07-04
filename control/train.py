from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# get parsing argument
from config import global_config
from graph.build_data import Build_Data
from graph.forward import Forward
from graph.backward import Backward

class Train(object):

    def __init__(self):
        # inception inital func
        self.init_fn = None
        self.saver = None
        self.optimize = None
        self.g = None
        self.mode = 'train'

    def run(self):
        global_config.assign_config()
        self.build_computation_graph()
        
        self.run_training()

    def build_computation_graph(self):
        # Build the TensorFlow graph.
        g = tf.Graph()
        with g.as_default():
            data = Build_Data(mode='train')
            model = Forward(mode='train',data=data)
            optimize = Backward(model = model)
            self.setup_inception_initializer()

            # Set up the Saver for saving and restoring model checkpoints.
            saver = tf.train.Saver(max_to_keep=global_config.global_config.max_checkpoints_to_keep)

            self.optimize = optimize
            self.saver = saver
        self.g = g

    def setup_inception_initializer(self):
        """Sets up the function to restore inception variables from checkpoint."""
        if self.mode != "inference":
            # Restore inception variables only.
            inception_variables = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")
            saver = tf.train.Saver(inception_variables)

            def restore_fn(sess):
                tf.logging.info("Restoring Inception variables from checkpoint file %s",
                                global_config.global_config.inception_checkpoint_file)
                saver.restore(sess, global_config.global_config.inception_checkpoint_file)

            self.init_fn = restore_fn

    def run_training(self):
        # Run training.
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth=True
        tf.contrib.slim.learning.train(
            self.optimize.train_op,
            global_config.global_config.train_dir,
            log_every_n_steps=global_config.parse_args.log_every_n_steps,
            graph=self.g,
            global_step=self.optimize.global_step,
            number_of_steps=global_config.parse_args.number_of_steps,
            init_fn=self.init_fn,
            saver=self.saver,
            session_config=sess_config)
