from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# get parsing argument
from utils.config import global_config
from graph.build_data import Build_Data
from graph.forward.lstm import LSTM
from graph.backward import Backward
from control.prepare_dataset import PrepareDataset

class Train(object):

    def __init__(self):
        # inception inital func
        self.init_fn = None
        self.saver = None
        self.optimize = None
        self.g = None
        self.mode = 'train'
        self.prepared_dataset = None
        
        self.data = None
        self.model = None

    def run(self):
        global_config.assign_config()
#         self.prepared_dataset = PrepareDataset()
        self.build_computation_graph()
        self.run_training()

        # export CUDA_VISIBLE_DEVICES=
        # export CUDA_VISIBLE_DEVICES=0
        # os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # self.run_partial_graph()

    def build_computation_graph(self):
        # Build the TensorFlow graph.
        g = tf.Graph()
        with g.as_default():
            data = Build_Data(mode='train')
            self.data = data
            
            model = LSTM(mode='train',data=data)
            self.model = model
            
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
            global_step=self.model.global_step,
            number_of_steps=global_config.parse_args.number_of_steps,
            save_summaries_secs=60,
            init_fn=self.init_fn,
            saver=self.saver,
            session_config=sess_config)
      
    def run_partial_graph(self):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth=True
        with tf.Session(graph=self.g, config=sess_config) as sess:
            # But now we build our coordinator to coordinate our child threads with
            # the main thread
            coord = tf.train.Coordinator()
            
            # Beware, if you don't start all your queues before runnig anything
            # The main threads will wait for them to start and you will hang again
            # This helper start all queues in tf.GraphKeys.QUEUE_RUNNERS
            threads = tf.train.start_queue_runners(coord=coord)
            
            # init all variables
            sess.run(tf.global_variables_initializer())
            
            # init inception net
            self.init_fn(sess)
            
            import ipdb; ipdb.set_trace()
            # print(sess.run(self.model.input_mask))
            # print(sess.run(self.model.sequence_length))
            # print(sess.run(self.model.lstm_outputs))
            
            print(sess.run(self.model.logits))
            print(sess.run(self.model.targets))
            print(sess.run(self.model.weights))
            # print(sess.run(self.data.target_seqs))
            # print(sess.run(self.data.input_mask))
            
            sess.close()