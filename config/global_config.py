from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

parse_args = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "/home/ipython/r-cnn-rnn/data/dataset/train-?????-of-00256", "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("inception_checkpoint_file", "/home/ipython/r-cnn-rnn/data/pretrain_model/inception_v3.ckpt", "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("train_dir", "/home/ipython/r-cnn-rnn/data/check_point", "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_inception", False, "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 10, "Frequency at which loss and global step are logged.")

tf.logging.set_verbosity(tf.logging.INFO)

global_config = None
def assign_config():
    global global_config
    global_config = Global_Config()
    

class Global_Config(object):

    def __init__(self):
        # File pattern of sharded TFRecord file containing SequenceExample protos.
        # Must be provided in training and evaluation modes.
        self.input_file_pattern = None

        #training directory.
        self.train_dir = None

        # Image format ("jpeg" or "png").
        self.image_format = "jpeg"

        # Approximate number of values per input shard. Used to ensure sufficient
        # mixing between shards in training.
        self.values_per_input_shard = 2300
        # Minimum number of shards to keep in the input queue.
        self.input_queue_capacity_factor = 2
        # Number of threads for prefetching SequenceExample protos.
        self.num_input_reader_threads = 1

        # Name of the SequenceExample context feature containing image data.
        self.image_feature_name = "image/data"
        # Name of the SequenceExample feature list containing integer captions.
        self.caption_feature_name = "image/caption_ids"

        # Number of unique words in the vocab (plus 1, for <UNK>).
        # The default value is larger than the expected actual vocab size to allow
        # for differences between tokenizer versions used in preprocessing. There is
        # no harm in using a value greater than the actual vocab size, but using a
        # value less than the actual vocab size will result in an error.
        self.vocab_size = 12000

        # Number of threads for image preprocessing. Should be a multiple of 2.
        self.num_preprocess_threads = 4

        # Batch size.
        self.batch_size = 16

        """Sets the default training hyperparameters."""
        # Number of examples per epoch of training data.
        self.num_examples_per_epoch = 586363

        # Optimizer for training the model.
        self.optimizer = "SGD"

        # Learning rate for the initial phase of training.
        self.initial_learning_rate = 2.0
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 8.0

        # Learning rate when fine tuning the Inception v3 parameters.
        self.train_inception_learning_rate = 0.0005

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 5
        
        # File containing an Inception v3 checkpoint to initialize the variables
        # of the Inception model. Must be provided when starting training for the
        # first time.
        self.inception_checkpoint_file = None

        # Dimensions of Inception v3 input images.
        self.image_height = 299
        self.image_width = 299

        # Scale used to initialize model variables.
        self.initializer_scale = 0.08

        # LSTM input and output dimensionality, respectively.
        self.embedding_size = 512
        self.num_lstm_units = 512

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.lstm_dropout_keep_prob = 0.7

        # assign parse argument to configuration object
        self.assign_global_config();

    def assign_global_config(self):
        assert parse_args.train_dir, "--train_dir is required"
        # Create training directory.
        self.train_dir=parse_args.train_dir
        if not tf.gfile.IsDirectory(self.train_dir):
            tf.logging.info("Creating training directory: %s", train_dir)
            tf.gfile.MakeDirs(train_dir)

        assert parse_args.input_file_pattern, "--input_file_pattern is required"
        self.input_file_pattern=parse_args.input_file_pattern
        
        assert parse_args.inception_checkpoint_file, "--inception_checkpoint_file is required"
        self.inception_checkpoint_file=parse_args.inception_checkpoint_file