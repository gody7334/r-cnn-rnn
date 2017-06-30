from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import _init_paths
from control.train import Train

def main(unused_argv):
    Train().run()

if __name__ == "__main__":
    tf.app.run()
