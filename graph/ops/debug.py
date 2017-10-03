# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from pprint import pprint as pp
from time import sleep
import random
import ipdb

def _debug_func(tensor, name="", print_tf=False, print_op=False, break_point=False):
  t = tensor
  debug_op = tf.py_func(debug_func, [t, t.name, str(t.op), str(t.dtype), t.device, print_tf, print_op, break_point, name], [tf.bool])

  with tf.control_dependencies(debug_op):
    tensor = tf.identity(tensor, name=name)

  return tensor

def debug_func(tf, tf_name, tf_op, tf_type, tf_device, print_tf, print_op, break_point, name):
  sleep((random.randint(0, 500) / 1000))
  print('name: {}'.format(name))
  print('tf_name: {}'.format(tf_name))
  print('tf_shape: {} '.format(tf.shape))
  print('tf_type: {} '.format(tf_type))
  print('tf_device: {} '.format(tf_device))
  print('')

  if print_tf:
    np.set_printoptions(threshold=50)
    print('tf_element: ')
    pp(tf)
  if print_op:
    print('tf_op: ')
    print(tf_op)
  if break_point:
    # name, tf_name, tf_shape, tf_type, tf_device, tf_op, tf
    ipdb.set_trace()  # BREAKPOINT

  return False
