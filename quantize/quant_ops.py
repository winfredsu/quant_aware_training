# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Python support for quantization operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import model_variable
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages

import tensorflow as tf

@add_arg_scope
def FixedQuantize(inputs, init_min=-6.0, init_max=6.0, scope=None):
  """Adds a fake quantize layer with fixed quantization interval.

  Args:
    inputs: a tensor containing values to be quantized.
    init_min: the lower end of quantization interval.
    init_max: the upper end of quantization interval.
    scope: Optional scope for name_scope.
  Returns:
    a tensor containing quantized values.
  """
  with ops.name_scope(scope, 'FixedQuantize', values=[inputs]):
    return array_ops.fake_quant_with_min_max_args(
        inputs, min=init_min, max=init_max)


@add_arg_scope
def LastValueQuantize(inputs,
                      per_channel=False,
                      init_min=-6.0,
                      init_max=6.0,
                      #vars_collection=ops.GraphKeys.MOVING_AVERAGE_VARIABLES,
                      vars_collection=None,
                      name_prefix='LastValueQuant',
                      reuse=None,
                      is_training=True,
                      num_bits=8,
                      narrow_range=False):
  """Adds a layer that collects quantization ranges as last input ranges.

  LastValueQuantize creates variables called 'min' and 'max', representing the
  interval used for quantization and clamping.

  Args:
    inputs: a tensor containing values to be quantized.
    per_channel: (Optional) a boolean specifying whether to use different
      quantization ranges per output channel.
    init_min: a float scalar, the initial value for variable min.
    init_max: a float scalar, the initial value for variable max.
    vars_collection: (Optional) collection where to store variables for
      quantization interval ends.
    name_prefix: name_prefix for created nodes.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    is_training: Whether the op is applied to a training or eval graph.
    num_bits: Number of bits to use for quantization, must be between 2 and 8.
    narrow_range: Whether to use the narrow quantization range
      [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
  Returns:
    a tensor containing quantized values.
  """
  with variable_scope.variable_scope(
      None, default_name=name_prefix, values=[inputs], reuse=reuse) as scope:
    scope.set_partitioner(None)
    input_shape = inputs.get_shape()
    input_dim = len(input_shape)
    if per_channel:
      # Only support quantizing 1-, 2- and 4-dimensional tensors.
      assert input_dim in [1, 2, 4], ('Expected 1D, 2D or 4D input, was: %s in '
                                      ' scope: %s' % (input_shape, name_prefix))
      min_max_shape = [input_shape[-1]]
    else:
      min_max_shape = []

    min_var = model_variable(
        'min',
        shape=min_max_shape,
        initializer=init_ops.constant_initializer(init_min),
        collections=[vars_collection],
        trainable=False)
    max_var = model_variable(
        'max',
        shape=min_max_shape,
        initializer=init_ops.constant_initializer(init_max),
        collections=[vars_collection],
        trainable=False)

    # modified by sf

    min_po2_var = model_variable(
        'min_po2',
        shape=min_max_shape,
        initializer=init_ops.constant_initializer(init_min),
        collections=[vars_collection],
        trainable=False)
    max_po2_var = model_variable(
        'max_po2',
        shape=min_max_shape,
        initializer=init_ops.constant_initializer(init_max),
        collections=[vars_collection],
        trainable=False)

    if not is_training:
      return _FakeQuantWithMinMaxVars(
          inputs,
          #min_var,
          #max_var,
          min_po2_var,
          max_po2_var,
     # end modified by sf
          per_channel=per_channel,
          num_bits=num_bits,
          narrow_range=narrow_range)

    if per_channel:
      if input_dim == 2:
        reduce_dims = [0]
      elif input_dim == 4:
        reduce_dims = [0, 1, 2]

    if per_channel:
      if input_dim >= 2:
        batch_min = math_ops.reduce_min(
            inputs, reduction_indices=reduce_dims, name='BatchMin')
      else:
        batch_min = inputs
    else:
      batch_min = math_ops.reduce_min(inputs, name='BatchMin')
    # TFLite requires that 0.0 if always in the [min; max] range.

    # modified by sf
    # make sure batch_min is not zero
    # print(batch_min)
    batch_min = math_ops.minimum(batch_min, -0.0000001)
    # end modified by sf

    assign_min = state_ops.assign(min_var, batch_min, name='AssignMinLastBak')

    if per_channel:
      if input_dim >= 2:
        batch_max = math_ops.reduce_max(
            inputs, reduction_indices=reduce_dims, name='BatchMax')
      else:
        batch_max = inputs
    else:
      batch_max = math_ops.reduce_max(inputs, name='BatchMax')
    # TFLite requires that 0.0 if always in the [min; max] range.

    # modified by sf
    # make sure batch_max is not zero
    batch_max = math_ops.maximum(batch_max, 0.0000001)
    # end modified by sf

    assign_max = state_ops.assign(max_var, batch_max, name='AssignMaxLastBak')

    # modified by sf
    min_log = tf.ceil(tf.log(tf.abs(assign_min))/tf.log(2.0))
    max_log = tf.ceil(tf.log(tf.abs(assign_max))/tf.log(2.0))
    minmax_log = math_ops.maximum(min_log,max_log)
    minmax_log = math_ops.minimum(minmax_log,3.0)
    min_po2 = tf.negative(tf.pow(2.0,minmax_log))
    max_po2 = tf.pow(2.0,minmax_log)-tf.pow(2.0,minmax_log+1-num_bits)

    assign_min_po2 = state_ops.assign(min_po2_var,min_po2,name='AssignMinLast')
    assign_max_po2 = state_ops.assign(max_po2_var,max_po2,name='AssignMaxLast')


    return _FakeQuantWithMinMaxVars(
        inputs,
        assign_min_po2,
        assign_max_po2,
        #assign_min,
        #assign_max,
        per_channel=per_channel,
        num_bits=num_bits,
        narrow_range=narrow_range)

    # end modified by sf

@add_arg_scope
def MovingAvgQuantize(inputs,
                      per_channel=False,
                      init_min=-6.0,
                      init_max=6.0,
                      ema_decay=0.999,
                      vars_collection=ops.GraphKeys.MOVING_AVERAGE_VARIABLES,
                      name_prefix='MovingAvgQuantize',
                      reuse=None,
                      is_training=True,
                      num_bits=8,
                      narrow_range=False):
  """Adds a layer that collects quantization ranges as EMAs of input ranges.

  MovingAvgQuantize creates variables called 'min' and 'max', representing the
  interval used for quantization and clamping.

  Args:
    inputs: a tensor containing values to be quantized.
    per_channel: (default False) a boolean specifying whether to use different
      quantization ranges per output channel.
    init_min: a float scalar, the initial value for variable min.
    init_max: a float scalar, the initial value for variable max.
    ema_decay: EMA decay parameter.
    vars_collection: (Optional) collection where to store variables for
      quantization interval ends.
    name_prefix: name_prefix for created nodes.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    is_training: Whether the op is applied to a training or eval graph.
    num_bits: Number of bits to use for quantization, must be between 2 and 8.
    narrow_range: Whether to use the narrow quantization range
      [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
  Returns:
    a tensor containing quantized values.
  """
  with variable_scope.variable_scope(
      None, default_name=name_prefix, values=[inputs], reuse=reuse) as scope:
    scope.set_partitioner(None)
    input_shape = inputs.get_shape()
    input_dim = len(input_shape)
    if per_channel:
      # Only support quantizing 1-, 2- and 4-dimensional tensors.
      assert input_dim in [1, 2, 4], ('Expected 1D, 2D or 4D input, was: %s in '
                                      ' scope: %s' % (input_shape, name_prefix))
      min_max_shape = [input_shape[-1]]
    else:
      min_max_shape = []

    min_var = model_variable(
        'min',
        shape=min_max_shape,
        initializer=init_ops.constant_initializer(init_min),
        collections=[vars_collection],
        trainable=False)
    max_var = model_variable(
        'max',
        shape=min_max_shape,
        initializer=init_ops.constant_initializer(init_max),
        collections=[vars_collection],
        trainable=False)

    # modified by sf

    min_po2_var = model_variable(
        'min_po2',
        shape=min_max_shape,
        initializer=init_ops.constant_initializer(init_min),
        collections=[vars_collection],
        trainable=False)
    max_po2_var = model_variable(
        'max_po2',
        shape=min_max_shape,
        initializer=init_ops.constant_initializer(init_max),
        collections=[vars_collection],
        trainable=False)

    if not is_training:
      return _FakeQuantWithMinMaxVars(
          inputs,
          #min_var,
          #max_var,
          min_po2_var,
          max_po2_var,
          per_channel=per_channel,
          num_bits=num_bits,
          narrow_range=narrow_range)
    # end modified by sf
    if per_channel:
      if input_dim == 2:
        reduce_dims = [0]
      elif input_dim == 4:
        reduce_dims = [0, 1, 2]

    if per_channel:
      if input_dim >= 2:
        batch_min = math_ops.reduce_min(
            inputs, reduction_indices=reduce_dims, name='BatchMin')
      else:
        batch_min = inputs
    else:
      batch_min = math_ops.reduce_min(inputs, name='BatchMin')
    # B-eng requires that 0.0 if always in the [min; max] range.

    # modified by sf
    # ensure batch_min is not 0
    batch_min = math_ops.minimum(batch_min, -0.0000001)
    # end modified by sf

    assign_min = moving_averages.assign_moving_average(
        min_var, batch_min, ema_decay, name='AssignMinEma')

    if per_channel:
      if input_dim >= 2:
        batch_max = math_ops.reduce_max(
            inputs, reduction_indices=reduce_dims, name='BatchMax')
      else:
        batch_max = inputs
    else:
      batch_max = math_ops.reduce_max(inputs, name='BatchMax')
    # B-eng requires that 0.0 if always in the [min; max] range.

    # modified by sf
    # ensure batch_max is not 0
    batch_max = math_ops.maximum(batch_max, 0.0000001)
    # end modified by sf

    assign_max = moving_averages.assign_moving_average(
        max_var, batch_max, ema_decay, name='AssignMaxEma')

    # modified by sf
    min_log = tf.ceil(tf.log(tf.abs(assign_min))/tf.log(2.0))
    max_log = tf.ceil(tf.log(tf.abs(assign_max))/tf.log(2.0))
    minmax_log = math_ops.maximum(min_log,max_log)
    min_po2 = tf.negative(tf.pow(2.0,minmax_log))
    max_po2 = tf.pow(2.0,minmax_log)-tf.pow(2.0,minmax_log-7)

    assign_min_po2 = state_ops.assign(min_po2_var,min_po2,name='AssignMinPo2MovAvg')
    assign_max_po2 = state_ops.assign(max_po2_var,max_po2,name='AssignMaxPo2MovAvg')

    return _FakeQuantWithMinMaxVars(
        inputs,
        #assign_min,
        #assign_max,
        assign_min_po2,
        assign_max_po2,
    # end modified by sf
        per_channel=per_channel,
        num_bits=num_bits,
        narrow_range=narrow_range)


def _FakeQuantWithMinMaxVars(inputs, min_var, max_var, per_channel, num_bits,
                             narrow_range):
  """Adds a fake quantization operation.

  Depending on value of per_channel, this operation may do global quantization
  or per channel quantization.  min_var and max_var should have corresponding
  shapes: [1] when per_channel == False and [d] when per_channel == True.

  Args:
    inputs: a tensor containing values to be quantized.
    min_var: a variable containing quantization range lower end(s).
    max_var: a variable containing quantization range upper end(s).
    per_channel: a boolean specifying whether to use per-channel quantization.
    num_bits: Number of bits to use for quantization, must be between 2 and 8.
    narrow_range: Whether to use the narrow quantization range
      [1; 2^num_bits - 1] or wide range [0; 2^num_bits - 1].
  Returns:
    a tensor containing quantized values.
  """

  if per_channel:
    assert len(min_var.get_shape()) == 1
    assert len(max_var.get_shape()) == 1
    return array_ops.fake_quant_with_min_max_vars_per_channel(
        inputs, min_var, max_var, num_bits=num_bits, narrow_range=narrow_range)
  else:
    assert min_var.get_shape() == []  # pylint: disable=g-explicit-bool-comparison
    assert max_var.get_shape() == []  # pylint: disable=g-explicit-bool-comparison
    return array_ops.fake_quant_with_min_max_vars(
        inputs, min_var, max_var, num_bits=num_bits, narrow_range=narrow_range)
