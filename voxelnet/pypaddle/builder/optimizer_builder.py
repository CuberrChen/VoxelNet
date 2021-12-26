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
"""Functions to build DetectionModel training optimizers."""

import paddle


def build(optimizer_config, lr_sche, params, clip=True, name=None):
    """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    optimizer_type = optimizer_config.optimizer_type
    optimizer = None
    if clip:
        cliper = paddle.nn.ClipGradByNorm(clip_norm=10.0)
    if optimizer_type == 'rms_prop_optimizer':
        config = optimizer_config.rms_prop_optimizer
        optimizer = paddle.optimizer.RMSProp(
            learning_rate=lr_sche,
            rho=config.decay,
            momentum=config.momentum_optimizer_value,
            parameters=params,
            epsilon=config.epsilon,
            weight_decay=config.weight_decay,grad_clip=cliper if clip else None)

    if optimizer_type == 'momentum_optimizer':
        config = optimizer_config.momentum_optimizer
        optimizer = paddle.optimizer.Momentum(
            parameters=params,
            learning_rate=lr_sche,
            momentum=config.momentum_optimizer_value,
            weight_decay=config.weight_decay,grad_clip=cliper if clip else None)

    if optimizer_type == 'adam_optimizer':
        config = optimizer_config.adam_optimizer
        optimizer = paddle.optimizer.Adam(
            parameters=params,
            learning_rate=lr_sche,
            weight_decay=config.weight_decay,grad_clip=cliper if clip else None)

    if optimizer is None:
        raise ValueError('Optimizer %s not supported.' % optimizer_type)

    if optimizer_config.use_moving_average:
        raise ValueError('paddle don\'t support moving average')
    if name is None:
        # assign a name to optimizer for checkpoint system
        optimizer.name = optimizer_type
    else:
        optimizer.name = name
    return optimizer