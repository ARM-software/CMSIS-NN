# SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from test_settings import TestSettings

import tensorflow as tf
import numpy as np


class ConvSettings(TestSettings):

    def __init__(self,
                 dataset,
                 testtype,
                 regenerate_weights,
                 regenerate_input,
                 regenerate_biases,
                 schema_file,
                 in_ch=1,
                 out_ch=1,
                 x_in=7,
                 y_in=7,
                 w_x=3,
                 w_y=3,
                 stride_x=2,
                 stride_y=2,
                 pad=True,
                 randmin=TestSettings.INT8_MIN,
                 randmax=TestSettings.INT8_MAX,
                 batches=1,
                 generate_bias=True,
                 relu6=False,
                 out_activation_min=None,
                 out_activation_max=None,
                 int16xint8=False,
                 bias_min=TestSettings.INT32_MIN,
                 bias_max=TestSettings.INT32_MAX,
                 dilation_x=1,
                 dilation_y=1,
                 interpreter="tensorflow"):
        super().__init__(dataset,
                         testtype,
                         regenerate_weights,
                         regenerate_input,
                         regenerate_biases,
                         schema_file,
                         in_ch,
                         out_ch,
                         x_in,
                         y_in,
                         w_x,
                         w_y,
                         stride_x,
                         stride_y,
                         pad,
                         randmin,
                         randmax,
                         batches,
                         generate_bias=generate_bias,
                         relu6=relu6,
                         out_activation_min=out_activation_min,
                         out_activation_max=out_activation_max,
                         int16xint8=int16xint8,
                         bias_min=bias_min,
                         bias_max=bias_max,
                         dilation_x=dilation_x,
                         dilation_y=dilation_y,
                         interpreter=interpreter)

        self.scaling_factors = []

        if self.test_type == 'depthwise_conv':
            self.channel_multiplier = self.output_ch // self.input_ch
            if self.output_ch % self.input_ch != 0:
                raise RuntimeError("out channel ({}) is not multiple of in channel ({})".format(out_ch, in_ch))

    def write_c_config_header(self) -> None:
        super().write_c_config_header()

        filename = self.config_data
        filepath = self.headers_dir + filename
        prefix = self.testdataset.upper()

        with open(filepath, "a") as f:
            self.write_common_config(f, prefix)
            if self.test_type == 'depthwise_conv':
                f.write("#define {}_CH_MULT {}\n".format(prefix, self.channel_multiplier))
            f.write("#define {}_INPUT_OFFSET {}\n".format(prefix, -self.input_zero_point))
            f.write("#define {}_OUTPUT_OFFSET {}\n".format(prefix, self.output_zero_point))
            f.write("#define {}_DILATION_X {}\n".format(prefix, self.dilation_x))
            f.write("#define {}_DILATION_Y {}\n".format(prefix, self.dilation_y))

    def generate_quantize_per_channel_multiplier(self):
        num_channels = self.output_ch
        per_channel_multiplier = []
        per_channel_shift = []

        if len(self.scaling_factors) != num_channels:
            raise RuntimeError("Missing scaling factors")

        for i in range(num_channels):
            effective_output_scale = self.input_scale * self.scaling_factors[i] / self.output_scale
            (quantized_multiplier, shift) = self.quantize_scale(effective_output_scale)

            per_channel_multiplier.append(quantized_multiplier)
            per_channel_shift.append(shift)

        return per_channel_multiplier, per_channel_shift

    def generate_data(self, input_data=None, weights=None, biases=None) -> None:
        if self.is_int16xint8:
            inttype = tf.int16
            datatype = "int16_t"
            bias_datatype = "int64_t"
        else:
            inttype = tf.int8
            datatype = "int8_t"
            bias_datatype = "int32_t"

        input_data = self.get_randomized_input_data(input_data)

        if self.test_type == 'conv':
            out_channel = self.output_ch
        elif self.test_type == 'depthwise_conv':
            out_channel = self.channel_multiplier

        if weights is not None:
            weights = tf.reshape(weights, [self.filter_y, self.filter_x, self.input_ch, out_channel])
        else:
            weights = self.get_randomized_data([self.filter_y, self.filter_x, self.input_ch, out_channel],
                                               self.kernel_table_file,
                                               minrange=TestSettings.INT32_MIN,
                                               maxrange=TestSettings.INT32_MAX,
                                               decimals=1,
                                               regenerate=self.regenerate_new_weights)

        biases = self.get_randomized_bias_data(biases)

        # Create a one layer Keras model.
        model = tf.keras.models.Sequential()
        input_shape = (self.batches, self.y_input, self.x_input, self.input_ch)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape[1:], batch_size=self.batches))
        if self.test_type == 'conv':
            conv_layer = tf.keras.layers.Conv2D(self.output_ch,
                                                kernel_size=(self.filter_y, self.filter_x),
                                                strides=(self.stride_y, self.stride_x),
                                                padding=self.padding,
                                                input_shape=input_shape[1:],
                                                dilation_rate=(self.dilation_y, self.dilation_x))
            model.add(conv_layer)
            conv_layer.set_weights([weights, biases])
        elif self.test_type == 'depthwise_conv':
            depthwise_layer = tf.keras.layers.DepthwiseConv2D(kernel_size=(self.filter_y, self.filter_x),
                                                              strides=(self.stride_y, self.stride_x),
                                                              padding=self.padding,
                                                              depth_multiplier=self.channel_multiplier,
                                                              input_shape=input_shape[1:],
                                                              dilation_rate=(self.dilation_y, self.dilation_x))
            model.add(depthwise_layer)
            depthwise_layer.set_weights([weights, biases])
        interpreter = self.convert_and_interpret(model, inttype, input_data)

        all_layers_details = interpreter.get_tensor_details()
        filter_layer = all_layers_details[2]
        bias_layer = all_layers_details[1]
        if weights.numpy().size != interpreter.get_tensor(filter_layer['index']).size or \
           (self.generate_bias and biases.numpy().size != interpreter.get_tensor(bias_layer['index']).size):
            raise RuntimeError(f"Dimension mismatch for {self.testdataset}")

        output_details = interpreter.get_output_details()
        self.set_output_dims_and_padding(output_details[0]['shape'][2], output_details[0]['shape'][1])

        self.generate_c_array(self.input_data_file_prefix, input_data, datatype=datatype)
        self.generate_c_array(self.weight_data_file_prefix, interpreter.get_tensor(filter_layer['index']))

        self.scaling_factors = filter_layer['quantization_parameters']['scales']
        per_channel_multiplier, per_channel_shift = self.generate_quantize_per_channel_multiplier()
        self.generate_c_array("output_mult", per_channel_multiplier, datatype='int32_t')
        self.generate_c_array("output_shift", per_channel_shift, datatype='int32_t')

        self.generate_c_array(self.bias_data_file_prefix, interpreter.get_tensor(bias_layer['index']), bias_datatype)

        # Generate reference
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        self.generate_c_array(self.output_data_file_prefix,
                              np.clip(output_data, self.out_activation_min, self.out_activation_max),
                              datatype=datatype)

        self.write_c_config_header()
        self.write_c_header_wrapper()
