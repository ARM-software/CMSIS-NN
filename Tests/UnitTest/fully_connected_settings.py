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


class FullyConnectedSettings(TestSettings):

    def __init__(self,
                 dataset,
                 testtype,
                 regenerate_weights,
                 regenerate_input,
                 regenerate_biases,
                 schema_file,
                 in_ch=1,
                 out_ch=1,
                 x_in=1,
                 y_in=1,
                 w_x=1,
                 w_y=1,
                 stride_x=1,
                 stride_y=1,
                 pad=False,
                 randmin=TestSettings.INT8_MIN,
                 randmax=TestSettings.INT8_MAX,
                 batches=1,
                 generate_bias=True,
                 out_activation_min=None,
                 out_activation_max=None,
                 int16xint8=False,
                 bias_min=TestSettings.INT32_MIN,
                 bias_max=TestSettings.INT32_MAX,
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
                         x_in,
                         y_in,
                         stride_x,
                         stride_y,
                         pad,
                         randmin,
                         randmax,
                         batches,
                         generate_bias=generate_bias,
                         out_activation_min=out_activation_min,
                         out_activation_max=out_activation_max,
                         int16xint8=int16xint8,
                         bias_min=bias_min,
                         bias_max=bias_max,
                         interpreter=interpreter)

    def write_c_config_header(self) -> None:
        super().write_c_config_header()

        filename = self.config_data
        filepath = self.headers_dir + filename
        prefix = self.testdataset.upper()

        with open(filepath, "a") as f:
            f.write("#define {}_OUTPUT_MULTIPLIER {}\n".format(prefix, self.quantized_multiplier))
            f.write("#define {}_OUTPUT_SHIFT {}\n".format(prefix, self.quantized_shift))
            f.write("#define {}_ACCUMULATION_DEPTH {}\n".format(prefix, self.input_ch * self.x_input * self.y_input))
            f.write("#define {}_INPUT_OFFSET {}\n".format(prefix, -self.input_zero_point))
            f.write("#define {}_OUTPUT_OFFSET {}\n".format(prefix, self.output_zero_point))

    def quantize_multiplier(self):
        input_product_scale = self.input_scale * self.weights_scale
        if input_product_scale < 0:
            raise RuntimeError("negative input product scale")
        real_multipler = input_product_scale / self.output_scale
        (self.quantized_multiplier, self.quantized_shift) = self.quantize_scale(real_multipler)

    def generate_data(self, input_data=None, weights=None, biases=None) -> None:
        input_data = self.get_randomized_input_data(input_data,
                                                    [self.batches, self.input_ch * self.x_input * self.y_input])

        if self.is_int16xint8:
            inttype = tf.int16
            datatype = "int16_t"
            bias_datatype = "int64_t"
        else:
            inttype = tf.int8
            datatype = "int8_t"
            bias_datatype = "int32_t"

        fc_weights_format = [self.input_ch * self.y_input * self.x_input, self.output_ch]

        if weights is not None:
            weights = tf.reshape(weights, fc_weights_format)
        else:
            weights = self.get_randomized_data(fc_weights_format,
                                               self.kernel_table_file,
                                               minrange=TestSettings.INT32_MIN,
                                               maxrange=TestSettings.INT32_MAX,
                                               regenerate=self.regenerate_new_weights)

        biases = self.get_randomized_bias_data(biases)

        # Create model with one fully_connected layer.
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.InputLayer(input_shape=(self.y_input * self.x_input * self.input_ch, ),
                                       batch_size=self.batches))
        fully_connected_layer = tf.keras.layers.Dense(self.output_ch, activation=None)
        model.add(fully_connected_layer)
        fully_connected_layer.set_weights([weights, biases])

        interpreter = self.convert_and_interpret(model, inttype, input_data)

        all_layers_details = interpreter.get_tensor_details()
        if self.generate_bias:
            filter_layer = all_layers_details[2]
            bias_layer = all_layers_details[1]
        else:
            filter_layer = all_layers_details[1]
        if weights.numpy().size != interpreter.get_tensor(filter_layer['index']).size or \
           (self.generate_bias and biases.numpy().size != interpreter.get_tensor(bias_layer['index']).size):
            raise RuntimeError(f"Dimension mismatch for {self.testdataset}")

        # The generic destination size calculation for these tests are: self.x_output * self.y_output * self.output_ch
        # * self.batches.
        self.x_output = 1
        self.y_output = 1
        output_details = interpreter.get_output_details()
        if self.output_ch != output_details[0]['shape'][1] or self.batches != output_details[0]['shape'][0]:
            raise RuntimeError("Fully connected out dimension mismatch")

        self.weights_scale = filter_layer['quantization_parameters']['scales'][0]
        self.quantize_multiplier()

        self.generate_c_array(self.input_data_file_prefix, input_data, datatype=datatype)
        self.generate_c_array(self.weight_data_file_prefix, interpreter.get_tensor(filter_layer['index']))

        if self.generate_bias:
            self.generate_c_array(self.bias_data_file_prefix, interpreter.get_tensor(bias_layer['index']),
                                  bias_datatype)
        else:
            self.generate_c_array(self.bias_data_file_prefix, biases, bias_datatype)

        # Generate reference
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        self.generate_c_array(self.output_data_file_prefix,
                              np.clip(output_data, self.out_activation_min, self.out_activation_max),
                              datatype=datatype)

        self.write_c_config_header()
        self.write_c_header_wrapper()
