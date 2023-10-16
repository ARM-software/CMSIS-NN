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
                 interpreter="tensorflow",
                 input_scale=0.1,
                 input_zp=0,
                 w_scale=0.005,
                 w_zp=0,
                 bias_scale=0.00002,
                 bias_zp=0,
                 state_scale=0.005,
                 state_zp=0,
                 output_scale=0.1,
                 output_zp=0,
                 packed_4bit = False
                 ):
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

        self.packed_4bit = packed_4bit
        if self.packed_4bit:
            if self.generate_bias:
                self.json_template = "TestCases/Common/fc_s4_weights_template.json"
            else:
                self.json_template = "TestCases/Common/fc_s4_weights_template_null_bias.json"

            self.in_activation_max = TestSettings.INT4_MAX
            self.in_activation_min = TestSettings.INT4_MIN

        self.json_replacements = {
            "batches" : batches,
            "input_size" : in_ch * x_in * y_in,
            "input_scale" : input_scale,
            "input_zp" : input_zp,
            "w_scale" : w_scale,
            "w_zp" : w_zp,
            "bias_size" : out_ch,
            "bias_scale" : bias_scale,
            "bias_zp" : bias_zp,
            "output_size" : out_ch,
            "output_scale" : output_scale,
            "output_zp" : output_zp
        }


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
        if self.packed_4bit:
            if not self.use_tflite_micro_interpreter:
                print("Warning: interpreter tflite_micro must be used for fully_connected int4. Skipping generating headers.")
                return

        if self.is_int16xint8:
            inttype = tf.int16
            datatype = "int16_t"
            bias_datatype = "int64_t"
        else:
            inttype = tf.int8
            datatype = "int8_t"
            bias_datatype = "int32_t"

        # Generate data
        fc_input_format =  [self.batches, self.input_ch * self.x_input * self.y_input]
        if input_data is not None:
            input_data = tf.reshape(input_data, fc_input_format)
        else:
           input_data = self.get_randomized_input_data(input_data, fc_input_format)

        # Generate bias
        biases = self.get_randomized_bias_data(biases)

        # Generate weights
        if self.packed_4bit:
            # Generate packed and unpacked model from JSON
            temp1 = self.model_path
            temp2 = self.json_template

            fc_weights_format = [self.input_ch * self.y_input * self.x_input * self.output_ch]
            if weights is not None:
                weights = tf.reshape(weights, fc_weights_format)
            else:
                weights = self.get_randomized_data(fc_weights_format, self.kernel_table_file, minrange=TestSettings.INT4_MIN, maxrange=TestSettings.INT4_MAX, regenerate=self.regenerate_new_weights)

            if not self.generate_bias:
                biases = None

            # Unpacked model is used for reference during debugging only and not used by default
            self.model_path = self.model_path + "_unpacked"
            self.json_template = self.json_template[:-5] + "_unpacked.json"
            generated_json = self.generate_json_from_template(weights, bias_data = biases, bias_buffer=2)
            self.flatc_generate_tflite(generated_json, self.schema_file)

            self.model_path = temp1
            self.json_template = temp2
            temp = np.reshape(weights, (len(weights)//2, 2)).astype(np.uint8)
            temp = 0xff & ((0xf0 & (temp[:,1] << 4)) | (temp[:,0] & 0xf))
            weights = tf.convert_to_tensor(temp)
            generated_json = self.generate_json_from_template(weights, bias_data = biases, bias_buffer=2)
            self.flatc_generate_tflite(generated_json, self.schema_file)

            interpreter = self.Interpreter(model_path=str(self.model_path_tflite), experimental_op_resolver_type=self.OpResolverType.BUILTIN_REF)
            interpreter.allocate_tensors()

        else:
            # Generate model in tensorflow with one fully_connected layer
            fc_weights_format = [self.input_ch * self.y_input * self.x_input, self.output_ch]
            if weights is not None:
                weights = tf.reshape(weights, fc_weights_format)
            else:
                weights = self.get_randomized_data(fc_weights_format, self.kernel_table_file, minrange=TestSettings.INT32_MIN, maxrange=TestSettings.INT32_MAX, regenerate=self.regenerate_new_weights)


            model = tf.keras.models.Sequential()
            model.add(
                tf.keras.layers.InputLayer(input_shape=(self.y_input * self.x_input * self.input_ch, ),
                                        batch_size=self.batches))
            fully_connected_layer = tf.keras.layers.Dense(self.output_ch, activation=None)
            model.add(fully_connected_layer)
            fully_connected_layer.set_weights([weights, biases])
            interpreter = self.convert_and_interpret(model, inttype, input_data)

        # Get layer information
        all_layers_details = interpreter.get_tensor_details()
        input_layer = all_layers_details[0]
        (self.input_scale, self.input_zero_point) = self.get_scale_and_zp(input_layer)
        filter_layer = all_layers_details[1]
        (self.weights_scale, self.weights_zero_point) = self.get_scale_and_zp(filter_layer)
        if self.generate_bias:
            output_layer = all_layers_details[3]
        else:
            output_layer = all_layers_details[2]
        (self.output_scale, self.output_zero_point) = self.get_scale_and_zp(output_layer)
        self.x_output = 1
        self.y_output = 1

        self.quantize_multiplier()

        # Generate reference output
        if self.packed_4bit:
            interpreter = self.tflite_micro.runtime.Interpreter.from_file(model_path=str(self.model_path_tflite))
            interpreter.set_input(tf.cast(input_data, tf.int8), input_layer["index"])
            interpreter.invoke()
            output_data = interpreter.get_output(0)
        else:
            output_details = interpreter.get_output_details()
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])

        # Save results
        self.generate_c_array(self.input_data_file_prefix, input_data, datatype=datatype)
        self.generate_c_array(self.weight_data_file_prefix, weights, datatype=datatype)
        if self.generate_bias:
            self.generate_c_array(self.bias_data_file_prefix, biases, datatype=bias_datatype)
        self.generate_c_array(self.output_data_file_prefix, np.clip(output_data, self.out_activation_min, self.out_activation_max), datatype=datatype)
        self.write_c_config_header()
        self.write_c_header_wrapper()

    def get_scale_and_zp(self, layer):
        return (layer['quantization_parameters']['scales'][0], layer['quantization_parameters']['zero_points'][0])
