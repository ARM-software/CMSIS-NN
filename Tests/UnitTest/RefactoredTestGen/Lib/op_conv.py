# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
import Lib.op_utils
import math

from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import OpResolverType
import tf_keras as keras
import numpy as np


class Op_conv(Lib.op_utils.Op_type):

    def get_shapes(params):
        shapes = {}

        # Common default parameters
        params["stride_x"] = 1 if "stride_x" not in params else params["stride_x"]
        params["stride_y"] = 1 if "stride_y" not in params else params["stride_y"]
        params["dilation_x"] = 1 if "dilation_x" not in params else params["dilation_x"]
        params["dilation_y"] = 1 if "dilation_y" not in params else params["dilation_y"]
        params["batch_size"] = 1 if "batch_size" not in params else params["batch_size"]
        params["generate_bias"] = True if "generate_bias" not in params else params["generate_bias"]
        if "out_activation_min" not in params:
            params["out_activation_min"] = Lib.op_utils.get_dtype_min(params["input_data_type"])
        if "out_activation_max" not in params:
            params["out_activation_max"] = Lib.op_utils.get_dtype_max(params["input_data_type"])
        if "bias_min" not in params:
            params["bias_min"] = Lib.op_utils.get_dtype_min("int32_t")
        if "bias_max" not in params:
            params["bias_max"] = Lib.op_utils.get_dtype_max("int32_t")
        if "weights_min" not in params:
            params["weights_min"] = Lib.op_utils.get_dtype_min("int32_t")
        if "weights_max" not in params:
            params["weights_max"] = Lib.op_utils.get_dtype_max("int32_t")

        in_ch = params["in_ch"]
        out_ch = params["out_ch"]
        groups = params["groups"]
        filter_ch = in_ch // groups

        if in_ch % groups != 0:
            raise RuntimeError("ERROR: Input channels {} must be an even multiple of groups {}".format(in_ch, groups))
        if out_ch % groups != 0:
            raise RuntimeError("ERROR: Output channels {} must be an even multiple of groups {}".format(out_ch, groups))

        shapes["input"] = (params["batch_size"], params["input_h"], params["input_w"], in_ch)
        shapes["weight_shape"] = [params["filter_y"], params["filter_x"], filter_ch, out_ch]

        if params["generate_bias"]:
            shapes["bias_shape"] = [out_ch]
        else:
            shapes["bias_shape"] = []

        shapes["representational_dataset"] = (params["batch_size"], params["input_h"], params["input_w"], in_ch)
        return shapes

    def generate_keras_model(shapes, params):

        model = keras.models.Sequential()
        input_shape = (params["batch_size"], params["input_h"], params["input_w"], params["in_ch"])
        model.add(keras.layers.InputLayer(input_shape=input_shape[1:], batch_size=params["batch_size"]))

        conv_layer = keras.layers.Conv2D(params["out_ch"],
                                         kernel_size=(params["filter_y"], params["filter_x"]),
                                         strides=(params["stride_y"], params["stride_x"]),
                                         padding=params["padding"],
                                         input_shape=input_shape[1:],
                                         dilation_rate=(params["dilation_y"], params["dilation_x"]),
                                         groups=params["groups"],
                                         use_bias=params["generate_bias"])
        model.add(conv_layer)

        weights = Lib.op_utils.generate_tf_tensor(
            shapes["weight_shape"], params["weights_min"], params["weights_max"], decimals=8)

        if params["generate_bias"]:
            bias = Lib.op_utils.generate_tf_tensor(
                shapes["bias_shape"], params["bias_min"], params["bias_max"])
            conv_layer.set_weights([weights, bias])
        else:
            conv_layer.set_weights([weights])

        return model

    def generate_data_tflite(tflite_fname, params):
        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}
        aliases = {}

        # To be removed
        aliases["output_multiplier"] = "output_mult"
        aliases["bias"] = "biases"
        aliases["output"] = "output_ref"

        interpreter = Interpreter(str(tflite_fname), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        tensor_details = interpreter.get_tensor_details()
        input_state = tensor_details[0]

        if params["generate_bias"]:
            filter_index = 1
            bias_index = 2
        else:
            filter_index = 2
            bias_index = 1

        filter_layer = tensor_details[filter_index]

        if params["generate_bias"]:
            bias_layer = tensor_details[bias_index]
        else:
            bias_layer = None

        input_details = interpreter.get_input_details()
        (scales["input_scale"], scales["input_zero_point"]) = input_details[0]['quantization']

        output_details = interpreter.get_output_details()
        (scales["output_scale"], scales["output_zero_point"]) = output_details[0]['quantization']

        x_output = output_details[0]['shape'][2]
        y_output = output_details[0]['shape'][1]

        def calculate_padding(x_output, y_output, params):
            x_input = params["input_w"]
            y_input = params["input_h"]

            if params["padding"] == "SAME":
                # Take dilation into account.
                filter_x = (params["filter_x"] - 1) * params["dilation_x"] + 1
                filter_y = (params["filter_y"] - 1) * params["dilation_y"] + 1

                pad_along_width = max((x_output - 1) * params["stride_x"] + filter_x - x_input, 0)
                pad_along_height = max((y_output - 1) * params["stride_y"] + filter_y - y_input, 0)

                pad_top = pad_along_height // 2
                pad_left = pad_along_width // 2
                pad_top_offset = pad_along_height % 2
                pad_left_offset = pad_along_width % 2

                pad_y_with_offset = pad_top + pad_top_offset
                pad_x_with_offset = pad_left + pad_left_offset
                pad_x = pad_left
                pad_y = pad_top
            else:
                pad_x = 0
                pad_y = 0
                pad_y_with_offset = 0
                pad_x_with_offset = 0

            return pad_y_with_offset, pad_x_with_offset, pad_y, pad_x

        pad_y_with_offset, pad_x_with_offset, pad_y, pad_x = calculate_padding(x_output, y_output, params)

        tensors["weights"] = interpreter.get_tensor(filter_layer['index'])

        if params["generate_bias"]:
            tensors["bias"] = interpreter.get_tensor(bias_layer['index'])
        else:
            tensors["bias"] = None

        scales["scaling_factors"] = filter_layer['quantization_parameters']['scales']

        def generate_quantize_per_channel_multiplier(params, scales):
            def quantize_scale(scale):
                significand, shift = math.frexp(scale)
                significand_q31 = round(significand * (1 << 31))
                return significand_q31, shift

            num_channels = params["out_ch"]
            per_channel_multiplier = []
            per_channel_shift = []

            if len(scales["scaling_factors"]) != num_channels:
                raise RuntimeError("Missing scaling factors")

            for i in range(num_channels):
                effective_output_scale = scales["input_scale"] * scales["scaling_factors"][i] / scales["output_scale"]
                (quantized_multiplier, shift) = quantize_scale(effective_output_scale)

                per_channel_multiplier.append(quantized_multiplier)
                per_channel_shift.append(shift)

            return per_channel_multiplier, per_channel_shift

        generated_params["input_batches"] = params["batch_size"]
        generated_params["pad_x"] = pad_x
        generated_params["pad_y"] = pad_y
        generated_params["output_h"] = y_output
        generated_params["output_w"] = x_output
        generated_params["dst_size"] = x_output * y_output * params["out_ch"] * params["batch_size"]
        generated_params["input_offset"] = -input_state['quantization_parameters']['zero_points'][0]
        generated_params["output_offset"] = output_details[0]['quantization'][1]

        per_channel_multiplier, per_channel_shift = generate_quantize_per_channel_multiplier(params, scales)

        tensors["output_multiplier"] = np.array(per_channel_multiplier)
        tensors["output_shift"] = np.array(per_channel_shift)

        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales, aliases)
