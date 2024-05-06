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

import numpy as np


class Op_fully_connected(Lib.op_utils.Op_type):

    def get_shapes(params):
        shapes = {}

        # Common default parameters
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

        shapes["input"] = (params["batch_size"], in_ch)
        shapes["weight_shape"] = (in_ch, 1, 1, out_ch)

        if params["generate_bias"]:
            shapes["bias_shape"] = [out_ch]
            params["json_template"] = "fully_connected.json"
        else:
            shapes["bias_shape"] = []
            params["json_template"] = "fully_connected_null_bias.json"

        return shapes

    def generate_data_json(shapes, params):
        tensors = {}
        effective_scales = {}
        scales = {}
        generated_params = {}
        aliases = {}

        generated_params["input_batches"] = params["batch_size"]
        generated_params["input_w"] = 1
        generated_params["input_h"] = 1
        generated_params["dst_size"] = params["out_ch"] * params["batch_size"]
        generated_params["accumulation_depth"] = params["in_ch"]
        generated_params["input_offset"] = -params["input_zp"]
        generated_params["output_offset"] = params["output_zp"]

        # To be removed
        aliases["input_bias"] = "biases"
        aliases["output"] = "output_ref"
        aliases["input_weights"] = "weights"

        # TODOx
        minval = -7
        maxval = 8
        weights = np.random.randint(minval, maxval, size=shapes["weight_shape"])

        uneven = weights.size % 2
        if uneven:
            weights = np.append(weights, 0)

        temp = np.reshape(weights, (weights.size // 2, 2)).astype(np.uint8)
        weights = 0xff & ((0xf0 & (temp[:, 1] << 4)) | (temp[:, 0] & 0xf))
        tensors["input_weights"] = weights

        if params["generate_bias"]:
            tensors["input_bias"] = np.random.randint(minval, maxval, size=shapes["bias_shape"])
        else:
            tensors["input_bias"] = None

        def quantize_multiplier(input_scale, weights_scale, output_scale):
            def quantize_scale(scale):
                significand, shift = math.frexp(scale)
                significand_q31 = round(significand * (1 << 31))
                return significand_q31, shift

            input_product_scale = input_scale * weights_scale
            if input_product_scale < 0:
                raise RuntimeError("negative input product scale")
            real_multipler = input_product_scale / output_scale
            return quantize_scale(real_multipler)

        generated_params["output_multiplier"], generated_params["output_shift"] = quantize_multiplier(
            params["input_scale"], params["w_scale"], params["output_scale"])

        return Lib.op_utils.Generated_data(generated_params, tensors, scales, effective_scales, aliases)
