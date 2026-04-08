/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_py_depthwise_conv_buffer_size.cpp
 * Description:  Depthwise convolve wrapper buffer size pybinds (optional Python module)
 *
 * $Date:        5 Mar 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Host/Python
 * -------------------------------------------------------------------- */

#include <array>
#include <sstream>

#include "arm_py_common.hpp"

extern "C" {
#include "arm_nnfunctions.h"
}

namespace py = pybind11;

static inline cmsis_nn_dw_conv_params make_dw_conv_params(const std::array<int32_t, 2> &padding_hw,
                                                          const std::array<int32_t, 2> &stride_hw,
                                                          const std::array<int32_t, 2> &dilation_hw,
                                                          int32_t input_offset,
                                                          int32_t output_offset,
                                                          int32_t ch_mult,
                                                          int32_t activation_min,
                                                          int32_t activation_max)
{
    cmsis_nn_dw_conv_params p{};
    p.input_offset = input_offset;
    p.output_offset = output_offset;
    p.ch_mult = ch_mult;

    p.padding.h = padding_hw[0];
    p.padding.w = padding_hw[1];
    p.stride.h = stride_hw[0];
    p.stride.w = stride_hw[1];
    p.dilation.h = dilation_hw[0];
    p.dilation.w = dilation_hw[1];

    p.activation.min = activation_min;
    p.activation.max = activation_max;

    return p;
}

void depthwise_conv_buffer_size(py::module_ &m)
{
    m.def(
        "depthwise_conv_wrapper_buffer_size",
        [](Backend backend,
           DataType data_type,
           const std::array<int32_t, 4> &input_nhwc,
           const std::array<int32_t, 4> &filter_nhwc,
           const std::array<int32_t, 4> &output_nhwc,
           const std::array<int32_t, 2> &padding_hw,
           const std::array<int32_t, 2> &stride_hw,
           const std::array<int32_t, 2> &dilation_hw,
           int32_t ch_mult,
           int32_t input_offset,
           int32_t output_offset,
           int32_t activation_min,
           int32_t activation_max) -> int32_t {
            const cmsis_nn_dw_conv_params dw_conv_params = make_dw_conv_params(padding_hw,
                                                                               stride_hw,
                                                                               dilation_hw,
                                                                               input_offset,
                                                                               output_offset,
                                                                               ch_mult,
                                                                               activation_min,
                                                                               activation_max);

            const cmsis_nn_dims input_dims = make_dims(input_nhwc);
            const cmsis_nn_dims filter_dims = make_dims(filter_nhwc);
            const cmsis_nn_dims output_dims = make_dims(output_nhwc);

            switch (data_type)
            {
            case DataType::A8W4:
                switch (backend)
                {
                case Backend::MVE:
                    return arm_depthwise_conv_wrapper_s4_get_buffer_size_mve(
                        &dw_conv_params, &input_dims, &filter_dims, &output_dims);
                case Backend::DSP:
                    return arm_depthwise_conv_wrapper_s4_get_buffer_size_dsp(
                        &dw_conv_params, &input_dims, &filter_dims, &output_dims);
                case Backend::SCALAR:
                    return arm_depthwise_conv_wrapper_s4_get_buffer_size(
                        &dw_conv_params, &input_dims, &filter_dims, &output_dims);
                }
                break;
            case DataType::A8W8:
                switch (backend)
                {
                case Backend::MVE:
                    return arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(
                        &dw_conv_params, &input_dims, &filter_dims, &output_dims);
                case Backend::DSP:
                    return arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp(
                        &dw_conv_params, &input_dims, &filter_dims, &output_dims);
                case Backend::SCALAR:
                    return arm_depthwise_conv_wrapper_s8_get_buffer_size(
                        &dw_conv_params, &input_dims, &filter_dims, &output_dims);
                }
                break;
            case DataType::A16W8:
                switch (backend)
                {
                case Backend::MVE:
                    return arm_depthwise_conv_wrapper_s16_get_buffer_size_mve(
                        &dw_conv_params, &input_dims, &filter_dims, &output_dims);
                case Backend::DSP:
                    return arm_depthwise_conv_wrapper_s16_get_buffer_size_dsp(
                        &dw_conv_params, &input_dims, &filter_dims, &output_dims);
                case Backend::SCALAR:
                    return arm_depthwise_conv_wrapper_s16_get_buffer_size(
                        &dw_conv_params, &input_dims, &filter_dims, &output_dims);
                }
                break;
            }
            std::ostringstream msg;
            msg << "invalid Backend/DataType combination: backend=" << static_cast<int>(backend)
                << " data_type=" << static_cast<int>(data_type);
            throw py::value_error(msg.str());
        },
        py::arg("backend"),
        py::arg("data_type"),
        py::arg("input_nhwc"),
        py::arg("filter_nhwc"),
        py::arg("output_nhwc"),
        py::arg("padding_hw"),
        py::arg("stride_hw"),
        py::arg("dilation_hw"),
        py::arg("ch_mult"),
        py::arg("input_offset") = 0,
        py::arg("output_offset") = 0,
        py::arg("activation_min") = -128,
        py::arg("activation_max") = 127);
}
