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
 * Title:        arm_py_convolve_packed.cpp
 * Description:  Python helpers for AOT convolution preparation
 *
 * $Date:        15 October 2026
 * $Revision:    V.1.0.0
 * -------------------------------------------------------------------- */

#include "arm_py_common.hpp"
#include <limits>
#include <pybind11/numpy.h>

extern "C" {
#include "arm_nnfunctions.h"
}

static cmsis_nn_dims packed_dims(const py::array &weights)
{
    if (!weights.dtype().is(py::dtype::of<int8_t>()))
    {
        throw py::type_error("weights must have dtype int8");
    }
    if ((weights.flags() & py::array::c_style) == 0)
    {
        throw py::value_error("weights must be C-contiguous");
    }
    if (weights.ndim() != 2 && weights.ndim() != 4)
    {
        throw py::value_error("weights must have shape [C_out, K] or [C_out, H, W, C_in_per_group]");
    }
    cmsis_nn_dims dims = {1, 1, 1, 1};
    for (py::ssize_t i = 0; i < weights.ndim(); i++)
    {
        if (weights.shape(i) <= 0 || weights.shape(i) > std::numeric_limits<int32_t>::max())
        {
            throw py::value_error("weights dimensions are outside the supported int32 range");
        }
    }
    dims.n = static_cast<int32_t>(weights.shape(0));
    dims.c = static_cast<int32_t>(weights.shape(weights.ndim() - 1));
    if (weights.ndim() == 4)
    {
        dims.h = static_cast<int32_t>(weights.shape(1));
        dims.w = static_cast<int32_t>(weights.shape(2));
    }
    return dims;
}

static py::array_t<int8_t> pack_weights(const py::array &weights, int32_t groups)
{
    const cmsis_nn_dims dims = packed_dims(weights);
    const int32_t size = arm_convolve_s8_packed_get_filter_size(&dims, groups);
    if (!size)
    {
        throw py::value_error("invalid groups, dimensions, or packed filter size exceeds INT32_MAX");
    }
    py::array_t<int8_t> packed(size);
    if (arm_convolve_s8_pack_weights(
            &dims, static_cast<const int8_t *>(weights.data()), groups, packed.mutable_data()) != ARM_CMSIS_NN_SUCCESS)
    {
        throw py::value_error("arm_convolve_s8_pack_weights failed");
    }
    return packed;
}

void convolve_packed(py::module_ &m)
{
    m.def(
        "convolve_s8_packed_filter_size",
        [](const std::array<int32_t, 4> &filter_nhwc, int32_t groups) {
            const cmsis_nn_dims dims = make_dims(filter_nhwc);
            return arm_convolve_s8_packed_get_filter_size(&dims, groups);
        },
        py::arg("filter_nhwc"),
        py::arg("groups") = 1);
    m.def("convolve_s8_pack_weights", &pack_weights, py::arg("weights"), py::arg("groups") = 1);
    const py::object vector_sum = m.attr("vector_sum_s8");
    m.def(
        "convolve_s8_prepare_weights",
        [vector_sum](const py::array &weights, int32_t input_offset, const py::object &bias, int32_t groups) {
            const auto packed = pack_weights(weights, groups);
            const auto effective_bias = vector_sum(weights, input_offset, bias);
            return py::make_tuple(packed, effective_bias);
        },
        py::arg("weights"),
        py::arg("input_offset"),
        py::arg("bias") = py::none(),
        py::arg("groups") = 1);
}
