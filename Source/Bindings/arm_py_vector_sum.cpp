/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_py_vector_sum.cpp
 * Description:  Python helper for effective-bias preparation
 *
 * $Date:        15 October 2026
 * $Revision:    V.1.0.0
 * -------------------------------------------------------------------- */

#include <limits>
#include <stdexcept>

#include <pybind11/numpy.h>

#include "arm_py_common.hpp"

extern "C" {
#include "arm_nnfunctions.h"
}

namespace py = pybind11;

void vector_sum(py::module_ &m)
{
    m.def(
        "vector_sum_s8",
        [](const py::array &weights, int32_t lhs_offset, const py::object &bias, int32_t rhs_offset) {
            if (!weights.dtype().is(py::dtype::of<int8_t>()))
            {
                throw py::type_error("weights must have dtype int8");
            }
            if ((weights.flags() & py::array::c_style) == 0)
            {
                throw py::value_error("weights must be C-contiguous");
            }
            if (weights.ndim() < 1 || weights.shape(0) <= 0)
            {
                throw py::value_error("weights must have a non-empty output-channel dimension");
            }

            const py::ssize_t vector_rows_size = weights.shape(0);
            const py::ssize_t vector_cols_size = weights.size() / vector_rows_size;
            if (vector_cols_size <= 0 || vector_rows_size > std::numeric_limits<int32_t>::max() ||
                vector_cols_size > std::numeric_limits<int32_t>::max())
            {
                throw py::value_error("weights shape is outside the supported int32 range");
            }

            const int32_t *bias_data = nullptr;
            py::array bias_array;
            if (!bias.is_none())
            {
                if (!py::isinstance<py::array>(bias))
                {
                    throw py::type_error("bias must be a NumPy array with dtype int32 or None");
                }
                bias_array = py::reinterpret_borrow<py::array>(bias);
                if (!bias_array.dtype().is(py::dtype::of<int32_t>()))
                {
                    throw py::type_error("bias must have dtype int32");
                }
                if ((bias_array.flags() & py::array::c_style) == 0)
                {
                    throw py::value_error("bias must be C-contiguous");
                }
                if (bias_array.ndim() != 1 || bias_array.shape(0) != vector_rows_size)
                {
                    throw py::value_error("bias must be one-dimensional with one entry per output channel");
                }
                bias_data = static_cast<const int32_t *>(bias_array.data());
            }

            py::array_t<int32_t> result(vector_rows_size);
            const arm_cmsis_nn_status status = arm_vector_sum_s8(result.mutable_data(),
                                                                 static_cast<int32_t>(vector_cols_size),
                                                                 static_cast<int32_t>(vector_rows_size),
                                                                 static_cast<const int8_t *>(weights.data()),
                                                                 lhs_offset,
                                                                 rhs_offset,
                                                                 bias_data);
            if (status != ARM_CMSIS_NN_SUCCESS)
            {
                throw std::runtime_error("arm_vector_sum_s8 failed");
            }
            return result;
        },
        py::arg("weights"),
        py::arg("lhs_offset"),
        py::arg("bias") = py::none(),
        py::arg("rhs_offset") = 0);
}
