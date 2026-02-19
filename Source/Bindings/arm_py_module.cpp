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
 * Title:        arm_py_module.cpp
 * Description:  Optional Python module
 *
 * $Date:        23 Feb 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Host/Python
 * -------------------------------------------------------------------- */

#include <pybind11/pybind11.h>

namespace py = pybind11;

void conv_buffer_size(py::module_ &m);
void backend_helpers(py::module_ &m);
void depthwise_conv_buffer_size(py::module_ &m);
void fully_connected_buffer_size(py::module_ &m);
void avgpool_buffer_size(py::module_ &m);
void transpose_conv_buffer_size(py::module_ &m);
void svdf_buffer_size(py::module_ &m);

PYBIND11_MODULE(cmsis_nn, m)
{
    m.doc() = "CMSIS-NN buffer size helpers";
    conv_buffer_size(m);
    backend_helpers(m);
    depthwise_conv_buffer_size(m);
    fully_connected_buffer_size(m);
    avgpool_buffer_size(m);
    transpose_conv_buffer_size(m);
    svdf_buffer_size(m);
}
