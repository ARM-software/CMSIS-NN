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
 * Title:        arm_py_svdf_buffer_size.cpp
 * Description:  SVDF buffer size pybinds (optional Python module)
 *
 * $Date:        23 Feb 2026
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

void svdf_buffer_size(py::module_ &m)
{
    m.def(
        "svdf_buffer_size",
        [](Backend backend, DataType data_type, const std::array<int32_t, 4> &filter_nhwc) -> int32_t {
            const cmsis_nn_dims filter_dims = make_dims(filter_nhwc);

            switch (data_type)
            {
            case DataType::A8W8:
                switch (backend)
                {
                case Backend::MVE:
                    return arm_svdf_s8_get_buffer_size_mve(&filter_dims);
                case Backend::DSP:
                    return arm_svdf_s8_get_buffer_size_dsp(&filter_dims);
                case Backend::SCALAR:
                    return arm_svdf_s8_get_buffer_size(&filter_dims);
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
        py::arg("filter_nhwc"));
}
