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
 * Title:        arm_py_avgpool_buffer_size.cpp
 * Description:  Average pooling buffer size pybinds (optional Python module)
 *
 * $Date:        23 Feb 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Host/Python
 * -------------------------------------------------------------------- */

#include <sstream>

#include "arm_py_common.hpp"

extern "C" {
#include "arm_nnfunctions.h"
}

namespace py = pybind11;

void avgpool_buffer_size(py::module_ &m)
{
    m.def(
        "avgpool_buffer_size",
        [](Backend backend, DataType data_type, int32_t dim_dst_width, int32_t ch_src) -> int32_t {
            switch (data_type)
            {
            case DataType::A8W8:
                switch (backend)
                {
                case Backend::MVE:
                    return arm_avgpool_s8_get_buffer_size_mve(dim_dst_width, ch_src);
                case Backend::DSP:
                    return arm_avgpool_s8_get_buffer_size_dsp(dim_dst_width, ch_src);
                case Backend::SCALAR:
                    return arm_avgpool_s8_get_buffer_size(dim_dst_width, ch_src);
                }
                break;
            case DataType::A16W8:
                switch (backend)
                {
                case Backend::MVE:
                    return arm_avgpool_s16_get_buffer_size_mve(dim_dst_width, ch_src);
                case Backend::DSP:
                    return arm_avgpool_s16_get_buffer_size_dsp(dim_dst_width, ch_src);
                case Backend::SCALAR:
                    return arm_avgpool_s16_get_buffer_size(dim_dst_width, ch_src);
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
        py::arg("dim_dst_width"),
        py::arg("ch_src"));
}
