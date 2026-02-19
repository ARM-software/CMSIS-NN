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
 * Title:        arm_py_backend.cpp
 * Description:  Backend resolution helpers for optional Python module
 *
 * $Date:        23 Feb 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Host/Python
 * -------------------------------------------------------------------- */

#include "arm_py_common.hpp"

namespace py = pybind11;

void backend_helpers(py::module_ &m)
{
    py::enum_<CortexM>(m, "CortexM")
        .value("M0", CortexM::M0)
        .value("M0PLUS", CortexM::M0PLUS)
        .value("M3", CortexM::M3)
        .value("M4", CortexM::M4)
        .value("M7", CortexM::M7)
        .value("M23", CortexM::M23)
        .value("M33", CortexM::M33)
        .value("M35P", CortexM::M35P)
        .value("M55", CortexM::M55)
        .value("M85", CortexM::M85);

    m.def(
        "resolve_backend",
        [](CortexM core) -> Backend {
            switch (core)
            {
            case CortexM::M55:
            case CortexM::M85:
                return Backend::MVE;
            case CortexM::M4:
            case CortexM::M7:
            case CortexM::M33:
            case CortexM::M35P:
                return Backend::DSP;
            case CortexM::M0:
            case CortexM::M0PLUS:
            case CortexM::M3:
            case CortexM::M23:
                return Backend::SCALAR;
            }
            throw py::value_error("unsupported Cortex-M core");
        },
        py::arg("core"));
}
