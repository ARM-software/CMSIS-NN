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
 * $Date:        15 October 2026
 * $Revision:    V.1.1.0
 *
 * Target :  Host/Python
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_config.h"
#include "arm_py_common.hpp"
#include <pybind11/numpy.h>

namespace py = pybind11;

void conv_buffer_size(py::module_ &m);
void backend_helpers(py::module_ &m);
void depthwise_conv_buffer_size(py::module_ &m);
void fully_connected_buffer_size(py::module_ &m);
void avgpool_buffer_size(py::module_ &m);
void transpose_conv_buffer_size(py::module_ &m);
void svdf_buffer_size(py::module_ &m);
void vector_sum(py::module_ &m);
void convolve_packed(py::module_ &m);

namespace {
// Incompatible pybind11/NumPy combinations can construct arrays whose
// Python-visible layout or values do not match the bytes written by C++, silently
// producing incorrect output without an immediate error. Write known values from
// C++ and check them and the layout through Python, independently of pybind11's
// internal NumPy metadata access.
template <typename T> void validate_numpy_array(const py::module_ &numpy, const char *dtype)
{
    py::array_t<T> array(4);
    T *data = array.mutable_data();
    for (int i = 0; i < 4; ++i)
    {
        data[i] = static_cast<T>((i + 1) * 10);
    }
    // Inspect Python-visible metadata, independently of pybind11's NumPy proxies.
    const py::object visible = array;
    if (!py::object(visible.attr("shape")).equal(py::make_tuple(4)) ||
        !py::object(visible.attr("dtype")).equal(numpy.attr("dtype")(dtype)) ||
        !py::object(visible.attr("strides")).equal(py::make_tuple(sizeof(T))) ||
        !visible.attr("flags").attr("c_contiguous").cast<bool>() ||
        !visible.attr("tolist")().equal(py::cast(std::array<int, 4>{{10, 20, 30, 40}})))
    {
        throw std::runtime_error(std::string("incorrect Python-visible ") + dtype + " array");
    }
}

void validate_numpy_runtime()
{
    std::string version = "unavailable";
    try
    {
        const py::module_ numpy = py::module_::import("numpy");
        version = py::str(numpy.attr("__version__"));
        const std::string major = version.substr(0, version.find('.'));
        if (major != "1" && major != "2")
        {
            throw std::runtime_error("unsupported NumPy major (require numpy>=1,<3)");
        }
        validate_numpy_array<int8_t>(numpy, "int8");
        validate_numpy_array<int32_t>(numpy, "int32");
    } catch (const std::exception &error)
    {
        throw py::import_error(
            "CMSIS-NN NumPy validation failed (compiled pybind11 " + std::to_string(PYBIND11_VERSION_MAJOR) + "." +
            std::to_string(PYBIND11_VERSION_MINOR) + "." + std::to_string(PYBIND11_VERSION_PATCH) + ", runtime NumPy " +
            version + "): " + error.what() +
            ". Install pybind11>=2.13.6,<4 and numpy>=1,<3, remove the old build directory and cached wheel, "
            "then clean-rebuild and reinstall CMSIS-NN.");
    }
}
} // namespace

PYBIND11_MODULE(cmsis_nn, m)
{
    validate_numpy_runtime();
    m.doc() = "CMSIS-NN helpers";
    conv_buffer_size(m);
    backend_helpers(m);
    depthwise_conv_buffer_size(m);
    fully_connected_buffer_size(m);
    avgpool_buffer_size(m);
    transpose_conv_buffer_size(m);
    svdf_buffer_size(m);
    vector_sum(m);
    m.attr("aot_enabled") = bool(ARM_NN_ENABLE_AOT);
#if ARM_NN_ENABLE_AOT
    convolve_packed(m);
#endif
}
