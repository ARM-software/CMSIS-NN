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
 * Title:        arm_py_transpose_conv_buffer_size.cpp
 * Description:  Transpose convolution buffer size pybinds (optional Python module)
 *
 * $Date:        29 Apr 2026
 * $Revision:    V.1.1.0
 *
 * Target :  Host/Python
 * -------------------------------------------------------------------- */

#include <array>
#include <sstream>
#include <string>

#include "arm_py_common.hpp"

extern "C" {
#include "arm_nnfunctions.h"
}

namespace py = pybind11;

/*
 * Note that this pybind implementation differs from the others. This is due to the
 * complex signature triggering error C1202 on MSVC (recursive type or function dependency context too complex)
 * This is resolved by implementing the python calling semantics manually.
 */

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

static inline cmsis_nn_transpose_conv_params
make_transpose_conv_params(const std::array<int32_t, 2> &padding_hw,
                           const std::array<int32_t, 2> &stride_hw,
                           const std::array<int32_t, 2> &dilation_hw,
                           const std::array<int32_t, 2> &padding_offsets_hw,
                           int32_t input_offset,
                           int32_t output_offset,
                           int32_t activation_min,
                           int32_t activation_max)
{
    cmsis_nn_transpose_conv_params p{};
    p.input_offset = input_offset;
    p.output_offset = output_offset;

    p.padding.h = padding_hw[0];
    p.padding.w = padding_hw[1];
    p.padding_offsets.h = padding_offsets_hw[0];
    p.padding_offsets.w = padding_offsets_hw[1];
    p.stride.h = stride_hw[0];
    p.stride.w = stride_hw[1];
    p.dilation.h = dilation_hw[0];
    p.dilation.w = dilation_hw[1];

    p.activation.min = activation_min;
    p.activation.max = activation_max;

    return p;
}

static inline void throw_invalid_combo(Backend backend, DataType data_type)
{
    std::ostringstream msg;
    msg << "invalid Backend/DataType combination: backend=" << static_cast<int>(backend)
        << " data_type=" << static_cast<int>(data_type);
    throw py::value_error(msg.str());
}

static inline bool is_valid_kwarg_name(const std::string &name, std::initializer_list<const char *> valid_names)
{
    for (const char *valid_name : valid_names)
    {
        if (name == valid_name)
        {
            return true;
        }
    }

    return false;
}

static inline void
validate_kwargs(const py::kwargs &kwargs, std::initializer_list<const char *> valid_names, const char *function_name)
{
    for (auto item : kwargs)
    {
        const std::string name = py::str(item.first).cast<std::string>();
        if (!is_valid_kwarg_name(name, valid_names))
        {
            throw py::type_error(std::string(function_name) + "() got an unexpected keyword argument '" + name + "'");
        }
    }
}

template <typename T>
static T parse_required_arg(const py::args &args, const py::kwargs &kwargs, size_t index, const char *name)
{
    const bool has_positional = index < args.size();
    const bool has_keyword = kwargs.contains(name);

    if (has_positional && has_keyword)
    {
        throw py::type_error(std::string("got multiple values for argument '") + name + "'");
    }

    if (has_positional)
    {
        return args[index].cast<T>();
    }

    if (has_keyword)
    {
        return kwargs[name].cast<T>();
    }

    throw py::type_error(std::string("missing required argument '") + name + "'");
}

template <typename T>
static T parse_optional_arg(const py::args &args,
                            const py::kwargs &kwargs,
                            size_t index,
                            const char *name,
                            const T &default_value)
{
    const bool has_positional = index < args.size();
    const bool has_keyword = kwargs.contains(name);

    if (has_positional && has_keyword)
    {
        throw py::type_error(std::string("got multiple values for argument '") + name + "'");
    }

    if (has_positional)
    {
        return args[index].cast<T>();
    }

    if (has_keyword)
    {
        return kwargs[name].cast<T>();
    }

    return default_value;
}

// ----------------------------------------------------------------------
// Wrapper implementations
// ----------------------------------------------------------------------

static int32_t transpose_conv_buffer_size_impl(Backend backend,
                                               DataType data_type,
                                               const std::array<int32_t, 4> &input_nhwc,
                                               const std::array<int32_t, 4> &filter_nhwc,
                                               const std::array<int32_t, 4> &output_nhwc,
                                               const cmsis_nn_transpose_conv_params &params)
{
    switch (data_type)
    {
    case DataType::A8W8: {
        const cmsis_nn_dims input_dims = make_dims(input_nhwc);
        const cmsis_nn_dims filter_dims = make_dims(filter_nhwc);
        const cmsis_nn_dims output_dims = make_dims(output_nhwc);

        switch (backend)
        {
        case Backend::MVE:
            return arm_transpose_conv_s8_get_buffer_size_mve(&params, &input_dims, &filter_dims, &output_dims);

        case Backend::DSP:
        case Backend::SCALAR:
            return arm_transpose_conv_s8_get_buffer_size(&params, &input_dims, &filter_dims, &output_dims);
        }
        break;
    }
    }

    throw_invalid_combo(backend, data_type);
}

static int32_t transpose_conv_reverse_conv_buffer_size_impl(Backend backend,
                                                            DataType data_type,
                                                            const std::array<int32_t, 4> &input_nhwc,
                                                            const std::array<int32_t, 4> &filter_nhwc,
                                                            const cmsis_nn_transpose_conv_params &params)
{
    switch (data_type)
    {
    case DataType::A8W8: {
        const cmsis_nn_dims input_dims = make_dims(input_nhwc);
        const cmsis_nn_dims filter_dims = make_dims(filter_nhwc);

        switch (backend)
        {
        case Backend::MVE:
        case Backend::DSP:
        case Backend::SCALAR:
            return arm_transpose_conv_s8_get_reverse_conv_buffer_size(&params, &input_dims, &filter_dims);
        }
        break;
    }
    }

    throw_invalid_combo(backend, data_type);
}

static int32_t transpose_conv_buffer_size_py(py::args args, py::kwargs kwargs)
{
    constexpr size_t max_args = 13;
    if (args.size() > max_args)
    {
        throw py::type_error("transpose_conv_buffer_size accepts at most 13 arguments");
    }

    validate_kwargs(kwargs,
                    {"backend",
                     "data_type",
                     "input_nhwc",
                     "filter_nhwc",
                     "output_nhwc",
                     "padding_hw",
                     "stride_hw",
                     "dilation_hw",
                     "padding_offsets_hw",
                     "input_offset",
                     "output_offset",
                     "activation_min",
                     "activation_max"},
                    "transpose_conv_buffer_size");

    const Backend backend = parse_required_arg<Backend>(args, kwargs, 0, "backend");
    const DataType data_type = parse_required_arg<DataType>(args, kwargs, 1, "data_type");
    const auto input_nhwc = parse_required_arg<std::array<int32_t, 4>>(args, kwargs, 2, "input_nhwc");
    const auto filter_nhwc = parse_required_arg<std::array<int32_t, 4>>(args, kwargs, 3, "filter_nhwc");
    const auto output_nhwc = parse_required_arg<std::array<int32_t, 4>>(args, kwargs, 4, "output_nhwc");
    const auto padding_hw = parse_required_arg<std::array<int32_t, 2>>(args, kwargs, 5, "padding_hw");
    const auto stride_hw = parse_required_arg<std::array<int32_t, 2>>(args, kwargs, 6, "stride_hw");
    const auto dilation_hw = parse_required_arg<std::array<int32_t, 2>>(args, kwargs, 7, "dilation_hw");
    const auto padding_offsets_hw =
        parse_optional_arg<std::array<int32_t, 2>>(args, kwargs, 8, "padding_offsets_hw", {0, 0});
    const int32_t input_offset = parse_optional_arg<int32_t>(args, kwargs, 9, "input_offset", 0);
    const int32_t output_offset = parse_optional_arg<int32_t>(args, kwargs, 10, "output_offset", 0);
    const int32_t activation_min = parse_optional_arg<int32_t>(args, kwargs, 11, "activation_min", -128);
    const int32_t activation_max = parse_optional_arg<int32_t>(args, kwargs, 12, "activation_max", 127);

    const cmsis_nn_transpose_conv_params params = make_transpose_conv_params(padding_hw,
                                                                             stride_hw,
                                                                             dilation_hw,
                                                                             padding_offsets_hw,
                                                                             input_offset,
                                                                             output_offset,
                                                                             activation_min,
                                                                             activation_max);

    return transpose_conv_buffer_size_impl(backend, data_type, input_nhwc, filter_nhwc, output_nhwc, params);
}

static int32_t transpose_conv_reverse_conv_buffer_size_py(py::args args, py::kwargs kwargs)
{
    constexpr size_t max_args = 12;
    if (args.size() > max_args)
    {
        throw py::type_error("transpose_conv_reverse_conv_buffer_size accepts at most 12 arguments");
    }

    validate_kwargs(kwargs,
                    {"backend",
                     "data_type",
                     "input_nhwc",
                     "filter_nhwc",
                     "padding_hw",
                     "stride_hw",
                     "dilation_hw",
                     "padding_offsets_hw",
                     "input_offset",
                     "output_offset",
                     "activation_min",
                     "activation_max"},
                    "transpose_conv_reverse_conv_buffer_size");

    const Backend backend = parse_required_arg<Backend>(args, kwargs, 0, "backend");
    const DataType data_type = parse_required_arg<DataType>(args, kwargs, 1, "data_type");
    const auto input_nhwc = parse_required_arg<std::array<int32_t, 4>>(args, kwargs, 2, "input_nhwc");
    const auto filter_nhwc = parse_required_arg<std::array<int32_t, 4>>(args, kwargs, 3, "filter_nhwc");
    const auto padding_hw = parse_required_arg<std::array<int32_t, 2>>(args, kwargs, 4, "padding_hw");
    const auto stride_hw = parse_required_arg<std::array<int32_t, 2>>(args, kwargs, 5, "stride_hw");
    const auto dilation_hw = parse_optional_arg<std::array<int32_t, 2>>(args, kwargs, 6, "dilation_hw", {1, 1});
    const auto padding_offsets_hw =
        parse_optional_arg<std::array<int32_t, 2>>(args, kwargs, 7, "padding_offsets_hw", {0, 0});
    const int32_t input_offset = parse_optional_arg<int32_t>(args, kwargs, 8, "input_offset", 0);
    const int32_t output_offset = parse_optional_arg<int32_t>(args, kwargs, 9, "output_offset", 0);
    const int32_t activation_min = parse_optional_arg<int32_t>(args, kwargs, 10, "activation_min", -128);
    const int32_t activation_max = parse_optional_arg<int32_t>(args, kwargs, 11, "activation_max", 127);

    const cmsis_nn_transpose_conv_params params = make_transpose_conv_params(padding_hw,
                                                                             stride_hw,
                                                                             dilation_hw,
                                                                             padding_offsets_hw,
                                                                             input_offset,
                                                                             output_offset,
                                                                             activation_min,
                                                                             activation_max);

    return transpose_conv_reverse_conv_buffer_size_impl(backend, data_type, input_nhwc, filter_nhwc, params);
}

// ----------------------------------------------------------------------
// Pybind module bindings
// ----------------------------------------------------------------------

void transpose_conv_buffer_size(py::module_ &m)
{
    m.def("transpose_conv_buffer_size", &transpose_conv_buffer_size_py);

    m.def("transpose_conv_reverse_conv_buffer_size", &transpose_conv_reverse_conv_buffer_size_py);
}
