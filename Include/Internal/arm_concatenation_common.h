/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_concatenation_common.h
 * Description:  Shared float/int concatenation helper templates
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_CONCATENATION_COMMON_H
#define ARM_CONCATENATION_COMMON_H

#include "Internal/arm_nn_compiler.h"
#include "arm_nn_math_types.h"

#define ARM_CONCATENATION_DEFINE(SUFFIX, TYPE)                                                                         \
    void arm_concatenation_##SUFFIX##_x(const TYPE *input,                                                             \
                                        const int32_t input_x,                                                         \
                                        const int32_t input_y,                                                         \
                                        const int32_t input_z,                                                         \
                                        const int32_t input_w,                                                         \
                                        TYPE *output,                                                                  \
                                        const int32_t output_x,                                                        \
                                        const uint32_t offset_x)                                                       \
    {                                                                                                                  \
        const uint32_t num_iterations = (uint32_t)input_y * (uint32_t)input_z * (uint32_t)input_w;                     \
        const TYPE *in = input;                                                                                        \
        TYPE *out = output + (size_t)offset_x;                                                                         \
        const uint32_t input_copy_elems = (uint32_t)input_x;                                                           \
        const uint32_t output_stride_elems = (uint32_t)output_x;                                                       \
        for (uint32_t i = 0; i < num_iterations; ++i)                                                                  \
        {                                                                                                              \
            arm_memcpy_##SUFFIX(out, in, input_copy_elems);                                                            \
            in += input_copy_elems;                                                                                    \
            out += output_stride_elems;                                                                                \
        }                                                                                                              \
    }                                                                                                                  \
    void arm_concatenation_##SUFFIX##_y(const TYPE *input,                                                             \
                                        const int32_t input_x,                                                         \
                                        const int32_t input_y,                                                         \
                                        const int32_t input_z,                                                         \
                                        const int32_t input_w,                                                         \
                                        TYPE *output,                                                                  \
                                        const int32_t output_y,                                                        \
                                        const uint32_t offset_y)                                                       \
    {                                                                                                                  \
        const uint32_t num_iterations = (uint32_t)input_z * (uint32_t)input_w;                                         \
        const uint32_t input_copy_elems = (uint32_t)input_x * (uint32_t)input_y;                                       \
        const uint32_t output_stride_elems = (uint32_t)input_x * (uint32_t)output_y;                                   \
        const TYPE *in = input;                                                                                        \
        TYPE *out = output + (size_t)offset_y * (size_t)input_x;                                                       \
        for (uint32_t i = 0; i < num_iterations; ++i)                                                                  \
        {                                                                                                              \
            arm_memcpy_##SUFFIX(out, in, input_copy_elems);                                                            \
            in += input_copy_elems;                                                                                    \
            out += output_stride_elems;                                                                                \
        }                                                                                                              \
    }                                                                                                                  \
    void arm_concatenation_##SUFFIX##_z(const TYPE *input,                                                             \
                                        const int32_t input_x,                                                         \
                                        const int32_t input_y,                                                         \
                                        const int32_t input_z,                                                         \
                                        const int32_t input_w,                                                         \
                                        TYPE *output,                                                                  \
                                        const int32_t output_z,                                                        \
                                        const uint32_t offset_z)                                                       \
    {                                                                                                                  \
        const uint32_t input_copy_elems = (uint32_t)input_x * (uint32_t)input_y * (uint32_t)input_z;                   \
        const uint32_t output_stride_elems = (uint32_t)input_x * (uint32_t)input_y * (uint32_t)output_z;               \
        const TYPE *in = input;                                                                                        \
        TYPE *out = output + (size_t)offset_z * (size_t)input_x * (size_t)input_y;                                     \
        for (uint32_t i = 0; i < (uint32_t)input_w; ++i)                                                               \
        {                                                                                                              \
            arm_memcpy_##SUFFIX(out, in, input_copy_elems);                                                            \
            in += input_copy_elems;                                                                                    \
            out += output_stride_elems;                                                                                \
        }                                                                                                              \
    }                                                                                                                  \
    void arm_concatenation_##SUFFIX##_w(const TYPE *input,                                                             \
                                        const int32_t input_x,                                                         \
                                        const int32_t input_y,                                                         \
                                        const int32_t input_z,                                                         \
                                        const int32_t input_w,                                                         \
                                        TYPE *output,                                                                  \
                                        const uint32_t offset_w)                                                       \
    {                                                                                                                  \
        const uint32_t input_copy_elems =                                                                              \
            (uint32_t)input_x * (uint32_t)input_y * (uint32_t)input_z * (uint32_t)input_w;                             \
        TYPE *out = output + (size_t)offset_w * (size_t)input_x * (size_t)input_y * (size_t)input_z;                   \
        arm_memcpy_##SUFFIX(out, input, input_copy_elems);                                                             \
    }

#endif /* ARM_CONCATENATION_COMMON_H */
