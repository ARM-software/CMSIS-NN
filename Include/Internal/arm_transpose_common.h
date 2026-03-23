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
 * Title:        arm_transpose_common.h
 * Description:  Shared transpose helper templates for CMSIS-NN
 *
 * $Date:        2 April 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_TRANSPOSE_COMMON_H
#define ARM_TRANSPOSE_COMMON_H

#include "arm_nn_types_flt.h"

#define ARM_TRANSPOSE_DEFINE(FUNC_NAME, SCALAR_T, PARAMS_T, LAYOUT_T, TRANSPOSE_2D_FUNC)                               \
    static void FUNC_NAME##_copy_elems(const SCALAR_T *input, SCALAR_T *output, size_t elems)                          \
    {                                                                                                                  \
        for (size_t i = 0; i < elems; ++i)                                                                             \
        {                                                                                                              \
            output[i] = input[i];                                                                                      \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    static arm_cmsis_nn_status FUNC_NAME##_transpose_2d(                                                               \
        const SCALAR_T *input, SCALAR_T *output, int32_t rows, int32_t cols)                                           \
    {                                                                                                                  \
        return TRANSPOSE_2D_FUNC(input, output, rows, cols);                                                           \
    }                                                                                                                  \
                                                                                                                       \
    static arm_cmsis_nn_status FUNC_NAME##_transpose_swap_last2_3d(                                                    \
        const SCALAR_T *input, SCALAR_T *output, int32_t d0, int32_t d1, int32_t d2)                                   \
    {                                                                                                                  \
        const size_t in_stride0 = (size_t)d1 * (size_t)d2;                                                             \
        const size_t out_stride0 = (size_t)d2 * (size_t)d1;                                                            \
        for (int32_t i0 = 0; i0 < d0; ++i0)                                                                            \
        {                                                                                                              \
            const SCALAR_T *in_base = input + (size_t)i0 * in_stride0;                                                 \
            SCALAR_T *out_base = output + (size_t)i0 * out_stride0;                                                    \
            FUNC_NAME##_transpose_2d(in_base, out_base, d1, d2);                                                       \
        }                                                                                                              \
        return ARM_CMSIS_NN_SUCCESS;                                                                                   \
    }                                                                                                                  \
                                                                                                                       \
    static arm_cmsis_nn_status FUNC_NAME##_transpose_swap_last2_4d(                                                    \
        const SCALAR_T *input, SCALAR_T *output, int32_t d0, int32_t d1, int32_t d2, int32_t d3)                       \
    {                                                                                                                  \
        const size_t in_stride0 = (size_t)d1 * (size_t)d2 * (size_t)d3;                                                \
        const size_t in_stride1 = (size_t)d2 * (size_t)d3;                                                             \
        const size_t in_stride2 = (size_t)d3;                                                                          \
        const size_t out_stride0 = (size_t)d1 * (size_t)d3 * (size_t)d2;                                               \
        const size_t out_stride1 = (size_t)d3 * (size_t)d2;                                                            \
        const size_t out_stride2 = (size_t)d2;                                                                         \
        for (int32_t i0 = 0; i0 < d0; ++i0)                                                                            \
        {                                                                                                              \
            for (int32_t i1 = 0; i1 < d1; ++i1)                                                                        \
            {                                                                                                          \
                const SCALAR_T *in_base = input + (size_t)i0 * in_stride0 + (size_t)i1 * in_stride1;                   \
                SCALAR_T *out_base = output + (size_t)i0 * out_stride0 + (size_t)i1 * out_stride1;                     \
                (void)in_stride2;                                                                                      \
                (void)out_stride2;                                                                                     \
                FUNC_NAME##_transpose_2d(in_base, out_base, d2, d3);                                                   \
            }                                                                                                          \
        }                                                                                                              \
        return ARM_CMSIS_NN_SUCCESS;                                                                                   \
    }                                                                                                                  \
                                                                                                                       \
    arm_cmsis_nn_status FUNC_NAME(const cmsis_nn_context *ctx,                                                         \
                                  const PARAMS_T *params,                                                              \
                                  const cmsis_nn_dims *input_dims,                                                     \
                                  const SCALAR_T *input,                                                               \
                                  const cmsis_nn_dims *output_dims,                                                    \
                                  SCALAR_T *output)                                                                    \
    {                                                                                                                  \
        (void)ctx;                                                                                                     \
        if (!params || !input_dims || !input || !output_dims || !output)                                               \
        {                                                                                                              \
            return ARM_CMSIS_NN_ARG_ERROR;                                                                             \
        }                                                                                                              \
                                                                                                                       \
        const int32_t num_dims = params->num_dims;                                                                     \
        if (num_dims < 1 || num_dims > 4)                                                                              \
        {                                                                                                              \
            return ARM_CMSIS_NN_ARG_ERROR;                                                                             \
        }                                                                                                              \
                                                                                                                       \
        const LAYOUT_T layout = params->layout;                                                                        \
        if (layout != ARM_NN_LAYOUT_NHWC)                                                                              \
        {                                                                                                              \
            return ARM_CMSIS_NN_ARG_ERROR;                                                                             \
        }                                                                                                              \
                                                                                                                       \
        int32_t in_all[4];                                                                                             \
        int32_t out_all[4];                                                                                            \
        in_all[0] = input_dims->n;                                                                                     \
        in_all[1] = input_dims->h;                                                                                     \
        in_all[2] = input_dims->w;                                                                                     \
        in_all[3] = input_dims->c;                                                                                     \
        out_all[0] = output_dims->n;                                                                                   \
        out_all[1] = output_dims->h;                                                                                   \
        out_all[2] = output_dims->w;                                                                                   \
        out_all[3] = output_dims->c;                                                                                   \
                                                                                                                       \
        int32_t in_dims[4] = {1, 1, 1, 1};                                                                             \
        int32_t out_dims[4] = {1, 1, 1, 1};                                                                            \
        int32_t perm[4] = {0, 1, 2, 3};                                                                                \
        for (int32_t i = 0; i < num_dims; ++i)                                                                         \
        {                                                                                                              \
            in_dims[i] = in_all[i];                                                                                    \
            out_dims[i] = out_all[i];                                                                                  \
            perm[i] = params->perm[i];                                                                                 \
            if (perm[i] < 0 || perm[i] >= num_dims)                                                                    \
            {                                                                                                          \
                return ARM_CMSIS_NN_ARG_ERROR;                                                                         \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        if (num_dims == 1)                                                                                             \
        {                                                                                                              \
            size_t elems = (size_t)in_dims[0];                                                                         \
            FUNC_NAME##_copy_elems(input, output, elems);                                                              \
            return ARM_CMSIS_NN_SUCCESS;                                                                               \
        }                                                                                                              \
                                                                                                                       \
        if (num_dims == 2)                                                                                             \
        {                                                                                                              \
            if (perm[0] == 0 && perm[1] == 1)                                                                          \
            {                                                                                                          \
                size_t elems = (size_t)in_dims[0] * (size_t)in_dims[1];                                                \
                FUNC_NAME##_copy_elems(input, output, elems);                                                          \
                return ARM_CMSIS_NN_SUCCESS;                                                                           \
            }                                                                                                          \
            if (perm[0] == 1 && perm[1] == 0)                                                                          \
            {                                                                                                          \
                return FUNC_NAME##_transpose_2d(input, output, in_dims[0], in_dims[1]);                                \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        if (num_dims == 3 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1)                                             \
        {                                                                                                              \
            return FUNC_NAME##_transpose_swap_last2_3d(input, output, in_dims[0], in_dims[1], in_dims[2]);             \
        }                                                                                                              \
                                                                                                                       \
        if (num_dims == 4 && perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2)                             \
        {                                                                                                              \
            return FUNC_NAME##_transpose_swap_last2_4d(input, output, in_dims[0], in_dims[1], in_dims[2], in_dims[3]); \
        }                                                                                                              \
                                                                                                                       \
        size_t in_strides[4] = {0, 0, 0, 0};                                                                           \
        size_t out_strides[4] = {0, 0, 0, 0};                                                                          \
        in_strides[num_dims - 1] = 1;                                                                                  \
        out_strides[num_dims - 1] = 1;                                                                                 \
        for (int32_t i = num_dims - 2; i >= 0; --i)                                                                    \
        {                                                                                                              \
            in_strides[i] = in_strides[i + 1] * (size_t)in_dims[i + 1];                                                \
            out_strides[i] = out_strides[i + 1] * (size_t)out_dims[i + 1];                                             \
        }                                                                                                              \
                                                                                                                       \
        int32_t identity = 1;                                                                                          \
        int32_t inv_perm[4] = {0, 1, 2, 3};                                                                            \
        for (int32_t i = 0; i < num_dims; ++i)                                                                         \
        {                                                                                                              \
            identity &= (perm[i] == i);                                                                                \
            inv_perm[perm[i]] = i;                                                                                     \
        }                                                                                                              \
        if (identity)                                                                                                  \
        {                                                                                                              \
            size_t elems = 1;                                                                                          \
            for (int32_t i = 0; i < num_dims; ++i)                                                                     \
            {                                                                                                          \
                elems *= (size_t)in_dims[i];                                                                           \
            }                                                                                                          \
            FUNC_NAME##_copy_elems(input, output, elems);                                                              \
            return ARM_CMSIS_NN_SUCCESS;                                                                               \
        }                                                                                                              \
                                                                                                                       \
        if (num_dims == 3)                                                                                             \
        {                                                                                                              \
            const size_t out_stride0 = out_strides[inv_perm[0]];                                                       \
            const size_t out_stride1 = out_strides[inv_perm[1]];                                                       \
            const size_t out_stride2 = out_strides[inv_perm[2]];                                                       \
            for (int32_t i0 = 0; i0 < in_dims[0]; ++i0)                                                                \
            {                                                                                                          \
                const SCALAR_T *in0 = input + (size_t)i0 * in_strides[0];                                              \
                SCALAR_T *out0 = output + (size_t)i0 * out_stride0;                                                    \
                for (int32_t i1 = 0; i1 < in_dims[1]; ++i1)                                                            \
                {                                                                                                      \
                    const SCALAR_T *in1 = in0 + (size_t)i1 * in_strides[1];                                            \
                    SCALAR_T *out1 = out0 + (size_t)i1 * out_stride1;                                                  \
                    for (int32_t i2 = 0; i2 < in_dims[2]; ++i2)                                                        \
                    {                                                                                                  \
                        out1[(size_t)i2 * out_stride2] = in1[i2];                                                      \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            return ARM_CMSIS_NN_SUCCESS;                                                                               \
        }                                                                                                              \
                                                                                                                       \
        const size_t out_stride0 = out_strides[inv_perm[0]];                                                           \
        const size_t out_stride1 = out_strides[inv_perm[1]];                                                           \
        const size_t out_stride2 = out_strides[inv_perm[2]];                                                           \
        const size_t out_stride3 = out_strides[inv_perm[3]];                                                           \
        for (int32_t i0 = 0; i0 < in_dims[0]; ++i0)                                                                    \
        {                                                                                                              \
            const SCALAR_T *in0 = input + (size_t)i0 * in_strides[0];                                                  \
            SCALAR_T *out0 = output + (size_t)i0 * out_stride0;                                                        \
            for (int32_t i1 = 0; i1 < in_dims[1]; ++i1)                                                                \
            {                                                                                                          \
                const SCALAR_T *in1 = in0 + (size_t)i1 * in_strides[1];                                                \
                SCALAR_T *out1 = out0 + (size_t)i1 * out_stride1;                                                      \
                for (int32_t i2 = 0; i2 < in_dims[2]; ++i2)                                                            \
                {                                                                                                      \
                    const SCALAR_T *in2 = in1 + (size_t)i2 * in_strides[2];                                            \
                    SCALAR_T *out2 = out1 + (size_t)i2 * out_stride2;                                                  \
                    for (int32_t i3 = 0; i3 < in_dims[3]; ++i3)                                                        \
                    {                                                                                                  \
                        out2[(size_t)i3 * out_stride3] = in2[i3];                                                      \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        return ARM_CMSIS_NN_SUCCESS;                                                                                   \
    }

#endif /* ARM_TRANSPOSE_COMMON_H */
