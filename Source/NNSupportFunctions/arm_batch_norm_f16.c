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
 * Title:        arm_batch_norm_f16.c
 * Description:  Batch normalization for float16 tensors
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F16

/**
 * @ingroup Public
 */

/**
 * @addtogroup NNSupport
 * @{
 */

/* Refer header file for details. */
arm_cmsis_nn_status arm_batch_norm_f16(const float16_t *input,
                                       float16_t *output,
                                       const float16_t *scale,
                                       const float16_t *bias,
                                       const cmsis_nn_dims *input_dims,
                                       arm_nn_tensor_layout layout)
{
    if (!input || !output || !scale || !bias || !input_dims)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t n = input_dims->n;
    const int32_t h = input_dims->h;
    const int32_t w = input_dims->w;
    const int32_t c = input_dims->c;

    if (n <= 0 || h <= 0 || w <= 0 || c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    if (layout != ARM_NN_LAYOUT_NHWC)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    for (int32_t bn = 0; bn < n; ++bn)
    {
        for (int32_t y = 0; y < h; ++y)
        {
            for (int32_t x = 0; x < w; ++x)
            {
                const size_t base = (((size_t)bn * (size_t)h + (size_t)y) * (size_t)w + (size_t)x) * (size_t)c;
    #if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)
                for (int32_t ch = 0; ch < c; ch += 8)
                {
                    const mve_pred16_t p = vctp16q((uint32_t)(c - ch));
                    const float16x8_t v_in = vld1q_z(input + base + (size_t)ch, p);
                    const float16x8_t v_scale = vld1q_z(scale + ch, p);
                    const float16x8_t v_bias = vld1q_z(bias + ch, p);
                    const float16x8_t v_out = vfmaq(v_bias, v_in, v_scale);
                    vst1q_p(output + base + (size_t)ch, v_out, p);
                }
    #else
                for (int32_t ch = 0; ch < c; ++ch)
                {
                    const _Float16 in_val = (_Float16)input[base + (size_t)ch];
                    const _Float16 scale_val = (_Float16)scale[ch];
                    const _Float16 bias_val = (_Float16)bias[ch];
                    output[base + (size_t)ch] = (float16_t)(in_val * scale_val + bias_val);
                }
    #endif
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

/** @} */

#endif /* ARM_NN_ENABLE_F16 */
