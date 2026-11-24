/*
 * SPDX-FileCopyrightText: Copyright 2026 Google LLC
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
 * Title:        arm_quantize_f32_s8.c
 * Description:  Affine quantize float32 vector to s8
 *
 * $Date:        24 November 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_nn_quantize_common.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    #if defined(__GNUC__) || defined(__ARMCC_VERSION)
__attribute__((noinline))
    #endif
static void arm_quantize_f32_s8_scalar(const float *input,
                                       int8_t *output,
                                       const float scale,
                                       const int32_t zero_point,
                                       const int32_t count)
{
    for (int32_t i = 0; i < count; ++i)
    {
        output[i] = (int8_t)arm_nn_quantize_scalar_elem_f32(input[i], scale, zero_point, NN_Q7_MIN, NN_Q7_MAX);
    }
}
#endif

/*
 * Affine quantize float32 vector to s8.
 *
 * Refer header file for details.
 */
arm_cmsis_nn_status arm_quantize_f32_s8(const float *input,
                                        int8_t *output,
                                        const float scale,
                                        const int32_t zero_point,
                                        const int32_t block_size)
{
    if (block_size < 0 || !arm_nn_quantize_f32_is_valid_scale(scale))
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    if (block_size == 0)
    {
        return ARM_CMSIS_NN_SUCCESS;
    }
    if (!input || !output)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (arm_nn_quantize_can_use_mve_f32(scale, zero_point))
    {
        const float inv_scale = 1.0f / scale;
        const float32x4_t half_vec = vdupq_n_f32(0.5f);
        const int32x4_t min_vec = vdupq_n_s32(NN_Q7_MIN);
        const int32x4_t max_vec = vdupq_n_s32(NN_Q7_MAX);
        int32_t count = block_size;

        while (count > 0)
        {
            const mve_pred16_t p = vctp32q((uint32_t)count);
            const float32x4_t v = vldrwq_z_f32(input, p);
            const float32x4_t scaled = vmulq_n_f32(v, inv_scale);
            const int32x4_t rounded = vcvtaq_s32_f32(scaled);

            /* Multiplying by 1/scale can differ from val/scale by up to 1 ULP.
             * Fall back to exact scalar division only when any active lane is
             * within 2 ULP of a half-integer tie (n + 0.5). */
            const float32x4_t diff = vabdq_f32(scaled, vcvtq_f32_s32(rounded));
            const float32x4_t dist_to_half = vabdq_f32(half_vec, diff);
            const float32x4_t tol = vmulq_n_f32(vabsq_f32(scaled), 0x1.0p-22f);
            const mve_pred16_t near_tie = vcmpleq_m_f32(dist_to_half, tol, p);

            if (near_tie == 0)
            {
                int32x4_t q = vqaddq_n_s32(rounded, zero_point);
                q = vmaxq_s32(q, min_vec);
                q = vminq_s32(q, max_vec);
                vstrbq_p_s32(output, q, p);
            }
            else
            {
                const int32_t lanes = (count < 4) ? count : 4;
                arm_quantize_f32_s8_scalar(input, output, scale, zero_point, lanes);
            }

            input += 4;
            output += 4;
            count -= 4;
        }

        return ARM_CMSIS_NN_SUCCESS;
    }

    arm_quantize_f32_s8_scalar(input, output, scale, zero_point, block_size);
    return ARM_CMSIS_NN_SUCCESS;
#else
    for (int32_t i = 0; i < block_size; ++i)
    {
        output[i] = (int8_t)arm_nn_quantize_scalar_elem_f32(input[i], scale, zero_point, NN_Q7_MIN, NN_Q7_MAX);
    }

    return ARM_CMSIS_NN_SUCCESS;
#endif
}

/**
 * @} end of groupElementwise group
 */
