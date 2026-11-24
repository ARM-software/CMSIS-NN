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
 * Title:        arm_requantize_s8.c
 * Description:  Requantize s8 vector to s8
 *
 * $Date:        24 November 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup groupElementwise
 * @{
 */

/*
 * Requantize s8 vector to s8.
 *
 * Refer header file for details.
 */
arm_cmsis_nn_status arm_requantize_s8(const int8_t *input,
                                      int8_t *output,
                                      const int32_t input_offset,
                                      const int32_t out_offset,
                                      const int32_t out_mult,
                                      const int32_t out_shift,
                                      const int32_t out_activation_min,
                                      const int32_t out_activation_max,
                                      const int32_t block_size)
{
    if (block_size < 0 || out_mult < 0 || out_shift < -30 || out_shift > 22 ||
        out_activation_min > out_activation_max || out_activation_min < NN_Q7_MIN || out_activation_max > NN_Q7_MAX)
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

#if defined(ARM_MATH_MVEI)
    int32_t count = block_size;
    const int32x4_t zero_vec = vdupq_n_s32(0);
    const int32x4_t min_vec = vdupq_n_s32(out_activation_min);
    const int32x4_t max_vec = vdupq_n_s32(out_activation_max);

    while (count > 0)
    {
        const mve_pred16_t p = vctp32q((uint32_t)count);
        int32x4_t val = vldrbq_z_s32(input, p);
        val = vaddq_m_n_s32(zero_vec, val, input_offset, p);
        val = arm_requantize_mve(val, out_mult, out_shift);
        val = vqaddq_n_s32(val, out_offset);
        val = vmaxq_s32(val, min_vec);
        val = vminq_s32(val, max_vec);
        vstrbq_p_s32(output, val, p);

        input += 4;
        output += 4;
        count -= 4;
    }
#else
    for (int32_t i = 0; i < block_size; ++i)
    {
        const int32_t shifted = (int32_t)input[i] + input_offset;
        int64_t val = (int64_t)arm_nn_requantize(shifted, out_mult, out_shift) + (int64_t)out_offset;
        if (val < (int64_t)out_activation_min)
        {
            val = (int64_t)out_activation_min;
        }
        else if (val > (int64_t)out_activation_max)
        {
            val = (int64_t)out_activation_max;
        }
        output[i] = (int8_t)val;
    }
#endif

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of groupElementwise group
 */
