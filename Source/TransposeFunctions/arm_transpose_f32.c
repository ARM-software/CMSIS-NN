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
 * Title:        arm_transpose_f32.c
 * Description:  Transpose a float32 tensor
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_transpose_common.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Transpose
 * @{
 */

static arm_cmsis_nn_status
arm_transpose_2d_kernel_f32(const float32_t *input, float32_t *output, int32_t rows, int32_t cols)
{
    #if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    if (rows > 0 && cols > 0 && (size_t)rows * (size_t)cols >= 128U && (size_t)cols <= ((size_t)UINT32_MAX / 3U))
    {
        const uint32_t cols_u32 = (uint32_t)cols;
        const uint32x4_t row_offsets = vmulq(vidupq_u32((uint32_t)0, 1), cols_u32);

        for (int32_t c = 0; c < cols; ++c)
        {
            const float32_t *input_col = input + c;
            float32_t *output_col = output + (size_t)c * (size_t)rows;

            for (int32_t r = 0; r < rows; r += 4)
            {
                const mve_pred16_t p = vctp32q((uint32_t)(rows - r));
                const float32x4_t vin = vldrwq_gather_shifted_offset_z(input_col, row_offsets, p);
                vst1q_p(output_col, vin, p);
                input_col += (size_t)4U * (size_t)cols;
                output_col += 4;
            }
        }

        return ARM_CMSIS_NN_SUCCESS;
    }
    #endif

    for (int32_t r = 0; r < rows; ++r)
    {
        const float32_t *in_row = input + (size_t)r * (size_t)cols;
        for (int32_t c = 0; c < cols; ++c)
        {
            output[(size_t)c * (size_t)rows + (size_t)r] = in_row[c];
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}

ARM_TRANSPOSE_DEFINE(arm_transpose_f32,
                     float32_t,
                     cmsis_nn_transpose_params_f32,
                     arm_nn_tensor_layout,
                     arm_transpose_2d_kernel_f32)

/**
 * @} end of Transpose group
 */

#endif /* ARM_NN_ENABLE_F32 */
