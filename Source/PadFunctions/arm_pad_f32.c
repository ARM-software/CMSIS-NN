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
 * Title:        arm_pad_f32.c
 * Description:  Pad a float32 tensor
 *
 * $Date:        19 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

/**
 * @ingroup Public
 */

/**
 * @addtogroup Pad
 * @{
 */

/* Refer header file for details. */
arm_cmsis_nn_status arm_pad_f32(const float32_t *input,
                                float32_t *output,
                                float32_t pad_value,
                                const cmsis_nn_dims *input_size,
                                const cmsis_nn_dims *pre_pad,
                                const cmsis_nn_dims *post_pad)
{
    if (!input || !output || !input_size || !pre_pad || !post_pad)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const cmsis_nn_dims output_size = {pre_pad->n + input_size->n + post_pad->n,
                                       pre_pad->h + input_size->h + post_pad->h,
                                       pre_pad->w + input_size->w + post_pad->w,
                                       pre_pad->c + input_size->c + post_pad->c};

    if (output_size.n <= 0 || output_size.h <= 0 || output_size.w <= 0 || output_size.c <= 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const uint32_t batch_block_size = (uint32_t)((size_t)output_size.h * (size_t)output_size.w * (size_t)output_size.c);
    const uint32_t row_block_size = (uint32_t)((size_t)output_size.w * (size_t)output_size.c);
    const uint32_t col_block_size = (uint32_t)output_size.c;

    arm_memset_f32(output, pad_value, batch_block_size * (uint32_t)pre_pad->n);
    output += batch_block_size * (uint32_t)pre_pad->n;
    for (int32_t b = 0; b < input_size->n; ++b)
    {
        arm_memset_f32(output, pad_value, row_block_size * (uint32_t)pre_pad->h);
        output += row_block_size * (uint32_t)pre_pad->h;
        for (int32_t y = 0; y < input_size->h; ++y)
        {
            arm_memset_f32(output, pad_value, col_block_size * (uint32_t)pre_pad->w);
            output += col_block_size * (uint32_t)pre_pad->w;
            if (input_size->c == output_size.c)
            {
                arm_memcpy_f32(output, input, (uint32_t)input_size->w * (uint32_t)input_size->c);
                output += (uint32_t)input_size->w * (uint32_t)input_size->c;
                input += (uint32_t)input_size->w * (uint32_t)input_size->c;
            }
            else
            {
                for (int32_t x = 0; x < input_size->w; ++x)
                {
                    arm_memset_f32(output, pad_value, (uint32_t)pre_pad->c);
                    output += pre_pad->c;

                    arm_memcpy_f32(output, input, (uint32_t)input_size->c);
                    output += input_size->c;
                    input += input_size->c;

                    arm_memset_f32(output, pad_value, (uint32_t)post_pad->c);
                    output += post_pad->c;
                }
            }

            arm_memset_f32(output, pad_value, col_block_size * (uint32_t)post_pad->w);
            output += col_block_size * (uint32_t)post_pad->w;
        }

        arm_memset_f32(output, pad_value, row_block_size * (uint32_t)post_pad->h);
        output += row_block_size * (uint32_t)post_pad->h;
    }
    arm_memset_f32(output, pad_value, batch_block_size * (uint32_t)post_pad->n);

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Pad group
 */

#endif /* ARM_NN_ENABLE_F32 */
