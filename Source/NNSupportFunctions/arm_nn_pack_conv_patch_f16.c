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
 * Title:        arm_nn_pack_conv_patch_f16.c
 * Description:  Support: pack one convolution patch row (float16)
 *
 * $Date:        2 Mar 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup supportConvolution
 * @{
 */

void arm_nn_pack_conv_patch_f16(const float16_t *__RESTRICT input,
                                int32_t in_h,
                                int32_t in_w,
                                int32_t in_c,
                                int32_t kernel_h,
                                int32_t kernel_w,
                                int32_t stride_h,
                                int32_t stride_w,
                                int32_t pad_h,
                                int32_t pad_w,
                                int32_t dilation_h,
                                int32_t dilation_w,
                                int32_t out_y,
                                int32_t out_x,
                                float16_t pad_value,
                                float16_t *__RESTRICT patch_row)
{
    const int32_t base_y = out_y * stride_h - pad_h;
    const int32_t base_x = out_x * stride_w - pad_w;

    /* This is effectively one im2row-style packed patch. */
    int32_t dst = 0;
    for (int32_t kh = 0; kh < kernel_h; ++kh)
    {
        const int32_t in_y = base_y + kh * dilation_h;
        for (int32_t kw = 0; kw < kernel_w; ++kw)
        {
            const int32_t in_x = base_x + kw * dilation_w;
            if ((uint32_t)in_y < (uint32_t)in_h && (uint32_t)in_x < (uint32_t)in_w)
            {
                const int32_t src = ((in_y * in_w) + in_x) * in_c;
                arm_memcpy_f16(patch_row + dst, input + src, (uint32_t)in_c);
            }
            else
            {
                arm_memset_f16(patch_row + dst, pad_value, (uint32_t)in_c);
            }
            dst += in_c;
        }
    }
}

/**
 * @} end of supportConvolution group
 */
