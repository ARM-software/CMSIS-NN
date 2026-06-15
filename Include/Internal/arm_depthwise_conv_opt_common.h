/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
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
 * Title:        arm_depthwise_conv_opt_common.h
 * Description:  Shared float depthwise-convolution specialization helpers
 *
 * $Date:        30 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_DEPTHWISE_CONV_OPT_COMMON_H
#define ARM_DEPTHWISE_CONV_OPT_COMMON_H

#include "Internal/arm_nn_compiler.h"

#define ARM_DW_SPEC_ENTRY(MATCH_FN, CALL_FN)                                                                           \
    {                                                                                                                  \
        (MATCH_FN), (CALL_FN)                                                                                          \
    }

#define ARM_DW_ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

/* NHWC input tensor linear index: ((y * width) + x) * channels + channel. */
static inline int32_t
arm_depthwise_conv_input_index_nhwc(int32_t x, int32_t y, int32_t c, int32_t input_x, int32_t input_ch)
{
    return (y * input_x + x) * input_ch + c;
}

/* NHWC output tensor linear index: ((y * width) + x) * channels + channel. */
static inline int32_t
arm_depthwise_conv_output_index_nhwc(int32_t out_x, int32_t out_y, int32_t out_ch, int32_t output_x, int32_t output_ch)
{
    return (out_y * output_x + out_x) * output_ch + out_ch;
}

#define ARM_DW_DISPATCH(TABLE, COUNT, ...)                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        for (size_t _i = 0; _i < (COUNT); ++_i)                                                                        \
        {                                                                                                              \
            if ((TABLE)[_i].match(__VA_ARGS__))                                                                        \
            {                                                                                                          \
                return (TABLE)[_i].call(__VA_ARGS__);                                                                  \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#endif /* ARM_DEPTHWISE_CONV_OPT_COMMON_H */
