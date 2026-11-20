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

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_arm_convolve_f32.c"
#include "unity.h"

#ifdef USING_FVP_CORSTONE_300
extern void uart_init(void);
#endif

void setUp(void)
{
#ifdef USING_FVP_CORSTONE_300
    uart_init();
#endif
}

void tearDown(void) {}
void test_conv_basic_f32_arm_convolve_f32(void) { conv_basic_f32_arm_convolve_f32(); }
void test_conv_basic_nhwc_f32_arm_convolve_f32(void) { conv_basic_nhwc_f32_arm_convolve_f32(); }
void test_conv_1x1_stride2_nhwc_f32_arm_convolve_f32(void) { conv_1x1_stride2_nhwc_f32_arm_convolve_f32(); }
void test_conv_k3_opt_f32_arm_convolve_f32(void) { conv_k3_opt_f32_arm_convolve_f32(); }
void test_conv_k5_opt_f32_arm_convolve_f32(void) { conv_k5_opt_f32_arm_convolve_f32(); }
void test_conv_k3_opt_nhwc_tuned_f32_arm_convolve_f32(void) { conv_k3_opt_nhwc_tuned_f32_arm_convolve_f32(); }
void test_conv_k5_opt_nhwc_tuned_f32_arm_convolve_f32(void) { conv_k5_opt_nhwc_tuned_f32_arm_convolve_f32(); }
void test_conv_kernel_2x2_f32_arm_convolve_f32(void) { conv_kernel_2x2_f32_arm_convolve_f32(); }
void test_conv_kernel_3x3_pad1_f32_arm_convolve_f32(void) { conv_kernel_3x3_pad1_f32_arm_convolve_f32(); }
void test_conv_match_basic_f32_arm_convolve_f32(void) { conv_match_basic_f32_arm_convolve_f32(); }
void test_conv_match_stride2pad1_f32_arm_convolve_f32(void) { conv_match_stride2pad1_f32_arm_convolve_f32(); }
void test_conv_match_conv_2_f32_arm_convolve_f32(void) { conv_match_conv_2_f32_arm_convolve_f32(); }
void test_conv_match_conv_3_f32_arm_convolve_f32(void) { conv_match_conv_3_f32_arm_convolve_f32(); }
void test_conv_match_conv_4_f32_arm_convolve_f32(void) { conv_match_conv_4_f32_arm_convolve_f32(); }
void test_conv_match_conv_5_f32_arm_convolve_f32(void)
{
#if defined(USING_FVP_CORSTONE_300)
    TEST_IGNORE_MESSAGE("Host-only due to FVP image size");
#else
    conv_match_conv_5_f32_arm_convolve_f32();
#endif
}
void test_conv_match_out_activation_f32_arm_convolve_f32(void) { conv_match_out_activation_f32_arm_convolve_f32(); }
void test_conv_match_dilation_golden_f32_arm_convolve_f32(void) { conv_match_dilation_golden_f32_arm_convolve_f32(); }
void test_conv_match_2x2_dilation_f32_arm_convolve_f32(void) { conv_match_2x2_dilation_f32_arm_convolve_f32(); }
void test_conv_match_2x3_dilation_f32_arm_convolve_f32(void) { conv_match_2x3_dilation_f32_arm_convolve_f32(); }
void test_conv_match_1x1_basic_f32_arm_convolve_f32(void) { conv_match_1x1_basic_f32_arm_convolve_f32(); }
void test_conv_match_1x1_stride_x_f32_arm_convolve_f32(void) { conv_match_1x1_stride_x_f32_arm_convolve_f32(); }
void test_conv_match_1x1_stride_x_y_f32_arm_convolve_f32(void) { conv_match_1x1_stride_x_y_f32_arm_convolve_f32(); }
void test_conv_match_1x1_stride_x_y_1_f32_arm_convolve_f32(void) { conv_match_1x1_stride_x_y_1_f32_arm_convolve_f32(); }
void test_conv_match_1x1_stride_x_y_2_f32_arm_convolve_f32(void) { conv_match_1x1_stride_x_y_2_f32_arm_convolve_f32(); }
void test_conv_match_1xn_1_f32_arm_convolve_f32(void) { conv_match_1xn_1_f32_arm_convolve_f32(); }
void test_conv_match_1xn_2_f32_arm_convolve_f32(void) { conv_match_1xn_2_f32_arm_convolve_f32(); }
void test_conv_match_1xn_3_f32_arm_convolve_f32(void) { conv_match_1xn_3_f32_arm_convolve_f32(); }
void test_conv_match_1xn_4_f32_arm_convolve_f32(void) { conv_match_1xn_4_f32_arm_convolve_f32(); }
void test_conv_match_1xn_5_f32_arm_convolve_f32(void) { conv_match_1xn_5_f32_arm_convolve_f32(); }
void test_conv_match_1xn_6_generic_f32_arm_convolve_f32(void) { conv_match_1xn_6_generic_f32_arm_convolve_f32(); }
void test_conv_match_1xn_7_f32_arm_convolve_f32(void) { conv_match_1xn_7_f32_arm_convolve_f32(); }
void test_conv_match_1xn_8_f32_arm_convolve_f32(void) { conv_match_1xn_8_f32_arm_convolve_f32(); }
void test_conv_match_3x2_dilation_f32_arm_convolve_f32(void) { conv_match_3x2_dilation_f32_arm_convolve_f32(); }
void test_conv_match_3x3_dilation_5x5_input_f32_arm_convolve_f32(void)
{
    conv_match_3x3_dilation_5x5_input_f32_arm_convolve_f32();
}
void test_conv_match_2x2_dilation_5x5_input_f32_arm_convolve_f32(void)
{
    conv_match_2x2_dilation_5x5_input_f32_arm_convolve_f32();
}
void test_conv_match_1x1_basic_f32_arm_convolve_f32_packed(void) { conv_match_1x1_basic_f32_arm_convolve_f32_packed(); }
void test_conv_basic_f32_arm_convolve_f32_packed(void) { conv_basic_f32_arm_convolve_f32_packed(); }
void test_conv_k3_opt_f32_arm_convolve_f32_packed(void) { conv_k3_opt_f32_arm_convolve_f32_packed(); }
void test_conv_k5_opt_f32_arm_convolve_f32_packed(void) { conv_k5_opt_f32_arm_convolve_f32_packed(); }

void test_conv_1xn_padding_only_windows_f32(void) { conv_1xn_padding_only_windows_f32(); }
