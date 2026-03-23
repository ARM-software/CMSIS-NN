/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test_arm_convolve_f16.c"
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
void test_conv_basic_f16_arm_convolve_f16(void) { conv_basic_f16_arm_convolve_f16(); }
void test_conv_basic_nhwc_f16_arm_convolve_f16(void) { conv_basic_nhwc_f16_arm_convolve_f16(); }
void test_conv_1x1_stride2_nhwc_f16_arm_convolve_f16(void) { conv_1x1_stride2_nhwc_f16_arm_convolve_f16(); }
void test_conv_k3_opt_f16_arm_convolve_f16(void) { conv_k3_opt_f16_arm_convolve_f16(); }
void test_conv_k5_opt_f16_arm_convolve_f16(void) { conv_k5_opt_f16_arm_convolve_f16(); }
void test_conv_k3_opt_nhwc_tuned_f16_arm_convolve_f16(void) { conv_k3_opt_nhwc_tuned_f16_arm_convolve_f16(); }
void test_conv_k5_opt_nhwc_tuned_f16_arm_convolve_f16(void) { conv_k5_opt_nhwc_tuned_f16_arm_convolve_f16(); }
void test_conv_kernel_2x2_f16_arm_convolve_f16(void) { conv_kernel_2x2_f16_arm_convolve_f16(); }
void test_conv_kernel_3x3_pad1_f16_arm_convolve_f16(void) { conv_kernel_3x3_pad1_f16_arm_convolve_f16(); }
void test_conv_match_basic_f16_arm_convolve_f16(void) { conv_match_basic_f16_arm_convolve_f16(); }
void test_conv_match_stride2pad1_f16_arm_convolve_f16(void) { conv_match_stride2pad1_f16_arm_convolve_f16(); }
void test_conv_match_conv_2_f16_arm_convolve_f16(void) { conv_match_conv_2_f16_arm_convolve_f16(); }
void test_conv_match_conv_3_f16_arm_convolve_f16(void) { conv_match_conv_3_f16_arm_convolve_f16(); }
void test_conv_match_conv_4_f16_arm_convolve_f16(void) { conv_match_conv_4_f16_arm_convolve_f16(); }
void test_conv_match_conv_5_f16_arm_convolve_f16(void) { conv_match_conv_5_f16_arm_convolve_f16(); }
void test_conv_match_out_activation_f16_arm_convolve_f16(void) { conv_match_out_activation_f16_arm_convolve_f16(); }
void test_conv_match_dilation_golden_f16_arm_convolve_f16(void) { conv_match_dilation_golden_f16_arm_convolve_f16(); }
void test_conv_match_2x2_dilation_f16_arm_convolve_f16(void) { conv_match_2x2_dilation_f16_arm_convolve_f16(); }
void test_conv_match_2x3_dilation_f16_arm_convolve_f16(void) { conv_match_2x3_dilation_f16_arm_convolve_f16(); }
void test_conv_match_1x1_basic_f16_arm_convolve_f16(void) { conv_match_1x1_basic_f16_arm_convolve_f16(); }
void test_conv_match_1x1_stride_x_f16_arm_convolve_f16(void) { conv_match_1x1_stride_x_f16_arm_convolve_f16(); }
void test_conv_match_1x1_stride_x_y_f16_arm_convolve_f16(void) { conv_match_1x1_stride_x_y_f16_arm_convolve_f16(); }
void test_conv_match_1x1_stride_x_y_1_f16_arm_convolve_f16(void) { conv_match_1x1_stride_x_y_1_f16_arm_convolve_f16(); }
void test_conv_match_1x1_stride_x_y_2_f16_arm_convolve_f16(void) { conv_match_1x1_stride_x_y_2_f16_arm_convolve_f16(); }
void test_conv_match_1xn_1_f16_arm_convolve_f16(void) { conv_match_1xn_1_f16_arm_convolve_f16(); }
void test_conv_match_1xn_2_f16_arm_convolve_f16(void) { conv_match_1xn_2_f16_arm_convolve_f16(); }
void test_conv_match_1xn_3_f16_arm_convolve_f16(void) { conv_match_1xn_3_f16_arm_convolve_f16(); }
void test_conv_match_1xn_4_f16_arm_convolve_f16(void) { conv_match_1xn_4_f16_arm_convolve_f16(); }
void test_conv_match_1xn_5_f16_arm_convolve_f16(void) { conv_match_1xn_5_f16_arm_convolve_f16(); }
void test_conv_match_1xn_6_generic_f16_arm_convolve_f16(void) { conv_match_1xn_6_generic_f16_arm_convolve_f16(); }
void test_conv_match_1xn_7_f16_arm_convolve_f16(void) { conv_match_1xn_7_f16_arm_convolve_f16(); }
void test_conv_match_1xn_8_f16_arm_convolve_f16(void) { conv_match_1xn_8_f16_arm_convolve_f16(); }
void test_conv_match_3x2_dilation_f16_arm_convolve_f16(void) { conv_match_3x2_dilation_f16_arm_convolve_f16(); }
void test_conv_match_3x3_dilation_5x5_input_f16_arm_convolve_f16(void)
{
    conv_match_3x3_dilation_5x5_input_f16_arm_convolve_f16();
}
void test_conv_match_2x2_dilation_5x5_input_f16_arm_convolve_f16(void)
{
    conv_match_2x2_dilation_5x5_input_f16_arm_convolve_f16();
}
void test_conv_match_1x1_basic_f16_arm_convolve_f16_packed(void) { conv_match_1x1_basic_f16_arm_convolve_f16_packed(); }
void test_conv_basic_f16_arm_convolve_f16_packed(void) { conv_basic_f16_arm_convolve_f16_packed(); }
void test_conv_k3_opt_f16_arm_convolve_f16_packed(void) { conv_k3_opt_f16_arm_convolve_f16_packed(); }
void test_conv_k5_opt_f16_arm_convolve_f16_packed(void) { conv_k5_opt_f16_arm_convolve_f16_packed(); }
