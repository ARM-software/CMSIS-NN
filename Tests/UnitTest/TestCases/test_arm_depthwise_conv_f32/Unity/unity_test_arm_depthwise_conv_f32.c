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

#include "../test_arm_depthwise_conv_f32.c"
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
void test_depthwise_basic_f32_arm_depthwise_conv_f32(void) { depthwise_basic_f32_arm_depthwise_conv_f32(); }
void test_depthwise_basic_smallc_nhwc_f32_arm_depthwise_conv_f32(void)
{
    depthwise_basic_smallc_nhwc_f32_arm_depthwise_conv_f32();
}
void test_depthwise_ic1_to_conv_nhwc_f32_arm_depthwise_conv_f32(void)
{
    depthwise_ic1_to_conv_nhwc_f32_arm_depthwise_conv_f32();
}
void test_depthwise_k3_1d_opt_batch2_f32_arm_depthwise_conv_f32(void)
{
    depthwise_k3_1d_opt_batch2_f32_arm_depthwise_conv_f32();
}
void test_depthwise_k3_1d_opt_nhwc_f32_arm_depthwise_conv_f32(void)
{
    depthwise_k3_1d_opt_nhwc_f32_arm_depthwise_conv_f32();
}
void test_depthwise_2x5_opt_batch2_f32_arm_depthwise_conv_f32(void)
{
    depthwise_2x5_opt_batch2_f32_arm_depthwise_conv_f32();
}
void test_depthwise_2x5_opt_nhwc_chmult16_f32_arm_depthwise_conv_f32(void)
{
    depthwise_2x5_opt_nhwc_chmult16_f32_arm_depthwise_conv_f32();
}
void test_depthwise_kernel_2x2_f32_arm_depthwise_conv_f32(void) { depthwise_kernel_2x2_f32_arm_depthwise_conv_f32(); }
void test_depthwise_kernel_3x3_f32_arm_depthwise_conv_f32(void) { depthwise_kernel_3x3_f32_arm_depthwise_conv_f32(); }
void test_depthwise_kernel_3x3_null_bias_f32_arm_depthwise_conv_f32(void)
{
    depthwise_kernel_3x3_null_bias_f32_arm_depthwise_conv_f32();
}
void test_depthwise_match_basic_f32_arm_depthwise_conv_f32(void) { depthwise_match_basic_f32_arm_depthwise_conv_f32(); }
void test_depthwise_match_sub_block_f32_arm_depthwise_conv_f32(void)
{
    depthwise_match_sub_block_f32_arm_depthwise_conv_f32();
}
void test_depthwise_match_dilation_f32_arm_depthwise_conv_f32(void)
{
    depthwise_match_dilation_f32_arm_depthwise_conv_f32();
}
void test_depthwise_match_out_activation_f32_arm_depthwise_conv_f32(void)
{
    depthwise_match_out_activation_f32_arm_depthwise_conv_f32();
}
void test_depthwise_match_stride2pad1_f32_arm_depthwise_conv_f32(void)
{
    depthwise_match_stride2pad1_f32_arm_depthwise_conv_f32();
}
void test_depthwise_ic1_to_conv_nhwc_f32_arm_depthwise_conv_f32_no_ctx(void)
{
    depthwise_ic1_to_conv_nhwc_f32_arm_depthwise_conv_f32_no_ctx();
}
