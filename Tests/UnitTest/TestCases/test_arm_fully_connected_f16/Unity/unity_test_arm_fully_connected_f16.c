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

#include "../test_arm_fully_connected_f16.c"
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
void test_fully_connected_small_f16_arm_fully_connected_f16(void)
{
    fully_connected_small_f16_arm_fully_connected_f16();
}
void test_fully_connected_medium_f16_arm_fully_connected_f16(void)
{
    fully_connected_medium_f16_arm_fully_connected_f16();
}
void test_fully_connected_large_f16_arm_fully_connected_f16(void)
{
    fully_connected_large_f16_arm_fully_connected_f16();
}
void test_fully_connected_2out_batch2_f16_arm_fully_connected_f16(void)
{
    fully_connected_2out_batch2_f16_arm_fully_connected_f16();
}
void test_fully_connected_2out_tail17_f16_arm_fully_connected_f16(void)
{
    fully_connected_2out_tail17_f16_arm_fully_connected_f16();
}
void test_fully_connected_2out_tail21_f16_arm_fully_connected_f16(void)
{
    fully_connected_2out_tail21_f16_arm_fully_connected_f16();
}
void test_fully_connected_null_bias_f16_arm_fully_connected_f16(void)
{
    fully_connected_null_bias_f16_arm_fully_connected_f16();
}
void test_fully_connected_out_activation_f16_arm_fully_connected_f16(void)
{
    fully_connected_out_activation_f16_arm_fully_connected_f16();
}
void test_fully_connected_match_basic_f16_arm_fully_connected_f16(void)
{
    fully_connected_match_basic_f16_arm_fully_connected_f16();
}
void test_fully_connected_match_mve_0_f16_arm_fully_connected_f16(void)
{
    fully_connected_match_mve_0_f16_arm_fully_connected_f16();
}
void test_fully_connected_match_mve_1_f16_arm_fully_connected_f16(void)
{
    fully_connected_match_mve_1_f16_arm_fully_connected_f16();
}
void test_fully_connected_match_fc_per_ch_f16_arm_fully_connected_f16(void)
{
    fully_connected_match_fc_per_ch_f16_arm_fully_connected_f16();
}
void test_fully_connected_match_mve_1_f16_arm_fully_connected_f16_packed(void)
{
    fully_connected_match_mve_1_f16_arm_fully_connected_f16_packed();
}
