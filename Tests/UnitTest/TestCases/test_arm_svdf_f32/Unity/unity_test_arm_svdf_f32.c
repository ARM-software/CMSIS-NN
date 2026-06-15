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
#include "../test_arm_svdf_f32.c"

void test_svdf_small_f32_arm_svdf_f32(void) { svdf_small_f32_arm_svdf_f32(); }
void test_svdf_batch2_f32_arm_svdf_f32(void) { svdf_batch2_f32_arm_svdf_f32(); }
void test_svdf_match_1_f32_arm_svdf_f32(void) { svdf_match_1_f32_arm_svdf_f32(); }
void test_svdf_match_2_f32_arm_svdf_f32(void) { svdf_match_2_f32_arm_svdf_f32(); }
