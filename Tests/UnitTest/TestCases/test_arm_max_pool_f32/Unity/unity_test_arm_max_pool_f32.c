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

#include "../test_arm_max_pool_f32.c"
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
void test_maxpooling_f32_arm_max_pool_f32(void) { maxpooling_f32_arm_max_pool_f32(); }
void test_maxpooling_f32_1_arm_max_pool_f32(void) { maxpooling_f32_1_arm_max_pool_f32(); }
void test_maxpooling_f32_2_arm_max_pool_f32(void) { maxpooling_f32_2_arm_max_pool_f32(); }
void test_maxpooling_f32_3_arm_max_pool_f32(void) { maxpooling_f32_3_arm_max_pool_f32(); }
void test_maxpooling_match_2_f32_arm_max_pool_f32(void) { maxpooling_match_2_f32_arm_max_pool_f32(); }
void test_maxpooling_match_3_f32_arm_max_pool_f32(void) { maxpooling_match_3_f32_arm_max_pool_f32(); }
void test_maxpooling_match_4_f32_arm_max_pool_f32(void) { maxpooling_match_4_f32_arm_max_pool_f32(); }
void test_maxpooling_match_5_f32_arm_max_pool_f32(void) { maxpooling_match_5_f32_arm_max_pool_f32(); }
void test_maxpooling_1d_k2s2_noclip_f32_arm_max_pool_f32(void) { maxpooling_1d_k2s2_noclip_f32_arm_max_pool_f32(); }
void test_maxpooling_1d_k2s2_clip_f32_arm_max_pool_f32(void) { maxpooling_1d_k2s2_clip_f32_arm_max_pool_f32(); }
void test_maxpooling_1d_k3s3_noclip_f32_arm_max_pool_f32(void) { maxpooling_1d_k3s3_noclip_f32_arm_max_pool_f32(); }
