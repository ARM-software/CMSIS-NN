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

#include "../test_arm_batch_matmul_f16.c"
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
void test_batch_matmul_1_f16_arm_batch_matmul_f16(void) { batch_matmul_1_f16_arm_batch_matmul_f16(); }
void test_batch_matmul_2_f16_arm_batch_matmul_f16(void) { batch_matmul_2_f16_arm_batch_matmul_f16(); }
void test_batch_matmul_3_f16_arm_batch_matmul_f16(void) { batch_matmul_3_f16_arm_batch_matmul_f16(); }
void test_batch_matmul_4_f16_arm_batch_matmul_f16(void) { batch_matmul_4_f16_arm_batch_matmul_f16(); }
void test_batch_matmul_5_f16_arm_batch_matmul_f16(void) { batch_matmul_5_f16_arm_batch_matmul_f16(); }
void test_batch_matmul_6_f16_arm_batch_matmul_f16(void) { batch_matmul_6_f16_arm_batch_matmul_f16(); }
void test_batch_matmul_et_small_f16_arm_batch_matmul_f16(void) { batch_matmul_et_small_f16_arm_batch_matmul_f16(); }
void test_batch_matmul_et_small_f16_arm_batch_matmul_f16_packed(void)
{
    batch_matmul_et_small_f16_arm_batch_matmul_f16_packed();
}
void test_batch_matmul_2_f16_arm_batch_matmul_f16_packed_invalid(void)
{
    batch_matmul_2_f16_arm_batch_matmul_f16_packed_invalid();
}
