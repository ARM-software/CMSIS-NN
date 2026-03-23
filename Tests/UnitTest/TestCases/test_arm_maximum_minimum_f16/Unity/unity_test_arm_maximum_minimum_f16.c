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
#include "../test_arm_maximum_minimum_f16.c"

void test_maximum_scalar_1_f16_arm_maximum_f16(void) { maximum_scalar_1_f16_arm_maximum_f16(); }
void test_maximum_scalar_2_f16_arm_maximum_f16(void) { maximum_scalar_2_f16_arm_maximum_f16(); }
void test_maximum_no_broadcast_f16_arm_maximum_f16(void) { maximum_no_broadcast_f16_arm_maximum_f16(); }
void test_maximum_broadcast_batch_f16_arm_maximum_f16(void) { maximum_broadcast_batch_f16_arm_maximum_f16(); }
void test_maximum_broadcast_height_f16_arm_maximum_f16(void) { maximum_broadcast_height_f16_arm_maximum_f16(); }
void test_maximum_broadcast_width_f16_arm_maximum_f16(void) { maximum_broadcast_width_f16_arm_maximum_f16(); }
void test_maximum_broadcast_ch_f16_arm_maximum_f16(void) { maximum_broadcast_ch_f16_arm_maximum_f16(); }
void test_minimum_scalar_1_f16_arm_minimum_f16(void) { minimum_scalar_1_f16_arm_minimum_f16(); }
void test_minimum_scalar_2_f16_arm_minimum_f16(void) { minimum_scalar_2_f16_arm_minimum_f16(); }
void test_minimum_no_broadcast_f16_arm_minimum_f16(void) { minimum_no_broadcast_f16_arm_minimum_f16(); }
void test_minimum_broadcast_batch_f16_arm_minimum_f16(void) { minimum_broadcast_batch_f16_arm_minimum_f16(); }
void test_minimum_broadcast_height_f16_arm_minimum_f16(void) { minimum_broadcast_height_f16_arm_minimum_f16(); }
void test_minimum_broadcast_width_f16_arm_minimum_f16(void) { minimum_broadcast_width_f16_arm_minimum_f16(); }
void test_minimum_broadcast_ch_f16_arm_minimum_f16(void) { minimum_broadcast_ch_f16_arm_minimum_f16(); }
