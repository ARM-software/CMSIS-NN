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
#include "../test_arm_transpose_f16.c"

void test_transpose_matrix_f16_arm_transpose_f16(void) { transpose_matrix_f16_arm_transpose_f16(); }
void test_transpose_matrix_small_f16_arm_transpose_f16(void) { transpose_matrix_small_f16_arm_transpose_f16(); }
void test_transpose_3dim_f16_arm_transpose_f16(void) { transpose_3dim_f16_arm_transpose_f16(); }
void test_transpose_3dim2_f16_arm_transpose_f16(void) { transpose_3dim2_f16_arm_transpose_f16(); }
void test_transpose_chwn_f16_arm_transpose_f16(void) { transpose_chwn_f16_arm_transpose_f16(); }
void test_transpose_default_f16_arm_transpose_f16(void) { transpose_default_f16_arm_transpose_f16(); }
void test_transpose_nhcw_f16_arm_transpose_f16(void) { transpose_nhcw_f16_arm_transpose_f16(); }
void test_transpose_nchw_f16_arm_transpose_f16(void) { transpose_nchw_f16_arm_transpose_f16(); }
void test_transpose_nwhc_f16_arm_transpose_f16(void) { transpose_nwhc_f16_arm_transpose_f16(); }
void test_transpose_ncwh_f16_arm_transpose_f16(void) { transpose_ncwh_f16_arm_transpose_f16(); }
void test_transpose_swap_last2_4d_f16_arm_transpose_f16(void) { transpose_swap_last2_4d_f16_arm_transpose_f16(); }
void test_transpose_wchn_f16_arm_transpose_f16(void) { transpose_wchn_f16_arm_transpose_f16(); }
