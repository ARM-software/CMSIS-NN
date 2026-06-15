/*
 * Copyright (c) 2026 Arm Limited. All rights reserved.
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

#ifndef TEST_UNITTEST_CORSTONE300_REGION_DEFS_H
#define TEST_UNITTEST_CORSTONE300_REGION_DEFS_H

#include "region_limits.h"

/*
 * This header is parsed both by the compiler and by the linker script
 * preprocessor. Keep expressions simple and avoid C-only syntax.
 */

#define S_CODE_START (S_ROM_ALIAS)
#define S_CODE_SIZE (TOTAL_S_ROM_SIZE)
#define S_CODE_LIMIT (S_CODE_START + S_CODE_SIZE)

#define S_DATA_START (S_RAM_ALIAS)
#define S_DATA_SIZE (TOTAL_S_RAM_SIZE)
#define S_DATA_LIMIT (S_DATA_START + S_DATA_SIZE)

#define S_DDR4_START (S_DDR4_ALIAS)
#define S_DDR4_SIZE (TOTAL_S_DDR4_SIZE)
#define S_DDR4_LIMIT (S_DDR4_START + S_DDR4_SIZE)

#define S_ISRAM_START (S_ISRAM_ALIAS)
#define S_ISRAM_SIZE (TOTAL_S_ISRAM_SIZE)
#define S_ISRAM_LIMIT (S_ISRAM_START + S_ISRAM_SIZE)

#define __ROM0_BASE (S_CODE_START)
#define __ROM0_SIZE (S_CODE_SIZE)
#define __RAM0_BASE (S_DATA_START)
#define __RAM0_SIZE (S_DATA_SIZE)

#ifndef HEAP_SIZE
    #define HEAP_SIZE (0x00010000)
#endif

#ifndef STACK_SIZE
    #define STACK_SIZE (0x00008000)
#endif

#define __HEAP_SIZE (HEAP_SIZE)
#define __STACK_SIZE (STACK_SIZE)

#endif
