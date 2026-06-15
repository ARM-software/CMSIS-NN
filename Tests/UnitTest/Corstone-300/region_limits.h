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

#ifndef TEST_UNITTEST_CORSTONE300_REGION_LIMITS_H
#define TEST_UNITTEST_CORSTONE300_REGION_LIMITS_H

/* Secure Code */
#define S_ROM_ALIAS (0x10000000)
#define TOTAL_S_ROM_SIZE (0x00080000)

/* Secure Data */
#define S_RAM_ALIAS (0x30000000)
#define TOTAL_S_RAM_SIZE (0x00080000)

/* Secure DDR */
#define S_DDR4_ALIAS (0x70000000)
#define TOTAL_S_DDR4_SIZE (0x02000000)

/* Secure SRAM */
#define S_ISRAM_ALIAS (0x31000000)
#define TOTAL_S_ISRAM_SIZE (0x00200000)

#endif
