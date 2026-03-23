/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
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

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_conv1x1_opt_common.h
 * Description:  Shared float conv1x1 specialization helpers
 *
 * $Date:        27 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_CONV1X1_OPT_COMMON_H
#define ARM_CONV1X1_OPT_COMMON_H

#define ARM_CONV1X1_SPEC_ENTRY(MATCH_FN, CALL_FN)                                                                      \
    {                                                                                                                  \
        (MATCH_FN), (CALL_FN)                                                                                          \
    }

#define ARM_CONV1X1_ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

#define ARM_CONV1X1_DISPATCH(TABLE, COUNT, ...)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        for (size_t _i = 0; _i < (COUNT); ++_i)                                                                        \
        {                                                                                                              \
            if ((TABLE)[_i].match(__VA_ARGS__))                                                                        \
            {                                                                                                          \
                return (TABLE)[_i].call(__VA_ARGS__);                                                                  \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

#endif /* ARM_CONV1X1_OPT_COMMON_H */
