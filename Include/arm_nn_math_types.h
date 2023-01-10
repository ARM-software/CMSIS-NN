/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_math_types.h
 * Description:  Compiler include and basic types
 *
 * $Date:        28 December 2022
 * $Revision:    V.1.3.2
 *
 * Target Processor:  Arm Cortex-M CPUs
 * -------------------------------------------------------------------- */

/**
   Copied from CMSIS/DSP/arm_math_types.h and modified
*/

#ifndef _ARM_NN_MATH_TYPES_H_

#define _ARM_NN_MATH_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Compiler specific diagnostic adjustment */
#if defined(__CC_ARM)

#elif defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)

#elif defined(__GNUC__)

#elif defined(__ICCARM__)

#elif defined(__TI_ARM__)

#elif defined(__CSMC__)

#elif defined(__TASKING__)

#elif defined(_MSC_VER)

#else
#error Unknown compiler
#endif

/* Included for instrinsics definitions */
#if defined(_MSC_VER)
#ifndef __STATIC_FORCEINLINE
#define __STATIC_FORCEINLINE static __forceinline
#endif
#ifndef __STATIC_INLINE
#define __STATIC_INLINE static __inline
#endif
#ifndef __ALIGNED
#define __ALIGNED(x) __declspec(align(x))
#endif

#elif defined(__GNUC_PYTHON__)
#ifndef __ALIGNED
#define __ALIGNED(x) __attribute__((aligned(x)))
#endif
#ifndef __STATIC_FORCEINLINE
#define __STATIC_FORCEINLINE static inline __attribute__((always_inline))
#endif
#ifndef __STATIC_INLINE
#define __STATIC_INLINE static inline
#endif

#else
#include "cmsis_compiler.h"
#include <arm_acle.h>
#endif

/* evaluate ARM DSP feature */
#if (defined(__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1))
#ifndef ARM_MATH_DSP
#define ARM_MATH_DSP 1
#endif
#endif

#if __ARM_FEATURE_MVE
#ifndef ARM_MATH_MVEI
#define ARM_MATH_MVEI
#endif
#endif

/* Compiler specific diagnostic adjustment */
#if defined(__CC_ARM)

#elif defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050)

#define ACLE_SMULBB __smulbb
#define ACLE_SMULTT __smultt

#elif defined(__GNUC__)

#if (__GNUC__ == 12 && (__GNUC_MINOR__ <= 2))
// Workaround for Internal Compiler Error for GCC 12.2.x
// https://gcc.gnu.org/pipermail/gcc-patches/2022-December/607963.html
#define ARM_GCC_12_2_ICE
#endif

#if defined(ARM_MATH_DSP)

// Inline assembly routines for ACLE intrinsics that are not defined by GCC toolchain
__STATIC_FORCEINLINE uint32_t ACLE_SMULBB(uint32_t op1, uint32_t op2)
{
    uint32_t result;

    __ASM volatile("smulbb %0, %1, %2" : "=r"(result) : "r"(op1), "r"(op2));
    return (result);
}

__STATIC_FORCEINLINE uint32_t ACLE_SMULTT(uint32_t op1, uint32_t op2)
{
    uint32_t result;

    __ASM volatile("smultt %0, %1, %2" : "=r"(result) : "r"(op1), "r"(op2));
    return (result);
}

#endif

#elif defined(__ICCARM__)
#define ACLE_SMULBB __smulbb
#define ACLE_SMULTT __smultt
#elif defined(__TI_ARM__)

#elif defined(__CSMC__)

#elif defined(__TASKING__)

#elif defined(_MSC_VER)

#else
#error Unknown compiler
#endif

#ifdef __cplusplus
}
#endif

#if __ARM_FEATURE_MVE
#include <arm_mve.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Add necessary typedefs
 */

#define NN_Q31_MAX ((int32_t)(0x7FFFFFFFL))
#define NN_Q15_MAX ((int16_t)(0x7FFF))
#define NN_Q7_MAX ((int8_t)(0x7F))
#define NN_Q31_MIN ((int32_t)(0x80000000L))
#define NN_Q15_MIN ((int16_t)(0x8000))
#define NN_Q7_MIN ((int8_t)(0x80))

/**
 * @brief Error status returned by some functions in the library.
 */

typedef enum
{
    ARM_CMSIS_NN_SUCCESS = 0,        /**< No error */
    ARM_CMSIS_NN_ARG_ERROR = -1,     /**< One or more arguments are incorrect */
    ARM_CMSIS_NN_NO_IMPL_ERROR = -2, /**<  No implementation available */
} arm_cmsis_nn_status;

#if defined(ARM_MATH_DSP)
#define ACLE_SMLABB __smlabb
#define ACLE_SMLATT __smlatt
#endif

#ifdef __cplusplus
}
#endif

#endif /*ifndef _ARM_NN_MATH_TYPES_H_ */
