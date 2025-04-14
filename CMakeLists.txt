#
# SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.15.6)

if (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMSIS_OPTIMIZATION_LEVEL "-Ofast" CACHE STRING "Compiler optimization level.")
endif ()

project(CMSISNN)

add_library(cmsis-nn STATIC)

target_compile_options(cmsis-nn PRIVATE ${CMSIS_OPTIMIZATION_LEVEL})

target_include_directories(cmsis-nn PUBLIC "Include")

add_subdirectory(Source)

if (CMAKE_BUILD_TYPE MATCHES Debug AND CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
    include(CodeCoverage)
    set(CODE_COVERAGE_VERBOSE ON)
    append_coverage_compiler_flags()
    set(COVERAGE_EXCLUDES "*build*" "Tests")
    setup_target_for_coverage_gcovr_xml(NAME coverage-xml EXECUTABLE ctest)
    setup_target_for_coverage_gcovr_html(NAME coverage-html EXECUTABLE ctest)

    enable_testing()
    set(CMSIS_PATH ${CMAKE_BINARY_DIR} CACHE PATH "Path to CMSIS.")
    set(BUILD_CMSIS_NN_UNIT ON)
    add_subdirectory(Tests/UnitTest)
endif ()
