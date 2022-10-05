#
# SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

project(CMSISNN)

set(CMSIS_PATH "</path/to/CMSIS>" CACHE PATH "Path to CMSIS.")

if (${CMSIS_PATH} STREQUAL "</path/to/CMSIS>")
  message(FATAL_ERROR "CMSIS_PATH not set. Did you provide -DCMSIS_PATH=<path/to/CMSIS>?")
endif()

option(BUILD_CMSIS_NN_FUNCTIONS "Build CMSIS-NN Source." ON)

if(BUILD_CMSIS_NN_FUNCTIONS)
    add_subdirectory(Source)
endif()
