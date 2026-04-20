#
# SPDX-FileCopyrightText: Copyright 2010-2023, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
set(CMSIS_OPTIMIZATION_LEVEL "-Ofast" CACHE STRING "Compiler optimization level.")

project(CMSISNN C CXX)

add_library(cmsis-nn STATIC)

target_compile_options(cmsis-nn PRIVATE ${CMSIS_OPTIMIZATION_LEVEL})

target_include_directories(cmsis-nn PUBLIC "Include")

add_subdirectory(Source)

option(CMSISNN_BUILD_PYBIND "Build pybind11 Python module for CMSIS-NN helpers" OFF)

if(CMSISNN_BUILD_PYBIND)

  # Add shared library for testing bindings with ctest
  add_library(cmsis-nn-shared SHARED
    $<TARGET_PROPERTY:cmsis-nn,SOURCES>
  )
  set_target_properties(cmsis-nn-shared PROPERTIES OUTPUT_NAME cmsis-nn)
  target_include_directories(cmsis-nn-shared PRIVATE
    $<TARGET_PROPERTY:cmsis-nn,INCLUDE_DIRECTORIES>
  )
  target_compile_options(cmsis-nn-shared PRIVATE ${CMSIS_OPTIMIZATION_LEVEL})

  set(Python3_FIND_VIRTUALENV FIRST)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_PYTHON_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  find_package(pybind11 CONFIG
              HINTS "${pybind11_PYTHON_CMAKE_DIR}"
  )
  if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found. Install pybind11, set pybind11_DIR, or set CMSISNN_BUILD_PYBIND=OFF.")
  endif()

  pybind11_add_module(cmsis_nn MODULE
    Source/Bindings/arm_py_module.cpp
    Source/Bindings/arm_py_backend.cpp
    Source/Bindings/arm_py_avgpool_buffer_size.cpp
    Source/Bindings/arm_py_conv_buffer_size.cpp
    Source/Bindings/arm_py_depthwise_conv_buffer_size.cpp
    Source/Bindings/arm_py_fully_connected_buffer_size.cpp
    Source/Bindings/arm_py_svdf_buffer_size.cpp
    Source/Bindings/arm_py_transpose_conv_buffer_size.cpp
  )
  set_target_properties(cmsis_nn PROPERTIES PREFIX "")
  target_include_directories(cmsis_nn PRIVATE
      ${Python3_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/Include
  )

  target_link_libraries(cmsis_nn PRIVATE
      cmsis-nn
      pybind11::headers
  )

  target_compile_options(cmsis_nn PRIVATE ${CMSIS_OPTIMIZATION_LEVEL})

  # Put the built .so into the python package directory inside the wheel
  install(TARGETS cmsis_nn
                  LIBRARY DESTINATION cmsis_nn
                  RUNTIME DESTINATION cmsis_nn
  )

endif()
