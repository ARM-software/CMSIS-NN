#!/bin/bash
# SPDX-FileCopyrightText: Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

# Version: 2.3
# Date: 2022-09-28
# This bash script generates a CMSIS Software Pack:
#

set -o pipefail

# Set version of gen pack library
REQUIRED_GEN_PACK_LIB="0.5.1"

# Set default command line arguments
DEFAULT_ARGS=(-c "v")

# Pack warehouse directory - destination
PACK_OUTPUT=./output

# Temporary pack build directory
PACK_BUILD=./build

# Specify directory names to be added to pack base directory
PACK_DIRS="
  Documentation
  Include
  Source
"

# Specify file names to be added to pack base directory
PACK_BASE_FILES="
  LICENSE.txt
  README.md
"

# Specify file names to be deleted from pack build directory
PACK_DELETE_FILES=""

# Specify patches to be applied
PACK_PATCH_FILES=""

# Specify addition argument to packchk
PACKCHK_ARGS=()

# Specify additional dependencies for packchk
PACKCHK_DEPS="
  ARM.CMSIS.pdsc
"

# custom pre-processing steps
function preprocess() {
  ./DoxyGen/gen_doc.sh
}

# custom post-processing steps
function postprocess() {
  # Set component version to match pack version
  VERSION=$(git_describe "${CHANGELOG}" | sed -e "s/+g.*$//")
  sed -i -e "s/Cgroup=\"NN Lib\" Cversion=\"[^\"]*\"/Cgroup=\"NN Lib\" Cversion=\"${VERSION}\"/" ${PACK_BUILD}/ARM.CMSIS-NN.pdsc
}

############ DO NOT EDIT BELOW ###########

function install_lib() {
  local URL="https://github.com/Open-CMSIS-Pack/gen-pack/archive/refs/tags/v$1.tar.gz"
  echo "Downloading gen-pack lib to '$2'"
  mkdir -p "$2"
  curl -L "${URL}" -s | tar -xzf - --strip-components 1 -C "$2" || exit 1
}

function load_lib() {
  if [[ -d ${GEN_PACK_LIB} ]]; then
    . "${GEN_PACK_LIB}/gen-pack"
    return 0
  fi
  local GLOBAL_LIB="/usr/local/share/gen-pack/${REQUIRED_GEN_PACK_LIB}"
  local USER_LIB="${HOME}/.local/share/gen-pack/${REQUIRED_GEN_PACK_LIB}"
  if [[ ! -d "${GLOBAL_LIB}" && ! -d "${USER_LIB}" ]]; then
    echo "Required gen_pack lib not found!" >&2
    install_lib "${REQUIRED_GEN_PACK_LIB}" "${USER_LIB}"
  fi

  if [[ -d "${GLOBAL_LIB}" ]]; then
    . "${GLOBAL_LIB}/gen-pack"
  elif [[ -d "${USER_LIB}" ]]; then
    . "${USER_LIB}/gen-pack"
  else
    echo "Required gen-pack lib is not installed!" >&2
    exit 1
  fi
}

load_lib
gen_pack "${DEFAULT_ARGS[@]}" "$@"

exit 0
