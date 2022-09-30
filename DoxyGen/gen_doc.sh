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

# Version: 1.0
# Date: 2022-10-30
# This bash script generates CMSIS-NN Documentation:
#
# Pre-requisites:
# - bash shell (for Windows: install git for Windows)
# - doxygen 1.9.2

set -o pipefail

DIRNAME=$(dirname $(readlink -f $0))
DOXYGEN=$(which doxygen)
DESCRIBE=$(readlink -f ${DIRNAME}/../Scripts/git_describe.sh)
CHANGELOG=$(readlink -f ${DIRNAME}/../Scripts/gen_changelog.sh)

if [[ ! -f "${DOXYGEN}" ]]; then
  echo "Doxygen not found!" >&2
  echo "Did you miss to add it to PATH?"
  exit 1
else
  version=$("${DOXYGEN}" --version)
  echo "DOXYGEN is ${DOXYGEN} at version ${version}"
  if [[ "${version}" != "1.8.6" ]]; then
    echo " >> Version is different from 1.8.6 !" >&2
  fi
fi

if [ -z $VERSION ]; then
  VERSION_FULL=$(${DESCRIBE} "v")
  VERSION=${VERSION_FULL%+*}
fi

echo "Generating documentation ..."

pushd $DIRNAME > /dev/null

rm -rf ${DIRNAME}/../Documentation/html
sed -e "s/{projectNumber}/${VERSION}/" "${DIRNAME}/nn.dxy.in" \
  > "${DIRNAME}/nn.dxy"

echo "${CHANGELOG} -f html > history.txt"
"${CHANGELOG}" -f html 1> history.txt 2>/dev/null

echo "${DOXYGEN} nn.dxy"
"${DOXYGEN}" nn.dxy
popd > /dev/null

if [[ $2 != 0 ]]; then
  cp -f "${DIRNAME}/templates/search.css" "${DIRNAME}/../Documentation/html/search/"
fi

projectName=$(grep -E "PROJECT_NAME\s+=" "${DIRNAME}/nn.dxy" | sed -r -e 's/[^"]*"([^"]+)"/\1/')
datetime=$(date -u +'%a %b %e %Y %H:%M:%S')
year=$(date -u +'%Y')
if [[ "${year}" != "2022" ]]; then 
  year="2022-${year}"
fi
sed -e "s/{datetime}/${datetime}/" "${DIRNAME}/templates/footer.js.in" \
  | sed -e "s/{year}/${year}/" \
  | sed -e "s/{projectName}/${projectName}/" \
  | sed -e "s/{projectNumber}/${VERSION_FULL}/" \
  > "${DIRNAME}/../Documentation/html/footer.js"

exit 0
