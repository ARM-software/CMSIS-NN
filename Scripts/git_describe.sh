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

if git rev-parse --git-dir 2>&1 >/dev/null; then
  gitversion=$(git describe --tags --long --match "$1*" --abbrev=7 || echo "0.0.0-dirty-0-g$(git describe --tags --match "$1*" --always --abbrev=7 2>/dev/null)")
  patch=$(sed -r -e 's/[0-9]+\.[0-9]+\.([0-9]+).*/\1/' <<< ${gitversion#$1})
  let patch+=1
  version=$(sed -r -e 's/-0-(g[0-9a-f]{7})//' <<< ${gitversion#$1})
  version=$(sed -r -e "s/\.[0-9]+-([0-9]+)-(g[0-9a-f]{7})/.${patch}-dev\1+\2/"  <<< ${version})
  version=$(sed -r -e "s/-([0-9]+)-(g[0-9a-f]{7})/+p\1+\2/"  <<< ${version})
  echo "Git version: '$version'" >&2
  echo $version
else
  echo "No Git repository: '0.0.0-nogit'" >&2
  echo "0.0.0-nogit"
fi

exit 0
