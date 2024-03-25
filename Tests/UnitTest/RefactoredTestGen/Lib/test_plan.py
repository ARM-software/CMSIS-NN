# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
import json
import Lib.test_suite


def generate(args):
    """Generate a number of test suites defined by a json-file test plan"""

    print(f"\nGenerating tests from {args.test_plan}")
    test_plan = args.test_plan.read_text()
    test_suite_params_list = json.loads(test_plan)

    test_suites = []
    for test_suite_params in test_suite_params_list:
        if (test_suite_params["suite_name"] in args.test_suites) or (args.test_suites == []):
            print(f"{test_suite_params['suite_name']}")
            test_suite = Lib.test_suite.generate(test_suite_params, args)
            test_suites.append(test_suite)
