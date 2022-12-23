/*
# SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#pragma once

// Lookup table for exp(x), where x uniform distributed between [-10.0 , 0.0].
const int16_t softmax_s16_exp_lut[513] = {
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     3,     3,     3,     3,     3,
    3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     3,     4,     4,     4,     4,
    4,     4,     4,     4,     4,     4,     4,     4,     4,     5,     5,     5,     5,     5,     5,     5,
    5,     5,     5,     6,     6,     6,     6,     6,     6,     6,     6,     7,     7,     7,     7,     7,
    7,     7,     7,     8,     8,     8,     8,     8,     8,     9,     9,     9,     9,     9,     9,     10,
    10,    10,    10,    10,    11,    11,    11,    11,    11,    12,    12,    12,    12,    13,    13,    13,
    13,    14,    14,    14,    14,    15,    15,    15,    16,    16,    16,    17,    17,    17,    18,    18,
    18,    19,    19,    19,    20,    20,    21,    21,    21,    22,    22,    23,    23,    24,    24,    25,
    25,    26,    26,    27,    27,    28,    28,    29,    29,    30,    30,    31,    32,    32,    33,    34,
    34,    35,    36,    36,    37,    37,    38,    39,    40,    40,    42,    42,    43,    44,    45,    45,
    46,    47,    48,    49,    50,    51,    52,    53,    54,    55,    56,    57,    59,    60,    60,    62,
    63,    65,    65,    67,    68,    69,    71,    73,    74,    75,    77,    78,    80,    81,    83,    85,
    86,    88,    90,    92,    93,    95,    97,    99,    101,   103,   105,   107,   109,   112,   114,   116,
    118,   121,   123,   126,   128,   131,   133,   135,   139,   141,   144,   147,   149,   152,   155,   158,
    162,   165,   168,   171,   174,   178,   181,   185,   189,   192,   196,   200,   204,   208,   212,   217,
    221,   225,   230,   234,   239,   243,   248,   253,   258,   263,   268,   273,   279,   284,   290,   296,
    302,   308,   314,   320,   327,   333,   340,   346,   353,   360,   366,   374,   381,   389,   397,   404,
    413,   421,   429,   437,   446,   455,   464,   473,   482,   492,   501,   511,   522,   532,   543,   553,
    564,   575,   586,   598,   610,   622,   634,   646,   659,   672,   685,   699,   713,   727,   741,   756,
    771,   786,   801,   817,   833,   850,   866,   884,   901,   919,   937,   955,   974,   993,   1013,  1033,
    1053,  1074,  1095,  1117,  1139,  1161,  1184,  1207,  1232,  1256,  1281,  1306,  1332,  1358,  1385,  1412,
    1440,  1468,  1497,  1527,  1557,  1587,  1619,  1651,  1683,  1716,  1750,  1785,  1820,  1856,  1892,  1930,
    1968,  2006,  2046,  2087,  2128,  2170,  2212,  2256,  2300,  2346,  2392,  2439,  2488,  2537,  2587,  2638,
    2690,  2743,  2796,  2852,  2908,  2966,  3024,  3084,  3145,  3207,  3270,  3334,  3400,  3467,  3535,  3605,
    3677,  3749,  3822,  3898,  3975,  4053,  4133,  4214,  4297,  4383,  4469,  4557,  4647,  4739,  4833,  4927,
    5024,  5124,  5225,  5328,  5433,  5541,  5649,  5761,  5875,  5991,  6109,  6230,  6352,  6477,  6605,  6736,
    6868,  7004,  7141,  7282,  7427,  7572,  7722,  7874,  8030,  8188,  8350,  8514,  8683,  8854,  9028,  9206,
    9387,  9572,  9762,  9954,  10151, 10351, 10555, 10763, 10976, 11191, 11412, 11637, 11867, 12102, 12341, 12583,
    12831, 13085, 13342, 13606, 13874, 14148, 14427, 14711, 15002, 15297, 15599, 15907, 16221, 16541, 16867, 17199,
    17539, 17884, 18237, 18597, 18964, 19338, 19719, 20108, 20505, 20909, 21322, 21742, 22171, 22608, 23054, 23509,
    23973, 24445, 24928, 25419, 25921, 26432, 26953, 27485, 28027, 28580, 29143, 29718, 30304, 30902, 31512, 32133,
    32767};