from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
import os

nan = np.infty
CLASSES = ['car', 'bus', 'truck', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
CLASSES_severity = ['AP', 'trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']
severityS = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']
PARAMETERS = dict(
    BEVFormer_Tiny = 33.6,
    BEVFormer_Tiny_Temp = 33.6,
    BEVFormer_Small = 59.6,
    BEVFormer_Small_Temp = 59.6,
    BEVFormer_Base = 69.1,
    BEVFormer_Base_Temp = 69.1,
    DETR3D_CBGS = 53.8,
    DETR3D = 53.8,
    PETR_R50 = 38.1,
    PETR_Vov = 83.1,
    BEVDepth_R50 = 53.1,
    BEVDepth4D_R50 = 53.4,
    BEVDepth_R101 = 72.1,
    BEVDet_R50 = 48.2,
    BEVDet4D_R50 = 48.2,
    BEVDet_R101 = 67.2,
    BEVDet_Swin_Tiny = 55.9,
    FCOS3D = 55.1,
    PGD_Det = 56.2,
    )

MARKER = dict(
    BEVFormer_Tiny = 'o', 
    BEVFormer_Tiny_Temp = 'o', 
    BEVFormer_Small = 'o', 
    BEVFormer_Small_Temp = 'o', 
    BEVFormer_Base = 'o', 
    BEVFormer_Base_Temp = 'o', 
    DETR3D_CBGS = 'v', 
    DETR3D = 'v', 
    PETR_R50 = 'X',
    PETR_Vov = 'X',
    FCOS3D = 's', 
    PGD_Det = 'p', 
    BEVDepth_R50 = 'D', 
    BEVDepth4D_R50 = 'D', 
    BEVDepth_R101 = 'D',
    BEVDet_R50 = 'd',
    BEVDet4D_R50 = 'd',
    BEVDet_R101 = 'd',
    BEVDet_Swin_Tiny = 'd',
)

COLORS = dict(
    BEVFormer_Tiny = '#4169E1', 
    BEVFormer_Tiny_Temp = '#0000FF', 
    BEVFormer_Small = '#9932CC', 
    BEVFormer_Small_Temp = '#800080', 
    BEVFormer_Base = '#F08080', 
    BEVFormer_Base_Temp = '#B22222', 
    DETR3D_CBGS = '#A9A9A9', 
    DETR3D = '#696969', 
    PETR_R50 = '#D2691E',
    PETR_Vov = '#8B4513',
    FCOS3D = '#32CD32', 
    PGD_Det = '#006400', 
    BEVDepth_R50 = '#C0C000', 
    BEVDepth4D_R50 = '#808000', 
    BEVDepth_R101 = '#9ACD32',
    BEVDet_R50 = '#DAA520',
    BEVDet4D_R50 = '#FFE4B5',
    BEVDet_R101 = '#FF8C00',
    BEVDet_Swin_Tiny = '#FFA500',
    )

VAL_MAP = dict(
    BEVFormer_Tiny = 0.1842,
    BEVFormer_Tiny_Temp = 0.2524,
    BEVFormer_Small = 0.1324,
    BEVFormer_Small_Temp = 0.3699,
    BEVFormer_Base = 0.3461,
    BEVFormer_Base_Temp = 0.4167,
    DETR3D_CBGS = 0.3494,
    DETR3D = 0.3469,
    PETR_R50 = 0.3174,
    PETR_Vov = 0.4035,
    FCOS3D = 0.3214,
    PGD_Det = 0.3360,
    BEVDepth_R50 = 0.3327,
    BEVDepth4D_R50 = 0.3609,
    BEVDepth_R101 = 0.3376,
    BEVDet_R50 = 0.2987,
    BEVDet4D_R50 = 0.3215,
    BEVDet_R101 = 0.3021,
    BEVDet_Swin_Tiny = 0.3080,
)
VAL_NDS = dict(
    BEVFormer_Tiny = 0.2548,
    BEVFormer_Tiny_Temp = 0.3542,
    BEVFormer_Small = 0.2623,
    BEVFormer_Small_Temp = 0.4786,
    BEVFormer_Base = 0.4128,
    BEVFormer_Base_Temp = 0.5176,
    DETR3D_CBGS = 0.4342,
    DETR3D = 0.4223,
    PETR_R50 = 0.3667,
    PETR_Vov = 0.4550,
    FCOS3D = 0.3949,
    PGD_Det = 0.4089,
    BEVDepth_R50 = 0.4057,
    BEVDepth4D_R50 = 0.4844,
    BEVDepth_R101 = 0.4167,
    BEVDet_R50 = 0.3770,
    BEVDet4D_R50 = 0.4570,
    BEVDet_R101 = 0.3864,
    BEVDet_Swin_Tiny = 0.4037,
)


pgd_attack_untarget = dict(
    severity = 'max_steps',
    max_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50],
    BEVFormer_Tiny = dict(
        NDS = [0.2316,0.2024,0.1833,0.1760,0.1500,0.1208,0.1079,0.1014,0.0850,0.0954,0.0884,0.0498,0.0001,0.0000,0.0000],
        mAP = [0.2015,0.1331,0.1053,0.0821,0.0663,0.0488,0.0443,0.0380,0.0358,0.0316,0.0216,0.0048,0.0003,0.0000,0.0000]
    ),

    BEVFormer_Tiny_Temp = dict(
        NDS = [0.3281,0.2393,0.2257,0.2108,0.2061,0.1855,0.1501,0.1415,0.1324,0.1375,0.1143,0.0533,0.0051,0.0000,0.0000],
        mAP = [0.2662,0.1838,0.1363,0.1100,0.0887,0.0686,0.0527,0.0423,0.0377,0.0321,0.0272,0.0012,0.0000,0.0000,0.0000],
    ),
    
    BEVFormer_Small = dict(
        NDS = [0.2790,0.2326,0.1845,0.1617,0.1292,0.1283,0.1254,0.1192,0.1254,0.1175,0.1187,0.0751,0.0630,0.0288,0.0287],
        mAP = [0.1893,0.1312,0.0839,0.0555,0.0322,0.0274,0.0206,0.0105,0.0133,0.0100,0.0105,0.0030,0.0007,0.0000,0.0000],
    ),

    BEVFormer_Small_Temp = dict(
        NDS = [0.3990,0.3063,0.2548,0.2454,0.2247,0.2046,0.1878,0.1740,0.1685,0.1549,0.1541,0.1147,0.0781,0.0833,0.0679], 
        mAP = [0.3546,0.2867,0.2129,0.1920,0.1516,0.1277,0.0985,0.0821,0.0661,0.0690,0.0522,0.0069,0.0019,0.0013,0.0012], 
    ),

    BEVFormer_Base = dict(
        NDS = [0.3563,0.2964,0.2609,0.2359,0.2233,0.1883,0.1859,0.1694,0.1450,0.1300,0.1264,0.1122,0.0772,0.0343,0.0026],
        mAP = [0.3185,0.2318,0.1723,0.1236,0.0901,0.0750,0.0552,0.0442,0.0325,0.0260,0.0188,0.0006,0.0000,0.0000,0.0000],
    ),

    BEVFormer_Base_Temp = dict(
        NDS = [0.4254,0.3109,0.2339,0.2052,0.1799,0.1574,0.1558,0.1229,0.1258,0.1204,0.1225,0.0484,0.0772,0.0000,0.0000],
        mAP = [0.3766,0.2598,0.1627,0.1156,0.0768,0.0609,0.0434,0.0250,0.0174,0.0102,0.0065,0.0000,0.0000,0.0000,0.0000],
    ),

    DETR3D_CBGS = dict(
        NDS = [0.3899,0.3479,0.3188,0.2771,0.2386,0.2362,0.2060,0.1879,0.1770,0.1703,0.1817,0.1214,0.1110,0.0675,0.0480],
        mAP = [0.3219,0.2643,0.2073,0.1545,0.1094,0.0908,0.0735,0.0567,0.0462,0.0373,0.0296,0.0029,0.0000,0.0000,0.0000],
    ),

    DETR3D = dict(
        NDS = [0.3743,0.3058,0.2855,0.2507,0.2152,0.1960,0.1854,0.1775,0.1653,0.1541,0.1487,0.1106,0.0905,0.0605,0.0353],
        mAP = [0.3112,0.2453,0.1954,0.1559,0.1251,0.0977,0.0832,0.0682,0.0581,0.0431,0.0327,0.0075,0.0019,0.0004,0.0000],
    ),

    PETR_R50 = dict(
        NDS = [0.3205,0.2726,0.2553,0.2240,0.1891,0.1641,0.1327,0.1223,0.1282,0.1106,0.0843,0.0511,0.0247,0.0000,0.0000],
        mAP = [0.2970,0.2187,0.1634,0.1154,0.0819,0.0657,0.0487,0.0355,0.0255,0.0180,0.0100,0.0000,0.0000,0.0000,0.0000],
    ),

    PETR_Vov = dict(
        NDS = [0.4024,0.3576,0.3018,0.2598,0.2487,0.2110,0.1990,0.1827,0.1688,0.1612,0.1562,0.0817,0.0558,0.0073,0.0000],
        mAP = [0.3804,0.2983,0.2264,0.1660,0.1316,0.1117,0.0864,0.0748,0.0558,0.0464,0.0383,0.0016,0.0000,0.0000,0.0000],
    ),

    FCOS3D = dict(
        NDS = [0.3309,0.2908,0.2701,0.2334,0.2191,0.2029,0.1747,0.1680,0.1678,0.1404,0.1438,0.1026,0.0373,0.0000,0.0000],
        mAP = [0.3083,0.2532,0.2082,0.1587,0.1385,0.1121,0.0929,0.0759,0.0627,0.0524,0.0423,0.0085,0.0005,0.0000,0.0000],
    ),

    PGD_Det = dict(
        NDS = [0.3525,0.3091,0.2748,0.2514,0.2329,0.2154,0.1999,0.1971,0.1819,0.1776,0.1759,0.1094,0.0399,0.0096,0.0000],
        mAP = [0.3344,0.2666,0.2203,0.1819,0.1504,0.1234,0.1017,0.0838,0.0722,0.0631,0.0502,0.0104,0.0021,0.0001,0.0000],
    ),

    BEVDepth_R50 = dict( # double checked
        NDS = [0.3759,0.2857,0.2459,0.2244,0.1945,0.1821,0.1544,0.1297,0.1434,0.1157,0.1063,0.0600,0.0276,0.0052,0.0000], 
        mAP = [0.3248,0.2126,0.1655,0.1328,0.0992,0.0733,0.0680,0.0496,0.0415,0.0310,0.0275,0.0041,0.0003,0.0000,0.0000],
    ), 

    BEVDepth4D_R50 = dict(
        NDS = [0.4309,0.3826,0.3586,0.3250,0.3006,0.2429,0.2674,0.2250,0.2166,0.1758,0.1998,0.1452,0.0729,0.0562,0.0319], 
        mAP = [0.3620,0.2941,0.2388,0.1804,0.1519,0.1191,0.0987,0.0860,0.0693,0.0560,0.0502,0.0104,0.0021,0.0003,0.0000],
    ), 

    BEVDepth_R101 = dict( # double checked
        NDS = [0.3767,0.3067,0.2743,0.2506,0.1965,0.1811,0.1635,0.1375,0.1346,0.1235,0.0996,0.0697,0.0374,0.0364,0.0000],
        mAP = [0.3276,0.2324,0.1794,0.1397,0.1071,0.0777,0.0698,0.0626,0.0492,0.0456,0.0421,0.0100,0.0010,0.0001,0.0000],
    ), 

    BEVDet_R50 = dict(
        NDS = [0.3289,0.2048,0.1757,0.1489,0.1309,0.1094,0.1060,0.0927,0.0917,0.0604,0.0080,0.0000,0.0000,0.0000,0.0000],
        mAP = [0.2831,0.1551,0.1170,0.0728,0.0540,0.0405,0.0384,0.0199,0.0161,0.0064,0.0080,0.0000,0.0000,0.0000,0.0000],
    ),

    BEVDet4D_R50 = dict(
        NDS = [0.3848,0.3008,0.2533,0.2310,0.1968,0.1865,0.1580,0.1445,0.1341,0.1275,0.1354,0.0843,0.0000,0.0000,0.0000],
        mAP = [0.3007,0.1901,0.1293,0.0975,0.0744,0.0542,0.0415,0.0245,0.0183,0.0173,0.0160,0.0004,0.0000,0.0000,0.0000]
    ),

    BEVDet_R101 = dict(
        NDS = [0.3491,0.2322,0.2027,0.1800,0.1682,0.1424,0.1583,0.1163,0.1056,0.1067,0.1058,0.0522,0.0001,0.0000,0.0000],
        mAP = [0.2977,0.1731,0.1373,0.1100,0.0823,0.0667,0.0524,0.0473,0.0470,0.0366,0.0301,0.0011,0.0001,0.0000,0.0000],
    ),

    BEVDet_Swin_Tiny = dict(
        NDS = [0.3306,0.2090,0.1821,0.1435,0.1269,0.1167,0.1068,0.0996,0.0835,0.0592,0.0720,0.0000,0.0000,0.0000,0.0000],
        mAP = [0.2847,0.1385,0.1031,0.0812,0.0615,0.0443,0.0409,0.0308,0.0286,0.0236,0.0170,0.0000,0.0000,0.0000,0.0000],
    ),
    
    )

# # rebenchmark
# PGD_Det_attack_target = dict(
#     severity = 'max_steps',
#     max_steps = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50],
#     BEVFormer_Tiny = dict(
#         mAP = [0.2015,0.1603,0.1333,0.1176,0.0979,0.0920,0.0396,0.0168,0.0132,0.0107],
#     ),
#     BEVFormer_Tiny_Temp = dict(
#         mAP = [0.2662,0.2177,0.1906,0.1685,0.1525,0.1328,0.0778,0.0449,0.0254,0.0180],
#     ),
#     BEVFormer_Small = dict(
#         mAP = [0.1893,0.1570,0.1386,0.1241,0.1098,0.0954,0.0576,0.0410,0.0295,0.0250],
#     ),
#     BEVFormer_Small_Temp = dict(
#         mAP = [0.3546,0.2645,0.2075,0.1943,0.1645,0.1397,0.1106,0.0578,0.0524,0.0516],
#     ),
#     BEVFormer_Base = dict(
#         mAP = [0.3185,0.2511,0.2276,0.2029,0.1952,0.1837,0.1464,0.1286,0.0981,0.0829],
#     ),
#     BEVFormer_Base_Temp = dict(
#         mAP = [0.3766,0.2871,0.2483,0.2149,0.1856,0.1788,0.1380,0.0863,0.0735,0.0654],
#     ),
#     DETR3D_CBGS = dict(
#         mAP = [0.3219,0.2820,0.2509,0.2228,0.1942,0.1797,0.1095,0.0770,0.0628,0.0501], 
#     ),
#     DETR3D = dict(
#         mAP = [0.3112,0.2915,0.2703,0.2441,0.2314,0.2172,0.1572,0.1159,0.0963,0.0754],
#     ),
#     FCOS3D = dict(
#         mAP = [0.3083,0.2652,0.2404,0.2132,0.1897,0.1784,0.1270,0.1041,0.0861,0.0799],
#     ),
#     PGD_Det = dict(
#         mAP = [0.3344,0.2997,0.2681,0.2471,0.2220,0.2131,0.1599,0.1326,0.1044,0.0951],
#     ),
#     BEVDepth_R50 = dict(
#         mAP = [0.3248,0.2681,0.2494,0.2374,0.2308,0.2064,0.1867,0.1436,0.1325,0.1063],
#     ),
#     BEVDet_R50 = dict(
#         mAP = [0.2831,0.2007,0.1698,0.1482,0.1401,0.1315,0.0863,0.0552,0.0320,0.0200],
#     )
# )


pgd_attack_local = dict(
    severity = 'max_steps',
    max_steps = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50],
    BEVFormer_Tiny = dict(
        NDS = [0.2316,0.1954,0.1668,0.1554,0.1405,0.1292,0.0982,0.0336,0.0230,0.0194],
        mAP = [0.2015,0.1572,0.1172,0.0971,0.0722,0.0511,0.0202,0.0098,0.0067,0.0038],
    ),
    BEVFormer_Tiny_Temp = dict(
        NDS = [0.3281,0.2146,0.1722,0.1452,0.1371,0.1218,0.0833,0.0521,0.0438,0.0419],
        mAP = [0.2662,0.1914,0.1479,0.1057,0.0849,0.0677,0.0276,0.0168,0.0138,0.0052],
    ),
    BEVFormer_Small = dict(
        NDS = [0.2790,0.2101,0.1854,0.1638,0.1555,0.1394,0.0989,0.0858,0.0743,0.0660],
        mAP = [0.1893,0.1703,0.1526,0.1285,0.1221,0.1001,0.0418,0.0221,0.0090,0.0062],
    ),
    BEVFormer_Small_Temp = dict(
        NDS = [0.3990,0.2506,0.1883,0.1647,0.1460,0.1376,0.0868,0.0551,0.0438,0.0263],
        mAP = [0.3546,0.2270,0.1619,0.1364,0.1102,0.0973,0.0443,0.0345,0.0221,0.0209],
    ),
    BEVFormer_Base = dict(
        NDS = [0.3563,0.2541,0.1961,0.1675,0.1565,0.1443,0.1107,0.0881,0.0713,0.0632],
        mAP = [0.3185,0.2253,0.1617,0.1303,0.1074,0.0923,0.0441,0.0239,0.0121,0.0062],
    ),
    BEVFormer_Base_Temp = dict(
        NDS = [0.4254,0.2645,0.2146,0.1842,0.1511,0.1402,0.0973,0.0554,0.0450,0.0292],
        mAP = [0.3766,0.2385,0.1965,0.1531,0.1241,0.1004,0.0518,0.0220,0.0183,0.0112],
    ),
    DETR3D_CBGS = dict(
        NDS = [0.3899,0.3080,0.2653,0.2335,0.2182,0.1856,0.1465,0.1205,0.1117,0.0969],
        mAP = [0.3219,0.2866,0.2410,0.2134,0.1960,0.1670,0.1105,0.0767,0.0554,0.0423], 
    ),
    DETR3D = dict(
        NDS = [0.3743,0.2878,0.2561,0.2180,0.1972,0.1807,0.1327,0.1040,0.0858,0.0764],
        mAP = [0.3112,0.2776,0.2413,0.2145,0.1837,0.1628,0.1020,0.0595,0.0395,0.0278],
    ),
    PETR_R50 = dict(
        NDS = [0.3205,0.2305,0.1881,0.1576,0.1439,0.1227,0.0920,0.0663,0.0312,0.0215],
        mAP = [0.2970,0.2110,0.1476,0.1119,0.0892,0.0648,0.0360,0.0201,0.0155,0.0114],
    ),
    PETR_Vov = dict(
        NDS = [0.4024,0.2721,0.1987,0.1650,0.1555,0.1420,0.1009,0.0702,0.0554,0.0298],
        mAP = [0.3804,0.2415,0.1540,0.1164,0.0963,0.0725,0.0351,0.0200,0.0145,0.0099],
    ),
    FCOS3D = dict(
        NDS = [0.3309,0.2115,0.1642,0.1466,0.1353,0.0927,0.0779,0.0596,0.0538,0.0517],
        mAP = [0.3083,0.1647,0.1015,0.0671,0.0544,0.0379,0.0185,0.0122,0.0091,0.0064],
    ),
    PGD_Det = dict(
        NDS = [0.3525,0.2475,0.1919,0.1717,0.1295,0.1144,0.0650,0.0468,0.0159,0.0125],
        mAP = [0.3344,0.1984,0.1244,0.0919,0.0715,0.0577,0.0320,0.0211,0.0159,0.0125],
    ),
    BEVDepth_R50 = dict( # double checked
        NDS = [0.3759,0.2671,0.2270,0.2072,0.1898,0.1770,0.1339,0.0975,0.0891,0.0751],
        mAP = [0.3248,0.2508,0.2125,0.1787,0.1533,0.1449,0.0892,0.0614,0.0434,0.0368],
    ),
    BEVDepth4D_R50 = dict(
        NDS = [0.4309,0.3665,0.2997,0.2504,0.2268,0.1896,0.1388,0.0964,0.0918,0.0632], 
        mAP = [0.3620,0.3120,0.2606,0.2141,0.1899,0.1463,0.0873,0.0539,0.0421,0.0338],
    ), 
    BEVDepth_R101 = dict( # double checked
        NDS = [0.3767,0.2827,0.2401,0.2158,0.1932,0.1793,0.1297,0.1074,0.0888,0.0850],
        mAP = [0.3276,0.2701,0.2334,0.2077,0.1801,0.1549,0.0974,0.0665,0.0563,0.0437],
    ),
    BEVDet_R50 = dict(
        NDS = [0.3289,0.2093,0.1765,0.1531,0.1466,0.1338,0.1163,0.0734,0.0593,0.0519],
        mAP = [0.2831,0.1897,0.1542,0.1246,0.1134,0.1018,0.0614,0.0400,0.0284,0.0253],
    ),
    BEVDet4D_R50 = dict(
        NDS = [0.3848,0.2817,0.2238,0.2032,0.1811,0.1644,0.0979,0.0802,0.0556,0.0614],
        mAP = [0.3007,0.2136,0.1700,0.1525,0.1318,0.1039,0.0619,0.0426,0.0316,0.0204]
    ),
    BEVDet_R101 = dict(
        NDS = [0.3491,0.2275,0.1954,0.1697,0.1567,0.1523,0.1216,0.1004,0.0717,0.0705],
        mAP = [0.2977,0.2047,0.1695,0.1431,0.1263,0.1184,0.0770,0.0540,0.0361,0.0281],
    ),
    BEVDet_Swin_Tiny = dict(
        NDS = [0.3306,0.1775,0.1513,0.1321,0.1250,0.1017,0.0999,0.0658,0.0450,0.0547],
        mAP = [0.2847,0.1614,0.1307,0.1115,0.1040,0.0859,0.0435,0.0186,0.0109,0.0049],
    ),
    )

dynamic_patch_untarget_attack = dict(
    severity = 'scale',
    scale = [0, 0.1, 0.2, 0.3, 0.4],
    BEVFormer_Tiny = dict(
        NDS = [0.2316,0.1838,0.1375,0.1014,0.0588],
        mAP = [0.2015,0.1028,0.0543,0.0213,0.0007],
    ),
    BEVFormer_Tiny_Temp = dict(
        NDS = [0.3281,0.2277,0.1709,0.1135,0.0467],
        mAP = [0.2662,0.1434,0.0652,0.0249,0.0056],
    ),
    BEVFormer_Small = dict(
        NDS = [0.2790,0.2145,0.1642,0.1088,0.0839],
        mAP = [0.1893,0.1116,0.0450,0.0120,0.0015],
    ),
    BEVFormer_Small_Temp = dict(
        NDS = [0.3990,0.2805,0.1924,0.1238,0.1018],
        mAP = [0.3546,0.2245,0.1048,0.0295,0.0040],
    ),
    BEVFormer_Base = dict(
        NDS = [0.3563,0.2767,0.2037,0.1430,0.0869],
        mAP = [0.3185,0.1856,0.0881,0.0117,0.0002],
    ),
    BEVFormer_Base_Temp = dict(
        NDS = [0.4254,0.2739,0.1976,0.1149,0.0895],
        mAP = [0.3766,0.2068,0.0911,0.0084,0.0001],
    ),
    DETR3D_CBGS = dict(
        NDS = [0.3899,0.3261,0.2327,0.1448,0.1049],
        mAP = [0.3219,0.2089,0.0782,0.0135,0.0006], 
    ),
    DETR3D = dict(
        NDS = [0.3743,0.2983,0.2106,0.1469,0.0630],
        mAP = [0.3112,0.1835,0.0713,0.0106,0.0003],
    ),
    PETR_R50 = dict(
        NDS = [0.3205,0.2221,0.0944,0.0268,0.0116],
        mAP = [0.2970,0.1292,0.0060,0.0000,0.0000],
    ),
    PETR_Vov = dict(
        NDS = [0.4024,0.2580,0.1359,0.0814,0.0658],
        mAP = [0.3804,0.1820,0.0224,0.0003,0.0000],
    ),
    FCOS3D = dict(
        NDS = [0.3309,0.2396,0.1396,0.1109,0.0000],
        mAP = [0.3083,0.1647,0.0458,0.0066,0.0000],
    ),
    PGD_Det = dict(
        NDS = [0.3525,0.2612,0.1676,0.0130,0.0085],
        mAP = [0.3344,0.1898,0.0551,0.0000,0.0000],
    ),
    BEVDepth_R50 = dict( # double checked
        NDS = [0.3759,0.2868,0.1721,0.0710,0.0057],
        mAP = [0.3248,0.1753,0.0617,0.0043,0.0000],
    ),
    BEVDepth4D_R50 = dict(
        NDS = [0.4309,0.3631,0.2911,0.1664,0.1348], 
        mAP = [0.3620,0.2404,0.1171,0.0262,0.0004],
    ), 
    BEVDepth_R101 = dict( # double checked
        NDS = [0.3767,0.2824,0.1655,0.0658,0.0066],
        mAP = [0.3276,0.1693,0.0607,0.0007,0.0000],
    ),
    BEVDet_R50 = dict(
        NDS = [0.3289,0.2308,0.1422,0.0580,0.0096],
        mAP = [0.2831,0.1285,0.0296,0.0005,0.0000],
    ),
    BEVDet4D_R50 = dict(
        NDS = [0.3848,0.3061,0.2541,0.1042,0.0904],
        mAP = [0.3007,0.1701,0.0746,0.0083,0.0000]
    ),
    BEVDet_R101 = dict(
        NDS = [0.3491,0.2395,0.1644,0.0622,0.0044],
        mAP = [0.2977,0.1209,0.0386,0.0007,0.0000],
    ),
    BEVDet_Swin_Tiny = dict(
        NDS = [0.3306,0.2735,0.1838,0.0817,0.0069],
        mAP = [0.2847,0.1778,0.0663,0.0088,0.0000],
    ),

    )


dynamic_patch_loc_attack = dict(
    severity = 'scale',
    scale = [0, 0.1, 0.2, 0.3, 0.4],
    BEVFormer_Tiny = dict(
        NDS = [0.2316,0.1874,0.1589,0.1346,0.1021],
        mAP = [0.2015,0.1501,0.0979,0.0595,0.0235],
    ),
    BEVFormer_Tiny_Temp = dict(
        NDS = [0.3281,0.2161,0.1714,0.1380,0.0849],
        mAP = [0.2662,0.1939,0.1284,0.0811,0.0358],
    ),
    BEVFormer_Small = dict(
        NDS = [0.2790,0.2341,0.1901,0.1503,0.1137],
        mAP = [0.1893,0.1827,0.1436,0.0807,0.0316],
    ),
    BEVFormer_Small_Temp = dict(
        NDS = [0.3990,0.2667,0.1808,0.1516,0.1249],
        mAP = [0.3546,0.2356,0.1493,0.1109,0.0592],
    ),
    BEVFormer_Base = dict(
        NDS = [0.3563,0.2830,0.2144,0.1490,0.1175],
        mAP = [0.3185,0.2622,0.1955,0.1012,0.0649],
    ),
    BEVFormer_Base_Temp = dict(
        NDS = [0.4254,0.2830,0.2144,0.1490,0.1175],
        mAP = [0.3766,0.2622,0.1955,0.1012,0.0649],
    ),
    DETR3D_CBGS = dict(
        NDS = [0.3899,0.3130,0.2520,0.1722,0.1360],
        mAP = [0.3219,0.2823,0.2223,0.1425,0.0825], 
    ),
    DETR3D = dict(
        NDS = [0.3743,0.2959,0.2248,0.1704,0.1209],
        mAP = [0.3112,0.2707,0.2027,0.1323,0.0596],
    ),
    PETR_R50 = dict(
        NDS = [0.3205,0.2269,0.1560,0.1012,0.0478],
        mAP = [0.2970,0.2043,0.1015,0.0457,0.0130],
    ),
    PETR_Vov = dict(
        NDS = [0.4024,0.2524,0.1627,0.1243,0.0798],
        mAP = [0.3804,0.2317,0.1092,0.0479,0.0070],
    ),
    FCOS3D = dict(
        NDS = [0.3309,0.2110,0.1494,0.0985,0.0596],
        mAP = [0.3083,0.1710,0.0897,0.0388,0.0201],
    ),
    PGD_Det = dict(
        NDS = [0.3525,0.2583,0.1707,0.1303,0.0891],
        mAP = [0.3344,0.2148,0.1246,0.0618,0.0182],
    ),
    BEVDepth_R50 = dict( # double checked
        NDS = [0.3759,0.2878,0.2039,0.1517,0.1128],
        mAP = [0.3248,0.2583,0.1532,0.0954,0.0393],
    ),
    BEVDepth4D_R50 = dict(
        NDS = [0.4309,0.3663,0.2919,0.1845,0.1275], 
        mAP = [0.3620,0.3152,0.2192,0.1045,0.0359],
    ), 
    BEVDepth_R101 = dict(
        NDS = [0.3767,0.2669,0.2104,0.1275,0.0956],
        mAP = [0.3276,0.2549,0.1689,0.0935,0.0482],
    ),
    BEVDet_R50 = dict(
        NDS = [0.3289,0.2419,0.1789,0.1219,0.0821],
        mAP = [0.2831,0.2180,0.1308,0.0614,0.0325],
    ),
    BEVDet4D_R50 = dict(
        NDS = [0.3848,0.3436,0.2495,0.1574,0.1123],
        mAP = [0.3007,0.2521,0.1710,0.0873,0.0327]
    ),
    BEVDet_R101 = dict(
        NDS = [0.3491,0.2478,0.1753,0.1264,0.0738],
        mAP = [0.2977,0.2119,0.1236,0.0679,0.0341],
    ),
    BEVDet_Swin_Tiny = dict(
        NDS = [0.3306,0.2692,0.1831,0.1054,0.0743],
        mAP = [0.2847,0.2356,0.1378,0.0665,0.0366],
    ),
)

def collect_data(data, severity, catagory):

    results = []

    for i in range(len(data)):
        data_ = data[i]
        result = dict()
        result['patch_scale'] = data_['patch_scale']
        result['step_size'] = data_['step_size']
        result['max_steps'] = data_['max_steps']
        result[f'{catagory}_{severity}'] = 0
        num = 0
        for key in data_.keys():
            if severity in key and catagory in key:
                result[f'{catagory}_{severity}'] += data_[key]
                num = num + 1
        # calculate average ap under different threshold
        result[f'{catagory}_{severity}'] /= num
        results.append(result)

    return results


def plot_api(x, y, xtitle, ytitle, out_path):
    
    plt.plot(x, y, 'o-', color='steelblue', markersize=2)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(out_path)
    plt.cla()


def visualize_object(data):
    """Visualize severity of different object according to experiment settings (e.x. patch_size, step_size, etc.)
    Args:
        data (list): each element is a dict output by the model
    """
    for class_ in CLASSES:
        for severity_ in CLASSES_severity:
            if severity_ == 'AP':
                result = collect_data(data, severity_, class_)
                result.sort(key=lambda k: k['patch_scale'])
                x = [exp['patch_scale'] for exp in result]
                y = [exp[f'{class_}_{severity_}'] for exp in result]
                plot_api(x, y, 'patch_scale', f'{class_}_{severity_}', out_path=os.path.join('visual', 'patch_scale', f'ap_{class_}.pdf'))
            

def multi_plot_api(xs, ys, labels, xtitle, ytitle, out_path, size=(10,6), legend_size=None, fontsize=None, tick_font_size=None):
    assert isinstance(xs, list or tuple)
    assert isinstance(ys, list or tuple)
    assert isinstance(labels, list or tuple)
    assert len(xs) == len(ys) and len(xs) == len(labels)

    plt.figure(figsize=size)
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if fontsize is None:
        fontsize = 20
    ax.set_xlabel(..., fontsize=fontsize)
    ax.set_ylabel(..., fontsize=fontsize)
    for i in range(len(xs)):
        label_ = labels[i].replace('_', '-')
        assert len(xs[i]) == len(ys[i]), f"x doesn't match y in {label_}"
        ax.plot(xs[i], ys[i], f'{MARKER[labels[i]]}-', c=COLORS[labels[i]], label=label_, markersize=5)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    if tick_font_size:
        plt.xticks(fontsize=tick_font_size)
        plt.yticks(fontsize=tick_font_size)
    if legend_size:
        plt.legend(prop={'size':legend_size})
    else:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.cla()


def plot_scatter_api(xs, ys, labels, xtitle, ytitle, parameters, out_path, size=(6, 6), change_size=True):
    assert isinstance(xs, list or tuple)
    assert isinstance(ys, list or tuple)
    assert isinstance(labels, list or tuple)
    assert len(xs) == len(ys) and len(xs) == len(labels)

    if change_size:
        plt.rcParams['figure.figsize'] = size
    fig, ax = plt.subplots()
    ax.grid(linestyle = '--', linewidth = 0.5)
    for i in range(len(xs)):
        label_ = labels[i].replace('_', '-')
        ax.scatter(x=[xs[i]], y=[ys[i]], marker=MARKER[labels[i]], c=COLORS[labels[i]], alpha=0.8, s=200, label=label_)
        
    plt.xlabel(xtitle, fontsize=15)
    plt.ylabel(ytitle, fontsize=15)
    plt.legend(markerscale=0.3, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.cla()


def plot_bar_api(values, models, types, out_path, total_width=0.8, ylabel='mAP'):
    assert isinstance(values, dict)

    x =list(range(len(values)))
    model_num = len(models)
    total_width = 0.5
    width = total_width / model_num

    hatches = [None, '/', '\\', '.']

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.grid(axis='y', linestyle = '--', linewidth = 0.5)

    y_major_locator=MultipleLocator(0.05)
    ax.yaxis.set_major_locator(y_major_locator)

    for i, model in enumerate(models):
        model_ = model.replace('_', '-')
        ax.bar(x, values[model], width=width, label=model_, fc=COLORS[model], hatch=hatches[i])
        for j in range(len(x)):
            x[j] = x[j] + width

    x =list(range(len(values)))
    for j in range(len(x)):
            x[j] = x[j] + total_width / 2
    plt.xticks(x, types, fontsize=15) 
    plt.ylabel(ylabel, fontsize=15)
    plt.tick_params(top=False,bottom=False,left=False,right=False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.cla()


def parse_data(results, relative=False, metric='mAP', models=None):
    """
    Args:
        results (dict)
        relative (bool): visualize relative performance drop
    """
    assert isinstance(results, dict)
    # keys = list(results.keys())
    if models is None:
        keys = MODELS
    else:
        keys = models
    xs = []
    ys = []
    labels = []
    for key in keys:
        if key == 'severity' or key == results['severity']:
            continue
        x = results[results['severity']]
        y = results[key][metric]
        if relative:
            y = [y[i] / y[0] for i in range(len(y))]
        xs.append(x)
        ys.append(y)
        labels.append(key)
    
    return xs, ys, labels


def collect_clean_accuracy(results_dict, val=False, metric='mAP'):
    assert isinstance(results_dict, dict)

    # models = list(results_dict.keys())
    models = MODELS
    acc_all = []
    model_names = []
    for model in models:
        if model == 'severity' or model == results_dict['severity']:
            continue
        if not val:
            # map on mini datasets
            acc = results_dict[model][metric][0]
        else:
            if metric == 'mAP':
                # map on validation datasets
                acc = VAL_MAP[model]
            elif metric == 'NDS':
                acc = VAL_NDS[model]
                
        acc_all.append(acc)
        model_names.append(model)
    
    return acc_all, model_names


def collect_model_size():
    models = MODELS
    size_all = []
    for i, model in enumerate(models):
        size_all.append(PARAMETERS[model])

    return size_all, models


def collect_robustness_acc(results_dicts, param=False, val=False, metric='mAP', range=-1):
    """Calculate average adversarial robustness v.s. clean accuracy
    """
    assert isinstance(results_dicts, List)
    assert isinstance(results_dicts[0], dict)

    if param:
        clean_accs, model_names = collect_model_size()
    else:
        clean_accs, model_names = collect_clean_accuracy(results_dicts[0], val, metric)

    adver_accs = []
    for model_name in model_names:
        adver_acc = []
        for results_dict in results_dicts:
            adver_acc.extend(results_dict[model_name][metric][1: ])
        adver_acc = np.mean(np.array(adver_acc))
        adver_accs.append(adver_acc)

    return clean_accs, adver_accs, model_names


def collect_average_adv_acc(results_dicts, metric='mAP', range=-1):
    """Collect average adversarial mAP/NDS under attacks
    """
    assert isinstance(results_dicts, List)
    assert isinstance(results_dicts[0], dict)

    model_names = MODELS

    model_accs = dict()
    # init dict value as list
    for model_name in model_names:
        model_accs[model_name] = []

    for model_name in model_names:
        for results_dict in results_dicts:
            adver_acc = []
            adver_acc.extend(results_dict[model_name][metric][1:])
            adver_acc = np.mean(np.array(adver_acc))
            model_accs[model_name].append(adver_acc)

    return model_accs


def appendix_curve_plot_api(model_name, models):
    metrics = ['mAP', 'NDS']
    for metric in metrics:
        xs, ys, labels = parse_data(pgd_attack_untarget, relative=False, metric=metric, models=models)
        multi_plot_api(xs, ys, labels, 'attack iterations', metric, f'visual/appendix/{model_name}-pgd-untarget-{metric}.pdf', \
                    size=(15, 8), legend_size=25, fontsize=25, tick_font_size=15)
        xs, ys, labels = parse_data(pgd_attack_local, relative=False, metric=metric, models=models)
        multi_plot_api(xs, ys, labels, 'attack iterations', metric, f'visual/appendix/{model_name}-pgd-loc-{metric}.pdf', \
                    size=(15, 8), legend_size=25, fontsize=25, tick_font_size=15)
        xs, ys, labels = parse_data(dynamic_patch_untarget_attack, relative=False, metric=metric, models=models)
        multi_plot_api(xs, ys, labels, 'patch scale', metric, f'visual/appendix/{model_name}-patch-untarget-{metric}.pdf', \
                    size=(15, 8), legend_size=25, fontsize=25, tick_font_size=15)
        xs, ys, labels = parse_data(dynamic_patch_loc_attack, relative=False, metric=metric, models=models)
        multi_plot_api(xs, ys, labels, 'patch scale', metric, f'visual/appendix/{model_name}-patch-loc-{metric}.pdf', \
                    size=(15, 8), legend_size=25, fontsize=25, tick_font_size=15)


if __name__ == '__main__':

    MODELS = [
        'BEVFormer_Tiny',
        'BEVFormer_Tiny_Temp',
        'BEVFormer_Small',
        'BEVFormer_Small_Temp',
        'BEVFormer_Base',
        'BEVFormer_Base_Temp',
        'DETR3D_CBGS',
        'DETR3D',
        'FCOS3D',
        'PGD_Det',
        'BEVDepth_R50',
        'BEVDet_R50',
    ]

    # ##############################################################
    # # Fig1 overall results
    # #  - [x] Add all the models here
    # #  - [ ] Add target attack results here
    # #  - [x] Add DETR wo CBGS Validation results here
    # #  - [x] Use BEVDepth-R101 and BEVDet-R50 here
    # # 'BEVFormer_Tiny','BEVFormer_Tiny_Temp',
    # MODELS = ['BEVFormer_Small','BEVFormer_Small_Temp','BEVFormer_Base','BEVFormer_Base_Temp',
    #           'DETR3D_CBGS','DETR3D','FCOS3D','PGD_Det','BEVDepth_R50','BEVDepth4D_R50','BEVDepth_R101','BEVDet_R50','BEVDet_R101','BEVDet_Swin_Tiny',
    #           'BEVDet4D_R50', 'PETR_R50', 'PETR_Vov']
    # metric = 'NDS'
    # clean_accs, adver_accs, model_names = collect_robustness_acc([PGD_Det_attack_untarget, PGD_Det_attack_local, dynamic_patch_untarget_attack,dynamic_patch_loc_attack], 
    #                                         param=False, val=True, metric=metric, range=-1)
    # for i in range(len(model_names)):
    #     print(f'{model_names[i]}  clean_{metric}: {clean_accs[i]}  adver_{metric}: {adver_accs[i]}') # 'visual/figure/overall_average.pdf'
    # plot_scatter_api(clean_accs, adver_accs, model_names, metric, f'Adversarial {metric}', PARAMETERS, 'visual/figure/overall_average.pdf', \
    #                  size=(6, 6)) # 



    # ###############################################################
    # # Fig2: PGD_Det Untarget Attack Results
    # #  - [x] Add PETR here
    # #  - [x] Use BEVDepth-R101 and BEVDet-R101 here
    # metric = 'mAP'
    # MODELS = ['BEVFormer_Base','DETR3D_CBGS','FCOS3D','PGD_Det','BEVDepth_R101','BEVDet_R101','PETR_Vov']
    # xs, ys, labels = parse_data(pgd_attack_untarget, relative=False, metric=metric)
    # multi_plot_api(xs, ys, labels, 'attack iterations', metric, 'visual/figure/pgd_attack_untarget_curve.pdf', \
    #                size=(10, 6), legend_size=20, fontsize=20) # visual/figure/PGD_Det_attack_untarget_curve.pdf


    # ###############################################################
    # # Fig3: PGD_Det Localization Attack Results
    # #  - [x] Add PETR here
    # #  - [x] Use BEVDepth-R101 and BEVDet-R101 here
    # metric = 'mAP'
    # MODELS = ['BEVFormer_Base','DETR3D_CBGS','FCOS3D','PGD_Det','BEVDepth_R101','BEVDet_R101', 'PETR_Vov']
    # xs, ys, labels = parse_data(pgd_attack_local, relative=False, metric=metric)
    # multi_plot_api(xs, ys, labels, 'attack iterations', metric, 'visual/figure/pgd_attack_local_curve.pdf', \
    #                size=(10, 6), legend_size=20, fontsize=20) # visual/figure/PGD_Det_attack_local_curve.pdf


    # ###############################################################
    # # Table: Average NDS and mAP
    # #  - [x] Add PETR, BEVDepth-R101, BEVDet-R101, BEVDet-Swin-Tiny, BEVDet4D_R50
    # metric = 'mAP'
    # MODELS = ['BEVFormer_Tiny','BEVFormer_Tiny_Temp','BEVFormer_Small','BEVFormer_Small_Temp','BEVFormer_Base','BEVFormer_Base_Temp',
    #     'DETR3D_CBGS','DETR3D','FCOS3D','PGD_Det','BEVDepth_R50','BEVDepth4D_R50','BEVDepth_R101','BEVDet_R50','BEVDet_R101','BEVDet_Swin_Tiny',
    #     'PETR_R50', 'PETR_Vov','BEVDet4D_R50'
    # ]
    # print(metric)
    # attacks = [dynamic_patch_loc_attack,dynamic_patch_untarget_attack,PGD_Det_attack_untarget,PGD_Det_attack_local]
    # clean_accs, adver_accs, model_names = collect_robustness_acc(attacks, param=False, val=True, metric=metric, range=-1)
    # for i in range(len(model_names)):
    #     print(f'{model_names[i]}  clean_{metric}: {clean_accs[i]}  adver_{metric}: {adver_accs[i]}')
    # metric = 'NDS' 
    # print(metric)
    # clean_accs, adver_accs, model_names = collect_robustness_acc(attacks, param=False, val=True, metric=metric, range=-1)
    # for i in range(len(model_names)):
    #     print(f'{model_names[i]}  clean_{metric}: {clean_accs[i]}  adver_{metric}: {adver_accs[i]}')

 
    # plot_scatter_api(clean_accs, adver_accs, model_names, 'mAP', 'Adversarial mAP', PARAMETERS, 'visual/test.png') # visual/test.png


    # ###############################################################
    # # Bar1: Ablation Study: BEV Model v.s. Non-BEV Model
    # # - [ ] Make it look better !
    # metric = 'NDS'
    # MODELS = ['FCOS3D','PGD_Det','BEVDepth_R101','BEVDet_R101'] # visual/figure/bev_nonbev.pdf
    # model_accs = collect_average_adv_acc([pgd_attack_untarget, pgd_attack_local, dynamic_patch_untarget_attack, dynamic_patch_loc_attack], metric=metric)
    # plot_bar_api(model_accs, MODELS, ['pixel cls', 'pixel loc', 'patch cls', 'patch loc'], out_path='visual/figure/bev_nonbev.pdf', ylabel=metric)


    # ##############################################################
    # # Fig4: Ablation: model size
    # #  - [] 
    # # 'BEVFormer_Tiny','BEVFormer_Tiny_Temp',
    # MODELS = ['BEVFormer_Small','BEVFormer_Base','DETR3D_CBGS',
    #         'FCOS3D','PGD_Det',
    #         'BEVDepth_R50','BEVDepth_R101','BEVDet_R50','BEVDet_R101','BEVDet_Swin_Tiny', 'PETR_R50','PETR_Vov']
    # metric = 'NDS'
    # clean_accs, adver_accs, model_names = collect_robustness_acc([pgd_attack_untarget, pgd_attack_local, dynamic_patch_untarget_attack,dynamic_patch_loc_attack], 
    #                                         param=True, val=True, metric=metric, range=-1)
    # for i in range(len(model_names)):
    #     print(f'{model_names[i]}  clean_{metric}: {clean_accs[i]}  adver_{metric}: {adver_accs[i]}') # visual/figure/model_size.pdf
    # plot_scatter_api(clean_accs, adver_accs, model_names, 'Model Size (M)', f'Adversarial {metric}', PARAMETERS, 'visual/figure/model_size.pdf', \
    #                  size=(6, 6), change_size=False) # 


    # ###############################################################
    # # Fig5: DETR vs DETR-CBGS
    # #  - [x] Add PETR here
    # #  - [x] Use BEVDepth-R101 and BEVDet-R101 here
    # metric = 'NDS'
    # MODELS = ['DETR3D','DETR3D_CBGS']
    # xs, ys, labels = parse_data(PGD_Det_attack_untarget, relative=False, metric=metric)
    # multi_plot_api(xs, ys, labels, 'attack iterations', metric, 'visual/figure/cbgs_PGD_Det_attack_untarget.pdf', \
    #                size=(5.5, 4.5), legend_size=17, fontsize=17)
    # xs, ys, labels = parse_data(PGD_Det_attack_local, relative=False, metric=metric)
    # multi_plot_api(xs, ys, labels, 'attack iterations', metric, 'visual/figure/cbgs_PGD_Det_attack_local.pdf', \
    #                size=(5.5, 4.5), legend_size=17, fontsize=17)
    # xs, ys, labels = parse_data(dynamic_patch_untarget_attack, relative=False, metric=metric)
    # multi_plot_api(xs, ys, labels, 'attack iterations', metric, 'visual/figure/cbgs_patch_untarget_attack.pdf', \
    #                size=(5.5, 4.5), legend_size=17, fontsize=17)
    # xs, ys, labels = parse_data(dynamic_patch_loc_attack, relative=False, metric=metric)
    # multi_plot_api(xs, ys, labels, 'attack iterations', metric, 'visual/figure/cbgs_patch_loc_attack.pdf', \
    #                size=(5.5, 4.5), legend_size=17, fontsize=17)

    ###############################################################
    # Appendix: All attacks curve results
    metric = 'mAP'
    MODELS = [
    # 'BEVFormer_Tiny','BEVFormer_Tiny_Temp',
    'BEVFormer_Small','BEVFormer_Small_Temp','BEVFormer_Base','BEVFormer_Base_Temp',
    # 'DETR3D','DETR3D_CBGS',
    # 'FCOS3D','PGD_Det','BEVDepth_R101','BEVDepth_R50','BEVDet_R50','BEVDet_R101','BEVDet4D_R50','PETR_R50'
    ]
    appendix_curve_plot_api('bevformer', MODELS)
    MODELS = [
    #'BEVFormer_Tiny','BEVFormer_Tiny_Temp','BEVFormer_Small','BEVFormer_Small_Temp','BEVFormer_Base','BEVFormer_Base_Temp',
    'DETR3D','DETR3D_CBGS', 'PETR_R50', 'PETR_Vov'
    # 'FCOS3D','PGD_Det','BEVDepth_R101','BEVDepth_R50','BEVDet_R50','BEVDet_R101','BEVDet4D_R50','PETR_R50'
    ]
    appendix_curve_plot_api('detr', MODELS)
    MODELS = [
    # 'BEVFormer_Tiny','BEVFormer_Tiny_Temp','BEVFormer_Small','BEVFormer_Small_Temp','BEVFormer_Base','BEVFormer_Base_Temp',
    # 'DETR3D','DETR3D_CBGS', 'PETR_R50'
    'FCOS3D','PGD_Det',
    # 'BEVDepth_R101','BEVDepth_R50','BEVDet_R50','BEVDet_R101','BEVDet4D_R50',
    ]
    appendix_curve_plot_api('mono', MODELS)
    MODELS = [
    # 'BEVFormer_Tiny','BEVFormer_Tiny_Temp','BEVFormer_Small','BEVFormer_Small_Temp','BEVFormer_Base','BEVFormer_Base_Temp',
    # 'DETR3D','DETR3D_CBGS', 'PETR_R50'
    # 'FCOS3D','PGD_Det',
    'BEVDepth_R101','BEVDepth_R50','BEVDepth4D_R50','BEVDet_R50','BEVDet_R101','BEVDet4D_R50',
    ]
    appendix_curve_plot_api('bevdet', MODELS)


