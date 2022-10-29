from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

nan = np.infty
CLASSES = ['car', 'bus', 'truck', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
CLASSES_METRIC = ['AP', 'trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']
METRICS = ['mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']
PARAMETERS = dict(
    BEVFormer_Tiny = 33.6,
    BEVFormer_Tiny_Temp = 33.6,
    BEVFormer_Small = 59.6,
    BEVFormer_Small_Temp = 59.6,
    BEVFormer_Base = 69.1,
    BEVFormer_Base_Temp = 69.1,
    DETR3D_CBGS = 53.8,
    DETR3D = 53.8,
    FCOS3D = 55.1,
    PGD = 56.2,
    BEVDepth_R50 = 53.1,
    BEVDet_R50 = 48.2
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
    FCOS3D = 's', 
    PGD = 'p', 
    BEVDepth_R50 = 'D', 
    BEVDet_R50 = 'd'
)

COLORS = dict(
    BEVFormer_Tiny = '#4169E1', 
    BEVFormer_Tiny_Temp = '#0000FF', 
    BEVFormer_Small = '#9932CC', 
    BEVFormer_Small_Temp = '#800080', 
    BEVFormer_Base = '#F08080', 
    BEVFormer_Base_Temp = '#B22222', 
    DETR3D_CBGS = '#FFE4B5', 
    DETR3D = '#FFA500', 
    FCOS3D = '#00FF00', 
    PGD = '#2E8B57', 
    BEVDepth_R50 = '#FFFF00', 
    BEVDet_R50 = '#FFD700')

MODELS = [
    # 'BEVFormer_Tiny',
    # 'BEVFormer_Tiny_Temp',
    # 'BEVFormer_Small',
    # 'BEVFormer_Small_Temp',
    # 'BEVFormer_Base',
    'BEVFormer_Base_Temp',
    # 'DETR3D_CBGS',
    'DETR3D',
    'FCOS3D',
    'PGD',
    'BEVDepth_R50',
    'BEVDet_R50',
]
VAL_MAP = dict(
    BEVFormer_Tiny = 0,
    BEVFormer_Tiny_Temp = 0.2524,
    BEVFormer_Small = 0,
    BEVFormer_Small_Temp = 0.3699,
    BEVFormer_Base = 0,
    BEVFormer_Base_Temp = 0.4167,
    DETR3D_CBGS = 0.3494,
    DETR3D = 0,
    FCOS3D = 0.3214,
    PGD = 0.3360,
    BEVDepth_R50 = 0.3327,
    BEVDet_R50 = 0.2987
)


pgd_attack_untarget = dict(
    metric = 'max_steps',
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
        NDS = [0.2790,0.2503,0.2195,0.1907,0.1679,0.1669,0.1650,0.1496,0.1281,0.1350,0.1050,0.0948,0.0524,0.0636,0.0256],
        mAP = [0.1893,0.1468,0.1155,0.0852,0.0694,0.0569,0.0414,0.0376,0.0338,0.0312,0.0186,0.0029,0.0006,0.0031,0.0001],
    ),

    BEVFormer_Small_Temp = dict(
        NDS = [0.3990,0.2869,0.2497,0.2272,0.2063,0.1760,0.1752,0.1536,0.1236,0.1460,0.1372,0.0979,0.0362,0.0,0.0], # remanchmark the last two
        mAP = [0.3546,0.2455,0.1835,0.1470,0.1262,0.1030,0.0813,0.0556,0.0421,0.0381,0.0375,0.0107,0.0005,0.0,0.0], # remanchmark the last two
    ),

    BEVFormer_Base = dict(
        NDS = [0.3563,0.3082,0.2828,0.2660,0.2518,0.2412,0.2309,0.2233,0.2083,0.2062,0.1938,0.1482,0.1287,0.0879,0.0587],
        mAP = [0.3185,0.2400,0.1955,0.1692,0.1462,0.1272,0.1194,0.1095,0.0905,0.0871,0.0775,0.0447,0.0331,0.0221,0.0159],
    ),

    BEVFormer_Base_Temp = dict(
        NDS = [0.4254,0.3201,0.2754,0.2291,0.2127,0.2174,0.1992,0.1831,0.1757,0.1460,0.1661,0.0974,0.0808,0.0334,0.0260],
        mAP = [0.3766,0.2700,0.2190,0.1637,0.1416,0.1348,0.1106,0.0980,0.0802,0.0662,0.0553,0.0221,0.0109,0.0017,0.0001],
    ),

    DETR3D_CBGS = dict(
        NDS = [0.3899,0.3660,0.3498,0.3409,0.3244,0.3151,0.3024,0.2981,0.2801,0.2895,0.2770,0.2257,0.1903,0.1901,0.1385],
        mAP = [0.3219,0.2756,0.2456,0.2283,0.2077,0.1932,0.1745,0.1705,0.1538,0.1551,0.1530,0.1017,0.0753,0.0559,0.0448], # double check this results
    ),

    DETR3D = dict(
        NDS = [0.3743,0.3213,0.2875,0.2622,0.2549,0.2179,0.2202,0.1949,0.1894,0.1751,0.1674,0.1133,0.0808,0.0859,0.0347],
        mAP = [0.3112,0.2579,0.2139,0.1835,0.1636,0.1353,0.1105,0.1053,0.0988,0.0895,0.0747,0.0251,0.0079,0.0015,0.000],
    ),

    FCOS3D = dict(
        NDS = [0.3309,0.2908,0.2701,0.2334,0.2191,0.2029,0.1747,0.1680,0.1678,0.1404,0.1438,0.1026,0.0373,0.0000,0.0000],
        mAP = [0.3083,0.2532,0.2082,0.1587,0.1385,0.1121,0.0929,0.0759,0.0627,0.0524,0.0423,0.0085,0.0005,0.0000,0.0000],
    ),

    PGD = dict(
        NDS = [0.3525,0.3091,0.2748,0.2514,0.2329,0.2154,0.1999,0.1971,0.1819,0.1776,0.1759,0.1094,0.0399,0.0096,0.0000], # rebenchmark the last one
        mAP = [0.3344,0.2666,0.2203,0.1819,0.1504,0.1234,0.1017,0.0838,0.0722,0.0631,0.0502,0.0104,0.0021,0.0001,0.0000],
    ),

    BEVDepth_R50 = dict(
        NDS = [0.3759,0.2857,0.2459,0.2244,0.1945,0.1821,0.1544,0.1297,0.1434,0.1157,0.1063,0.0600,0.0276,0.0052,0.0000],
        mAP = [0.3248,0.2126,0.1655,0.1328,0.0992,0.0733,0.0680,0.0496,0.0415,0.0310,0.0275,0.0041,0.0003,0.0000,0.0000], # rebenchmark this part
    ), 

    BEVDet_R50 = dict(
        NDS = [0.3289,0.2048,0.1757,0.1489,0.1309,0.1094,0.1060,0.0927,0.0917,0.0604,0.0080,0.0000,0.0000,0.0000,0.0000],
        mAP = [0.2831,0.1551,0.1170,0.0728,0.0540,0.0405,0.0384,0.0199,0.0161,0.0064,0.0080,0.0000,0.0000,0.0000,0.0000],
    )
    
    )

# rebenchmark
pgd_attack_target = dict(
    metric = 'max_steps',
    max_steps = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50],
    BEVFormer_Tiny = dict(
        mAP = [0.2015,0.1603,0.1333,0.1176,0.0979,0.0920,0.0396,0.0168,0.0132,0.0107],
    ),
    BEVFormer_Tiny_Temp = dict(
        mAP = [0.2662,0.2177,0.1906,0.1685,0.1525,0.1328,0.0778,0.0449,0.0254,0.0180],
    ),
    BEVFormer_Small = dict(
        mAP = [0.1893,0.1570,0.1386,0.1241,0.1098,0.0954,0.0576,0.0410,0.0295,0.0250],
    ),
    BEVFormer_Small_Temp = dict(
        mAP = [0.3546,0.2645,0.2075,0.1943,0.1645,0.1397,0.1106,0.0578,0.0524,0.0516],
    ),
    BEVFormer_Base = dict(
        mAP = [0.3185,0.2511,0.2276,0.2029,0.1952,0.1837,0.1464,0.1286,0.0981,0.0829],
    ),
    BEVFormer_Base_Temp = dict(
        mAP = [0.3766,0.2871,0.2483,0.2149,0.1856,0.1788,0.1380,0.0863,0.0735,0.0654],
    ),
    DETR3D_CBGS = dict(
        mAP = [0.3219,0.2820,0.2509,0.2228,0.1942,0.1797,0.1095,0.0770,0.0628,0.0501], 
    ),
    DETR3D = dict(
        mAP = [0.3112,0.2915,0.2703,0.2441,0.2314,0.2172,0.1572,0.1159,0.0963,0.0754],
    ),
    FCOS3D = dict(
        mAP = [0.3083,0.2652,0.2404,0.2132,0.1897,0.1784,0.1270,0.1041,0.0861,0.0799],
    ),
    PGD = dict(
        mAP = [0.3344,0.2997,0.2681,0.2471,0.2220,0.2131,0.1599,0.1326,0.1044,0.0951],
    ),
    BEVDepth_R50 = dict(
        mAP = [0.3248,0.2681,0.2494,0.2374,0.2308,0.2064,0.1867,0.1436,0.1325,0.1063],
    ),
    BEVDet_R50 = dict(
        mAP = [0.2831,0.2007,0.1698,0.1482,0.1401,0.1315,0.0863,0.0552,0.0320,0.0200],
    )
)


pgd_attack_local = dict(
    metric = 'max_steps',
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
    FCOS3D = dict(
        NDS = [0.3309,0.2115,0.1642,0.1466,0.1353,0.0927,0.0779,0.0596,0.0538,0.0517],
        mAP = [0.3083,0.1647,0.1015,0.0671,0.0544,0.0379,0.0185,0.0122,0.0091,0.0064],
    ),
    PGD = dict(
        NDS = [0.3525,0.2475,0.1919,0.1717,0.1295,0.1144,0.0650,0.0468,0.0159,0.0125],
        mAP = [0.3344,0.1984,0.1244,0.0919,0.0715,0.0577,0.0320,0.0211,0.0159,0.0125],
    ),
    BEVDepth_R50 = dict(
        NDS = [0.3759,0.2671,0.2270,0.2072,0.1898,0.1770,0.1339,0.0975,0.0891,0.0751],
        mAP = [0.3248,0.2508,0.2125,0.1787,0.1533,0.1449,0.0892,0.0614,0.0434,0.0368],
    ),
    BEVDet_R50 = dict(
        NDS = [0.3289,0.2093,0.1765,0.1531,0.1466,0.1338,0.1163,0.0734,0.0593,0.0519],
        mAP = [0.2831,0.1897,0.1542,0.1246,0.1134,0.1018,0.0614,0.0400,0.0284,0.0253],
    )
    )

dynamic_patch_untarget_attack = dict(
    metric = 'scale',
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
        NDS = [0.2790,0.2248,0.1714,0.1238,0.0721],
        mAP = [0.1893,0.1142,0.0544,0.0187,0.0023],
    ),
    BEVFormer_Small_Temp = dict(
        NDS = [0.3990,0.2767,0.1902,0.1248,0.0708],
        mAP = [0.3546,0.1794,0.0848,0.0344,0.0078],
    ),
    BEVFormer_Base = dict(
        NDS = [0.3563,0.3068,0.2335,0.1555,0.1378],
        mAP = [0.3185,0.2175,0.1225,0.0529,0.0132],
    ),
    BEVFormer_Base_Temp = dict(
        NDS = [0.4254,0.2757,0.2170,0.1649,0.0730],
        mAP = [0.3766,0.2145,0.1244,0.0444,0.0031],
    ),
    DETR3D_CBGS = dict(
        NDS = [0.3899,0.3485,0.2880,0.2237,0.1503],
        mAP = [0.3219,0.2376,0.1551,0.0893,0.0218], 
    ),
    DETR3D = dict(
        NDS = [0.3743],
        mAP = [0.3112,0.2059,0.1034,0.0237,0.0007],
    ),
    FCOS3D = dict(
        NDS = [0.3309,0.2396,0.1396,0.1109,0.0000],
        mAP = [0.3083,0.1647,0.0458,0.0066,0.0000],
    ),
    PGD = dict(
        NDS = [0.3525,0.2612,0.1676,0.0130,0.0085],
        mAP = [0.3344,0.1898,0.0551,0.0000,0.0000], # rebenchmark PGD patch scale 0.3
    ),
    BEVDepth_R50 = dict(
        NDS = [0.3759,0.2868,0.1721,0.0710,0.0057],
        mAP = [0.3248,0.1753,0.0617,0.0043,0.0],
    ),
    BEVDet_R50 = dict(
        NDS = [0.3289,0.2308,0.1422,0.0580,0.0096],
        mAP = [0.2831,0.1285,0.0296,0.0005,0.0000],
    )

    )


dynamic_patch_loc_attack = dict(
    metric = 'scale',
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
        NDS = [0.3563, ],       # benchmark this part
        mAP = [0.3185],
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
        NDS = [0.3743],
        mAP = [0.3112,0.2705], # rebenchmark
    ),
    FCOS3D = dict(
        NDS = [0.3309,0.2110,0.1494,0.0985,0.0596],
        mAP = [0.3083,0.1710,0.0897,0.0388,0.0201],
    ),
    PGD = dict(
        NDS = [0.3525,0.2583,0.1707,0.1303,0.0891],
        mAP = [0.3344,0.2148,0.1246,0.0618,0.0182],
    ),
    BEVDepth_R50 = dict(
        NDS = [0.3759,0.2878,0.2039,0.1517,0.1128],
        mAP = [0.3248,0.2583,0.1532,0.0954,0.0393],
    ),
    BEVDet_R50 = dict(
        NDS = [0.3289,0.2419,0.1789,0.1219,0.0821],
        mAP = [0.2831,0.2180,0.1308,0.0614,0.0325],
    )
)

def collect_data(data, metric, catagory):

    results = []

    for i in range(len(data)):
        data_ = data[i]
        result = dict()
        result['patch_scale'] = data_['patch_scale']
        result['step_size'] = data_['step_size']
        result['max_steps'] = data_['max_steps']
        result[f'{catagory}_{metric}'] = 0
        num = 0
        for key in data_.keys():
            if metric in key and catagory in key:
                result[f'{catagory}_{metric}'] += data_[key]
                num = num + 1
        # calculate average ap under different threshold
        result[f'{catagory}_{metric}'] /= num
        results.append(result)

    return results


def plot_api(x, y, xtitle, ytitle, out_path):
    
    plt.plot(x, y, 'o-', color='steelblue', markersize=2)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(out_path)
    plt.cla()


def visualize_object(data):
    """Visualize metric of different object according to experiment settings (e.x. patch_size, step_size, etc.)
    Args:
        data (list): each element is a dict output by the model
    """
    for class_ in CLASSES:
        for metric_ in CLASSES_METRIC:
            if metric_ == 'AP':
                result = collect_data(data, metric_, class_)
                result.sort(key=lambda k: k['patch_scale'])
                x = [exp['patch_scale'] for exp in result]
                y = [exp[f'{class_}_{metric_}'] for exp in result]
                plot_api(x, y, 'patch_scale', f'{class_}_{metric_}', out_path=os.path.join('visual', 'patch_scale', f'ap_{class_}.pdf'))
            

def multi_plot_api(xs, ys, labels, xtitle, ytitle, out_path):
    assert isinstance(xs, list or tuple)
    assert isinstance(ys, list or tuple)
    assert isinstance(labels, list or tuple)
    assert len(xs) == len(ys) and len(xs) == len(labels)

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i in range(len(xs)):
        label_ = labels[i].replace('_', '-')
        assert len(xs[i]) == len(ys[i]), f"x doesn't match y in {label_}"
        ax.plot(xs[i], ys[i], f'{MARKER[labels[i]]}-', c=COLORS[labels[i]], label=label_, markersize=8)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend()
    plt.savefig(out_path)
    plt.cla()


def plot_scatter_api(xs, ys, labels, xtitle, ytitle, parameters, out_path):
    assert isinstance(xs, list or tuple)
    assert isinstance(ys, list or tuple)
    assert isinstance(labels, list or tuple)
    assert len(xs) == len(ys) and len(xs) == len(labels)


    fig, ax = plt.subplots()
    for i in range(len(xs)):
        label_ = labels[i].replace('_', '-')
        ax.scatter(x=[xs[i]], y=[ys[i]], marker=MARKER[labels[i]], c=COLORS[labels[i]], alpha=0.8, s=200, label=label_)
        
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(markerscale=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(out_path)
    plt.cla()


def parse_data(results, relative=False):
    """
    Args:
        results (dict)
        relative (bool): visualize relative performance drop
    """
    assert isinstance(results, dict)
    # keys = list(results.keys())
    keys = MODELS
    xs = []
    ys = []
    labels = []
    for key in keys:
        if key == 'metric' or key == results['metric']:
            continue
        x = results[results['metric']]
        y = results[key]
        if relative:
            y = [y[i] / y[0] for i in range(len(y))]
        xs.append(x)
        ys.append(y)
        labels.append(key)
    
    return xs, ys, labels


def collect_clean_accuracy(results_dict, val=False):
    assert isinstance(results_dict, dict)

    # models = list(results_dict.keys())
    models = MODELS
    acc_all = []
    model_names = []
    for model in models:
        if model == 'metric' or model == results_dict['metric']:
            continue
        if not val:
            # map on mini datasets
            acc = results_dict[model][0]
        else:
            # map on validation datasets
            acc = VAL_MAP[model]
        acc_all.append(acc)
        model_names.append(model)
    
    return acc_all, model_names


def collect_model_size():
    models = MODELS
    size_all = []
    for i, model in enumerate(models):
        size_all.append(PARAMETERS[model])

    return size_all, models


def collect_robustness_acc(results_dicts, param=False, val=False, range=-1):
    """Calculate average adversarial robustness v.s. clean accuracy
    """
    assert isinstance(results_dicts, List)
    assert isinstance(results_dicts[0], dict)

    if param:
        clean_accs, model_names = collect_model_size()
    else:
        clean_accs, model_names = collect_clean_accuracy(results_dicts[0])

    if val:
        clean_accs = [VAL_MAP[model] for model in model_names]

    adver_accs = []
    for model_name in model_names:
        adver_acc = []
        for results_dict in results_dicts:
            adver_acc.extend(results_dict[model_name][1: range])
        adver_acc = np.mean(np.array(adver_acc))
        adver_accs.append(adver_acc)

    return clean_accs, adver_accs, model_names


if __name__ == '__main__':
    xs, ys, labels = parse_data(dynamic_patch_untarget_attack, relative=False)
    multi_plot_api(xs, ys, labels, 'attack iterations', 'mAP', 'visual/test.png')

    # pgd_attack_untarget, pgd_attack_target, pgd_attack_local, dynamic_patch_untarget_attack
    # clean_accs, adver_accs, model_names = collect_robustness_acc([pgd_attack_local], param=False, val=False, range=-1)
    # for i in range(len(model_names)):
    #     print(f'{model_names[i]}  clean_acc: {clean_accs[i]}  adver_accs: {adver_accs[i]}')
    # plot_scatter_api(clean_accs, adver_accs, model_names, 'mAP', 'Adversarial mAP', PARAMETERS, 'visual/test.png') # visual/test.png