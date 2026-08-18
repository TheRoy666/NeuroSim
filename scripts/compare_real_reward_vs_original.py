import re
import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

REAL_REWARD_TEXT = """
[  1/238] 100610 Sev=0 (Social_Drinker) E_crav→rest=0.2469  teleport=3.55x
[  2/238] 102311 Sev=0 (Social_Drinker) E_crav→rest=0.3098  teleport=3.16x
[  3/238] 103818 Sev=0 (Social_Drinker) E_crav→rest=0.1623  teleport=5.15x
[  4/238] 104416 Sev=0 (Social_Drinker) E_crav→rest=0.2641  teleport=3.64x
[  5/238] 105115 Sev=0 (Social_Drinker) E_crav→rest=0.1045  teleport=2.97x
[  6/238] 106016 Sev=0 (Social_Drinker) E_crav→rest=0.4586  teleport=2.13x
[  7/238] 107018 Sev=0 (Social_Drinker) E_crav→rest=0.3729  teleport=2.60x
[  8/238] 107422 Sev=0 (Social_Drinker) E_crav→rest=0.1883  teleport=3.98x
[  9/238] 108323 Sev=0 (Social_Drinker) E_crav→rest=0.2132  teleport=4.45x
[ 10/238] 109123 Sev=2 (Dependent     ) E_crav→rest=0.2867  teleport=3.35x
[ 11/238] 111312 Sev=0 (Social_Drinker) E_crav→rest=0.2661  teleport=3.38x
[ 12/238] 111413 Sev=0 (Social_Drinker) E_crav→rest=0.1938  teleport=3.53x
[ 13/238] 111514 Sev=1 (Abuser        ) E_crav→rest=0.2217  teleport=4.19x
[ 14/238] 113922 Sev=0 (Social_Drinker) E_crav→rest=0.2864  teleport=2.85x
[ 15/238] 115017 Sev=0 (Social_Drinker) E_crav→rest=0.2091  teleport=4.06x
[ 16/238] 115320 Sev=0 (Social_Drinker) E_crav→rest=0.3193  teleport=3.03x
[ 17/238] 116726 Sev=0 (Social_Drinker) E_crav→rest=0.2901  teleport=2.61x
[ 18/238] 117021 Sev=0 (Social_Drinker) E_crav→rest=0.3084  teleport=3.01x
[ 19/238] 118023 Sev=2 (Dependent     ) E_crav→rest=0.1887  teleport=4.87x
[ 20/238] 118225 Sev=0 (Social_Drinker) E_crav→rest=0.2493  teleport=3.29x
[ 21/238] 118831 Sev=0 (Social_Drinker) E_crav→rest=0.2367  teleport=4.11x
[ 22/238] 120111 Sev=0 (Social_Drinker) E_crav→rest=0.0898  teleport=4.44x
[ 23/238] 122822 Sev=0 (Social_Drinker) E_crav→rest=0.4597  teleport=2.12x
[ 24/238] 124422 Sev=0 (Social_Drinker) E_crav→rest=0.1215  teleport=6.00x
[ 25/238] 124826 Sev=0 (Social_Drinker) E_crav→rest=0.3650  teleport=2.62x
[ 26/238] 126325 Sev=0 (Social_Drinker) E_crav→rest=0.2576  teleport=3.58x
[ 27/238] 126426 Sev=0 (Social_Drinker) E_crav→rest=0.4752  teleport=1.85x
[ 28/238] 127731 Sev=0 (Social_Drinker) E_crav→rest=0.1611  teleport=5.20x
[ 29/238] 127832 Sev=0 (Social_Drinker) E_crav→rest=0.2133  teleport=4.54x
[ 30/238] 128935 Sev=0 (Social_Drinker) E_crav→rest=0.2679  teleport=3.11x
[ 31/238] 129028 Sev=1 (Abuser        ) E_crav→rest=0.1786  teleport=4.43x
[ 32/238] 130518 Sev=0 (Social_Drinker) E_crav→rest=0.1340  teleport=4.89x
[ 33/238] 130619 Sev=0 (Social_Drinker) E_crav→rest=0.1797  teleport=5.03x
[ 34/238] 131217 Sev=0 (Social_Drinker) E_crav→rest=0.3267  teleport=2.89x
[ 35/238] 131722 Sev=2 (Dependent     ) E_crav→rest=0.3827  teleport=2.53x
[ 36/238] 131823 Sev=0 (Social_Drinker) E_crav→rest=0.1665  teleport=4.27x
[ 37/238] 133019 Sev=0 (Social_Drinker) E_crav→rest=0.1293  teleport=5.79x
[ 38/238] 133625 Sev=1 (Abuser        ) E_crav→rest=0.1821  teleport=3.35x
[ 39/238] 134728 Sev=0 (Social_Drinker) E_crav→rest=0.4069  teleport=2.39x
[ 40/238] 134829 Sev=0 (Social_Drinker) E_crav→rest=0.2159  teleport=3.91x
[ 41/238] 135124 Sev=0 (Social_Drinker) E_crav→rest=0.1425  teleport=4.45x
[ 42/238] 137128 Sev=0 (Social_Drinker) E_crav→rest=0.1811  teleport=4.14x
[ 43/238] 137431 Sev=0 (Social_Drinker) E_crav→rest=0.2082  teleport=3.99x
[ 44/238] 139435 Sev=0 (Social_Drinker) E_crav→rest=0.2272  teleport=3.77x
[ 45/238] 139637 Sev=0 (Social_Drinker) E_crav→rest=0.3127  teleport=3.14x
[ 46/238] 139839 Sev=0 (Social_Drinker) E_crav→rest=0.2846  teleport=3.37x
[ 47/238] 140117 Sev=1 (Abuser        ) E_crav→rest=0.2616  teleport=2.92x
[ 48/238] 141422 Sev=0 (Social_Drinker) E_crav→rest=0.2566  teleport=3.12x
[ 49/238] 143325 Sev=0 (Social_Drinker) E_crav→rest=0.3294  teleport=2.88x
[ 50/238] 144226 Sev=0 (Social_Drinker) E_crav→rest=0.1810  teleport=4.46x
[ 51/238] 144731 Sev=0 (Social_Drinker) E_crav→rest=0.1050  teleport=6.39x
[ 52/238] 144933 Sev=1 (Abuser        ) E_crav→rest=0.2694  teleport=3.40x
[ 53/238] 145632 Sev=2 (Dependent     ) E_crav→rest=0.2172  teleport=4.04x
[ 54/238] 145834 Sev=0 (Social_Drinker) E_crav→rest=0.2503  teleport=3.62x
[ 55/238] 146735 Sev=0 (Social_Drinker) E_crav→rest=0.4172  teleport=1.97x
[ 56/238] 148436 Sev=0 (Social_Drinker) E_crav→rest=0.2044  teleport=3.06x
[ 57/238] 148941 Sev=0 (Social_Drinker) E_crav→rest=0.2806  teleport=3.33x
[ 58/238] 149741 Sev=0 (Social_Drinker) E_crav→rest=0.3438  teleport=2.70x
[ 59/238] 151425 Sev=0 (Social_Drinker) E_crav→rest=0.2999  teleport=3.18x
[ 60/238] 151526 Sev=0 (Social_Drinker) E_crav→rest=0.3754  teleport=2.65x
[ 61/238] 151930 Sev=0 (Social_Drinker) E_crav→rest=0.2730  teleport=3.52x
[ 62/238] 153934 Sev=1 (Abuser        ) E_crav→rest=0.2353  teleport=3.55x
[ 63/238] 154431 Sev=0 (Social_Drinker) E_crav→rest=0.2999  teleport=2.74x
[ 64/238] 157336 Sev=0 (Social_Drinker) E_crav→rest=0.2772  teleport=3.38x
[ 65/238] 159138 Sev=0 (Social_Drinker) E_crav→rest=0.3759  teleport=2.51x
[ 66/238] 159239 Sev=0 (Social_Drinker) E_crav→rest=0.1964  teleport=3.33x
[ 67/238] 161327 Sev=0 (Social_Drinker) E_crav→rest=0.2591  teleport=2.74x
[ 68/238] 162026 Sev=0 (Social_Drinker) E_crav→rest=0.2618  teleport=3.30x
[ 69/238] 162935 Sev=0 (Social_Drinker) E_crav→rest=0.2171  teleport=3.03x
[ 70/238] 163129 Sev=0 (Social_Drinker) E_crav→rest=0.2884  teleport=2.83x
[ 71/238] 164131 Sev=0 (Social_Drinker) E_crav→rest=0.3208  teleport=2.96x
[ 72/238] 164636 Sev=0 (Social_Drinker) E_crav→rest=0.2460  teleport=3.20x
[ 73/238] 165638 Sev=0 (Social_Drinker) E_crav→rest=0.4073  teleport=2.40x
[ 74/238] 166438 Sev=1 (Abuser        ) E_crav→rest=0.2652  teleport=3.56x
[ 75/238] 168341 Sev=0 (Social_Drinker) E_crav→rest=0.2598  teleport=2.50x
[ 76/238] 168947 Sev=0 (Social_Drinker) E_crav→rest=0.2592  teleport=2.81x
[ 77/238] 169343 Sev=0 (Social_Drinker) E_crav→rest=0.1976  teleport=4.78x
[ 78/238] 171330 Sev=0 (Social_Drinker) E_crav→rest=0.1670  teleport=5.13x
[ 79/238] 171633 Sev=1 (Abuser        ) E_crav→rest=0.3937  teleport=1.99x
[ 80/238] 172029 Sev=0 (Social_Drinker) E_crav→rest=0.2916  teleport=2.96x
[ 81/238] 172130 Sev=2 (Dependent     ) E_crav→rest=0.1057  teleport=6.83x
[ 82/238] 172332 Sev=0 (Social_Drinker) E_crav→rest=0.3332  teleport=2.68x
[ 83/238] 172938 Sev=0 (Social_Drinker) E_crav→rest=0.3406  teleport=2.87x
[ 84/238] 173334 Sev=0 (Social_Drinker) E_crav→rest=0.1900  teleport=4.91x
[ 85/238] 174841 Sev=0 (Social_Drinker) E_crav→rest=0.3047  teleport=3.05x
[ 86/238] 175237 Sev=0 (Social_Drinker) E_crav→rest=0.4012  teleport=2.45x
[ 87/238] 175439 Sev=0 (Social_Drinker) E_crav→rest=0.4008  teleport=2.30x
[ 88/238] 175540 Sev=0 (Social_Drinker) E_crav→rest=0.2561  teleport=3.49x
[ 89/238] 176542 Sev=0 (Social_Drinker) E_crav→rest=0.1666  teleport=4.75x
[ 90/238] 177140 Sev=0 (Social_Drinker) E_crav→rest=0.1916  teleport=4.87x
[ 91/238] 177645 Sev=0 (Social_Drinker) E_crav→rest=0.2280  teleport=4.25x
[ 92/238] 178647 Sev=0 (Social_Drinker) E_crav→rest=0.3100  teleport=2.83x
[ 93/238] 181232 Sev=0 (Social_Drinker) E_crav→rest=0.2792  teleport=3.46x
[ 94/238] 182436 Sev=0 (Social_Drinker) E_crav→rest=0.3206  teleport=2.59x
[ 95/238] 182840 Sev=0 (Social_Drinker) E_crav→rest=0.3377  teleport=2.28x
[ 96/238] 185139 Sev=0 (Social_Drinker) E_crav→rest=0.2598  teleport=3.47x
[ 97/238] 185947 Sev=0 (Social_Drinker) E_crav→rest=0.2348  teleport=3.90x
[ 98/238] 186848 Sev=0 (Social_Drinker) E_crav→rest=0.3375  teleport=2.74x
[ 99/238] 187345 Sev=0 (Social_Drinker) E_crav→rest=0.1578  teleport=6.12x
[100/238] 187547 Sev=0 (Social_Drinker) E_crav→rest=0.1810  teleport=4.77x
[101/238] 189349 Sev=0 (Social_Drinker) E_crav→rest=0.1696  teleport=5.10x
[102/238] 191437 Sev=0 (Social_Drinker) E_crav→rest=0.2372  teleport=3.22x
[103/238] 191841 Sev=0 (Social_Drinker) E_crav→rest=0.2711  teleport=3.49x
[104/238] 192641 Sev=0 (Social_Drinker) E_crav→rest=0.3182  teleport=2.86x
[105/238] 193845 Sev=0 (Social_Drinker) E_crav→rest=0.2610  teleport=3.62x
[106/238] 194140 Sev=0 (Social_Drinker) E_crav→rest=0.3026  teleport=2.94x
[107/238] 194443 Sev=0 (Social_Drinker) E_crav→rest=0.2052  teleport=4.67x
[108/238] 195041 Sev=0 (Social_Drinker) E_crav→rest=0.4024  teleport=1.99x
[109/238] 195445 Sev=1 (Abuser        ) E_crav→rest=0.2009  teleport=4.43x
[110/238] 196750 Sev=1 (Abuser        ) E_crav→rest=0.2663  teleport=3.57x
[111/238] 197348 Sev=0 (Social_Drinker) E_crav→rest=0.3351  teleport=2.61x
[112/238] 198047 Sev=0 (Social_Drinker) E_crav→rest=0.3270  teleport=2.79x
[113/238] 198249 Sev=0 (Social_Drinker) E_crav→rest=0.3390  teleport=2.68x
[114/238] 198653 Sev=0 (Social_Drinker) E_crav→rest=0.1260  teleport=6.19x
[115/238] 200008 Sev=0 (Social_Drinker) E_crav→rest=0.2475  teleport=2.75x
[116/238] 200109 Sev=0 (Social_Drinker) E_crav→rest=0.2803  teleport=2.90x
[117/238] 200311 Sev=0 (Social_Drinker) E_crav→rest=0.2746  teleport=3.36x
[118/238] 203418 Sev=0 (Social_Drinker) E_crav→rest=0.2899  teleport=3.13x
[119/238] 204521 Sev=2 (Dependent     ) E_crav→rest=0.3103  teleport=2.61x
[120/238] 205119 Sev=0 (Social_Drinker) E_crav→rest=0.3221  teleport=2.62x
[121/238] 205220 Sev=1 (Abuser        ) E_crav→rest=0.1488  teleport=5.30x
[122/238] 205826 Sev=0 (Social_Drinker) E_crav→rest=0.1265  teleport=3.32x
[123/238] 207426 Sev=0 (Social_Drinker) E_crav→rest=0.4632  teleport=2.09x
[124/238] 212419 Sev=0 (Social_Drinker) E_crav→rest=0.3893  teleport=2.51x
[125/238] 212823 Sev=1 (Abuser        ) E_crav→rest=0.2325  teleport=3.81x
[126/238] 214524 Sev=2 (Dependent     ) E_crav→rest=0.1525  teleport=3.16x
[127/238] 214625 Sev=1 (Abuser        ) E_crav→rest=0.1713  teleport=4.20x
[128/238] 221319 Sev=0 (Social_Drinker) E_crav→rest=0.1724  teleport=4.92x
[129/238] 233326 Sev=1 (Abuser        ) E_crav→rest=0.3539  teleport=2.67x
[130/238] 246133 Sev=0 (Social_Drinker) E_crav→rest=0.2443  teleport=3.87x
[131/238] 249947 Sev=0 (Social_Drinker) E_crav→rest=0.2823  teleport=2.43x
[132/238] 257845 Sev=0 (Social_Drinker) E_crav→rest=0.2820  teleport=3.14x
[133/238] 263436 Sev=0 (Social_Drinker) E_crav→rest=0.2215  teleport=4.06x
[134/238] 280739 Sev=0 (Social_Drinker) E_crav→rest=0.2352  teleport=4.03x
[135/238] 283543 Sev=0 (Social_Drinker) E_crav→rest=0.1554  teleport=2.83x
[136/238] 297655 Sev=0 (Social_Drinker) E_crav→rest=0.5311  teleport=1.79x
[137/238] 299760 Sev=0 (Social_Drinker) E_crav→rest=0.2483  teleport=3.71x
[138/238] 300719 Sev=1 (Abuser        ) E_crav→rest=0.1512  teleport=5.95x
[139/238] 303624 Sev=0 (Social_Drinker) E_crav→rest=0.1482  teleport=3.98x
[140/238] 308129 Sev=0 (Social_Drinker) E_crav→rest=0.1885  teleport=4.72x
[141/238] 310621 Sev=0 (Social_Drinker) E_crav→rest=0.2469  teleport=3.00x
[142/238] 316835 Sev=1 (Abuser        ) E_crav→rest=0.2093  teleport=4.55x
[143/238] 318637 Sev=0 (Social_Drinker) E_crav→rest=0.1309  teleport=3.58x
[144/238] 320826 Sev=0 (Social_Drinker) E_crav→rest=0.2506  teleport=2.53x
[145/238] 321323 Sev=0 (Social_Drinker) E_crav→rest=0.1582  teleport=6.10x
[146/238] 322224 Sev=0 (Social_Drinker) E_crav→rest=0.4702  teleport=1.88x
[147/238] 325129 Sev=0 (Social_Drinker) E_crav→rest=0.3636  teleport=2.65x
[148/238] 341834 Sev=2 (Dependent     ) E_crav→rest=0.1914  teleport=4.66x
[149/238] 352132 Sev=1 (Abuser        ) E_crav→rest=0.1679  teleport=4.18x
[150/238] 352738 Sev=2 (Dependent     ) E_crav→rest=0.2529  teleport=3.09x
[151/238] 353740 Sev=0 (Social_Drinker) E_crav→rest=0.2323  teleport=3.96x
[152/238] 360030 Sev=0 (Social_Drinker) E_crav→rest=0.2901  teleport=3.23x
[153/238] 365343 Sev=0 (Social_Drinker) E_crav→rest=0.3085  teleport=2.47x
[154/238] 380036 Sev=0 (Social_Drinker) E_crav→rest=0.3927  teleport=2.49x
[155/238] 385450 Sev=2 (Dependent     ) E_crav→rest=0.4479  teleport=2.14x
[156/238] 389357 Sev=2 (Dependent     ) E_crav→rest=0.2597  teleport=3.61x
[157/238] 390645 Sev=0 (Social_Drinker) E_crav→rest=0.2129  teleport=4.02x
[158/238] 391748 Sev=0 (Social_Drinker) E_crav→rest=0.3321  teleport=2.93x
[159/238] 393247 Sev=0 (Social_Drinker) E_crav→rest=0.2002  teleport=3.90x
[160/238] 393550 Sev=0 (Social_Drinker) E_crav→rest=0.3343  teleport=2.62x
[161/238] 397760 Sev=1 (Abuser        ) E_crav→rest=0.3317  teleport=2.54x
[162/238] 397861 Sev=0 (Social_Drinker) E_crav→rest=0.2148  teleport=2.90x
[163/238] 406836 Sev=0 (Social_Drinker) E_crav→rest=0.3461  teleport=2.25x
[164/238] 412528 Sev=0 (Social_Drinker) E_crav→rest=0.3544  teleport=2.81x
[165/238] 413934 Sev=0 (Social_Drinker) E_crav→rest=0.2171  teleport=4.26x
[166/238] 414229 Sev=0 (Social_Drinker) E_crav→rest=0.1593  teleport=5.79x
[167/238] 429040 Sev=1 (Abuser        ) E_crav→rest=0.2049  teleport=3.67x
[168/238] 433839 Sev=0 (Social_Drinker) E_crav→rest=0.2676  teleport=3.24x
[169/238] 453441 Sev=0 (Social_Drinker) E_crav→rest=0.1740  teleport=4.31x
[170/238] 461743 Sev=2 (Dependent     ) E_crav→rest=0.3357  teleport=2.91x
[171/238] 463040 Sev=0 (Social_Drinker) E_crav→rest=0.1053  teleport=3.93x
[172/238] 467351 Sev=0 (Social_Drinker) E_crav→rest=0.1738  teleport=5.05x
[173/238] 500222 Sev=2 (Dependent     ) E_crav→rest=0.1869  teleport=4.79x
[174/238] 506234 Sev=1 (Abuser        ) E_crav→rest=0.1972  teleport=4.73x
[175/238] 512835 Sev=0 (Social_Drinker) E_crav→rest=0.4043  teleport=2.24x
[176/238] 519647 Sev=2 (Dependent     ) E_crav→rest=0.4532  teleport=2.12x
[177/238] 525541 Sev=1 (Abuser        ) E_crav→rest=0.2593  teleport=3.68x
[178/238] 541943 Sev=0 (Social_Drinker) E_crav→rest=0.2413  teleport=3.85x
[179/238] 548250 Sev=0 (Social_Drinker) E_crav→rest=0.3068  teleport=2.83x
[180/238] 558657 Sev=0 (Social_Drinker) E_crav→rest=0.2862  teleport=3.19x
[181/238] 559053 Sev=0 (Social_Drinker) E_crav→rest=0.3313  teleport=2.80x
[182/238] 561949 Sev=1 (Abuser        ) E_crav→rest=0.2009  teleport=4.21x
[183/238] 562345 Sev=0 (Social_Drinker) E_crav→rest=0.1458  teleport=6.37x
[184/238] 562446 Sev=0 (Social_Drinker) E_crav→rest=0.2254  teleport=3.00x
[185/238] 568963 Sev=0 (Social_Drinker) E_crav→rest=0.2055  teleport=3.82x
[186/238] 572045 Sev=0 (Social_Drinker) E_crav→rest=0.2152  teleport=3.10x
[187/238] 573451 Sev=0 (Social_Drinker) E_crav→rest=0.2180  teleport=4.40x
[188/238] 581450 Sev=0 (Social_Drinker) E_crav→rest=0.1507  teleport=6.33x
[189/238] 585256 Sev=0 (Social_Drinker) E_crav→rest=0.0875  teleport=4.45x
[190/238] 586460 Sev=0 (Social_Drinker) E_crav→rest=0.2131  teleport=4.28x
[191/238] 617748 Sev=0 (Social_Drinker) E_crav→rest=0.2348  teleport=3.92x
[192/238] 627549 Sev=0 (Social_Drinker) E_crav→rest=0.2118  teleport=4.36x
[193/238] 638049 Sev=0 (Social_Drinker) E_crav→rest=0.2843  teleport=2.90x
[194/238] 656657 Sev=0 (Social_Drinker) E_crav→rest=0.1988  teleport=4.87x
[195/238] 664757 Sev=1 (Abuser        ) E_crav→rest=0.1293  teleport=4.54x
[196/238] 677968 Sev=0 (Social_Drinker) E_crav→rest=0.3026  teleport=3.05x
[197/238] 680452 Sev=0 (Social_Drinker) E_crav→rest=0.3011  teleport=2.96x
[198/238] 686969 Sev=0 (Social_Drinker) E_crav→rest=0.2390  teleport=3.94x
[199/238] 690152 Sev=1 (Abuser        ) E_crav→rest=0.2086  teleport=3.72x
[200/238] 698168 Sev=0 (Social_Drinker) E_crav→rest=0.2721  teleport=3.61x
[201/238] 707749 Sev=0 (Social_Drinker) E_crav→rest=0.2334  teleport=4.15x
[202/238] 709551 Sev=0 (Social_Drinker) E_crav→rest=0.3025  teleport=3.13x
[203/238] 715647 Sev=0 (Social_Drinker) E_crav→rest=0.3076  teleport=3.04x
[204/238] 720337 Sev=0 (Social_Drinker) E_crav→rest=0.2195  teleport=4.30x
[205/238] 724446 Sev=0 (Social_Drinker) E_crav→rest=0.2361  teleport=3.32x
[206/238] 725751 Sev=0 (Social_Drinker) E_crav→rest=0.3305  teleport=2.94x
[207/238] 731140 Sev=0 (Social_Drinker) E_crav→rest=0.2689  teleport=3.55x
[208/238] 732243 Sev=0 (Social_Drinker) E_crav→rest=0.1435  teleport=4.94x
[209/238] 735148 Sev=0 (Social_Drinker) E_crav→rest=0.1601  teleport=4.50x
[210/238] 744553 Sev=1 (Abuser        ) E_crav→rest=0.2011  teleport=4.59x
[211/238] 749058 Sev=0 (Social_Drinker) E_crav→rest=0.2434  teleport=3.85x
[212/238] 757764 Sev=0 (Social_Drinker) E_crav→rest=0.2606  teleport=3.72x
[213/238] 763557 Sev=1 (Abuser        ) E_crav→rest=0.2340  teleport=3.63x
[214/238] 767464 Sev=2 (Dependent     ) E_crav→rest=0.3290  teleport=2.73x
[215/238] 782561 Sev=0 (Social_Drinker) E_crav→rest=0.1838  teleport=3.60x
[216/238] 784565 Sev=0 (Social_Drinker) E_crav→rest=0.0917  teleport=9.47x
[217/238] 788674 Sev=0 (Social_Drinker) E_crav→rest=0.3718  teleport=2.46x
[218/238] 789373 Sev=0 (Social_Drinker) E_crav→rest=0.2221  teleport=4.18x
[219/238] 792564 Sev=0 (Social_Drinker) E_crav→rest=0.3761  teleport=2.59x
[220/238] 814649 Sev=0 (Social_Drinker) E_crav→rest=0.4651  teleport=1.97x
[221/238] 820745 Sev=0 (Social_Drinker) E_crav→rest=0.2447  teleport=3.06x
[222/238] 825048 Sev=0 (Social_Drinker) E_crav→rest=0.2915  teleport=2.77x
[223/238] 832651 Sev=2 (Dependent     ) E_crav→rest=0.1896  teleport=4.90x
[224/238] 852455 Sev=0 (Social_Drinker) E_crav→rest=0.1667  teleport=3.48x
[225/238] 859671 Sev=0 (Social_Drinker) E_crav→rest=0.2068  teleport=4.47x
[226/238] 871762 Sev=0 (Social_Drinker) E_crav→rest=0.2577  teleport=3.08x
[227/238] 878776 Sev=0 (Social_Drinker) E_crav→rest=0.5131  teleport=1.93x
[228/238] 884064 Sev=0 (Social_Drinker) E_crav→rest=0.1849  teleport=3.62x
[229/238] 885975 Sev=0 (Social_Drinker) E_crav→rest=0.1843  teleport=4.39x
[230/238] 905147 Sev=0 (Social_Drinker) E_crav→rest=0.1925  teleport=3.10x
[231/238] 910241 Sev=0 (Social_Drinker) E_crav→rest=0.3386  teleport=2.89x
[232/238] 911849 Sev=0 (Social_Drinker) E_crav→rest=0.1078  teleport=5.85x
[233/238] 912447 Sev=0 (Social_Drinker) E_crav→rest=0.3376  teleport=2.75x
[234/238] 926862 Sev=0 (Social_Drinker) E_crav→rest=0.2153  teleport=4.50x
[235/238] 927359 Sev=0 (Social_Drinker) E_crav→rest=0.2782  teleport=3.31x
[236/238] 951457 Sev=0 (Social_Drinker) E_crav→rest=0.1482  teleport=5.64x
[237/238] 958976 Sev=0 (Social_Drinker) E_crav→rest=0.2901  teleport=2.72x
[238/238] 984472 Sev=0 (Social_Drinker) E_crav→rest=0.2680  teleport=3.67x
"""

ORIGINAL_TEXT = """
[  1/238] 100610 Sev=0 (Social_Drinker) E_crav→rest=0.3021  teleport=2.99x
[  2/238] 102311 Sev=0 (Social_Drinker) E_crav→rest=0.3081  teleport=3.26x
[  3/238] 103818 Sev=0 (Social_Drinker) E_crav→rest=0.1986  teleport=4.25x
[  4/238] 104416 Sev=0 (Social_Drinker) E_crav→rest=0.2960  teleport=3.28x
[  5/238] 105115 Sev=0 (Social_Drinker) E_crav→rest=0.1739  teleport=2.98x
[  6/238] 106016 Sev=0 (Social_Drinker) E_crav→rest=0.4805  teleport=2.02x
[  7/238] 107018 Sev=0 (Social_Drinker) E_crav→rest=0.3855  teleport=2.51x
[  8/238] 107422 Sev=0 (Social_Drinker) E_crav→rest=0.1989  teleport=4.19x
[  9/238] 108323 Sev=0 (Social_Drinker) E_crav→rest=0.2377  teleport=4.03x
[ 10/238] 109123 Sev=2 (Dependent     ) E_crav→rest=0.3032  teleport=3.20x
[ 11/238] 111312 Sev=0 (Social_Drinker) E_crav→rest=0.2849  teleport=3.14x
[ 12/238] 111413 Sev=0 (Social_Drinker) E_crav→rest=0.2148  teleport=3.71x
[ 13/238] 111514 Sev=1 (Abuser        ) E_crav→rest=0.2526  teleport=3.79x
[ 14/238] 113922 Sev=0 (Social_Drinker) E_crav→rest=0.2942  teleport=2.98x
[ 15/238] 115017 Sev=0 (Social_Drinker) E_crav→rest=0.2358  teleport=3.65x
[ 16/238] 115320 Sev=0 (Social_Drinker) E_crav→rest=0.3203  teleport=2.97x
[ 17/238] 116726 Sev=0 (Social_Drinker) E_crav→rest=0.3229  teleport=2.53x
[ 18/238] 117021 Sev=0 (Social_Drinker) E_crav→rest=0.3468  teleport=2.67x
[ 19/238] 118023 Sev=2 (Dependent     ) E_crav→rest=0.1993  teleport=4.36x
[ 20/238] 118225 Sev=0 (Social_Drinker) E_crav→rest=0.3013  teleport=2.95x
[ 21/238] 118831 Sev=0 (Social_Drinker) E_crav→rest=0.2380  teleport=3.97x
[ 22/238] 120111 Sev=0 (Social_Drinker) E_crav→rest=0.1061  teleport=5.44x
[ 23/238] 122822 Sev=0 (Social_Drinker) E_crav→rest=0.4605  teleport=2.17x
[ 24/238] 124422 Sev=0 (Social_Drinker) E_crav→rest=0.1506  teleport=5.58x
[ 25/238] 124826 Sev=0 (Social_Drinker) E_crav→rest=0.3639  teleport=2.66x
[ 26/238] 126325 Sev=0 (Social_Drinker) E_crav→rest=0.2966  teleport=3.03x
[ 27/238] 126426 Sev=0 (Social_Drinker) E_crav→rest=0.4808  teleport=1.85x
[ 28/238] 127731 Sev=0 (Social_Drinker) E_crav→rest=0.2015  teleport=4.17x
[ 29/238] 127832 Sev=0 (Social_Drinker) E_crav→rest=0.2253  teleport=4.41x
[ 30/238] 128935 Sev=0 (Social_Drinker) E_crav→rest=0.2885  teleport=2.94x
[ 31/238] 129028 Sev=1 (Abuser        ) E_crav→rest=0.2349  teleport=3.56x
[ 32/238] 130518 Sev=0 (Social_Drinker) E_crav→rest=0.1797  teleport=4.33x
[ 33/238] 130619 Sev=0 (Social_Drinker) E_crav→rest=0.2111  teleport=4.27x
[ 34/238] 131217 Sev=0 (Social_Drinker) E_crav→rest=0.3399  teleport=2.87x
[ 35/238] 131722 Sev=2 (Dependent     ) E_crav→rest=0.4082  teleport=2.38x
[ 36/238] 131823 Sev=0 (Social_Drinker) E_crav→rest=0.1679  teleport=4.23x
[ 37/238] 133019 Sev=0 (Social_Drinker) E_crav→rest=0.1348  teleport=6.06x
[ 38/238] 133625 Sev=1 (Abuser        ) E_crav→rest=0.2328  teleport=3.23x
[ 39/238] 134728 Sev=0 (Social_Drinker) E_crav→rest=0.4185  teleport=2.28x
[ 40/238] 134829 Sev=0 (Social_Drinker) E_crav→rest=0.2187  teleport=4.29x
[ 41/238] 135124 Sev=0 (Social_Drinker) E_crav→rest=0.1643  teleport=4.28x
[ 42/238] 137128 Sev=0 (Social_Drinker) E_crav→rest=0.2059  teleport=4.06x
[ 43/238] 137431 Sev=0 (Social_Drinker) E_crav→rest=0.2699  teleport=3.30x
[ 44/238] 139435 Sev=0 (Social_Drinker) E_crav→rest=0.2632  teleport=3.32x
[ 45/238] 139637 Sev=0 (Social_Drinker) E_crav→rest=0.3131  teleport=3.22x
[ 46/238] 139839 Sev=0 (Social_Drinker) E_crav→rest=0.2988  teleport=3.14x
[ 47/238] 140117 Sev=1 (Abuser        ) E_crav→rest=0.2625  teleport=3.33x
[ 48/238] 141422 Sev=0 (Social_Drinker) E_crav→rest=0.3033  teleport=2.83x
[ 49/238] 143325 Sev=0 (Social_Drinker) E_crav→rest=0.3329  teleport=2.97x
[ 50/238] 144226 Sev=0 (Social_Drinker) E_crav→rest=0.2162  teleport=3.79x
[ 51/238] 144731 Sev=0 (Social_Drinker) E_crav→rest=0.1116  teleport=6.07x
[ 52/238] 144933 Sev=1 (Abuser        ) E_crav→rest=0.3043  teleport=3.11x
[ 53/238] 145632 Sev=2 (Dependent     ) E_crav→rest=0.2407  teleport=3.78x
[ 54/238] 145834 Sev=0 (Social_Drinker) E_crav→rest=0.2584  teleport=3.67x
[ 55/238] 146735 Sev=0 (Social_Drinker) E_crav→rest=0.4365  teleport=2.00x
[ 56/238] 148436 Sev=0 (Social_Drinker) E_crav→rest=0.2359  teleport=3.05x
[ 57/238] 148941 Sev=0 (Social_Drinker) E_crav→rest=0.3221  teleport=2.87x
[ 58/238] 149741 Sev=0 (Social_Drinker) E_crav→rest=0.3875  teleport=2.37x
[ 59/238] 151425 Sev=0 (Social_Drinker) E_crav→rest=0.3082  teleport=3.04x
[ 60/238] 151526 Sev=0 (Social_Drinker) E_crav→rest=0.3857  teleport=2.51x
[ 61/238] 151930 Sev=0 (Social_Drinker) E_crav→rest=0.2967  teleport=3.28x
[ 62/238] 153934 Sev=1 (Abuser        ) E_crav→rest=0.2508  teleport=3.23x
[ 63/238] 154431 Sev=0 (Social_Drinker) E_crav→rest=0.3122  teleport=2.76x
[ 64/238] 157336 Sev=0 (Social_Drinker) E_crav→rest=0.2930  teleport=3.22x
[ 65/238] 159138 Sev=0 (Social_Drinker) E_crav→rest=0.3985  teleport=2.40x
[ 66/238] 159239 Sev=0 (Social_Drinker) E_crav→rest=0.2411  teleport=3.08x
[ 67/238] 161327 Sev=0 (Social_Drinker) E_crav→rest=0.2494  teleport=2.81x
[ 68/238] 162026 Sev=0 (Social_Drinker) E_crav→rest=0.2948  teleport=3.06x
[ 69/238] 162935 Sev=0 (Social_Drinker) E_crav→rest=0.2379  teleport=3.17x
[ 70/238] 163129 Sev=0 (Social_Drinker) E_crav→rest=0.2949  teleport=2.75x
[ 71/238] 164131 Sev=0 (Social_Drinker) E_crav→rest=0.3387  teleport=2.80x
[ 72/238] 164636 Sev=0 (Social_Drinker) E_crav→rest=0.2706  teleport=3.23x
[ 73/238] 165638 Sev=0 (Social_Drinker) E_crav→rest=0.4343  teleport=2.25x
[ 74/238] 166438 Sev=1 (Abuser        ) E_crav→rest=0.3021  teleport=3.10x
[ 75/238] 168341 Sev=0 (Social_Drinker) E_crav→rest=0.2662  teleport=3.02x
[ 76/238] 168947 Sev=0 (Social_Drinker) E_crav→rest=0.2979  teleport=2.68x
[ 77/238] 169343 Sev=0 (Social_Drinker) E_crav→rest=0.2130  teleport=4.51x
[ 78/238] 171330 Sev=0 (Social_Drinker) E_crav→rest=0.2339  teleport=3.67x
[ 79/238] 171633 Sev=1 (Abuser        ) E_crav→rest=0.4473  teleport=1.96x
[ 80/238] 172029 Sev=0 (Social_Drinker) E_crav→rest=0.3394  teleport=2.59x
[ 81/238] 172130 Sev=2 (Dependent     ) E_crav→rest=0.1188  teleport=6.29x
[ 82/238] 172332 Sev=0 (Social_Drinker) E_crav→rest=0.3597  teleport=2.47x
[ 83/238] 172938 Sev=0 (Social_Drinker) E_crav→rest=0.3490  teleport=2.73x
[ 84/238] 173334 Sev=0 (Social_Drinker) E_crav→rest=0.2218  teleport=4.21x
[ 85/238] 174841 Sev=0 (Social_Drinker) E_crav→rest=0.3292  teleport=2.84x
[ 86/238] 175237 Sev=0 (Social_Drinker) E_crav→rest=0.4215  teleport=2.36x
[ 87/238] 175439 Sev=0 (Social_Drinker) E_crav→rest=0.4067  teleport=2.25x
[ 88/238] 175540 Sev=0 (Social_Drinker) E_crav→rest=0.2598  teleport=3.25x
[ 89/238] 176542 Sev=0 (Social_Drinker) E_crav→rest=0.1718  teleport=4.71x
[ 90/238] 177140 Sev=0 (Social_Drinker) E_crav→rest=0.2197  teleport=4.35x
[ 91/238] 177645 Sev=0 (Social_Drinker) E_crav→rest=0.2376  teleport=4.06x
[ 92/238] 178647 Sev=0 (Social_Drinker) E_crav→rest=0.3402  teleport=2.58x
[ 93/238] 181232 Sev=0 (Social_Drinker) E_crav→rest=0.2885  teleport=3.34x
[ 94/238] 182436 Sev=0 (Social_Drinker) E_crav→rest=0.3447  teleport=2.43x
[ 95/238] 182840 Sev=0 (Social_Drinker) E_crav→rest=0.3650  teleport=2.18x
[ 96/238] 185139 Sev=0 (Social_Drinker) E_crav→rest=0.3073  teleport=2.93x
[ 97/238] 185947 Sev=0 (Social_Drinker) E_crav→rest=0.2374  teleport=3.85x
[ 98/238] 186848 Sev=0 (Social_Drinker) E_crav→rest=0.3638  teleport=2.54x
[ 99/238] 187345 Sev=0 (Social_Drinker) E_crav→rest=0.1631  teleport=5.93x
[100/238] 187547 Sev=0 (Social_Drinker) E_crav→rest=0.1951  teleport=4.58x
[101/238] 189349 Sev=0 (Social_Drinker) E_crav→rest=0.1921  teleport=4.84x
[102/238] 191437 Sev=0 (Social_Drinker) E_crav→rest=0.2631  teleport=3.19x
[103/238] 191841 Sev=0 (Social_Drinker) E_crav→rest=0.2702  teleport=3.55x
[104/238] 192641 Sev=0 (Social_Drinker) E_crav→rest=0.3460  teleport=2.68x
[105/238] 193845 Sev=0 (Social_Drinker) E_crav→rest=0.2795  teleport=3.40x
[106/238] 194140 Sev=0 (Social_Drinker) E_crav→rest=0.3216  teleport=2.70x
[107/238] 194443 Sev=0 (Social_Drinker) E_crav→rest=0.2401  teleport=3.98x
[108/238] 195041 Sev=0 (Social_Drinker) E_crav→rest=0.4464  teleport=1.90x
[109/238] 195445 Sev=1 (Abuser        ) E_crav→rest=0.2244  teleport=4.17x
[110/238] 196750 Sev=1 (Abuser        ) E_crav→rest=0.2799  teleport=3.41x
[111/238] 197348 Sev=0 (Social_Drinker) E_crav→rest=0.3678  teleport=2.44x
[112/238] 198047 Sev=0 (Social_Drinker) E_crav→rest=0.3476  teleport=2.68x
[113/238] 198249 Sev=0 (Social_Drinker) E_crav→rest=0.3534  teleport=2.60x
[114/238] 198653 Sev=0 (Social_Drinker) E_crav→rest=0.1485  teleport=5.72x
[115/238] 200008 Sev=0 (Social_Drinker) E_crav→rest=0.3002  teleport=2.52x
[116/238] 200109 Sev=0 (Social_Drinker) E_crav→rest=0.3015  teleport=2.64x
[117/238] 200311 Sev=0 (Social_Drinker) E_crav→rest=0.3076  teleport=2.99x
[118/238] 203418 Sev=0 (Social_Drinker) E_crav→rest=0.3337  teleport=2.74x
[119/238] 204521 Sev=2 (Dependent     ) E_crav→rest=0.3612  teleport=2.39x
[120/238] 205119 Sev=0 (Social_Drinker) E_crav→rest=0.3638  teleport=2.52x
[121/238] 205220 Sev=1 (Abuser        ) E_crav→rest=0.1542  teleport=5.32x
[122/238] 205826 Sev=0 (Social_Drinker) E_crav→rest=0.2148  teleport=2.78x
[123/238] 207426 Sev=0 (Social_Drinker) E_crav→rest=0.5000  teleport=1.93x
[124/238] 212419 Sev=0 (Social_Drinker) E_crav→rest=0.4041  teleport=2.42x
[125/238] 212823 Sev=1 (Abuser        ) E_crav→rest=0.2660  teleport=3.22x
[126/238] 214524 Sev=2 (Dependent     ) E_crav→rest=0.1756  teleport=3.61x
[127/238] 214625 Sev=1 (Abuser        ) E_crav→rest=0.1847  teleport=4.48x
[128/238] 221319 Sev=0 (Social_Drinker) E_crav→rest=0.1656  teleport=5.34x
[129/238] 233326 Sev=1 (Abuser        ) E_crav→rest=0.3571  teleport=2.61x
[130/238] 246133 Sev=0 (Social_Drinker) E_crav→rest=0.2491  teleport=3.80x
[131/238] 249947 Sev=0 (Social_Drinker) E_crav→rest=0.3037  teleport=2.67x
[132/238] 257845 Sev=0 (Social_Drinker) E_crav→rest=0.3096  teleport=2.92x
[133/238] 263436 Sev=0 (Social_Drinker) E_crav→rest=0.2218  teleport=4.29x
[134/238] 280739 Sev=0 (Social_Drinker) E_crav→rest=0.2339  teleport=4.01x
[135/238] 283543 Sev=0 (Social_Drinker) E_crav→rest=0.2158  teleport=2.93x
[136/238] 297655 Sev=0 (Social_Drinker) E_crav→rest=0.5375  teleport=1.74x
[137/238] 299760 Sev=0 (Social_Drinker) E_crav→rest=0.2721  teleport=3.39x
[138/238] 300719 Sev=1 (Abuser        ) E_crav→rest=0.1654  teleport=5.59x
[139/238] 303624 Sev=0 (Social_Drinker) E_crav→rest=0.1803  teleport=3.91x
[140/238] 308129 Sev=0 (Social_Drinker) E_crav→rest=0.2232  teleport=4.18x
[141/238] 310621 Sev=0 (Social_Drinker) E_crav→rest=0.2636  teleport=3.27x
[142/238] 316835 Sev=1 (Abuser        ) E_crav→rest=0.2351  teleport=4.05x
[143/238] 318637 Sev=0 (Social_Drinker) E_crav→rest=0.1779  teleport=3.62x
[144/238] 320826 Sev=0 (Social_Drinker) E_crav→rest=0.2834  teleport=2.66x
[145/238] 321323 Sev=0 (Social_Drinker) E_crav→rest=0.1807  teleport=5.18x
[146/238] 322224 Sev=0 (Social_Drinker) E_crav→rest=0.4791  teleport=1.83x
[147/238] 325129 Sev=0 (Social_Drinker) E_crav→rest=0.3722  teleport=2.60x
[148/238] 341834 Sev=2 (Dependent     ) E_crav→rest=0.2019  teleport=4.54x
[149/238] 352132 Sev=1 (Abuser        ) E_crav→rest=0.1691  teleport=4.00x
[150/238] 352738 Sev=2 (Dependent     ) E_crav→rest=0.2863  teleport=3.12x
[151/238] 353740 Sev=0 (Social_Drinker) E_crav→rest=0.2355  teleport=3.84x
[152/238] 360030 Sev=0 (Social_Drinker) E_crav→rest=0.3218  teleport=2.89x
[153/238] 365343 Sev=0 (Social_Drinker) E_crav→rest=0.3326  teleport=2.45x
[154/238] 380036 Sev=0 (Social_Drinker) E_crav→rest=0.4097  teleport=2.39x
[155/238] 385450 Sev=2 (Dependent     ) E_crav→rest=0.4646  teleport=2.07x
[156/238] 389357 Sev=2 (Dependent     ) E_crav→rest=0.2824  teleport=3.25x
[157/238] 390645 Sev=0 (Social_Drinker) E_crav→rest=0.2240  teleport=3.80x
[158/238] 391748 Sev=0 (Social_Drinker) E_crav→rest=0.3429  teleport=2.85x
[159/238] 393247 Sev=0 (Social_Drinker) E_crav→rest=0.2047  teleport=3.81x
[160/238] 393550 Sev=0 (Social_Drinker) E_crav→rest=0.3584  teleport=2.53x
[161/238] 397760 Sev=1 (Abuser        ) E_crav→rest=0.3668  teleport=2.45x
[162/238] 397861 Sev=0 (Social_Drinker) E_crav→rest=0.2433  teleport=3.04x
[163/238] 406836 Sev=0 (Social_Drinker) E_crav→rest=0.3736  teleport=2.35x
[164/238] 412528 Sev=0 (Social_Drinker) E_crav→rest=0.3810  teleport=2.60x
[165/238] 413934 Sev=0 (Social_Drinker) E_crav→rest=0.2391  teleport=3.81x
[166/238] 414229 Sev=0 (Social_Drinker) E_crav→rest=0.1631  teleport=5.93x
[167/238] 429040 Sev=1 (Abuser        ) E_crav→rest=0.2326  teleport=3.36x
[168/238] 433839 Sev=0 (Social_Drinker) E_crav→rest=0.3200  teleport=2.78x
[169/238] 453441 Sev=0 (Social_Drinker) E_crav→rest=0.2054  teleport=3.83x
[170/238] 461743 Sev=2 (Dependent     ) E_crav→rest=0.3437  teleport=2.86x
[171/238] 463040 Sev=0 (Social_Drinker) E_crav→rest=0.1239  teleport=4.90x
[172/238] 467351 Sev=0 (Social_Drinker) E_crav→rest=0.1911  teleport=4.76x
[173/238] 500222 Sev=2 (Dependent     ) E_crav→rest=0.2334  teleport=3.93x
[174/238] 506234 Sev=1 (Abuser        ) E_crav→rest=0.2062  teleport=4.53x
[175/238] 512835 Sev=0 (Social_Drinker) E_crav→rest=0.4105  teleport=2.32x
[176/238] 519647 Sev=2 (Dependent     ) E_crav→rest=0.4656  teleport=2.06x
[177/238] 525541 Sev=1 (Abuser        ) E_crav→rest=0.2645  teleport=3.48x
[178/238] 541943 Sev=0 (Social_Drinker) E_crav→rest=0.2781  teleport=3.36x
[179/238] 548250 Sev=0 (Social_Drinker) E_crav→rest=0.3766  teleport=2.39x
[180/238] 558657 Sev=0 (Social_Drinker) E_crav→rest=0.3159  teleport=3.02x
[181/238] 559053 Sev=0 (Social_Drinker) E_crav→rest=0.3422  teleport=2.73x
[182/238] 561949 Sev=1 (Abuser        ) E_crav→rest=0.2109  teleport=4.21x
[183/238] 562345 Sev=0 (Social_Drinker) E_crav→rest=0.1697  teleport=5.32x
[184/238] 562446 Sev=0 (Social_Drinker) E_crav→rest=0.2759  teleport=2.81x
[185/238] 568963 Sev=0 (Social_Drinker) E_crav→rest=0.2255  teleport=3.55x
[186/238] 572045 Sev=0 (Social_Drinker) E_crav→rest=0.2298  teleport=3.08x
[187/238] 573451 Sev=0 (Social_Drinker) E_crav→rest=0.2508  teleport=3.84x
[188/238] 581450 Sev=0 (Social_Drinker) E_crav→rest=0.1563  teleport=6.12x
[189/238] 585256 Sev=0 (Social_Drinker) E_crav→rest=0.1177  teleport=5.04x
[190/238] 586460 Sev=0 (Social_Drinker) E_crav→rest=0.2249  teleport=4.28x
[191/238] 617748 Sev=0 (Social_Drinker) E_crav→rest=0.2536  teleport=3.58x
[192/238] 627549 Sev=0 (Social_Drinker) E_crav→rest=0.2595  teleport=3.52x
[193/238] 638049 Sev=0 (Social_Drinker) E_crav→rest=0.3008  teleport=2.93x
[194/238] 656657 Sev=0 (Social_Drinker) E_crav→rest=0.2161  teleport=4.48x
[195/238] 664757 Sev=1 (Abuser        ) E_crav→rest=0.1698  teleport=4.12x
[196/238] 677968 Sev=0 (Social_Drinker) E_crav→rest=0.2975  teleport=3.11x
[197/238] 680452 Sev=0 (Social_Drinker) E_crav→rest=0.2906  teleport=2.98x
[198/238] 686969 Sev=0 (Social_Drinker) E_crav→rest=0.2550  teleport=3.70x
[199/238] 690152 Sev=1 (Abuser        ) E_crav→rest=0.1852  teleport=4.72x
[200/238] 698168 Sev=0 (Social_Drinker) E_crav→rest=0.2893  teleport=3.37x
[201/238] 707749 Sev=0 (Social_Drinker) E_crav→rest=0.2799  teleport=3.48x
[202/238] 709551 Sev=0 (Social_Drinker) E_crav→rest=0.3299  teleport=2.85x
[203/238] 715647 Sev=0 (Social_Drinker) E_crav→rest=0.3190  teleport=2.93x
[204/238] 720337 Sev=0 (Social_Drinker) E_crav→rest=0.2524  teleport=3.77x
[205/238] 724446 Sev=0 (Social_Drinker) E_crav→rest=0.2813  teleport=2.81x
[206/238] 725751 Sev=0 (Social_Drinker) E_crav→rest=0.3534  teleport=2.82x
[207/238] 731140 Sev=0 (Social_Drinker) E_crav→rest=0.2815  teleport=3.48x
[208/238] 732243 Sev=0 (Social_Drinker) E_crav→rest=0.1726  teleport=4.69x
[209/238] 735148 Sev=0 (Social_Drinker) E_crav→rest=0.1949  teleport=3.78x
[210/238] 744553 Sev=1 (Abuser        ) E_crav→rest=0.2105  teleport=4.55x
[211/238] 749058 Sev=0 (Social_Drinker) E_crav→rest=0.2745  teleport=3.47x
[212/238] 757764 Sev=0 (Social_Drinker) E_crav→rest=0.3002  teleport=3.23x
[213/238] 763557 Sev=1 (Abuser        ) E_crav→rest=0.2401  teleport=3.76x
[214/238] 767464 Sev=2 (Dependent     ) E_crav→rest=0.3679  teleport=2.52x
[215/238] 782561 Sev=0 (Social_Drinker) E_crav→rest=0.2039  teleport=3.89x
[216/238] 784565 Sev=0 (Social_Drinker) E_crav→rest=0.1019  teleport=8.66x
[217/238] 788674 Sev=0 (Social_Drinker) E_crav→rest=0.4037  teleport=2.29x
[218/238] 789373 Sev=0 (Social_Drinker) E_crav→rest=0.2504  teleport=3.64x
[219/238] 792564 Sev=0 (Social_Drinker) E_crav→rest=0.3713  teleport=2.53x
[220/238] 814649 Sev=0 (Social_Drinker) E_crav→rest=0.4592  teleport=1.91x
[221/238] 820745 Sev=0 (Social_Drinker) E_crav→rest=0.2624  teleport=2.57x
[222/238] 825048 Sev=0 (Social_Drinker) E_crav→rest=0.3205  teleport=2.77x
[223/238] 832651 Sev=2 (Dependent     ) E_crav→rest=0.1913  teleport=4.83x
[224/238] 852455 Sev=0 (Social_Drinker) E_crav→rest=0.1862  teleport=3.91x
[225/238] 859671 Sev=0 (Social_Drinker) E_crav→rest=0.2391  teleport=3.86x
[226/238] 871762 Sev=0 (Social_Drinker) E_crav→rest=0.2825  teleport=2.75x
[227/238] 878776 Sev=0 (Social_Drinker) E_crav→rest=0.5095  teleport=1.90x
[228/238] 884064 Sev=0 (Social_Drinker) E_crav→rest=0.1963  teleport=3.92x
[229/238] 885975 Sev=0 (Social_Drinker) E_crav→rest=0.1777  teleport=4.56x
[230/238] 905147 Sev=0 (Social_Drinker) E_crav→rest=0.2334  teleport=3.20x
[231/238] 910241 Sev=0 (Social_Drinker) E_crav→rest=0.3511  teleport=2.79x
[232/238] 911849 Sev=0 (Social_Drinker) E_crav→rest=0.1647  teleport=4.31x
[233/238] 912447 Sev=0 (Social_Drinker) E_crav→rest=0.3635  teleport=2.59x
[234/238] 926862 Sev=0 (Social_Drinker) E_crav→rest=0.2419  teleport=4.11x
[235/238] 927359 Sev=0 (Social_Drinker) E_crav→rest=0.2991  teleport=2.95x
[236/238] 951457 Sev=0 (Social_Drinker) E_crav→rest=0.1688  teleport=5.14x
[237/238] 958976 Sev=0 (Social_Drinker) E_crav→rest=0.3193  teleport=2.57x
[238/238] 984472 Sev=0 (Social_Drinker) E_crav→rest=0.2889  teleport=3.33x
"""


def parse(text):
    pattern = re.compile(
        r"(\d+)\s+Sev=(\d)\s+\(([\w_]+)\s*\)\s+E_crav→rest=([\d.]+)\s+teleport=([\d.]+)x"
    )
    out = {}
    for m in pattern.finditer(text):
        sid, sev, label, e, tp = m.groups()
        out[sid] = {"sev": int(sev), "label": label, "E": float(e), "tp": float(tp)}
    return out


real = parse(REAL_REWARD_TEXT)
orig = parse(ORIGINAL_TEXT)

print(f"Parsed: real_reward={len(real)}, original={len(orig)}")

common = sorted(set(real) & set(orig))
print(f"Common subjects: {len(common)}")

E_real = np.array([real[s]["E"] for s in common])
E_orig = np.array([orig[s]["E"] for s in common])
tp_real = np.array([real[s]["tp"] for s in common])
tp_orig = np.array([orig[s]["tp"] for s in common])
sev = np.array([real[s]["sev"] for s in common])

print("\n=== Per-subject correlation: real-reward vs original-proxy ===")
rho_E, p_E = spearmanr(E_real, E_orig)
r_E, pr_E = pearsonr(E_real, E_orig)
print(f"E_craving_to_rest: Spearman rho={rho_E:.4f} (p={p_E:.2e}), Pearson r={r_E:.4f}")

rho_tp, p_tp = spearmanr(tp_real, tp_orig)
print(f"teleport_ratio:    Spearman rho={rho_tp:.4f} (p={p_tp:.2e})")

print("\n=== Per-subject absolute change ===")
diff = E_real - E_orig
print(f"E_craving_to_rest change: mean={diff.mean():.4f}, std={diff.std():.4f}, "
      f"median={np.median(diff):.4f}")

print("\n=== Group comparison, BOTH versions ===")
for label_sev, label_name in [(0, "Social_Drinker"), (1, "Abuser"), (2, "Dependent")]:
    mask = sev == label_sev
    print(f"{label_name} (N={mask.sum()}): "
          f"real_median={np.median(E_real[mask]):.4f}  "
          f"orig_median={np.median(E_orig[mask]):.4f}")

print("\n=== Mann-Whitney U: Social_Drinker vs Abuser, BOTH versions ===")
sd_real = E_real[sev == 0]; ab_real = E_real[sev == 1]
sd_orig = E_orig[sev == 0]; ab_orig = E_orig[sev == 1]
u_real, p_u_real = mannwhitneyu(sd_real, ab_real, alternative='two-sided')
u_orig, p_u_orig = mannwhitneyu(sd_orig, ab_orig, alternative='two-sided')
print(f"Real-reward:  U={u_real:.0f}, p={p_u_real:.4f}")
print(f"Original:     U={u_orig:.0f}, p={p_u_orig:.4f}")

print("\n=== Mann-Whitney U: Social_Drinker vs Dependent, BOTH versions ===")
dp_real = E_real[sev == 2]; dp_orig = E_orig[sev == 2]
u_real2, p_u_real2 = mannwhitneyu(sd_real, dp_real, alternative='two-sided')
u_orig2, p_u_orig2 = mannwhitneyu(sd_orig, dp_orig, alternative='two-sided')
print(f"Real-reward:  U={u_real2:.0f}, p={p_u_real2:.4f}")
print(f"Original:     U={u_orig2:.0f}, p={p_u_orig2:.4f}")
