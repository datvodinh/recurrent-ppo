import numpy as np
from numba import jit, njit
from numba.typed import List

TILE_TILE = np.array(
    [
        [1, 11, 12, 17, -1, -1],  #  0
        [0, 2, 17, -1, -1, -1],  #  1
        [1, 3, 16, 17, -1, -1],  #  2
        [2, 4, 16, -1, -1, -1],  #  3
        [3, 5, 15, 16, -1, -1],  #  4
        [4, 6, 15, -1, -1, -1],  #  5
        [5, 7, 14, 15, -1, -1],  #  6
        [6, 8, 14, -1, -1, -1],  #  7
        [7, 9, 13, 14, -1, -1],  #  8
        [8, 10, 13, -1, -1, -1],  #  9
        [9, 11, 12, 13, -1, -1],  #  10
        [0, 10, 12, -1, -1, -1],  #  11
        [0, 10, 11, 13, 17, 18],  #  12
        [8, 9, 10, 12, 14, 18],  #  13
        [6, 7, 8, 13, 15, 18],  #  14
        [4, 5, 6, 14, 16, 18],  #  15
        [2, 3, 4, 15, 17, 18],  #  16
        [0, 1, 2, 12, 16, 18],  #  17
        [12, 13, 14, 15, 16, 17],
    ]  #  18
)

POINT_POINT = np.array(
    [
        [1, 29, -1],  #  0
        [0, 2, 46],  #  1
        [1, 3, -1],  #  2
        [2, 4, -1],  #  3
        [3, 5, 45],  #  4
        [4, 6, -1],  #  5
        [5, 7, 43],  #  6
        [6, 8, -1],  #  7
        [7, 9, -1],  #  8
        [8, 10, 42],  #  9
        [9, 11, -1],  #  10
        [10, 12, 40],  #  11
        [11, 13, -1],  #  12
        [12, 14, -1],  #  13
        [13, 15, 39],  #  14
        [14, 16, -1],  #  15
        [15, 17, 37],  #  16
        [16, 18, -1],  #  17
        [17, 19, -1],  #  18
        [18, 20, 36],  #  19
        [19, 21, -1],  #  20
        [20, 22, 34],  #  21
        [21, 23, -1],  #  22
        [22, 24, -1],  #  23
        [23, 25, 33],  #  24
        [24, 26, -1],  #  25
        [25, 27, 31],  #  26
        [26, 28, -1],  #  27
        [27, 29, -1],  #  28
        [0, 28, 30],  #  29
        [29, 31, 47],  #  30
        [26, 30, 32],  #  31
        [31, 33, 49],  #  32
        [24, 32, 34],  #  33
        [21, 33, 35],  #  34
        [34, 36, 50],  #  35
        [19, 35, 37],  #  36
        [16, 36, 38],  #  37
        [37, 39, 51],  #  38
        [14, 38, 40],  #  39
        [11, 39, 41],  #  40
        [40, 42, 52],  #  41
        [9, 41, 43],  #  42
        [6, 42, 44],  #  43
        [43, 45, 53],  #  44
        [4, 44, 46],  #  45
        [1, 45, 47],  #  46
        [30, 46, 48],  #  47
        [47, 49, 53],  #  48
        [32, 48, 50],  #  49
        [35, 49, 51],  #  50
        [38, 50, 52],  #  51
        [41, 51, 53],  #  52
        [44, 48, 52],
    ]  #  53
)

ROAD_POINT = np.array(
    [
        [0, 29],  #  0
        [0, 1],  #  1
        [1, 46],  #  2
        [1, 2],  #  3
        [2, 3],  #  4
        [3, 4],  #  5
        [4, 45],  #  6
        [4, 5],  #  7
        [5, 6],  #  8
        [6, 43],  #  9
        [6, 7],  #  10
        [7, 8],  #  11
        [8, 9],  #  12
        [9, 42],  #  13
        [9, 10],  #  14
        [10, 11],  #  15
        [11, 40],  #  16
        [11, 12],  #  17
        [12, 13],  #  18
        [13, 14],  #  19
        [14, 39],  #  20
        [14, 15],  #  21
        [15, 16],  #  22
        [16, 37],  #  23
        [16, 17],  #  24
        [17, 18],  #  25
        [18, 19],  #  26
        [19, 36],  #  27
        [19, 20],  #  28
        [20, 21],  #  29
        [21, 34],  #  30
        [21, 22],  #  31
        [22, 23],  #  32
        [23, 24],  #  33
        [24, 33],  #  34
        [24, 25],  #  35
        [25, 26],  #  36
        [26, 31],  #  37
        [26, 27],  #  38
        [27, 28],  #  39
        [28, 29],  #  40
        [29, 30],  #  41
        [30, 31],  #  42
        [31, 32],  #  43
        [32, 49],  #  44
        [32, 33],  #  45
        [33, 34],  #  46
        [34, 35],  #  47
        [35, 50],  #  48
        [35, 36],  #  49
        [36, 37],  #  50
        [37, 38],  #  51
        [38, 51],  #  52
        [38, 39],  #  53
        [39, 40],  #  54
        [40, 41],  #  55
        [41, 52],  #  56
        [41, 42],  #  57
        [42, 43],  #  58
        [43, 44],  #  59
        [44, 53],  #  60
        [44, 45],  #  61
        [45, 46],  #  62
        [46, 47],  #  63
        [47, 48],  #  64
        [30, 47],  #  65
        [48, 49],  #  66
        [49, 50],  #  67
        [50, 51],  #  68
        [51, 52],  #  69
        [52, 53],  #  70
        [48, 53],
    ]  #  71
)

POINT_ROAD = np.array(
    [
        [0, 1, -1],  #  0
        [1, 2, 3],  #  1
        [3, 4, -1],  #  2
        [4, 5, -1],  #  3
        [5, 6, 7],  #  4
        [7, 8, -1],  #  5
        [8, 9, 10],  #  6
        [10, 11, -1],  #  7
        [11, 12, -1],  #  8
        [12, 13, 14],  #  9
        [14, 15, -1],  #  10
        [15, 16, 17],  #  11
        [17, 18, -1],  #  12
        [18, 19, -1],  #  13
        [19, 20, 21],  #  14
        [21, 22, -1],  #  15
        [22, 23, 24],  #  16
        [24, 25, -1],  #  17
        [25, 26, -1],  #  18
        [26, 27, 28],  #  19
        [28, 29, -1],  #  20
        [29, 30, 31],  #  21
        [31, 32, -1],  #  22
        [32, 33, -1],  #  23
        [33, 34, 35],  #  24
        [35, 36, -1],  #  25
        [36, 37, 38],  #  26
        [38, 39, -1],  #  27
        [39, 40, -1],  #  28
        [0, 40, 41],  #  29
        [41, 42, 65],  #  30
        [37, 42, 43],  #  31
        [43, 44, 45],  #  32
        [34, 45, 46],  #  33
        [30, 46, 47],  #  34
        [47, 48, 49],  #  35
        [27, 49, 50],  #  36
        [23, 50, 51],  #  37
        [51, 52, 53],  #  38
        [20, 53, 54],  #  39
        [16, 54, 55],  #  40
        [55, 56, 57],  #  41
        [13, 57, 58],  #  42
        [9, 58, 59],  #  43
        [59, 60, 61],  #  44
        [6, 61, 62],  #  45
        [2, 62, 63],  #  46
        [63, 64, 65],  #  47
        [64, 66, 71],  #  48
        [44, 66, 67],  #  49
        [48, 67, 68],  #  50
        [52, 68, 69],  #  51
        [56, 69, 70],  #  52
        [60, 70, 71],
    ]  #  53
)

PORT_POINT = np.array(
    [[0, 1], [4, 5], [7, 8], [10, 11], [14, 15], [17, 18], [20, 21], [24, 25], [27, 28]]
)

POINT_TILE = np.array(
    [
        [0, -1, -1],  #  0
        [0, 1, -1],  #  1
        [1, -1, -1],  #  2
        [1, -1, -1],  #  3
        [1, 2, -1],  #  4
        [2, -1, -1],  #  5
        [2, 3, -1],  #  6
        [3, -1, -1],  #  7
        [3, -1, -1],  #  8
        [3, 4, -1],  #  9
        [4, -1, -1],  #  10
        [4, 5, -1],  #  11
        [5, -1, -1],  #  12
        [5, -1, -1],  #  13
        [5, 6, -1],  #  14
        [6, -1, -1],  #  15
        [6, 7, -1],  #  16
        [7, -1, -1],  #  17
        [7, -1, -1],  #  18
        [7, 8, -1],  #  19
        [8, -1, -1],  #  20
        [8, 9, -1],  #  21
        [9, -1, -1],  #  22
        [9, -1, -1],  #  23
        [9, 10, -1],  #  24
        [10, -1, -1],  #  25
        [10, 11, -1],  #  26
        [11, -1, -1],  #  27
        [11, -1, -1],  #  28
        [0, 11, -1],  #  29
        [0, 11, 12],  #  30
        [10, 11, 12],  #  31
        [10, 12, 13],  #  32
        [9, 10, 13],  #  33
        [8, 9, 13],  #  34
        [8, 13, 14],  #  35
        [7, 8, 14],  #  36
        [6, 7, 14],  #  37
        [6, 14, 15],  #  38
        [5, 6, 15],  #  39
        [4, 5, 15],  #  40
        [4, 15, 16],  #  41
        [3, 4, 16],  #  42
        [2, 3, 16],  #  43
        [2, 16, 17],  #  44
        [1, 2, 17],  #  45
        [0, 1, 17],  #  46
        [0, 12, 17],  #  47
        [12, 17, 18],  #  48
        [12, 13, 18],  #  49
        [13, 14, 18],  #  50
        [14, 15, 18],  #  51
        [15, 16, 18],  #  52
        [16, 17, 18],
    ]  #  53
)

TILE_POINT = np.array(
    [
        [0, 1, 29, 30, 46, 47],  #  0
        [1, 2, 3, 4, 45, 46],  #  1
        [4, 5, 6, 43, 44, 45],  #  2
        [6, 7, 8, 9, 42, 43],  #  3
        [9, 10, 11, 40, 41, 42],  #  4
        [11, 12, 13, 14, 39, 40],  #  5
        [14, 15, 16, 37, 38, 39],  #  6
        [16, 17, 18, 19, 36, 37],  #  7
        [19, 20, 21, 34, 35, 36],  #  8
        [21, 22, 23, 24, 33, 34],  #  9
        [24, 25, 26, 31, 32, 33],  #  10
        [26, 27, 28, 29, 30, 31],  #  11
        [30, 31, 32, 47, 48, 49],  #  12
        [32, 33, 34, 35, 49, 50],  #  13
        [35, 36, 37, 38, 50, 51],  #  14
        [38, 39, 40, 41, 51, 52],  #  15
        [41, 42, 43, 44, 52, 53],  #  16
        [44, 45, 46, 47, 48, 53],  #  17
        [48, 49, 50, 51, 52, 53],
    ]  #  18
)

ROAD_PRICE = np.array([1, 1, 0, 0, 0])
SETTLEMENT_PRICE = np.array([1, 1, 1, 1, 0])
CITY_PRICE = np.array([0, 0, 0, 2, 3])
DEV_PRICE = np.array([0, 0, 1, 1, 1])

LEN_P_STATE = 220
AMOUNT_ACTION = 95
MAX_TURN_IN_ONE_GAME = (
    1200  #  Càng cao thì chạy càng lâu, nhưng tỉ lệ không end được game càng thấp
)

import warnings

warnings.filterwarnings("ignore")
from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaExperimentalFeatureWarning,
    NumbaPendingDeprecationWarning,
    NumbaWarning,
)

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
warnings.simplefilter("ignore", category=NumbaWarning)


@njit()
def initEnv() -> np.ndarray:
    env = np.zeros(281)

    #  [0:19]: Tài nguyên trên các ô đất
    temp = np.array([5, 0, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 4, 4, 4])
    np.random.shuffle(temp)
    env[0:19] = temp

    #  [19]: Ví trí Robber
    env[19] = np.argmax(temp)

    #  [20:39]: Số trên các ô đất
    temp_prob = np.zeros(19)
    temp = np.ones(19)
    temp[int(env[19])] = 0
    temp_1 = temp.copy()
    for i in range(4):
        temp_2 = np.where(temp == 1)[0]
        k = temp_2[np.random.randint(0, len(temp_2))]
        temp[k] = 0
        temp_1[k] = 0
        if i < 2:
            temp_prob[k] = 6
        else:
            temp_prob[k] = 8

        for j in TILE_TILE[k]:
            if j == -1:
                break
            else:
                temp[j] = 0

    temp = np.array([3, 3, 4, 4, 5, 5, 9, 9, 10, 10, 11, 11, 2, 12])
    np.random.shuffle(temp)
    temp_prob[np.where(temp_1 == 1)[0]] = temp
    env[20:39] = temp_prob

    #  [39:48]: Các cảng
    temp = np.array([5, 5, 5, 5, 0, 1, 2, 3, 4])
    np.random.shuffle(temp)
    env[39:48] = temp

    #  [48:53]: Tài nguyên ngân hàng
    env[48:53] = 15

    #  [53:58]: Thẻ dev bank
    env[53:58] = np.array([14, 2, 2, 2, 5])

    #  Thông tin người chơi: [58,100], [100:142], [142,184], [184:226]
    for p_idx in range(4):
        s_ = 58 + 42 * p_idx  #  58, 100, 142, 184
        #  [+0:+5]: Tài nguyên
        #  [+5:+10]: Thẻ dev
        #  [+10]: Điểm

        #  [+11:+26]: Đường
        #  [+26:+31]: Nhà
        #  [+31:+35]: Thành phố
        env[s_ + 11 : s_ + 35] = -1

        #  [+35]: Số thẻ knight đã dùng
        #  [+36]: Con đường dài nhất

        #  [+37:+42]: Tỉ lệ trao đổi với Bank
        env[s_ + 37 : s_ + 42] = 4

    #  [226]: Danh hiệu quân đội mạnh nhất
    #  [227]: Danh hiệu con đường dài nhất
    #  [228]: Tổng xx
    env[226:229] = -1

    #  [229]: Pha
    env[229] = 0
    #  [230]: Turn

    #  [231]: Điểm đặt thứ nhất
    env[231] = -1

    #  [232]: Số tài nguyên trả do bị chia

    #  [233]: Đang dùng thẻ dev gì
    env[233] = -1

    #  [234]: Số lần sử dụng thẻ dev
    #  [235:239]: Loại thẻ dev được sử dụng trong turn hiện tại
    #  [239:244]: Lượng nguyên liệu còn lại khi bị chia đầu game
    resource_turn_0 = np.random.randint(0, 5)
    env[239:244] = 4
    env[239 + resource_turn_0] = 3
    #  [244]: Người chơi đang action (không hẳn là người chơi chính)
    #  [245:249], số nguyên liệu đã lấy trong turn đầu game
    env[245] = 1  # Người chơi đầu tiên đã nhận 1 nguyên liệu
    #  [249:254]: Tài nguyên đưa ra trong trade offer
    #  [254:259]: Tài nguyên yêu cầu trong trade offer
    #  [254:274]: Tài nguyên trong kho dự trữ của các người chơi
    env[254 + resource_turn_0] = 1
    env[280] = 0
    return env


@njit()
def getAgentState(env: np.ndarray) -> np.ndarray:
    p_state = np.zeros(P_END_GAME_N + 1)
    p_idx = int(env[244])

    #  p_state[0:48] = env[0:48]

    for id, res in enumerate(env[:19]):  # Tài nguyên trên các ô đất
        p_state[id * 6 + int(res)] = 1
    p_state[P_START_ROBBLE + int(env[19])] = 1  # Vị trí robble
    for id, count in enumerate(env[20:39]):  # Số trên các ô đất
        if count == 0:
            p_state[P_START_COUNT_LAND + id * 12] = 1
        else:
            p_state[P_START_COUNT_LAND + id * 12 + int(count) - 1] = 1
    for id, port in enumerate(env[39:48]):  # Các cảng
        p_state[P_START_PORT + id * 6 + int(port)] = 1

    p_state[P_START_RES_BANK:P_END_RES_BANK] = (
        env[48:53] > 0
    )  # Tài nguyên ngân hàng dạng 0, 1
    p_state[P_DEV_BANK] = (env[53:58] > 0).any()  # Thẻ dev Bank dạng không hoặc có

    s_ = 58 + 42 * p_idx
    p_state[P_PLAYER : P_PLAYER + 11] = env[s_ : s_ + 11]  # Tài nguyên, thẻ dev, điểm
    if env[280] == 0:
        p_state[P_PLAYER + 10] += p_state[P_PLAYER + 9]

    road_have = env[s_ + 11 : s_ + 26]
    road_have = road_have[road_have >= 0].astype(np.int64)
    p_state[P_PLAYER + 11 + road_have] = 1

    stm_have = env[s_ + 26 : s_ + 31]
    stm_have = stm_have[stm_have >= 0].astype(np.int64)
    p_state[P_PLAYER + 83 + stm_have] = 1

    city_have = env[s_ + 31 : s_ + 35]
    city_have = city_have[city_have >= 0].astype(np.int64)
    p_state[P_PLAYER + 137 + city_have] = 1

    p_state[P_PLAYER + 191] = env[s_ + 35]
    p_state[P_PLAYER + 192] = env[s_ + 36]

    for id, trade in enumerate(env[s_ + 37 : s_ + 42]):
        p_state[P_PLAYER + 193 + 3 * id + int(trade) - 2] = 1

    for i in range(1, 4):
        e_idx = (p_idx + i) % 4
        s_e = 58 + 42 * e_idx

        s_new = P_OTHER_PLAYER + 185 * (i - 1)
        p_state[s_new + 0] = np.sum(env[s_e : s_e + 5])  #  [+0]: Tổng tài nguyên
        p_state[s_new + 1] = np.sum(env[s_e + 5 : s_e + 10])  #  [+1]: Tổng số thẻ dev
        p_state[s_new + 2] = env[s_e + 10]  #  [+2]: Điểm

        road_have = env[s_e + 11 : s_e + 26]
        road_have = road_have[road_have >= 0].astype(np.int64)
        p_state[s_new + 3 + road_have] = 1

        stm_have = env[s_e + 26 : s_e + 31]
        stm_have = stm_have[stm_have >= 0].astype(np.int64)
        p_state[s_new + 75 + stm_have] = 1

        city_have = env[s_e + 31 : s_e + 35]
        city_have = city_have[city_have >= 0].astype(np.int64)
        p_state[s_new + 129 + city_have] = 1

        p_state[s_new + 183] = env[s_e + 35]
        p_state[s_new + 184] = env[s_e + 36]

    if env[226] != -1:
        temp = env[226]
        idx = int((temp - p_idx) % 4)
        p_state[P_START_STRONGEST_ARMY + idx] = 1  # Quân đội mạnh nhất
    if env[227] != -1:
        temp = env[227]
        idx = int((temp - p_idx) % 4)
        p_state[P_START_LONGEST_ROAD + idx] = 1  # Con đường dài nhất
    p_state[P_START_DICE + int(env[228]) - 2] = 1  # Tổng xúc xắc
    p_state[P_RES_REMAIN_START_GAME] = env[
        245 + p_idx
    ]  #  Nguyên liệu còn lại có thể lấy ở đầu game
    if int(env[231]) != -1:  # Điểm đặt thứ nhất(Có -1)
        p_state[P_START_FIRST_POINT + int(env[231])] = 1

    p_state[P_SPLIT_RES] = env[232]  # Số tài nguyên phải bỏ do bị chia
    if int(env[233]) != -1:  # Đang dùng thẻ dev gì
        p_state[P_START_DEV_USING + int(env[233])] = 1
    p_state[P_COUNT_USING_DEV] = env[234]  # Số lần dùng thẻ dev

    p_state[P_START_DEV_CAN_USE:P_END_DEV_CAN_USE] = env[235:239]  #
    #  p_state[P_START_RES_WAREHOUSE:P_END_RES_WAREHOUSE] = env[239:244] #Số nguyên liệu còn lại ở trong kho
    p_state[P_START_RES_WAREHOUSE_PLAYER:P_END_RES_WAREHOUSE_PLAYER] = env[
        254 + 5 * p_idx : 259 + 5 * p_idx
    ]  # Các ngyên liệu có trong kho của người chơi
    phase = int(env[229])
    p_state[P_START_PHASE + phase] = 1
    if np.max(env[249:254]) > 0:
        p_state[
            P_START_TRADE_BANK + np.argmax(env[249:254])
        ] = 1  # Tài nguyên đưa ra trong trade offer để trade với bank
    p_state[P_END_GAME_N] = env[280]
    return p_state


P_PHASE = 205
TOTAL_PHASE = 13
P_TRADE_BANK = 218
P_END_GAME = 219

LEN_P_STATE_NEW = 992
P_START_ROBBLE = 114
P_END_ROBBLE = 133
P_START_COUNT_LAND = 133
P_END_COUNT_LAND = 361
P_START_PORT = 361
P_END_PORT = 415
P_START_RES_BANK = 415
P_END_RES_BANK = 420
P_DEV_BANK = 420
P_PLAYER = 421
P_END_PLAYER = 629
P_OTHER_PLAYER = 629
P_END_OTHER_PLAYER = 1184
P_START_STRONGEST_ARMY = 1184
P_END_STRONGEST_ARMY = 1188
P_START_LONGEST_ROAD = 1188
P_END_LONGEST_ROAD = 1192
P_START_DICE = 1192
P_END_DICE = 1203
P_RES_REMAIN_START_GAME = 1203
P_START_FIRST_POINT = 1204
P_END_FIRST_POINT = 1258
P_SPLIT_RES = 1258
P_START_DEV_USING = 1259
P_END_DEV_USING = 1263
P_COUNT_USING_DEV = 1263
P_START_DEV_CAN_USE = 1264
P_END_DEV_CAN_USE = 1268
#  P_START_RES_WAREHOUSE = 1268
#  P_END_RES_WAREHOUSE = 1273
P_START_RES_WAREHOUSE_PLAYER = 1268
P_END_RES_WAREHOUSE_PLAYER = 1273

P_START_PHASE = 1273
P_END_PHASE = 1286
P_START_TRADE_BANK = 1286
P_END_TRADE_BANK = 1291
P_END_GAME_N = 1291


@njit()
def new_to_old(p_state_new):
    p_state_old = np.zeros(220)

    for i in range(19):  # Tài nguyên trên các ô đất
        p_state_old[i] = np.argmax(p_state_new[i * 6 : (i + 1) * 6])

    p_state_old[19] = np.argmax(
        p_state_new[P_START_ROBBLE:P_END_ROBBLE]
    )  # Vị trí robble

    for i in range(19):  # Số trên các ô đất
        count = np.argmax(
            p_state_new[P_START_COUNT_LAND + i * 12 : P_START_COUNT_LAND + (i + 1) * 12]
        )
        if count != 0:
            count += 1
        p_state_old[20 + i] = count

    for i in range(9):  # Các cảng
        p_state_old[39 + i] = np.argmax(
            p_state_new[P_START_PORT + i * 6 : P_START_PORT + (i + 1) * 6]
        )

    p_state_old[48:53] = p_state_new[
        P_START_RES_BANK:P_END_RES_BANK
    ]  # Tài nguyên ngân hàng dạng 0, 1
    p_state_old[53] = p_state_new[P_DEV_BANK]  # Thẻ dev Bank dạng không hoặc có
    p_state_old[54 : 54 + 11] = p_state_new[
        P_PLAYER : P_PLAYER + 11
    ]  # Tài nguyên, thẻ dev, điểm

    road_have = np.where(p_state_new[P_PLAYER + 11 : P_PLAYER + 83] == 1)[0]
    p_state_old[54 + 11 : 54 + 26] = np.full(15, -1)
    if len(road_have) > 0:
        p_state_old[54 + 11 : 54 + 11 + len(road_have)] = road_have

    stm_have = np.where(p_state_new[P_PLAYER + 83 : P_PLAYER + 137] == 1)[0]
    p_state_old[54 + 26 : 54 + 31] = np.full(5, -1)
    if len(stm_have) > 0:
        p_state_old[54 + 26 : 54 + 26 + len(stm_have)] = stm_have

    city_have = np.where(p_state_new[P_PLAYER + 137 : P_PLAYER + 191] == 1)[0]
    p_state_old[54 + 31 : 54 + 35] = np.full(4, -1)
    if len(city_have) > 0:
        p_state_old[54 + 31 : 54 + 31 + len(city_have)] = city_have

    p_state_old[54 + 35] = p_state_new[P_PLAYER + 191]
    p_state_old[54 + 36] = p_state_new[P_PLAYER + 192]

    for i in range(5):  # Tỉ lệ trade
        p_state_old[54 + 37 + i] = (
            np.argmax(
                p_state_new[P_PLAYER + 193 + i * 3 : P_PLAYER + 193 + (i + 1) * 3]
            )
            + 2
        )

    for id_player in range(3):
        s_ = 96 + 29 * id_player
        s_new = P_OTHER_PLAYER + 185 * id_player
        p_state_old[s_ + 0] = p_state_new[s_new + 0]  #  [+0]: Tổng tài nguyên
        p_state_old[s_ + 1] = p_state_new[s_new + 1]  #  [+1]: Tổng số thẻ dev
        p_state_old[s_ + 2] = p_state_new[s_new + 2]  #  [+2]: Điểm

        road_have = np.where(p_state_new[s_new + 3 : s_new + 75] == 1)[0]
        p_state_old[s_ + 3 : s_ + 18] = np.full(15, -1)
        if 15 >= len(road_have) > 0:
            #  print(len(road_have), len(p_state_old[s_+3:s_+3+len(road_have)]))
            p_state_old[s_ + 3 : s_ + 3 + len(road_have)] = road_have

        stm_have = np.where(p_state_new[s_new + 75 : s_new + 129] == 1)[0]
        p_state_old[s_ + 18 : s_ + 23] = np.full(5, -1)
        if 5 >= len(stm_have) > 0:
            p_state_old[s_ + 18 : s_ + 18 + len(stm_have)] = stm_have

        city_have = np.where(p_state_new[s_new + 129 : s_new + 183] == 1)[0]
        p_state_old[s_ + 23 : s_ + 27] = np.full(4, -1)
        if 4 >= len(city_have) > 0:
            p_state_old[s_ + 23 : s_ + 23 + len(city_have)] = city_have

        p_state_old[s_ + 27] = p_state_new[s_new + 183]
        p_state_old[s_ + 28] = p_state_new[s_new + 184]

    if np.sum(p_state_new[P_START_STRONGEST_ARMY:P_END_STRONGEST_ARMY]) == 0:
        p_state_old[183] = -1
    else:
        p_state_old[183] = np.argmax(
            p_state_new[P_START_STRONGEST_ARMY:P_END_STRONGEST_ARMY]
        )

    if np.sum(p_state_new[P_START_LONGEST_ROAD:P_END_LONGEST_ROAD]) == 0:
        p_state_old[184] = -1
    else:
        p_state_old[184] = np.argmax(
            p_state_new[P_START_LONGEST_ROAD:P_END_LONGEST_ROAD]
        )

    if np.sum(p_state_new[P_START_DICE:P_END_DICE]) == 0:
        p_state_old[185] = -1
    else:
        p_state_old[185] = np.argmax(p_state_new[P_START_DICE:P_END_DICE]) + 2

    p_state_old[186] = p_state_new[
        P_RES_REMAIN_START_GAME
    ]  #  Nguyên liệu còn lại có thể lấy ở đầu game

    if np.sum(p_state_new[P_START_FIRST_POINT:P_END_FIRST_POINT]) == 0:
        p_state_old[187] = -1
    else:
        p_state_old[187] = np.argmax(p_state_new[P_START_FIRST_POINT:P_END_FIRST_POINT])

    p_state_old[188] = p_state_new[P_SPLIT_RES]  # Số tài nguyên phải bỏ do bị chia

    if np.sum(p_state_new[P_START_DEV_USING:P_END_DEV_USING]) == 0:
        p_state_old[189] = -1
    else:
        p_state_old[189] = np.argmax(p_state_new[P_START_DEV_USING:P_END_DEV_USING])

    p_state_old[190] = p_state_new[P_COUNT_USING_DEV]  # Số lần dùng thẻ dev
    p_state_old[191:195] = p_state_new[P_START_DEV_CAN_USE:P_END_DEV_CAN_USE]
    p_state_old[200:218] = p_state_new[P_START_RES_WAREHOUSE_PLAYER:P_END_PHASE]
    p_state_old[218] = np.argmax(p_state_new[P_START_TRADE_BANK:P_END_TRADE_BANK])
    return p_state_old.astype(np.int64)


@njit()
def getValidActions(p_state_new: np.ndarray):
    p_state = new_to_old(p_state_new)
    list_action = np.zeros(AMOUNT_ACTION)
    #  phase = np.where(p_state[P_PHASE:P_PHASE+TOTAL_PHASE] == 1)[0]
    #  if len(phase) == 0:
    #      return list_action
    #  else: phase = phase[0]

    if np.sum(p_state[P_PHASE : P_PHASE + TOTAL_PHASE]) == 0:
        return list_action
    else:
        phase = np.argmax(p_state[P_PHASE : P_PHASE + TOTAL_PHASE])

    if phase == 0:  #  Chọn điểm đặt nhà đầu game
        list_action[0:54] = 1

        temp = p_state[np.array([80, 81, 114, 115, 143, 144, 172, 173])].astype(
            np.int32
        )
        for stm in temp:
            if stm != -1:
                list_action[stm] = 0
                for point in POINT_POINT[stm]:
                    if point != -1:
                        list_action[point] = 0

        return list_action.astype(np.float64)

    if phase == 6:  #  Chọn các mô đun giữa turn
        check_useDev(p_state, list_action)

        if (p_state[54:59] >= ROAD_PRICE).all():
            #  Mua đường (83)
            if p_state[79] == -1 and check_firstPoint(p_state):
                list_action[83] = 1

            #  Mua nhà (84)
            if (p_state[54:59] >= SETTLEMENT_PRICE).all() and p_state[84] == -1:
                p_point, all_stm_and_city = get_p_point_n_all_stm_city(p_state)
                list_point = np.where(p_point == 1)[0]
                for point in list_point:
                    check = True
                    for road in POINT_ROAD[point]:
                        if road != -1:
                            for neares_point in ROAD_POINT[road]:
                                if all_stm_and_city[neares_point] == 1:
                                    check = False
                                    break

                            if not check:
                                break

                    if check:
                        list_action[84] = 1
                        break

        #  Mua thành phố (85)
        if (
            (p_state[54:59] >= CITY_PRICE).all()
            and p_state[80] != -1
            and p_state[88] == -1
        ):
            list_action[85] = 1

        #  Mua thẻ dev (86)
        if (p_state[54:59] >= DEV_PRICE).all() and p_state[53] == 1:
            list_action[86] = 1

        if (p_state[54:59] > 0).any():  #  Trade với bank (87)
            if (p_state[54:59] >= p_state[91:96]).any():
                temp = np.where(p_state[54:59] >= p_state[91:96])[0]
                for res in temp:
                    for res_1 in range(5):
                        if res_1 != res and p_state[48 + res_1] == 1:
                            list_action[87] = 1
                            break

                    if list_action[87] == 1:
                        break

        #  Action lấy nguyên liệu từ kho
        if (p_state[200:205] > 0).any() and p_state[186] == 1:
            #  print('hi', p_state[200:205], (p_state[200:205] > 0).all())
            list_action[94] = 1

        #  Kết thúc lượt (92)
        list_action[88] = 1

        return list_action.astype(np.float64)

    if phase == 3:  #  Trả tài nguyên do bị chia bài
        for i in range(5):
            if p_state[54 + i] > 0:
                list_action[89 + i] = 1

        return list_action.astype(np.float64)

    if phase == 11:  #  Chọn tài nguyên muốn nhận từ ngân hàng
        #  Chọn những tài nguyên mà ngân hàng có, khác tài nguyên đưa ra
        list_action[59:64] = p_state[48:53]
        #  list_action[59+np.argmax(p_state[196:201])] = 0
        list_action[int(59 + p_state[P_TRADE_BANK])] = 0

        return list_action.astype(np.float64)

    if phase == 10:  #  Chọn tài nguyên khi trade với ngân hàng
        #  Chọn những tài nguyên mà khi chọn, ngân hàng còn ít nhất 1 loại tài nguyên khác
        for res in range(5):
            if p_state[54 + res] >= p_state[91 + res]:
                for res_1 in range(5):
                    if res_1 != res and p_state[48 + res_1] == 1:
                        list_action[89 + res] = 1
                        break
        return list_action.astype(np.float64)

    if phase == 1:  #  Chọn các điểm đầu mút của đường
        if p_state[187] == -1:  #  Chọn điểm thứ nhất
            p_point, all_road = get_p_point_n_all_road(p_state)
            list_point = np.where(p_point == 1)[0]
            #   print('lst point', list_point)
            for point in list_point:
                for road in POINT_ROAD[point]:
                    if road != -1 and all_road[road] == 0:
                        list_action[point] = 1
                        break
            return list_action.astype(np.float64)

        all_road = np.zeros(72)

        temp = p_state[65:80].astype(np.int32)
        for i in temp:
            if i == -1:
                break
            else:
                all_road[i] = 1

        for j in range(3):
            s_ = 96 + 29 * j

            temp = p_state[s_ + 3 : s_ + 18].astype(np.int32)
            for i in temp:
                if i == -1:
                    break
                else:
                    all_road[i] = 1

        first_point = int(p_state[187])
        for road in POINT_ROAD[first_point]:
            if road != -1 and all_road[road] == 0:
                list_action[ROAD_POINT[road]] = 1

        list_action[first_point] = 0
        #   print(np.where(list_action == 1)[0])
        return list_action.astype(np.float64)

    if phase == 4:  #  Di chuyển Robber
        list_action[64:83] = 1
        list_action[int(64 + p_state[19])] = 0

        return list_action.astype(np.float64)

    if phase == 5:  #  Lấy nguyên liệu đầu game
        list_action[59:64] = (p_state[195:200] > 0) * 1

    if phase == 2:  #  Đổ xx hoặc dùng thẻ dev
        #  Đổ xx
        list_action[54] = 1

        check_useDev(p_state, list_action)

        return list_action.astype(np.float64)

    if phase == 8:  #  Chọn các điểm mua nhà
        p_point, all_stm_and_city = get_p_point_n_all_stm_city(p_state)
        list_point = np.where(p_point == 1)[0]
        for point in list_point:
            list_road = POINT_ROAD[point][POINT_ROAD[point] != -1]
            nearest_points = ROAD_POINT[list_road].flatten()
            if (all_stm_and_city[nearest_points] == 0).all():
                list_action[point] = 1

        return list_action.astype(np.float64)

    if phase == 7:  #  Chọn tài nguyên khi dùng thẻ dev
        if p_state[189] == 2:  #  Đang dùng yearofplenty
            list_action[59:64] = p_state[48:53]
        elif p_state[189] == 3:  #  Đang dùng monopoly
            list_action[59:64] = 1

        return list_action.astype(np.float64)

    if phase == 9:  #  Chọn các điểm mua thành phố
        temp = p_state[80:85].astype(np.int32)
        for p_stm in temp:
            if p_stm == -1:
                break
            else:
                list_action[p_stm] = 1

        return list_action.astype(np.float64)

    if phase == 12:  # Lấy nguyên liệu từ kho ở trước hoặc sau khi đổ xúc xắc
        list_action[59:64] = (p_state[200:205] > 0) * 1
        return list_action.astype(np.float64)
    return list_action.astype(np.float64)


@njit()
def random_choice_weight(a):
    b = []
    for id_res, count in enumerate(a):
        for number_res in range(int(count)):
            b.append(id_res)
    res_choice = np.random.randint(0, len(b))
    return b[res_choice]


@njit()
def stepEnv(env: np.ndarray, action: int):
    phase = env[229]
    p_idx = int(env[244])
    s_ = 58 + 42 * p_idx

    if phase == 6:  #  Chọn các mô đun giữa turn
        if action == 88:  #  Kết thúc lượt
            env[239] = 1  #  Reset lại số lần tạo trade offer
            env[230] += 1  #  Tăng turn
            n_idx = int(env[230] % 4)
            env[245 + n_idx] = 1
            s_n = int(58 + 42 * n_idx)
            env[244] = n_idx  #  Người đi tiếp theo

            #  Kiểm tra thẻ dev có thể sử dụng
            env[235:239] = env[s_n + 5 : s_n + 9] > 0
            #  print(env[235:239])

            #  Tổng xx
            env[228] = -1
            #  print('Trả lại giá trị cho xúc xắc và đổi người chơi', env[228])

            #  Pha
            env[229] = 2

            if (env[235:239] == 0).all():
                roll_xx(env)
                #  print('sau khi roll', env[228])

            return

        if action == 87:  #  Trade với bank
            env[229] = 10

            return

        if action == 83:  #  Mua đường
            env[s_ : s_ + 5] -= ROAD_PRICE  #  Trả tài nguyên
            env[48:53] += ROAD_PRICE
            env[229] = 1  #  Chuyển sang pha đặt đường

            return

        if action == 86:  #  Mua thẻ dev, không chuyển pha
            env[s_ : s_ + 5] -= DEV_PRICE  #  Trả tài nguyên mua thẻ dev
            env[48:53] += DEV_PRICE
            temp = env[53:58]
            #  print('Các thẻ dev còn lại trong game', temp)
            dev_idx = weighted_random(temp)
            env[53 + dev_idx] -= 1  #  type: ignore #  Trừ thẻ dev ở ngân hàng
            env[s_ + 5 + dev_idx] += 1  #  type: ignore #  Cộng thẻ dev cho người chơi

            #  Nếu thẻ dev vừa nhận là thẻ vp (dev_idx = 4) thì cộng điểm cho người chơi
            #  if dev_idx == 4:
            #      env[s_+10] += 1
            #  Note: Không được cộng do như thế các người khác sẽ biết

            return

        if action >= 55 and action < 59:  #  Dùng thẻ dev
            useDev(env, action, s_, p_idx)

            return

        if action == 84:  #  Mua nhà
            env[s_ : s_ + 5] -= SETTLEMENT_PRICE
            env[48:53] += SETTLEMENT_PRICE
            env[229] = 8  #  Sang pha mua nhà

            return

        if action == 85:  #  Mua thành phố
            env[s_ : s_ + 5] -= CITY_PRICE
            env[48:53] += CITY_PRICE
            env[229] = 9  #  Chuyển sang pha mua thành phố

            return

        if action == 94:  # Lấy thẻ từ kho dự trữ phát triển
            env[229] = 12
            env[245:249] = 0
            return

    if phase == 3:  #  Trả tài nguyên do bị chia bài
        env[s_ + action - 89] -= 1  #  Trừ tài nguyên của người chơi
        env[action - 41] += 1  #  Trả tài nguyên này cho ngân hàng

        env[232] -= 1  #  Giảm số lượng thẻ cần trả
        if env[232] == 0:  #  Đã trả đủ
            for i in range(1, 5):
                e_idx = (p_idx + i) % 4
                s_e = 58 + 42 * e_idx
                if e_idx == env[230] % 4:  #  Quay về người chơi chính
                    env[229] = 4  #  Sang pha di chuyển Robber
                    env[244] = e_idx  #  Chuyển người action
                    break
                else:
                    if np.sum(env[s_e : s_e + 5]) > 7:
                        #  Cập nhật số lá phải bỏ
                        env[232] = np.sum(env[s_e : s_e + 5]) // 2
                        env[244] = e_idx  #  Chuyển người action
                        break

        return

    if phase == 11:  #  Chọn tài nguyên muốn nhận từ ngân hàng
        res_idx = action - 59

        #  Người chơi
        env[s_ + res_idx] += 1
        env[s_ : s_ + 5] -= env[249:254]

        #  Ngân hàng
        env[48 + res_idx] -= 1
        env[48:53] += env[249:254]

        env[249:254] = 0
        env[229] = 6

        return

    if phase == 10:  #  Chọn tài nguyên khi trade với ngân hàng
        res_idx = action - 89
        env[249 + res_idx] = env[s_ + 37 + res_idx]

        env[229] = 11

        return

    if phase == 1:  #  Chọn các điểm đầu mút của đường
        if env[231] == -1:  #  Chưa có điểm đặt thứ nhất
            env[231] = action

            return

        #  Tìm con đường tương ứng và thay đổi tập đường
        for road in np.where(ROAD_POINT == env[231])[0]:
            if action in ROAD_POINT[road]:
                i = np.count_nonzero(env[s_ + 11 : s_ + 26] != -1)
                env[s_ + 11 + i] = road
                break

        #  Check con đường dài nhất
        env[s_ + 36] = get_p_longest_road(env, p_idx)
        list_p_longestRoad = env[np.array([94, 136, 178, 220])]
        longestRoad = np.max(list_p_longestRoad)

        #  Check lại danh hiệu
        if env[s_ + 36] >= 5 and env[s_ + 36] == longestRoad and env[227] != p_idx:
            #  Nếu cddn < 5 thì ko cần xét do ko thể giành danh hiệu
            #  Nếu cddn != longestRoad thì cũng ko cần xét do có xét cũng ko được danh hiệu
            #  Nếu đang có danh hiệu thì việc xây đường giúp củng cố danh hiệu, nên ko cần xét
            list_p_check = np.where(list_p_longestRoad == longestRoad)[0]
            if len(list_p_check) == 1:  #  Ghi nhận danh hiệu hoặc thay đổi danh hiệu
                if env[227] == -1:  #  Chưa có danh hiệu
                    env[227] = p_idx  #  Ghi nhận danh hiệu
                    env[s_ + 10] += 2  #  Cộng điểm
                else:  #  Thay đổi danh hiệu
                    env[int(58 + 42 * env[227] + 10)] -= 2  #  Trừ điểm người cũ
                    env[227] = p_idx  #  Ghi nhận người mới
                    env[s_ + 10] += 2  #  Cộng điểm người cũ
            #  Nếu có 2 người cùng có con đường dài nhất, không thay đổi danh hiệu

        #  Xóa điểm đặt thứ nhất
        env[231] = -1

        #  16 turn đầu: Kết thúc turn, chuyển người chơi, nếu là turn 8 thì chỉ chuyển pha
        if env[230] <= 15:
            env[230] += 2
            turn = env[230]
            if env[230] == 16:  #  Chỉ chuyển pha
                env[239] = 1  #  Cài lại số lần tạo trade offer
                env[229] = 2  #  Sang pha 2: Đổ xx hoặc dùng thẻ dev
                roll_xx(env)  #  Chưa thể có thẻ dev trong trường hợp này
            else:  #  Thay đổi người chơi, chuyển pha
                env[229] = 0
                if env[230] < 8:
                    env[244] = env[230] // 2
                    total_resource_take = int(turn // 2 + 1)
                    for i_res in range(total_resource_take):
                        #  res = random.choices([0,1,2,3,4], weights = env[239:244])[0]
                        res = random_choice_weight(env[239:244])
                        env[239 + res] -= 1
                        if p_idx != 3:
                            o_idx = p_idx + 1
                        else:
                            o_idx = 3
                        env[254 + 5 * o_idx + res] += 1
                        env[245 + o_idx] += 1
                else:
                    env[244] = 7 - env[230] // 2
                    if turn == 8:
                        o_idx = int(p_idx)
                        total_resource_take = int(5 - env[245 + o_idx])
                    else:
                        o_idx = int(p_idx - 1)
                        total_resource_take = int(5 - env[245 + o_idx])
                    for i_res in range(total_resource_take):
                        #  res = random.choices([0,1,2,3,4], weights = env[239:244])[0]
                        res = random_choice_weight(env[239:244])
                        env[239 + res] -= 1
                        env[254 + 5 * o_idx + res] += 1
                        env[245 + o_idx] += 1

        else:  #  Các turn giữa game
            if env[233] != 1:  #  Không dùng thẻ roadbuilding, về pha 6
                env[229] = 6

                return

            if env[233] == 1:  #  Đang dùng thẻ roadbuilding
                env[234] -= 1
                #  Hoặc là hết lượt dùng, hoặc đủ 15 đường, hoặc hết vị trí xây đường
                check = True
                #  if env[234] == 0 or (env[s_+11:s_+26] != -1).all():
                if env[234] == 0 or env[s_ + 25] != -1:
                    check = False

                if check:
                    check = False
                    p_point = np.zeros(54)
                    all_road = np.zeros(72)

                    #  Đường và điểm của người chơi
                    temp = env[s_ + 11 : s_ + 26].astype(np.int32)
                    #  list_road = temp[temp != -1].astype(np.int32)
                    #  all_road[list_road] = 1
                    #  p_point[ROAD_POINT[list_road].flatten()] = 1

                    for road in temp:
                        if road == -1:
                            break
                        else:
                            all_road[road] = 1
                            p_point[ROAD_POINT[road]] = 1

                    for i in range(1, 4):
                        e_idx = (p_idx + i) % 4
                        s_e = 58 + 42 * e_idx

                        temp = env[s_e + 11 : s_e + 26].astype(np.int32)
                        #  list_road = temp[temp != -1].astype(np.int32)
                        #  all_road[list_road] = 1
                        for road in temp:
                            if road == -1:
                                break
                            else:
                                all_road[road] = 1

                        temp = env[s_e + 26 : s_e + 31].astype(np.int32)
                        #  stm_and_city = temp[temp != -1].astype(np.int32)
                        #  p_point[stm_and_city] = 0

                        for stm in temp:
                            if stm == -1:
                                break
                            else:
                                p_point[stm] = 0

                        temp = env[s_e + 31 : s_e + 35].astype(np.int32)
                        for city in temp:
                            if city == -1:
                                break
                            else:
                                p_point[city] = 0

                    list_point = np.where(p_point == 1)[0]
                    for point in list_point:
                        #  nearest_roads = POINT_ROAD[point][POINT_ROAD[point] != -1]
                        #  if (all_road[nearest_roads] == 0).any():
                        #      check = True
                        #      break
                        for road in POINT_ROAD[point]:
                            if road != -1 and all_road[road] == 0:
                                check = True
                                break

                        if check:
                            break

                if not check:  #  Kết thúc trạng thái sử dụng thẻ roadbuilding
                    after_useDev(env)

                return

        return

    if phase == 4:  #  Di chuyển Robber
        robber_pos = action - 64
        env[19] = robber_pos  #  Cập nhật vị trí Robber
        res = int(env[int(env[19])])
        if res != 5 and env[48 + res] > 0:
            env[s_ + res] += 1  # Người chơi
            env[48 + res] -= 1  # Ngân hàng
        after_Rob(env)

        return

    if phase == 5:  #  Lấy nguyên liệu đầu game
        res_idx = action - 59
        env[254 + 5 * p_idx + res_idx] += 1
        env[239 + res_idx] -= 1
        turn = env[230]
        env[245 + p_idx] += 1
        if turn <= 7:
            if env[245 + p_idx] < (turn // 2 + 1):
                env[229] = 5
            else:
                env[229] = 0  # Lấy nguyên liệu xong chuyển sang phase 0
                env[230] += 1
                env[244] = turn // 2
        elif turn <= 15:
            if env[245 + p_idx] < 5:
                env[229] = 5
            else:
                env[229] = 0  # Lấy nguyên liệu xong chuyển sang phase 0
                env[230] += 1
                env[244] = 3 - (turn - 8) // 2
                #  print('Chuyển người chơi thành, ', (3 - (turn - 8)//2), env[244])

        return

    if phase == 2:  #  Đổ xx hoặc dùng thẻ dev
        if action == 54:  #  Đổ xúc xắc
            roll_xx(env)

            return

        useDev(env, action, s_, p_idx)

        return

    if phase == 0:  #  Chọn điểm đặt nhà đầu game
        update_new_stm(env, action, s_)

        #  Nếu là lần 2 thì cộng tài nguyên
        if env[s_ + 10] == 2:
            temp = POINT_TILE[action][POINT_TILE[action] != -1]
            nearest_tiles = temp[env[temp] != 5]
            for tile in nearest_tiles:
                env_tile = int(env[tile])
                env[s_ + env_tile] += 1  #  Cộng tài nguyên cho người chơi
                env[48 + env_tile] -= 1  #  Trừ tài nguyên ở ngân hàng

        #  Chuyển sang pha đặt đường
        env[231] = action  #  Điểm đặt thứ nhất
        env[229] = 1  #  Chuyển sang pha 1: Đặt đường
        if env[230] == 15:
            env[245:249] = 0
        return

    if phase == 8:  #  Chọn các điểm mua nhà
        update_new_stm(env, action, s_)

        #  Check xem nhà vừa xây có nằm trên con đường của ai khác hay ko
        check = False
        w_idx = 9999

        for i in range(1, 4):
            e_idx = (p_idx + i) % 4
            s_e = 58 + 42 * e_idx

            temp = env[s_e + 11 : s_e + 26]
            list_road = temp[temp != -1].astype(np.int32)
            count = 0
            for road in POINT_ROAD[action]:
                if road != -1 and road in list_road:
                    count += 1

            if count == 2:
                check = True
                w_idx = e_idx
                break

        if check:  #  Người này vừa xây nhà cắt đường người khác
            s_w = 58 + 42 * w_idx
            env[s_w + 36] = get_p_longest_road(env, w_idx)
            list_p_longestRoad = env[np.array([94, 136, 178, 220])]
            longestRoad = np.max(list_p_longestRoad)

            if longestRoad < 5:
                if env[227] != -1:  #  Có ông có danh hiệu, chứng tỏ vừa bị cướp
                    env[int(58 + 42 * env[227] + 10)] -= 2  #  Trừ điểm
                    env[227] = -1  #  Thu hồi danh hiệu
                #   Nếu chưa ai có danh hiệu thì bỏ qua
            else:
                list_p_check = np.where(list_p_longestRoad == longestRoad)[0]
                if len(list_p_check) == 1:  #  Có đúng 1 ông có con đường dài nhất
                    if env[227] == -1:  #  Chưa có ai có danh hiệu
                        env[227] = list_p_check[0]  #  Bàn chơi ghi nhận
                        env[int(58 + 42 * env[227] + 10)] += 2  #  Cộng điểm
                    else:  #  Có ông đã có danh hiệu
                        if env[227] != list_p_check[0]:  #  Thay đổi danh hiệu
                            env[int(58 + 42 * env[227] + 10)] -= 2  #  Trừ điểm ông cũ
                            env[227] = list_p_check[0]  #  Bàn chơi ghi nhận
                            #  Cộng điểm ông mới
                            env[int(58 + 42 * env[227] + 10)] += 2
                        #   Nếu cùng là một người thì không thay đổi danh hiệu
                else:  #  Có từ 2 ông trở lên cùng có con đường dài nhất
                    #  Đang có người có danh hiệu, nhưng con đường không dài nhất
                    if env[227] != -1 and env[227] not in list_p_check:
                        env[int(58 + 42 * env[227] + 10)] -= 2  #  Trừ điểm
                        env[227] = -1
                    #  Hoặc là chưa ai có danh hiệu, hoặc là danh hiệu nằm trong list thì ko cần check

        env[229] = 6  #  Về pha chọn mô đun

        return

    if phase == 7:  #  Chọn tài nguyên khi dùng thẻ dev
        if env[233] == 2:  #  Đang dùng thẻ yearofplenty
            env[action - 11] -= 1  #  Trừ tài nguyên cho ngân hàng
            env[s_ + action - 59] += 1  #  Cộng tài nguyên cho người chơi

            #  Giảm số lần sử dụng thẻ xuống
            env[234] -= 1

            #  Nếu hết số lần sử dụng hoặc ngân hàng hết tài nguyên
            if env[234] == 0 or (env[48:53] == 0).all():
                after_useDev(env)

            return

        if env[233] == 3:  #  Đang dùng thẻ monopoly
            sum_res = 0
            res_idx = action - 59
            if env[48 + res_idx] > 0:
                res_thieu = 19 - env[48 + res_idx]
                if res_thieu > env[48 + res_idx]:
                    res_thieu = env[48 + res_idx]

                env[48 + res_idx] -= res_thieu
                env[s_ + res_idx] += res_thieu
            #  for i in range(1, 4):
            #      e_idx = (p_idx + i) % 4
            #      s_e = 58 + 42*e_idx
            #      sum_res += env[s_e+res_idx]
            #      env[s_e+res_idx] = 0

            #  env[s_+res_idx] += sum_res
            after_useDev(env)

            return

    if phase == 9:  #  Chọn các điểm mua thành phố
        #  Cập nhật tập nhà
        r_ = np.count_nonzero(env[s_ + 31 : s_ + 35] != -1)
        env[s_ + 31 + r_] = action

        #  Xóa nhà vừa được nâng cấp lên thành phố
        temp = env[s_ + 26 : s_ + 31]
        for i in range(5):
            if temp[i] == action:
                temp[i] = -1
                break

        temp = temp[temp != -1]
        env[s_ + 26 : s_ + 31] = -1
        env[s_ + 26 : s_ + 26 + len(temp)] = temp

        #  Cộng điểm
        env[s_ + 10] += 1

        env[229] = 6  #  Sang pha chọn mô đun

        return

    if phase == 12:  #  Lấy nguyên liệu ở trong kho trước hoặc sau khi đổ xúc xắc
        res = action - 59
        env[58 + 42 * p_idx + res] += 1  # Cộng thêm nguyên liệu cho người chơi
        env[254 + 5 * p_idx + res] -= 1  # Trừ nguyên liệu cho kho của người chơi
        env[245 + p_idx] = 0
        if env[228] == -1:
            env[229] = 2
            roll_xx(env)
        else:
            env[229] = 6

        return

    return


@njit()
def get_p_point_n_all_road(p_state: np.ndarray):
    p_point = np.zeros(54)
    all_road = np.zeros(72)

    temp = p_state[65:80].astype(np.int32)
    for road in temp:
        if road == -1:
            break
        else:
            all_road[road] = 1
            p_point[ROAD_POINT[road]] = 1

    for j in range(3):
        s_ = 96 + 29 * j

        temp = p_state[s_ + 3 : s_ + 18].astype(np.int32)
        for road in temp:
            if road == -1:
                break
            else:
                all_road[road] = 1

        temp = p_state[s_ + 18 : s_ + 23].astype(np.int32)
        for i in temp:
            if i == -1:
                break
            else:
                p_point[i] = 0

        temp = p_state[s_ + 23 : s_ + 27].astype(np.int32)
        for i in temp:
            if i == -1:
                break
            else:
                p_point[i] = 0

    return p_point, all_road


@njit()
def check_firstPoint(p_state: np.ndarray):
    p_point, all_road = get_p_point_n_all_road(p_state)
    list_point = np.where(p_point == 1)[0]
    for point in list_point:
        for road in POINT_ROAD[point]:
            if road != -1 and all_road[road] == 0:
                #   print('road', list_point)
                return True

    return False


@njit()
def check_useDev(p_state: np.ndarray, list_action: np.ndarray):
    #  Knight: Có là được dùng
    if p_state[191] == 1:
        list_action[55] = 1

    #  Roadbuilding: Còn đường và còn vị trí xây đường
    if p_state[192] == 1 and p_state[79] == -1 and check_firstPoint(p_state):
        list_action[56] = 1

    #  Yearofplenty: Ngân hàng có tài nguyên
    if p_state[193] == 1 and (p_state[48:53] == 1).any():
        list_action[57] = 1

    #  Monopoly: Có là được dùng
    if p_state[194] == 1:
        list_action[58] = 1


@njit()
def get_p_point_n_all_stm_city(p_state: np.ndarray):
    p_point = np.zeros(54)
    all_stm_and_city = np.zeros(54)

    temp = p_state[65:80].astype(np.int32)
    for i in temp:
        if i == -1:
            break
        else:
            p_point[ROAD_POINT[i]] = 1

    temp = p_state[80:85].astype(np.int32)
    for i in temp:
        if i == -1:
            break
        else:
            all_stm_and_city[i] = 1
            p_point[i] = 0

    temp = p_state[85:89].astype(np.int32)
    for i in temp:
        if i == -1:
            break
        else:
            all_stm_and_city[i] = 1
            p_point[i] = 0

    for j in range(3):
        s_ = 96 + 29 * j

        temp = p_state[s_ + 18 : s_ + 23].astype(np.int32)
        for i in temp:
            if i == -1:
                break
            else:
                all_stm_and_city[i] = 1
                p_point[i] = 0

        temp = p_state[s_ + 23 : s_ + 27].astype(np.int32)
        for i in temp:
            if i == -1:
                break
            else:
                all_stm_and_city[i] = 1
                p_point[i] = 0

    return p_point, all_stm_and_city


@njit()
def find_max_road(
    len_: int,
    diemXP: int,
    duongDaDi: np.ndarray,
    p_road: np.ndarray,
    p_point: np.ndarray,
):
    duongCanCheck = np.zeros(72)
    for road in POINT_ROAD[diemXP]:
        if road != -1 and p_road[road] == 1 and duongDaDi[road] == 0:
            duongCanCheck[road] = 1

    list_duongCanCheck = np.where(duongCanCheck == 1)[0]
    if len(list_duongCanCheck) == 0 or p_point[diemXP] == 0:
        if len_ > 0:
            return len_

    max_len = len_
    for road in list_duongCanCheck:
        duongDaDi_copy = duongDaDi.copy()
        duongDaDi_copy[road] = 1
        diemXP_1 = 9999
        for diem in ROAD_POINT[road]:
            if diem != diemXP:
                diemXP_1 = diem
                break

        len_1 = find_max_road(len_ + 1, diemXP_1, duongDaDi_copy, p_road, p_point)
        if len_1 > max_len:
            max_len = len_1

    return max_len


@njit()
def get_p_longest_road(env: np.ndarray, p_idx: int):
    s_ = 58 + 42 * p_idx
    p_road = np.zeros(72)
    p_full_point = np.zeros(54)

    temp = env[s_ + 11 : s_ + 26]
    list_road = temp[temp != -1].astype(np.int32)
    p_road[list_road] = 1
    p_full_point[ROAD_POINT[list_road].flatten()] = 1

    p_point = p_full_point.copy()
    for i in range(1, 4):
        e_idx = (p_idx + i) % 4
        s_e = 58 + 42 * e_idx
        temp = env[s_e + 26 : s_e + 35]
        stm_and_city = temp[temp != -1].astype(np.int32)
        p_point[stm_and_city] = 0

    duongDaDi = np.zeros(72)
    max_len = 0
    list_full_point = np.where(p_full_point == 1)[0]
    for point in list_full_point:
        len_ = find_max_road(0, point, duongDaDi, p_road, p_point)
        if len_ > max_len:
            max_len = len_

    return max_len


@njit()
def roll_xx(env: np.ndarray):
    p_idx = int(env[244])
    dice1 = np.random.randint(1, 7)
    dice2 = np.random.randint(1, 7)
    env[228] = dice1 + dice2  #  Cập nhật xx vào env

    if env[228] == 7:  #  Check xem ai phải bỏ bài
        #  Giả sử không ai bị chia bài, thì sẽ sang phase di chuyển Robber
        env[229] = 4
        for i in range(0, 4):
            e_idx = (p_idx + i) % 4
            s_e = 58 + 42 * e_idx
            if np.sum(env[s_e : s_e + 5]) > 7:  #  Thừa, sang pha bỏ bài
                env[229] = 3  #  Sang pha bỏ tài nguyên do bị chia
                #  Cập nhật số lượng tài nguyên phải bỏ
                env[232] = np.sum(env[s_e : s_e + 5]) // 2
                env[244] = e_idx  #  Đổi người chơi action
                env[235:239] = 0
                break

    else:  #  Trả tài nguyên từ ngân hàng
        temp = np.where(env[20:39] == env[228])[0]
        list_tile = temp[temp != env[19]]
        #  print('Ngân hàng trả tài nguyên', list_tile, env[list_tile])
        p_res_receive = np.zeros((4, 5))
        for i in range(4):
            s_i = 58 + 42 * i

            temp = env[s_i + 26 : s_i + 31]
            p_stm = temp[temp != -1].astype(np.int32)

            temp = env[s_i + 31 : s_i + 35]
            p_city = temp[temp != -1].astype(np.int32)

            for tile in list_tile:
                env_tile = int(env[tile])
                for stm in p_stm:
                    if stm in TILE_POINT[tile]:
                        p_res_receive[i][env_tile] += 1

                for city in p_city:
                    if city in TILE_POINT[tile]:
                        p_res_receive[i][env_tile] += 2

        total_res_receive = np.sum(p_res_receive, axis=0)

        #  Chỉ trả những loại tài nguyên mà ngân hàng có đủ
        temp = np.where(env[48:53] < total_res_receive)[0]
        p_res_receive[:, temp] = 0
        total_res_receive = np.sum(p_res_receive, axis=0)

        #  Trả tài nguyên
        env[48:53] -= total_res_receive  #  Trừ tài nguyên ngân hàng
        for i in range(4):
            s_i = 58 + 42 * i
            #  Cộng tài nguyên cho người chơi
            env[s_i : s_i + 5] += p_res_receive[i]

        #  Sang phase chọn mô đun
        env[229] = 6


@njit()
def update_new_stm(env: np.ndarray, new_stm_pos: int, s_: int):
    #  Cập nhật tập nhà
    r_ = np.count_nonzero(env[s_ + 26 : s_ + 31] != -1)
    env[s_ + 26 + r_] = new_stm_pos

    #  Cộng 1 điểm
    env[s_ + 10] += 1

    #  Kiểm tra cảng
    for port in range(9):
        if new_stm_pos in PORT_POINT[port]:
            if env[39 + port] == 5:
                env[s_ + 37 : s_ + 42] = np.minimum(3, env[s_ + 37 : s_ + 42])
            else:
                env[int(s_ + 37 + env[39 + port])] = 2

            break


@njit()
def useDev(env: np.ndarray, action: int, s_: int, p_idx: int):
    env[s_ + action - 50] -= 1  #  Giảm số lá dev của người chơi
    env[235:239] = 0  #  Mỗi turn chỉ được sử dụng 1 thẻ dev
    if action == 55:  #  Thẻ knight
        env[233] = 0
        env[234] = 1
        env[229] = 4  #  Chuyển sang pha di chuyển Robber
        env[s_ + 35] += 1

        list_p_usedKnight = env[np.array([93, 135, 177, 219])]
        largestArmy = np.max(list_p_usedKnight)

        #  Check lại danh hiệu
        if env[s_ + 35] >= 3 and env[s_ + 35] == largestArmy and env[226] != p_idx:
            #  Nếu cddn < 3 thì ko cần xét do ko thể giành danh hiệu
            #  Nếu cddn != longestRoad thì cũng ko cần xét do có xét cũng ko được danh hiệu
            #  Nếu đang có danh hiệu thì việc xây đường giúp củng cố danh hiệu, nên ko cần xét
            list_p_check = np.where(list_p_usedKnight == largestArmy)[0]
            if len(list_p_check) == 1:  #  Ghi nhận danh hiệu hoặc thay đổi danh hiệu
                if env[226] == -1:  #  Chưa có danh hiệu
                    env[226] = p_idx  #  Ghi nhận danh hiệu
                    env[s_ + 10] += 2  #  Cộng điểm
                else:  #  Thay đổi danh hiệu
                    #  Trừ điểm người cũ
                    env[int(58 + 42 * env[226] + 10)] -= 2
                    env[226] = p_idx  #  Ghi nhận người mới
                    env[s_ + 10] += 2  #  Cộng điểm người cũ
            #  Nếu có 2 người cùng có con đường dài nhất, không thay đổi danh hiệu

    elif action == 56:  #  Roadbuilding
        env[233] = 1
        env[234] = 2
        env[229] = 1  #  Chuyển về pha đặt đường

    elif action == 57:  #  Yearofplenty
        env[233] = 2
        env[234] = 2
        env[229] = 7

    elif action == 58:  #  Monopoly
        env[233] = 3
        env[234] = 1
        env[229] = 7


@njit()
def after_useDev(env: np.ndarray):
    env[233] = -1
    env[234] = 0
    if env[228] == -1:  #  Chưa roll xx
        env[229] = 2
        roll_xx(env)
    else:  #  Đã roll xx
        env[229] = 6


@njit()
def after_Rob(env: np.ndarray):
    if env[233] == 0:  #  Đang dùng thẻ knight
        after_useDev(env)
    else:  #  Vừa đổ ra 7
        env[229] = 6


@njit()
def weighted_random(p: np.ndarray):
    a = np.sum(p)
    b = np.random.uniform(0, a)
    for i in range(len(p)):
        b -= p[i]
        if b <= 0:
            return i


@njit()
def checkEnded(env: np.ndarray):
    #  Người chơi chỉ có thể chiến thắng khi tới lượt của mình
    #  Kể cả trường hợp: A có 9 điểm, B có con đường dài nhất,
    #  C có 9 điểm. C xây nhà phá đường của B, A được con đường
    #  dài nhất, có 11 điểm, C chỉ có 10 điểm nhưng vẫn chiến thắng
    #  mặc cho A có nhiều điểm hơn C
    ##
    main_p_idx = int(env[230]) % 4
    all_total_score = (
        env[np.array([68, 110, 152, 194])] + env[np.array([67, 109, 151, 193])]
    )

    if all_total_score[main_p_idx] > 9:
        list_more_than_9 = np.where(all_total_score > 9)[0]
        max_score = np.max(all_total_score)

        #  Chỉ có thể có tối đa 2 người chơi trên 9 điểm cùng lúc
        #  Nếu có 2 người chơi trên 9 điểm thì không thể có người chơi có 12 điểm
        if len(list_more_than_9) == 2:
            #  2 người chơi có cùng số điểm cao nhất
            if (all_total_score[list_more_than_9] == max_score).all():
                return main_p_idx + 100

            #  Người chơi chiến thắng có điểm cao nhất
            if all_total_score[main_p_idx] == max_score:
                return main_p_idx + 200

            #  Người chơi chiến thắng không có điểm cao nhất
            return main_p_idx + 300

        #  Đến đây thì tức là người chơi chính có điểm cao nhất
        if all_total_score[main_p_idx] == 12:
            return main_p_idx + 400

        #  Chỉ đơn giản là chiến thắng với 10 hoặc 11 điểm, không có gì đặc biệt
        return main_p_idx

    return -1


@njit()
def getReward(p_state: np.ndarray):
    if p_state[P_END_GAME_N] == 0:
        return -1
    else:
        if p_state[P_PLAYER + 10] > 9:  #  Hơn 9 điểm => auto thắng
            return 1

        else:
            return 0  #  Chưa có trên 9 điểm, chưa kết thúc game


@njit()
def getStateSize():
    return P_END_GAME_N + 1


@njit()
def getActionSize():
    return AMOUNT_ACTION


@njit()
def getAgentSize():
    return 4


@njit()
def random_Env(p_state, per):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], per


@njit()
def one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env = initEnv()

    winner = -1
    while env[230] < 1000:
        p_idx = int(env[244])
        state = getAgentState(env)
        if list_other[p_idx] == -1:
            action, per_player = p0(state, per_player)
        elif list_other[p_idx] == 1:
            action, per1 = p1(state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(state, per3)
        else:
            raise Exception("Sai list_other.")

        stepEnv(env, action)
        winner = checkEnded(env)
        if winner != -1:
            break

    env[280] = 1
    env[np.array([68, 110, 152, 194])] += env[np.array([67, 109, 151, 193])]

    env[229] = 2
    p0_idx = np.where(list_other == -1)[0][0]
    for p_idx in range(4):
        env[244] = p_idx
        state = getAgentState(env)
        if list_other[p_idx] == -1:
            action, per_player = p0(state, per_player)
        elif list_other[p_idx] == 1:
            action, per1 = p1(state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(state, per3)
        else:
            raise Exception("Sai list_other.")

    if p0_idx == winner:
        result = 1
    else:
        result = 0
    return result, per_player


@njit()
def n_games_numba(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_numba(
            p0, list_other, per_player, per1, per2, per3, p1, p2, p3
        )
        win += winner

    return win, per_player


def one_game_normal(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env = initEnv()

    winner = -1
    while env[230] < 1000:
        p_idx = int(env[244])
        state = getAgentState(env)
        if list_other[p_idx] == -1:
            action, per_player = p0(state, per_player)
        elif list_other[p_idx] == 1:
            action, per1 = p1(state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(state, per3)
        else:
            raise Exception("Sai list_other.")

        stepEnv(env, action)
        winner = checkEnded(env)
        if winner != -1:
            break

    env[280] = 1
    env[np.array([68, 110, 152, 194])] += env[np.array([67, 109, 151, 193])]

    env[229] = 2
    p0_idx = np.where(list_other == -1)[0][0]
    for p_idx in range(4):
        env[244] = p_idx
        state = getAgentState(env)
        if list_other[p_idx] == -1:
            action, per_player = p0(state, per_player)
        elif list_other[p_idx] == 1:
            action, per1 = p1(state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(state, per3)
        else:
            raise Exception("Sai list_other.")

    if p0_idx == winner:
        result = 1
    else:
        result = 0
    return result, per_player


def n_games_normal(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_normal(
            p0, list_other, per_player, per1, per2, per3, p1, p2, p3
        )
        win += winner

    return win, per_player


@njit()
def bot_lv0(state, perData):
    validActions = getValidActions(state)
    arr_action = np.where(validActions == 1)[0]
    idx = np.random.randint(0, arr_action.shape[0])
    return arr_action[idx], perData


import importlib.util
import json, sys

try:
    from env import SHORT_PATH
except:
    pass


def load_module_player(player, game_name=None):
    if game_name == None:
        spec = importlib.util.spec_from_file_location(
            "Agent_player", f"{SHORT_PATH}src/Agent/{player}/Agent_player.py"
        )
    else:
        spec = importlib.util.spec_from_file_location(
            "Agent_player", f"{SHORT_PATH}src/Agent/ifelse/{game_name}/{player}.py"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_agent(level, *args):
    num_bot = getAgentSize() - 1

    if "_level_" not in globals():
        global _level_
        _level_ = level
        init = True
    else:
        if _level_ != level:
            _level_ = level
            init = True
        else:
            init = False

    if init:
        global _list_per_level_
        global _list_bot_level_
        _list_per_level_ = []
        _list_bot_level_ = []

        if _level_ == 0:
            _list_per_level_ = [
                np.array([[0.0]], dtype=np.float64) for _ in range(num_bot)
            ]
            _list_bot_level_ = [bot_lv0 for _ in range(num_bot)]
        else:
            env_name = sys.argv[1]
            if len(args) > 0:
                dict_level = json.load(
                    open(f"{SHORT_PATH}src/Log/check_system_about_level.json")
                )
            else:
                dict_level = json.load(open(f"{SHORT_PATH}src/Log/level_game.json"))

            if str(_level_) not in dict_level[env_name]:
                raise Exception("Hiện tại không có level này")

            lst_agent_level = dict_level[env_name][str(level)][2]

            for i in range(num_bot):
                if level == -1:
                    module_agent = load_module_player(
                        lst_agent_level[i], game_name=env_name
                    )
                    _list_per_level_.append(module_agent.DataAgent())
                else:
                    data_agent_level = np.load(
                        f"{SHORT_PATH}src/Agent/{lst_agent_level[i]}/Data/{env_name}_{level}/Train.npy",
                        allow_pickle=True,
                    )
                    module_agent = load_module_player(lst_agent_level[i])
                    _list_per_level_.append(
                        module_agent.convert_to_test(data_agent_level)
                    )
                _list_bot_level_.append(module_agent.Test)

    return _list_bot_level_, _list_per_level_


@njit()
def check_run_under_njit(agent, perData):
    return True


def run(
    p0: any = bot_lv0,
    num_game: int = 100,
    per_player: np.ndarray = np.array([[0.0]]),
    level: int = 0,
    *args,
):
    num_bot = getAgentSize() - 1
    list_other = np.array([-1] + [i + 1 for i in range(num_bot)])
    try:
        check_njit = check_run_under_njit(p0, per_player)
    except:
        check_njit = False

    load_agent(level, *args)

    if check_njit:
        return n_games_numba(
            p0,
            num_game,
            per_player,
            list_other,
            _list_per_level_[0],
            _list_per_level_[1],
            _list_per_level_[2],
            _list_bot_level_[0],
            _list_bot_level_[1],
            _list_bot_level_[2],
        )
    else:
        return n_games_normal(
            p0,
            num_game,
            per_player,
            list_other,
            _list_per_level_[0],
            _list_per_level_[1],
            _list_per_level_[2],
            _list_bot_level_[0],
            _list_bot_level_[1],
            _list_bot_level_[2],
        )
