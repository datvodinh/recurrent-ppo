import warnings

# Building Card
import numpy as np
from numba import jit, njit

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
BUILDING_CARDS = np.array(
    [
        [10, 2, 1, 0, 0, 0, 0, 0],
        [11, 1, 2, 0, 0, 0, 0, 0],
        [11, 2, 0, 1, 0, 0, 0, 0],
        [12, 1, 1, 1, 0, 0, 0, 0],
        [12, 1, 1, 1, 0, 0, 0, 0],
        [12, 2, 0, 0, 1, 0, 0, 0],
        [13, 1, 1, 0, 1, 0, 0, 0],
        [13, 1, 1, 0, 1, 0, 0, 0],
        [13, 1, 0, 2, 0, 0, 0, 0],
        [13, 0, 2, 1, 0, 0, 0, 0],
        [14, 1, 0, 1, 1, 0, 0, 0],
        [14, 1, 0, 1, 1, 0, 0, 0],
        [14, 0, 2, 0, 1, 0, 0, 0],
        [14, 0, 1, 2, 0, 0, 0, 0],
        [15, 0, 1, 1, 1, 0, 0, 0],
        [15, 0, 1, 1, 1, 0, 0, 0],
        [16, 0, 0, 2, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 7, 0, 0],
        [0, 0, 0, 0, 0, 7, 0, 0],
        [0, 0, 0, 0, 0, 7, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 1],
        [0, 0, 0, 0, 0, 0, 4, 2],
        [0, 0, 0, 0, 0, 0, 4, 3],
        [0, 0, 0, 0, 0, 0, 4, 4],
        [0, 0, 0, 0, 0, 0, 5, 1],
        [0, 0, 0, 0, 0, 0, 5, 2],
        [0, 0, 0, 0, 0, 0, 5, 3],
        [0, 0, 0, 0, 0, 0, 5, 4],
    ]
)

CIV_CARDS = np.array(
    [
        [
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
        ],  # Xúc, đá
        [
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # Xúc, Xe
        [
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # Xúc, lu
        [
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # Xúc Đồng hồ
        [
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 1Thực, Dệt
        [
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 3Thực, Dệt
        [
            0,
            0,
            5,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 5Thực, Cây
        [
            0,
            0,
            7,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 7Thực, Lu
        [
            0,
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # Điểm, sáo
        [
            0,
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # Điểm, sáo
        [
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 2Bạc, xe
        [
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # Roll, Vàng, người
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # Cụ, người
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # Lúa, đồng hồ
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # 2NL, cây
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
        ],  # Nâng cấp, đá
        [
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
        ],  # Xúc, 1, lúa
        [
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
        ],  # Xúc, 1, nhà
        [
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            1,
            0,
            0,
        ],  # Xúc, 2, cụ
        [
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            1,
            0,
            0,
        ],  # Xúc, 2, cụ
        [
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            1,
        ],  # Xúc, 2, lúa
        [
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            1,
            0,
            0,
            0,
        ],  # Xúc, 2, nhà
        [
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            1,
            0,
            0,
            0,
        ],  # 2Thực, 2, nhà
        [
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            1,
        ],  # 3Thực, 2, lúa
        [
            0,
            0,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
        ],  # 4Thực, 1, nhà
        [
            0,
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            1,
            0,
            0,
            0,
        ],  # Điểm, 3, nhà
        [
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            1,
            0,
        ],  # gạch, 2, người
        [
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
        ],  # bạc, 1, lúa
        [
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
        ],  # bạc, 1, người
        [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
        ],  # vàng, 1, người
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            1,
            0,
            0,
        ],  # 2Cụ, 2, cụ
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
        ],  # 3Cụ, 1, cụ
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
        ],  # 4Cụ, 1, cụ
        [
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            1,
            0,
        ],  # Roll, gỗ, 2, người
        [
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
        ],  # Roll, bạc, 1, người
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    ]
)  # Lúc, 1, lua


BUILD_4_2 = np.array(
    [
        [0, 0, 1, 3],
        [0, 1, 0, 3],
        [2, 0, 0, 2],
        [0, 1, 3, 0],
        [2, 2, 0, 0],
        [0, 3, 1, 0],
        [3, 0, 1, 0],
        [0, 0, 3, 1],
        [0, 3, 0, 1],
        [3, 0, 0, 1],
        [3, 1, 0, 0],
        [0, 2, 2, 0],
        [1, 0, 3, 0],
        [1, 0, 0, 3],
        [1, 3, 0, 0],
        [0, 2, 0, 2],
        [0, 0, 2, 2],
        [2, 0, 2, 0],
    ]
)

BUILD_4_3 = np.array(
    [
        [0, 1, 1, 2],
        [1, 2, 0, 1],
        [0, 2, 1, 1],
        [0, 1, 2, 1],
        [2, 1, 0, 1],
        [1, 1, 2, 0],
        [1, 2, 1, 0],
        [2, 1, 1, 0],
        [1, 0, 2, 1],
        [2, 0, 1, 1],
        [1, 0, 1, 2],
        [1, 1, 0, 2],
    ]
)

BUILD_4_4 = np.array([[1, 1, 1, 1]])

BUILD_5_2 = np.array(
    [
        [0, 3, 2, 0],
        [4, 0, 0, 1],
        [0, 0, 2, 3],
        [0, 0, 1, 4],
        [3, 0, 0, 2],
        [4, 1, 0, 0],
        [2, 0, 3, 0],
        [3, 2, 0, 0],
        [0, 2, 3, 0],
        [1, 0, 4, 0],
        [1, 4, 0, 0],
        [4, 0, 1, 0],
        [0, 0, 3, 2],
        [0, 0, 4, 1],
        [0, 4, 0, 1],
        [2, 3, 0, 0],
        [0, 1, 4, 0],
        [2, 0, 0, 3],
        [0, 2, 0, 3],
        [0, 3, 0, 2],
        [0, 4, 1, 0],
        [3, 0, 2, 0],
        [1, 0, 0, 4],
        [0, 1, 0, 4],
    ]
)

BUILD_5_3 = np.array(
    [
        [1, 2, 2, 0],
        [3, 1, 1, 0],
        [1, 3, 0, 1],
        [0, 2, 2, 1],
        [1, 1, 0, 3],
        [0, 1, 2, 2],
        [0, 1, 3, 1],
        [1, 0, 3, 1],
        [1, 3, 1, 0],
        [2, 2, 0, 1],
        [1, 2, 0, 2],
        [3, 0, 1, 1],
        [2, 1, 0, 2],
        [0, 1, 1, 3],
        [2, 2, 1, 0],
        [0, 3, 1, 1],
        [3, 1, 0, 1],
        [2, 0, 1, 2],
        [1, 1, 3, 0],
        [1, 0, 1, 3],
        [0, 2, 1, 2],
        [2, 0, 2, 1],
        [2, 1, 2, 0],
        [1, 0, 2, 2],
    ]
)

BUILD_5_4 = np.array([[1, 1, 2, 1], [2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 1, 2]])


P_ID_PLAYER = 142
ID_START_PEOPLE = 318
ID_END_PEOPLE = 322
P_START_CIV = 322
P_END_CIV = 326
P_START_BUILD = 326
P_END_BUILD = 330
P_START_PUSH_RECENT = 354
P_END_PUSH_RECENT = 370
P_START_PULL_RECENT = 370
P_END_PULL_RECENT = 386
P_ALL_TOOL = 386
P_START_DICE_FOR_CIV = 387
P_END_DICE_FOR_CIV = 411
P_TOTAL_DICE = 411
P_PHASE = 412

E_ID_PLAYER = 211
TOTAL_INDEX_PLAYER = 44
TOTAL_CARD_STATE = 8
E_START_CIV = 387
E_END_CIV = 391
E_START_BUILD = 391
E_END_BUILD = 395
E_PUSH_RECENT = 395
E_PULL_RECENT = 396
E_ALL_TOOL = 397
E_START_DICE_FOR_CIV = 398
E_END_DICE_FOR_CIV = 402
E_TOTAL_DICE = 402
E_PHASE = 403


STATE_ENV_SIZE = E_PHASE + 11
STATE_PLAYER_SIZE = P_PHASE + 11
ALL_ACTION_SIZE = 68


@njit()
def initEnv(BUILDING_CARDS, CIV_CARDS):
    all_build_card = np.zeros((4, 7))
    all_build_card[:, 0] = 1
    all_build_card = all_build_card.flatten()

    all_civ_card = np.zeros(36)
    all_civ_card[:4] = 1

    env = np.zeros(STATE_ENV_SIZE)
    env[0] = 1  # Người chơi đầu tiên
    build_card = np.arange(28)
    np.random.shuffle(build_card)
    civ_card = np.arange(36)
    np.random.shuffle(civ_card)

    env[5:41] = civ_card
    env[41:69] = build_card
    env[69:73] = [100, 100, 100, 100]
    env[73:77] = 7
    env[83:179] = CIV_CARDS[civ_card[np.where(all_civ_card == 1)[0]]].flatten()
    env[179:211] = BUILDING_CARDS[
        build_card[np.where(all_build_card == 1)[0]]
    ].flatten()

    for i in range(4):
        p_id_env = 211 + TOTAL_INDEX_PLAYER * i
        env[p_id_env + 3] = 12
        env[p_id_env + 2] = 5
        env[p_id_env] = 400
    env[E_START_CIV:E_END_BUILD] = -1
    env[E_PHASE] = 1
    return env, all_build_card.reshape((4, 7)), all_civ_card


@njit()
def getAgentState(env):
    p_state = np.zeros(STATE_PLAYER_SIZE)
    p_state[:P_ID_PLAYER] = env[69:E_ID_PLAYER]  # Các thông tin ở trên bàn

    p_state[ID_START_PEOPLE:ID_END_PEOPLE] = env[0:4]  # đến người nào đang chơi
    p_idx = np.where(env[0:4] == 1)[0][0]

    infor_card_env = env[
        E_START_CIV:E_END_BUILD
    ]  # Người đang chơi là 0, người khác lần lượt là 1, 2, 3

    s_ = E_ID_PLAYER + TOTAL_INDEX_PLAYER * p_idx
    p_state[P_ID_PLAYER : P_ID_PLAYER + TOTAL_INDEX_PLAYER] = env[
        s_ : s_ + TOTAL_INDEX_PLAYER
    ]  # Thông tin của người đang chơi
    p_state[
        P_START_CIV + np.where(infor_card_env == p_idx)[0]
    ] = 1  # Thông tin thẻ của người đang chơi đã đặt, hoặc đã lấy
    #  print('Trong người chơi', np.where(infor_card_env == p_idx)[0], p_state[P_START_CIV:P_END_BUILD])

    for i in range(1, 4):
        p_other_idx = (p_idx + i) % 4
        s_o = E_ID_PLAYER + TOTAL_INDEX_PLAYER * p_other_idx
        p_state[
            P_ID_PLAYER
            + TOTAL_INDEX_PLAYER * i : P_ID_PLAYER
            + TOTAL_INDEX_PLAYER * (i + 1)
        ] = env[s_o : s_o + TOTAL_INDEX_PLAYER]

        p_state[
            P_START_CIV
            + TOTAL_CARD_STATE * i
            + np.where(infor_card_env == p_other_idx)[0]
        ] = 1

    p_state[P_START_PUSH_RECENT + int(env[E_PUSH_RECENT])] = 1
    p_state[P_START_PULL_RECENT + int(env[E_PULL_RECENT])] = 1
    p_state[P_ALL_TOOL] = env[E_ALL_TOOL]
    for i in range(4):
        if env[E_START_DICE_FOR_CIV + i] > 0:
            p_state[
                P_START_DICE_FOR_CIV + i * 6 + int(env[E_START_DICE_FOR_CIV + i]) - 1
            ] = 1

    p_state[P_TOTAL_DICE] = env[E_TOTAL_DICE]
    p_state[P_PHASE:] = env[E_PHASE:]

    return p_state.astype(np.float64)


@njit()
def RollDice(count_dice):
    total_of_dice = 0
    for i in range(count_dice):
        total_of_dice += np.random.randint(1, 7)
    return total_of_dice


@njit()
def RollDiceUseTool(env, e_idp):
    id_warehouse = int(env[E_PULL_RECENT])
    #  env[e_idp + 9:e_idp + 12] = np.abs(env[e_idp + 9:e_idp + 12]) #Trả lại công cụ như bình thường
    env[e_idp + 31 + id_warehouse] = 0
    #  env[e_idp + 15 + np.where(env[e_idp + 9:e_idp + 12] > 0)[0]] = 1
    if id_warehouse != 7:
        count_source_take = (env[E_TOTAL_DICE] + env[E_ALL_TOOL]) // id_warehouse
        env[e_idp + 5 + id_warehouse - 3] += count_source_take
        env[69 + id_warehouse - 3] -= count_source_take
    else:
        count_source_take = (env[E_TOTAL_DICE] + env[E_ALL_TOOL]) // 2
        env[e_idp + 3] += count_source_take
    ##print('Tổng số công cụ', env[E_ALL_TOOL], 'xúc xắc', env[E_TOTAL_DICE], 'Tổng', env[E_TOTAL_DICE] + env[E_ALL_TOOL])
    ##print('Nguyên liệu lấy', id_warehouse - 3, '---Lấy---:', count_source_take)
    env[E_TOTAL_DICE] = 0
    env[E_ALL_TOOL] = 0

    env[E_PHASE:] = 0  # Đổi phase
    env[E_PHASE + 2] = 1
    return env


@njit()
def RollDiceGetRes(count_dice, env, id_warehouse, e_idp):
    total_of_dice = 0
    for i in range(count_dice):
        total_of_dice += np.random.randint(1, 7)
    if id_warehouse != 7:  # Gỗ, gạch, bạc, vàng
        count_source_take = (total_of_dice) // id_warehouse
        env[e_idp + 5 + id_warehouse - 3] += count_source_take
        env[69 + id_warehouse - 3] -= count_source_take
    else:
        count_source_take = (total_of_dice) // 2
        env[e_idp + 3] += count_source_take
    ##print('Không có công cụ, Tổng số dice', total_of_dice, 'Nguyên liệu lấy', id_warehouse - 3, '---Lấy---:', count_source_take)
    return env


@njit()
def GetScoreEndGame(env):
    for id_score in range(4):
        e_o_idp = int(E_ID_PLAYER + TOTAL_INDEX_PLAYER * id_score)
        score_civ_yellow = (
            env[e_o_idp + 42] * env[e_o_idp + 1]
            + env[e_o_idp + 39] * env[e_o_idp + 4]
            + env[e_o_idp + 40] * np.sum(env[e_o_idp + 9 : e_o_idp + 12])
            + env[e_o_idp + 41] * env[e_o_idp + 2]
        )
        civ_player = env[e_o_idp + 22 : e_o_idp + 30]
        count_civ_player = civ_player[civ_player > 0]
        score_civ_green = len(count_civ_player) * len(count_civ_player)
        env[e_o_idp] += score_civ_yellow + score_civ_green
        env[e_o_idp] += np.sum(env[e_o_idp + 5 : e_o_idp + 9])
    return env


@njit()
def getAgentSize():
    return 4


@njit()
def getActionSize():
    return 68


@njit()
def getStateSize():
    return STATE_PLAYER_SIZE


@njit()
def checkEnded(env):
    if env[82] == 0:
        return np.array([-1])
    else:
        arr_score = env[np.array([211, 255, 299, 343])]
        return np.where(arr_score == np.max(arr_score))[0]


@njit()
def getValidActions(p_state):
    p_state = p_state.astype(np.int64)
    list_action = np.zeros(ALL_ACTION_SIZE)
    lst_phase = np.where(p_state[P_PHASE:] == 1)[0]
    if len(lst_phase) == 0:
        return list_action.astype(np.float64)
    if p_state[13] == 1:
        list_action[0] = 1
        return list_action.astype(np.float64)
    phase = lst_phase[0]
    s_ = P_ID_PLAYER

    p_push_recent = np.where(p_state[P_START_PUSH_RECENT:P_END_PUSH_RECENT] == 1)[0]
    p_pull_recent = np.where(p_state[P_START_PULL_RECENT:P_END_PULL_RECENT] == 1)[0]

    card_state = np.zeros(TOTAL_CARD_STATE)
    for i in range(4):
        card_state += p_state[
            P_START_CIV
            + TOTAL_CARD_STATE * i : P_START_CIV
            + TOTAL_CARD_STATE * (i + 1)
        ]
        #  print(p_state[P_START_CIV + TOTAL_CARD_STATE*i:P_START_CIV + TOTAL_CARD_STATE*(i+1)])

    if phase == 0:  # Chọn ô đặt người
        so_nguoi_co_the_dat = (
            p_state[s_ + 2]
            - np.sum(p_state[s_ + 31 : s_ + 39])
            - len(np.where(p_state[P_START_CIV:P_END_BUILD] == 1)[0])
        )
        #  print(card_state, p_state[s_ + 31:s_ + 39])
        civ_can_get = np.where(card_state[:4] == 0)[0]
        build_can_get = np.where(card_state[4:] == 0)[0]

        if so_nguoi_co_the_dat > 0:
            list_action[19 + civ_can_get] = 1
            list_action[23 + build_can_get] = 1

        people_in_warehouse = np.zeros(8)
        for i in range(4):
            people_in_warehouse += p_state[
                173 + TOTAL_INDEX_PLAYER * i : 181 + TOTAL_INDEX_PLAYER * i
            ]  # Số người ở mỗi ô

        list_action[11:13] = people_in_warehouse[0:2] == 0  # Lúa, cụ

        if so_nguoi_co_the_dat >= 2 and p_state[142 + 2] < 10:
            list_action[13] = people_in_warehouse[2] == 0  # Sinh sản

        for id_res, res in enumerate(p_state[s_ + 34 : s_ + 38]):
            if res == 0:
                list_action[14 + id_res] = (
                    people_in_warehouse[3 + id_res] < 7
                )  # Gỗ, bạc, gạch, vàng
        list_action[18] = 1  # Lương thực

    if phase == 1:  # Đặt số người
        so_nguoi_da_dat = np.sum(p_state[s_ + 31 : s_ + 39]) + len(
            np.where(p_state[P_START_CIV:P_END_BUILD] == 1)[0]
        )
        if len(p_push_recent) > 0:
            if p_push_recent[0] != 7:  # Gỗ gạch bạc vàng
                people_in_warehouse = np.zeros(8)
                for i in range(4):
                    people_in_warehouse += p_state[
                        s_
                        + 31
                        + TOTAL_INDEX_PLAYER * i : s_
                        + 39
                        + TOTAL_INDEX_PLAYER * i
                    ]
                count_people_can_push = np.minimum(
                    7 - people_in_warehouse, (p_state[s_ + 2]) - so_nguoi_da_dat
                )[int(p_push_recent[0])]
                ##print('Số người tối đa có thể đặt', count_people_can_push, np.sum(p_state[173:181]), 'tổng số người', p_state[s_ + 2], 'ô vừa đặt', p_state[P_PUSH_RECENT])
                list_action[1 : int(count_people_can_push + 1)] = 1
            else:  # Lương thực
                so_nguoi_co_the_dat = p_state[s_ + 2] - so_nguoi_da_dat
                list_action[1 : int(so_nguoi_co_the_dat + 1)] = 1

    if phase == 2:  # Lấy người từ các ô
        ##print(p_state[s_ + 31:s_ + 39], p_state[P_START_CIV:P_END_BUILD])
        list_action[
            29 + np.where(p_state[s_ + 31 : s_ + 39] > 0)[0]
        ] = 1  # Các ô nguyên liệu
        list_action[
            48 + np.where(p_state[P_START_CIV:P_END_CIV] == 1)[0]
        ] = 1  # Người đang có ở ô civ
        list_action[
            52 + np.where(p_state[P_START_BUILD:P_END_BUILD] == 1)[0]
        ] = 1  # Người đang có ở ô build
        if p_state[s_ + 30] != 0:
            list_action[63] = 1

    if phase == 3:  # Trả nguyên liệu mua thẻ civ
        list_action[40:44] = p_state[s_ + 5 : s_ + 9] > 0

    if phase == 4:  # Dùng công cụ(end hoặc hết thì qua roll xúc xắc)
        list_action[37:40] = p_state[s_ + 15 : s_ + 18] > 0
        list_action[44:47] = p_state[s_ + 12 : s_ + 15] > 0
        list_action[0] = 1

    if phase == 5:  # trả nguyên liệu khi mua thẻ build 1 -> 7
        list_action[40:44] = p_state[s_ + 5 : s_ + 9] > 0
        list_action[47] = 1

    if phase == 6:  # Chọn trừ nguyên liệu hoặc trừ điểm khi không đủ thức ăn
        list_action[27] = 1
        list_action[28] = 1

    if phase == 7:  # Chọn giá trị xúc xắc khi dùng thẻ civ xúc xắc
        #  dice_after = p_state[P_START_DICE_FOR_CIV:P_END_DICE_FOR_CIV].astype(np.int64)
        for i in range(4):
            dice_after = np.where(
                p_state[
                    P_START_DICE_FOR_CIV + i * 6 : P_START_DICE_FOR_CIV + (i + 1) * 6
                ]
                == 1
            )[0]
            if len(dice_after) > 0:
                #  print(p_state[P_START_DICE_FOR_CIV + i*6: P_START_DICE_FOR_CIV + (i+1)*6], int(dice_after[0]))
                list_action[56 + int(dice_after[0]) + 1] = 1

    if (
        phase == 8
    ):  # Lấy nguyên liệu từ ngân hàng khi dùng thẻ civ lấy 2 nguyên liệu bất kỳ
        list_action[64:68] = p_state[4:8] > 0

    if phase == 9:  # trả nguyên liệu khi mua thẻ civ có số lượng mặc định
        if len(p_pull_recent) > 0:
            id_build_card = int(p_pull_recent[0] - 12)
            all_infor_build_card = p_state[110:142].reshape((4, 8))
            build_card_take = all_infor_build_card[id_build_card]
            p_stock = p_state[s_ + 5 : s_ + 9]
            ##print('Thông tin thẻ build', build_card_take, 'res đã lấy', p_state[8:12], p_stock)

            card_civ_need = np.zeros(4)

            if build_card_take[6] == 4 and build_card_take[7] == 2:
                for case_card in BUILD_4_2:
                    if ((p_stock + p_state[8:12]) >= case_card).all() and (
                        case_card >= p_state[8:12]
                    ).all():
                        card_civ_need += case_card - p_state[8:12]

            elif build_card_take[6] == 4 and build_card_take[7] == 3:
                for case_card in BUILD_4_3:
                    if ((p_stock + p_state[8:12]) >= case_card).all() and (
                        case_card >= p_state[8:12]
                    ).all():
                        card_civ_need += case_card - p_state[8:12]

            elif build_card_take[6] == 4 and build_card_take[7] == 4:
                for case_card in BUILD_4_4:
                    if ((p_stock + p_state[8:12]) >= case_card).all() and (
                        case_card >= p_state[8:12]
                    ).all():
                        card_civ_need += case_card - p_state[8:12]

            elif build_card_take[6] == 5 and build_card_take[7] == 2:
                for case_card in BUILD_5_2:
                    if ((p_stock + p_state[8:12]) >= case_card).all() and (
                        case_card >= p_state[8:12]
                    ).all():
                        card_civ_need += case_card - p_state[8:12]

            elif build_card_take[6] == 5 and build_card_take[7] == 3:
                for case_card in BUILD_5_3:
                    if ((p_stock + p_state[8:12]) >= case_card).all() and (
                        case_card >= p_state[8:12]
                    ).all():
                        card_civ_need += case_card - p_state[8:12]

            elif build_card_take[6] == 5 and build_card_take[7] == 4:
                for case_card in BUILD_5_4:
                    if ((p_stock + p_state[8:12]) >= case_card).all() and (
                        case_card >= p_state[8:12]
                    ).all():
                        card_civ_need += case_card - p_state[8:12]

            list_action[40:44] = card_civ_need > 0

    if phase == 10:  # trả nguyên liệu nếu không đủ thức ăn
        list_action[40:44] = p_state[s_ + 5 : s_ + 9] > 0

    return list_action.astype(np.float64)


@njit()
def stepEnv(action, env, all_build_card, all_civ_card):
    action = int(action)
    phase = np.where(env[E_PHASE:] == 1)[0][0]
    idp = np.where(env[0:4] == 1)[0][0]
    e_idp = int(E_ID_PLAYER + TOTAL_INDEX_PLAYER * idp)

    check_end_pull_people = True
    if phase == 0:  # Đặt người
        #  print('action ở step', action)
        id_warehouse = action - 11
        if action in np.arange(14, 19):
            ##print('Vào action cần phải đặt số người')
            env[E_PHASE:] = 0  # Đổi phase
            env[E_PHASE + 1] = 1

            env[E_PUSH_RECENT] = id_warehouse

        else:
            if action == 13:
                env[e_idp + 31 + id_warehouse] += 2
            if action in [11, 12]:
                env[e_idp + 31 + id_warehouse] += 1
            if action in np.arange(19, 27):
                #  print('Đặt người lỗi')
                env[E_START_CIV + action - 19] = idp

            env[0:4] = 0  # Đổi người chơi
            check_people = True
            for i in range(1, 5):
                o_idp = (idp + i) % 4
                e_o_idp = int(211 + TOTAL_INDEX_PLAYER * o_idp)
                so_nguoi_da_dat = np.sum(env[e_o_idp + 31 : e_o_idp + 39]) + len(
                    np.where(env[E_START_CIV:E_END_BUILD] == o_idp)[0]
                )
                #  print('Người chơi', o_idp, 'đã đặt', 'Số người đã đặt', so_nguoi_da_dat, env[e_o_idp + 31:e_o_idp + 39], env[E_START_CIV:E_END_BUILD])
                if so_nguoi_da_dat < env[e_o_idp + 2]:
                    env[o_idp] = 1
                    ##print('Đổi sang người chơi', np.where(env[0:4] == 1)[0])
                    check_people = False
                    break
            if check_people == True:
                env[E_PHASE:] = 0  # Đổi phase
                env[E_PHASE + 2] = 1
                env[int(env[4]) % 4] = 1

    elif phase == 1:  # Đặt số người
        id_warehouse = int(env[E_PUSH_RECENT])
        env[e_idp + 31 + id_warehouse] += action
        ##print('Đặt', action, 'người')
        env[0:4] = 0  # Đổi người chơi
        check_people = True
        for i in range(1, 5):
            o_idp = (idp + i) % 4
            e_o_idp = int(211 + TOTAL_INDEX_PLAYER * o_idp)
            so_nguoi_da_dat = np.sum(env[e_o_idp + 31 : e_o_idp + 39]) + len(
                np.where(env[E_START_CIV:E_END_BUILD] == o_idp)[0]
            )
            ##print('Người chơi', o_idp, 'đã đặt', 'Số người đã đặt', so_nguoi_da_dat, env[e_o_idp + 31:e_o_idp + 39], env[E_START_CIV:E_END_BUILD])
            if so_nguoi_da_dat < env[e_o_idp + 2]:
                env[o_idp] = 1
                ##print('Đổi sang người chơi', np.where(env[0:4] == 1)[0][0])
                check_people = False
                break
        env[E_PHASE:] = 0  # Đổi phase
        if check_people == True:
            env[E_PHASE + 2] = 1
            env[int(env[4]) % 4] = 1
        else:
            env[E_PHASE] = 1

    elif phase == 2:  # Lấy người từ các ô
        if action in np.arange(29, 37):
            id_warehouse = action - 29

        elif action in np.arange(48, 56):
            id_warehouse = action - 48 + 8

        elif action == 63:
            id_warehouse = (
                -1
            )  # action lấy 2 nguyên liệu từ thẻ civ và sang phase trả nguyên liệu

        if id_warehouse in [0, 1, 2]:
            env[e_idp + 31 + id_warehouse] = 0

        if id_warehouse in [8, 9, 10, 11, 12, 13, 14, 15]:
            env[E_START_CIV + id_warehouse - 8] = -1

        if id_warehouse == -1:  # Chuyển sang lấy 2 nguyên liệu bất kỳ trên bàn
            env[E_PHASE:] = 0  # Đổi phase
            env[E_PHASE + 8] = 1

        if id_warehouse == 0:  # Thêm lúa
            env[e_idp + 1] += 1

        elif id_warehouse == 1:  # Thêm công cụ vào ô có số lượng công cụ bé nhất
            id_min = np.argmin(env[e_idp + 9 : e_idp + 12])
            env[e_idp + 9 + id_min] += 1
            if env[e_idp + 15 + id_min] == 0 and env[e_idp + 9 + id_min] == 1:
                env[e_idp + 15 + id_min] = 1
        elif id_warehouse == 2:  # Thêm số người
            env[e_idp + 2] += 1

        elif id_warehouse in [3, 4, 5, 6, 7]:  # Lấy tài nguyên
            env[E_PULL_RECENT] = id_warehouse
            if (
                np.sum(env[e_idp + 12 : e_idp + 15]) > 0
                or (env[e_idp + 15 : e_idp + 18] > 0).any()
            ):  # Nếu còn thẻ công cụ để dùng
                count_dice = int(env[e_idp + 31 + id_warehouse])
                env[E_TOTAL_DICE] = RollDice(count_dice)
                env[E_PHASE:] = 0  # Đổi phase
                env[E_PHASE + 4] = 1
            else:  # Không có công cụ thì roll luôn
                count_dice = int(env[e_idp + 31 + id_warehouse])
                env[e_idp + 31 + id_warehouse] = 0
                env = RollDiceGetRes(count_dice, env, id_warehouse, e_idp)

        elif id_warehouse in [8, 9, 10, 11]:  # Lấy thẻ civ
            env[E_PULL_RECENT] = id_warehouse
            if np.sum(env[e_idp + 5 : e_idp + 9]) >= (id_warehouse - 7):
                env[E_PHASE:] = 0  # Đổi phase
                env[E_PHASE + 3] = 1

                check_end_pull_people = False

        elif id_warehouse in [12, 13, 14, 15]:  # Lấy thẻ build
            env[E_PULL_RECENT] = id_warehouse

            id_build_card = id_warehouse - 12
            all_infor_build_card = env[179:211].reshape((4, 8))
            build_card_take = all_infor_build_card[id_build_card]
            ##print('Thông tin thẻ build', build_card_take)

            if (
                build_card_take[0] > 0
            ):  # Đây là card lấy điểm, trả nguyên liệu và cộng điểm luôn
                if (
                    env[e_idp + 5 : e_idp + 9] >= build_card_take[1:5]
                ).all():  # Nếu thừa nguyên liệu
                    env[e_idp + 5 : e_idp + 9] -= build_card_take[
                        1:5
                    ]  # Trừ nguyên liệu của bản thân
                    env[69:73] += build_card_take[1:5]  # Cộng nguyên liẹu cho ngân hàng
                    env[e_idp] += build_card_take[0]  # Cộng điểm
                    env[e_idp + 4] += 1  # Cộng thêm số nhà

                    env[
                        E_START_CIV + int(env[E_PULL_RECENT]) - 8
                    ] = -2  # Đánh dấu là đã lấy

            else:  # Đây là card đổi nguyên liệu lấy thẻ build
                if build_card_take[5] == 7:  # Nếu là thẻ trả 1-> nguyên liệu
                    if np.sum(env[e_idp + 5 : e_idp + 9]) > 0:
                        env[E_PHASE:] = 0  # Đổi phase
                        env[E_PHASE + 5] = 1

                        check_end_pull_people = False

                if (
                    build_card_take[6] > 0
                ):  # Nếu là thẻ trả số lượng nguyên liệu cố định
                    if (
                        np.sum(env[e_idp + 5 : e_idp + 9]) >= build_card_take[6]
                    ):  # Tổng số lượng nguyên liệu phải lớn hơn thẻ cần
                        check_phase = False
                        if build_card_take[6] == 4 and build_card_take[7] == 2:
                            for case_card in BUILD_4_2:
                                if (env[e_idp + 5 : e_idp + 9] >= case_card).all():
                                    check_phase = True
                                    break

                        elif build_card_take[6] == 4 and build_card_take[7] == 3:
                            for case_card in BUILD_4_3:
                                if (env[e_idp + 5 : e_idp + 9] >= case_card).all():
                                    check_phase = True
                                    break

                        elif build_card_take[6] == 4 and build_card_take[7] == 4:
                            for case_card in BUILD_4_4:
                                if (env[e_idp + 5 : e_idp + 9] >= case_card).all():
                                    check_phase = True
                                    break

                        elif build_card_take[6] == 5 and build_card_take[7] == 2:
                            for case_card in BUILD_5_2:
                                if (env[e_idp + 5 : e_idp + 9] >= case_card).all():
                                    check_phase = True
                                    break

                        elif build_card_take[6] == 5 and build_card_take[7] == 3:
                            for case_card in BUILD_5_3:
                                if (env[e_idp + 5 : e_idp + 9] >= case_card).all():
                                    check_phase = True
                                    break

                        elif build_card_take[6] == 5 and build_card_take[7] == 4:
                            for case_card in BUILD_5_4:
                                if (env[e_idp + 5 : e_idp + 9] >= case_card).all():
                                    check_phase = True
                                    break

                        if check_phase == True:
                            env[E_PHASE:] = 0  # Đổi phase
                            env[E_PHASE + 9] = 1

                            check_end_pull_people = False

    elif phase == 3:  # Trả nguyên liệu mua thẻ civ
        count_res_give = int(env[E_PULL_RECENT] - 7)
        id_card_civ = count_res_give - 1
        card_civ_id = 83 + 24 * id_card_civ

        res_giv = int(action - 40)
        env[77 + res_giv] += 1  # Các nguyên liệu đã trả trong turn trả nguyên liệu
        env[e_idp + 5 + res_giv] -= 1  # Trừ nguyên liệu của bản thân
        env[69 + res_giv] += 1  #  Cộng nguyên liệu cho ngân hàng
        ##print('Các nguyên liệu đã trả', env[77:81], 'cần trả', count_res_give, env[e_idp + 5:e_idp + 9])
        check_end_pull_people = False
        ##print('trong phase 3', check_end_pull_people)
        if (
            np.sum(env[77:81]) == count_res_give
        ):  # Khi đủ nguyên liệu thì mở thẻ civ mới, lấy thẻ cho người chơi, sang người chơi khác
            # print('Civ card lấy', env[5:41][np.where(all_civ_card == 1)[0][id_card_civ]])
            env[77:81] = 0  # Các nguyên liệu đã lấy gán lại bằng 0

            env[E_START_CIV + id_card_civ] = -2  # Đánh dấu là đã lấy

            env[E_PHASE:] = 0  # Đổi phase về phase 2
            env[E_PHASE + 2] = 1
            ##print('Vẫn trả được')
            check_end_pull_people = True

            env[e_idp + 22 : e_idp + 30] += env[
                card_civ_id + 11 : card_civ_id + 19
            ]  # Thêm thẻ văn minh
            env[e_idp + 39 : e_idp + 43] += (env[card_civ_id + 19]) * env[
                card_civ_id + 20 : card_civ_id + 24
            ]  # Thêm thẻ số người tính điểm cuối game

            env[e_idp + 1] += env[card_civ_id + 8]  # Thêm lúa
            env[e_idp + 3] += env[
                card_civ_id + 2
            ]  # Thẻ có thêm thức ăn(1,2, 3, 4, 5, 6, 7)

            if env[card_civ_id] == 0:
                if env[card_civ_id + 5] > 0:
                    env[e_idp + 5 + 2] += env[card_civ_id + 5]  # Thêm nguyên liệu bạc
                    env[69 + 2] -= env[card_civ_id + 5]
                if env[card_civ_id + 6] > 0:
                    env[e_idp + 5 + 3] += env[card_civ_id + 6]  # Thêm nguyên liệu vàng
                    env[69 + 3] -= env[card_civ_id + 6]

                if env[card_civ_id] == 0:
                    if env[card_civ_id + 4] > 0:
                        env[e_idp + 5 + 1] += env[
                            card_civ_id + 4
                        ]  # Thêm nguyên liệu gạch
                        env[69 + 1] -= env[card_civ_id + 4]

            env[e_idp] += env[card_civ_id + 3]  # Thêm điểm

            if env[card_civ_id + 7] == 1:  # Thêm công cụ
                id_min = np.argmin(env[e_idp + 9 : e_idp + 12])
                env[e_idp + 9 + id_min] += 1
                env[e_idp + 15 + np.where(env[e_idp + 9 : e_idp + 12] > 0)[0]] = 1

            if env[card_civ_id + 7] > 1:  # Thêm 2, 3, 4 công cụ dùng sau
                ##print('Thẻ có thêm công cụ', env[card_civ_id:card_civ_id+25])
                id_civ_tool_1_use = int(env[card_civ_id + 7]) - 2
                env[e_idp + 12 + id_civ_tool_1_use] = 1

            if env[card_civ_id] > 0:  # Thẻ roll 2 xúc xắc để lấy tài nguyên
                if env[card_civ_id + 4] > 0:  # Gỗ
                    res_take = 3
                elif env[card_civ_id + 5] > 0:  # Vàng
                    res_take = 5
                elif env[card_civ_id + 6] > 0:  # Gạch
                    res_take = 6

                if (
                    np.sum(env[e_idp + 12 : e_idp + 15]) > 0
                    or (env[e_idp + 9 : e_idp + 12] > 0).any()
                ):  # Nếu còn thẻ công cụ để dùng
                    env[E_TOTAL_DICE] = RollDice(2)
                    env[E_PULL_RECENT] = res_take
                    env[E_PHASE:] = 0  # Đổi phase
                    env[E_PHASE + 4] = 1
                    check_end_pull_people = False
                else:  # Không có công cụ thì roll luôn
                    env = RollDiceGetRes(2, env, res_take, e_idp)

            if env[card_civ_id + 9] == 2:  # thẻ lấy thêm 2 nguyên liệu
                check_end_pull_people = False
                env[e_idp + 30] = 2

            if env[card_civ_id + 1] == 1:
                check_end_pull_people = False
                for dice in range(4):
                    env[E_START_DICE_FOR_CIV + dice] = np.random.randint(1, 7)
                env[E_PHASE:] = 0  # Đổi phase
                env[E_PHASE + 7] = 1

            if (
                env[card_civ_id + 10] == 1
            ):  # Thẻ chọn thêm 1 thẻ civ mới từ chồng bài úp
                ##print('Thẻ lấy 1 thẻ mới từ ô văn minh')
                card_has_not_been_collected = np.where(all_civ_card == 0)[0]
                if len(card_has_not_been_collected) > 0:
                    id_card_top_most_close = int(card_has_not_been_collected[0])
                    ##print('Civ card lấy thêm', env[5:41][id_card_top_most_close])
                    all_civ_card[id_card_top_most_close] = -1

                    id_card_in_deck = int(env[5:41][id_card_top_most_close])
                    infor_card_top_most_close = CIV_CARDS[
                        id_card_in_deck
                    ]  # Lấy thông tin thẻ sẽ mở thêm

                    env[e_idp + 22 : e_idp + 30] += infor_card_top_most_close[
                        11:19
                    ]  # Thêm thẻ văn minh
                    env[e_idp + 39 : e_idp + 43] += (
                        infor_card_top_most_close[19]
                    ) * infor_card_top_most_close[
                        20:24
                    ]  # Thêm thẻ số người tính điểm cuối game

    elif phase == 4:  # Dùng công cụ(end hoặc hết thì qua roll xúc xắc)
        check_use_tool = False
        check_end_pull_people = False
        if action == 0:
            env = RollDiceUseTool(env, e_idp)
            check_end_pull_people = True

            ###Nếu còn có thể lấy người thì lấy, không thì sang người chơi khác
        else:
            if action in [37, 38, 39]:
                id_action = action - 37
                env[E_ALL_TOOL] += env[e_idp + 9 + id_action]
                #  env[e_idp + 9 + id_action] = - env[e_idp + 9 + id_action]
                env[e_idp + 15 + id_action] = 0
                ###Nếu còn có thể lấy công cụ thì lấy công cụ
                if (
                    np.sum(env[e_idp + 12 : e_idp + 15]) > 0
                    or (env[e_idp + 15 : e_idp + 18] > 0).any()
                ):
                    check_use_tool = True

            elif action in [44, 45, 46]:
                id_action = action - 44
                env[E_ALL_TOOL] += id_action + 2
                env[e_idp + 12 + id_action] = 0
                if (
                    np.sum(env[e_idp + 12 : e_idp + 15]) > 0
                    or (env[e_idp + 15 : e_idp + 18] > 0).any()
                ):
                    check_use_tool = True

            if check_use_tool == False:
                env = RollDiceUseTool(env, e_idp)
                check_end_pull_people = True

    elif (
        phase == 5
    ):  # trả nguyên liệu khi mua thẻ build trả nguyên liệu bất kỳ từ 1 -> 7
        if action != 47:
            res_giv = int(action - 40)
            env[77 + res_giv] += 1  # Các nguyên liệu đã trả trong turn trả nguyên liệu
            env[e_idp + 5 + res_giv] -= 1  # Trừ nguyên liệu của bản thân
            env[69 + res_giv] += 1  #  Cộng nguyên liệu cho ngân hàng

        check_end_pull_people = False
        if (
            np.sum(env[77:81]) == 7
            or action == 47
            or np.sum(env[e_idp + 5 : e_idp + 9]) == 0
        ):
            env[E_START_CIV + int(env[E_PULL_RECENT]) - 8] = -2  # Đánh dấu là đã lấy

            score_add = env[77:81] * (np.array([3, 4, 5, 6]))
            env[e_idp] += np.sum(score_add)  # Cộng thêm điểm
            env[e_idp + 4] += 1  # Cộng thêm số nhà

            env[E_PHASE:] = 0  # Đổi phase
            env[E_PHASE + 2] = 1

            check_end_pull_people = True

            env[77:81] = 0  # reset lại nguyên liệu đã trả

    elif phase == 6:  # Chọn trừ nguyên liệu hoặc trừ điểm khi không đủ thức ăn
        check_end_pull_people = False
        if action == 28:  # Chọn trừ điểm, sang người chơi khác
            check_end_pull_people = True
            env[e_idp] -= 10
            env[E_PHASE:] = 0  # Đổi phase
            env[E_PHASE + 2] = 1
            env[e_idp + 21] = 1  # Đánh dấu đã trả nguyên liệu

            # Đổi lại người chơi cuối cùng

            main_player = int(env[4]) % 4
            #  env[0:4] = 0 #Đổi người chơi
            id_move = int((main_player + 3) % 4)
        ##print('đổi người chươi ở phase 6', 'main_player', main_player, 'chuyển sang', int((main_player+3)%4))

        if action == 27:  # Chọn trừ nguyên liệu, sang phase 10
            env[E_PHASE:] = 0  # Đổi phase
            env[E_PHASE + 10] = 1

    elif phase == 7:  # Chọn giá trị xúc xắc khi dùng thẻ civ xúc xắc
        ##print('giá trị xúc xắc', env[E_START_DICE_FOR_CIV:E_END_DICE_FOR_CIV], 'Người chơi tiếp theo', idp)
        dice = action - 56
        #  print('dice', dice, env[E_START_DICE_FOR_CIV:E_END_DICE_FOR_CIV])
        if dice == 6:
            env[e_idp + 1] += 1  # Thêm lúa
        if dice == 1:
            env[e_idp + 5] += 1  # Thêm gỗ
            env[69] -= 1
        if dice == 2:
            env[e_idp + 6] += 1  # Thêm gạch
            env[69 + 1] -= 1
        if dice == 3:
            env[e_idp + 7] += 1  # Thêm bạc
            env[69 + 2] -= 1
        if dice == 4:
            env[e_idp + 8] += 1  # Thêm vàng
            env[69 + 3] -= 1
        if dice == 5:
            id_min = np.argmin(env[e_idp + 9 : e_idp + 12])
            env[e_idp + 15 + np.where(env[e_idp + 9 : e_idp + 12] > 0)[0]] = 1
            env[e_idp + 9 + id_min] += 1

        id_dice = np.where(env[E_START_DICE_FOR_CIV:E_END_DICE_FOR_CIV] == dice)[0][0]
        env[E_START_DICE_FOR_CIV + id_dice] = 0

        env[0:4] = 0  # Đổi người chơi
        o_idp = (idp + 1) % 4
        idp = o_idp
        env[o_idp] = 1
        e_idp = int(E_ID_PLAYER + TOTAL_INDEX_PLAYER * idp)

        check_end_pull_people = False
        if (env[E_START_DICE_FOR_CIV:E_END_DICE_FOR_CIV] == 0).all():
            check_end_pull_people = True
            env[E_PHASE:] = 0  # Đổi phase
            env[E_PHASE + 2] = 1

    elif (
        phase == 8
    ):  # Lấy nguyên liệu từ ngân hàng khi dùng thẻ civ lấy 2 nguyên liệu bất kỳ
        env[e_idp + 30] -= 1
        id_res = action - 64
        env[e_idp + 5 + id_res] += 1  # thêm nguyên liệu cho người chơi
        env[69 + id_res] -= 1  # Trừ nguyên liệu ở ngân hàng

        if env[e_idp + 30] == 0:
            env[E_PHASE:] = 0  # Đổi phase
            env[E_PHASE + 2] = 1

    elif phase == 9:  # trả nguyên liệu khi mua thẻ build có số lượng mặc định
        check_end_pull_people = False
        res_giv = int(action - 40)
        env[77 + res_giv] += 1  # Các nguyên liệu đã trả trong turn trả nguyên liệu
        env[e_idp + 5 + res_giv] -= 1  # Trừ nguyên liệu của bản thân
        env[69 + res_giv] += 1  #  Cộng nguyên liệu cho ngân hàng

        id_build_card = int(env[E_PULL_RECENT] - 12)
        all_infor_build_card = env[179:211].reshape((4, 8))
        build_card_take = all_infor_build_card[id_build_card]

        check_get_card = False
        if build_card_take[6] == 4 and build_card_take[7] == 2:
            for case_card in BUILD_4_2:
                if (env[77:81] == case_card).all():  # Lấy thẻ
                    check_get_card = True
                    break

        elif build_card_take[6] == 4 and build_card_take[7] == 3:
            for case_card in BUILD_4_3:
                if (env[77:81] == case_card).all():  # Lấy thẻ
                    check_get_card = True
                    break

        elif build_card_take[6] == 4 and build_card_take[7] == 4:
            for case_card in BUILD_4_4:
                if (env[77:81] == case_card).all():  # Lấy thẻ
                    check_get_card = True
                    break

        elif build_card_take[6] == 5 and build_card_take[7] == 2:
            for case_card in BUILD_5_2:
                if (env[77:81] == case_card).all():  # Lấy thẻ
                    check_get_card = True
                    break

        elif build_card_take[6] == 5 and build_card_take[7] == 3:
            for case_card in BUILD_5_3:
                if (env[77:81] == case_card).all():  # Lấy thẻ
                    check_get_card = True
                    break

        elif build_card_take[6] == 5 and build_card_take[7] == 4:
            for case_card in BUILD_5_4:
                if (env[77:81] == case_card).all():  # Lấy thẻ
                    check_get_card = True
                    break

        if check_get_card == True:
            check_end_pull_people = True

            score_add = env[77:81] * (np.array([3, 4, 5, 6]))
            env[e_idp] += np.sum(score_add)  # Cộng thêm điểm
            env[e_idp + 4] += 1  # Cộng thêm số nhà

            env[E_START_CIV + int(env[E_PULL_RECENT]) - 8] = -2  # Đánh dấu là đã lấy

            env[E_PHASE:] = 0  # Đổi phase
            env[E_PHASE + 2] = 1

            env[77:81] = 0

    elif phase == 10:  # trả nguyên liệu nếu không đủ thức ăn
        check_end_pull_people = False
        res_giv = int(action - 40)
        env[77 + res_giv] += 1  # Các nguyên liệu đã trả trong turn trả nguyên liệu
        env[e_idp + 5 + res_giv] -= 1  # Trừ nguyên liệu của bản thân
        env[69 + res_giv] += 1  #  Cộng nguyên liệu cho ngân hàng
        ##print('Số lượng nguyên liệu cần phải trả', env[81])
        if np.sum(env[77:81]) == env[81]:
            check_end_pull_people = True
            env[77:81] = 0

            env[E_PHASE:] = 0  # Đổi phase
            env[E_PHASE + 2] = 1

            env[e_idp + 21] = 1  # Đánh dấu đã trả nguyên liệu
            # Đổi lại người chơi cuối cùng

            main_player = int(env[4]) % 4
            #  env[0:4] = 0 #Đổi người chơi
            #  env[int((main_player+3)%4)] = 1
            id_move = int((main_player + 3) % 4)
            env[81] = 0
        ##print('đổi người chươi ở phase 10', 'main_player', main_player, 'chuyển sang', int((main_player+3)%4))

    #  Nếu hết người thì chuyển sang người chơi khác, không thì vẫn giữ người chơi dó
    #  Nếu tất cả mọi người hết người thì cộng thêm turn, turn này cũng chính là người chơi chính
    so_nguoi_da_dat = np.sum(env[e_idp + 31 : e_idp + 39]) + len(
        np.where(env[E_START_CIV:E_END_BUILD] == idp)[0]
    )

    ##print('Số người đã đặt', so_nguoi_da_dat)
    if check_end_pull_people == True:
        #  print(env[e_idp + 31:e_idp + 39], np.where(env[E_START_CIV:E_END_BUILD] == idp)[0], idp, env[E_START_CIV:E_END_BUILD])
        if so_nguoi_da_dat == 0 and phase != 1 and phase != 0:
            main_player = int(env[4]) % 4
            env[e_idp + 15 + np.where(env[e_idp + 9 : e_idp + 12] > 0)[0]] = 1
            env[0:4] = 0  # Đổi người chơi
            if env[e_idp + 21] == 1:
                idp = id_move
            ##print('Tại lúc trả nguyên liệu',  'main_player', main_player, 'chuyển sang', idp)
            if (idp + 1) % 4 == main_player:
                ##print('Xong 1 vòng')
                # Nếu là người cuối cùng lấy hết người thì trả thức ăn
                # Không đủ thức ăn thì chọn trừ nguyên liệu hoặc trừ 10 điểm
                # Không đủ nguyên liệu thì auto chọn trừ điểm
                check_tra_thuc_an = True
                for id_return in range(4):
                    e_id_return = int(E_ID_PLAYER + TOTAL_INDEX_PLAYER * id_return)
                    if env[e_id_return + 21] == 0:  # Chưa trả thức ăn
                        if (env[e_id_return + 3] + env[e_id_return + 1]) >= env[
                            e_id_return + 2
                        ]:
                            if (env[e_id_return + 2] - env[e_id_return + 1]) > 0:
                                env[e_id_return + 3] -= (
                                    env[e_id_return + 2] - env[e_id_return + 1]
                                )
                                env[e_id_return + 21] = 1
                        elif (
                            np.sum(env[e_id_return + 5 : e_id_return + 9])
                            + env[e_id_return + 3]
                            + env[e_id_return + 1]
                        ) < env[e_id_return + 2]:
                            env[e_id_return] -= 10  # Trừ điểm
                            env[e_id_return + 3] = 0
                            env[e_id_return + 21] = 1
                        else:
                            env[81] = (
                                env[e_id_return + 2]
                                - env[e_id_return + 3]
                                - env[e_id_return + 1]
                            )
                            env[e_id_return + 3] = 0
                            check_tra_thuc_an = False

                            # Đổi người chơi
                            env[0:4] = 0
                            env[id_return] = 1

                            # Đổi phase
                            env[E_PHASE:] = 0
                            env[E_PHASE + 6] = 1
                            break

                if (
                    check_tra_thuc_an == True
                ):  # Đã trả thức ăn hết cho mn, sang vòng mới, mở thẻ mới
                    ##print('Sang turn mới')

                    for id_return in range(
                        4
                    ):  # Trả lại trạng thái thành chưa trả thức ăn
                        e_id_return = int(E_ID_PLAYER + TOTAL_INDEX_PLAYER * id_return)
                        env[e_id_return + 21] = 0

                    env[4] += 1  # Sang vòng chơi tiếp theo

                    env[
                        int(env[4]) % 4
                    ] = 1  # Người chơi chính tiếp theo sẽ bắt đầu đặt người

                    # Đổi phase
                    env[E_PHASE:] = 0
                    env[E_PHASE] = 1

                    all_card_civ_open = np.where(all_civ_card == 1)[0]
                    id_da_lay = np.where(env[E_START_CIV:E_END_CIV] == -2)[0]
                    ##print('thẻ cũ', all_card_civ_open, 'ID dã lấy', id_da_lay)
                    all_civ_card[all_card_civ_open[id_da_lay]] = -1  # đã có người lấy
                    ##print(all_civ_card)

                    so_luong_can_mo_them = len(id_da_lay)
                    card_co_the_mo = np.where(all_civ_card == 0)[0]
                    if len(card_co_the_mo) < so_luong_can_mo_them:
                        env[82] = 1
                        env = GetScoreEndGame(env)

                    else:
                        all_civ_card[card_co_the_mo[:so_luong_can_mo_them]] = 1
                        civ_card = env[5:41]
                        ##print(np.where(all_civ_card == 1)[0])
                        id_civ_card_open = civ_card[
                            np.where(all_civ_card == 1)[0]
                        ].astype(np.int64)
                        env[E_START_CIV:E_END_CIV] = -1
                        env[83:179] = CIV_CARDS[id_civ_card_open].flatten()
                    ##print('all civ card', all_civ_card)
                    id_da_lay = np.where(env[E_START_BUILD:E_END_BUILD] == -2)[0]
                    if len(id_da_lay) > 0:
                        #  print('Lấy thẻ build', id_da_lay, all_build_card)
                        for id_deck_card in range(4):
                            if id_deck_card in id_da_lay:
                                id_card_open = np.where(
                                    all_build_card[id_deck_card] == 1
                                )[0]
                                if id_card_open == 6:
                                    env[82] = 1
                                    env = GetScoreEndGame(env)
                                    break
                                else:
                                    all_build_card[id_deck_card][id_card_open] = -1
                                    all_build_card[id_deck_card][id_card_open + 1] = 1
                                    all_build_card = all_build_card.flatten()
                                    build_card = env[41:69]
                                    id_build_card_open = build_card[
                                        np.where(all_build_card == 1)[0]
                                    ].astype(np.int64)
                                    env[179:211] = BUILDING_CARDS[
                                        id_build_card_open
                                    ].flatten()
                                    all_build_card = all_build_card.reshape((4, 7))
                        #  print(all_build_card)
                        id_build_desk, id_card_in_desk = np.where(all_build_card == 1)
                        env[73 + id_build_desk] = 7 - id_card_in_desk

                        env[E_START_BUILD:E_END_BUILD] = -1

            else:
                ##print('Đổi người chơi khác lấy người', env[E_START_CIV:E_END_BUILD], env[77:81])
                env[(idp + 1) % 4] = 1

                env[E_PHASE:] = 0
                env[E_PHASE + 2] = 1

    return env, all_build_card, all_civ_card


@njit()
def getReward(p_state):
    p_state = p_state.astype(np.int64)
    if p_state[13] == 0:
        return -1
    else:
        if p_state[P_ID_PLAYER] == np.max(p_state[np.array([142, 186, 230, 274])]):
            return 1
        else:
            return 0


@njit()
def one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env, all_build_card, all_civ_card = initEnv(BUILDING_CARDS, CIV_CARDS)
    _cc = 0
    while _cc <= 1000:
        idx = np.where(env[0:4] == 1)[0][0]
        player_state = getAgentState(env)

        if list_other[idx] == -1:
            action, per_player = p0(player_state, per_player)
            list_action = getValidActions(player_state)
            if list_action[action] != 1:
                raise Exception("Action không hợp lệ")
        elif list_other[idx] == 1:
            action, per1 = p1(player_state, per1)
        elif list_other[idx] == 2:
            action, per2 = p2(player_state, per2)
        elif list_other[idx] == 3:
            action, per3 = p3(player_state, per3)

        env, all_build_card, all_civ_card = stepEnv(
            action, env, all_build_card, all_civ_card
        )
        if checkEnded(env)[0] != -1:
            break

        _cc += 1

    env[82] = 1  # Gán cho biến kết thúc game
    for p_idx in range(4):
        env[83] = p_idx
        env[0:4] = 0
        env[p_idx] = 1
        p_state = getAgentState(env)
        if list_other[p_idx] == -1:
            act, per_player = p0(p_state, per_player)
        elif list_other[p_idx] == 1:
            action, per1 = p1(p_state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(p_state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(p_state, per3)
        else:
            raise Exception("Sai list_other.")

    winner = False

    if np.where(list_other == -1)[0][0] in checkEnded(env):
        winner = True
    else:
        winner = False

    return winner, per_player


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
    env, all_build_card, all_civ_card = initEnv(BUILDING_CARDS, CIV_CARDS)
    _cc = 0
    while _cc <= 1000:
        idx = np.where(env[0:4] == 1)[0][0]
        player_state = getAgentState(env)

        if list_other[idx] == -1:
            action, per_player = p0(player_state, per_player)
            list_action = getValidActions(player_state)
            if list_action[action] != 1:
                raise Exception("Action không hợp lệ")
        elif list_other[idx] == 1:
            action, per1 = p1(player_state, per1)
        elif list_other[idx] == 2:
            action, per2 = p2(player_state, per2)
        elif list_other[idx] == 3:
            action, per3 = p3(player_state, per3)

        env, all_build_card, all_civ_card = stepEnv(
            action, env, all_build_card, all_civ_card
        )
        if checkEnded(env)[0] != -1:
            break

        _cc += 1

    env[82] = 1  # Gán cho biến kết thúc game
    for p_idx in range(4):
        env[83] = p_idx
        env[0:4] = 0
        env[p_idx] = 1
        p_state = getAgentState(env)
        if list_other[p_idx] == -1:
            act, per_player = p0(p_state, per_player)
        elif list_other[p_idx] == 1:
            action, per1 = p1(p_state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(p_state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(p_state, per3)
        else:
            raise Exception("Sai list_other.")

    winner = False

    if np.where(list_other == -1)[0][0] in checkEnded(env):
        winner = True
    else:
        winner = False

    return winner, per_player


def n_games_normal(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_normal(
            p0, list_other, per_player, per1, per2, per3, p1, p2, p3
        )
        win += winner

    return win, per_player


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


@njit()
def bot_lv0(state, perData):
    validActions = getValidActions(state)
    arr_action = np.where(validActions == 1)[0]
    idx = np.random.randint(0, arr_action.shape[0])
    return arr_action[idx], perData


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
