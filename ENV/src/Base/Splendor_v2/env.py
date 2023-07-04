import random as rd

import numpy as np
from numba import jit, njit

#  normal_cards_infor = np.array([[0, 2, 2, 2, 0, 0, 0], [0, 2, 3, 0, 0, 0, 0], [0, 2, 1, 1, 0, 2, 1], [0, 2, 0, 1, 0, 0, 2], [0, 2, 0, 3, 1, 0, 1], [0, 2, 1, 1, 0, 1, 1], [1, 2, 0, 0, 0, 4, 0], [0, 2, 2, 1, 0, 2, 0], [0, 1, 2, 0, 2, 0, 1], [0, 1, 0, 0, 2, 2, 0], [0, 1, 1, 0, 1, 1, 1], [0, 1, 2, 0, 1, 1, 1], [0, 1, 1, 1, 3, 0, 0], [0, 1, 0, 0, 0, 2, 1], [0, 1, 0, 0, 0, 3, 0], [1, 1, 4, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 1, 1, 1, 2], [0, 0, 0, 0, 1, 2, 2], [0, 0, 1, 0, 0, 3, 1], [0, 0, 2, 0, 0, 0, 2], [0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 2, 1, 0, 0], [0, 4, 0, 2, 2, 1, 0], [0, 4, 1, 1, 2, 1, 0], [0, 4, 0, 1, 0, 1, 3], [1, 4, 0, 0, 4, 0, 0], [0, 4, 0, 2, 0, 2, 0], [0, 4, 2, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 0], [0, 4, 0, 3, 0, 0, 0], [0, 3, 1, 0, 2, 0, 0], [0, 3, 1, 1, 1, 0, 1], [1, 3, 0, 4, 0, 0, 0], [0, 3, 1, 2, 0, 0, 2], [0, 3, 0, 0, 3, 0, 0], [0, 3, 0, 0, 2, 0, 2], [0, 3, 3, 0, 1, 1, 0], [0, 3, 1, 2, 1, 0, 1], [1, 2, 0, 3, 0, 2, 2], [2, 2, 0, 2, 0, 1, 4], [1, 2, 3, 0, 2, 0, 3], [2, 2, 0, 5, 3, 0, 0], [2, 2, 0, 0, 5, 0, 0], [3, 2, 0, 0, 6, 0, 0], [3, 1, 0, 6, 0, 0, 0], [2, 1, 1, 0, 0, 4, 2], [2, 1, 0, 5, 0, 0, 0], [2, 1, 0, 3, 0, 0, 5], [1, 1, 0, 2, 3, 3, 0], [1, 1, 3, 2, 2, 0, 0], [3, 0, 6, 0, 0, 0, 0], [2, 0, 0, 0, 0, 5, 3], [2, 0, 0, 0, 0, 5, 0], [2, 0, 0, 4, 2, 0, 1], [1, 0, 2, 3, 0, 3, 0], [1, 0, 2, 0, 0, 3, 2], [3, 4, 0, 0, 0, 0, 6], [2, 4, 5, 0, 0, 3, 0], [2, 4, 5, 0, 0, 0, 0], [1, 4, 3, 3, 0, 0, 2], [1, 4, 2, 0, 3, 2, 0], [2, 4, 4, 0, 1, 2, 0], [1, 3, 0, 2, 2, 0, 3], [1, 3, 0, 0, 3, 2, 3], [2, 3, 2, 1, 4, 0, 0], [2, 3, 3, 0, 5, 0, 0], [2, 3, 0, 0, 0, 0, 5], [3, 3, 0, 0, 0, 6, 0], [4, 2, 0, 7, 0, 0, 0], [4, 2, 0, 6, 3, 0, 3], [5, 2, 0, 7, 3, 0, 0], [3, 2, 3, 3, 0, 3, 5], [3, 1, 3, 0, 3, 5, 3], [4, 1, 0, 0, 0, 0, 7], [5, 1, 0, 3, 0, 0, 7], [4, 1, 0, 3, 0, 3, 6], [3, 0, 0, 5, 3, 3, 3], [4, 0, 0, 0, 7, 0, 0], [5, 0, 3, 0, 7, 0, 0], [4, 0, 3, 3, 6, 0, 0], [5, 4, 0, 0, 0, 7, 3], [3, 4, 5, 3, 3, 3, 0], [4, 4, 0, 0, 0, 7, 0], [4, 4, 3, 0, 0, 6, 3], [3, 3, 3, 3, 5, 0, 3], [5, 3, 7, 0, 0, 3, 0], [4, 3, 6, 0, 3, 3, 0], [4, 3, 7, 0, 0, 0, 0]])
normal_cards_infor = np.array(
    [
        [0, 0, 0, 1, 0, 0, 2, 2, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 2],
        [0, 0, 0, 1, 0, 0, 0, 3, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 1, 0, 0, 2, 1, 0, 2, 0],
        [0, 0, 1, 0, 0, 0, 2, 0, 2, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 2, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 1, 1, 3, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0],
        [1, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 2],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 3, 1],
        [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 2, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 3],
        [1, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 2],
        [0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 2],
        [0, 0, 0, 0, 1, 0, 3, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 2, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 3, 0, 2, 2],
        [2, 0, 0, 1, 0, 0, 0, 2, 0, 1, 4],
        [1, 0, 0, 1, 0, 0, 3, 0, 2, 0, 3],
        [2, 0, 0, 1, 0, 0, 0, 5, 3, 0, 0],
        [2, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0],
        [3, 0, 0, 1, 0, 0, 0, 0, 6, 0, 0],
        [3, 0, 1, 0, 0, 0, 0, 6, 0, 0, 0],
        [2, 0, 1, 0, 0, 0, 1, 0, 0, 4, 2],
        [2, 0, 1, 0, 0, 0, 0, 5, 0, 0, 0],
        [2, 0, 1, 0, 0, 0, 0, 3, 0, 0, 5],
        [1, 0, 1, 0, 0, 0, 0, 2, 3, 3, 0],
        [1, 0, 1, 0, 0, 0, 3, 2, 2, 0, 0],
        [3, 1, 0, 0, 0, 0, 6, 0, 0, 0, 0],
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 5, 3],
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0],
        [2, 1, 0, 0, 0, 0, 0, 4, 2, 0, 1],
        [1, 1, 0, 0, 0, 0, 2, 3, 0, 3, 0],
        [1, 1, 0, 0, 0, 0, 2, 0, 0, 3, 2],
        [3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 6],
        [2, 0, 0, 0, 0, 1, 5, 0, 0, 3, 0],
        [2, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 3, 3, 0, 0, 2],
        [1, 0, 0, 0, 0, 1, 2, 0, 3, 2, 0],
        [2, 0, 0, 0, 0, 1, 4, 0, 1, 2, 0],
        [1, 0, 0, 0, 1, 0, 0, 2, 2, 0, 3],
        [1, 0, 0, 0, 1, 0, 0, 0, 3, 2, 3],
        [2, 0, 0, 0, 1, 0, 2, 1, 4, 0, 0],
        [2, 0, 0, 0, 1, 0, 3, 0, 5, 0, 0],
        [2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 5],
        [3, 0, 0, 0, 1, 0, 0, 0, 0, 6, 0],
        [4, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0],
        [4, 0, 0, 1, 0, 0, 0, 6, 3, 0, 3],
        [5, 0, 0, 1, 0, 0, 0, 7, 3, 0, 0],
        [3, 0, 0, 1, 0, 0, 3, 3, 0, 3, 5],
        [3, 0, 1, 0, 0, 0, 3, 0, 3, 5, 3],
        [4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7],
        [5, 0, 1, 0, 0, 0, 0, 3, 0, 0, 7],
        [4, 0, 1, 0, 0, 0, 0, 3, 0, 3, 6],
        [3, 1, 0, 0, 0, 0, 0, 5, 3, 3, 3],
        [4, 1, 0, 0, 0, 0, 0, 0, 7, 0, 0],
        [5, 1, 0, 0, 0, 0, 3, 0, 7, 0, 0],
        [4, 1, 0, 0, 0, 0, 3, 3, 6, 0, 0],
        [5, 0, 0, 0, 0, 1, 0, 0, 0, 7, 3],
        [3, 0, 0, 0, 0, 1, 5, 3, 3, 3, 0],
        [4, 0, 0, 0, 0, 1, 0, 0, 0, 7, 0],
        [4, 0, 0, 0, 0, 1, 3, 0, 0, 6, 3],
        [3, 0, 0, 0, 1, 0, 3, 3, 5, 0, 3],
        [5, 0, 0, 0, 1, 0, 7, 0, 0, 3, 0],
        [4, 0, 0, 0, 1, 0, 6, 0, 3, 3, 0],
        [4, 0, 0, 0, 1, 0, 7, 0, 0, 0, 0],
    ]
)
noble_cards_infor = np.array(
    [
        [0, 4, 4, 0, 0],
        [3, 0, 3, 3, 0],
        [3, 3, 3, 0, 0],
        [3, 0, 0, 3, 3],
        [0, 3, 0, 3, 3],
        [4, 0, 4, 0, 0],
        [4, 0, 0, 4, 0],
        [0, 3, 3, 0, 3],
        [0, 4, 0, 0, 4],
        [0, 0, 0, 4, 4],
    ]
)

P_STATE_SIZE = 221
P_ACTION_SIZE = 42
P_SCORE = 17

P_END_BOARD_INFOR = 6
P_START_AGENT_INFOR = 6
P_END_AGENT_INFOR = 18
P_START_CARD_NORMAL = 18
P_END_CARD_NORMAL = 150
P_START_CARD_NOBLE = 150
P_END_CARD_NOBLE = 175
P_START_CARD_UPSIDE_DOWN = 175
P_END_CARD_UPSIDE_DOWN = 208
P_START_GET_RES = 208
P_END_GET_RES = 213
P_START_OTHER_SCORE = 213
P_END_OTHER_SCORE = 216
P_START_UPSIDE_DOWN_HIDE_CARD = 216
P_END_UPSIDE_DOWN_HIDE_CARD = 219
P_COUNT_CAN_UPSIDE_DOWN = 219
P_CLOSE_GAME = 220

P_START_STOCK = 6
P_END_STOCK = 11
P_YELLOW_STOCK = 11
P_START_STOCK_COUNT = 12
P_END_STOCK_COUNT = 17
TOTAL_INFOR_NORMAL_CARD = 11


@njit()
def initEnv():
    env_state = np.full(164, 0)
    env_state[:] = 0
    env_state[101:107] = np.array([7, 7, 7, 7, 7, 5])
    lv1 = np.arange(40)
    lv2 = np.arange(40, 70)
    lv3 = np.arange(70, 90)
    nob = np.arange(90, 100)
    for lv in [lv1, lv2, lv3]:
        np.random.shuffle(lv)
        env_state[lv[:4]] = 5
    np.random.shuffle(nob)
    env_state[nob[:5]] = 5
    env_state[161] = lv1[4]
    env_state[162] = lv2[4]
    env_state[163] = lv3[4]

    return env_state, lv1, lv2, lv3


@njit()
def get_list_id_card_on_lv(lv):  # Get id card
    if len(lv) >= 4:
        return lv[:4]
    else:
        return lv[: len(lv)]


@njit()
def checkEnded(env_state):
    score_arr = np.array([env_state[118 + 12 * p_id] for p_id in range(4)])
    max_score = np.max(score_arr)
    if max_score >= 15 and env_state[100] % 4 == 0:
        lst_p = np.where(score_arr == max_score)[0] + 1
        if len(lst_p) == 1:
            return lst_p[0]
        else:
            lst_p_c = []
            for p_id in lst_p:
                lst_p_c.append(np.count_nonzero(env_state[:90] == p_id))

            lst_p_c = np.array(lst_p_c)
            min_p_c = np.min(lst_p_c)
            lst_p_win = np.where(lst_p_c == min_p_c)[0]
            if len(lst_p_win) == 1:
                return lst_p[lst_p_win[0]]
            else:
                id_max = -1
                a = -1
                for i in lst_p_win:
                    b = max(np.where(env_state[:90] == lst_p[i])[0])
                    if b > a:
                        id_max = lst_p[i]
                        a = b

                return id_max

    else:
        return 0


@njit()
def concatenate_all_lv_card(lv1, lv2, lv3):
    card_lv1 = normal_cards_infor[get_list_id_card_on_lv(lv1)]
    card_lv2 = normal_cards_infor[get_list_id_card_on_lv(lv2)]
    card_lv3 = normal_cards_infor[get_list_id_card_on_lv(lv3)]
    list_open_card = np.append(card_lv1, card_lv2)
    list_open_card = np.append(list_open_card, card_lv3)
    return list_open_card


@njit()
def get_id_card_normal_in_lv(lv1, lv2, lv3):
    list_card_normal_on_board = np.append(
        get_list_id_card_on_lv(lv1), get_list_id_card_on_lv(lv2)
    )
    list_card_normal_on_board = np.append(
        list_card_normal_on_board, get_list_id_card_on_lv(lv3)
    )
    return list_card_normal_on_board


@njit()
def getAgentState(env_state, lv1, lv2, lv3):
    p_id = env_state[100] % 4  # Lấy người đang chơi
    b_infor = env_state[101:107]  #  Lấy 6 loại nguyên liệu của bàn chơi
    p_infor = env_state[
        107 + 12 * p_id : 119 + 12 * p_id
    ]  # Lấy thông tin người đang chơi, 6 nguyên liệu trên bàn, 5 nguyên liệu mặc định, điểm

    list_open_card = concatenate_all_lv_card(
        lv1, lv2, lv3
    )  # Lấy list thẻ normal đang mở trên bàn
    list_open_noble = noble_cards_infor[
        np.where(env_state[90:100] == 5)
    ].flatten()  # Lấy list thẻ Noble đang mở trên bàn

    state_card_normal = np.full(132, 0)
    state_card_noble = np.full(25, 0)
    state_card_normal[: len(list_open_card)] = list_open_card
    state_card_noble[: len(list_open_noble)] = list_open_noble

    list_upside_down_card = normal_cards_infor[np.where(env_state[:90] == -(p_id + 1))]
    p_upside_down_card = np.full(33, 0)
    if len(list_upside_down_card) > 0:
        array_hide_card = list_upside_down_card.flatten()
        p_upside_down_card[: len(array_hide_card)] = array_hide_card

    st_getting = env_state[155:160]  # Lấy thông tin 5 nguyên liệu đang lấy trong turn
    other_scores = [
        env_state[118 + 12 * id_other_player]
        for id_other_player in range(4)
        if id_other_player != p_id
    ]  # Lấy điểm của người chơi khác

    p_state = np.zeros(P_STATE_SIZE)
    p_state[:P_END_BOARD_INFOR] = b_infor
    p_state[P_START_AGENT_INFOR:P_END_AGENT_INFOR] = p_infor
    p_state[
        P_START_CARD_NORMAL:P_END_CARD_NORMAL
    ] = state_card_normal  # Lấy thông tin 12 thẻ đang mở ở trên bàn
    p_state[P_START_CARD_NOBLE:P_END_CARD_NOBLE] = state_card_noble
    p_state[
        P_START_CARD_UPSIDE_DOWN:P_END_CARD_UPSIDE_DOWN
    ] = p_upside_down_card  # Lấy thông tin 3 thẻ đang úp
    p_state[
        P_START_GET_RES:P_END_GET_RES
    ] = st_getting  # Lấy thông tin 5 nguyên liệu đang lấy trong turn
    p_state[
        P_START_OTHER_SCORE:P_END_OTHER_SCORE
    ] = other_scores  # Lấy điểm của người chơi khác
    p_state[P_START_UPSIDE_DOWN_HIDE_CARD:P_END_UPSIDE_DOWN_HIDE_CARD] = (
        env_state[161:164] != 100
    ) * 1  # Lấy thông tin của các thẻ ẩn có thẻ úp, nếu có thể úp thì là 1
    p_state[P_COUNT_CAN_UPSIDE_DOWN] = len(
        np.where(env_state[:90] == 5)[0]
    )  # Số lượng thẻ có thể úp trong bàn

    cls_game = int(checkEnded(env_state))
    if cls_game == 0:
        p_state[P_CLOSE_GAME] = 0
    else:
        p_state[P_CLOSE_GAME] = 1
    return p_state.astype(np.float64)


@njit()
def getReward(p_state):
    scores = p_state[P_START_OTHER_SCORE:P_END_OTHER_SCORE]
    owner_score = p_state[P_SCORE]

    if p_state[P_CLOSE_GAME] == 0:
        return -1
    if owner_score >= 15 and max(scores) <= owner_score:
        return 1
    if max(scores) >= 15 and max(scores) > owner_score:
        return 0
    if p_state[P_CLOSE_GAME] == 1:
        return 0


@njit()
def get_remove_card_on_lv_and_add_new_card(
    env_state, lv, p_id, id_card_hide, type_action, card_id
):
    if type_action == 2:
        env_state[lv[4]] = -(p_id + 1)
        id_card_in_level = 4
    else:
        if len(lv) > 4:
            env_state[lv[4]] = 5
        id_card_in_level = np.where(lv == card_id)[0][0]
        if type_action == 1:
            env_state[card_id] = p_id + 1

    lv = np.delete(lv, id_card_in_level)
    if len(lv) > 4:
        env_state[id_card_hide] = lv[4]
    else:
        env_state[id_card_hide] = 100
    return env_state, lv


@njit()
def stepEnv(action, env_state, lv1, lv2, lv3):
    p_id = env_state[100] % 4
    cur_p = env_state[107 + 12 * p_id : 119 + 12 * p_id]
    b_stocks = env_state[101:107]

    if action == 0:
        env_state[100] += 1  # Sang turn mới
        env_state[155:160] = [0, 0, 0, 0, 0]
    else:
        if 1 <= action and action < 16:  # Mua thẻ
            if 1 <= action and action < 13:
                id_action = action - 1
                id_card_normal = get_id_card_normal_in_lv(lv1, lv2, lv3)
            else:
                id_action = action - 13
                id_card_normal = np.where(env_state[:90] == -(p_id + 1))[0]
            card_id = id_card_normal[id_action]
            card_infor = normal_cards_infor[card_id]
            card_price = card_infor[-5:]
            nl_bo_ra = (card_price > cur_p[6:11]) * (card_price - cur_p[6:11])
            nl_bt = np.minimum(nl_bo_ra, cur_p[:5])
            nl_auto = np.sum(nl_bo_ra - nl_bt)

            #  Trả nguyên liệu
            cur_p[:5] -= nl_bt
            cur_p[5] -= nl_auto
            b_stocks[:5] += nl_bt
            b_stocks[5] += nl_auto

            x_ = env_state[card_id]
            env_state[card_id] = p_id + 1
            if x_ == 5:  # Type_action == 1
                if card_id < 40:
                    env_state, lv1 = get_remove_card_on_lv_and_add_new_card(
                        env_state, lv1, p_id, 161, 1, card_id
                    )
                elif card_id >= 40 and card_id < 70:
                    env_state, lv2 = get_remove_card_on_lv_and_add_new_card(
                        env_state, lv2, p_id, 162, 1, card_id
                    )
                    env_state[card_id] = p_id + 1
                else:
                    env_state, lv3 = get_remove_card_on_lv_and_add_new_card(
                        env_state, lv3, p_id, 163, 1, card_id
                    )

            stock_count_card = np.where(card_infor[1:6] == 1)[0][0]

            cur_p[6:11][stock_count_card] += 1  # const_stock
            cur_p[11] += card_infor[0]  # Score

            #  Check Noble
            noble_lst = []
            nobles = [i for i in range(90, 100) if env_state[:100][i] == 5]
            for noble_id in nobles:
                if (noble_cards_infor[noble_id - 90][-5:] <= cur_p[6:11]).all():
                    noble_lst.append(noble_id)

            for noble_id in noble_lst:
                env_state[noble_id] = p_id + 1
                cur_p[11] += 3

            env_state[100] += 1  #  Sang turn mới

        elif 16 <= action and action < 31:  #  Úp thẻ có trên bàn
            id_action = action - 16
            #  print('Chon lay the', id_action)
            if b_stocks[5] > 0:
                b_stocks[5] -= 1
                cur_p[5] += 1
            if id_action == 12:  # Úp thẻ ẩn cấp 1
                env_state, lv1 = get_remove_card_on_lv_and_add_new_card(
                    env_state, lv1, p_id, 161, 2, 0
                )
            elif id_action == 13:  # Úp thẻ ẩn cấp 2
                env_state, lv2 = get_remove_card_on_lv_and_add_new_card(
                    env_state, lv2, p_id, 162, 2, 0
                )
            elif id_action == 14:  # Úp thẻ ẩn cấp 3
                env_state, lv3 = get_remove_card_on_lv_and_add_new_card(
                    env_state, lv3, p_id, 163, 2, 0
                )
            else:  # úp thẻ bình thường trên bàn
                id_card_normal = get_id_card_normal_in_lv(lv1, lv2, lv3)
                card_id = id_card_normal[id_action]
                env_state[card_id] = -(p_id + 1)
                if card_id < 40:
                    env_state, lv1 = get_remove_card_on_lv_and_add_new_card(
                        env_state, lv1, p_id, 161, 3, card_id
                    )
                elif card_id >= 40 and card_id < 70:
                    env_state, lv2 = get_remove_card_on_lv_and_add_new_card(
                        env_state, lv2, p_id, 162, 3, card_id
                    )
                else:
                    env_state, lv3 = get_remove_card_on_lv_and_add_new_card(
                        env_state, lv3, p_id, 163, 3, card_id
                    )

            if np.sum(cur_p[:6]) <= 10:
                env_state[100] += 1  #  Sang turn mới

        elif 31 <= action and action < 36:  # Lấy nguyên liệu
            check_phase3 = False
            taken = env_state[155:160]  # Các nguyên liệu đang lấy
            id_action = action - 31  # Id loại nguyên liệu lấy trong turn
            b_stocks[id_action] -= 1  # Trừ nguyên liệu bàn chơi
            cur_p[id_action] += 1  # thêm nguyên liệu của người chơi
            taken[id_action] += 1  #  thêm nguyên liệu lấy trong turn
            #  print('Lấy nguyên liệu:', id_action, 'Taken:',taken)
            s_taken = np.sum(taken)
            env_state[155:160] = taken

            if s_taken == 1:  #  Chỉ còn đúng loại nl vừa lấy nhưng sl < 3
                if (
                    b_stocks[id_action] < 3
                    and (np.sum(b_stocks[:5]) - b_stocks[id_action]) == 0
                ):
                    check_phase3 = True
            elif s_taken == 2:  #  Lấy double, hoặc không còn nl nào khác 2 cái vừa lấy
                if (
                    np.max(taken) == 2
                    or (np.sum(b_stocks[:5]) - np.sum(b_stocks[np.where(taken > 0)[0]]))
                    == 0
                ):
                    check_phase3 = True
            else:  #  sum(taken) = 3
                check_phase3 = True

            if check_phase3:
                env_state[155:160] = [0, 0, 0, 0, 0]
                if np.sum(cur_p[:6]) <= 10:
                    env_state[100] += 1  #  Sang turn mới

        elif 36 <= action and action < 42:  # Trả nguyên liệu
            st_ = action - 36
            cur_p[st_] -= 1
            b_stocks[st_] += 1

            if np.sum(cur_p[:6]) <= 10:  #  Thỏa mãn điều kiện này thì sang turn mới
                env_state[100] += 1  #  Sang turn mới
                #  env_state[155:160] = [0,0,0,0,0]

    env_state[107 + 12 * p_id : 119 + 12 * p_id] = cur_p
    env_state[101:107] = b_stocks
    return env_state, lv1, lv2, lv3


@njit()
def getActionSize():
    return P_ACTION_SIZE


@njit()
def getAgentSize():
    return 4


@njit()
def getStateSize():
    return P_STATE_SIZE


@njit()
def getValidActions(player_state_origin: np.int64):
    list_action_return = np.zeros(P_ACTION_SIZE)
    p_state = player_state_origin.copy()
    p_state = p_state.astype(np.int64)
    b_stocks = p_state[:P_END_BOARD_INFOR]  # Các nguyên liệu trên bàn chơi
    p_st = p_state[P_START_STOCK:P_END_STOCK]  # Các nguyên liệu của bản thân đang có
    yellow_count = p_state[P_YELLOW_STOCK]  # Số thẻ vàng đang có
    normal_cards = p_state[
        P_START_CARD_NORMAL:P_END_CARD_NORMAL
    ]  # Thông tin 12 thẻ đang mở
    p_upside_down_cards = p_state[
        P_START_CARD_UPSIDE_DOWN:P_END_CARD_UPSIDE_DOWN
    ]  # thông tin 3 thẻ đang úp
    taken = p_state[P_START_GET_RES:P_END_GET_RES]  # các nguyên liệu đã lấy trong turn
    p_count_st = p_state[
        P_START_STOCK_COUNT:P_END_STOCK_COUNT
    ]  # Nguyên liệu mặc định của người chơi
    list_action_return[0] = 1
    check_action_0 = False

    # Lấy nguyên liệu
    s_taken = np.sum(taken)
    temp_ = [i_ + 31 for i_ in range(5) if b_stocks[i_] != 0]
    if s_taken == 1:
        lst_s_ = np.where(taken == 1)[0]
        if len(lst_s_) > 0:
            s_ = np.where(taken == 1)[0][0]
            if b_stocks[s_] < 3:  #  Có thể lấy double
                if (s_ + 31) in temp_:
                    temp_.remove(
                        s_ + 31
                    )  # Xóa action đã lấy ở file temp nếu nguyên liệu không trên 4
            if len(temp_) > 0:
                list_action_return[np.array(temp_)] = 1
                check_action_0 = True
    elif s_taken == 2:
        lst_s_ = np.where(taken == 1)[0]
        for s_ in lst_s_:
            if (s_ + 31) in temp_:
                temp_.remove(s_ + 31)
        if len(temp_) > 0:
            list_action_return[np.array(temp_)] = 1
            check_action_0 = True
    elif s_taken == 0:
        if len(temp_) > 0 and np.sum(p_state[P_START_STOCK:P_START_STOCK_COUNT]) <= 10:
            #  list_action_return[0] = 0
            list_action_return[np.array(temp_)] = 1
    if s_taken > 0:
        list_action_return[0] = 0
        return list_action_return.astype(np.float64)

    # Trả nguyên liệu
    p_st_have_auto = p_state[P_START_STOCK:P_START_STOCK_COUNT]
    sum_p_st_have_auto = sum(p_st_have_auto)
    if sum_p_st_have_auto > 10:
        list_action_return_stock = [
            i_ + 36 for i_ in range(6) if p_st_have_auto[i_] != 0
        ]
        list_action_return[0] = 0
        list_action_return[np.array(list_action_return_stock)] = 1
        return list_action_return.astype(np.float64)

    #  Kiểm tra 15 thẻ có thể mở, action từ [1:16]
    for id_card in range(12):
        card = normal_cards[
            TOTAL_INFOR_NORMAL_CARD * id_card : TOTAL_INFOR_NORMAL_CARD * (id_card + 1)
        ]
        if sum(card) > 0:
            card_need = p_st + p_count_st - card[-5:]
            if -sum(card_need[np.where(card_need < 0)]) <= yellow_count:  # (x*x>0)
                list_action_return[id_card + 1] = 1
    for id_card in range(3):
        card = p_upside_down_cards[
            TOTAL_INFOR_NORMAL_CARD * id_card : TOTAL_INFOR_NORMAL_CARD * (id_card + 1)
        ]
        if sum(card) > 0:
            card_need = p_st + p_count_st - card[-5:]
            if -sum(card_need[np.where(card_need < 0)]) <= yellow_count:
                list_action_return[id_card + 13] = 1
    count_upside_down = 0
    for id_card in range(3):
        card_upside_down = p_upside_down_cards[
            TOTAL_INFOR_NORMAL_CARD * id_card : TOTAL_INFOR_NORMAL_CARD * (id_card + 1)
        ]
        if sum(card_upside_down) > 0:
            count_upside_down += 1
        else:
            break
    if count_upside_down < 3:  #  Nếu chưa có đủ 3 thẻ úp thì có thể úp thêm một thẻ
        count_can_up = p_state[P_COUNT_CAN_UPSIDE_DOWN]
        if count_can_up > 12:
            count_can_up = 12
        if count_can_up < 0:
            count_can_up = 0
        list_action_upside_down = np.array([i + 16 for i in range(0, count_can_up)])
        if len(list_action_upside_down) > 0:
            list_action_return[list_action_upside_down] = 1
        list_card_hide = (
            np.where(
                p_state[P_START_UPSIDE_DOWN_HIDE_CARD:P_END_UPSIDE_DOWN_HIDE_CARD] == 1
            )[0]
            + 28
        )
        list_action_return[list_card_hide] = 1

    if check_action_0 == False and np.sum(list_action_return) > 1:
        list_action_return[0] = 0

    return list_action_return.astype(np.float64)


@njit()
def one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env, lv1, lv2, lv3 = initEnv()
    _cc = 0

    while env[100] <= 400 and _cc <= 10000:
        idx = env[100] % 4
        player_state = getAgentState(env, lv1, lv2, lv3)
        if list_other[idx] == -1:
            action, per_player = p0(player_state, per_player)
        elif list_other[idx] == 1:
            action, per1 = p1(player_state, per1)
        elif list_other[idx] == 2:
            action, per2 = p2(player_state, per2)
        elif list_other[idx] == 3:
            action, per3 = p3(player_state, per3)

        list_action = getValidActions(player_state)
        if list_action[action] != 1:
            raise Exception("Action không hợp lệ")

        env, lv1, lv2, lv3 = stepEnv(action, env, lv1, lv2, lv3)
        if checkEnded(env) != 0:
            break

        _cc += 1

    for p_idx in range(4):
        env[100] = p_idx
        p_state = getAgentState(env, lv1, lv2, lv3)
        p_state[P_CLOSE_GAME] = 1
        if list_other[p_idx] == -1:
            act, per_player = p0(p_state, per_player)
            if getReward(p_state) == 1:
                winner = True
            else:
                winner = False
        elif list_other[p_idx] == 1:
            action, per1 = p1(p_state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(p_state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(p_state, per3)
        else:
            raise Exception("Sai list_other.")

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
    env, lv1, lv2, lv3 = initEnv()
    _cc = 0

    while env[100] <= 400 and _cc <= 10000:
        idx = env[100] % 4
        player_state = getAgentState(env, lv1, lv2, lv3)
        if list_other[idx] == -1:
            action, per_player = p0(player_state, per_player)
        elif list_other[idx] == 1:
            action, per1 = p1(player_state, per1)
        elif list_other[idx] == 2:
            action, per2 = p2(player_state, per2)
        elif list_other[idx] == 3:
            action, per3 = p3(player_state, per3)

        list_action = getValidActions(player_state)
        if list_action[action] != 1:
            raise Exception("Action không hợp lệ")

        env, lv1, lv2, lv3 = stepEnv(action, env, lv1, lv2, lv3)
        if checkEnded(env) != 0:
            break

        _cc += 1

    for p_idx in range(4):
        env[100] = p_idx
        p_state = getAgentState(env, lv1, lv2, lv3)
        p_state[P_CLOSE_GAME] = 1
        if list_other[p_idx] == -1:
            act, per_player = p0(p_state, per_player)
            if getReward(p_state) == 1:
                winner = True
            else:
                winner = False
        elif list_other[p_idx] == 1:
            action, per1 = p1(p_state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(p_state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(p_state, per3)
        else:
            raise Exception("Sai list_other.")

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
