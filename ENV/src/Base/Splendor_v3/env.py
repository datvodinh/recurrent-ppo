import numpy as np
from numba import njit

# Splendor Game Environment
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
P_ACTION_SIZE = 15
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
def get_list_id_card_on_lv(lv):
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
def get_card_can_get(
    env_state, p_id, cur_p, b_stocks, card_id, nl_auto, nl_bt, card_infor, lv1, lv2, lv3
):
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

    cur_p[6:11][np.where(card_infor[1:6] == 1)[0][0]] += 1  # const_stock
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

    env_state[107 + 12 * p_id : 119 + 12 * p_id] = cur_p
    env_state[101:107] = b_stocks

    return env_state, lv1, lv2, lv3


@njit()
def get_card(env_state, action_, cur_p, b_stocks, p_id, lv1, lv2, lv3):
    if action_ < 12:
        id_action = int(action_)
        id_card_normal_ = get_id_card_normal_in_lv(lv1, lv2, lv3)
    else:
        #   print('Thẻ này đang úp')
        id_action = int(action_) - 12
        id_card_normal_ = np.where(env_state[:90] == -(p_id + 1))[0]

    card_id = id_card_normal_[id_action]
    card_infor = normal_cards_infor[card_id]
    card_price = card_infor[-5:]
    nl_bo_ra = (card_price > cur_p[6:11]) * (card_price - cur_p[6:11])
    nl_bt = np.minimum(nl_bo_ra, cur_p[:5])
    nl_auto = np.sum(nl_bo_ra - nl_bt)

    card_need = cur_p[:5] + cur_p[6:11] - card_price

    #   print('Taget', get_id_card(card_id))
    if (
        -sum(card_need[np.where(card_need < 0)]) <= cur_p[5] or min(card_need) >= 0
    ):  # (x*x>0)
        #   print('Lấy thẻ', card_infor, card_need, get_id_card(card_id))

        return get_card_can_get(
            env_state,
            p_id,
            cur_p,
            b_stocks,
            card_id,
            nl_auto,
            nl_bt,
            card_infor,
            lv1,
            lv2,
            lv3,
        )


@njit()
def return_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id):
    cur_p[:5] += array_res_buy
    b_stocks[:5] -= array_res_buy
    env_state[107 + 12 * p_id : 119 + 12 * p_id] = cur_p
    env_state[101:107] = b_stocks
    return env_state, lv1, lv2, lv3


@njit()
def return_2_res_to_board(env_state, lv1, lv2, lv3, cur_p, b_stocks, res_, p_id):
    cur_p[:5][res_] += 2
    b_stocks[:5][res_] -= 2

    env_state[107 + 12 * p_id : 119 + 12 * p_id] = cur_p
    env_state[101:107] = b_stocks
    return env_state, lv1, lv2, lv3


@njit()
def stepEnv(action, env_state, lv1, lv2, lv3, all_actions):
    all_actions = np.where(all_actions == 1)[0]
    p_id = env_state[100] % 4
    cur_p = env_state[107 + 12 * p_id : 119 + 12 * p_id]
    b_stocks = env_state[101:107]
    env_state[100] += 1
    if action < 12:
        id_action = action
        id_card_normal = get_id_card_normal_in_lv(lv1, lv2, lv3)
    else:
        #   print('Thẻ này đang úp')
        id_action = action - 12
        id_card_normal = np.where(env_state[:90] == -(p_id + 1))[0]

    card_id = id_card_normal[id_action]
    card_infor = normal_cards_infor[card_id]
    card_price = card_infor[-5:]
    nl_bo_ra = (card_price > cur_p[6:11]) * (card_price - cur_p[6:11])
    nl_bt = np.minimum(nl_bo_ra, cur_p[:5])
    nl_auto = np.sum(nl_bo_ra - nl_bt)
    #   print(nl_bo_ra)

    card_need = cur_p[:5] + cur_p[6:11] - card_price

    #   print('Taget', get_id_card(card_id))
    if (
        -np.sum(card_need[np.where(card_need < 0)]) <= cur_p[5]
        or np.min(card_need) >= 0
    ):  # (x*x>0)
        #   print('Lấy thẻ', card_infor, card_need, cur_p[5], get_id_card(card_id))
        return get_card_can_get(
            env_state,
            p_id,
            cur_p,
            b_stocks,
            card_id,
            nl_auto,
            nl_bt,
            card_infor,
            lv1,
            lv2,
            lv3,
        )

    res_max = np.argmax(nl_bo_ra)
    if np.sum(cur_p[:6]) <= 8:
        if np.max(nl_bo_ra - cur_p[:5]) >= 2 and b_stocks[res_max] >= 4:
            #   print('Lấy 2 nguyên liệu', res_max, card_infor, nl_bo_ra)
            return return_2_res_to_board(
                env_state, lv1, lv2, lv3, cur_p, b_stocks, res_max, p_id
            )

        for res in np.argsort(nl_bo_ra)[::-1]:
            if (nl_bo_ra[res] - cur_p[:5][res]) >= 2 and b_stocks[res] >= 4:
                #   print('Lấy 2 nguyên liệu nhiều thứ 2__', res, card_infor, nl_bo_ra)
                return return_2_res_to_board(
                    env_state, lv1, lv2, lv3, cur_p, b_stocks, res, p_id
                )

    res_board_have = b_stocks[:5]
    res_can_buy = np.where(((nl_bo_ra - cur_p[:5]) > 0) & (res_board_have > 0), 1, 0)
    array_res_buy = np.full(5, 0)
    if np.sum(cur_p[:6]) <= 7:
        for id, res in enumerate(res_can_buy):
            if res == 1:
                array_res_buy[id] = 1
                if np.sum(array_res_buy) == 3:
                    #   print('Mua nguyên liệu_I:', array_res_buy, '----', res_can_buy, card_infor, nl_bo_ra)
                    return return_res_to_board(
                        env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id
                    )

        res_du = np.where(((res_board_have > 0) - array_res_buy) > 0)[0]
        if np.sum(array_res_buy) == 2 and len(res_du) >= 1:
            res = np.random.randint(0, len(res_du))
            array_res_buy[res_du[res]] = 1
            #   print('Mua nguyên liệu_II:', array_res_buy, '----', res_du, card_infor, nl_bo_ra)
            return return_res_to_board(
                env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id
            )

        if np.sum(array_res_buy) == 1 and len(res_du) >= 2:
            res = np.random.choice(res_du, 2, replace=False)
            array_res_buy[res] = 1
            #   print('Mua nguyên liệu_III:', array_res_buy, '----', res_du, res, card_infor, nl_bo_ra)
            return return_res_to_board(
                env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id
            )

    if len(np.where(env_state[:90] == -(p_id + 1))[0]) < 3 and (action < 12):  # Úp thẻ
        card_can_upside_down = np.where(env_state[:90] == 5)[0]
        if len(card_can_upside_down) > 0:
            #   print('Úp thẻ', card_infor, card_infor, nl_bo_ra, p_id, get_id_card(card_id))
            env_state[card_id] = -(p_id + 1)
            if b_stocks[5] > 0:
                if np.sum(cur_p[:6]) == 10:
                    for res in np.argsort(nl_bo_ra):
                        if cur_p[res] > 0:
                            #   print('Trả nguyên liệu', res)
                            cur_p[res] -= 1
                            b_stocks[res] += 1
                            break
                if np.sum(cur_p[:6]) <= 9:
                    cur_p[5] += 1
                    b_stocks[5] -= 1

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

            env_state[107 + 12 * p_id : 119 + 12 * p_id] = cur_p
            env_state[101:107] = b_stocks
            return env_state, lv1, lv2, lv3

    action_have_res_count = np.array([-99])
    soluong_nl_bo_ra = np.array([99])
    action_co_the_lay_the = np.array([-1])

    for action_ in all_actions:  # Chọn những thẻ có thể lấy trên bàn và đang úp
        if action_ < 12:
            id_action_ = int(action_)
            id_card_normal_ = get_id_card_normal_in_lv(lv1, lv2, lv3)
        else:
            id_action_ = int(action_) - 12
            id_card_normal_ = np.where(env_state[:90] == -(p_id + 1))[0]

        card_id_ = id_card_normal_[id_action_]
        card_infor_ = normal_cards_infor[card_id_]
        card_price_ = card_infor_[-5:]
        nl_bo_ra_ = (card_price_ > cur_p[6:11]) * (card_price_ - cur_p[6:11])
        card_need_ = cur_p[:5] + cur_p[6:11] - card_price_

        if (
            -np.sum(card_need_[np.where(card_need_ < 0)]) <= cur_p[5]
            or np.min(card_need_) >= 0
        ):  # (x*x>0)
            action_co_the_lay_the = np.append(action_co_the_lay_the, action_)
            soluong_nl_bo_ra = np.append(soluong_nl_bo_ra, np.sum(nl_bo_ra_))
            #   print('Thẻ có thể lấy', get_id_card(card_id_), 'Số lượng nguyên liệu bỏ ra', np.sum(nl_bo_ra_))
            if (nl_bo_ra[card_infor_[1]] - cur_p[card_infor_[1]]) > 0:
                #   print('Thẻ này có nguyên liệu cần')
                action_have_res_count = np.append(action_have_res_count, action_)
            else:
                action_have_res_count = np.append(action_have_res_count, -99)

    action_chon_lay_the_co_nl_mac_dinh = np.where(action_have_res_count != -99)[0]
    check = False
    if len(action_chon_lay_the_co_nl_mac_dinh) > 0:  # Lấy thẻ có nguyên liệu mặc định
        nl_bo_ra_min = 99
        #   print("Lấy thẻ có nguyên liệu mặc định", soluong_nl_bo_ra, action_chon_lay_the_co_nl_mac_dinh)
        for action_id in action_chon_lay_the_co_nl_mac_dinh:
            if soluong_nl_bo_ra[action_id] < nl_bo_ra_min:
                nl_bo_ra_min = soluong_nl_bo_ra[action_id]
                action_ = action_have_res_count[action_id]
                check = True
        if check == True:
            return get_card(env_state, action_, cur_p, b_stocks, p_id, lv1, lv2, lv3)

    sum_res_need = 10 - np.sum(cur_p[:6])
    if sum_res_need >= 3:
        sum_res_need = 3

    res_board_have = b_stocks[:5]
    res_can_buy = np.where(((nl_bo_ra - cur_p[:5]) > 0) & (res_board_have > 0), 1, 0)
    array_res_buy = np.full(5, 0)

    for id, res in enumerate(res_can_buy):
        if res == 1:
            if np.sum(array_res_buy) == sum_res_need:
                break
            array_res_buy[id] = 1

    if np.sum(array_res_buy) > 1:
        #   print('Mua nguyên liệu_IIII:', array_res_buy, '----', res_can_buy, card_infor, nl_bo_ra)
        return return_res_to_board(
            env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id
        )

    if len(soluong_nl_bo_ra) > 1 and np.min(soluong_nl_bo_ra) == 0:
        action_ = action_co_the_lay_the[np.argmin(soluong_nl_bo_ra)]
        #   print("Lấy thẻ không có nguyên liệu mặc định miễn phí:", np.min(soluong_nl_bo_ra))
        return get_card(env_state, action_, cur_p, b_stocks, p_id, lv1, lv2, lv3)

    if np.max(nl_bo_ra) > 0 and b_stocks[res_max] >= 4 and np.sum(cur_p[:6]) <= 8:
        #   print('Lấy 2 nguyên liệu_II:', res_max, '----', card_infor, nl_bo_ra)
        return return_2_res_to_board(
            env_state, lv1, lv2, lv3, cur_p, b_stocks, res_max, p_id
        )

    if np.sum(array_res_buy) > 0:
        for id, res in enumerate(b_stocks[:5]):
            if res > 0:
                if np.sum(array_res_buy) == sum_res_need:
                    break
                array_res_buy[id] = 1
        #   print('Mua nguyên liệu_IIIII:', array_res_buy, '----', res_can_buy, card_infor, nl_bo_ra)
        return return_res_to_board(
            env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id
        )

    res_max = np.argmax(b_stocks[:5])
    if b_stocks[res_max] >= 4 and np.sum(cur_p[:6]) <= 8:
        #   print('Lấy 2 nguyên liệu_III, bất kỳ:', res_max, '----', card_infor, nl_bo_ra)
        return return_2_res_to_board(
            env_state, lv1, lv2, lv3, cur_p, b_stocks, res_max, p_id
        )

    for id, res in enumerate(b_stocks[:5]):
        if res > 0:
            if np.sum(array_res_buy) == sum_res_need:
                break
            array_res_buy[id] = 1

    if np.sum(array_res_buy) > 0:
        #   print('Mua nguyên liệu còn thừa trên bàn_IIIIII:', array_res_buy, '----', res_can_buy, card_infor, nl_bo_ra, sum_res_need)
        return return_res_to_board(
            env_state, lv1, lv2, lv3, cur_p, b_stocks, array_res_buy, p_id
        )

    if len(soluong_nl_bo_ra) > 1:
        action_ = action_co_the_lay_the[np.argmin(soluong_nl_bo_ra)]
        #   print("Lấy thẻ không có nguyên liệu mặc định:", np.min(soluong_nl_bo_ra))
        return get_card(env_state, action_, cur_p, b_stocks, p_id, lv1, lv2, lv3)

    #   print('Không làm gì cả')
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
    list_action_return = np.zeros(15)
    list_action_return[: int(player_state_origin[P_COUNT_CAN_UPSIDE_DOWN])] = 1
    for id_card in range(3):
        card = player_state_origin[P_START_CARD_UPSIDE_DOWN:P_END_CARD_UPSIDE_DOWN][
            11 * id_card : 11 + 11 * id_card
        ]
        if np.sum(card) > 0:
            list_action_return[12 + id_card] = 1
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

        env, lv1, lv2, lv3 = stepEnv(action, env, lv1, lv2, lv3, list_action)
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

        env, lv1, lv2, lv3 = stepEnv(action, env, lv1, lv2, lv3, list_action)
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
