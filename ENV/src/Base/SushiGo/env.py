import numpy as np
from numba import njit
from src.Utils import load_module_player


@njit()
def initEnv(n):
    """
    n : số lượng người chơi\n
    id| Name card    | Amount\n
    0 | Tempura      | 14\n
    1 | Sashimi      | 14\n
    2 | Dumpling     | 14\n
    3 | 1 Maki Roll  | 6\n
    4 | 2 Maki Roll  | 12\n
    5 | 3 Maki Roll  | 8\n
    6 | Salmon Nigiri| 10\n
    7 | Squid Nigiri | 5\n
    8 | Egg Nigiri   | 5\n
    9| Pudding      | 10\n
    10| Wasabi       | 6\n
    11| Chopsticks   | 4\n
    """
    amount_card = np.array([14, 14, 14, 6, 12, 8, 10, 5, 5, 10, 6, 4])
    id_card = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    state_sys = np.array([1, 1, n])
    list_card = np.array([0])
    index = 0
    for amount in amount_card:
        list_c = np.array([id_card[index] for i in range(amount)])
        if len(list_card) == 1:
            list_card = list_c
        else:
            list_card = np.concatenate((list_card, list_c))
        index += 1
    np.random.shuffle(list_card)
    cards = list_card[: 3 * n * (12 - n)]
    state_sys = np.concatenate((state_sys, cards))
    for i in range(n):
        state_sys = np.concatenate(
            (state_sys, np.array([0, 0]), np.array([-1 for i in range(12 - n)]))
        )
    return state_sys


@njit()
def getAgentState(state_sys, index_player):
    amount_player = 5
    round = state_sys[0] - 1
    turn = state_sys[1] + 4
    index_start_card_board = (
        (((turn - index_player) % amount_player)) * 7 + round * 7 * amount_player + 3
    )
    index_end_card_board = index_start_card_board + 7
    state_player = np.zeros(86)
    state_player[0] = state_sys[0]
    state_player[1] = state_sys[1]
    state_player[84:86] = np.array([0, 1])
    step = 2
    for index_player_relative in range(index_player, index_player + amount_player):
        index_start_player_s = (
            (index_player_relative % amount_player) * 7
            + 3 * amount_player * 7
            + (index_player_relative % amount_player + 1) * 2
            + 1
        )
        index_end_player_s = index_start_player_s + 9
        if index_player == index_player_relative:
            for i in state_sys[index_start_card_board:index_end_card_board]:
                if i != -1:
                    state_player[step + i] += 1
            #  print("cover:",index_player,state_sys,state_sys[index_start_card_board:index_end_card_board],state_player[step:step+12])
            step += 12
            state_player[step : step + 2] = state_sys[
                index_start_player_s : index_start_player_s + 2
            ]
            step += 2
            for i in state_sys[index_start_player_s + 2 : index_end_player_s]:
                if i != -1:
                    state_player[step + i] += 1
            step += 12
        else:
            state_player[step : step + 2] = state_sys[
                index_start_player_s : index_start_player_s + 2
            ]
            step += 2
            for i in state_sys[index_start_player_s + 2 : index_end_player_s]:
                if i != -1:
                    state_player[step + i] += 1
            step += 12
    return state_player.astype(np.float64)


@njit()
def calculator_for_one(card):
    score_dumpling = [0, 1, 3, 6, 10, 15]
    score = card[0]
    card = card[2:]
    #  print("card",card,end = " ")
    tempura = np.count_nonzero(card == 0)

    score += tempura // 2 * 5
    #  print("tempura: ",score,end = " ")

    sashimi = np.count_nonzero(card == 1)
    score += sashimi // 3 * 10
    #  print("sashimi: ",score,end = " ")

    dumpling = np.count_nonzero(card == 2)
    if dumpling > 5:
        score += score_dumpling[dumpling - 5]
        dumpling = 5
    score += score_dumpling[dumpling]

    #  print("dumpling: ",score,end = " ")

    salmon_nigiri = np.count_nonzero(card == 6)
    squid_nigiri = np.count_nonzero(card == 7)
    egg_nigiri = np.count_nonzero(card == 8)
    wasabi = np.count_nonzero(card == 10)
    list_ = [squid_nigiri, salmon_nigiri, egg_nigiri]
    for i in range(len(list_)):
        for j in range(list_[i]):
            if wasabi != 0:
                score += (3 - i) * 3
                wasabi -= 1
            else:
                score += 3 - i

    return score, np.count_nonzero(card == 9)


@njit()
def count_maki(card):
    maki_1 = np.count_nonzero(card == 3)
    maki_2 = np.count_nonzero(card == 4)
    maki_3 = np.count_nonzero(card == 5)
    return maki_1 + maki_2 * 2 + maki_3 * 3


@njit()
def get_index(arr, first, second):
    return np.where(arr == first)[0], np.where(arr == second)[0]


@njit()
def calculator_pudding(state_sys, amount_player):
    amount_player = state_sys[2]
    arr_pudding = np.array([0 for i in range(amount_player)])
    for index_player_relative in range(0, amount_player):
        index_start_player_s = (
            (index_player_relative) * 7
            + 3 * amount_player * 7
            + (index_player_relative + 1) * 2
            + 1
        )
        pudding = state_sys[index_start_player_s + 1]
        arr_pudding[index_player_relative] = pudding

    max_p, min_p = max(arr_pudding), min(arr_pudding)
    list_ = get_index(arr_pudding, max_p, min_p)
    #  print(list_)
    for top in range(len(list_)):
        for index_player_relative in list_[top]:
            score = (6 - top * 12) // len(list_[top])
            index_start_player_s = (
                (index_player_relative) * 7
                + 3 * amount_player * 7
                + (index_player_relative + 1) * 2
                + 1
            )
            state_sys[index_start_player_s] += score
    return state_sys


@njit()
def calculator_score(state_sys, amount_player):
    amount_player = state_sys[2]
    round = state_sys[0] - 1
    arr_maki = np.array([0 for i in range(amount_player)])
    first, second = -1, -1
    if round != 4:
        for index_player_relative in range(0, amount_player):
            index_start_player_s = (
                (index_player_relative) * 7
                + 3 * amount_player * 7
                + (index_player_relative + 1) * 2
                + 1
            )
            index_end_player_s = (
                (index_player_relative + 1) * 7
                + 3 * amount_player * 7
                + (index_player_relative + 2) * 2
                + 1
            )
            card = state_sys[index_start_player_s:index_end_player_s]
            state_sys[index_start_player_s], puding = calculator_for_one(card)
            state_sys[index_start_player_s + 1] += puding
            c_maki = count_maki(card)
            arr_maki[index_player_relative] = c_maki
            if c_maki > first:
                second = first
                first = c_maki
            elif c_maki > second:
                second = c_maki

        list_ = get_index(arr_maki, first, second)
        #  print(list_)
        for top in range(len(list_)):
            for index_player_relative in list_[top]:
                score = (6 - top * 3) // len(list_[top])
                index_start_player_s = (
                    (index_player_relative) * 7
                    + 3 * amount_player * 7
                    + (index_player_relative + 1) * 2
                    + 1
                )
                state_sys[index_start_player_s] = (
                    state_sys[index_start_player_s] + score
                )
                #  print(score)
            if len(list_[top]) > 1:
                break
    return state_sys


@njit()
def getReward(state_player):
    if state_player[1] <= 21:
        return -1
    list_score = state_player[14::14]
    Max_Score = max(list_score)
    list_winner = np.where(list_score == Max_Score)[0]
    check_winer_self = np.where(list_winner == 0)[0]
    if len(check_winer_self) == 1:
        if len(list_winner) == 1:
            return 1
        else:
            list_pudding = state_player[15::14]
            pudding_victoryer = list_pudding[list_winner]
            max_puding = max(pudding_victoryer)
            if list_pudding[0] == max_puding:
                return 1
            else:
                return 0
    else:
        return 0


@njit()
def winner_victory(state_sys):
    amount_player = int(state_sys[2])
    list_score = state_sys[3 + 3 * amount_player * 7 :: 14 - amount_player]
    max_score = max(list_score)
    winner = np.where(list_score == max_score)[0]
    #  print("Diem:",list_score)
    if len(winner) == 1:
        return winner
    else:
        list_pudding = state_sys[3 + 3 * amount_player * 7 + 1 :: 14 - amount_player]
        pudding_victoryer = list_pudding[winner]
        max_puding = max(pudding_victoryer)
        winner_puding = np.where(pudding_victoryer == max_puding)[0]
        return winner[winner_puding]


@njit()
def stepEnv(state_sys, list_action, amount_player, turn, round):
    player = 0
    turn += 4
    for a in range(turn + amount_player, turn, -1):
        index_board_s = (
            (a % amount_player) * (12 - amount_player)
            + round * (12 - amount_player) * amount_player
            + 3
        )
        index_board_e = index_board_s + (12 - amount_player)
        index_player_s = (
            (player % amount_player) * (12 - amount_player)
            + 3 * amount_player * (12 - amount_player)
            + (player % amount_player + 1) * 2
            + 3
        )
        index_player_e = index_player_s + (12 - amount_player)
        l_a = list_action[player]
        l_a = l_a[np.where(l_a >= 0)[0]]
        for i in l_a:
            if i == 13:
                break
            #  print("----------------------",i,state_sys[index_player_s:index_player_e],state_sys[index_board_s:index_board_e],state_sys.shape)
            if i == 12:
                state_sys = move_card_step(
                    state_sys,
                    11,
                    index_player_s,
                    index_player_e,
                    index_board_s,
                    index_board_e,
                )
                continue
            state_sys = move_card_step(
                state_sys,
                i,
                index_board_s,
                index_board_e,
                index_player_s,
                index_player_e,
            )
        player += 1
    return state_sys


@njit()
def reset_card_player(state_sys):
    amount_player = state_sys[2]
    for player in range(amount_player):
        index_player_s = (
            (player % amount_player) * (12 - amount_player)
            + 3 * amount_player * 7
            + (player % amount_player + 1) * 2
            + 3
        )
        index_player_e = index_player_s + (12 - amount_player)
        for i in range(index_player_s, index_player_e):
            state_sys[i] = -1
    return state_sys


@njit()
def test_action(player_state, action):
    index_between = 14
    if action == 12:
        player_state = move_card(player_state, 11, index_between + 2, 2)
        player_state[-2] = 1
        return player_state
    #  player_state[-2] = 0
    if player_state[-1] > 0:
        player_state = move_card(player_state, action, 2, index_between + 2)
        player_state[-1] -= 1
    else:
        player_state = move_card(player_state, action, 2, index_between + 2)
        player_state[-2] -= 1
    return player_state


@njit()
def move_card(state, card, start_1=0, start_2=0):
    state[start_1 + card] -= 1
    state[start_2 + card] += 1
    return state


@njit()
def move_card_step(state, card, start_1=0, end_1=0, start_2=0, end_2=0):
    index_relative_from = np.where(state[start_1:end_1] == card)[0]
    index_relative_to = np.where(state[start_2:end_2] == -1)[0]
    index_relative_from = index_relative_from[0] + start_1
    index_relative_to = index_relative_to[0] + start_2
    temp = state[index_relative_from]
    state[index_relative_from] = state[index_relative_to]
    state[index_relative_to] = temp
    return state


@njit()
def getActionSize():
    return 14


@njit()
def getAgentSize():
    return 5


@njit()
def getStateSize():
    return 86


@njit()
def getValidActions(player_state_origin: np.int64):
    list_action_return = np.zeros(14)
    player_state = player_state_origin.copy()
    player_state = player_state.astype(np.int64)
    amount = 5
    card = player_state[2:14]
    #  print("card:", card)
    if player_state[-2] == 1:
        card[11] -= 1
    list_action = np.where(card > 0)[0]
    if (12 - amount) * 3 < player_state[1]:
        list_action = np.array([13])

    of_card = player_state[16:28]
    if of_card[11] != 0 and player_state[-2] != 1 and sum(card) > 1:
        list_action = np.append(list_action, np.array([12]))

    list_action_return[np.unique(list_action)] = 1

    return list_action_return.astype(np.float64)


@njit()
def one_game_numba(p0, list_other, per_player, per1, per2, per3, per4, p1, p2, p3, p4):
    amount_player = 5
    state_sys = initEnv(amount_player)
    amount_player = state_sys[2]
    turn = state_sys[1]

    while turn < 7 * 3:
        round = state_sys[0] - 1
        turn = state_sys[1]
        list_action = np.full((amount_player, 3), 13)
        for idx in range(amount_player):
            player_state = getAgentState(state_sys, idx)

            count = 0
            while player_state[-1] + player_state[-2] > 0:
                if list_other[idx] == -1:
                    action, per_player = p0(player_state, per_player)
                    lst_action = getValidActions(player_state)
                    if lst_action[action] != 1:
                        raise Exception("Action không hợp lệ")
                elif list_other[idx] == 1:
                    action, per1 = p1(player_state, per1)
                elif list_other[idx] == 2:
                    action, per2 = p2(player_state, per2)
                elif list_other[idx] == 3:
                    action, per3 = p3(player_state, per3)
                elif list_other[idx] == 4:
                    action, per4 = p4(player_state, per4)
                list_action[idx][count] = action
                count += 1
                player_state = test_action(player_state, action)
            #  player_state = getAgentState(state_sys,idx)
        state_sys = stepEnv(state_sys, list_action, amount_player, turn, round)
        if turn % 7 == 0:
            state_sys = calculator_score(state_sys, amount_player)
            if state_sys[0] < 3:
                state_sys[0] += 1
                state_sys = reset_card_player(state_sys)
        if turn == 7 * 3:
            state_sys = calculator_pudding(state_sys, amount_player)
        if turn <= 7 * 3:
            state_sys[1] += 1

    for idx in range(amount_player):
        player_state = getAgentState(state_sys, idx)
        if list_other[idx] == -1:
            action, per_player = p0(player_state, per_player)
        elif list_other[idx] == 1:
            action, per1 = p1(player_state, per1)
        elif list_other[idx] == 2:
            action, per2 = p2(player_state, per2)
        elif list_other[idx] == 3:
            action, per3 = p3(player_state, per3)
        elif list_other[idx] == 4:
            action, per4 = p4(player_state, per4)
    winner = winner_victory(state_sys)
    p0_idx = np.where(list_other == -1)[0][0]
    if p0_idx in winner:
        result = 1
    else:
        result = 0
    return result, per_player


@njit()
def n_games_numba(
    p0, num_game, per_player, list_other, per1, per2, per3, per4, p1, p2, p3, p4
):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_numba(
            p0, list_other, per_player, per1, per2, per3, per4, p1, p2, p3, p4
        )
        win += winner

    return win, per_player


def one_game_normal(p0, list_other, per_player, per1, per2, per3, per4, p1, p2, p3, p4):
    amount_player = 5
    state_sys = initEnv(amount_player)
    amount_player = state_sys[2]
    turn = state_sys[1]

    while turn < 7 * 3:
        round = state_sys[0] - 1
        turn = state_sys[1]
        list_action = np.full((amount_player, 3), 13)
        for idx in range(amount_player):
            player_state = getAgentState(state_sys, idx)

            count = 0
            while player_state[-1] + player_state[-2] > 0:
                if list_other[idx] == -1:
                    action, per_player = p0(player_state, per_player)
                    lst_action = getValidActions(player_state)
                    if lst_action[action] != 1:
                        raise Exception("Action không hợp lệ")
                elif list_other[idx] == 1:
                    action, per1 = p1(player_state, per1)
                elif list_other[idx] == 2:
                    action, per2 = p2(player_state, per2)
                elif list_other[idx] == 3:
                    action, per3 = p3(player_state, per3)
                elif list_other[idx] == 4:
                    action, per4 = p4(player_state, per4)
                list_action[idx][count] = action
                count += 1
                player_state = test_action(player_state, action)
            player_state = getAgentState(state_sys, idx)
        state_sys = stepEnv(state_sys, list_action, amount_player, turn, round)
        if turn % 7 == 0:
            state_sys = calculator_score(state_sys, amount_player)
            if state_sys[0] < 3:
                state_sys[0] += 1
                state_sys = reset_card_player(state_sys)
        if turn == 7 * 3:
            state_sys = calculator_pudding(state_sys, amount_player)
        if turn <= 7 * 3:
            state_sys[1] += 1

    for idx in range(amount_player):
        player_state = getAgentState(state_sys, idx)
        if list_other[idx] == -1:
            action, per_player = p0(player_state, per_player)
        elif list_other[idx] == 1:
            action, per1 = p1(player_state, per1)
        elif list_other[idx] == 2:
            action, per2 = p2(player_state, per2)
        elif list_other[idx] == 3:
            action, per3 = p3(player_state, per3)
        elif list_other[idx] == 4:
            action, per4 = p4(player_state, per4)
    winner = winner_victory(state_sys)
    p0_idx = np.where(list_other == -1)[0][0]
    if p0_idx in winner:
        result = 1
    else:
        result = 0
    return result, per_player


def n_games_normal(
    p0, num_game, per_player, list_other, per1, per2, per3, per4, p1, p2, p3, p4
):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_normal(
            p0, list_other, per_player, per1, per2, per3, per4, p1, p2, p3, p4
        )
        win += winner

    return win, per_player


import json, sys

try:
    from env import SHORT_PATH
except:
    pass


@njit()
def bot_lv0(state, perData):
    validActions = getValidActions(state)
    arr_action = np.where(validActions == 1)[0]
    idx = np.random.randint(0, arr_action.shape[0])
    return arr_action[idx], perData


@njit()
def check_run_under_njit(agent, perData):
    return True


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
            _list_per_level_[3],
            _list_bot_level_[0],
            _list_bot_level_[1],
            _list_bot_level_[2],
            _list_bot_level_[3],
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
            _list_per_level_[3],
            _list_bot_level_[0],
            _list_bot_level_[1],
            _list_bot_level_[2],
            _list_bot_level_[3],
        )
