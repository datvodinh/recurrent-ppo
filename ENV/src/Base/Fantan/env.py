import numpy as np
from numba import njit, jit
from numba.typed import List

perData = np.array([0])
CARD = 52
cards = np.arange(1, 53)  # Build cards
sevens = np.array([6, 19, 32, 45])
idxChip = np.array([21, 35, 49, 63])
ACTION_SIZE = 53
AGENT_SIZE = 4
STATE_SIZE = 112


# Work function
@njit()
def initEnv():
    cards = np.arange(0, 52, dtype=np.int64)
    np.random.shuffle(cards)
    env = np.full(71, 0)
    env[0:8] = np.array([-1, 6, -1, 19, -1, 32, -1, 45])

    env[8:21] = cards[0:13]
    env[22:35] = cards[13:26]
    env[36:49] = cards[26:39]
    env[50:63] = cards[39:52]
    idx = np.array([21, 35, 49, 63])
    env[idx] = 50
    env[64] = 0
    env[65] = 0
    env[66] = 0
    return env


@njit()
def getAgentSize():
    return AGENT_SIZE


@njit()
def getActionSize():
    return ACTION_SIZE


@njit()
def getStateSize():
    return STATE_SIZE


@njit()
def getAgentState(env):
    state = np.zeros(112)
    pIdx = env[64]
    player_card_idx = env[8 + pIdx * 14 : 8 + pIdx * 14 + 13]
    state[player_card_idx] = 1
    card_on_board = env[0:8]
    card_on_board = card_on_board[card_on_board > -1]
    card_on_board_id = np.full(52, 0)
    card_on_board_id[card_on_board] = 1
    state[52:104] = card_on_board_id
    state[104] = env[8 + pIdx * 14 + 13]
    cards_len = list([0, 0, 0])
    count = 0
    for i in range(getAgentSize()):
        if i == pIdx:
            continue
        cards = env[8 + i * 14 : 8 + i * 14 + 13]
        len_ = len(np.where(cards > -1)[0])
        cards_len[count] = len_
        count += 1
    state[105:108] = cards_len
    state[108] = env[66]  # Game da ket thuc hay chua
    chipArr = np.zeros(3)
    count = 0
    for i in range(4):
        if i == pIdx:
            continue
        chip = env[8 + i * 14 + 13]
        chipArr[count] = chip
        count += 1
    state[109:112] = chipArr
    return state


@njit()
def getValidActions(state):
    p_cards_binary = state[0:52]
    p_cards = np.where(p_cards_binary == 1)[0]
    card_on_board_binary = state[52:104]
    card_on_board = np.where(card_on_board_binary == 1)[0]
    arr_action = np.zeros(53)
    arr_action[52] = 1
    for i in range(len(arr_action)):
        if i in p_cards and i in card_on_board:
            arr_action[i] = 1
    return arr_action


# ------------------------------------------------------------------------
@njit()
def stepEnv(action, env):
    player_Id = env[64]
    player_Card = env[8 + player_Id * 14 : 8 + player_Id * 14 + 13]
    current_card_on_board = env[0:8]
    if action == 52:
        env[8 + player_Id * 14 + 13] -= 1
        env[65] += 1
        if env[8 + player_Id * 14 + 13] <= 0:
            return -2
    if action != 52:
        player_Card[np.where(player_Card == action)[0]] = -1
        if action == 6 or action == 19 or action == 32 or action == 45:  # action bang 7
            if action == 6:
                current_card_on_board[0:2] = [5, 7]
            if action == 19:
                current_card_on_board[2:4] = [18, 20]
            if action == 32:
                current_card_on_board[4:6] = [31, 33]
            if action == 45:
                current_card_on_board[6:8] = [44, 46]
        else:  # Check các action hợp lệ
            if 0 <= action < 6:
                current_card_on_board[0] -= 1
            if 6 < action < 12:
                current_card_on_board[1] += 1
            if 12 < action < 19:
                current_card_on_board[2] -= 1
            if 19 < action < 25:
                current_card_on_board[3] += 1
            if 25 < action < 32:
                current_card_on_board[4] -= 1
            if 32 < action < 38:
                current_card_on_board[5] += 1
            if 38 < action < 45:
                current_card_on_board[6] -= 1
            if 45 < action < 52:
                current_card_on_board[7] += 1
    env[0:8] = current_card_on_board
    env[8 + player_Id * 14 : 8 + player_Id * 14 + 13] = player_Card
    if max(env[8 + player_Id * 14 : 8 + player_Id * 14 + 13]) == -1:
        return -1
    return 0


# ----------------------------------------------------------------
@njit()
def getReward(state):
    IsEnd = state[108]
    if IsEnd == 0:
        return -1
    if IsEnd == 1:
        p_chip = state[104]
        chip_arr = state[109:112]
        if p_chip > max(chip_arr):
            return 1
        elif p_chip == max(chip_arr):
            id_max_chip = np.argmax(chip_arr)
            cards_len = state[105:108]
            player_cards = state[0:52]
            player_cards_id = np.where(player_cards > -1)
            player_cards_len = len(player_cards_id)
            if player_cards_len >= cards_len[id_max_chip]:
                return 1
            else:
                return 0
        else:
            return 0


# ----------------------------------------------------------------
@njit()
def one_game_numba(
    p0,
    list_other,
    per_player,
    per1,
    per2,
    per3,
    p1,
    p2,
    p3,
):
    allGame = True
    saveStoreChip = np.array([50, 50, 50, 50])
    idxPlayerChip = np.array([21, 35, 49, 63])
    while allGame:
        env = initEnv()
        env[idxPlayerChip] = saveStoreChip
        oneGame = True
        while oneGame:
            count = 10000
            if count > 0:
                count -= 1
                for i in range(getAgentSize()):
                    env[64] = i
                    state = getAgentState(env)
                    if list_other[i] == -1:
                        action, per_player = p0(state, per_player)
                    if list_other[i] == 1:
                        action, per1 = p1(state, per1)
                    if list_other[i] == 2:
                        action, per2 = p2(state, per2)
                    if list_other[i] == 3:
                        action, per3 = p3(state, per3)
                    stepEnvReturn = stepEnv(action, env)
                    if stepEnvReturn == -1:
                        oneGame = False
                        env[8 + i * 14 + 13] += env[65]
                        saveStoreChip = env[idxPlayerChip]
                        env[65] = 0
                    elif stepEnvReturn == -2:  # Khi nguoi choi het chip
                        env[66] = 1
                        # Cong chip cho nguoi choi con it bai nhat
                        player_chip = env[idxPlayerChip]
                        player_id_not_0_chip = np.where(player_chip > 0)[0]
                        arr_player_cards = np.zeros(13 * 3)
                        for i in range(len(player_id_not_0_chip)):
                            player_cards = env[
                                8
                                + player_id_not_0_chip[i] * 13 : 8
                                + player_id_not_0_chip[i] * 13
                                + 13
                            ]  # bai cua nhung nguoi khong bi 0 chip
                            arr_player_cards[
                                i * 13 : i * 13 + 13
                            ] = player_cards.astype(np.float64)
                        arr_player_cards = np.reshape(arr_player_cards, (3, 13))
                        player_card_len = np.array(
                            [
                                len(np.where(player_cards > -1))
                                for player_cards in arr_player_cards
                            ]
                        )
                        player_lowest_card = np.argmax(player_card_len)
                        player_lowest_card_id = player_id_not_0_chip[player_lowest_card]
                        env[idxPlayerChip[player_lowest_card_id]] += env[65]
                        env[65] = 0

                        for pIdx in range(4):
                            env[64] = pIdx
                            state = getAgentState(env)
                            if list_other[pIdx] == -1:
                                action, per_player = p0(getAgentState(env), per_player)
                                if getReward(state) == 1:
                                    winner = True
                                else:
                                    winner = False
                            elif list_other[pIdx] == 1:
                                action, per1 = p1(getAgentState(env), per1)
                            elif list_other[pIdx] == 2:
                                action, per2 = p2(getAgentState(env), per2)
                            elif list_other[pIdx] == 3:
                                action, per3 = p3(getAgentState(env), per3)
                        allGame = False
                        return winner, per_player
            if count < 0:
                return False, per_player


# # ------------------------------------------------------------------------
def one_game_normal(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    allGame = True
    saveStoreChip = np.array([50, 50, 50, 50])
    idxPlayerChip = np.array([21, 35, 49, 63])
    while allGame:
        env = initEnv()
        env[idxPlayerChip] = saveStoreChip
        oneGame = True
        while oneGame:
            count = 10000
            if count > 0:
                count -= 1
                for i in range(getAgentSize()):
                    env[64] = i
                    state = getAgentState(env)
                    if list_other[i] == -1:
                        action, per_player = p0(state, per_player)
                    if list_other[i] == 1:
                        action, per1 = p1(state, per1)
                    if list_other[i] == 2:
                        action, per2 = p2(state, per2)
                    if list_other[i] == 3:
                        action, per3 = p3(state, per3)
                    stepEnvReturn = stepEnv(action, env)
                    if stepEnvReturn == -1:
                        oneGame = False
                        env[8 + i * 14 + 13] += env[65]
                        saveStoreChip = env[idxPlayerChip]
                        env[65] = 0
                    elif stepEnvReturn == -2:
                        env[66] = 1
                        # Cong chip cho nguoi choi con it bai nhat

                        player_chip = env[idxPlayerChip]
                        player_id_not_0_chip = np.where(player_chip > 0)[0]
                        arr_player_cards = np.zeros(13 * 3)
                        for i in range(len(player_id_not_0_chip)):
                            player_cards = env[
                                8
                                + player_id_not_0_chip[i] * 13 : 8
                                + player_id_not_0_chip[i] * 13
                                + 13
                            ]  # bai cua nhung nguoi khong bi 0 chip
                            arr_player_cards[
                                i * 13 : i * 13 + 13
                            ] = player_cards.astype(np.float64)
                        arr_player_cards = np.reshape(arr_player_cards, (13, 3))
                        player_card_len = np.array(
                            [
                                len(np.where(player_cards > -1))
                                for player_cards in arr_player_cards
                            ]
                        )
                        player_lowest_card = np.argmax(player_card_len)
                        player_lowest_card_id = player_id_not_0_chip[player_lowest_card]
                        env[idxPlayerChip[player_lowest_card_id]] += env[65]
                        env[65] = 0
                        for pIdx in range(4):
                            env[64] = pIdx
                            state = getAgentState(env)
                            if list_other[pIdx] == -1:
                                action, per_player = p0(getAgentState(env), per_player)
                                if getReward(state) == 1:
                                    winner = True
                                else:
                                    winner = False
                            elif list_other[pIdx] == 1:
                                action, per1 = p1(getAgentState(env), per1)
                            elif list_other[pIdx] == 2:
                                action, per2 = p2(getAgentState(env), per2)
                            elif list_other[pIdx] == 3:
                                action, per3 = p3(getAgentState(env), per3)
                        allGame = False
                        return winner, per_player
            if count < 0:
                return False, per_player


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
def n_games_numba(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_numba(
            p0, list_other, per_player, per1, per2, per3, p1, p2, p3
        )
        win += winner
    return win, per_player


# -----------------------------------------------------------------------------------
import importlib.util, json, sys

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
            lst_module_level = [
                load_module_player(lst_agent_level[i]) for i in range(num_bot)
            ]
            for i in range(num_bot):
                data_agent_level = np.load(
                    f"{SHORT_PATH}src/Agent/{lst_agent_level[i]}/Data/{env_name}_{level}/Train.npy",
                    allow_pickle=True,
                )
                _list_per_level_.append(
                    lst_module_level[i].convert_to_test(data_agent_level)
                )
                _list_bot_level_.append(lst_module_level[i].Test)

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
