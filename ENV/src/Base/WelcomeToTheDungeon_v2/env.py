import numpy as np
from numba import njit, jit
import json

__STATE_SIZE__ = 83
__ACTION_SIZE__ = 14
__AGENT_SIZE__ = 4


@njit()
def initEnv():
    env = np.zeros(114)
    env[16:65:4] = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 9])
    return env.astype(np.int64)


@njit()
def resetRound(env):
    # Người chơi đã bị thua 2 lần thì bỏ lượt
    env[0:4] = 0
    for i in range(4):
        if env[2 * i + 5] == 2:
            env[i] = 1

    # Không lá quái vật nào được lật
    for i in range(16, 65, 4):
        env[i + 1] = -1  # Không ai chọn
        env[i + 2] = 0  # Không trong hang
        env[i + 3] = 0  # Chưa bị bỏ

    # Xào lại bài quái vật
    monsters = np.arange(13)
    np.random.shuffle(monsters)
    env[68:81] = monsters

    # Số quái chưa mở là 13
    env[81] = 13

    # Số quái trong hang là 0
    env[82] = 0

    # Thông tin anh hùng
    env[83:99] = 0

    env[100] = 0  # Bắt đầu từ Bidding

    env[101:114] = 0
    return env


@njit()
def getStateSize():
    return __STATE_SIZE__


@njit()
def getAgentState(env):
    env_ = env.copy()
    state = np.zeros(__STATE_SIZE__)

    # Index của người chơi nhận state
    pIdx = env_[99] % 4
    passArr = env_[0:4]
    score = env_[4:12]
    for i in range(4):
        pIdxEnv = (pIdx + i) % 4
        state[i] = passArr[pIdxEnv]
        state[4 + i * 2 : 6 + i * 2] = score[pIdxEnv * 2 : pIdxEnv * 2 + 2]

    # Số thẻ quái vật chưa mở
    state[12] = env_[81]

    # Số thẻ quái vật trong hang
    state[13] = env_[82]

    # Các thẻ quái vật người chơi đã mở
    state[14:53] = 0
    j = 14
    for i in range(16, 65, 4):
        if env_[i + 1] == pIdx:  # Nếu đây là thẻ do người này rút
            state[j] = env_[i]
            state[j + 1] = env_[i + 2]
            state[j + 2] = env_[i + 3]
        j = j + 3

    # Anh hùng và trang bị
    state[53:69] = env_[83:99]
    if state[54] < 0:
        state[54] = 0
    if state[62] < 0:
        state[62] = 0

    # Đang trong Bidding phase hay Dungeon phase
    state[69] = env_[100]

    # Lá quái vật vừa mở
    state[70:83] = env_[101:114]

    return state


@njit()
def getActionSize():
    return __ACTION_SIZE__


@njit()
def checkEndBidding(state):
    """
    Nếu đã có 3/4 người bỏ lượt thì kết thúc bidding
    """
    return np.sum(state[0:4].copy()) == 3


@njit()
def getValidActions(state):
    """
    Trong Bidding phase:
        [0]: Bỏ lượt
        [1]: Rút thẻ quái vật và bỏ vào hang
        [2:8]: Rút thẻ quái vật và chọn 1 trang bị để bỏ đi
    Trong Dungeon phase:
        [8]: Sử dụng Vorpal Axe anh hùng Barbarian
        [9]: Sử dụng Polymorph của anh hùng Mage
        [10]: Chuyển phase
    Chung:
        [11]: Chọn xem bài
        [12]: Chọn Barbarian
        [13]: Chọn Mage
    """

    state_ = state.copy()
    validActions = np.zeros(__ACTION_SIZE__)
    # Nếu người chơi đã bỏ lượt
    if state_[53] != 0 or state_[61] != 0:
        if state_[0] == 1:
            validActions[0] = 1
        elif checkEndBidding(state):
            validActions[10] = 1
            if state_[53] == 1:  # Anh hùng đang là Barbarian
                if state_[59] == 1:
                    validActions[8] = 1
            else:  # Anh hùng đang là Mage
                if state_[66] == 1 and state_[12] > 0:
                    validActions[9] = 1
        else:
            if state_[12] == 0:  # Nếu hết bài:
                validActions[0] = 1
            # Nếu chưa hết bài
            elif np.all(state_[70:83] == 0):  # Nếu chưa xem
                validActions[0] = 1
                validActions[11] = 1
            else:  # Đã xem bài
                if state_[53] == 1:
                    for i in range(6):
                        if state_[55 + i] == 1:
                            validActions[2 + i] = 1
                else:
                    for i in range(6):
                        if state_[63 + i] == 1:
                            validActions[2 + i] = 1
                validActions[1] = 1
        return validActions
    else:
        validActions[12] = 1
        validActions[13] = 1
        return validActions


@njit()
def stepEnv(action, env):
    pIdx = env[99] % 4
    if action == 0:  # Bỏ lượt
        env[pIdx] = 1  # Gán người chơi bỏ qua
        while env[env[99] % 4] == 1:
            env[99] += 1  # Chuyển đến người chưa bỏ lượt
    elif action == 11:  # Chọn xem thẻ quái vật
        topCardIdx = env[81] + 67
        env[env[topCardIdx] + 101] = 1
        env[17 + 4 * env[topCardIdx]] = pIdx  # Người rút thẻ
    elif action == 1:  # Rút thẻ quái vật và bỏ vào hang
        topCardIdx = env[81] + 67
        env[81] -= 1  # Giảm số quái vật trên bàn đi 1
        env[82] += 1  # Tăng số quái vật trong hang lên 1
        env[18 + 4 * env[topCardIdx]] = env[82]  # Thêm vào hang
        env[topCardIdx] = -1
        env[101:114] = 0  # Đánh dấu đã hành động sau khi xem
        env[99] += 1  # Chuyển người tiếp theo
        while env[env[99] % 4] == 1:
            env[99] += 1  # Chuyển đến người chưa bỏ lượt
    elif 2 <= action <= 7:  # Rút thẻ và chọn trang bị bỏ đi
        topCardIdx = env[81] + 67
        env[81] -= 1  # Giảm số quái vật trên bàn đi 1
        env[19 + 4 * env[topCardIdx]] = 1  # Đánh dấu đã bỏ thẻ
        if env[83] == 1:
            env[action + 83] = 0  # Đánh dấu đã bỏ trang bị của Barbarian
        else:
            env[action + 91] = 0  # Đánh dấu đã bỏ trang bị của Mage
        env[topCardIdx] = -1
        env[101:114] = 0  # Đánh dấu đã hành động sau khi xem
        env[99] += 1  # Chuyển người tiếp theo
        while env[env[99] % 4] == 1:
            env[99] += 1  # Chuyển đến người chưa bỏ lượt
    elif 8 <= action <= 10:
        # Đánh quái trong hang
        env[12:16] = 0
        env[pIdx + 12] = 1
        env[100] = 1
        position = 0
        monster = np.array([0, 0, 0, 0])
        if env[82] > 0:
            for i in range(16, 65, 4):
                if env[i + 2] == env[82]:
                    monster = env[i : i + 4]
                    position = i
            env[82] -= 1  # Giảm số quái trong hang đi 1
            env[position + 2] = 0  # Bỏ bài này khỏi hang và
            env[position + 3] = 1  # Vứt bài này
            attack = monster[0]
            if action == 8:  # Sử dụng Vorpal Axe anh hùng Barbarian
                env[89] = 0  # Bỏ đi trang bị Vorpal Axe
                attack = 0
            if action == 9:  # Sử dụng Polymorph của anh hùng Mage
                env[81] -= 1  # Giảm số quái vật trên bàn đi 1
                topCardIdx = env[81] + 67
                env[96] = 0  # Bỏ đi trang bị Polymorph
                env[17 + 4 * env[topCardIdx]] = pIdx  # Người rút thẻ
                env[topCardIdx] = -1
                attack = env[16 + 4 * env[topCardIdx]]
            if env[83] == 1:  # Tính điểm Barbarian
                env[84] += 3 * env[85] + 4 * env[90]
                env[85] = 0
                env[90] = 0
                if env[87] == 1:  # Có Torch
                    if attack <= 3:
                        attack = 0
                if env[88] == 1:  # Có War Hammer
                    if attack == 5:
                        attack = 0
                if env[86] == 1:  # Có Healing Potion
                    if attack >= env[84]:
                        attack = 0
                        env[84] = 4
                        env[86] = 0
                env[84] = env[84] - attack
            else:  # Tính điểm Mage
                if env[93] == 1:  # Có Omnipotence
                    monsters = env[16:68].copy().reshape(13, 4)
                    monsters = monsters[monsters[:, 2] > 0]
                    unique_monster = np.unique(monsters[:, 0])
                    if len(unique_monster) == len(monsters[:, 0]):
                        # Bỏ tất cả quái vật
                        env[82] = 0  # Xóa tất cả số quái trong hang
                        for i in range(16, 65, 4):
                            env[i + 2] = 0  # Bỏ bài này khỏi hang và
                            env[i + 3] = 1  # Vứt bài này
                    env[93] = 0
                env[92] += 6 * env[97] + 3 * env[98]
                env[97] = 0
                env[98] = 0
                if env[94] == 1:  # Có Holy Grail
                    if attack % 2 == 0:
                        attack = 0
                if env[95] == 1:  # Có Demonic Pact
                    if attack == 7:
                        attack = 0
                env[92] -= attack
        elif env[82] == 0:
            if env[83] == 1:
                if env[84] > 0:
                    env[pIdx * 2 + 4] += 1
                else:
                    env[pIdx * 2 + 5] += 1
            else:
                if env[92] > 0:
                    env[pIdx * 2 + 4] += 1
                else:
                    env[pIdx * 2 + 5] += 1
            if checkEnded(env) == -1:
                resetRound(env)
    elif action == 12:  # Chọn Barbarian
        env[83:91] = 1
        env[84] = 4
    elif action == 13:  # Chọn Mage
        env[91:99] = 1
        env[92] = 2


@njit()
def getAgentSize():
    return __AGENT_SIZE__


@njit()
def checkEnded(env):
    win_scores = env[4:11:2]
    lose_scores = env[5:12:2]
    if np.any(win_scores == 2):
        winner = np.where(win_scores == 2)[0][0]
        return winner
    elif len(np.where(lose_scores == 2)[0]) == 3:
        winner = np.where(lose_scores < 2)[0][0]
        return winner
    else:
        return -1


@njit
def getReward(state):
    winPoint = state[4:11:2]
    losePoint = state[5:12:2]

    if len(np.where(winPoint == 2)[0]) == 1:
        if state[4] == 2:
            return 1
        else:
            return 0
    if len(np.where(losePoint == 2)[0]) == 3:
        if state[5] == 2:
            return 0
        else:
            return 1

    return -1


def one_game_normal(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env = initEnv()
    resetRound(env)
    winner = -1
    while env[99] < 10000:
        idx = env[99] % 4
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

        stepEnv(action, env)
        winner = checkEnded(env)
        if winner != -1:
            break

    for idx in range(4):
        env[99] = idx
        p_state = getAgentState(env)
        if list_other[idx] == -1:
            action, per_player = p0(p_state, per_player)
        elif list_other[idx] == 1:
            action, per1 = p1(p_state, per1)
        elif list_other[idx] == 2:
            action, per2 = p2(p_state, per2)
        elif list_other[idx] == 3:
            action, per3 = p3(p_state, per3)

    winner__ = False
    if np.where(list_other == -1)[0][0] == winner:
        winner__ = True
    else:
        winner__ = False
    return winner__, per_player


def n_games_normal(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_normal(
            p0, list_other, per_player, per1, per2, per3, p1, p2, p3
        )
        win += winner
    return win, per_player


@jit()
def one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env = initEnv()
    resetRound(env)
    winner = -1
    while env[99] < 10000:
        idx = env[99] % 4
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

        stepEnv(action, env)
        winner = checkEnded(env)
        if winner != -1:
            break

    for idx in range(4):
        env[99] = idx
        p_state = getAgentState(env)
        if list_other[idx] == -1:
            action, per_player = p0(p_state, per_player)
        elif list_other[idx] == 1:
            action, per1 = p1(p_state, per1)
        elif list_other[idx] == 2:
            action, per2 = p2(p_state, per2)
        elif list_other[idx] == 3:
            action, per3 = p3(p_state, per3)

    winner__ = False
    if np.where(list_other == -1)[0][0] == winner:
        winner__ = True
    else:
        winner__ = False
    return winner__, per_player


@jit()
def n_games_numba(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_numba(
            p0, list_other, per_player, per1, per2, per3, p1, p2, p3
        )
        win += winner
    return win, per_player


import importlib.util, json, sys

try:
    from env import SHORT_PATH
except:
    pass


@njit()
def random_Env(p_state, per):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx], per


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

    _list_bot_level_, _list_per_level_ = load_agent(level, *args)

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
