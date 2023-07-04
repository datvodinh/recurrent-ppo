import numpy as np
from numba import njit

# Gofish


@njit
def getActionSize():
    return 17


@njit
def getAgentSize():
    return 4


@njit
def getStateSize():
    return 107


@njit
def initEnv():
    all = np.arange(52) % 13
    np.random.shuffle(all)

    env = np.zeros(170)  ### --- update
    env[:52] = all

    for i in range(20):
        idx = i % 4
        laBai = int(env[i])
        env[53 + idx * 15 + laBai] += 1

    env[52] = 32
    for i in range(4):
        arr = env[53 + 15 * i : 66 + 15 * i]
        env[67 + i * 15] = np.where(arr == 4)[0].size
        env[66 + i * 15] = 5 - env[67 + i * 15] * 4

    env[116] = -1
    return env


@njit
def getAgentState(env):
    state = np.zeros(107)

    idx = int(env[113] % 4)  # player chính
    state[:15] = env[53 + idx * 15 : 68 + idx * 15]
    for k in range(3):
        pidx = int((idx + k + 1) % 4)
        arr_ = env[53 + pidx * 15 : 66 + pidx * 15]
        arr = state[15 + k * 15 : 30 + k * 15]
        arr[arr_ == 4] += 1
        arr[13:15] = env[66 + pidx * 15 : 68 + pidx * 15]

        state[67 + k * 13 : 80 + k * 13] = env[
            117 + pidx * 13 : 130 + pidx * 13
        ]  ### --- update

    state[60] = env[52]  ### số lá còn lại

    state[61 + int(env[114])] = 1  # phase
    if env[115]:
        state[64 + int(env[115]) - 1] = 1  # người bị yêu cầu

    state[-1] = env[-1]
    return state


@njit
def getValidActions(state):
    validActions = np.zeros(17)
    arrPhase = np.where(state[61:64] == 1)[0]
    if arrPhase.size:
        phase = arrPhase[0]
        if phase == 0:
            arr = state[28:60:15]
            idxs = np.where(arr > 0)[0]
            validActions[idxs + 1] += 1
        elif phase == 1:
            arr = state[0:13]
            idx = np.where((arr > 0) & (arr < 4))[0]
            validActions[idx + 4] = 1
        elif phase == 2:
            validActions[0] = 1

    if state[-1] == 1:
        validActions = np.ones(17)

    return validActions


@njit
def stepEnv(action, env):
    player_0 = env[113] % 4
    arr_0 = env[53 + 15 * player_0 : 68 + 15 * player_0]

    if action == 0:  # Bốc 1 là bài từ bộ bài
        laBai = int(env[52 - int(env[52])])
        env[52] -= 1
        arr_0[laBai] += 1
        arr_0[13] += 1

        # tinh lai diem
        if arr_0[laBai] == 4:
            arr_0[14] += 1  # điểm tăng 1
            arr_0[13] -= 4

        # neu het bai tren tay thi phai lay du 5 la tu bo bai
        if arr_0[13] == 0:
            while env[52] > 0 and arr_0[13] < 5:
                l_ = int(env[52 - int(env[52])])
                env[52] -= 1
                arr_0[l_] += 1
                arr_0[13] += 1
            # Tính điểm
            laBaicuaToi = arr_0[:13]
            arr_0[13] = np.sum(laBaicuaToi[laBaicuaToi < 4])
            arr_0[14] = np.where(laBaicuaToi == 4)[0].size

        if laBai == env[116] and arr_0[13] != 0:
            env[114:117] = np.array([0, 0, -1])
        else:
            if np.sum(env[66:113:15]) - arr_0[13] != 0:
                env[114:117] = np.array([3, 0, -1])
            elif env[52] != 0:
                env[114:117] = np.array([2, 0, -1])

    elif action < 4:
        env[114] = 1
        env[115] = action
    elif action < 17:
        laBai = int(action - 4)
        player_1 = int(player_0 + env[115]) % 4
        arr_1 = env[53 + 15 * player_1 : 68 + 15 * player_1]

        env[117 + int(player_0) * 13 + laBai] = 1  ###--- update

        if (
            arr_1[laBai] > 0
        ):  ### không cần nhỏ hơn 4 vì nếu người này đủ thì chắc chắn người kia không yêu cầu được
            arr_0[laBai] += arr_1[laBai]  # đưa thẻ
            arr_0[13] += arr_1[laBai]
            arr_1[13] -= arr_1[laBai]
            arr_1[laBai] = 0

            # tinh lai diem
            if arr_0[laBai] == 4:
                arr_0[14] += 1  # điểm tăng 1
                arr_0[13] -= 4

            # neu het bai tren tay thi phai lay du 5 la tu bo bai
            if arr_0[13] == 0:
                while env[52] > 0 and arr_0[13] < 5:
                    l_ = int(env[52 - int(env[52])])
                    env[52] -= 1
                    arr_0[l_] += 1
                    arr_0[13] += 1
                # tinh diem
                laBaicuaToi = arr_0[:13]
                arr_0[13] = np.sum(laBaicuaToi[laBaicuaToi < 4])
                arr_0[14] = np.where(laBaicuaToi == 4)[0].size

            if arr_0[13] != 0:
                if np.sum(env[66:113:15]) - arr_0[13] != 0:
                    env[114:117] = np.array([0, 0, -1])
                elif env[52]:
                    env[114:117] = np.array([2, 0, -1])
            else:
                env[114:117] = np.array([3, 0, -1])

        else:
            if env[52]:
                env[114:117] = np.array([2, 0, laBai])
            else:
                env[114:117] = np.array([3, 0, -1])

    # phase = 3---------------
    if np.sum(env[67:113:15]) != 13:
        while env[114] == 3:
            env[113] += 1  # chuyển sang người chơi khác
            new_pl = int(env[113] % 4)

            # Nếu người tiếp theo hết bài và trên bàn còn bài ---> thì bốc đủ cho họ
            new_arr = env[53 + new_pl * 15 : 68 + new_pl * 15]
            if env[52] and new_arr[13] == 0:
                while env[52] > 0 and new_arr[13] < 5:
                    l_ = int(env[52 - int(env[52])])
                    env[52] -= 1
                    new_arr[l_] += 1
                    new_arr[13] += 1
                # tinh diem
                laBaicuaHo = new_arr[:13]
                new_arr[13] = np.sum(laBaicuaHo[laBaicuaHo < 4])
                new_arr[14] = np.where(laBaicuaHo == 4)[0].size

            if new_arr[13] != 0:  # người chơi này còn bài
                env[114:117] = np.array([0, 0, -1])

            env[117 + new_pl * 13 : 130 + new_pl * 13] = np.zeros(13)  ###--- update
    else:
        env[-1] = 1


@njit
def checkEnded(env):
    if env[-1] == 1:
        scoreArr = env[67:113:15]
        max = np.max(scoreArr)
        if np.where(scoreArr == max)[0].size == 1:
            winner = np.argmax(scoreArr)
        else:
            laMax = 0
            for i in range(4):
                if env[67 + i * 15] == max:
                    arr_player = env[53 + i * 15 : 66 + i * 15]
                    l = np.where(arr_player == 4)[0][-1]
                    if l > laMax:
                        laMax = l
                        winner = i
        return winner
    else:
        return -1


@njit
def getReward(state):
    if state[-1] == 0:
        return -1
    else:
        arrMax = state[29:60:15]
        max_ = np.max(arrMax)
        if state[14] > max_:
            return 1
        elif state[14] == max_:
            laMax = 0
            for i in range(3):
                if arrMax[i] == max_:
                    arr = state[15 + i * 15 : 28 + i * 15]
                    l = np.where(arr == 1)[0][-1]
                    if laMax < l:
                        laMax = l
            if laMax < np.where(state[:13] == 4)[0][-1]:
                return 1
    return 0


# def visualizeEnv(env):
#   print('ENV-------')
#   print('Những lá còn lại:',env[52], env[52- int(env[52]): 52] + 1)
#   for i in range(4):
#     print('--> p:',i, np.where((env[53 + 15*i: 66+ 15*i] > 0) & (env[53 + 15*i: 66+ 15*i] < 4) )[0] + 1, env[66+ 15*i: 68+ 15*i])
#     print('arr', env[53 + 15*i: 66+ 15*i] )

#   print(env[113: 117])
#   print(env[117: 169].reshape(4,13))
#   print('--- PLAYER:', env[113] %4)

# def visualizeState(state):
#   print('arr: ', state[: 13] , np.where((state[:13] > 0) & (state[:13]< 4) )[0] + 1, state[13: 15])
#   for i in range(3):
#     print('-p:', np.where( state[15 + i*15: 28 + i*15] )[0] + 1, state[28 + i*15: 30 + i*15])
#   print('so la con lai:', state[60])
#   print('phase:', state[61: 64])
#   print('ngBiYeuCau:', state[64: 67])
#   print(state[67: 106].reshape(3,13))
#   print('endgame:', state[106])


@njit
def bot_lv0(state, per):
    validActions = getValidActions(state)
    actions = np.where(validActions == 1)[0]
    action = np.random.choice(actions)
    return action, per


def one_game_normal(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env = initEnv()
    # check = 0
    while env[113] < 400:
        idx = int(env[113]) % 4
        player_state = getAgentState(env)

        # if env[113] == check:
        #   check += 1
        #   print('-------------------------------------')
        # visualizeEnv(env)
        # visualizeState(player_state)

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
        else:
            raise Exception("Sai list_other.")

        stepEnv(action, env)
        winner = checkEnded(env)
        if winner != -1:
            break

    env[-1] = 1
    p0_idx = np.where(list_other == -1)[0][0]
    for p_idx in range(4):
        env[113] = p_idx
        p_state = getAgentState(env)
        if list_other[p_idx] == -1:
            action, per_player = p0(p_state, per_player)
        elif list_other[p_idx] == 1:
            action, per1 = p1(p_state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(p_state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(p_state, per3)
        else:
            raise Exception("Sai list_other.")

    if p0_idx == winner:
        result = 1
    else:
        result = 0
    # print(list_other)
    # print(env[53: 113].reshape(4,15))
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


@njit
def one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env = initEnv()
    while env[113] < 400:
        idx = int(env[113]) % 4
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
        else:
            raise Exception("Sai list_other.")

        stepEnv(action, env)
        winner = checkEnded(env)
        if winner != -1:
            break

    env[-1] = 1
    p0_idx = np.where(list_other == -1)[0][0]
    for p_idx in range(4):
        env[113] = p_idx
        p_state = getAgentState(env)
        if list_other[p_idx] == -1:
            action, per_player = p0(p_state, per_player)
        elif list_other[p_idx] == 1:
            action, per1 = p1(p_state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(p_state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(p_state, per3)
        else:
            raise Exception("Sai list_other.")

    if p0_idx == winner:
        result = 1
    else:
        result = 0
    return result, per_player


@njit
def n_games_numba(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_numba(
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
