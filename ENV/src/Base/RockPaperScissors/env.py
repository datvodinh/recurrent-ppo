import numpy as np
from numba import njit


@njit
def getActionSize():
    return 4


@njit
def getAgentSize():
    return 2


@njit
def getStateSize():
    return 8


@njit
def initEnv():
    env = np.zeros(7)
    env[0:2] -= 1  # Each player has not made any choice yet
    env[5] = -1  # No winner
    return env


@njit
def getAgentState(env):
    state = np.zeros(8)
    state[6] = env[4]  # phase

    if env[4] == 1:  # The phase where players confirm the information of the game turn.
        player = int(env[3])  # player who is currently playing
        state[0 + int(env[player])] = 1  # Player's choice
        state[3 + int(env[(player + 1) % 2])] = 1  # Opponent's choice

    state[-1] = env[-1]  # endgame?
    return state


@njit
def getValidActions(state):
    validActions = np.zeros(4)
    if state[6] == 0:  # phase = 0
        validActions[:3] += 1  # The player makes a choice
    else:
        validActions[3] = 1  # phase = 1
    return validActions


@njit
def stepEnv(action, env):
    if action < 3:
        env[int(env[3])] = action  # Player's choice
        env[3] = (env[3] + 1) % 2  # Switch to another player

        if (
            env[3] == 0
        ):  # If it goes back to player 0, it means that all players have made their choices
            env[4] = 1  # phase = 1

            # check winner
            check = env[0] - env[1]
            if check == 1 or check == -2:
                env[5] = 0
            if check == -1 or check == 2:
                env[5] = 1

    else:  # confirmation phase
        env[3] = (env[3] + 1) % 2  # Switch to another player
        if env[3] == 0:
            if env[5] != -1:
                env[6] = 1  # game over
            else:  # game continue
                env[4] = 0
                env[2] += 1  # turn += 1
                env[0:2] = np.zeros(2) - 1


@njit
def checkEnded(env):
    if env[-1] == 1:  # game over
        return env[5]  # winner
    return -1


@njit
def getReward(state):
    if state[-1] == 0:  # game over# The game is not yet over
        return -1

    # check winner
    check = np.where(state[:3])[0][0] - np.where(state[3:6])[0][0]
    if check == 1 or check == -2:
        return 1  # Player win

    return 0


@njit
def bot_lv0(state, per):
    validActions = getValidActions(state)
    actions = np.where(validActions == 1)[0]
    action = np.random.choice(actions)
    return action, per


def one_game_normal(p0, list_other, per_player, per1, p1):
    env = initEnv()
    while env[2] < 100:  # turn < 100
        idx = int(env[3])  # idx_player
        player_state = getAgentState(env)

        if list_other[idx] == -1:
            action, per_player = p0(player_state, per_player)
            list_action = getValidActions(player_state)
            if list_action[action] != 1:
                raise Exception("Action không hợp lệ")
        elif list_other[idx] == 1:
            action, per1 = p1(player_state, per1)
        else:
            raise Exception("Sai list_other.")

        stepEnv(action, env)
        winner = checkEnded(env)
        if winner != -1:
            break
    p0_idx = np.where(list_other == -1)[0][0]
    for p_idx in range(2):
        env[3] = p_idx
        p_state = getAgentState(env)
        if list_other[p_idx] == -1:
            action, per_player = p0(p_state, per_player)
        elif list_other[p_idx] == 1:
            action, per1 = p1(p_state, per1)
        else:
            raise Exception("Sai list_other.")

    if p0_idx == winner:
        result = 1
    else:
        result = 0
    return result, per_player


def n_games_normal(p0, num_game, per_player, list_other, per1, p1):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_normal(p0, list_other, per_player, per1, p1)
        win += winner

    return win, per_player


@njit
def one_game_numba(p0, list_other, per_player, per1, p1):
    env = initEnv()
    while env[2] < 100:
        idx = int(env[3])
        player_state = getAgentState(env)
        if list_other[idx] == -1:
            action, per_player = p0(player_state, per_player)
            list_action = getValidActions(player_state)
            if list_action[action] != 1:
                raise Exception("Action không hợp lệ")
        elif list_other[idx] == 1:
            action, per1 = p1(player_state, per1)
        else:
            raise Exception("Sai list_other.")
        stepEnv(action, env)
        winner = checkEnded(env)
        if winner != -1:
            break

    p0_idx = np.where(list_other == -1)[0][0]
    for p_idx in range(2):
        env[3] = p_idx
        p_state = getAgentState(env)
        if list_other[p_idx] == -1:
            action, per_player = p0(p_state, per_player)
        elif list_other[p_idx] == 1:
            action, per1 = p1(p_state, per1)
        else:
            raise Exception("Sai list_other.")

    if p0_idx == winner:
        result = 1
    else:
        result = 0
    return result, per_player


def n_games_numba(p0, num_game, per_player, list_other, per1, p1):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_numba(p0, list_other, per_player, per1, p1)
        win += winner

    return win, per_player


import importlib.util, json, sys

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


@njit
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
            _list_bot_level_[0],
        )
    else:
        return n_games_normal(
            p0,
            num_game,
            per_player,
            list_other,
            _list_per_level_[0],
            _list_bot_level_[0],
        )
