import sys
import warnings

import numba
import numpy as np
from numba import jit, njit
from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaWarning,
)
from numba.typed import List

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaWarning)


@njit()
def initEnv():
    env = np.zeros(81)
    card = np.arange(52)  # card
    np.random.shuffle(card)
    for i in range(4):
        env[card[i * 8 : (i + 1) * 8]] = i + 1
    env[52] = card[-1]  # trump suit card
    env[53] = 0  # mode: attack or defense
    env[54:57] = [1, 3, 4]
    env[57] = 1  # num of people choose attack this round
    env[58] = 2  # player_id defending
    env[59] = 0  # index player attack in env[54:57]
    env[60:80] = card[32:52]  # card on deck
    env[80] = 0
    return env


@njit()
def getStateSize():
    return 167


@njit()
def getAgentSize():
    return 4


@njit()
def getAgentState(env):
    state = np.zeros(getStateSize())
    if env[53] == 0:
        player_id = env[54:57][int(env[59])]  # attack player id
    elif env[53] == 1:
        player_id = env[58]  # defend player id
    state[0:52][np.where(env[0:52] == player_id)] = 1  #  card player hold
    state[52:104][np.where(env[0:52] == 6)] = 1  # all card defender defend this round
    state[104:156][np.where(env[0:52] == 5)] = 1  # card have to defend this round
    if env[53] == 1:
        state[156:158] = [0, 1]  #  attack, defend
    elif env[53] == 0:
        state[156:158] = [1, 0]
    state[158:162][int(env[52]) // 13] = 1  # trump suit
    state[162] = len(np.where(env[0:52] == 0)[0])  # num card on deck

    for i in range(3):
        state[163 + i] = np.where(env[0:52] == (player_id + i) % 4 + 1)[0].shape[0]

    state[166] = env[80]
    return state


@njit()
def getActionSize():
    return 53


@njit()
def getDefenseCard(state):
    card = np.zeros(52)
    idx = np.argmax(state[158:162])  # trump suit: 0:spade,1:club,2:diamond,3:heart
    card_def_id = np.argmax(state[104:156])
    if card_def_id // 13 != idx:  # card have to defend not a trump card
        card[13 * idx : 13 * (idx + 1)][
            np.where(state[13 * idx : 13 * (idx + 1)] == 1)
        ] = 1  #  trump card on hand
        card[card_def_id + 1 : 13 * (card_def_id // 13 + 1)][
            np.where(state[card_def_id + 1 : 13 * (card_def_id // 13 + 1)] == 1)
        ] = 1  # same type card, higher value on hand.
    else:  # card have to defend is a trump card
        card[card_def_id + 1 : 13 * (idx + 1)][
            np.where(state[card_def_id + 1 : 13 * (idx + 1)] == 1)
        ] = 1  # higher value trump card only.
    return card


@njit()
def getAttackCard(state):
    card = np.zeros(52)
    card_on_board = np.where(state[52:104] == 1)[0]
    card_value_on_board = card_on_board % 13  # value of that card (ex: 4 diamond is 4)
    card_on_hand = np.where(state[0:52] == 1)[0]  # card on player's hand
    for c in card_on_hand:
        if c % 13 in card_value_on_board:
            card[c] = 1
    return card


@njit()
def getValidActions(state):
    list_action = np.zeros(getActionSize())
    # attack
    if (
        state[156] == 1 and np.sum(state[52:104]) == 0
    ):  # main attacker, defender have nothing to defend yet.
        list_action[0:52] = state[0:52]
    elif (
        state[156] == 1 and np.sum(state[52:104]) != 0
    ):  # side attacker, attack only card with same value on the defend board( 4 heart on hand if have 4 spade on board)
        list_action[0:52] = getAttackCard(state)
        list_action[52] = 1
    # defense
    if state[157] == 1:  # defender
        list_action[0:52] = getDefenseCard(state)
        list_action[52] = 1
    if np.sum(list_action) == 0:
        list_action[52] = 1
    return list_action


@njit()
def drawCard(env):
    turn_draw_card = np.zeros(4)
    turn_draw_card[np.array([0, 2, 3])] = env[
        54:57
    ]  # attack player,main attack draw first.
    turn_draw_card[1] = env[58]  # defend player draw second.
    for p_id in turn_draw_card:  # draw card
        num_card_on_deck = len(np.where(env[0:52] == 0)[0])  # num cards left on deck
        if num_card_on_deck > 0:
            num_card_player = len(np.where(env[0:52] == p_id)[0])
            if num_card_player < 8:
                num_card_need = 8 - num_card_player
                if num_card_on_deck >= num_card_need:
                    env[
                        env[60:80].astype(np.int64)[
                            20
                            - num_card_on_deck : 20
                            - num_card_on_deck
                            + num_card_need
                        ]
                    ] = p_id
                else:
                    env[env[60:80].astype(np.int64)[20 - num_card_on_deck :]] = p_id
    return env


@njit()
def changeAttackPlayer(env):  # change the defender and attacker
    if env[58] == 1:
        env[54:57] = [4, 2, 3]
    elif env[58] == 2:
        env[54:57] = [1, 3, 4]
    elif env[58] == 3:
        env[54:57] = [2, 4, 1]
    elif env[58] == 4:
        env[54:57] = [3, 1, 2]


@njit()
def stepEnv(action, env):
    if action == 52:  # skip
        if env[53] == 1:  # defense
            env[0:52][np.where(env[0:52] == 5)] = env[58]  # Defender hold all card
            env[0:52][np.where(env[0:52] == 6)] = env[58]  # Defender hold all card
            env = drawCard(env)  # draw card
            env[58] = (
                (env[58] + 2) % 4 if env[58] > 2 else env[58] + 2
            )  # change defend player
            changeAttackPlayer(env)  # change attack player
            env[53] = 0  # reset mode: attack
            env[59] = 0
        elif env[53] == 0:  # attack
            env[57] += 1  # num attacker skip this round
            env[59] = (env[59] + 1) % 3
            if env[57] == 3:  # all attacker skip this round
                env[0:52][np.where(env[0:52] == 5)] = -1  # Thrown away card
                env[0:52][np.where(env[0:52] == 6)] = -1  # Thrown away card
                env = drawCard(env)  # draw card
                env[58] = 1 if env[58] == 4 else env[58] + 1  # change defend player
                changeAttackPlayer(env)  # change attack players
                env[57] = 0
                env[59] = 0

    else:  # attack or defend any card
        if env[53] == 1:  # defense
            env[0:52][np.where(env[0:52] == 5)] = 6  # defense this card successful
            env[action] = 6
            if len(np.where(env[0:52] == env[58])[0]) == 0:
                env[0:52][np.where(env[0:52] == 5)] = -1  # Thrown away card
                env[0:52][np.where(env[0:52] == 6)] = -1  # Thrown away card
                env = drawCard(env)  # draw card
                env[58] = 1 if env[58] == 4 else env[58] + 1  # change defend player
                changeAttackPlayer(env)  # change attack players
                env[57] = 0
                env[59] = 0
            env[53] = 0  # change mode: attack
        elif env[53] == 0:  # attack
            env[action] = 5  # this card have to defend
            env[53] = 1  # change mode: defend
            env[57] = 0  # change num player attack skip turn to 0
            env[59] = 0  # change attack player

    #  return env


@njit()
def checkEnded(env):
    if len(np.where(env[0:52] == 0)[0]) == 0:  # if no card left on deck
        for i in range(1, 5):
            if len(np.where(env[0:52] == i)[0]) == 0:
                return i - 1
    return -1


@njit()
def getReward(state):
    if state[162] == 0 and state[166] == 1:
        if np.sum(state[0:52]) == 0:
            return 1
        elif np.min(state[163:166]) == 0:
            return 0
        else:
            return -1
    else:
        return -1


@njit()
def one_game_numba(p0, pIdOrder, per_player, per1, per2, per3, p1, p2, p3):
    env = initEnv()
    winner = -1
    turn = 0
    while True:
        turn += 1
        if env[53] == 1:  # defend
            pIdx = int(env[58] - 1)
        else:  #  attack
            pIdx = int(env[54:57][int(env[59])] - 1)
        if pIdOrder[pIdx] == -1:
            action, per_player = p0(getAgentState(env), per_player)
        elif pIdOrder[pIdx] == 1:
            action, per1 = p1(getAgentState(env), per1)
        elif pIdOrder[pIdx] == 2:
            action, per2 = p2(getAgentState(env), per2)
        elif pIdOrder[pIdx] == 3:
            action, per3 = p3(getAgentState(env), per3)
        stepEnv(action, env)

        winner = checkEnded(env)
        if winner != -1:
            #  print(winner)
            break
    env[80] = 1
    #  env[0:52] = 6
    for pIdx in range(4):
        env[53] = 1
        env[58] = pIdx * 1.0 + 1.0
        if pIdOrder[pIdx] == -1:
            action, per_player = p0(getAgentState(env), per_player)
        elif pIdOrder[pIdx] == 1:
            action, per1 = p1(getAgentState(env), per1)
        elif pIdOrder[pIdx] == 2:
            action, per2 = p2(getAgentState(env), per2)
        elif pIdOrder[pIdx] == 3:
            action, per3 = p3(getAgentState(env), per3)

    win = False
    if np.where(pIdOrder == -1)[0][0] == checkEnded(env):
        win = True
    else:
        win = False

        #  #  print('ok')
    return win, per_player


@njit()
def n_game_numba(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_numba(
            p0, list_other, per_player, per1, per2, per3, p1, p2, p3
        )
        #  print(winner)
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


def one_game_normal(p0, pIdOrder, per_player, per1, per2, per3, p1, p2, p3):
    env = initEnv()
    winner = -1
    turn = 0
    while True:
        turn += 1
        if env[53] == 1:
            pIdx = int(env[58] - 1)
        else:
            pIdx = int(env[54:57][int(env[59])] - 1)
        if pIdOrder[pIdx] == -1:
            action, per_player = p0(getAgentState(env), per_player)
        elif pIdOrder[pIdx] == 1:
            action, per1 = p1(getAgentState(env), per1)
        elif pIdOrder[pIdx] == 2:
            action, per2 = p2(getAgentState(env), per2)
        elif pIdOrder[pIdx] == 3:
            action, per3 = p3(getAgentState(env), per3)
        stepEnv(action, env)

        winner = checkEnded(env)
        if winner != -1:
            break
    env[80] = 1
    #  env[0:52] = 6
    for pIdx in range(4):
        env[53] = 1
        env[58] = pIdx * 1.0 + 1.0
        if pIdOrder[pIdx] == -1:
            action, per_player = p0(getAgentState(env), per_player)
        elif pIdOrder[pIdx] == 1:
            action, per1 = p1(getAgentState(env), per1)
        elif pIdOrder[pIdx] == 2:
            action, per2 = p2(getAgentState(env), per2)
        elif pIdOrder[pIdx] == 3:
            action, per3 = p3(getAgentState(env), per3)

    win = False
    if np.where(pIdOrder == -1)[0][0] == checkEnded(env):
        win = True
    else:
        win = False

        #  #  print('ok')
    return win, per_player


def n_game_normal(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_normal(
            p0, list_other, per_player, per1, per2, per3, p1, p2, p3
        )
        #  print(winner)
        win += winner
    return win, per_player


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
        return n_game_numba(
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
        return n_game_normal(
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
