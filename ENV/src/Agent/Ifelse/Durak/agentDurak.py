import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


@njit
def DataAgent():
    per = []
    per.append(np.zeros(1))
    return per


@njit
def noTrump(
    state, validActions
):  ###  đánh lá có số nhỏ nhất có thể mà không phải trump
    trump = state[158:162]
    trump = np.where(trump)[0][0]
    action = -1
    stt = 15
    for act in validActions:
        if (act < (trump * 13) or act >= 13 * (trump + 1)) and act != 52:
            stt_ = act % 13
            if stt_ < stt:
                stt = stt_
                action = act

    return action


@njit
def defense(state, validActions):  ###  thủ bằng lá lớn nhất có thể mà không phải trump
    trump = state[158:162]
    trump = np.where(trump)[0][0]
    if state[157] and state[162] > 0:
        for act in np.flip(validActions):
            if (act < (trump * 13) or act >= 13 * (trump + 1)) and act != 52:
                return act
    return -1


@njit
def Test(state, per):
    validActions = env.getValidActions(state)
    validActions = np.where(validActions)[0]

    if defense(state, validActions) != -1:
        action = defense(state, validActions)
        return action, per

    if state[157] and state[162] > 0:
        return 52, per

    if noTrump(state, validActions) != -1:
        action = noTrump(state, validActions)
        return action, per

    if state[156] and state[162] > 2:
        return 52, per

    action = validActions[0]
    return action, per
