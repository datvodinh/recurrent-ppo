import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


def DataAgent():
    return np.array([0])


@njit()
def valueOf(action, master, nLeftCards):
    if action == 52:
        if nLeftCards > 0:
            return 3
        return 1
    if action // 13 == master:
        return 2
    return 20 - action % 13


@njit()
def Test(state, per):
    validActions = env.getValidActions(state)
    validActions = np.where(validActions == 1)[0]
    master = np.where(state[158:162] == 1)[0][0]

    valueOfActions = np.zeros_like(validActions)
    for i in range(len(validActions)):
        valueOfActions[i] = valueOf(validActions[i], master, state[162])
    action = validActions[np.argmax(valueOfActions)]
    return action, per
