import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


def DataAgent():
    per = []
    per.append(np.zeros(1))
    return per


def Test(state, per):
    validActions = env.getValidActions(state)
    actions = np.where(validActions)[0]

    monster = np.where(state[28:36])[0]
    if state[13] > 4:
        if 0 in actions:
            return 0, per

    if 1 in actions:
        return 1, per

    for i in np.array([6, 7, 5, 8, 4, 3, 12, 13, 11, 14, 10, 9]):
        if i in actions:
            return i, per

    act = np.random.choice(actions)
    return act, per
