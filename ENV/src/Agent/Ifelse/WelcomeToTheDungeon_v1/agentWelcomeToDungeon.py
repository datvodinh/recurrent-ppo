import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


@njit()
def DataAgent():
    return np.array([0.0])


@njit()
def Test(state, per):
    actions = env.getValidActions(state)
    actions = np.where(actions == 1)[0]
    if state[12] > 8:
        if 0 in actions:
            return 0, per
    if 1 in actions:
        return 1, per
    if 2 in actions:
        return 2, per
    for action in actions:
        if action in range(15, 19):
            return action, per
    return np.random.choice(actions), per
