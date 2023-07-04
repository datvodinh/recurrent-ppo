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
def Test(state, per):
    validActions = env.getValidActions(state)
    validActions = np.where(validActions)[0]
    if 61 in validActions:
        return 61, per
    if 77 in validActions:
        return 77, per
    if 78 in validActions:
        return 78, per
    action = np.random.choice(validActions)
    return action, per
