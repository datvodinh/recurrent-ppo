import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


def DataAgent():
    return np.array([])


@njit()
def Test(state, per):
    validActions = env.getValidActions(state)
    validActions[79] = 0
    validActions = np.where(validActions == 1)[0]

    if 61 in validActions:
        return 61, per

    if 78 in validActions:
        return 78, per

    if 81 in validActions:
        return 81, per

    if 77 in validActions:
        return 77, per

    if 80 in validActions:
        return 80, per

    if 44 in validActions:
        return 44, per

    action = validActions[np.random.randint(len(validActions))]
    return action, per
