import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


def DataAgent():
    return List([np.zeros((1, 1))])


@njit()
def Test(state, per):
    ValidActions = env.getValidActions(state)
    ValidActions = np.where(ValidActions == 1)[0]

    if state[12] <= 9 and 1 in ValidActions:
        return 1, per
    if state[13] >= 4 and 0 in ValidActions:
        return 0, per

    for i in range(3, 15):
        if i in ValidActions:
            return i, per

    action = ValidActions[np.random.randint(len(ValidActions))]
    return action, per
