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

    #  print(state[12:14])
    #  print(ValidActions,"-----------")
    if state[12] >= 5 and 11 in ValidActions:
        return 11, per
    for i in range(2, 8):
        if state[12] >= 5 and i in ValidActions:
            return i, per

    #  if state[13] >= 4 and 0 in ValidActions:
    #      return 0, per

    action = ValidActions[np.random.randint(len(ValidActions))]
    return action, per
