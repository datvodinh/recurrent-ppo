import sys
import numpy as np
from numba import njit
import env

game_name = sys.argv[1]
env.make(game_name)


from numba.typed import List


@njit()
def Test(state, perData):
    validActions = env.getValidActions(state)
    arr_action = np.where(validActions == 1)[0]
    idx = np.random.randint(0, arr_action.shape[0])
    return arr_action[idx], perData


def DataAgent():
    return np.array([0])


def convert_to_test(perData):
    return List(perData)
