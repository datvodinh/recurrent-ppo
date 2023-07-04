import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


def DataAgent():
    return [np.zeros((1, 1))]


@njit()
def Test(state, per):
    validActions = env.getValidActions(state)
    validActions[52] = validActions[52] * state[18] * state[17] * state[16]
    validActions[50] = validActions[50] * state[18] * state[16]
    validActions = np.where(validActions == 1)[0]

    if 1 in validActions:  #  Đổ xúc xắc
        return 1, per

    if (0 in validActions) and (state[117] not in (2, 3, 4)):
        return 0, per

    if (
        (np.sum(state[16:20] == 2))
        and (validActions[52] == 0)
        and (validActions[50] == 0)
    ):
        if 53 in validActions:
            return 53, per

    if np.sum(state[1:16]) >= 8:
        for i in (51, 52, 50, 49):
            if i in validActions:
                return i, per

    for i in (35, 34, 36, 38, 39, 37):
        if (i in validActions) and (state[i - 33] <= 4):
            return i, per

    if state[39] + state[59] + state[79] >= 1:
        for i in (43, 44, 42):
            if (i in validActions) and (state[i - 33] <= 1):
                return i, per

    if 53 in validActions:
        return 53, per

    action = validActions[np.random.randint(len(validActions))]
    return action, per
