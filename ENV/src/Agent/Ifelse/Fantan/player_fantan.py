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
def valueOf(action, pCards):
    if action == 52:
        return 0
    pSuits = pCards // 13
    counts = np.zeros(4)
    for suit in pSuits:
        counts[suit] += 1
    value = counts[action // 13] + 1
    pValues = pCards % 13
    countValues = np.zeros(13)
    for val in pValues:
        countValues[val] += 1
    small = np.sum(countValues[0:6])
    big = np.sum(countValues[7:13])
    if action % 13 < 7:
        if small > big:
            value += 1
    else:
        if big > small:
            value += 1
    return value + np.abs(action % 13 - 6)


@njit()
def Test(state, per):
    validActions = env.getValidActions(state)
    validActions = np.where(validActions == 1)[0]

    valueOfActions = np.zeros_like(validActions)
    pCards = np.where(state[0:52] == 1)[0]
    for i in range(len(validActions)):
        valueOfActions[i] = valueOf(validActions[i], pCards)
    action = validActions[np.argmax(valueOfActions)]
    return action, per
