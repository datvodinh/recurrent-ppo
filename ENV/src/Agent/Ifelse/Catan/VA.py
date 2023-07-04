import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


from src.Base.Catan.env import POINT_TILE


@njit
def DataAgent():
    per = []
    per.append(np.zeros(1))  #  đếm số lượt
    return per


@njit()
def valueOf(position, valueDiceInTile):
    score = 0
    # [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    proba = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]
    adjTilesOfPoss = POINT_TILE[position]
    for pos in adjTilesOfPoss:
        if pos != -1:
            val = valueDiceInTile[:, pos]
            val = np.where(val == 1)[0]
            if len(val) > 0:
                idxVal = val[0]
                score += proba[idxVal]
    return score


@njit()
def Test(state, per):
    validActions = env.getValidActions(state)
    validActions = np.where(validActions == 1)[0]

    selectPositionActions = validActions[(validActions >= 0) & (validActions < 54)]
    if len(selectPositionActions) > 0:
        valueDiceInTile = np.append(state[114:133], state[1048:1219])
        valueDiceInTile = valueDiceInTile.reshape(10, -1)
        valueOfActions = np.zeros_like(selectPositionActions)
        for i in range(len(selectPositionActions)):
            valueOfActions[i] = valueOf(selectPositionActions[i], valueDiceInTile)
        action = selectPositionActions[np.argmax(valueOfActions)]
        return action, per

    for i in (87, 86, 88, 89, 90, 91, 93, 103, 104):
        if i in validActions:
            return i, per

    returnResources = validActions[(validActions >= 95) & (validActions < 100)]
    if len(returnResources) > 0:
        valueOfActions = np.zeros_like(returnResources)
        for i in range(len(returnResources)):
            valueOfActions[i] = state[returnResources[i] - 95 + 193]
        action = returnResources[np.argmax(valueOfActions)]
        return action, per

    action = validActions[np.random.randint(len(validActions))]
    return action, per
