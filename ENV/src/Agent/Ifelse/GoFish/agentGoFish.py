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
    per.append(np.zeros((1, 4)))
    per.append(np.zeros((3, 13)))
    per.append(np.zeros((3, 13)))
    per.append(np.zeros((1, 2)))
    return per


@njit
def checkEndTurn(state, per):
    if state[61] == 1:
        preNum = per[0][0][0]
        per[0][0][0] = state[60]
        pre1 = per[0][0][1]
        pre2 = per[0][0][2]
        per[0][0][1] = state[13]
        per[0][0][2] = state[14]

        if preNum - state[60] >= 2:
            return True
        if state[60] == 0:
            if (
                state[13] - pre1 <= 0 and state[14] - pre2 == 0
            ):  ###  số lá giảm và điểm không đổi
                return True

    return False


@njit
def duDoan(state, per):
    arr = state[67:106].reshape(3, 13)
    for i in range(3):
        arr_0 = arr[i]
        num_0 = np.where(arr_0)[0]
        if num_0.size > 0:
            for k in num_0:
                per[1][i][k] = 1
                p2 = (i + 5) % 3
                p1 = (i + 4) % 3
                if per[1][p1][k] == 1:
                    per[1][p1][k] = 0
                if per[1][p2][k] == 1:
                    per[1][p2][k] = 0

    arr = np.zeros((3, 13)).T
    arr_idx = np.sum(per[1], 0)
    for i in range(13):
        if arr_idx[i] > 0:
            arr[i] = np.ones(3)
    per[2] = arr.T

    return per


@njit
def comboActions(myCard, arr, arr1):
    arr_idx = np.where(myCard)[0]

    arr_T = arr.T
    size = arr_idx.size
    for i in arr_idx[size - 1 :: -1]:
        if sum(arr_T[i]) > 0:
            act = np.where(arr_T[i])[0][0]
            return act + 1, i + 4

    arr_T = arr1.T
    size = arr_idx.size
    for i in arr_idx[size - 1 :: -1]:
        if sum(arr_T[i]) > 0:
            act = np.where(arr_T[i])[0][0]
            return act + 1, i + 4
    return 0, 0


@njit
def Test(state, per):
    if env.getReward(state) != -1:
        per = DataAgent()

    validActions = env.getValidActions(state)
    actions = np.where(validActions)[0]

    phase = state[61:64]

    myCard = state[:13]
    myCard[myCard == 4] -= 4

    if checkEndTurn(state, per):
        per = duDoan(state, per)

    for i in range(3):
        if state[28 + i * 15] == 0:
            per[1][i] = np.zeros(13)
            per[2][i] = np.zeros(13)

    if phase[0]:
        per[3][0][0], per[3][0][1] = comboActions(myCard, per[1], per[2])

    act_p0 = int(per[3][0][0])
    act_p1 = int(per[3][0][1])
    if phase[0]:
        if act_p0:
            if act_p0 in actions:
                return act_p0, per

        action = np.random.choice(actions)
        return action, per

    if phase[1]:
        if act_p1:
            per[1][act_p0 - 1][act_p1 - 4] = 0
            per[2][act_p0 - 1][act_p1 - 4] = 0

            if act_p1 in actions:
                return act_p1, per

    action = actions[-1]
    return action, per
