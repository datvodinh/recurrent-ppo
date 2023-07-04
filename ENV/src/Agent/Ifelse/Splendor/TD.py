import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


@njit()
def DataAgent():
    per = List()
    per.append(np.zeros(5))
    return per


@njit()
def nhamThe(state):
    arr_card = state[36:201].reshape(15, 11)
    ngLieuCanGet = np.zeros(5)
    check = False

    for i in range(14, 3, -1):  ###quan tâm tới thẻ cấp 2,3, hold
        card = arr_card[i]
        ngLieuCanGet = env.getValidActions(state)[:5]
        cost = card[6:11]
        ngLieuCanGet[cost == 0] = np.zeros(cost[cost == 0].size)
        ngLieu = state[201:206] + state[207:212]

        arr = cost - ngLieuCanGet - ngLieu
        number = sum(arr[arr > 0]) - state[6]  # so nguyen lieu cần lấy
        arrGet = cost - ngLieu
        ngLieuCanGet[arrGet <= 0] = np.zeros(
            arrGet[arrGet <= 0].size
        )  #  các nguyen lieu can lay <= 3
        if number <= 0 and ngLieuCanGet[ngLieuCanGet == 1].size <= 3:
            check = True
            break

    if check:
        return ngLieuCanGet
    return np.zeros(5)


@njit()
def checkGetStock(state):  # lay đủ 3 nguyên liệu
    stockBoard = state[:5] + state[258:263]
    stockBanDau = state[201:206] - state[258:263]

    if sum(stockBanDau) + state[206] <= 7 and stockBoard[stockBoard > 0].size >= 3:
        return True
    return False


@njit()
def Test(state, per):
    validActions = env.getValidActions(state)
    #  Lấy thẻ-----------------------------------
    arr = validActions[5:95]
    if sum(arr):
        actions = np.where(arr)[0]
        action = actions[-1] + 5
        return action, per

    #  Nhặt nguyên liệu----------------------------------
    stockOnHand = state[258:263]
    if sum(stockOnHand) == 0:
        per[0] = nhamThe(state)

    if checkGetStock(state):
        action = 0
        actionGetStock = validActions[:5]
        actGet = per[0]
        if sum(actGet) != 0:
            action = np.argmax(actGet)
            per[0][action] = 0
        else:
            stockUuTien = np.array([4, 3, 0, 1, 2])
            for i in stockUuTien:
                if actionGetStock[i] and stockOnHand[i] == 0:
                    action = i
                    break
        return action, per

    validActions = np.where(validActions)[0]
    idx = np.random.randint(len(validActions))
    action = validActions[idx]
    return action, per
