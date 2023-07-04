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
    per.append(np.zeros((1, 1)))
    return per


@njit
def civCard(state):
    return state[14:110].reshape(4, 24)


@njit
def buildingCard(state):
    return state[110:142].reshape(4, 8)


@njit
def infor(state):
    return state[142:318].reshape(4, 44)


@njit
def checkCiv(state, validActions):
    myInfor = infor(state)[0]
    go = myInfor[5]
    if sum(state[326:330]) + sum(state[322:326]) >= 2:
        return 0

    if go:
        for i in range(4):
            action = i + 19
            card = civCard(state)[i]
            green = card[11:19]
            green = np.where(green)[0]
            if action in validActions:
                if card[22] or card[23] or card[21]:
                    return action
                if green.size:
                    return action
    return 0


@njit
def thuHoach(state, validActions):
    arr1 = env.getValidActions(state)[29:37]
    arr2 = env.getValidActions(state)[48:56]
    action = 0
    if arr1[arr1 > 0].size:
        policy = np.array([8, 7, 6, 2, 3, 4, 5, 1])
        arrActions = arr1 * policy
        action = np.argmax(arrActions) + 29

    elif arr2[arr2 > 0].size:
        policy = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        arrActions = arr2 * policy
        action = np.argmax(arrActions) + 48

    return action


@njit
def datLayNgL(state, validActions):
    arr = np.array([3, 2, 1])
    arr = arr * env.getValidActions(state)[15:18]
    action = 0
    if sum(arr):
        act = np.argmax(arr)
        action = act + 15
    return action


@njit
def chonXX(state, validActions):
    arr = np.arange(6) + 1
    arrAct = arr * env.getValidActions(state)[57:63]
    action = 0
    if sum(arrAct):
        action = np.argmax(arrAct) + 57
    return action


@njit
def chonNgL(state, validActions):
    arr = np.arange(4) + 1
    arrAct = arr * env.getValidActions(state)[64:68]
    action = 0
    if sum(arrAct):
        action = np.argmax(arrAct) + 64
    return action


@njit
def checkLuongThuc(state, validActions):
    myInfor = infor(state)[0]
    check = myInfor[2] - myInfor[1] - myInfor[3]
    if check >= 0 and 18 in validActions:
        return 18
    return 0


@njit
def checkCongCu(state, validActions):
    arr1 = env.getValidActions(state)[37:40]
    arr2 = env.getValidActions(state)[44:47]
    arrAct = np.array([37, 38, 39, 44, 45, 46])
    action = 0
    if sum(arr1) + sum(arr2):
        for i in arrAct:
            if i in validActions:
                action = i
    return action


@njit
def traNgL(state, validActions):
    arr = np.arange(4) + 1
    arr = arr * env.getValidActions(state)[40:44]
    phase = state[412:423]
    action = 0
    if sum(arr) and phase[3]:
        act = np.argmax(arr)
        action = act + 40
        return action
    return action


@njit
def traNgLNuoi(state, validActions):
    arr = np.array([4, 3, 2, 1])
    arr = arr * env.getValidActions(state)[40:44]
    phase = state[412:423]
    action = 0
    if sum(arr) and phase[10]:
        act = np.argmax(arr)
        action = act + 40
        return action
    return action


@njit
def Test(state, per):
    validActions = env.getValidActions(state)
    validActions = np.where(validActions)[0]

    # Đặt dân --------------------------------------
    phase = state[412:423]
    myInfor = infor(state)[0]
    if 11 in validActions and myInfor[1] < 10:  #  lúa
        return 11, per
    if (
        13 in validActions and myInfor[2] < 8 and (myInfor[2] - myInfor[1]) > 5
    ):  #  người
        return 13, per
    if myInfor[33] == 0:  # đặt công cụ
        if 12 in validActions:
            return 12, per

    if checkCiv(state, validActions):
        action = checkCiv(state, validActions)
        return action, per

    if 14 in validActions:
        return 14, per

    if checkLuongThuc(state, validActions):
        return 18, per

    if datLayNgL(state, validActions):
        action = datLayNgL(state, validActions)
        return action, per

    taiNguyen = state[354:370]
    if sum(taiNguyen[3:8]) and phase[1]:
        arr = np.array([3, 2, 1])
        for action in arr:
            if action in validActions:
                return action, per

    if checkCongCu(state, validActions):
        action = checkCongCu(state, validActions)
        return action, per

    # Thu hoạch -------------------------------------
    if thuHoach(state, validActions):
        action = thuHoach(state, validActions)
        return action, per
    if traNgL(state, validActions):
        action = traNgL(state, validActions)
        return action, per

    # Nuôi bộ tộc -----------------------------------
    if 27 in validActions:
        return 27, per
    if traNgLNuoi(state, validActions):
        action = traNgLNuoi(state, validActions)
        return action, per

    # chon xuc xac
    if chonXX(state, validActions):
        action = chonXX(state, validActions)
        return action, per
    # chon nguyen lieu
    if chonNgL(state, validActions):
        action = chonNgL(state, validActions)
        return action, per

    action = np.random.choice(validActions)
    return action, per
