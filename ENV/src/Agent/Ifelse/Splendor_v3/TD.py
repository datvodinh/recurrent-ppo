import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


@njit()
def DataAgent():
    per = []
    per.append(np.zeros(1))
    per.append(np.zeros(11))
    return per


@njit()
def allCard(state):
    card12 = state[18:150].reshape(12, 11).copy()
    cardHold = state[175:208].reshape(3, 11).copy()
    arrCard = np.concatenate((card12, cardHold))
    return arrCard


@njit()
def theCap2(state, validActions):
    const = state[12:17]
    ngLieu = state[6:11] + const
    auto = state[11]
    actionColor = 0
    actionGtri = 0
    gTriMin = 15
    for i in range(4, 8):
        if i in validActions:
            card = allCard(state)[i]
            ngLthe = card[6:11]
            #  màu thẻ
            color = card[1:6]
            color = np.where(color)[0][0]
            arrColor = np.array([0, 3, 4])
            # giá trị
            ngLieuMua = card[6:11] - ngLieu
            gTri = sum(ngLieuMua > 0) - auto
            ngLmax = np.max(ngLthe)
            ngL = np.where(ngLthe == ngLmax)[0][0]
            if (ngLthe[ngL] - const[ngL] <= 4 and ngLmax >= 5) or ngLmax < 5:
                if gTriMin > gTri:
                    gTriMin = gTri
                    actionGtri = i
                    if color in arrColor:
                        actionColor = i

    if actionColor:
        return actionColor
    elif actionGtri:
        return actionGtri
    else:
        return validActions[-1]  # phòng trường hợp xấu nhất


@njit()
def theCap3(state, validActions):
    const = state[12:17]
    ngLieu = state[6:11] + const
    auto = state[11]

    action = 0
    ngLcanMin = 7  #  số nguyên liệu cần của thẻ lớn hơn 10 ngL
    ngLcanMinM = 4  #  số nguyên liệu cần của màu nhiều nhất
    for i in range(8, 12):
        if i in validActions:
            card = allCard(state)[i]
            ngLthe = card[6:11]
            # gtri
            sumNgLieu = sum(card[6:11])

            if sumNgLieu > 10:
                ngLieuCan = ngLthe - const
                if sum(ngLieuCan) <= ngLcanMin:
                    ngLcanMin = sum(ngLieuCan)
                    action = i
            else:
                ngLmax = np.max(ngLthe)
                ngL = np.where(ngLthe == ngLmax)[0][0]
                if ngLthe[ngL] - const[ngL] <= ngLcanMinM:
                    ngLcanMinM = ngLthe[ngL] - const[ngL]
                    action = i

    return action


@njit()
def lay3ngL(card, ngLieu, auto, ngLban):
    ngLban_ = np.zeros(5)
    ngLban_[ngLban > 0] = np.ones(len(ngLban[ngLban > 0]))
    arr = card[6:11] - ngLieu
    ngLcan = sum(arr[arr > 0]) - auto
    if ngLcan > 3:
        return False

    arr = card[6:11] - ngLieu - ngLban_
    ngLcan = sum(arr[arr > 0]) - auto
    if ngLcan > 0:
        return False

    return True


@njit()
def lay2ngL(card, ngLieu, auto, ngLban):
    arr = card[6:11] - ngLieu
    ngLcan = sum(arr[arr > 0]) - auto
    if ngLcan > 2:
        return False
    if arr[arr > 0].size != 1:
        return False

    color = np.where(arr > 0)[0][0]
    if ngLban[color] < 4:
        return False
    return True


@njit()
def checkCard(state, validActions):
    const = state[12:17]
    ngLieu = state[6:11] + const
    auto = state[11]
    ngLban = state[:5]
    if const[const > 0].size == 5:
        for i in range(15, 4, -1):
            idx = i - 1
            if idx in validActions:
                card = allCard(state)[idx]
                if lay3ngL(card, ngLieu, auto, ngLban) or lay2ngL(
                    card, ngLieu, auto, ngLban
                ):
                    return idx
    else:
        for i in range(15, 0, -1):
            idx = i - 1
            if idx in validActions:
                card = allCard(state)[idx]
                color = card[1:6]
                color = np.where(color)[0][0]
                if (card[0] == 0 and const[color] == 0) or card[0]:
                    if lay3ngL(card, ngLieu, auto, ngLban) or lay2ngL(
                        card, ngLieu, auto, ngLban
                    ):
                        return idx

    return 0


@njit()
def theHold(state, validActions):
    ngLieu = state[6:11] + state[12:17]
    ngLban = state[:5]
    auto = state[11]
    action = 0
    p = 0
    for i in validActions[validActions >= 12]:
        point = 0
        card = allCard(state)[i]
        ngLcan = card[6:11] - ngLieu
        ngLcan = np.where(ngLcan > 0)[0]
        for nl in ngLcan:
            if ngLban[nl]:
                point += 1
        if point > p:
            p = point
            action = i
    return action


@njit()
def getCard(state, validActions):
    const = state[12:17]
    ngLieu = state[6:11] + const
    ngLban = state[:5]
    auto = state[11]
    action = 0
    point = 0
    for i in range(15, 0, -1):
        idx = i - 1
        if idx in validActions:
            card = allCard(state)[idx]
            ngLcan = card[6:11] - ngLieu
            number = sum(ngLcan[ngLcan > 0]) - auto

            if number <= 0:
                if point < card[0]:
                    point = card[0]
                    action = idx
                elif point == card[0]:  #  bằng điểm thì xét màu
                    color = card[1:6]
                    color = np.where(color)[0][0]
                    arrColor = np.array([0, 3, 4])
                    if color in arrColor or const[color] == 0:
                        action = idx

    return action


@njit()
def Test(state, per):
    if env.getReward(state) != -1:
        per = DataAgent()

    validActions = env.getValidActions(state)
    validActions = np.where(validActions)[0]
    arrCard = allCard(state)
    const = state[12:17]

    if getCard(state, validActions):  # lấy được thẻ nào thì lấy luôn
        action = getCard(state, validActions)
        return action, per

    if checkCard(state, validActions):  # lấy nguyên liệu để turn sau ăn
        action = checkCard(state, validActions)
        #  print( '--checkCard:', action, arrCard[action])
        return action, per

    #  action = int(per[0][0])
    #  arr1 = per[1]
    #  arr2 = arrCard[action]
    #  if np.array_equal( arr1, arr2): #  nhắm thẻ đang nhắm
    #    #  print( '--1:', action, arrCard[action])
    #    return action, per

    if theHold(state, validActions):  # lấy nguyên liệu để ăn thẻ hold
        action = theHold(state, validActions)
        #  per[0][0] = action
        #  per[1] = arrCard[action]
        #  print( '--2:', action, arrCard[action])
        return action, per

    if theCap3(state, validActions):  #  lấy nguyên liệu ăn thẻ cấp 3
        action = theCap3(state, validActions)
        #  print( 't3: ', action, allCard(state)[action])
    else:
        action = theCap2(state, validActions)  #  lấy nguyên liệu ăn thẻ cấp 3
        #  print( 't2: ', action, allCard(state)[action])

    per[0][0] = action
    per[1] = arrCard[action]
    return action, per
