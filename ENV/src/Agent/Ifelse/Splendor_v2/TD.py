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
    per.append(np.zeros(5))
    return per


@njit()
def getCardOnBoard(p_state):  #  12 thẻ đang có trên bàn
    return p_state[18:150].reshape(12, 11).copy()


@njit()
def getNoble(p_state):  #  5 thẻ Noble
    return p_state[150:175].reshape(5, 5).copy()


@njit()
def getCardOnHold(p_state):
    return p_state[175:208].reshape(3, 11).copy()


@njit()
def allCardYouCanGet(p_state):  # tất cả các thẻ có thẻ mua: 12 + 3 thẻ đã úp
    arr_card = np.concatenate((getCardOnBoard(p_state), getCardOnHold(p_state)), axis=0)
    return arr_card


@njit()
def checkGetCard(state):
    actCard = env.getValidActions(state)[1:16]
    actCard = np.where(actCard)[0]
    arr_card = allCardYouCanGet(state)
    size = actCard.size
    action = 0
    if size != 0:
        for idx in range(size, 0, -1):
            value = actCard[idx - 1]
            card = arr_card[value]
            if card[0] > 0 or value >= 12:  # có điểm hoặc đang hold thì ăn luon
                action = value + 1
            else:
                stockConst = state[12:17]
                if (
                    stockConst[stockConst == 0].size == 0
                ):  # đã có đủ các thẻ của mỗi nguyên liệu chưa
                    if card[1] or card[4] or card[5]:
                        action = value + 1
                else:
                    action = value + 1

            if action:
                break
    return action


@njit()
def nhamThe(state):
    arr_card = allCardYouCanGet(state)
    ngLieuCanGet = np.zeros(5)
    check = False

    for i in range(14, 3, -1):  ###quan tâm tới thẻ cấp 2,3, hold
        card = arr_card[i]
        ngLieuCanGet = env.getValidActions(state)[31:36]
        cost = card[6:11]
        ngLieuCanGet[cost == 0] = np.zeros(cost[cost == 0].size)
        ngLieu = state[6:11] + state[12:17]

        arr = cost - ngLieuCanGet - ngLieu
        number = sum(arr[arr > 0]) - state[11]  # so nguyen lieu cần lấy
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
    stockBoard = state[:5] + state[208:213]
    stockBanDau = state[6:11] - state[208:213]

    if sum(stockBanDau) + state[11] <= 7 and stockBoard[stockBoard > 0].size >= 3:
        return True
    return False


###cần thiết thì hold thẻ đang chuẩn bị ăn
@njit()
def checkHold(state):
    actHold = env.getValidActions(state)[16:31]
    if sum(actHold) and state[11] < 3 and sum(state[6:12]) < 10:
        return True
    return False


@njit()
def holdCard(state):
    card = getCardOnBoard(state)
    actHold = env.getValidActions(state)[16:28]
    stt = np.where(actHold == 1)[0]
    size = stt.size

    number = 15
    action = 29
    for idx in range(size, 0, -1):
        k = stt[idx - 1]
        if k in range(4, 8):
            cost = card[k][6:11]
            ngLieu = state[6:11] + state[12:17]
            arr = cost - ngLieu
            arr[arr <= 0] = np.zeros(arr[arr <= 0].size)
            if sum(arr) < number and arr[1] + arr[2] <= 2:
                number = sum(arr[arr > 0])
                action = k + 16

    if action >= 16 and action <= 27:
        return action  ###  hold thẻ cấp 3 ẩn
    else:
        if env.getValidActions(state)[29]:
            return 29
        else:
            return stt[np.random.randint(size)] + 16


@njit()
def Test(state, per):
    validActions = env.getValidActions(state)
    actions = np.where(validActions)[0]
    #  print(state)
    #  print(actions)

    # Lấy thẻ-------------------------------------------
    if checkGetCard(state):
        action = checkGetCard(state)
        #  print('Lay the: ', action)
        return action, per

    # Nhặt nguyên liệu----------------------------------
    stockOnHand = state[208:213]
    if sum(stockOnHand) == 0:
        per[0] = nhamThe(state)

    if checkGetStock(state):
        action = 0
        actionGetStock = validActions[31:36]
        actGet = per[0]
        if sum(actGet) != 0:
            action = np.argmax(actGet)
            per[0][action] = 0
            action += 31
        else:
            stockUuTien = np.array([4, 3, 0, 1, 2])
            for i in stockUuTien:
                if actionGetStock[i] and stockOnHand[i] == 0:
                    action = i + 31
                    break
        #  print('Lay nguyen lieu: ', action)
        return action, per

    # Hold thẻ
    if checkHold(state):
        actHold = holdCard(state)
        #  print('Hold: ',   actHold)
        return actHold, per

    #  print(0)
    #  return 0, per
    idx = np.random.randint(len(actions))
    #  print( '-----', actions[idx])
    return actions[idx], per
