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
    per.append(np.zeros(1))
    per.append(np.zeros(3))
    return per


@njit()
def checkSa(state, validActions, per):
    mySushi = state[16:28]
    turn = state[1] - (state[0] - 1) * 7
    sa = mySushi[1]

    if per[1][1] == 0 and sa == 0 and turn >= 4:
        return False
    if per[1][1] == 1 and sa == 0 and turn >= 5:
        return False

    if 1 in validActions:
        if turn == 6 and sa == 0:
            return False
        if turn == 7 and sa <= 1:
            return False
        if sa < 3:
            return 1

    return False


@njit()
def checkSquPud(state, validActions):
    if 7 in validActions:
        return 7
    if 9 in validActions and state[14 + 1] < 4:
        if state[14 + 3] == 2 and 1 in validActions:
            return 0
        else:
            return 9
    return 0


@njit()
def checkNigiri(state, validActions):
    wa = state[16 + 10]
    if wa:
        arr = np.array([7, 6, 8])
        for i in arr:
            if i in validActions:
                return i
    return 0


@njit()
def checkMaki(state, validActions):
    mySushi = state[16:28]
    if mySushi[3] + mySushi[4] * 2 + mySushi[5] * 2 < 8:
        if 5 in validActions:
            return 5
        if 4 in validActions:
            return 4
    return 0


@njit()
def checkSal(state, validActions):
    arr = np.array([6])
    for i in arr:
        if i in validActions:
            return i
    return 0


@njit()
def checkTempura(state, validActions, per):
    mySushi = state[16:28]
    turn = state[1] - (state[0] - 1) * 7
    te = mySushi[0]

    if per[1][0] == 0 and (te == 0 or te == 2) and turn >= 5:
        return False
    if 0 in validActions:
        return True
    return False


@njit()
def checkDumpling(state, validActions, per):
    mySushi = state[16:28]
    turn = state[1] - (state[0] - 1) * 7
    du = mySushi[0]

    if per[1][2] == 0 and du == 0 and turn >= 5:
        return False
    if 2 in validActions:
        return True
    return False


@njit()
def checkMaki1(state, validActions):
    mySushi = state[16:28]
    myMaki = mySushi[3] + mySushi[4] * 2 + mySushi[5] * 3
    makiP = np.zeros(4)
    for i in range(4):
        sushiP = state[30 + i * 14 : 42 + i * 14]
        makiP[i] = sushiP[3] + sushiP[4] * 2 + sushiP[5] * 3

    sorted_makiP = np.sort(makiP)[::-1]
    max = sorted_makiP[0]
    arr = np.where(sorted_makiP == max)[0]
    size = arr.size
    if 3 in validActions:
        max2 = sorted_makiP[size]
        if max2 - myMaki == 1 or max - myMaki <= 1:
            return 3

    return 0


@njit()
def Test(state, per):
    #  if per[0][0] != state[0]:
    #    per[0][0] = state[0]
    #    print('-----Vong', state[0], '------------------')
    #  print(state[1])

    if env.getReward(state) != -1:
        per = DataAgent()
        #  print('điểm')
        #  for i in range(5):
        #    print(state[14 + i*14: 16 + i*14])

    if per[0][0] != state[0]:
        per[0][0] = state[0]
        per[1] = np.zeros(3)

    validActions = env.getValidActions(state)
    validActions = np.where(validActions)[0]

    turn = state[1] - (state[0] - 1) * 7
    if turn <= 2:
        if 0 in validActions:
            per[1][0] += 1
        if 1 in validActions:
            per[1][1] += 1
        if 2 in validActions:
            per[1][2] += 1

    if checkNigiri(state, validActions):  #  kích hoạt Wasabi
        action = checkNigiri(state, validActions)
        #  print( 'nigiri', action, validActions)
        return action, per

    if state[17] == 2 and 1 in validActions:  ##  nếu có sashimi thì nhắt thêm nữa
        action = 1
        #  print( 'sashimi',action, validActions)
        return action, per

    if state[18] >= 3 and 2 in validActions:  ##  nhặt dumpling
        action = 2
        return action, per

    if checkSquPud(state, validActions):  #  lấy 3 điểm và nhiều pudding
        action = checkSquPud(state, validActions)
        #  print( 'squpud', action, validActions)
        return action, per

    if (
        state[16] == 1 or state[16] == 3
    ) and 0 in validActions:  ##  nếu có tempura thì nhắt thêm nữa
        action = 0
        #  print( 'tempura',action, validActions)
        return action, per

    if checkSa(state, validActions, per):  # lấy Sashimi
        action = checkSa(state, validActions, per)
        #  print( 'sashimi', action, validActions)
        return action, per

    if 10 in validActions and turn < 7:  #  lấy wasabi
        action = 10
        #  print( 'wasabi',action, validActions)
        return 10, per

    if checkMaki(state, validActions):  #  maki
        action = checkMaki(state, validActions)
        #  print( 'maki',action, validActions)
        return action, per

    if checkDumpling(state, validActions, per):  # dumpling
        action = 2
        #  print( 'dumpling', action, validActions)
        return action, per

    if checkMaki1(state, validActions):
        action = 3
        return action, per

    if checkSal(state, validActions):
        action = checkSal(state, validActions)
        #  print( 'salegg',action, validActions)
        return action, per

    if checkTempura(state, validActions, per):  ###  nhặt tempura
        action = 0
        #  print( 'te',action, validActions)
        return 0, per

    if 8 in validActions:
        action = 8
        return action, per

    if validActions.size > 1:  # xóa sashimi
        idx = np.where(validActions == 1)[0]
        if idx.size:
            #  print('xóa')
            validActions = np.delete(validActions, idx[0])

    if validActions.size > 1:  # xóa tempura
        idx = np.where(validActions == 0)[0]
        if idx.size:
            #  print('xóa')
            validActions = np.delete(validActions, idx[0])

    if validActions.size > 1:  # xóa đũa
        idx = np.where(validActions == 11)[0]
        if idx.size:
            #  print('xóa')
            validActions = np.delete(validActions, idx[0])

    action = np.random.choice(validActions)
    #  print( 'random',action, validActions)
    return action, per
