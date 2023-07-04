import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


from src.Base.CatanNoExchange.env import *


@njit
def DataAgent():
    per = []
    per.append(np.zeros(1))
    return per


@njit
def dinhKe(x):  ###  các đỉnh kề với đỉnh x
    x = int(x)
    dinhKe = POINT_POINT[x]
    dinhKe = dinhKe[dinhKe != -1]
    return dinhKe


@njit
def point_tile(point):
    point = int(point)
    vungKe = POINT_TILE[point]
    vungKe = vungKe[vungKe != -1]
    return vungKe


@njit
def point_port(p):
    p = int(p)
    for i in range(9):
        arr = PORT_POINT[i]
        if p in arr:
            return i
    return 0


@njit
def featureTile(state, tile):
    x = int(tile)
    feature = np.zeros(7)  #  số, cây, gạch, cừu, lúa, đá, sa mạc
    ngLcacTile = state[:114].reshape(19, 6)
    feature[1:7] = ngLcacTile[x]

    numberTile = state[133:361].reshape(19, 12)
    number = numberTile[x]
    num = np.where(number)[0][0]
    feature[0] = 6 - abs(num - 7)
    return feature


@njit
def datNha1(state, validActions):
    action = -1
    max = 0
    for act in validActions:
        total = np.zeros(7)
        checkSaMac = 0
        if act in range(30, 54):
            for tile in point_tile(act):
                fea = featureTile(state, tile)
                if fea[-1]:
                    checkSaMac = 1  # samac
                if fea[4] or fea[5]:  # lúa or đá ------------------có thể thêm gạch
                    total += fea
                #  else:
                #    total[0] += fea[0]

            if checkSaMac == 0 and max < total[0]:
                action = act
                max = total[0]

    if action != -1:
        return action
    else:
        action = np.random.choice(validActions)
        return action


@njit
def datNha2(state, validActions):
    action = -1
    # thông tin nhà 1
    myInfor = state[421:629]
    nha1 = myInfor[83:137]
    nha1 = np.where(nha1)[0][0]
    arr = np.zeros(5)  ###  xác suất các ngL
    for tile in point_tile(nha1):
        feature = featureTile(state, tile)
        ngL = feature[1:6]
        ngL = np.where(ngL)[0][0]
        arr[ngL] += feature[0]

    ngLtot = np.argmax(arr)  # ngL tốt nhất= có xác suất cao nhất
    allPort = state[361:415].reshape(9, 6)

    # Vị trí cảng cần tìm
    portToiUu = np.array([1, 4, 11, 14, 21, 24])
    for i in portToiUu:
        if i in validActions:
            idx = point_port(i)
            inforPort = allPort[idx]  ###  thông tin cảng đang xét
            if inforPort[ngLtot]:
                action = i
                return action

    numPort = np.array(
        [0, 1, 4, 5, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28]
    )
    # tìm cangr 3/1
    max = 0
    for i in numPort:
        if i in validActions:
            idx = point_port(i)
            inforPort = allPort[idx]
            if inforPort[-1]:
                m = 0
                for tile in point_tile(i):
                    m += featureTile(state, tile)[0]
                if m > max:
                    m = max
                    action = i
    if action != -1:
        return action
    else:
        action = np.random.choice(validActions)
        return action


@njit
def diChuyenRobber(state, validActions):
    myInfor = state[421:629]
    house = myInfor[83:137] + myInfor[137:191]
    house = np.where(house)[0]

    #  những ô kề với nhà của mình
    tileKeNha = np.zeros(19)
    for h in house:
        arrTile = point_tile(h)
        size = arrTile.size
        tileKeNha[arrTile] = np.ones(size)
    #  những ô kề ng chơi khác:------------------------

    #  ngL thiếu
    allNgL = myInfor[:5]
    ngLthieu = np.min(allNgL)
    if allNgL[3] < 2 and allNgL[4] >= 3:
        ngLthieu = 3
    if allNgL[3] >= 2 and allNgL[4] < 3:
        ngLthieu = 4
    if np.where(allNgL[:4] == 0)[0].size == 1:
        ngLthieu = np.where(allNgL[:4] == 0)[0][0]

    #  turn dichuyen Robber
    action = 0
    for i in range(64, 83):
        if i in validActions and tileKeNha[i - 64] == 0:
            tile = i - 64
            feature = featureTile(state, tile)
            if feature[int(ngLthieu) + 1]:
                action = i
                return action
    return 0


@njit
def checkBuildRoad(state, validActions):
    if 83 in validActions:
        #  Khu của bản thân-----------------
        myInfor = state[421:629]
        nhaToi = myInfor[83:137] + myInfor[137:191]  #  nhà
        #  Đường
        myRoad = myInfor[11:83]
        khuCuaToi = np.zeros(54)
        for i in np.where(myRoad)[0]:
            khuCuaToi[ROAD_POINT[i]] = np.array([1, 1])

        #  khu của player khác-----------------
        nhaPlayer = np.zeros(54)  #  nhà
        for i in range(3):
            player_ = (
                state[629 + 185 * (i - 1) + 75 : 629 + 185 * (i - 1) + 129]
                + state[629 + 185 * (i - 1) + 129 : 629 + 185 * (i - 1) + 183]
            )
            nhaPlayer += player_

        # Lân cận nhà của tôi
        lanCanNhaToi = nhaToi
        for i in np.where(nhaToi)[0]:
            lanCanNhaToi[dinhKe(i)] = np.ones(len(dinhKe(i)))

        # Lân cận nhà của player khác
        lanCanNhaPlayer = nhaPlayer
        for i in np.where(nhaPlayer)[0]:
            lanCanNhaPlayer[dinhKe(i)] = np.ones(len(dinhKe(i)))

        arr = (khuCuaToi - lanCanNhaToi) * (
            1 - lanCanNhaPlayer
        )  #  những chỗ có thể xây nhà
        if arr[arr > 0].size == 0:
            return True

    return False


@njit
def dungKnightTruoc(state, validActions):
    if 54 in validActions and 55 in validActions:
        #  Khu của bản thân-----------------
        myInfor = state[421:629]
        nhaToi = myInfor[83:137] + myInfor[137:191]  #  nhà
        vungToi = np.zeros(19)  # -------------
        for i in np.where(nhaToi)[0]:
            vungToi[point_tile(i)] = np.ones(len(point_tile(i)))

        if sum(state[114:133] * vungToi) == 1:
            return 55
        return 54
    else:
        return 0


@njit
def tradeBank(state, validActions):
    myInfor = state[421:629]
    allNgL = myInfor[:5]
    infPort = myInfor[193:208].reshape(5, 3)
    for action in validActions:
        if action >= 89 and action < 94:
            ngL = action - 89
            if ngL > 2:
                if allNgL[ngL] >= 5 and (infPort[ngL][0] or infPort[ngL][1]):
                    return action
                if allNgL[ngL] >= 6 and infPort[ngL][2]:
                    return action
            if ngL <= 2:
                if allNgL[ngL] >= 4:
                    return action
    return 0


@njit
def checkBuyDev(state, validActions):
    myInfor = state[421:629]
    if 86 in validActions:
        if myInfor[5] < 2:
            return True
        nha = myInfor[83:137]
        if sum(nha) == 0:  ###  nếu không còn xây được thành phố
            return True
    return False


@njit
def devCard(state, validActions):
    for i in range(56, 59):
        if i in validActions:
            return i
    if 55 in validActions and state[421 + 5] > 1:
        return 55
    return 0


@njit
def Test(state, per):
    if env.getReward(state) != -1:
        per[0][0] = 0
    else:
        per[0][0] += 1

    validActions = env.getValidActions(state)
    validActions = np.where(validActions)[0]
    phase = state[1273:1286]

    if per[0][0] == 1:  #  đặt nhà đầu tiên
        action = datNha1(state, validActions)
        #  print(action)
        return action, per

    if per[0][0] == 3:  #  đặt nhà thứ hai
        action = datNha2(state, validActions)
        #  print(action)
        return action, per

    if 85 in validActions:
        return 85, per

    if 84 in validActions:
        return 84, per

    if phase[8] or phase[9]:
        max = 0
        act = -1
        for p in validActions:
            if p < 54:
                total = 0
                for tile in point_tile(p):
                    total += featureTile(state, tile)[0]
                if total > max:
                    max = total
                    act = p
        if act != -1:
            return act, per

    if checkBuildRoad(state, validActions):
        return 83, per

    if dungKnightTruoc(state, validActions):
        action = dungKnightTruoc(state, validActions)
        return action, per

    if 94 in validActions:
        myInfor = state[421:629]
        kho = state[1268:1273]
        #  ngL thiếu
        allNgL = myInfor[:5]
        ngLthieu = -1
        if allNgL[-2] >= 2 and allNgL[-1] < 3:
            ngLthieu = 4
        if allNgL[-2] < 2 and allNgL[-1] >= 3:
            ngLthieu = 3
        if np.where(allNgL[:4] == 0)[0].size == 1:
            ngLthieu = np.where(allNgL[:4] == 0)[0][0]
        if ngLthieu != -1 and kho[int(ngLthieu)]:
            return 94, per

    # the Dev
    if checkBuyDev(state, validActions):
        return 86, per
    if devCard(state, validActions):
        action = devCard(state, validActions)
        return action, per

    if diChuyenRobber(state, validActions):
        action = diChuyenRobber(state, validActions)
        return action, per

    if phase[11] or phase[7]:
        myInfor = state[421:629]
        #  ngL thiếu
        allNgL = myInfor[:5]
        ngLthieu = 1
        if allNgL[-2] >= 2 and allNgL[-1] < 3:
            ngLthieu = 4
        if allNgL[-2] < 2 and allNgL[-1] >= 3:
            ngLthieu = 3
        if np.where(allNgL[:4] == 0)[0].size == 1:
            ngLthieu = np.where(allNgL[:4] == 0)[0][0]
        action = ngLthieu + 59
        if action in validActions:
            return action, per

    if tradeBank(state, validActions):
        action = tradeBank(state, validActions)
        return action, per

    action = validActions[0]
    return action, per
