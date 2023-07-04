import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


from src.Base.Catan.env import POINT_POINT, POINT_TILE, ROAD_POINT


@njit
def DataAgent():
    per = []
    per.append(np.zeros(1))  #  đếm số lượt
    return per


@njit
def vungKe(x):  ###  các ô gắn với đỉnh x
    x = int(x)
    vungKe = POINT_TILE[x]
    vungKe = vungKe[vungKe != -1]
    return vungKe


@njit
def dinhKe(x):  ###  các đỉnh kề với đỉnh x
    x = int(x)
    dinhKe = POINT_POINT[x]
    dinhKe = dinhKe[dinhKe != -1]
    return dinhKe


@njit
def thuocTinhVung(state, x):  ###  thuộc tính của ô x
    x = int(x)
    thuocTinh = np.zeros(6)
    ngLieuCacVung = state[:95].reshape(19, 5)
    thuocTinh[1:6] = ngLieuCacVung[x]
    value = np.array([3, 4, 5, 6, 8, 9, 10, 11, 12])
    for i in range(9):
        arr = state[1048 + 19 * i : 1048 + 19 * (i + 1)]
        if arr[x] == 1:
            thuocTinh[0] = value[i]
            break
    arr = state[114:133]
    if arr[x] == 1:
        thuocTinh[0] = 2  ##  xác suất ô = 2

    return thuocTinh


@njit
def xacDinhDuongPhuHop(state, validActions):
    #  nhà của bản thân
    me = state[276:330] + state[330:384]
    #  nhà của player
    player = np.zeros(54)
    for i in range(3):
        player_ = (
            state[391 + 185 * (i - 1) + 75 : 391 + 185 * (i - 1) + 129]
            + state[391 + 185 * (i - 1) + 129 : 391 + 185 * (i - 1) + 183]
        )
        player += player_

    #  những điểm không thể xây nhà vì kề nhà đối phương
    map = np.zeros(54)
    map += player
    arr_nha = np.where(player)[0]
    for k in arr_nha:
        map[dinhKe(k)] = np.ones(len(dinhKe(k)))

    arr_p1 = state[963:1017]  #  xác định mút đầu tiên
    if sum(arr_p1) == 0:
        for i in validActions:
            if me[i]:  #  mút đầu là nhà
                #  những điểm không xây nhà được quanh i
                arr_i = np.zeros(54)
                arr_i[i] = 1
                arr_i[dinhKe(i)] = np.ones(len(dinhKe(i)))

                #  những điểm không xây nhà được quanh các nhà khác của mình trừ nha i
                arr_k = np.zeros(54)
                arr_k += me
                arr_k[i] = 0
                for k in np.where(me)[0]:
                    if k != i:
                        arr_k[dinhKe(k)] = np.ones(len(dinhKe(k)))
                check = False

                for id in dinhKe(i):
                    for id_ in dinhKe(id):
                        if map[id_] == 0 and arr_k[id_] == 0 and arr_i[id_] == 0:
                            check = True
                            return i
                ####-------điểm này dùng được

            else:  #  mút đầu không phải là nhà
                #  những điểm không xây nhà được quanh các nhà của mình
                arr_k = np.zeros(54)
                arr_k += me
                for k in np.where(me)[0]:
                    arr_k[dinhKe(k)] = np.ones(len(dinhKe(k)))
                check = False

                for id in dinhKe(i):
                    if map[id] == 0 and arr_k[id] == 0:
                        check = True
                        return i
                ####-------điểm này dùng được

    else:  #  xác định mút thứ 2
        p1 = np.where(arr_p1)[0][0]
        for i in validActions:
            if map[i] == 0:
                if me[p1] == 0:
                    #  những điểm không xây nhà được quanh các nhà của mình
                    arr_k = np.zeros(54)
                    arr_k += me
                    for k in np.where(me)[0]:
                        arr_k[dinhKe(k)] = np.ones(len(dinhKe(k)))
                    check = False

                    if arr_k[i] == 0:
                        check = True
                        return i
                        ####---------điểm này dùng được
                else:
                    #  những điểm không xây nhà được quanh p1
                    arr_i = np.zeros(54)
                    arr_i[p1] = 1
                    arr_i[dinhKe(p1)] = np.ones(len(dinhKe(p1)))

                    #  những điểm không xây nhà được quanh các nhà khác của mình
                    arr_k = np.zeros(54)
                    arr_k += me
                    arr_k[p1] = 0
                    for k in np.where(me)[0]:
                        if k != p1:
                            arr_k[dinhKe(k)] = np.ones(len(dinhKe(k)))
                    check = False

                    for id_ in dinhKe(i):
                        if map[id_] == 0 and arr_k[id_] == 0 and arr_i[id_] == 0:
                            check = True
                            return i
    action = validActions[np.random.randint(len(validActions))]
    #  print('---- random phase xây đường', action)
    return action


@njit
def firstSettlements(state, validActions):  ###  đặt nhà đầu tiên gần mỏ đá + lúa
    action = -1
    totalToiUu = np.zeros(6)  ###  điểm của đỉnh i

    for i in validActions:
        if i in range(30, 54):
            total = np.zeros(6)
            checkSaMac = 0
            for o in vungKe(i):
                tt = thuocTinhVung(state, o)
                if tt[0] == 0:
                    checkSaMac = 1
                tt[0] = 6 - abs(tt[0] - 7)  ###  điểm của value
                if tt[5] or tt[4]:  # check mỏ đá or lúa
                    total += tt
            if checkSaMac == 0 and totalToiUu[0] < total[0]:
                action = i
                totalToiUu = total

    if action != -1:
        return action
    else:
        return validActions[np.random.randint(len(validActions))]


@njit
def secondSettlements(state, validActions):  ###  đặt nhà thứ hai gần cảng
    # thông tin về nhà 1 để chọn cảng
    stateNha = state[276:330] + state[330:384]
    nha1 = np.where(stateNha)[0][0]
    arr = np.zeros(6)  #  value + 5ngLieu

    for i in range(1, 6):
        gTriNgL = np.zeros(6)
        for vung in vungKe(nha1):
            if thuocTinhVung(state, vung)[i]:
                gTriNgL += thuocTinhVung(state, vung)
                gTriNgL[0] = 6 - abs(thuocTinhVung(state, vung)[0] - 7)
        if arr[0] < gTriNgL[0]:
            arr = gTriNgL

    # Vi tri cảng cần tìm
    ngL = np.where(arr > 0)[0][-1] - 1  ##  nguyên liệu để chọn cảng
    action = -1
    patu = np.array([1, 4, 11, 14, 21, 24])
    typePort = np.array([0, 1, 3, 4, 6, 7])
    maxPoint = 0
    for i in validActions:
        if i in patu:
            arrPort = state[133:187].reshape(9, 6)
            idx = int(np.where(patu == i)[0][0])
            type_i = typePort[idx]
            port_i = arrPort[type_i]  ###gỗ, gạch, cừu, lúa, đá, (1/3)
            point = 0
            if port_i[ngL]:
                point = 3
            elif port_i[5]:
                point = 2

            for vung in vungKe(i):
                point_vung = thuocTinhVung(state, vung)[0]
                point_vung = 6 - abs(point_vung - 7)
                point += point_vung

            if maxPoint < point:
                action = i
                maxPoint = point

    if action == -1:
        maxPoint = 0
        for i in validActions:
            point = 0
            for vung in vungKe(i):
                point_vung = thuocTinhVung(state, vung)[0]
                point = 6 - abs(point_vung - 7)
            if maxPoint < point:
                action = i
                maxPoint = point
    return action


@njit
def diChuyenRobber(state):
    validActions = env.getValidActions(state)
    if sum(validActions[64:83]):
        #  khu của player khác-----------------
        nhaPlayer = np.zeros(54)  #  nhà
        for i in range(3):
            player_ = (
                state[391 + 185 * (i - 1) + 75 : 391 + 185 * (i - 1) + 129]
                + state[391 + 185 * (i - 1) + 129 : 391 + 185 * (i - 1) + 183]
            )
            nhaPlayer += player_
        #  Khu của bản thân-----------------
        nhaToi = state[276:330] + state[330:384]  #  nhà

        vungPlayer = np.zeros(19)  # ----------
        for i in np.where(nhaPlayer)[0]:
            vungPlayer[vungKe(i)] = np.ones(len(vungKe(i)))
        vungToi = np.zeros(19)  # -------------
        for i in np.where(nhaToi)[0]:
            vungToi[vungKe(i)] = np.ones(len(vungKe(i)))

        vungRobber = vungPlayer - vungToi
        act = validActions[64:83] * vungRobber
        if act[act > 0].size != 0:
            return np.where(act > 0)[0][0] + 64
        else:
            return 64
    else:
        return 0


@njit
def dungKnightTruoc(state, validActions):
    if 54 in validActions and 55 in validActions:
        #  Khu của bản thân-----------------
        nhaToi = state[276:330] + state[330:384]  #  nhà
        vungToi = np.zeros(19)  # -------------
        for i in np.where(nhaToi)[0]:
            vungToi[vungKe(i)] = np.ones(len(vungKe(i)))

        if sum(state[95:114] * vungToi) == 1:
            return 55
        return 54
    else:
        return 0


@njit
def buildRoad(state, validActions):
    if 86 in validActions:
        #  Khu của bản thân-----------------
        nhaToi = state[276:330] + state[330:384]  #  nhà
        #  Đường
        myRoad = state[204:276]
        khuCuaToi = np.zeros(54)
        for i in np.where(myRoad)[0]:
            khuCuaToi[ROAD_POINT[i]] = np.array([1, 1])

        #  khu của player khác-----------------
        nhaPlayer = np.zeros(54)  #  nhà
        for i in range(3):
            player_ = (
                state[391 + 185 * (i - 1) + 75 : 391 + 185 * (i - 1) + 129]
                + state[391 + 185 * (i - 1) + 129 : 391 + 185 * (i - 1) + 183]
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
        )  # những chỗ có thể xây nhà
        if arr[arr > 0].size == 0:
            return True

    return False


@njit
def building(state, validActions):
    arrBuilding = np.array([87, 88])
    for k in arrBuilding:
        if k in validActions:
            return k
    return 0


@njit
def checkBuyDev(state, validActions):
    if 89 in validActions:
        if state[198] < 2:
            return True
        nha = state[276:330]
        city = state[330:384]
        if sum(nha) == 0:  ###  nếu không còn xây được thành phố
            return True
    return False


@njit
def Test(state, per):
    if env.getReward(state) != -1:
        per[0][0] = 0
    else:
        per[0][0] += 1
    validActions = env.getValidActions(state)
    validActions = np.where(validActions)[0]

    phase = state[947:963]
    if per[0][0] == 1:  #  đặt nhà đầu tiên---------------
        action = firstSettlements(state, validActions)
        #  print('xaynha1', action)
        return action, per
    if per[0][0] == 3:  #  đặt nhà thứ hai-----------------
        action = secondSettlements(state, validActions)
        #  print('xaynha2', action)
        return action, per

    if diChuyenRobber(state):  # di chuyển Robber --------
        action = diChuyenRobber(state)
        #  print('dichuyenRobber', action)
        return action, per

    if dungKnightTruoc(
        state, validActions
    ):  # dùng knight đầu ván khi nhà đang bị cướp----------
        action = dungKnightTruoc(state, validActions)
        #  print('dungKnightTruoc', action)
        return action, per

    if building(state, validActions):  ##  xây nhà, thành phố khi có thể
        action = building(state, validActions)
        #  print('building', action)
        return action, per

    if buildRoad(state, validActions):  ##  có nên xây thêm đường không
        action = 86
        #  print('buildRoad', action)
        return action, per

    if checkBuyDev(state, validActions):
        action = 89
        #  print('buyDev', action)
        return action, per

    if phase[1]:
        action = xacDinhDuongPhuHop(state, validActions)
        #  print('xayDuong', action)
        return action, per

    if phase[15]:
        ngLieu = state[193:198]
        min = np.min(ngLieu)
        action = np.where(ngLieu == min)[0][0] + 59
        if action in validActions:
            #  print('phase15:', action)
            return action, per

    if 91 in validActions:
        action = 91
        #  print('trade: ', action)
        return action, per

    if 105 in validActions:
        action = 105
        #  print('end trade: ', action)
        return action, per
    idx = np.random.randint(len(validActions))
    action = validActions[idx]
    return action, per
