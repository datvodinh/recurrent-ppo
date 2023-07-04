import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


from src.Base.Century.env import ALL_CARD_IN4


@njit
def DataAgent():
    per = []
    per.append(np.zeros(1))
    return per


@njit
def actionCard(state):
    return state[120:174].reshape(6, 9)


@njit
def cardPoint(state):
    return state[194:219].reshape(5, 5)


@njit
def muaTheDiem(state, validActions):  # lấy thẻ điểm ngay khi có thẻ
    action = 0
    point = 0
    for act in range(7, 12):
        if act in validActions:
            p = cardPoint(state)[act - 7][-1]
            if act == 7:
                p += 30
            if act == 8:
                p += 10
            if p > point:
                action = act
                point = p
    return action


@njit
def muaActCard(state, validActions):  #  lấy thẻ action: ưu tiên thẻ cho ngL miễn phí
    action = 0
    for act in range(1, 7):
        card = actionCard(state)[act - 1]
        if act in validActions and act <= state[2]:
            if sum(card[:4]) == 0:
                action = act
                return action

    for act in range(1, 7):
        card_ = actionCard(state)[act - 1]
        ngLmat = card_[:4]
        ngLthem = card_[4:8]
        if act in validActions and act <= state[2] + state[3]:
            if ngLthem[3]:
                return act
            if ngLthem[2] and ngLmat[3] == 0:
                return act
            if ngLthem[1] and ngLmat[3] == 0:
                return act

    return action


@njit
def sinhNgLnau(state, validActions):  # dùng thẻ sinh ra Nâu
    myNgL = state[2:6]
    action = 0
    # sinh ra nâu
    for act in range(12, 57):
        if act in validActions:
            card_ = ALL_CARD_IN4[act - 12]
            ngLmat = card_[:4]
            ngLthem = card_[4:8]
            if ngLthem[3]:
                action = act
                return action
    return action


@njit
def sinhNgLxanh(state, validActions):  #  dùng thẻ sinh ra Xanh
    myNgL = state[2:6]
    action = 0
    # sinh ra xanh
    for act in range(12, 57):
        if act in validActions:
            card_ = ALL_CARD_IN4[act - 12]
            ngLmat = card_[:4]
            ngLthem = card_[4:8]
            if ngLthem[2]:
                action = act
                return action
    return action


@njit
def sinhNgLdo(state, validActions):  #  dùng thẻ sinh ra Đỏ
    myNgL = state[2:6]
    action = 0
    # sinh ra đỏ
    for act in range(12, 57):
        if act in validActions:
            card_ = ALL_CARD_IN4[act - 12]
            ngLmat = card_[:4]
            ngLthem = card_[4:8]
            if ngLthem[1]:
                action = act
                return action
    return action


@njit
def sinhNgLvang(state, validActions):  #  dùng thẻ sinh ra Vàng
    myNgL = state[2:6]
    action = 0
    # sinh ra vang
    for act in range(12, 57):
        if act in validActions:
            card_ = ALL_CARD_IN4[act - 12]
            ngLmat = card_[:4]
            ngLthem = card_[4:8]
            if ngLthem[0]:
                action = act
                return action
    return action


@njit
def dungActionCard(state, validActions):  # dùng thẻ sinh ra Xanh or Nâu
    myNgL = state[2:6]
    action = 0
    # sinh ra nâu
    for act in range(12, 57):
        if act in validActions:
            card_ = ALL_CARD_IN4[act - 12]
            ngLmat = card_[:4]
            ngLthem = card_[4:8]
            if ngLthem[3]:
                action = act
                return action
    # sinh ra xanh
    for act in range(12, 57):
        if act in validActions:
            card_ = ALL_CARD_IN4[act - 12]
            ngLmat = card_[:4]
            ngLthem = card_[4:8]
            if ngLthem[2] and ngLmat[3] == 0:
                action = act
                return action
    return action


@njit
def ngLfree(state, validActions):  #  lấy ngL miễn phí
    action = 0
    for act in range(12, 57):
        if act in validActions:
            card = ALL_CARD_IN4[act - 12]
            if sum(card[:4]) == 0 and card[-1] == 0:
                action = act
                return action
    return action


@njit
def act_point(state, validActions):  #  dùng action để mua đc thẻ
    action = 0
    myNgL = state[2:6]
    for card in cardPoint(state):
        for act in range(12, 57):
            if act in validActions:
                card_ = ALL_CARD_IN4[act - 12]
                ngLmat = card_[:4]
                ngLthem = card_[4:8]

                if sum(ngLmat) != 0:
                    for i in range(1, 11):
                        ngLsau = myNgL - i * ngLmat
                        if ngLsau[ngLsau < 0].size == 0:
                            ngLsau = ngLsau + i * ngLthem - card[:4]
                            if ngLsau[ngLsau >= 0].size == 4:
                                action = act
                                return action
                else:
                    ngLsau = myNgL - ngLmat
                    if ngLsau[ngLsau < 0].size == 0:
                        ngLsau = ngLsau + ngLthem - card[:4]
                        if ngLsau[ngLsau >= 0].size == 4:
                            action = act
                            return action
    return 0


@njit
def truNgL(state, validActions):
    for act in range(57, 61):
        if act in validActions:
            return act
    return 0


@njit
def levelUp(state, validActions):
    myNgL = state[2:6]
    if validActions[validActions >= 62].size == 0:
        return 0
    for card in cardPoint(state):
        arr = myNgL - card[:4]
        for i in range(3, 0, -1):
            if arr[i] < 0:
                if myNgL[i - 1] > 0:
                    return 62 + i - 1
                if myNgL[i - 2] > 0 and i - 2 >= 0:
                    return 62 + i - 2
                if myNgL[i - 3] > 0 and i - 3 >= 0 and state[221 + 43]:
                    return 62 + i - 3
    return np.random.choice(validActions)


@njit
def nhamThePoint(state, validActions):
    myNgL = state[2:6]
    targetCard = cardPoint(state)[0]
    ngLcard = targetCard[:4]
    if myNgL[3] < ngLcard[3]:
        return sinhNgLnau(state, validActions)
    if myNgL[2] < ngLcard[2]:
        return sinhNgLxanh(state, validActions)
    if myNgL[1] < ngLcard[1]:
        return sinhNgLdo(state, validActions)
    if myNgL[0] < ngLcard[0]:
        return sinhNgLvang(state, validActions)
    return 0


@njit
def Test(state, per):
    validActions = env.getValidActions(state)
    validActions = np.where(validActions)[0]

    if muaTheDiem(state, validActions):
        action = muaTheDiem(state, validActions)
        return action, per

    if act_point(state, validActions):
        action = act_point(state, validActions)
        return action, per

    if muaActCard(state, validActions):
        action = muaActCard(state, validActions)
        return action, per

    if dungActionCard(state, validActions):
        action = dungActionCard(state, validActions)
        return action, per

    if nhamThePoint(state, validActions):
        action = nhamThePoint(state, validActions)
        return action, per

    if 61 in validActions:
        return validActions[-2], per

    if ngLfree(state, validActions):
        action = ngLfree(state, validActions)
        return action, per

    if truNgL(state, validActions):
        action = truNgL(state, validActions)
        return action, per

    if levelUp(state, validActions):
        action = levelUp(state, validActions)
        return action, per

    if 0 in validActions:
        return 0, per

    action = np.random.choice(validActions)
    return action, per
