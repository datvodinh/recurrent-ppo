import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


@njit
def DataAgent():
    return np.zeros(1)


@njit
def locPhomBo(myCard):
    temp = np.zeros(52)
    count = 0

    cards = np.where(myCard)[0]
    for i in cards[cards.size - 1 :: -1]:
        if temp[i] == 0:
            k = i // 4
            arr = myCard[k * 4 : k * 4 + 4]
            if np.sum(arr) >= 3:
                _ = k * 4 + np.where(arr)[0]
                temp[_] = np.ones(_.size)
                count += 1

    return temp, count


@njit
def locPhomDay(myCard):
    temp = np.zeros(52)
    count = 0

    cards = np.where(myCard)[0]
    for i in cards[cards.size - 1 :: -1]:
        if temp[i] == 0 and i >= 8:
            if myCard[i - 4] and myCard[i - 8]:
                count += 1
                k = i
                while myCard[k]:
                    temp[k] = 1
                    k -= 4
                    if k < 0:
                        break

    return temp, count


@njit
def toiUuPhom(phom1, c1, phom2, c2):
    laLap = phom1 * phom2
    phom = phom1 + phom2 - laLap

    cards = np.where(laLap)[0]

    for i in cards[cards.size - 1 :: -1]:
        k = i // 4
        arrBo = phom[k * 4 : k * 4 + 4]

        a = b = 0
        x = i - 4
        while x >= 0:
            if phom[x]:
                x -= 4
                a += 1
            else:
                break

        x = i + 4
        while x < 52:
            if phom[x]:
                x += 4
                b += 1
            else:
                break

        arrBo_ = laLap[k * 4 : k * 4 + 4]

        if sum(arrBo) == 3 and sum(arrBo_) == 1:
            if a + b + 1 == 3:  ###  độ dài dây
                phom[i - 4 * a : i + 4 * b + 1 : 4] -= 1
                phom[i] = 1

        if sum(arrBo) == 3 and sum(arrBo_) > 1:
            arrBo = arrBo_

    return phom


@njit
def Test(state, per):
    validActions = env.getValidActions(state)
    actions = np.where(validActions == 1)[0]

    myCard = state[:52] + state[104 + 52 + 52 : 104 + 52 + 52 + 52]

    phom1, c1 = locPhomBo(myCard)
    phom2, c2 = locPhomDay(myCard)

    phom = toiUuPhom(phom1, c1, phom2, c2)
    if 0 in actions:
        return 0, per

    if np.sum(state[104 : 104 + 52]) <= 2:
        for act in actions[16:]:
            if act >= 2:
                id = act - 2
                if phom[id] == 0:
                    #  print('ACTION----: ', act)
                    return act, per

    xxx = np.sum(state[104 : 104 + 52 * 6].reshape(6, 52), 0) + phom
    myCard = myCard - phom

    for act in actions[actions.size - 1 :: -1]:
        if act >= 2:
            id = act - 2
            if phom[id] == 0 and sum(state[104 : 104 + 52]) < 3:
                k = id // 4

                if sum(myCard[k * 4 : k * 4 + 4]) == 2:
                    if sum(xxx[k * 4 : k * 4 + 4]):
                        return act, per

                if sum(myCard[k * 4 : k * 4 + 4]) == 1:
                    #  print('ACTION: ', act)
                    return act, per

            elif phom[id] == 0 and sum(state[104 : 104 + 52]) >= 3:
                return act, per

    action = actions[-1]
    #  print('ACTION:random ', action)
    return action, per
