import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


@njit
def DataAgent():
    return [np.zeros((1, 1))]


@njit()
def Test(state, per):
    ValidAction = env.getValidActions(state)
    ValidAction = np.where(ValidAction == 1)[0]
    returnAction = -1
    if 11 in ValidAction:
        returnAction = 11
    elif 13 in ValidAction and state[144] / (state[143] + 0.5) < 2:
        returnAction = 13
    elif 12 in ValidAction:
        returnAction = 12

    #  if returnAction == -1:
    #      civCards = state[14:110].reshape(4,-1)
    #      for i in range(4):
    #          if civCards[i][8] == 1 and i+19 in ValidAction:
    #              returnAction = i+19

    if returnAction == -1:
        buildingCards = state[110:142].reshape(4, -1)
        myResource = state[147:151]

        for i in range(4):
            price = buildingCards[i][1:5]
            if np.sum(price) > 0:
                buy = myResource - price
                if len(np.where(buy < 0)[0]) == 0 and i + 23 in ValidAction:
                    #  print(buy, myResource, price)
                    returnAction = i + 23

    if returnAction == -1:
        if 19 in ValidAction:
            returnAction = 19
        elif 20 in ValidAction:
            returnAction = 20
        elif state[143] + state[145] < state[144] and 18 in ValidAction:
            returnAction = 18
        elif state[147] >= 10 and 21 in ValidAction:
            returnAction = 21
        elif state[147] >= 14 and 22 in ValidAction:
            returnAction = 22
        elif 14 in ValidAction:
            returnAction = 14
        elif 15 in ValidAction:
            returnAction = 15
        elif 16 in ValidAction:
            returnAction = 16
        elif 17 in ValidAction:
            returnAction = 17

    if returnAction == -1:
        for i in range(40, 44):
            if i in ValidAction:
                returnAction = i
                break
        if ValidAction[0] == 0:
            returnAction = np.max(ValidAction)

        if 4 in ValidAction and state[144] >= 7:
            returnAction = 4
        elif 3 in ValidAction:
            returnAction = 3
        elif 30 in ValidAction:
            returnAction = 30
        elif 32 in ValidAction:
            returnAction = 32

    if 63 in ValidAction:
        returnAction = 63

    resourceAction = np.zeros(4)
    for action in ValidAction:
        if action in range(64, 68):
            resourceAction[action - 64] = 1

    resourceAction = np.where(resourceAction == 1)[0]
    if len(resourceAction) > 0:
        myResource = state[147:151]
        resource = np.min(myResource[resourceAction])
        for i in range(4):
            if myResource[3 - i] == resource and 3 - i + 64 in ValidAction:
                returnAction = 67 - i
                break

    if 27 in ValidAction:
        returnAction = 27
    if ValidAction[0] in range(57, 63):
        returnAction = np.max(ValidAction)

    if returnAction == -1:
        returnAction = ValidAction[np.random.randint(len(ValidAction))]
    #      print('random')

    #  print(ValidAction, returnAction)
    #  print(state[142:318].reshape(4,44))
    #  print(state[14:110].reshape(4,-1))
    #  print(state[110:142].reshape(4,-1))
    #  print('----------')
    return returnAction, per
