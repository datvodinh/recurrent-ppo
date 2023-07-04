import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


def DataAgent():
    return [np.zeros((1, 1))]


@njit()
def getCardInf():
    ALL_CARD_IN4 = np.array(
        [
            [0, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 2, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [2, 0, 0, 0, 0, 2, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 1, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 1, 0],
            [3, 0, 0, 0, 0, 3, 0, 0, 0],
            [3, 0, 0, 0, 0, 1, 1, 0, 0],
            [4, 0, 0, 0, 0, 0, 2, 0, 0],
            [4, 0, 0, 0, 0, 0, 1, 1, 0],
            [5, 0, 0, 0, 0, 0, 0, 2, 0],
            [5, 0, 0, 0, 0, 0, 3, 0, 0],
            [0, 1, 0, 0, 3, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 2, 0, 0, 3, 0, 1, 0, 0],
            [0, 2, 0, 0, 2, 0, 0, 1, 0],
            [0, 3, 0, 0, 0, 0, 3, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 2, 0],
            [0, 3, 0, 0, 1, 0, 1, 1, 0],
            [0, 3, 0, 0, 2, 0, 2, 0, 0],
            [0, 0, 1, 0, 4, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 2, 0, 0, 0],
            [0, 0, 1, 0, 0, 2, 0, 0, 0],
            [0, 0, 2, 0, 2, 1, 0, 1, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0],
            [0, 0, 2, 0, 2, 3, 0, 0, 0],
            [0, 0, 2, 0, 0, 2, 0, 1, 0],
            [0, 0, 3, 0, 0, 0, 0, 3, 0],
            [0, 0, 0, 1, 0, 0, 2, 0, 0],
            [0, 0, 0, 1, 3, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 3, 0, 0, 0],
            [0, 0, 0, 1, 2, 2, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 2, 1, 1, 3, 0, 0],
            [0, 0, 0, 2, 0, 3, 2, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 0],
            [2, 0, 1, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 3],
        ]
    )
    return ALL_CARD_IN4


@njit()
def Test(state, per):
    ValidAction = env.getValidActions(state)
    ValidAction = np.where(ValidAction == 1)[0]
    returnAction = -1

    if 1 in ValidAction:
        returnAction = 1
    elif ValidAction[0] in range(57, 61):
        returnAction = np.argmax(state[2:6]) + 57
    elif ValidAction[0] in range(62, 65):
        for action in ValidAction:
            returnAction = action

    if returnAction == -1:
        for action in ValidAction:
            if action in range(7, 12):
                returnAction = action
                break

    if returnAction == -1:
        pointCards = state[194:219].reshape(5, 5)
        myResource = state[2:6]

        actionCard = np.zeros(45)
        for action in ValidAction:
            if action in range(12, 57) and action != 55:
                actionCard[action - 12] = 1
        idxCard = np.where(actionCard == 1)[0]
        actionCards = getCardInf()
        actionCards = actionCards[idxCard]

        for i in range(5):
            priceCard = pointCards[i][0:4]
            if len(actionCards) > 0:
                for j in range(len(actionCards)):
                    card = actionCards[j]
                    if np.sum(state[2:6] - card[0:4] + card[4:8]) <= 10:
                        resourceAction = state[2:6] - card[0:4] + card[4:8]
                        if len(np.where(resourceAction - priceCard < 0)[0]) == 0:
                            returnAction = idxCard[j] + 12
                            break
                if returnAction != -1:
                    break

        if returnAction == -1:
            if len(actionCards) > 0:
                for i in range(len(actionCards)):
                    card = actionCards[i]
                    if np.sum(state[2:6] - card[0:4] + card[4:8]) <= 10:
                        returnAction = idxCard[i] + 12
                        break

    if returnAction == -1 and np.sum(state[51:96]) <= 1:
        #  print('?')
        returnAction = ValidAction[np.random.randint(len(ValidAction))]

    if returnAction == -1:
        if 0 in ValidAction:
            returnAction = 0
        else:
            returnAction = ValidAction[np.random.randint(len(ValidAction))]

    #  print(state[0:6])
    #  print(state[96:120].reshape(4,6))

    #  print(state[194:219].reshape(5,5))
    #  print(state[120:174].reshape(6,9))
    #  print(ValidAction, returnAction)

    return returnAction, per
