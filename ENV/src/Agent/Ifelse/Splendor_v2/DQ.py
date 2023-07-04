import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


def DataAgent():
    return List([np.zeros((1, 1))])


@njit()
def getGemsValue(state):
    gemsValue = np.zeros(5)
    myGems = state[6:11]
    myCards = state[12:17]
    cards = state[18:150].reshape(12, 11)
    prices = np.zeros((12, 5))
    #  print(cards)
    for i in range(12):
        if np.sum(cards[i]) != 0:
            prices[i] = cards[i][6:11] - myGems - myCards
            for j in range(5):
                if prices[i][j] < 0:
                    prices[i][j] = 0
                else:
                    prices[i][j] = prices[i][j] ** 1.4
        else:
            prices[i] = np.full(5, 99)
    #  print(prices)
    cardsValue = np.zeros(12)
    for i in range(12):
        cardsValue[i] = (0.5 * cards[i][0] + 20.02) / (np.sum(prices[i]) + 0.75)
    #  print(cardsValue)
    idxmax = np.argmax(cardsValue)
    gemsValue = prices[idxmax]
    if np.sum(state[208:213]) == 1:
        gemsValue[np.where(state[208:213] == 1)[0][0]] = 0.0001
    for i in range(5):
        if gemsValue[i] == 0:
            gemsValue[i] = 0.01
    return gemsValue


@njit()
def Test(state, per):
    ValidAction = env.getValidActions(state)
    ValidAction = np.where(ValidAction == 1)[0]

    cardCanBuy = np.zeros(12)
    for action in ValidAction:
        if action in range(1, 13):
            cardCanBuy[action - 1] = 1
    cardCanBuy = np.where(cardCanBuy == 1)[0]
    if len(cardCanBuy) > 0:
        action = cardCanBuy[np.random.randint(len(cardCanBuy))] + 1
        return action, per

    gems = np.zeros(5)
    for action in ValidAction:
        if action in range(31, 36):
            gems[action - 31] = 1
    #  gems = np.where(gems ==1)[0] //bỏ
    if np.sum(gems) > 0:
        gemsValue = getGemsValue(state)
        #  print(gemsValue)
        gemsValue = gemsValue * gems
        #  print(gemsValue)

        action = np.argmax(gemsValue) + 31
        #  print(action, ValidAction,'-------------')
        return action, per
    action = ValidAction[np.random.randint(len(ValidAction))]
    return action, per
