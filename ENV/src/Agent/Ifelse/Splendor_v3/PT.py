import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


def DataAgent():
    return np.array([])


@njit()
def getCardValue(state, card):
    card = np.array(card)
    if len(np.where(card > 0)[0]) == 0:
        return -999
    type_card = card[1:6]
    type_card = np.where(type_card == 1)[0][0]
    player_stock_const = state[6 + 6 : 6 + 6 + 5]
    #  print('player_stock_const',player_stock_const)
    had = player_stock_const[type_card]
    point = card[0]
    stock_require = card[6:11]
    #  print('stock_require',stock_require)
    result = player_stock_const - stock_require
    #  print(result)
    need = sum(result[result < 0])
    #  print('need',need)

    value = (point + 0.1) * (need + 0.1) - (had * 0.1)
    return value


@njit()
def Test(state, per):
    actions = env.getValidActions(state)
    actions = np.where(actions == 1)[0]
    temp_cards = list(state[18:150]) + list(state[175:208])
    cards = []
    for i in range(0, 15):
        temp_card = temp_cards[i * 11 : i * 11 + 11]
        cards.append(temp_card)
    value_arr = np.array([getCardValue(state, card) for card in cards])
    return np.argmax(value_arr), per
