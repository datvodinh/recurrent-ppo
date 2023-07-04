import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


@njit()
def DataAgent():
    return np.zeros(2)


@njit()
def getActionChain(state):
    players_order = state[67:106].reshape(3, 13)
    my_cards = state[0:13]
    my_cards[my_cards == 4] = 0
    my_cards_id = np.where(my_cards > 0)[0]
    players_infor = state[15:60].reshape(3, 15)
    player_cards_len = np.array([player_infor[13] for player_infor in players_infor])
    max_card_len_player_id = np.argmax(player_cards_len)
    if True:
        player_order = players_order[max_card_len_player_id]
        player_order = np.where(player_order == 1)[0]
        for j in range(len(player_order)):
            if player_order[j] in my_cards_id:
                return max_card_len_player_id + 1, player_order[j] + 4
    for i in range(0, 3):
        player_order = players_order[i]
        player_order = np.where(player_order == 1)[0]
        if len(player_order) > 1:
            for j in range(len(player_order)):
                if player_order[j] in my_cards_id:
                    return i + 1, player_order[j] + 4
    return 0, 0


@njit()
def Test(state, per):
    actions = env.getValidActions(state)
    actions = np.where(actions == 1)[0]
    phase = state[61:64]
    phase = np.where(phase == 1)[0]
    if len(phase) == 0:
        return np.random.choice(actions), per
    if phase == 2:
        per = np.zeros(2)
    if phase == 0:
        per[0], per[1] = getActionChain(state)
        if per[0] != 0:
            action = int(per[0])
            if action in actions:
                return action, per
        elif per[0] == 0:
            player_cards_len = np.zeros(3)
            player_infor = state[15:60].reshape(3, 15)
            for i in range(3):
                player_cards_len[i] = player_infor[i][13]
            max_cards_len = np.argmax(player_cards_len)
            action = max_cards_len + 1
            if action in actions:
                return action, per
    if phase == 1 and per[1] != 0:
        action = int(per[1])
        return action, per
    elif phase == 1 and per[1] == 0:
        my_cards = state[0:13]
    return np.random.choice(actions), per
