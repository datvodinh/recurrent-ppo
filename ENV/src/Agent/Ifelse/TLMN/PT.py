import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


@njit
def DataAgent():
    return np.array([0.0])


__ACTIONS__ = env.__ACTIONS__
# ---------------------------------------------------------------------------------------------------------
from numba.typed import List


@njit()
def getCardOnHand(state):
    card_on_hand = state[0:52]
    return np.where(card_on_hand == 1)[0]


@njit()
def checkArrInArr(arr1, arr2):
    #  print('arr1,arr2',arr1,arr2)
    arr_return = []
    for i in range(len(arr1)):
        if arr1[i] in arr2:
            arr_return.append(i)
    return np.array(arr_return)


@njit()
def getSpecialCard(state):
    p_cards = state[0:52]
    p_cards = np.where(p_cards == 1)[0]
    p_cards_value = p_cards // 4
    policy = np.zeros(52)
    for i in range(52):
        arr_copy = np.copy(p_cards)
        arr_temp = np.where(arr_copy // 4 == i)[0]
    #  check day
    arr_day = np.zeros(11) - 1
    arr_unique = np.unique(p_cards_value)
    for i in range(1, len(arr_unique)):
        if arr_unique[i] == arr_unique[i - 1] + 1:
            arr_day[i] = arr_unique[i]
        elif p_cards_value[i] != p_cards_value[i - 1] - 1:
            if len(np.where(arr_day > 0)[0]) >= 3:
                arr_id = checkArrInArr(p_cards_value, arr_day)
                arr_id = p_cards[arr_id]
                policy[arr_id] = 1
            arr_day = np.zeros(11)
    return np.where(policy == 1)[0]


@njit()
def getBoDay(state):
    p_cards = state[0:52]
    p_cards = np.where(p_cards == 1)[0]
    p_cards_value = p_cards // 4
    policy = np.zeros(52)
    for i in range(52):
        arr_copy = np.copy(p_cards)
        arr_temp = np.where(arr_copy // 4 == i)[0]
    #  check day
    arr_day = np.zeros(11)
    arr_unique = np.unique(p_cards_value)
    for i in range(1, len(arr_unique)):
        if arr_unique[i] == arr_unique[i - 1] + 1:
            arr_day[i] = arr_unique[i]
        elif arr_unique[i] != arr_unique[i - 1] - 1:
            if len(np.where(arr_day > 0)[0]) >= 3:
                arr_id = checkArrInArr(p_cards_value, arr_day)
                #  print('arr_id',arr_id)
                arr_id = p_cards[arr_id]
                policy[arr_id] = 1
            arr_day = np.zeros(11)
    return np.where(policy == 1)[0]


@njit()
def getHighestCard(action):
    return __ACTIONS__[action][1]


@njit()
def getTypeAction(action):
    return __ACTIONS__[action][0]


@njit()
def getValueActions(action):
    return __ACTIONS__[action][1]


@njit()
def getLowestValueOfChain(action):
    type_ = getTypeAction(action) - 3
    highestValue = getValueActions(action)
    return highestValue // 4 - type_


@njit()
def countCardInChain(state, cardValue):
    cards_on_hand = state[0:52]
    cards_on_hand = np.where(cards_on_hand == 1)[0]
    #   print(cards_on_hand//4)
    #   print(cardValue//4)
    #   cards_on_hand_value = cards_on_hand // 4
    count = 0
    for card_value_ in cards_on_hand:
        if card_value_ // 4 == cardValue // 4 and card_value_ != cardValue:
            count += 1
    return count


@njit()
def getAction(state, actions):
    boDay = getBoDay(state) // 4
    for action in actions:
        type_ = getTypeAction(action)
        value_ = getValueActions(action)
        if value_ // 4 not in boDay and value_ // 4 != 12:
            #  print('value_//4',value_//4)
            return action
        else:
            count_chain = countCardInChain(state, getValueActions(action))
            if count_chain > 2:
                #  print('count_chain',count_chain)
                return action
    return -1


@njit()
def getActionLaDon(state, actions):
    boDay = getBoDay(state) // 4
    #  check xem co trong bo day va bo doi hay k
    for action in actions:
        type_ = getTypeAction(action)
        value_ = getValueActions(action)
        specialCard = getSpecialCard(state)
        if value_ // 4 == 12:
            return action
        if value_ // 4 not in specialCard // 4:
            #  print('value_//4',value_//4)
            #   count_chain = countCardInChain(state,getValueActions(action))
            #   if count_chain == 0 or  value_ == 12:
            #    #  print('count_chain',count_chain)
            return action
    return -1


# ----------------------------------------------------------------------------------------------------
@njit()
def Test(state, per):
    cards_on_hand = state[0:52]
    cards_on_hand = np.where(cards_on_hand == 1)[0]
    actions = env.getValidActions(state)
    specialCards = getSpecialCard(state)
    boDay = getBoDay(state)
    if np.min(state[107:110]) >= 8:
        if actions[0] == 1:
            return 0, per
    actions = np.where(actions == 1)[0]
    boDay = getBoDay(state) // 4
    type_action = [getTypeAction(action) for action in actions]
    id = np.argmax(np.array(type_action))
    #  Xu ly danh la 2 ------------------------
    if (
        getTypeAction(actions[id]) == 2
        and getValueActions(actions[id]) // 4 in specialCards // 4
    ):
        action_ = getAction(state, actions)
        if action_ != -1:
            return action_, per
        elif np.min(state[107:110]) > 8:
            return actions[0], per
        else:
            return actions[1], per
    #  Xu ly danh la 1 ----------------------
    if (
        getTypeAction(actions[id]) == 1
        and getValueActions(actions[id]) // 4 in specialCards // 4
    ):
        action_ = getActionLaDon(state, actions[id:])
        if action_ != -1:
            return action_, per
        elif np.min(state[107:110]) > 8:
            return actions[0], per
        else:
            return actions[1], per
    valueCard = __ACTIONS__[actions[id]][1] // 4
    cardsOnHand = (len(cards_on_hand) + np.sum(state[107:110])) / 4
    if valueCard > 13 / cardsOnHand * 3.33 + 6:
        return actions[0], per
    return actions[id], per
