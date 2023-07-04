import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


@njit
def DataAgent():
    per = []
    per.append(np.zeros(1))  #  đếm số lượt
    return per


from src.Base.TLMN.env import __ACTIONS__

#  (0 1 2)
#  (1 2 3)
#  (2 3 4)
#  (3 4 5)
#  (4 5 6)
#  (5 6 7)
#  (6 7 8)
#  (7 8 9)
#  (8 9 10)
#  (9 10 11)


@njit()
def inGroup(card, cards):
    value = card // 4
    values = np.unique(cards // 4)
    if len(values[values == value]) >= 2:
        return True
    if value == 0:  # (0 1 2)
        if (1 in values) and (2 in values):
            return True
    if value == 1:  # (0 1 2) (1 2 3)
        if (0 in values) and (2 in values):
            return True
        if (2 in values) and (3 in values):
            return True
    if value in (2, 3, 4, 5, 6, 7, 8, 9):
        if ((value - 1) in values) and ((value - 2) in values):
            return True
        if ((value - 1) in values) and ((value + 1) in values):
            return True
        if ((value + 1) in values) and ((value + 2) in values):
            return True
    if value == 10:  # (8 9 10) (9 10 11)
        if (8 in values) and (9 in values):
            return True
        if (9 in values) and (11 in values):
            return True
    if value == 11:  # (9 10 11)
        if (9 in values) and (10 in values):
            return True
    return False


@njit()
def inStraight(card, cards):
    value = card // 4
    values = np.unique(cards // 4)
    if value == 0:  # (0 1 2)
        if (1 in values) and (2 in values):
            return True
    if value == 1:  # (0 1 2) (1 2 3)
        if (0 in values) and (2 in values):
            return True
        if (2 in values) and (3 in values):
            return True
    if value in (2, 3, 4, 5, 6, 7, 8, 9):
        if ((value - 1) in values) and ((value - 2) in values):
            return True
        if ((value - 1) in values) and ((value + 1) in values):
            return True
        if ((value + 1) in values) and ((value + 2) in values):
            return True
    if value == 10:  # (8 9 10) (9 10 11)
        if (8 in values) and (9 in values):
            return True
        if (9 in values) and (11 in values):
            return True
    if value == 11:  # (9 10 11)
        if (9 in values) and (10 in values):
            return True
    return False


@njit()
def Test(state, per):
    validActions = env.getValidActions(state)
    validActions = np.where(validActions == 1)[0]

    actionTypes = np.zeros_like(validActions)
    for i in range(len(validActions)):
        actionTypes[i] = __ACTIONS__[validActions[i]][0]

    actionType1 = validActions[np.where(actionTypes == 1)[0]]
    actionType2 = validActions[np.where(actionTypes == 2)[0]]
    actionType3 = validActions[np.where(actionTypes == 3)[0]]
    actionType4 = validActions[np.where(actionTypes == 4)[0]]
    actionType5 = validActions[np.where(actionTypes == 5)[0]]
    actionType6 = validActions[np.where(actionTypes == 6)[0]]
    actionType7 = validActions[np.where(actionTypes == 7)[0]]
    actionType8 = validActions[np.where(actionTypes == 8)[0]]
    actionType9 = validActions[np.where(actionTypes == 9)[0]]
    actionType10 = validActions[np.where(actionTypes == 10)[0]]
    actionType11 = validActions[np.where(actionTypes == 11)[0]]
    actionType12 = validActions[np.where(actionTypes == 12)[0]]
    actionType13 = validActions[np.where(actionTypes == 13)[0]]
    actionType14 = validActions[np.where(actionTypes == 14)[0]]
    actionType15 = validActions[np.where(actionTypes == 15)[0]]

    actionTypes = [
        actionType1,
        actionType2,
        actionType3,
        actionType4,
        actionType5,
        actionType6,
        actionType7,
        actionType8,
        actionType9,
        actionType10,
        actionType11,
        actionType12,
        actionType13,
        actionType14,
        actionType15,
    ]

    if np.sum(state[110:113]) == 0:
        cards = np.where(state[0:52] == 1)[0]
        for action in actionType3:
            card = __ACTIONS__[action][1]
            if (card <= 3) and not inStraight(card, cards):
                return action, per

        for action in actionType2:
            card = __ACTIONS__[action][1]
            if (card <= 3) and not inStraight(card, cards):
                return action, per

        for action in actionType1:
            card = __ACTIONS__[action][1]
            if (card <= 3) and not inGroup(card, cards):
                return action, per

    if np.min(state[107:110]) >= 8:
        if 0 in validActions:
            return 0, per

    for i in (15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1):
        if (len(actionTypes[i - 1]) == 1) and (len(validActions) == 1):
            return actionTypes[i - 1][0], per

    for i in (13, 12, 11, 10, 9, 8, 7, 6, 5):
        if len(actionTypes[i - 1]) > 0:
            return actionTypes[i - 1][0], per

    for i in (3, 2, 1, 14, 15, 4):
        if len(actionTypes[i - 1]) > 0:
            return actionTypes[i - 1][0], per

    action = validActions[np.random.randint(len(validActions))]
    return action, per
