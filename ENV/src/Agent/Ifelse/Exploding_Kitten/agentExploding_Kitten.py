import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


import numpy as np


@njit
def DataAgent():
    return np.zeros(1)


@njit
def uuTien(state, actions):
    arr = state[:12].copy()
    arr[0:3] = np.zeros(3)
    arr[5] = 0
    arr[11] = 0

    if state[23]:
        if 9 in actions and np.where(arr)[0].size >= 5:
            return 9
        if 50 in actions:
            return 50

    arr[0:6] = np.zeros(6)
    if 7 in actions and np.max(arr) >= 2 and state[25] < 23:
        idx = np.argmax(arr)
        return 7

    max = np.argmax(state[87:91])
    if 11 + max in actions:
        return 11 + max

    if 38 in actions and state[23] <= 3:
        return 38
    if 28 in actions and state[13] <= 2:
        return 28
    if 29 in actions and state[14] <= 2:
        return 29
    if 31 in actions and state[16] <= 2:
        return 31
    if 27 in actions and state[12] <= 3:
        return 27
    if 30 in actions and state[15] <= 2:
        return 30
    if 29 in actions:
        return 29

    return -1


@njit
def seeTheFuture(state, actions):
    future = state[28:67].reshape(3, 13)
    la = np.zeros(3)
    for i in range(3):
        arr = future[i]
        if np.where(arr)[0].size:
            la[i] = np.where(arr)[0][0]
        else:
            la[i] = -1

    if np.sum(la) != -3:
        if la[0] == 12 and state[71] >= 1:
            if 1 in actions:
                return 1
            elif 2 in actions:
                return 2
            elif 4 in actions:
                return 4

        elif la[1] == 12:
            if state[71] == 1 and 6 in actions:
                return 6
            elif state[71] >= 2:
                if 1 in actions:
                    return 1
                elif 2 in actions:
                    return 2
                elif 4 in actions:
                    return 4

        elif la[2] == 12:
            if state[71] <= 2 and 6 in actions:
                return 6
            elif state[71] >= 3:
                if 1 in actions:
                    return 1
                elif 2 in actions:
                    return 2
                elif 4 in actions:
                    return 4
        else:
            if 6 in actions:
                return 6

    return -1


@njit
def giveCard(state, actions):
    if actions[0] >= 15 and actions[-1] < 27:
        arr = state[:12].copy()
        arr[0:6] = np.zeros(6)
        arr[11] = 0
        idx = np.where(arr)[0]
        max = np.max(arr)
        for i in idx:
            if i + 15 in actions:
                if arr[i] != max and arr[i] > 1:
                    return i + 15

        if idx.size:
            if idx[0] + 15 in actions:
                return idx[0] + 15

        arr = np.array([18, 15, 19, 20, 17, 16])
        for act in arr:
            if act in actions:
                return act
        return np.random.choice(actions)

    return -1


@njit
def Test(state, per):
    validActions = env.getValidActions(state)
    actions = np.where(validActions)[0]

    #  print(state[:12],state[24: 27])
    #  print(actions)
    if uuTien(state, actions) != -1:
        action = uuTien(state, actions)
        #  print('ACTIONuuTien: ', action)
        return action, per

    if giveCard(state, actions) != -1:
        action = giveCard(state, actions)
        #  print('ACTIONgive: ', action)
        return action, per

    if 3 in actions and state[25] < 23:
        act = 3
        #  print('ACTION3: ', act)
        return 3, per

    if 0 in actions:
        if np.sum(state[72:82]) > 0 and state[71] > 0:
            act = 0
            #  print('ACTION0: ', act)
            return 0, per
        elif 10 in actions:
            act = 10
            #  print('ACTION10: ', act)
            return 10, per

    if state[11] or state[25] > 15:
        if 6 in actions:
            act = 6
            #  print('ACTION6: ', act)
            return 6, per

    if 5 in actions and state[11] == 0 and state[25] < 15:
        act = 5
        #  print('ACTION5: ', act)
        return 5, per

    if seeTheFuture(state, actions) != -1:
        action = seeTheFuture(state, actions)
        #  print('ACTIONsee: ', action)
        return action, per

    #  if 1 in actions and state[1] > 1:
    #    #  act = 1
    #    #  print('ACTION1: ', act)
    #    return 1, per

    if state[1] + state[2] >= 1 and state[26] < 3:
        act = actions[0]
    else:
        act = actions[-1]
    #  print('ACTIONrandon: ', act)
    return act, per
