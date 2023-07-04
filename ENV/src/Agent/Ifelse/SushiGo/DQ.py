import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


@njit()
def DataAgent():
    return List([np.zeros((1, 1))])


@njit()
def Test(state, per):
    ValidAction = env.getValidActions(state)
    ValidAction = np.where(ValidAction == 1)[0]
    returnAction = -1

    if 10 in ValidAction and np.sum(state[16:28]) <= 2 and state[26] == 0:
        returnAction = 10
    elif state[26] > 0:
        if 7 in ValidAction:
            returnAction = 7
        elif 6 in ValidAction:
            returnAction = 6
    #  if returnAction == -1:
    if 1 in ValidAction and state[17] < 3:  #  and state[16] == 0:
        if np.sum(state[16:28]) <= 2:
            returnAction = 1
        elif np.sum(state[16:28]) <= 4 and state[17] == 1:
            returnAction = 1
        elif state[17] == 2:
            returnAction = 1
    elif 7 in ValidAction:
        returnAction = 7
    elif 5 in ValidAction and state[21] <= 1:
        returnAction = 5
    if 9 in ValidAction:
        returnAction = 9

    if returnAction == -1:
        if 0 in ValidAction and state[16] < 2:
            if state[16] == 0:
                if np.sum(state[16:28]) <= 4:  # and state[17] == 0:
                    returnAction = 0

            else:
                returnAction = 0

    if returnAction == -1:
        if 7 in ValidAction:
            returnAction = 7
        elif 6 in ValidAction:
            returnAction = 6
        elif 4 in ValidAction:
            returnAction = 4
        elif 2 in ValidAction:
            returnAction = 2
        elif 8 in ValidAction:
            returnAction = 8
        elif 11 in ValidAction:
            returnAction = 11

    if returnAction == -1:
        returnAction = ValidAction[np.random.randint(len(ValidAction))]
    #      print('random')
    #  print(state[0:14])
    #  print(state[14:84].reshape(5,14))
    #  print(ValidAction, returnAction)

    if 12 in ValidAction:
        returnAction = 12
    return returnAction, per
