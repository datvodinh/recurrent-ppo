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
def Test(state, per):
    ValidAction = env.getValidActions(state)
    ValidAction = np.where(ValidAction == 1)[0]

    returnAction = -1
    if 1 in ValidAction:
        returnAction = 1
    elif (
        43 in ValidAction
        and state[39] + state[59] + state[79] >= 1
        and state[10] <= 0
        and state[18] == 1
    ):
        returnAction = 43
    elif (
        43 in ValidAction
        and state[39] + state[59] + state[79] >= 3
        and state[10] <= 1
        and state[18] == 1
    ):
        returnAction = 43

    elif 35 in ValidAction:
        returnAction = 35
    elif 34 in ValidAction:
        returnAction = 34
    elif 36 in ValidAction:
        returnAction = 36
    elif 38 in ValidAction:
        returnAction = 38
    elif 39 in ValidAction:
        returnAction = 39
    elif 37 in ValidAction and state[4] <= 2:
        returnAction = 37

    #  elif 46 in ValidAction and state[18] == 1 and state[13] < 1 and state[0] <= 8:
    #      returnAction = 46
    #  elif 47 in ValidAction and state[18] == 1 and state[14] < 1 and state[0] <= 8:
    #      returnAction = 47

    #  elif 40 in ValidAction and state[2] >= 2 and state[7] <= 1:
    #      returnAction = 40

    elif 51 in ValidAction:
        returnAction = 51

    #  elif 44 in ValidAction and state[11] <= 0 and state[39] + state[59] + state[79] >= 2:
    #      returnAction = 44
    #  elif 42 in ValidAction and state[9] <= 0 and state[39] + state[59] + state[79] >= 2:
    #      returnAction = 42

    elif 53 in ValidAction:
        returnAction = 53

    if np.sum(state[1:20]) >= 10:
        if 51 in ValidAction:
            returnAction = 51
        elif 49 in ValidAction:
            returnAction = 49
        elif 50 in ValidAction and np.sum(state[16:19]) == 2:
            returnAction = 50
        elif 52 in ValidAction and np.sum(state[16:19]) == 3:
            returnAction = 52

    #  if 0 in ValidAction and (state[117] == 2 or state[117] == 3):
    #          returnAction = 0

    if 0 in ValidAction:
        if state[117] == 2 or state[117] == 4:
            returnAction = 0
        if (
            state[117] == 3
            and state[3] * (1 + state[18])
            - state[24] * (1 + state[38])
            - state[44] * (1 + state[58])
            - state[64] * (1 + state[78])
            > 0
        ):
            returnAction = 0

    #  if returnAction != 51:
    #      if np.sum(state[1:19]) >= 10 and state[18] == 0 and 53 in ValidAction:
    #            returnAction = 53

    if returnAction == -1:
        returnAction = ValidAction[np.random.randint(len(ValidAction))]
    #    print('random')
    #  print(ValidAction, returnAction)
    #  print(state[0:80].reshape(4,20))
    #  print(state[80:92])
    #  print(state[92:104])
    #  print(state[104:116])
    #  print(state[116:118])
    #  print(state[118:130])
    #  print('-----------')

    return returnAction, per
