import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


def DataAgent():
    return np.zeros((3, 13))


@njit()
def Test(state, per):
    ValidAction = env.getValidActions(state)
    ValidAction = np.where(ValidAction == 1)[0]

    returnAction = -1

    myCards = np.where(state[0:13] > 0)[0]
    otherCardRequest = state[67:106].reshape(3, 13)
    per += otherCardRequest
    for j in range(13):
        for i in range(3):
            if per[i][j] == -1:
                per[i][j] = 0
    for j in range(13):
        for i in range(2):
            for k in range(i + 1, 3):
                if per[i][j] < per[k][j] and per[i][j] > 0:
                    per[k][j] = 0

    if ValidAction[0] in range(1, 4):
        for i in range(3):
            for card in myCards:
                if per[i][card] > 0:
                    if i + 1 in ValidAction:
                        returnAction = i + 1

    if ValidAction[0] in range(4, 17):
        person = np.where(state[64:67] == 1)[0][0]
        for card in myCards:
            #  if len(otherCardRequest) > 0:
            if per[person][card] > 0:
                if card + 4 in ValidAction:
                    returnAction = card + 4

    if returnAction == -1:
        returnAction = ValidAction[np.random.randint(len(ValidAction))]
    #      print("random")

    #  print(state[0:13],myCards)
    #  print("Person: ", state[64:67])
    #  print("other info:\n",state[15:60].reshape(3,15))
    #  print("Request Card:\n", otherCardRequest)
    #  print(per)
    #  print(ValidAction, returnAction)
    #  print("-------------")
    #  print()
    #  print()

    if returnAction in range(4, 17):
        person = np.where(state[64:67] == 1)[0]
        #  print(person)
        if len(person) > 0:
            person = person[0]
            per[person][returnAction - 4] = -1

    #  if state[60] > 10 and state[60] < 15:
    #      per = np.zeros((3,13))

    if state[60] <= 6:
        per = np.zeros((3, 13))

    if env.getReward(state) != -1:
        per = np.zeros((3, 13))

    return returnAction, per
