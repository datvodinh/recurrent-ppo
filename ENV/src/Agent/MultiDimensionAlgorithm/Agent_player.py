#  không gian n chiều
#  small NN deep
import sys
import numpy as np
from numba import njit
import env

game_name = sys.argv[1]
env.make(game_name)


from numba.typed import List


# Agent function
def DataAgent():
    per = List(
        [
            np.random.rand(env.getActionSize(), env.getStateSize()),  # [0]
            np.zeros((env.getActionSize(), env.getStateSize())),  # [1]
            np.zeros((1, 1)),  # [2]
            np.random.rand(env.getActionSize(), env.getStateSize()) * 10,  # [3]
            np.zeros((env.getActionSize(), env.getStateSize())),  # [4]
        ]
    )
    return per


@njit()
def findOut(state, geo):
    return np.sum((geo * state) ** 2, axis=1)


@njit()
def Train(state, per):
    actions = env.getValidActions(state)
    #  nState = state - 1
    nState = state - per[3]
    output = np.sum((per[0] * nState) ** 2, axis=1)
    output = actions * output + actions
    action = np.argmax(output)
    win = env.getReward(state)
    if win == 1:
        per[1] += per[0]
        per[4] += per[3]
        per[2][0] += 1
    if win == 0:
        per[0] = np.random.rand(env.getActionSize(), env.getStateSize())
        per[3] = np.random.rand(env.getActionSize(), env.getStateSize()) * 10
    return action, per


@njit()
def Test(state, per):
    actions = env.getValidActions(state)
    #  nState = state - 1
    nState = state - per[0]
    output = np.sum((per[1] * nState) ** 2, axis=1)
    output = actions * output + actions
    #  action = np.argmax(output)
    list_action = np.where(actions == 1)[0]
    action = list_action[np.argmax(output[list_action])]
    return action, per


def convert_to_save(perData):
    if len(perData) == 2:
        raise Exception("Data này đã được convert rồi.")
    data = List()
    data.append(perData[4] / perData[2][0])
    data.append(perData[1] / perData[2][0])
    return data


def convert_to_test(perData):
    return List(perData)
