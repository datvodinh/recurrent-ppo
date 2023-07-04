import sys
import numpy as np
from numba import njit
from numba.typed import List

import env

game_name = sys.argv[1]
env.make(game_name)


def convert_to_save(perData):
    if len(perData) == 2:
        raise Exception("Data này đã được convert rồi.")

    data = List()
    data.append(np.zeros((1, env.getActionSize())))
    temp = np.zeros((env.getActionSize(), env.getActionSize()))
    for i in range(temp.shape[0]):
        temp[i] = np.argsort(np.argsort(perData[2][i])) + 1e-6 * np.random.rand(
            env.getActionSize()
        )

    data.append(temp)
    return data


def convert_to_test(perData):
    return List(perData)


@njit()
def DataAgent():
    return List(
        [
            np.zeros((1, env.getActionSize())),
            np.random.rand(env.getActionSize(), env.getActionSize()),
            np.zeros((env.getActionSize(), env.getActionSize())),
        ]
    )


@njit()
def Train(state, per):
    actions = env.getValidActions(state)
    weight = per[0][0]

    output = actions * weight + actions
    c = np.where(output == np.max(output))[0]
    action = c[np.random.randint(0, c.shape[0])]

    per[0] += per[1][action]
    win = env.getReward(state)

    if win != -1:
        per[0][:, :] = 0.0
        if win == 1:
            per[2] += per[1]
        else:
            per[1] = np.random.rand(env.getActionSize(), env.getActionSize())

    return action, per


@njit()
def Test(state, per):
    actions = env.getValidActions(state)
    weight = per[0][0]

    output = actions * weight + actions
    action = np.argmax(output)

    weight[:] += per[1][action]
    win = env.getReward(state)
    if win != -1:
        per[0][:, :] = 0.0
    return action, per
