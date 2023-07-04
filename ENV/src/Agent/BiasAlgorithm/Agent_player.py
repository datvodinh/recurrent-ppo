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
            np.random.choice(
                np.arange(env.getActionSize()), size=env.getActionSize(), replace=False
            )
            * 1.0,
            np.zeros(env.getActionSize()),
        ]
    )
    return per


def convert_to_save(perData):
    if type(perData) == np.ndarray:
        raise Exception("Data này đã được convert rồi.")
    return perData[1]


def convert_to_test(perData):
    return perData


@njit()
def Train(state, per):
    actions = env.getValidActions(state)
    output = actions * per[0] + actions
    action = np.argmax(output)
    win = env.getReward(state)
    if win == 1:
        per[1] += per[0]
    if win == 0:
        np.random.shuffle(per[0])
    return action, per


@njit()
def Test(state, per):
    actions = env.getValidActions(state)
    output = per * actions + actions
    action = np.argmax(output)
    return action, per
