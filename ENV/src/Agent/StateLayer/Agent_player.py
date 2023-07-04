import sys
import numpy as np
from numba import njit
import env

game_name = sys.argv[1]
env.make(game_name)


from numba.typed import List


def DataAgent():
    perx_ = List(
        [
            np.random.choice(
                env.getActionSize(), size=env.getActionSize(), replace=False
            ).reshape(1, -1)
            * 1.0,
            np.random.rand(env.getStateSize(), env.getActionSize()) * 2 - 1,
            np.random.rand(env.getActionSize(), env.getActionSize()),
            np.zeros((1, env.getActionSize())),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.random.rand(env.getStateSize(), env.getActionSize()) * 2 - 1,
            np.random.rand(env.getActionSize(), env.getActionSize()),
        ]
    )
    return perx_


@njit
def Train(state, per):
    if per[5][0][0] <= 10000:  # phase1
        actions = env.getValidActions(state)
        if per[4][0][0] == 0:
            action = np.argmax(actions * per[0][0] + actions)
        else:
            output = np.random.rand(env.getActionSize()) * actions + actions
            action = np.argmax(output)
        per[4][0][0] += 1
        if env.getReward(state) != -1:
            per[4][0][0] = 0
            if env.getReward(state) == 1:
                per[3][0] += per[0][0]
            else:
                per[0][0] = (
                    np.random.choice(
                        env.getActionSize(), size=env.getActionSize(), replace=False
                    )
                    * 1.0
                )
            per[5][0][0] += 1
            if per[5][0][0] == 10000:
                per[0][0] = per[3][0] / np.max(per[3][0])
                per[3][0] = per[0][0].copy()

    elif per[5][0][0] > 10000 and per[5][0][0] <= 110000:  # phase 2
        actions = state.reshape(1, -1).dot(per[1])
        per[0][0] += actions[0] / np.max(actions)
        list_action = np.where(env.getValidActions(state) == 1)[0]
        action = list_action[np.argmax(per[0][0][list_action])]
        if env.getReward(state) != -1:
            per[5][0][0] += 1
            per[0][0] = per[3][0].copy()
            if env.getReward(state) == 1:
                per[6][0][0] += 1
            if per[5][0][0] % 1000 == 0:
                if per[6][0][0] / 1000 > per[7][0][0]:
                    per[7][0][0] = per[6][0][0] / 1000
                    per[9] = per[1].copy()
                per[1] = np.random.rand(env.getStateSize(), env.getActionSize()) * 2 - 1
                per[6][0][0] = 0

    elif per[5][0][0] > 110000 and per[5][0][0] <= 210000:  # phase 3
        actions = state.reshape(1, -1).dot(per[1])
        per[0][0] += actions[0] / np.max(actions)
        list_action = np.where(env.getValidActions(state) == 1)[0]
        action = list_action[np.argmax(per[0][0][list_action])]
        per[0][0] += per[2][int(action)]
        if env.getReward(state) != -1:
            per[5][0][0] += 1
            per[0][0] = per[3][0].copy()
            if env.getReward(state) == 1:
                per[6][0][0] += 1
            if per[5][0][0] % 1000 == 0:
                if per[6][0][0] / 1000 > per[8][0][0]:
                    per[8][0][0] = per[6][0][0] / 1000
                    per[10] = per[2].copy()
                per[2] = np.random.rand(env.getActionSize(), env.getActionSize())
                per[6][0][0] = 0

    else:
        actions = state.reshape(1, -1).dot(per[1])
        per[0][0] += actions[0] / np.max(actions)
        list_action = np.where(env.getValidActions(state) == 1)[0]
        action = list_action[np.argmax(per[0][0][list_action])]
        per[0][0] += per[2][int(action)]
        if env.getReward(state) != -1:
            per[5][0][0] += 1
            per[0][0] = per[3][0].copy()
            if env.getReward(state) == 1:
                per[6][0][0] += 1
            if per[5][0][0] % 1000 == 0:
                if (per[5][0][0] % 100000) % 2 == 1:
                    if per[6][0][0] / 1000 > per[8][0][0]:
                        per[8][0][0] = per[6][0][0] / 1000
                        per[10] = per[2].copy()
                    per[2] = np.random.rand(env.getActionSize(), env.getActionSize())
                    per[6][0][0] = 0
                else:
                    if per[6][0][0] / 1000 > per[7][0][0]:
                        per[7][0][0] = per[6][0][0] / 1000
                        per[9] = per[1].copy()
                    per[1] = (
                        np.random.rand(env.getStateSize(), env.getActionSize()) * 2 - 1
                    )
                    per[6][0][0] = 0
    return action, per


perx = [
    np.random.choice(
        env.getActionSize(), size=env.getActionSize(), replace=False
    ).reshape(1, -1)
    * 1.0,
    np.random.rand(env.getStateSize(), env.getActionSize()) * 2 - 1,
    np.random.rand(env.getActionSize(), env.getActionSize()),
    np.zeros((1, env.getActionSize())),
    np.zeros((1, 1)),
    np.zeros((1, 1)),
    np.zeros((1, 1)),
    np.zeros((1, 1)),
    np.zeros((1, 1)),
    np.random.rand(env.getStateSize(), env.getActionSize()) * 2 - 1,
    np.random.rand(env.getActionSize(), env.getActionSize()),
]

# per[0]: arr ban đầu: size = ActionSize, value in range(ActionSize)
# per[1]: arr weight: np.random.rand(StateSize,ActionSize) * 2 -1
# per[2]: arr bias: np.random.rand(ActionSize,ActionSize)
# per[3]: sum all per[0] victory
# per[4]: count turn in one game
# per[5]: count game
#  per[6]: count win in 1000 games
# per[7]: max win in 1000 games phase 2
# per[8]: max win in 1000 games phase 3
# per[9]: per[1] best.
# per[10]: per[2] best


@njit
def Test(state, per):
    actions = state.reshape(1, -1).dot(per[1])
    per[0][0] += actions[0] / np.max(actions)
    list_action = np.where(env.getValidActions(state) == 1)[0]
    action = list_action[np.argmax(per[0][0][list_action])]
    per[0][0] += per[2][action]
    if env.getReward(state) != -1:
        per[0][0] = per[3][0]
    return action, per


def convert_to_save(perData):
    if len(perData) == 4:
        raise Exception("Data này đã được convert rồi.")
    data = List()
    data.append(perData[0])
    data.append(perData[9])
    data.append(perData[10])
    data.append(perData[3])
    return data


def convert_to_test(perData):
    return List(perData)
