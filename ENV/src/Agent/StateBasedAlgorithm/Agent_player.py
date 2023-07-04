#  state based
#  small NN deep
import sys
import numpy as np
from numba import njit
import env

game_name = sys.argv[1]
env.make(game_name)


from numba.typed import List


def DataAgent():
    per = List(
        [
            np.zeros(
                (100, env.getStateSize(), env.getActionSize())
            ),  # [0][value][idS] khi mở đầu game
            np.zeros(
                (100, env.getStateSize(), env.getActionSize())
            ),  # [1][value][ids] lưu lại cuối cùng
            np.zeros((1, 1, 1)),  # [2][0][0][0] vừa thắng hay thua
        ]
    )
    return per


@njit()
def Train(state, per):
    actions = env.getValidActions(state)
    weights = np.zeros(env.getActionSize())
    if per[2][0][0][0] == 0:
        temp = np.arange(env.getActionSize(), dtype=np.float64)
        np.random.shuffle(temp)
        weights += temp
        for ids in range(env.getStateSize()):
            if state[ids] < 100:
                per[0][int(state[ids])][ids] += temp / np.max(temp)
    else:
        for ids in range(env.getStateSize()):
            if state[ids] < 100:
                weights += (
                    np.argsort(np.argsort(per[0][int(state[ids])][ids]))
                    / env.getActionSize()
                )
    output = weights * actions + actions
    action = np.argmax(output)
    win = env.getReward(state)
    if win != -1:
        if win == 1:
            #  print("đây")
            per[1] += per[0]
            per[2][0][0][0] = 1
        else:
            per[0][:, :, :] = 0.0
            per[2][0][0][0] = 0
    return action, per


@njit
def Test(state, per):
    state_int = state.astype(np.int64)
    stateSize = env.getStateSize()
    actionSize = env.getActionSize()
    where_ = np.where((state_int < per[stateSize][0]) & (state_int >= 0))[0]
    weight = np.zeros(actionSize)
    for i in where_:
        weight += per[i][state_int[i]]

    actions = env.getValidActions(state)
    output = (weight + 1) * actions
    #  action = np.argmax(output)
    list_action = np.where(actions == 1)[0]
    action = list_action[np.argmax(output[list_action])]
    return action, per


def convert_to_save(perData):
    if len(perData) == env.getStateSize() + 1:
        raise Exception("Data này đã được convert rồi.")
    data = List()
    arr = np.zeros(env.getStateSize(), int)
    for i in range(env.getStateSize()):
        for j in range(100):
            if (perData[1][j, i] == 0).all():
                check = True
                for k in range(j, 100):
                    if (perData[1][k, i] != 0).any():
                        check = False
                        break
                if check:
                    arr[i] = j
                    break
        else:
            arr[i] = 100

    for i in range(env.getStateSize()):
        data.append(perData[1][: arr[i], i])
        for j in range(data[i].shape[0]):
            data[i][j] = np.argsort(np.argsort(data[i][j])) / float(env.getActionSize())

    data.append(np.array([arr], float))
    return data


def convert_to_test(perData):
    return List(perData)
