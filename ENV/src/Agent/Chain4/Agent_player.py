import sys
import numpy as np
from numba import njit
import env

game_name = sys.argv[1]
env.make(game_name)


from numba.typed import Dict

CHAIN_LENGTH = 4


def DataAgent():
    perData = Dict()
    temp_ = Dict()
    temp_[0] = np.full(0, 0, dtype=np.int64)
    temp_.clear()

    perData[0] = temp_.copy()  #  Các biến dùng trong thuật toán
    perData[1] = temp_.copy()  #  Temp bias của các chuỗi
    perData[2] = temp_.copy()  #  Per bias của các chuỗi
    perData[3] = temp_.copy()  #  Đếm số lần xuất hiện của các chuỗi

    #  Các biến dùng trong thuật toán
    perData[0][0] = np.full(1, 0, dtype=np.int64)  #  Đếm số trận
    perData[0][1] = np.full(CHAIN_LENGTH, 0, dtype=np.int64)  #  Lưu lịch sử
    perData[0][2] = np.full(1, 0, dtype=np.int64)  #  Đếm số turn trong trận
    perData[0][3] = np.full(10000, -1, dtype=np.int64)  #  Lưu các chuỗi xuất hiện

    #  Bias tự do (Temp)
    perData[1][-1] = np.arange(env.getActionSize(), dtype=np.int64) + 1
    np.random.shuffle(perData[1][-1])

    #  Bias tự do (Per)
    perData[2][-1] = np.full(env.getActionSize(), 0, dtype=np.int64) + 1

    #
    return perData


@njit()
def encode(arr_chain):
    res = 0
    for i in range(arr_chain.shape[0]):
        res += arr_chain[i] * env.getActionSize() ** i

    return res


@njit()
def Train(state, perData):
    count_match = perData[0][0]
    action_history = perData[0][1]
    count_turn = perData[0][2]
    chain_history = perData[0][3]

    temp_bias = perData[1]
    per_bias = perData[2]
    count_bias = perData[3]

    reward = env.getReward(state)
    validActions = env.getValidActions(state)

    if count_match[0] < 10000:
        if count_match[0] == -1:
            #  print("ahihi")
            arr_key = np.full(len(count_bias), 0, dtype=np.int64)
            arr_value = np.full(len(count_bias), 0, dtype=np.int64)
            i = 0
            for key in count_bias:
                arr_key[i] = key
                arr_value[i] = count_bias[key][0]
                i += 1

            for i in range(arr_key.shape[0]):
                if arr_value[i] <= 1:
                    #  print(arr_key[i], arr_value[i], "drop")
                    count_bias.pop(arr_key[i])
                    temp_bias.pop(arr_key[i])
                    per_bias.pop(arr_key[i])
                else:
                    count_bias[arr_key[i]][0] = 0

            count_match[0] += 1

        if reward == -1:
            if count_turn[0] < CHAIN_LENGTH:
                action = np.argmax(validActions * temp_bias[-1])
                action_history[count_turn[0]] = action
            else:
                key = encode(action_history)
                if key not in count_bias:
                    count_bias[key] = np.full(1, 0, dtype=np.int64)
                    per_bias[key] = np.full(env.getActionSize(), 0, dtype=np.int64) + 1
                    temp_bias[key] = np.arange(env.getActionSize(), dtype=np.int64) + 1
                    np.random.shuffle(temp_bias[key])

                action = np.argmax(validActions * temp_bias[key])
                action_history[0 : CHAIN_LENGTH - 1] = action_history[1:CHAIN_LENGTH]
                action_history[CHAIN_LENGTH - 1] = action
                chain_history[count_turn[0] - CHAIN_LENGTH] = key

            count_turn[0] += 1
        else:
            if reward == 1:
                per_bias[-1] += temp_bias[-1]
                for i in range(count_turn[0] - CHAIN_LENGTH):
                    per_bias[chain_history[i]] += temp_bias[chain_history[i]]
                    count_bias[chain_history[i]][0] += 1
            else:
                np.random.shuffle(temp_bias[-1])
                for i in range(count_turn[0] - CHAIN_LENGTH):
                    np.random.shuffle(temp_bias[chain_history[i]])
                    count_bias[chain_history[i]][0] += 1

            count_match[0] += 1
            count_turn[0] = 0
            action_history[:] = 0
            chain_history[:] = -1

            action = np.argmax(validActions * temp_bias[-1])

        return action, perData

    elif count_match[0] < 100000:
        if reward == -1:
            if count_turn[0] < CHAIN_LENGTH:
                action = np.argmax(validActions * temp_bias[-1])
                action_history[count_turn[0]] = action
            else:
                key = encode(action_history)
                if key not in count_bias:
                    action = np.argmax(validActions * temp_bias[-1])
                    check = False
                else:
                    action = np.argmax(validActions * temp_bias[key])
                    check = True

                action_history[0 : CHAIN_LENGTH - 1] = action_history[1:CHAIN_LENGTH]
                action_history[CHAIN_LENGTH - 1] = action
                if check:
                    chain_history[count_turn[0] - CHAIN_LENGTH] = key

            count_turn[0] += 1
        else:
            if reward == 1:
                per_bias[-1] += temp_bias[-1]
                for i in range(count_turn[0] - CHAIN_LENGTH):
                    if chain_history[i] == -1:
                        break

                    per_bias[chain_history[i]] += temp_bias[chain_history[i]]
                    count_bias[chain_history[i]][0] += 1
            else:
                np.random.shuffle(temp_bias[-1])
                for i in range(count_turn[0] - CHAIN_LENGTH):
                    if chain_history[i] == -1:
                        break

                    np.random.shuffle(temp_bias[chain_history[i]])
                    count_bias[chain_history[i]][0] += 1

            count_match[0] += 1
            count_turn[0] = 0
            action_history[:] = 0
            chain_history[:] = -1
            if count_match[0] == 100000:
                count_match[0] = -1

            action = np.argmax(validActions * temp_bias[-1])

        return action, perData


@njit()
def Test(state, perData):
    per_bias = perData[2]
    action_history = perData[0][1]
    count_turn = perData[0][2]

    reward = env.getReward(state)
    validActions = env.getValidActions(state)

    if reward == -1:
        if count_turn[0] < CHAIN_LENGTH:
            action = np.argmax(validActions * per_bias[-1] + validActions)
            action_history[count_turn[0]] = action
        else:
            key = encode(action_history)
            if key not in per_bias:
                action = np.argmax(validActions * per_bias[-1] + validActions)
            else:
                action = np.argmax(validActions * per_bias[key] + validActions)

            action_history[0 : CHAIN_LENGTH - 1] = action_history[1:CHAIN_LENGTH]
            action_history[CHAIN_LENGTH - 1] = action

        count_turn[0] += 1
    else:
        count_turn[0] = 0
        action_history[:] = 0

        action = np.argmax(validActions * per_bias[-1] + validActions)

    return action, perData


def convert_to_save(perData):
    if 1 not in perData.keys():
        raise Exception("Data này đã được convert rồi.")
    temp = dict()
    for key in perData.keys():
        temp[key] = dict()
        for key1 in perData[key].keys():
            temp[key][key1] = perData[key][key1]

    data = dict()
    data[0] = temp[0]
    data[0].pop(0)
    data[0].pop(3)
    for key in data[0].keys():
        data[0][key] = data[0][key].astype(np.int16)

    data[2] = temp[2]
    for key in data[2].keys():
        data[2][key] = np.argsort(np.argsort(data[2][key])).astype(np.int16)

    return data


def convert_to_test(perSave):
    perSave = perSave[()]
    temp = Dict()
    for key in perSave.keys():
        temp1 = Dict()
        for key1 in perSave[key].keys():
            temp1[key1] = perSave[key][key1]

        temp[key] = temp1

    return temp
