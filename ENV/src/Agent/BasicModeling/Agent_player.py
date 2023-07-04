import sys
import numpy as np
from numba import njit
import env

game_name = sys.argv[1]
env.make(game_name)


from numba.typed import List

MAX_DEPTH = 7


def DataAgent():
    perData = List()

    perData.append(np.array([[0.0]], dtype=np.float64))  #  0: Đếm số trận đấu đã train
    perData.append(
        np.full((2, env.getActionSize()), 1e-9, dtype=np.float64)
    )  #  Bias và temp Bias
    perData[1][1] = np.arange(1, env.getActionSize() + 1, dtype=np.float64)
    np.random.shuffle(perData[1][1])

    perData.append(
        np.full((2, env.getStateSize()), 0.0, dtype=np.float64)
    )  #  Minimum state và maximum state
    perData[2][0] = 1e9
    perData[2][1] = -1e9

    return perData


@njit
def Train(state, perData):
    if perData[0][0][0] < 10000:
        validActions = env.getValidActions(state)
        bias = perData[1][1] * validActions
        action = np.argmax(bias)

        reward = env.getReward(state)
        if reward != -1:
            perData[0][0][0] += 1
            if reward == 1:
                perData[1][0] += perData[1][1]
            elif reward == 0:
                perData[1][0] -= perData[1][1]

            np.random.shuffle(perData[1][1])
            if perData[0][0][0] == 10000:
                min_ = np.min(perData[1][0])
                if min_ < 0:
                    perData[1][0] += 1.0 - min_

                perData[1][0] /= np.max(perData[1][0])
                perData[1][0] **= 2

        return action, perData

    elif perData[0][0][0] < 20000 and len(perData) < 11:
        perData[2][0] = np.minimum(perData[2][0], state)
        perData[2][1] = np.maximum(perData[2][1], state)

        validActions = env.getValidActions(state)
        actions = np.where(validActions == 1)[0]
        a_idx = np.random.randint(0, actions.shape[0])

        reward = env.getReward(state)
        if reward != -1:
            perData[0][0][0] += 1
            if perData[0][0][0] == 20000:
                size_ = int(np.round(np.max(perData[2][1] - perData[2][0])))
                if len(perData) < 11:
                    perData.append(
                        np.full((env.getStateSize(), size_), 0.0, dtype=np.float64)
                    )  #  3 win_state
                    perData.append(
                        np.full((env.getStateSize(), size_), 1e-9, dtype=np.float64)
                    )  #  4 last_state
                    perData.append(
                        np.full(
                            (env.getActionSize(), env.getStateSize()),
                            0.0,
                            dtype=np.float64,
                        )
                    )  #  5 Delta_state
                    perData.append(
                        np.full(
                            (env.getActionSize(), env.getStateSize()),
                            0.0,
                            dtype=np.float64,
                        )
                    )  #  6 Pre_state
                    perData.append(
                        np.full((env.getActionSize(), 1), 0.0, dtype=np.float64)
                    )  #  7 Số lần xuất hiện
                    perData.append(
                        np.array([[-1.0]], dtype=np.float64)
                    )  #  8 Pre_action
                    perData.append(
                        np.full((env.getStateSize(), size_), 0.01, dtype=np.float64)
                    )  #  9 = 3 / 4
                    perData.append(
                        np.full(
                            (env.getActionSize(), env.getStateSize()),
                            0.0,
                            dtype=np.float64,
                        )
                    )  #  10 = 5 / 7

        return actions[a_idx], perData

    elif perData[0][0][0] < 100000:
        if perData[8][0][0] > -1.0:
            perData[5][int(perData[8][0][0])] += (
                state - perData[6][int(perData[8][0][0])]
            )
            perData[7][0][int(perData[8][0][0])] += 1

        validActions = env.getValidActions(state)
        actions = np.where(validActions == 1)[0]
        a_idx = np.random.randint(0, actions.shape[0])

        perData[8][0][0] = actions[a_idx]
        perData[6][actions[a_idx]] = state

        reward = env.getReward(state)
        if reward != -1:
            perData[0][0][0] += 1
            perData[8][0][0] = -1.0
            arr = np.where(state < perData[2][0], perData[2][0], state)
            arr = np.where(arr > perData[2][1], perData[2][1], arr)
            arr = np.round(arr, 0, arr).astype(np.int64)
            for i in range(arr.shape[0]):
                perData[4][i][arr[i]] += 1

            if reward == 1:
                for i in range(arr.shape[0]):
                    perData[3][i][arr[i]] += 1

            if perData[0][0][0] == 100000:
                perData[0][0][0] = 0
                perData[9] = perData[3] / perData[4]
                perData[10] = perData[5] / perData[7]

        return actions[a_idx], perData


@njit
def get_value_state(state, value):
    score = 0.0
    state = state.astype(np.int64)
    for i in range(state.shape[0]):
        score += value[i][state[i]]

    return score / state.shape[0]


@njit
def weighted_random(p: np.ndarray):
    a = np.sum(p)
    b = np.random.uniform(0, a)
    for i in range(len(p)):
        b -= p[i]
        if b <= 0:
            return i

    return 999999


@njit
def scoring(action, state, depth, delta, bias, score):
    new_state = state + delta[action]
    new_state_round = np.empty_like(new_state)
    new_state_round = np.round(new_state, 0, new_state_round)
    validActions = env.getValidActions(new_state_round)
    actions = np.where(validActions == 1)[0]
    reward = env.getReward(new_state_round)

    if depth >= MAX_DEPTH:
        return get_value_state(new_state_round, score)

    if reward == 1:
        return (MAX_DEPTH - depth + 1) / (MAX_DEPTH + 1) + get_value_state(
            new_state_round, score
        )

    if reward == 0:
        return (depth - MAX_DEPTH - 1) / (MAX_DEPTH + 1) + get_value_state(
            new_state_round, score
        )

    if actions.shape[0] == 0:
        return (depth - MAX_DEPTH) / (MAX_DEPTH + 1) + get_value_state(
            new_state_round, score
        )

    temp = bias[actions]
    a_idx = weighted_random(temp)
    depth_ = depth + 1
    return scoring(actions[a_idx], new_state, depth_, delta, bias, score)


@njit
def Test(state, perData):
    validActions = env.getValidActions(state)
    actions = np.where(validActions == 1)[0]
    score = np.full(actions.shape, 0.0, dtype=np.float64)
    for i in range(actions.shape[0]):
        score[i] = scoring(actions[i], state, 0, perData[0], perData[1][0], perData[2])

    a_idx = np.argmax(score)
    return actions[a_idx], perData


def convert_to_save(perData):
    data = List()
    data.append(perData[10])
    min_ = np.min(perData[1][0])
    if min_ < 0:
        perData[1][0] += 1.0 - min_

    perData[1][0] /= np.max(perData[1][0])
    perData[1][0] **= 2
    data.append(perData[1])
    data.append(perData[9])
    return data


def convert_to_test(perData):
    return List(perData)
