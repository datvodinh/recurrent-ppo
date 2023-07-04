import sys
import numpy as np
from numba import njit
import env

game_name = sys.argv[1]
env.make(game_name)


from numba.typed import List


def convert_to_save(perData):
    if type(perData) == np.ndarray:
        raise Exception("Data này đã được convert rồi.")
    return perData[0][0]


def convert_to_test(perData):
    return perData


@njit()
def Train(state, per):
    if per[4][0][0] < 10000:
        if per[3][0][0] == -1:
            choice = np.argmin(per[2][0])
            per[3][0][0] = choice
            per[2][0][choice] += 1

        choice_id = int(per[3][0][0])
        bias = per[0][choice_id]
        actions = env.getValidActions(state)
        kq = actions * bias
        action = np.argmax(kq)

        win = env.getReward(state)
        if win != -1:
            if win == 1:
                per[1][0][choice_id] += 1

            per[4][0][0] += 1
            if per[4][0][0] == 10000:
                win_rate = per[1][0] / per[2][0]
                best = np.argmax(win_rate)
                per[0][0] = per[0][best]
                per[3][0][0] = win_rate[best]
                per[2][0] = 0
                per[1][0] = 0
            else:
                per[3][0][0] = -1

        return action, per

    if per[4][0][0] < 11000:
        actions = env.getValidActions(state)
        kq = actions * per[0][0]
        action = np.argmax(kq)
        list_action = np.where(actions == 1)[0]
        if len(list_action) > 1:
            per[6][0][action] += 1
            for a in list_action:
                per[5][a][list_action] += 1
                per[5][a][a] = 0

        win = env.getReward(state)
        if win != -1:
            id_match = int(per[4][0][0]) % 1000
            per[4][0][0] += 1
            per[7][0][id_match] = win
            if per[4][0][0] == 11000:
                win_rate = np.sum(per[7][0]) / 100
                if win_rate > per[3][0][0]:
                    per[3][0][0] = win_rate

        return action, per

    if per[2][0][99] == 0:
        actions = env.getValidActions(state)
        kq = actions * per[0][0]
        action = np.argmax(kq)
        list_action = np.where(actions == 1)[0]
        if len(list_action) > 1:
            per[6][0][action] += 1
            for a in list_action:
                per[5][a][list_action] += 1
                per[5][a][a] = 0

        win = env.getReward(state)
        if win != -1:
            id_match = int(per[4][0][0]) % 1000
            per[4][0][0] += 1
            per[7][0][id_match] = win
            win_rate = np.sum(per[7][0]) / 100
            if win_rate > per[3][0][0]:
                per[3][0][0] = win_rate
            else:
                if int(per[4][0][0]) % 3000 == 0:
                    num_1 = np.count_nonzero(per[6][0] != 0)
                    check = False
                    for l_i in range(num_1):
                        action_max = np.argmax(per[6][0])
                        per[6][0][action_max] = 0
                        aptgt = per[5][action_max].copy()
                        num_2 = np.count_nonzero(aptgt != 0)
                        for l_j in range(num_2):
                            action_choice = np.argmax(aptgt)
                            aptgt[action_choice] = 0
                            if per[8][action_max][action_choice] == 0:
                                per[8][action_max][action_choice] = 1
                                per[8][action_choice][action_max] = 1
                                per[0][1] = per[0][0]
                                temp_s = per[0][1][action_max]
                                per[0][1][action_max] = per[0][1][action_choice]
                                per[0][1][action_choice] = temp_s

                                check = True
                                break

                        if check:
                            break

                    if check:
                        per[2][0][99] = 1
                        per[1][0][0:2] = 0
                        per[2][0][0:2] = 0

        return action, per

    if per[2][0][99] == 1:
        if per[2][0][0] < 2000:
            actions = env.getValidActions(state)
            kq = actions * per[0][0]
            action = np.argmax(kq)

            win = env.getReward(state)
            if win != -1:
                per[2][0][0] += 1
                if win == 1:
                    per[1][0][0] += 1

        else:
            actions = env.getValidActions(state)
            kq = actions * per[0][1]
            action = np.argmax(kq)

            win = env.getReward(state)
            if win != -1:
                per[2][0][1] += 1
                if win == 1:
                    per[1][0][1] += 1

                if per[2][0][1] == 500:
                    num_1 = per[1][0][0] / per[2][0][0]
                    num_2 = per[1][0][1] / per[2][0][1]

                    if num_2 > num_1:
                        per[0][0] = per[0][1]

                    per[2][0][99] = 0

                    per[6][:, :] = 0.0
                    per[5][:, :] = 0.0

        return action, per


@njit()
def Test(state, per):
    actions = env.getValidActions(state)
    kq = actions * per + actions
    action = np.argmax(kq)
    return action, per


@njit()
def DataAgent():
    per_Ann = List()
    per_Ann.append(np.random.rand(100, env.getActionSize()))  #  0
    per_Ann.append(np.zeros((1, 100)))  #  1
    per_Ann.append(np.zeros((1, 100)))  #  2
    per_Ann.append(np.array([[-1.0]]))  #  3
    per_Ann.append(np.array([[0.0]]))  #  4
    per_Ann.append(np.zeros((env.getActionSize(), env.getActionSize())))  #  5
    per_Ann.append(np.zeros((1, env.getActionSize())))  #  6
    per_Ann.append(np.zeros((1, 1000)))  #  7
    per_Ann.append(np.zeros((env.getActionSize(), env.getActionSize())))  #  8
    return per_Ann
