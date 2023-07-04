import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


@njit()
def DataAgent():
    return np.array(0.0)


@njit()
def getPointTILE():
    return np.array(
        [
            [0, -1, -1],  #  0
            [0, 1, -1],  #  1
            [1, -1, -1],  #  2
            [1, -1, -1],  #  3
            [1, 2, -1],  #  4
            [2, -1, -1],  #  5
            [2, 3, -1],  #  6
            [3, -1, -1],  #  7
            [3, -1, -1],  #  8
            [3, 4, -1],  #  9
            [4, -1, -1],  #  10
            [4, 5, -1],  #  11
            [5, -1, -1],  #  12
            [5, -1, -1],  #  13
            [5, 6, -1],  #  14
            [6, -1, -1],  #  15
            [6, 7, -1],  #  16
            [7, -1, -1],  #  17
            [7, -1, -1],  #  18
            [7, 8, -1],  #  19
            [8, -1, -1],  #  20
            [8, 9, -1],  #  21
            [9, -1, -1],  #  22
            [9, -1, -1],  #  23
            [9, 10, -1],  #  24
            [10, -1, -1],  #  25
            [10, 11, -1],  #  26
            [11, -1, -1],  #  27
            [11, -1, -1],  #  28
            [0, 11, -1],  #  29
            [0, 11, 12],  #  30
            [10, 11, 12],  #  31
            [10, 12, 13],  #  32
            [9, 10, 13],  #  33
            [8, 9, 13],  #  34
            [8, 13, 14],  #  35
            [7, 8, 14],  #  36
            [6, 7, 14],  #  37
            [6, 14, 15],  #  38
            [5, 6, 15],  #  39
            [4, 5, 15],  #  40
            [4, 15, 16],  #  41
            [3, 4, 16],  #  42
            [2, 3, 16],  #  43
            [2, 16, 17],  #  44
            [1, 2, 17],  #  45
            [0, 1, 17],  #  46
            [0, 12, 17],  #  47
            [12, 17, 18],  #  48
            [12, 13, 18],  #  49
            [13, 14, 18],  #  50
            [14, 15, 18],  #  51
            [15, 16, 18],  #  52
            [16, 17, 18],
        ]  #  53
    )


@njit()
def getValuestLand(state, actions):
    resourceLand_binary = state[0:114].reshape(19, 6)
    hightest_pro = np.array([7, 6, 8])
    numb_on_TILE = state[133:361].reshape(19, 12)
    #  print('numb_on_TILE',numb_on_TILE)
    PointTILE = getPointTILE()
    PointTILE = PointTILE[actions]
    point_point = np.zeros(len(actions))
    numb_pro_1 = [6]
    numb_pro_2 = [5, 7]
    numb_pro_3 = [4, 8]
    numb_pro_4 = [3, 9]
    numb_pro_5 = [2, 10]
    numb_pro_6 = [1, 11]
    for i in range(len(point_point)):
        land_TILEs = PointTILE[i]

        for TILE in land_TILEs:
            if TILE != -1:
                numb_TILE = numb_on_TILE[TILE]
                numb_TILE = np.where(numb_TILE == 1)[0][0]
                if numb_TILE in numb_pro_1:
                    point_point[i] += 0.6
                if numb_TILE in numb_pro_2:
                    point_point[i] += 0.5
                if numb_TILE in numb_pro_3:
                    point_point[i] += 0.4
                if numb_TILE in numb_pro_4:
                    point_point[i] += 0.3
                if numb_TILE in numb_pro_5:
                    point_point[i] += 0.2
                if numb_TILE in numb_pro_6:
                    point_point[i] += 0.1
    max_point_point = np.argmax(point_point)
    return actions[max_point_point]


@njit()
def Test(state, per):
    actions = env.getValidActions(state)
    actions = np.where(actions == 1)[0]
    phase = state[1273:1286]
    phase = np.where(phase == 1)[0]
    my_infor = state[421:629]
    my_stock = my_infor[0:5]
    if 94 in actions:
        return 94, per
    elif 84 in actions:
        return 84, per
    elif 85 in actions:
        return 85, per
    elif 86 in actions:
        return 86, per
    elif 87 in actions and sum(my_stock) > 7:
        return 87, per
    elif 83 in actions:
        return 83, per
    if phase == 0:
        land = getValuestLand(state, actions)
        return land, per
    return np.random.choice(actions), per
