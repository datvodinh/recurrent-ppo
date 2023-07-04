import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


@njit
def DataAgent():
    return [np.zeros((1, 1))]


@njit()
def getTheTileNeighbor():
    POINT_TILE = np.array(
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
    return POINT_TILE


@njit()
def getThePointNeighbor():
    TILE_POINT = np.array(
        [
            [0, 1, 29, 30, 46, 47],  #  0
            [1, 2, 3, 4, 45, 46],  #  1
            [4, 5, 6, 43, 44, 45],  #  2
            [6, 7, 8, 9, 42, 43],  #  3
            [9, 10, 11, 40, 41, 42],  #  4
            [11, 12, 13, 14, 39, 40],  #  5
            [14, 15, 16, 37, 38, 39],  #  6
            [16, 17, 18, 19, 36, 37],  #  7
            [19, 20, 21, 34, 35, 36],  #  8
            [21, 22, 23, 24, 33, 34],  #  9
            [24, 25, 26, 31, 32, 33],  #  10
            [26, 27, 28, 29, 30, 31],  #  11
            [30, 31, 32, 47, 48, 49],  #  12
            [32, 33, 34, 35, 49, 50],  #  13
            [35, 36, 37, 38, 50, 51],  #  14
            [38, 39, 40, 41, 51, 52],  #  15
            [41, 42, 43, 44, 52, 53],  #  16
            [44, 45, 46, 47, 48, 53],  #  17
            [48, 49, 50, 51, 52, 53],
        ]  #  18
    )
    return TILE_POINT


@njit()
def getTileScore(state):
    tileScore = np.zeros(19)
    tileDice = np.zeros(19)

    idx2 = np.where(state[114:133] == 1)
    tileDice[idx2] = 2
    score = state[1048:1219].reshape(9, 19)  # Điểm dưới dạng nhị phân
    for i in range(9):
        idx = np.where(score[i] == 1)
        if i <= 3:
            tileDice[idx] = i + 3
        else:
            tileDice[idx] = i + 4
    #  print(tileDice)

    scoreDice = [0, 0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1]
    for i in range(19):
        tileScore[i] = scoreDice[int(tileDice[i])]
    #  print(tileScore)
    return tileScore


@njit
def getTileRobber(state):
    tileScore = getTileScore(state)
    robberScore = np.zeros(19)
    neighborPoint = getThePointNeighbor()
    myHouse = state[276:330] + 2 * state[330:384]
    rivalHouse = (
        state[466:520]
        + state[651:705]
        + state[836:890]
        + 2 * state[520:574]
        + 2 * state[705:759]
        + 2 * state[890:944]
    )
    for i in range(19):
        for point in neighborPoint[i]:
            robberScore[i] += (-5 * myHouse[point] + rivalHouse[point]) * tileScore[i]

    robberTile = np.argmax(robberScore)
    #  print(tileScore)
    #  print(robberScore)
    #  for i in range(54):
    #      print(i, myHouse[i],rivalHouse[i])
    #  print(robberTile,'-------------------------------')
    return robberTile + 64


@njit()
def Test(state, per):
    ValidAction = env.getValidActions(state)
    ValidAction = np.where(ValidAction == 1)[0]
    returnAction = -1
    if ValidAction[0] in range(0, 54):  # ACTION chọn POINT
        valuePosition = np.zeros(len(ValidAction))
        tileNeighbor = getTheTileNeighbor()
        tileScore = getTileScore(state)
        for i in range(len(ValidAction)):
            point = ValidAction[i]
            for tile in tileNeighbor[point]:
                if tile != -1:
                    valuePosition[i] += tileScore[tile] + 1.5
        idxmax = np.argmax(valuePosition)
        returnAction = ValidAction[idxmax]

    elif ValidAction[0] in range(64, 82):
        returnAction = getTileRobber(state)

    if 87 in ValidAction:
        returnAction = 87
    elif 88 in ValidAction:
        returnAction = 88
    elif 89 in ValidAction:
        returnAction = 89
    elif 90 in ValidAction:
        returnAction = 90
    elif state[957] == 1:
        #  if np.sum(state[1028:1033]) >= 1 and 103 in ValidAction:
        if 103 in ValidAction:
            returnAction = 103
    elif 93 in ValidAction and 94 in ValidAction:
        if np.sum(state[1044:1047]) > 0:  # Người khác trade
            if np.sum(state[1028:1033]) > np.sum(state[1033:1038]):
                returnAction = 93

            #  if np.sum(state[1028:1033]) == np.sum(state[1033:1038]):
            #      myResource = state[193:198]
            #      if np.sum(myResource[np.where(state[1028:1033]>0)])+1  < np.sum(myResource[np.where(state[1033:1038]>0)]):
            #          returnAction = 93

            if returnAction == -1:
                returnAction = 94
            #  print('My resource:',state[193:198])
            #  print(state[1028:1033],state[1033:1038])
            #  print(returnAction,"----------")

    if returnAction == -1:
        returnAction = ValidAction[np.random.randint(len(ValidAction))]
    #  print('------------')
    #  print(state[193:198])
    #  print(ValidAction, returnAction)
    return returnAction, per
