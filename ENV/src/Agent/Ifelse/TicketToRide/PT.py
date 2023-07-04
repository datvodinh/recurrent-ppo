import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


@njit()
def DataAgent():
    return np.array([0.0])


@njit()
def getRoadLen():
    return np.array(
        [
            2,
            3,
            3,
            2,
            3,
            3,
            2,
            4,
            4,
            4,
            4,
            4,
            3,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            4,
            4,
            3,
            4,
            3,
            3,
            2,
            2,
            2,
            4,
            4,
            4,
            2,
            4,
            4,
            2,
            4,
            3,
            2,
            4,
            6,
            3,
            3,
            4,
            4,
            4,
            8,
            2,
            6,
            2,
            3,
            3,
            2,
            4,
            4,
            3,
            3,
            3,
            3,
            3,
            4,
            2,
            4,
            4,
            4,
            1,
            1,
            2,
            3,
            2,
            3,
            3,
            4,
            4,
            3,
            2,
            2,
            4,
            4,
            3,
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            3,
            3,
            2,
            2,
            2,
            3,
            2,
            2,
            2,
            2,
            3,
            4,
        ]
    )


@njit()
def getRouteScore():
    return np.array(
        [
            5,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            9,
            9,
            10,
            10,
            10,
            10,
            10,
            11,
            11,
            12,
            12,
            13,
            20,
            20,
            20,
            21,
            21,
            21,
        ]
    )


@njit()
def Test(state, per):
    actions = env.getValidActions(state)
    actions = np.where(actions == 1)[0]
    pickTrainCarActions = actions[(actions >= 147) & (actions < 157)]
    if len(pickTrainCarActions) > 0:
        if 156 in actions:
            return 156, per
        else:
            for i in range(147, 156):
                if i in actions:
                    return i, per
    dropRouteActions = actions[(actions >= 101) & (actions < 147)] - 101
    if len(dropRouteActions) > 0:
        routeScore = getRouteScore()
        scoreRouteDropActions = routeScore[dropRouteActions]
        maxScore = np.argmax(scoreRouteDropActions)
        return actions[maxScore], per
    chooseTrainCardAction = actions[(actions >= 157) & (actions < 166)]
    if len(chooseTrainCardAction) > 0:
        my_train_car_amount = state[10:19]
        my_train_car_amount_can_play = my_train_car_amount[chooseTrainCardAction - 157]
        max_id = np.argmax(my_train_car_amount_can_play)
        if max_id + 157 in actions:
            return max_id + 157, per
        else:
            return np.random.choice(actions), per
    buildRoadActions = actions[(actions >= 0) & (actions < 101)]
    if len(buildRoadActions) > 0:
        roadLen = getRoadLen()
        myBuildRoadActionsLen = roadLen[buildRoadActions]
        max_road_len_id = np.argmax(myBuildRoadActionsLen)
        return actions[max_road_len_id], per
    if 170 in actions:
        return 170, per
    return np.random.choice(actions), per
