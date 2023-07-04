import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


def DataAgent():
    per = np.array([0])
    return per


@njit()
def Test(state, per):
    validActions = env.getValidActions(state)
    validActions = np.where(validActions == 1)[0]

    purchaseCardActions = validActions[(validActions >= 1) & (validActions < 13)]
    if len(purchaseCardActions) > 0:
        action = purchaseCardActions[0]
        for purchaseCardAction in purchaseCardActions:
            if (
                state[18 + 11 * (purchaseCardAction - 1)]
                > state[18 + 11 * (action - 1)]
            ):
                action = purchaseCardAction
        return action, per

    reverseFacedownCardActions = validActions[
        (validActions >= 13) & (validActions < 16)
    ]
    if len(reverseFacedownCardActions) > 0:
        action = reverseFacedownCardActions[
            np.random.randint(len(reverseFacedownCardActions))
        ]
        return action, per

    faceupCards = state[18:150].reshape(12, -1)
    faceupCards_l0_l1 = faceupCards[:8]
    faceupCards_l2 = faceupCards[8:]
    totalCostFaceUpCards_l0_l1 = np.sum(faceupCards_l0_l1, axis=1)[-5:]
    totalCostFaceUpCards_l2 = np.sum(faceupCards_l2, axis=1)[-5:]
    if np.sum(totalCostFaceUpCards_l0_l1) > 0:
        mostThreeTokens = (-totalCostFaceUpCards_l0_l1).argsort()[:3] + 31
    else:
        mostThreeTokens = (-totalCostFaceUpCards_l2).argsort()[:3] + 31
    takeTokenActions = validActions[(validActions >= 31) & (validActions < 36)]
    mostAvalableThreeTokens = np.intersect1d(mostThreeTokens, takeTokenActions)
    takeTokenActions = np.concatenate(
        (takeTokenActions, takeTokenActions, takeTokenActions, mostAvalableThreeTokens)
    )
    if len(takeTokenActions) > 0:
        action = takeTokenActions[np.random.randint(len(takeTokenActions))]
        return action, per

    action = validActions[np.random.randint(len(validActions))]
    return action, per
