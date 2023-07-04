import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


def DataAgent():
    return np.array([])


from src.Base.Century.docs.index import ALL_CARD_IN4


@njit()
def valueOf(cards, action, resources, pointCards):
    card = cards[int(action - 12)]
    returnTokens = card[0:4]
    rewardTokens = card[4:8]
    nUpgradeTokens = card[8]
    pTokens = rewardTokens + resources - returnTokens
    score = 0
    for pointCard in pointCards:
        balance = pTokens - pointCard
        if np.all(balance >= 0):
            score += 1000000
        if np.sum(pTokens) <= 10:
            balance[balance < 0] = 0
            score += np.sum(pTokens * np.array([1, 1, 2, 3]))
        elif nUpgradeTokens > 0:
            score += nUpgradeTokens
        else:
            score += np.sum(pTokens * np.array([1, 1, 2, 3])) - 20
    return score


@njit()
def Test(state, per):
    validActions = env.getValidActions(state)
    validActions = np.where(validActions == 1)[0]

    if 1 in validActions:
        return 1, per

    cards = state[194:219].reshape(5, -1)
    purchasePointCardActions = validActions[(validActions >= 7) & (validActions < 12)]
    if len(purchasePointCardActions) > 0:
        valueOfCards = np.zeros_like(purchasePointCardActions)
        for i in range(len(purchasePointCardActions)):
            valueOfCards[i] = cards[int(purchasePointCardActions[i] - 7)][0]
        action = purchasePointCardActions[np.argmax(valueOfCards)]
        return action, per

    if (0 in validActions) and (np.sum(state[51:96]) >= 0.5 * np.sum(state[6:51])):
        return 0, per

    resources = state[2:6]
    actionCards = ALL_CARD_IN4
    performActionCardsActions = validActions[(validActions >= 12) & (validActions < 57)]
    if len(performActionCardsActions) > 0:
        valueOfActionCards = np.zeros_like(performActionCardsActions)
        for i in range(len(performActionCardsActions)):
            valueOfActionCards[i] = valueOf(
                actionCards, performActionCardsActions[i], resources, cards[:, 0:4]
            )
        action = performActionCardsActions[np.argmax(valueOfActionCards)]
        return action, per

    returnTokenActions = validActions[(validActions >= 57) & (validActions < 61)]
    if len(returnTokenActions) > 0:
        return returnTokenActions[0], per

    upgradeTokenActions = validActions[(validActions >= 62) & (validActions < 65)]
    if len(upgradeTokenActions) > 0:
        return upgradeTokenActions[-1], per

    if 61 in validActions:
        return 61, per

    if 0 in validActions:
        return 0, per

    action = validActions[np.random.randint(len(validActions))]
    return action, per
