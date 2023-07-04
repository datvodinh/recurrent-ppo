import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


def DataAgent():
    return np.array([0.0])


from src.Base.Century.docs.index import ALL_CARD_IN4


@njit()
def getValueActionsCard(state, infor_action_cards_on_board):
    id = 0
    ls_return = []
    for infor_action_card in infor_action_cards_on_board:
        arr_cal = np.array([0.1, 0.2, 0.4, 0.8])
        token_cost = infor_action_card[0:4]
        token_get = infor_action_card[4:8]
        sum_cost_point = sum(arr_cal * token_cost) * 0.4
        sum_get_point = sum(arr_cal * token_get) * 0.4
        times_upgrade = infor_action_card[-1]
        value_ = sum_get_point - sum_cost_point - id * 0.3 + times_upgrade * 0.1
        ls_return.append(value_)
        id += 1
    arr_return = np.array(ls_return)
    return arr_return


@njit()
def getArrValueActionsCard(state, infor_action_cards_on_board):
    id = 0
    ls_return = []
    for infor_action_card in infor_action_cards_on_board:
        arr_cal = np.array([0.1, 0.2, 0.4, 0.8])
        token_cost = infor_action_card[0:4]
        token_get = infor_action_card[4:8]
        sum_cost_point = sum(arr_cal * token_cost) * 0.4
        sum_get_point = sum(arr_cal * token_get) * 0.4
        times_upgrade = infor_action_card[-1]
        value_ = sum_get_point - sum_cost_point - id * 0.3 + times_upgrade * 0.1
        ls_return.append(value_)
        id += 1

    arr_return = np.array(ls_return)
    #  print('arr_return',arr_return)
    return arr_return


@njit()
def chooseGetFreeTokenActionCardOnBoard(state, infor_action_cards_on_board):
    id = 1
    for action_card in infor_action_cards_on_board:
        cost_ = sum(action_card[0:4])
        get_ = sum(action_card[4:8])
        if cost_ == 0 and get_ > 0 and id < 3:
            return id
        id += 0
    return -1


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
            score += 10000000000000
        if np.sum(pTokens) <= 10:
            balance[balance < 0] = 0
            score += np.sum(pTokens * np.array([1, 1, 2, 3]))
        elif nUpgradeTokens > 0:
            score += nUpgradeTokens
        else:
            score += np.sum(pTokens * np.array([1, 1, 2, 3])) - 20
    return score


@njit()
def choosePointCard(state, id_card_can_buy, infor_point_cards_on_board):
    point_ls = []
    for id_card_can_buy in id_card_can_buy:
        id_card_can_buy = id_card_can_buy - 7
        point = infor_point_cards_on_board[id_card_can_buy][-1]
        point_ls.append(point)
    point_arr = np.array(point_ls)
    return np.argmax(point_arr)


@njit()
def Test(state, per):
    actions = env.getValidActions(state)
    actions = np.where(actions == 1)[0]
    player_token = state[2:6]
    infor_point_cards_on_board = state[194:219].reshape(5, 5)
    infor_action_cards_on_board = state[120:174].reshape(6, 9)

    actions_card_rank = getValueActionsCard(state, infor_action_cards_on_board)
    highest_action_card_value = np.argmax(actions_card_rank)
    buy_actions_card = actions[(actions >= 7) & (actions < 12)]
    # ---------------Working on
    if 1 in actions:
        return 1, per
    if 0 in actions:
        total_actions_card_on_hand = len(np.where(state[6:51] == 1)[0])
        total_actions_card_had_use = len(np.where(state[51:96] == 1)[0])
        if (total_actions_card_had_use) >= total_actions_card_on_hand * 0.5:
            return 0, per
    if len(buy_actions_card):
        arr_cham_diem_action_card = getArrValueActionsCard(
            state, infor_action_cards_on_board
        )
        id = np.argmax(arr_cham_diem_action_card) - 1
        return buy_actions_card[id], per

    idGetFreeTokenAction = chooseGetFreeTokenActionCardOnBoard(
        state, infor_action_cards_on_board
    )
    if idGetFreeTokenAction != -1:
        return idGetFreeTokenAction, per

    action_buy_point_card = actions[(actions >= 7) & (actions < 12)]
    if len(action_buy_point_card) > 0:
        #  id_buy_point_card = choosePointCard(state,action_buy_point_card,infor_point_cards_on_board)
        return action_buy_point_card[0], per

    removeToken = actions[(actions >= 57) & (actions < 61)]
    if len(removeToken) > 0:
        my_token = state[2:6]
        return np.argmax(my_token) + 57, per
        #

    resources = state[2:6]
    actionCards = ALL_CARD_IN4
    performActionCardsActions = actions[(actions >= 12) & (actions < 57)]
    if len(performActionCardsActions) > 0:
        valueOfActionCards = np.zeros_like(performActionCardsActions)
        for i in range(len(performActionCardsActions)):
            valueOfActionCards[i] = valueOf(
                actionCards,
                performActionCardsActions[i],
                resources,
                infor_point_cards_on_board[:, 0:4],
            )
        action = performActionCardsActions[np.argmax(valueOfActionCards)]
        return action, per
    #  if sum(state[51:96]) < 1:
    #    return np.random.choice(actions),per

    upgradeToken = actions[(actions >= 62) & (actions < 65)]
    if len(upgradeToken) > 0:
        return upgradeToken[-1], per

    return np.random.choice(actions), per
