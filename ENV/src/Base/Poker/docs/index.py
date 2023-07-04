import numpy as np
from numba import njit

# Poker Cash out

ALL_CARD_STR = np.array(
    [
        "2H",
        "2D",
        "2C",
        "2S",
        "3H",
        "3D",
        "3C",
        "3S",
        "4H",
        "4D",
        "4C",
        "4S",
        "5H",
        "5D",
        "5C",
        "5S",
        "6H",
        "6D",
        "6C",
        "6S",
        "7H",
        "7D",
        "7C",
        "7S",
        "8H",
        "8D",
        "8C",
        "8S",
        "9H",
        "9D",
        "9C",
        "9S",
        "TH",
        "TD",
        "TC",
        "TS",
        "JH",
        "JD",
        "JC",
        "JS",
        "QH",
        "QD",
        "QC",
        "QS",
        "KH",
        "KD",
        "KC",
        "KS",
        "AH",
        "AD",
        "AC",
        "AS",
    ]
)

ACTIONS_MEAN = np.array(["call", "check", "fold", "bet/raise", "all in", "stop_bet"])


NUMBER_PLAYER = 9
NUMBER_CARD = 52
NUMBER_BURN = 3
NUMBER_CARD_OPEN = 5
SMALL_CHIP = 1
BIG_CHIP = 2
ATTRIBUTE_PLAYER = 3
NUMBER_STATUS_GAME = 5  # (preflop, flop,, turn, river, showdown)
NUMBER_PHASE = 2


"""
env_state:
0-52: lá bài trong bộ bài, các lá bài còn nằm ở đầu, số là đã chia là số lá -1 nằm ở cuối
52-55: các lá bài burn, trống là -1
55-60: các lá open
60-69: tổng chip còn lại của các người chơi
69-78: chip đã bỏ ra trong lượt
78-87: tổng chip đã bỏ ra

87-96: trạng thái người chơi còn chơi hay k
96-105: lá bài thứ nhất của các người chơi
105-114: lá bài thứ hai của các người chơi
114-end:[button dealer, temp_button, status game, phase,
        id_action, cash to call_old, cash to call_new, sum pot, ván chơi thứ bao nhiêu

        ]
"""

INDEX = 0
# thẻ trên bàn
ENV_ALL_CARD_ON_BOARD = INDEX
INDEX += NUMBER_CARD

# card open
ENV_CARD_OPEN = INDEX
INDEX += NUMBER_CARD_OPEN

# chip of player
ENV_ALL_PLAYER_CHIP = INDEX
INDEX += NUMBER_PLAYER

# chip người chơi đã bỏ ra để theo
ENV_ALL_PLAYER_CHIP_GIVE = INDEX
INDEX += NUMBER_PLAYER

# tổng chip người chơi đã bỏ ra trong ván
ENV_ALL_PLAYER_CHIP_IN_POT = INDEX
INDEX += NUMBER_PLAYER

# player foled or still play
ENV_ALL_PLAYER_STATUS = INDEX
INDEX += NUMBER_PLAYER

# player first_card
ENV_ALL_FIRST_CARD = INDEX
INDEX += NUMBER_PLAYER

# player second_card
ENV_ALL_SECOND_CARD = INDEX
INDEX += NUMBER_PLAYER

# player first_card_showdown
ENV_ALL_FIRST_CARD_SHOWDOWN = INDEX
INDEX += NUMBER_PLAYER

# player second_card_showdown
ENV_ALL_SECOND_CARD_SHOWDOWN = INDEX
INDEX += NUMBER_PLAYER

# other in4
ENV_BUTTON_PLAYER = INDEX
INDEX += 1
ENV_TEMP_BUTTON = INDEX
INDEX += 1
ENV_STATUS_GAME = INDEX
INDEX += 1
ENV_PHASE = INDEX
INDEX += 1
ENV_ID_ACTION = INDEX
INDEX += 1
ENV_CASH_TO_CALL_OLD = INDEX
INDEX += 1
ENV_CASH_TO_CALL_NEW = INDEX
INDEX += 1
ENV_POT_VALUE = INDEX
INDEX += 1
ENV_NUMBER_GAME_PLAYED = INDEX
INDEX += 1
ENV_CHECK_END = INDEX
INDEX += 1


ENV_LENGTH = INDEX


"""
player_state:
0-52: lá bài trên bàn và của mình (giá trị 1 là của mình và đã mở, 0 là chưa mở và của người khác mình ko biết)
52-63: số chip còn lại của mỗi người
63:72: trạng thái người chơi (0 là đã bỏ game, 1 là còn chơi)
72:83: button (theo thứ tự từ phải qua trái của mình, tại đâu thì giá trị đó = 1)
83: cash to call
84: pot value
85: phase
86: status game
87: số ván đã chơi
"""

P_INDEX = 0
P_ALL_CARD = P_INDEX
P_INDEX += NUMBER_CARD * NUMBER_PLAYER

# chip of player
P_ALL_PLAYER_CHIP = P_INDEX
P_INDEX += NUMBER_PLAYER

# chip người chơi đã bỏ ra để theo
P_ALL_PLAYER_CHIP_GIVE = P_INDEX
P_INDEX += NUMBER_PLAYER

# status of player
P_ALL_PLAYER_STATUS = P_INDEX
P_INDEX += NUMBER_PLAYER

# button
P_BUTTON_DEALER = P_INDEX
P_INDEX += NUMBER_PLAYER

# other in4
P_CASH_TO_CALL = P_INDEX
P_INDEX += 1

P_CASH_TO_BET = P_INDEX
P_INDEX += 1

P_POT_VALUE = P_INDEX
P_INDEX += 1

P_PHASE = P_INDEX
P_INDEX += 1  # (vì chỉ có 2 phase nên 1 vị trí thể hiện đc 2 phase)

P_STATUS_GAME = P_INDEX
P_INDEX += NUMBER_STATUS_GAME

P_NUMBER_GAME_PLAY = P_INDEX
P_INDEX += 1

P_CHECK_END = P_INDEX
P_INDEX += 1

PLAYER_STATE_LENGTH = P_INDEX


@njit()
def combinations_using_numba(pool, r):
    n = len(pool)
    indices = list(range(r))
    empty = not (n and (0 < r <= n))
    a = []
    if not empty:
        result = [pool[i] for i in indices]
        #  yield result
        a.append(result)
    while not empty:
        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1
        if i < 0:
            empty = True
        else:
            indices[i] += 1
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1
            result = [pool[i] for i in indices]
            #  yield result
            a.append(result)
    return a


@njit()
def evaluate_num_numba(hand, id_player):
    """
    input:  - hand: các lá bài của người chơi
            - id_player: index của người chơi trong danh sách người chơi
    out_put: list gồm:  + điểm của hand bài
                        + rank kicker
                        + danh sách bài trong hand
                        +id người chơi
    """
    hand = hand.astype(np.int64)
    #  ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    rank_card = hand // 4
    suit_card = hand % 4
    all_index_card = [0, 1, 2, 3, 4, 5, 6]
    all_score = []
    all_hands = np.array(list(combinations_using_numba(all_index_card, 5)))
    for id in range(len(all_hands)):
        sm_hand = all_hands[id]
        rank_sm_hand = rank_card[sm_hand]
        suit_sm_hand = suit_card[sm_hand]
        score = np.array([0, 0, 0, 0, 0])
        rankss = [-1, -1, -1, -1, -1]
        arr_rank_score = [[0, -1], [0, -1], [0, -1], [0, -1], [0, -1]]
        for i in range(5):
            rank = rank_sm_hand[i]
            if rank not in rankss:
                rankss[i] = rank
                arr_rank_score[i] = [len(rank_sm_hand[rank_sm_hand == rank]), rank]
        arr_rank_score = np.array(sorted(arr_rank_score, reverse=True))
        rankss = list(arr_rank_score[:, 1])
        score = arr_rank_score[:, 0]
        if (
            len(score[score > 0]) == 5
        ):  #  if there are 5 different ranks it could be a straight or a flush (or both)
            if rankss[0:2] == [12, 3]:
                rankss = [3, 2, 1, 0, -1]  #  adjust if 5 high straight
            all_type_hand = [
                [np.array([1, 0, 0, 0, 0]), np.array([3, 1, 2, 0, 0])],
                [np.array([3, 1, 3, 0, 0]), np.array([5, 0, 0, 0, 0])],
            ]
            score = all_type_hand[int(len(np.unique(suit_sm_hand)) == 1)][
                int(rankss[0] - rankss[4] == 4)
            ]  #  high card, straight, flush, straight flush
        score = list(score)
        sm_hand = list(hand[sm_hand])
        all_score.append([score, rankss, sm_hand, [id_player, -1, -1, -1, -1]])
    return max(all_score)


@njit()
def holdem(board, hands):
    """
    board: các thẻ bài chung
    hands: list các bộ bài của từng người chơi
    """
    scores = []
    for i in range(len(hands)):
        result = evaluate_num_numba(np.append(board, hands[i]), i)
        scores.append(result)

    all_best_hand_player = np.zeros((NUMBER_PLAYER, 5))

    for i in range(len(scores)):
        all_best_hand_player[i] = np.array(scores[i][2])
    scores = sorted(scores, reverse=True)
    top = np.zeros((9, 9))
    topth = 0
    top[topth][scores[0][3][0]] = 1
    for i in range(len(scores) - 1):
        if scores[i][0] == scores[i + 1][0]:
            if scores[i][1] == scores[i + 1][1]:
                top[topth][scores[i + 1][3][0]] = 1
            else:
                topth += 1
                top[topth][scores[i + 1][3][0]] = 1
        else:
            topth += 1
            top[topth][scores[i + 1][3][0]] = 1
    ranks = np.full(9, -1)
    for i in range(len(top)):
        ranks[np.where(top[i] == 1)[0]] = i
    all_best_hand_player_str = []
    for item in all_best_hand_player:
        all_best_hand_player_str.append(ALL_CARD_STR[item.astype(np.int64)])
    return ranks, all_best_hand_player


@njit()
def showdown(env_state):
    chip_give = env_state[ENV_ALL_PLAYER_CHIP_IN_POT:ENV_ALL_PLAYER_STATUS]
    status = env_state[ENV_ALL_PLAYER_STATUS:ENV_ALL_FIRST_CARD]
    player_chip_receive = np.zeros(9)
    all_pot_val = np.sum(chip_give)
    if np.count_nonzero(status) == 1:
        player_chip_receive[np.argmax(status)] = all_pot_val
        # update chip sau ván chơi của các người chơi
        env_state[ENV_ALL_PLAYER_CHIP:ENV_ALL_PLAYER_CHIP_GIVE] += player_chip_receive
    else:
        board = env_state[ENV_CARD_OPEN:ENV_ALL_PLAYER_CHIP]
        hands = [
            np.array(
                [env_state[ENV_ALL_FIRST_CARD + i], env_state[ENV_ALL_SECOND_CARD + i]]
            )
            for i in range(9)
        ]

        # tính rank bài của các người chơi
        ranks, all_player_hand = holdem(board, hands)

        # split_pot:
        max_chip_can_get = np.zeros(9)
        while all_pot_val > 0:
            # tạo side pot
            side_pot = np.min(chip_give[chip_give > 0])
            player_join = np.where(chip_give >= side_pot)[0]
            player_get_pot = np.where((chip_give >= side_pot) & (status == 1))
            side_pot_val = len(player_join) * side_pot
            max_chip_can_get[player_get_pot] += side_pot_val
            # lấy rank
            rank_in_side_pot = np.full(9, 10)
            rank_in_side_pot[player_get_pot] = ranks[player_get_pot]

            best_rank = np.min(rank_in_side_pot)
            player_win_pot = np.where(rank_in_side_pot == best_rank)[0]
            delta_chip = side_pot_val / len(player_win_pot)
            player_chip_receive[player_win_pot] += delta_chip
            # cập nhật chip còn lại
            all_pot_val -= side_pot_val
            temp = chip_give - side_pot
            chip_give = (temp) * (temp > 0)
        # update chip sau ván chơi của các người
        for i in range(len(player_chip_receive)):
            player_chip_receive[i] = int(player_chip_receive[i])
        #  player_chip_receive = player_chip_receive // 1
        env_state[ENV_ALL_PLAYER_CHIP:ENV_ALL_PLAYER_CHIP_GIVE] += player_chip_receive
        # show card on hand, update vào env_state xem ai phải show bài, lá nào k cần show thì gán thành -1
        rank_temp = np.full(9, 10)
        chip_receive_max = np.zeros(9)
        # lấy rank hand người mở đầu tiên và lượng chip tối đa họ có thể ăn
        id_temp_button = int(env_state[ENV_TEMP_BUTTON])
        rank_temp[id_temp_button] = ranks[id_temp_button]
        chip_receive_max[id_temp_button] = max_chip_can_get[id_temp_button]
        if (
            env_state[ENV_ALL_FIRST_CARD + id_temp_button]
            in all_player_hand[id_temp_button]
        ):
            env_state[ENV_ALL_FIRST_CARD_SHOWDOWN + id_temp_button] = env_state[
                ENV_ALL_FIRST_CARD + id_temp_button
            ]
        if (
            env_state[ENV_ALL_SECOND_CARD + id_temp_button]
            in all_player_hand[id_temp_button]
        ):
            env_state[ENV_ALL_SECOND_CARD_SHOWDOWN + id_temp_button] = env_state[
                ENV_ALL_SECOND_CARD + id_temp_button
            ]

        for i in range(1, 9):
            id = int(env_state[ENV_TEMP_BUTTON] + i) % NUMBER_PLAYER
            # nếu người chơi đã bỏ, thì bài của người chơi là bài trên bàn
            if status[id] == 0:
                continue
            # nếu người chơi bài cao hơn hoặc bằng (share pot) hoặc có thể ăn side pot thì phải show card
            else:
                if (
                    np.min(rank_temp) >= ranks[id]
                    or np.max(chip_receive_max[np.where(rank_temp < ranks[id])[0]])
                    < max_chip_can_get[id]
                ):
                    if env_state[ENV_ALL_FIRST_CARD + id] in all_player_hand[id]:
                        env_state[ENV_ALL_FIRST_CARD_SHOWDOWN + id] = env_state[
                            ENV_ALL_FIRST_CARD + id
                        ]
                    if env_state[ENV_ALL_SECOND_CARD + id] in all_player_hand[id]:
                        env_state[ENV_ALL_SECOND_CARD_SHOWDOWN + id] = env_state[
                            ENV_ALL_SECOND_CARD + id
                        ]
                    rank_temp[id] = ranks[id]
                    chip_receive_max[id] = max_chip_can_get[id]

    return env_state
