import numba as nb
import numpy as np
from numba import njit

from src.Base.Poker.docs.index import *


@njit()
def getActionSize():
    return 6


@njit()
def getAgentSize():
    return 9


@njit()
def getStateSize():
    return 515


@njit()
def initEnv():
    env_state = np.zeros(ENV_LENGTH)
    env_state[ENV_CARD_OPEN:ENV_ALL_PLAYER_CHIP] = -1
    env_state[ENV_ALL_PLAYER_CHIP:ENV_ALL_PLAYER_CHIP_GIVE] = np.full(
        NUMBER_PLAYER, 100 * BIG_CHIP
    )
    env_state[ENV_ALL_PLAYER_STATUS:ENV_ALL_FIRST_CARD] = np.ones(NUMBER_PLAYER)
    env_state[ENV_ALL_FIRST_CARD:ENV_BUTTON_PLAYER] = np.full(4 * NUMBER_PLAYER, -1)
    env_state[ENV_BUTTON_PLAYER] = -1  # người chơi ở vị trí đầu tiên giữ button
    return env_state


@njit()
def reset_round(old_env_state):
    env_state = np.zeros(ENV_LENGTH)
    #  print(np.sum(old_env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE]), 'tổng chip')

    if (
        np.sum(
            old_env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP + NUMBER_PLAYER] > 0
        )
        == 1
    ):
        env_state[ENV_ALL_PLAYER_CHIP:ENV_ALL_PLAYER_CHIP_GIVE] = old_env_state[
            ENV_ALL_PLAYER_CHIP:ENV_ALL_PLAYER_CHIP_GIVE
        ]
        env_state[ENV_ALL_PLAYER_STATUS:ENV_ALL_FIRST_CARD] = (
            env_state[ENV_ALL_PLAYER_CHIP:ENV_ALL_PLAYER_CHIP_GIVE] > 0
        ) * 1
        env_state[ENV_NUMBER_GAME_PLAYED] = old_env_state[ENV_NUMBER_GAME_PLAYED]
        #  print('ENDDDDDDDD GAME')
        #  print(np.sum(old_env_state[ENV_ALL_PLAYER_CHIP : ENV_ALL_PLAYER_CHIP_GIVE]), 'tổng chip')
        return env_state

    # tính toán chip còn lại của người chơi, từ đó tính ra trạng thái của người chơi ở game mới
    env_state[ENV_ALL_PLAYER_CHIP:ENV_ALL_PLAYER_CHIP_GIVE] = old_env_state[
        ENV_ALL_PLAYER_CHIP:ENV_ALL_PLAYER_CHIP_GIVE
    ]
    env_state[ENV_ALL_PLAYER_STATUS:ENV_ALL_FIRST_CARD] = (
        env_state[ENV_ALL_PLAYER_CHIP:ENV_ALL_PLAYER_CHIP_GIVE] > 0
    ) * 1

    # thiết lập button dealer và temp button, trừ chip của small và big để cộng vào sum_pot_value, thiết lập id_action, status_game, phase, số ván đã chơi
    env_state[ENV_BUTTON_PLAYER] = (
        old_env_state[ENV_BUTTON_PLAYER] + 1
    ) % NUMBER_PLAYER
    while env_state[int(ENV_ALL_PLAYER_CHIP + env_state[ENV_BUTTON_PLAYER])] == 0:
        env_state[ENV_BUTTON_PLAYER] = (
            env_state[ENV_BUTTON_PLAYER] + 1
        ) % NUMBER_PLAYER
    sum_pot = 0
    # trừ chip của small_player
    sm_player = (env_state[ENV_BUTTON_PLAYER] + 1) % NUMBER_PLAYER
    while env_state[int(ENV_ALL_PLAYER_CHIP + sm_player)] == 0:
        sm_player = (sm_player + 1) % NUMBER_PLAYER
    sum_pot += SMALL_CHIP
    env_state[int(ENV_ALL_PLAYER_CHIP + sm_player)] -= SMALL_CHIP
    env_state[int(ENV_ALL_PLAYER_CHIP_GIVE + sm_player)] += SMALL_CHIP
    env_state[int(ENV_ALL_PLAYER_CHIP_IN_POT + sm_player)] += SMALL_CHIP

    # trừ chip của big_player
    big_player = (sm_player + 1) % NUMBER_PLAYER
    while env_state[int(ENV_ALL_PLAYER_CHIP + big_player)] == 0:
        big_player = (big_player + 1) % NUMBER_PLAYER
    sum_pot += min(BIG_CHIP, env_state[int(ENV_ALL_PLAYER_CHIP + big_player)])
    env_state[int(ENV_ALL_PLAYER_CHIP + big_player)] -= min(
        2, env_state[int(ENV_ALL_PLAYER_CHIP + big_player)]
    )
    env_state[int(ENV_ALL_PLAYER_CHIP_GIVE + big_player)] += min(
        2, env_state[int(ENV_ALL_PLAYER_CHIP + big_player)]
    )
    env_state[int(ENV_ALL_PLAYER_CHIP_IN_POT + big_player)] += min(
        2, env_state[int(ENV_ALL_PLAYER_CHIP + big_player)]
    )

    temp_button = (big_player + 1) % NUMBER_PLAYER
    while env_state[int(ENV_ALL_PLAYER_CHIP + temp_button)] == 0:
        temp_button = (temp_button + 1) % NUMBER_PLAYER
    env_state[ENV_TEMP_BUTTON] = temp_button

    env_state[ENV_CASH_TO_CALL_NEW] = BIG_CHIP
    env_state[ENV_CASH_TO_CALL_OLD] = BIG_CHIP
    env_state[ENV_POT_VALUE] = sum_pot

    # người chơi action kế tiếp là người thứ 1 trở đi từ button dealer mà còn chip
    env_state[ENV_ID_ACTION] = temp_button

    while env_state[int(ENV_ALL_PLAYER_CHIP + env_state[ENV_ID_ACTION])] == 0:
        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER

    env_state[ENV_NUMBER_GAME_PLAYED] = old_env_state[ENV_NUMBER_GAME_PLAYED] + 1
    #  print('ván chơi thứ ', env_state[ENV_NUMBER_GAME_PLAYED])
    # reset 2 lá bài của người chơi và bài showdown, sau đó chia bài
    env_state[ENV_ALL_FIRST_CARD:ENV_BUTTON_PLAYER] = np.full(4 * NUMBER_PLAYER, -1)
    all_card_num = np.arange(52)
    np.random.shuffle(all_card_num)
    env_state[ENV_CARD_OPEN:ENV_ALL_PLAYER_CHIP] = all_card_num[:NUMBER_CARD_OPEN]
    env_state[ENV_ALL_FIRST_CARD:ENV_ALL_FIRST_CARD_SHOWDOWN] = all_card_num[
        NUMBER_CARD_OPEN : NUMBER_CARD_OPEN + 2 * NUMBER_PLAYER
    ]
    env_state[ENV_ALL_CARD_ON_BOARD:ENV_CARD_OPEN] = np.append(
        all_card_num[NUMBER_CARD_OPEN + 2 * NUMBER_PLAYER :],
        np.full(NUMBER_CARD_OPEN + 2 * NUMBER_PLAYER, -1),
    )
    return env_state


@njit()
def getAgentState(env_state):
    player_state = np.zeros(PLAYER_STATE_LENGTH)
    id_action = int(env_state[ENV_ID_ACTION])
    # cập nhật chip còn lại
    all_player_chip = env_state[ENV_ALL_PLAYER_CHIP:ENV_ALL_PLAYER_CHIP_GIVE]
    player_state[P_ALL_PLAYER_CHIP:P_ALL_PLAYER_CHIP_GIVE] = np.concatenate(
        (all_player_chip[id_action:], all_player_chip[:id_action])
    )
    # cập nhật status
    all_player_status = env_state[ENV_ALL_PLAYER_STATUS:ENV_ALL_FIRST_CARD]
    player_state[P_ALL_PLAYER_STATUS:P_BUTTON_DEALER] = np.concatenate(
        (all_player_status[id_action:], all_player_status[:id_action])
    )
    # cập nhật game kết thúc chưa
    player_state[P_CHECK_END] = env_state[ENV_CHECK_END]
    player_state[P_NUMBER_GAME_PLAY] = env_state[ENV_NUMBER_GAME_PLAYED]

    if env_state[ENV_CHECK_END] == 0:
        # Cập nhật chip đã bỏ ra
        all_player_chip_give = env_state[
            ENV_ALL_PLAYER_CHIP_GIVE:ENV_ALL_PLAYER_CHIP_IN_POT
        ]
        player_state[P_ALL_PLAYER_CHIP_GIVE:P_ALL_PLAYER_STATUS] = np.concatenate(
            (all_player_chip_give[id_action:], all_player_chip_give[:id_action])
        )
        # cập nhật button dealer
        player_state[
            P_BUTTON_DEALER
            + int(env_state[ENV_BUTTON_PLAYER] - id_action) % NUMBER_PLAYER
        ] = 1
        # cập nhật chip to call
        player_state[P_CASH_TO_CALL] = max(
            0,
            env_state[ENV_CASH_TO_CALL_OLD]
            - env_state[ENV_ALL_PLAYER_CHIP_GIVE + id_action],
        )
        # cập nhật chip to bet
        player_state[P_CASH_TO_BET] = env_state[ENV_CASH_TO_CALL_OLD]
        # cập nhật pot value, phase, status game
        player_state[P_POT_VALUE] = env_state[ENV_POT_VALUE]
        player_state[P_PHASE] = env_state[ENV_PHASE]
        player_state[int(P_STATUS_GAME + max(env_state[ENV_STATUS_GAME] - 2, 0))] = 1

        # cập nhật card
        player_card = np.array(
            [
                env_state[ENV_ALL_FIRST_CARD + id_action],
                env_state[ENV_ALL_SECOND_CARD + id_action],
            ]
        ).astype(np.int64)
        if env_state[ENV_STATUS_GAME] == 0:
            player_state[P_ALL_CARD : P_ALL_CARD + NUMBER_CARD][player_card] = 1
        elif env_state[ENV_STATUS_GAME] != 6:
            # nếu ko phải showdown và pre flop
            card_open = env_state[
                ENV_CARD_OPEN : int(ENV_CARD_OPEN + env_state[ENV_STATUS_GAME])
            ].astype(np.int64)
            for id in range(1, NUMBER_PLAYER):
                status_id = player_state[P_ALL_PLAYER_STATUS:P_BUTTON_DEALER][id]
                if status_id == 1:
                    player_state[
                        P_ALL_CARD
                        + NUMBER_CARD * id : P_ALL_CARD
                        + NUMBER_CARD * (id + 1)
                    ][card_open] = 1

            player_card = np.append(player_card, card_open)
            player_state[P_ALL_CARD : P_ALL_CARD + NUMBER_CARD][player_card] = 1
        else:
            # duyệt từng người, ông nào tham gia showdown thì cho xem bài, ông nào bài xấu ko mở thì đã bị gán thành -1 ở step, nên lọc bài lớn hơn -1 là đc
            card_open = env_state[ENV_CARD_OPEN:ENV_ALL_PLAYER_CHIP].astype(np.int64)
            card_open = card_open[card_open > -1]  # thêm mới
            for id in range(NUMBER_PLAYER):
                status_id = player_state[P_ALL_PLAYER_STATUS:P_BUTTON_DEALER][id]
                if status_id == 1:
                    id_env = (id_action + id) % NUMBER_PLAYER
                    player_i_card = np.array(
                        [
                            env_state[ENV_ALL_FIRST_CARD_SHOWDOWN + id_env],
                            env_state[ENV_ALL_SECOND_CARD_SHOWDOWN + id_env],
                        ]
                    ).astype(np.int64)
                    player_i_card = np.append(player_i_card, card_open)
                    if id_env == id_action:
                        player_i_card = np.append(player_i_card, player_card)
                    player_i_card = player_i_card[player_i_card > -1]
                    player_state[
                        P_ALL_CARD
                        + NUMBER_CARD * id : P_ALL_CARD
                        + NUMBER_CARD * (id + 1)
                    ][player_i_card] = 1

    return player_state


@njit()
def getValidActions(player_state):
    list_action = np.zeros(6)
    if player_state[P_PHASE] == 0:
        # nếu cash to call == 0 thì check, bet, allin
        if player_state[P_CASH_TO_CALL] == 0:
            list_action[1:5] = 1
            list_action[2] = 0
        # nếu cash to call >= 0
        else:
            # nếu cash_to_call > cash (fold, allin)
            if player_state[P_ALL_PLAYER_CHIP] <= player_state[P_CASH_TO_CALL]:
                # fold, allin
                list_action[2] = 1
                list_action[4] = 1
            elif (
                player_state[P_ALL_PLAYER_CHIP] > player_state[P_CASH_TO_CALL]
                and player_state[P_ALL_PLAYER_CHIP] - player_state[P_CASH_TO_CALL]
                < player_state[P_CASH_TO_BET]
            ):
                # fold, call, allin
                list_action[0] = 1
                list_action[2] = 1
                list_action[4] = 1
            elif (
                player_state[P_ALL_PLAYER_CHIP] - player_state[P_CASH_TO_CALL]
                >= player_state[P_CASH_TO_BET]
            ):
                # call, fold, bet, allin
                list_action[:5] = 1
                list_action[1] = 0
                if (
                    np.count_nonzero(
                        player_state[P_ALL_PLAYER_CHIP:P_ALL_PLAYER_CHIP_GIVE]
                        * player_state[
                            P_ALL_PLAYER_STATUS : P_ALL_PLAYER_STATUS + NUMBER_PLAYER
                        ]
                    )
                    < 2
                ):
                    list_action[3:5] = 0

            else:
                raise Exception("xét thieu trường hợp")
    # nếu đang bet dở thì bet, allin, dừng bet
    elif player_state[P_PHASE] == 1:
        if player_state[P_ALL_PLAYER_CHIP] >= player_state[P_CASH_TO_BET]:
            list_action[3:] = 1
        else:
            list_action[4:] = 1
    return list_action


@njit()
def stepEnv(env_state, action):
    phase_env = int(env_state[ENV_PHASE])
    id_action = int(env_state[ENV_ID_ACTION])
    if phase_env == 0:
        if action == 0:  # người chơi call
            # trừ chip người chơi và tăng giá trị pot, bổ sung giá trị số tiền đã bỏ ra
            chip_to_call = (
                env_state[ENV_CASH_TO_CALL_OLD]
                - env_state[ENV_ALL_PLAYER_CHIP_GIVE + id_action]
            )
            env_state[ENV_ALL_PLAYER_CHIP_GIVE + id_action] += chip_to_call
            env_state[ENV_ALL_PLAYER_CHIP_IN_POT + id_action] += chip_to_call
            env_state[ENV_ALL_PLAYER_CHIP + id_action] -= chip_to_call
            env_state[ENV_POT_VALUE] += chip_to_call
        elif action == 1:  # người chơi check
            pass
        elif action == 2:  # người chơi fold
            # nếu người chơi fold, chuyển trạng thái của người đó.
            env_state[ENV_ALL_PLAYER_CHIP_GIVE + id_action] = 0
            env_state[ENV_ALL_PLAYER_STATUS + id_action] = 0
        elif action == 3:  # người chơi bet/raise
            # nếu người chơi bet, trừ tiền người chơi, tăng tiền pot, tăng cash to call_new
            # bổ sung giá trị số tiền đã bỏ ra
            chip_to_bet_raise = 0
            if env_state[ENV_CASH_TO_CALL_OLD] == 0:  # nếu chưa ai bet thì mình bet
                env_state[ENV_CASH_TO_CALL_OLD] = SMALL_CHIP
                chip_to_bet_raise = SMALL_CHIP
            else:  # nếu đã có người bet thì mình raise
                chip_to_bet_raise = (
                    2 * env_state[ENV_CASH_TO_CALL_OLD]
                    - env_state[ENV_ALL_PLAYER_CHIP_GIVE + id_action]
                )

            env_state[ENV_ALL_PLAYER_CHIP + id_action] -= chip_to_bet_raise
            env_state[ENV_ALL_PLAYER_CHIP_GIVE + id_action] += chip_to_bet_raise
            env_state[ENV_ALL_PLAYER_CHIP_IN_POT + id_action] += chip_to_bet_raise
            env_state[ENV_POT_VALUE] += chip_to_bet_raise
            env_state[ENV_CASH_TO_CALL_NEW] += env_state[ENV_CASH_TO_CALL_OLD]
            # nếu bet thì sang phase xem có bet tiếp k

            env_state[ENV_TEMP_BUTTON] = id_action
            #  print('gán lại temp button', id_action, env_state[ENV_TEMP_BUTTON])
            env_state[ENV_PHASE] = 1
        elif action == 4:
            # nếu người chơi all_in, bổ sung giá trị chip người chơi bỏ ra và tăng pot
            chip_to_allin = env_state[ENV_ALL_PLAYER_CHIP + id_action]
            env_state[ENV_ALL_PLAYER_CHIP_IN_POT + id_action] += chip_to_allin
            env_state[ENV_POT_VALUE] += chip_to_allin
            #  print('check', chip_to_allin, env_state[ENV_CASH_TO_CALL_OLD])
            # điều chỉnh cash to call nếu cần, điều chỉnh button nếu người chơi bet nhiều chip nhất
            if env_state[ENV_ALL_PLAYER_CHIP_GIVE + id_action] + chip_to_allin > np.max(
                env_state[ENV_ALL_PLAYER_CHIP_GIVE:ENV_ALL_PLAYER_CHIP_IN_POT]
            ):
                env_state[ENV_CASH_TO_CALL_OLD] = (
                    env_state[ENV_ALL_PLAYER_CHIP_GIVE + id_action] + chip_to_allin
                )
                env_state[ENV_CASH_TO_CALL_NEW] = env_state[ENV_CASH_TO_CALL_OLD]
                env_state[ENV_TEMP_BUTTON] = id_action
            env_state[ENV_ALL_PLAYER_CHIP_GIVE + id_action] += chip_to_allin
            # trừ hết tiền người chơi all in
            env_state[ENV_ALL_PLAYER_CHIP + id_action] = 0

        if action != 3:
            id_action_next = (id_action + 1) % NUMBER_PLAYER
            """
            hết lượt của người chơi hiện tại, chuyển người chơi, loop đến khi tìm đc người kế tiếp có thể action loop trở về đến
            vị trí hiện tại thì cũng dừng
            """
            if id_action_next != env_state[ENV_TEMP_BUTTON]:
                while (
                    env_state[ENV_ALL_PLAYER_STATUS + id_action_next] == 0
                    or env_state[ENV_ALL_PLAYER_CHIP + id_action_next] == 0
                ):
                    id_action_next = (id_action_next + 1) % NUMBER_PLAYER
                    #  print('while 1')
                    if id_action_next == env_state[ENV_TEMP_BUTTON]:
                        break
            #  print('check', id_action_next, env_state[ENV_STATUS_GAME], env_state[ENV_TEMP_BUTTON])
            if id_action_next == env_state[ENV_TEMP_BUTTON]:
                # kết thúc vòng chơi, chuyển status game, reset give chip, xác định id_action mới là người còn chơi và còn chip. Nếu hết 1 vòng => các người chơi đã allin hết, đi thẳng đến showdown
                if env_state[ENV_STATUS_GAME] != 5:
                    env_state[ENV_ALL_PLAYER_CHIP_GIVE:ENV_ALL_PLAYER_CHIP_IN_POT] = 0
                    env_state[ENV_ID_ACTION] = (
                        env_state[ENV_BUTTON_PLAYER] + 1
                    ) % NUMBER_PLAYER
                    env_state[ENV_CASH_TO_CALL_OLD], env_state[ENV_CASH_TO_CALL_NEW] = (
                        0,
                        0,
                    )

                    if (
                        np.count_nonzero(
                            env_state[ENV_ALL_PLAYER_STATUS:ENV_ALL_FIRST_CARD]
                        )
                        == 1
                        or np.count_nonzero(
                            env_state[ENV_ALL_PLAYER_CHIP:ENV_ALL_PLAYER_CHIP_GIVE]
                            * env_state[ENV_ALL_PLAYER_STATUS:ENV_ALL_FIRST_CARD]
                        )
                        == 1
                    ):
                        if (
                            np.count_nonzero(
                                env_state[ENV_ALL_PLAYER_STATUS:ENV_ALL_FIRST_CARD]
                            )
                            == 1
                        ):
                            env_state[ENV_CARD_OPEN : ENV_CARD_OPEN + NUMBER_CARD_OPEN][
                                max(0, int(env_state[ENV_STATUS_GAME])) :
                            ] = -1
                        env_state[ENV_STATUS_GAME] = 6
                        #  print('gnas đi showdown khi chỉ còn 1 người còn tiền hoặc cả làng đã fold')
                        env_state[ENV_ID_ACTION] = env_state[ENV_TEMP_BUTTON]
                    else:
                        # loop cho đến khi gặp người vừa còn chơi và vừa còn chip
                        while (
                            env_state[
                                int(ENV_ALL_PLAYER_STATUS + env_state[ENV_ID_ACTION])
                            ]
                            == 0
                            or env_state[
                                int(ENV_ALL_PLAYER_CHIP + env_state[ENV_ID_ACTION])
                            ]
                            == 0
                        ):
                            #  print('while 2', env_state[ENV_ID_ACTION], env_state[ENV_BUTTON_PLAYER], env_state[int(ENV_ALL_PLAYER_STATUS + env_state[ENV_ID_ACTION])], env_state[int(ENV_ALL_PLAYER_CHIP + env_state[ENV_ID_ACTION])], env_state[ENV_BUTTON_PLAYER] )
                            env_state[ENV_ID_ACTION] = (
                                env_state[ENV_ID_ACTION] + 1
                            ) % NUMBER_PLAYER
                            # nếu mọi người hết khả năng action tiếp thì đi thẳng đến showdown
                            if env_state[ENV_ID_ACTION] == env_state[ENV_BUTTON_PLAYER]:
                                #  if np.count_nonzero(env_state[ENV_ALL_PLAYER_STATUS : ENV_ALL_FIRST_CARD]) != 1:
                                env_state[ENV_STATUS_GAME] = 6
                                #  print('gnas đi showdown')
                                env_state[ENV_ID_ACTION] = env_state[ENV_TEMP_BUTTON]
                                break

                    if env_state[ENV_STATUS_GAME] != 6:
                        temp_button = (env_state[ENV_BUTTON_PLAYER] + 1) % NUMBER_PLAYER
                        while env_state[int(ENV_ALL_PLAYER_STATUS + temp_button)] == 0:
                            temp_button = (temp_button + 1) % NUMBER_PLAYER
                        env_state[ENV_TEMP_BUTTON] = temp_button

                if env_state[ENV_STATUS_GAME] == 0:
                    env_state[ENV_STATUS_GAME] = 3
                elif env_state[ENV_STATUS_GAME] == 3:
                    env_state[ENV_STATUS_GAME] = 4
                elif env_state[ENV_STATUS_GAME] == 4:
                    env_state[ENV_STATUS_GAME] = 5
                elif env_state[ENV_STATUS_GAME] == 5:
                    env_state[ENV_STATUS_GAME] = 6

                if env_state[ENV_STATUS_GAME] == 6:
                    env_state = showdown(env_state)
                    env_state[ENV_ID_ACTION] = env_state[ENV_TEMP_BUTTON]
            else:
                # cập nhật người chơi
                env_state[ENV_ID_ACTION] = id_action_next
                # new thêm mới
                if (
                    np.count_nonzero(
                        env_state[ENV_ALL_PLAYER_STATUS:ENV_ALL_FIRST_CARD]
                    )
                    == 1
                ):
                    if env_state[ENV_STATUS_GAME] != 5:
                        env_state[ENV_CARD_OPEN : ENV_CARD_OPEN + NUMBER_CARD_OPEN][
                            max(0, int(env_state[ENV_STATUS_GAME])) :
                        ] = -1

                    env_state[ENV_STATUS_GAME] = 6
                    env_state = showdown(env_state)

    elif phase_env == 1:
        if action == 3:
            # nếu người chơi bet tiếp, trừ tiền người chơi, tăng tiền pot, tăng cash to call_new
            # bổ sung giá trị số tiền đã bỏ ra
            chip_to_bet_raise = env_state[ENV_CASH_TO_CALL_OLD]
            env_state[ENV_ALL_PLAYER_CHIP_GIVE + id_action] += chip_to_bet_raise
            env_state[ENV_ALL_PLAYER_CHIP_IN_POT + id_action] += chip_to_bet_raise
            env_state[ENV_POT_VALUE] += chip_to_bet_raise
            env_state[ENV_CASH_TO_CALL_NEW] += chip_to_bet_raise
            env_state[ENV_ALL_PLAYER_CHIP + id_action] -= chip_to_bet_raise
        elif action == 4:
            # nếu all in khi bet tiếp, update chip give, pot value, cash_to_call_old and cash_to_call_new
            chip_to_allin = env_state[ENV_ALL_PLAYER_CHIP + id_action]
            env_state[ENV_ALL_PLAYER_CHIP + id_action] = 0
            env_state[ENV_ALL_PLAYER_CHIP_GIVE + id_action] += chip_to_allin
            env_state[ENV_ALL_PLAYER_CHIP_IN_POT + id_action] += chip_to_allin
            env_state[ENV_POT_VALUE] += chip_to_allin
            env_state[ENV_CASH_TO_CALL_OLD] = (
                env_state[ENV_CASH_TO_CALL_NEW] + chip_to_allin
            )
            env_state[ENV_CASH_TO_CALL_NEW] = env_state[ENV_CASH_TO_CALL_OLD]
            env_state[ENV_TEMP_BUTTON] = id_action
        elif action == 5:
            env_state[ENV_CASH_TO_CALL_OLD] = env_state[ENV_CASH_TO_CALL_NEW]
            env_state[ENV_CASH_TO_CALL_NEW] = env_state[ENV_CASH_TO_CALL_OLD]
            env_state[ENV_TEMP_BUTTON] = id_action

        if action != 3:
            """
            hết lượt của người chơi hiện tại, chuyển người chơi, loop đến khi tìm đc người kế tiếp có thể action loop trở về đến
            vị trí hiện tại thì cũng dừng
            """
            id_action_next = (id_action + 1) % NUMBER_PLAYER
            while (
                env_state[ENV_ALL_PLAYER_STATUS + id_action_next] == 0
                or env_state[ENV_ALL_PLAYER_CHIP + id_action_next] == 0
            ):
                id_action_next = (id_action_next + 1) % NUMBER_PLAYER
                #  print('while 3')
                if id_action_next == env_state[ENV_TEMP_BUTTON]:
                    break
            if id_action_next == env_state[ENV_TEMP_BUTTON]:
                # kết thúc vòng chơi, chuyển status game, reset give chip, xác định id_action mới là người còn chơi và còn chip. Nếu hết 1 vòng => các người chơi đã allin hết, đi thẳng đến showdown
                if env_state[ENV_STATUS_GAME] != 5:
                    env_state[ENV_ALL_PLAYER_CHIP_GIVE:ENV_ALL_PLAYER_CHIP_IN_POT] = 0
                    env_state[ENV_ID_ACTION] = (
                        env_state[ENV_BUTTON_PLAYER] + 1
                    ) % NUMBER_PLAYER
                    env_state[ENV_CASH_TO_CALL_OLD], env_state[ENV_CASH_TO_CALL_NEW] = (
                        0,
                        0,
                    )
                    # loop cho đến khi gặp người vừa còn chơi và vừa còn chip
                    while (
                        env_state[int(ENV_ALL_PLAYER_STATUS + env_state[ENV_ID_ACTION])]
                        == 0
                        or env_state[
                            int(ENV_ALL_PLAYER_CHIP + env_state[ENV_ID_ACTION])
                        ]
                        == 0
                    ):
                        env_state[ENV_ID_ACTION] = (
                            env_state[ENV_ID_ACTION] + 1
                        ) % NUMBER_PLAYER
                        # nếu mọi người hết khả năng action tiếp thì đi thẳng đến showdown
                        if env_state[ENV_ID_ACTION] == env_state[ENV_BUTTON_PLAYER]:
                            #  if np.count_nonzero(env_state[ENV_ALL_PLAYER_STATUS : ENV_ALL_FIRST_CARD]) != 1:
                            env_state[ENV_STATUS_GAME] = 6
                            #  print('gnas đi showdown')
                            env_state[ENV_ID_ACTION] = env_state[ENV_TEMP_BUTTON]
                            break
                    if env_state[ENV_STATUS_GAME] != 6:
                        temp_button = (env_state[ENV_BUTTON_PLAYER] + 1) % NUMBER_PLAYER
                        while env_state[int(ENV_ALL_PLAYER_STATUS + temp_button)] == 0:
                            temp_button = (temp_button + 1) % NUMBER_PLAYER
                        env_state[ENV_TEMP_BUTTON] = temp_button

                if env_state[ENV_STATUS_GAME] == 0:
                    env_state[ENV_STATUS_GAME] = 3
                elif env_state[ENV_STATUS_GAME] == 3:
                    env_state[ENV_STATUS_GAME] = 4
                elif env_state[ENV_STATUS_GAME] == 4:
                    env_state[ENV_STATUS_GAME] = 5
                elif env_state[ENV_STATUS_GAME] == 5:
                    env_state[ENV_STATUS_GAME] = 6

                if env_state[ENV_STATUS_GAME] == 6:
                    env_state = showdown(env_state)
                    env_state[ENV_ID_ACTION] = env_state[ENV_TEMP_BUTTON]

            else:
                # cập nhật người chơi
                env_state[ENV_ID_ACTION] = id_action_next
                env_state[ENV_PHASE] = 0

    return env_state


@njit()
def checkEnded(env_state):
    all_player_chip = env_state[ENV_ALL_PLAYER_CHIP:ENV_ALL_PLAYER_CHIP_GIVE]
    all_player_status = env_state[ENV_ALL_PLAYER_STATUS:ENV_ALL_FIRST_CARD]
    number_game = env_state[ENV_NUMBER_GAME_PLAYED]
    if (
        np.count_nonzero(all_player_status) == 1
        and np.count_nonzero(all_player_chip) == 1
    ) or number_game == 100:
        return True
    return False


@njit()
def getReward(agent_state):
    all_player_chip = agent_state[P_ALL_PLAYER_CHIP:P_ALL_PLAYER_CHIP_GIVE]
    all_player_status = agent_state[P_ALL_PLAYER_STATUS:P_BUTTON_DEALER]

    if agent_state[P_CHECK_END] == 1 and (
        (
            np.count_nonzero(all_player_status) == 1
            and np.count_nonzero(all_player_chip) == 1
        )
        or agent_state[P_NUMBER_GAME_PLAY] == 100
    ):
        if np.argmax(all_player_chip) == 0:
            return 1
        else:
            return 0
    else:
        return -1


@njit()
def check_winner(env_state):
    all_player_chip = env_state[ENV_ALL_PLAYER_CHIP:ENV_ALL_PLAYER_CHIP_GIVE]
    winner = np.argmax(all_player_chip)
    return winner


@njit()
def bot_lv0(state, perData):
    validActions = getValidActions(state)
    arr_action = np.where(validActions == 1)[0]
    idx = np.random.randint(0, arr_action.shape[0])
    return arr_action[idx], perData

    validActions = getValidActions(state)
    arr_action = np.where(validActions == 1)[0]
    idx = np.random.randint(0, arr_action.shape[0])
    action = arr_action[idx]
    if 0 in arr_action:
        action = 0
    elif 1 in arr_action:
        action = 1
    #  print(list(state))
    return action, perData


def one_game_print_mode(
    p0,
    list_other,
    per_player,
    per1,
    per2,
    per3,
    per4,
    per5,
    per6,
    per7,
    per8,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
):
    env_state = initEnv()
    env_state = reset_round(env_state)
    while not checkEnded(env_state):
        print(
            "bài chung: ",
            ALL_CARD_STR[env_state[ENV_CARD_OPEN:ENV_ALL_PLAYER_CHIP].astype(np.int64)],
        )
        print(
            "bài riêng1: ",
            ALL_CARD_STR[
                env_state[ENV_ALL_FIRST_CARD:ENV_ALL_SECOND_CARD].astype(np.int64)
            ],
        )
        print(
            "bài riêng2: ",
            ALL_CARD_STR[
                env_state[ENV_ALL_SECOND_CARD:ENV_ALL_FIRST_CARD_SHOWDOWN].astype(
                    np.int64
                )
            ],
        )

        while env_state[ENV_STATUS_GAME] != 6:
            idx = int(env_state[ENV_ID_ACTION])
            player_state = getAgentState(env_state)
            if list_other[idx] == -1:
                action, per_player = p0(player_state, per_player)
                if getValidActions(player_state)[action] != 1:
                    raise Exception("bot dua ra action khong hop le")
            elif list_other[idx] == 1:
                action, per1 = p1(player_state, per1)
            elif list_other[idx] == 2:
                action, per2 = p2(player_state, per2)
            elif list_other[idx] == 3:
                action, per3 = p3(player_state, per3)
            elif list_other[idx] == 4:
                action, per4 = p4(player_state, per4)
            elif list_other[idx] == 5:
                action, per5 = p5(player_state, per5)
            elif list_other[idx] == 6:
                action, per6 = p6(player_state, per6)
            elif list_other[idx] == 7:
                action, per7 = p7(player_state, per7)
            elif list_other[idx] == 8:
                action, per8 = p8(player_state, per8)
            else:
                raise Exception("Sai list_other.")

            env_state = stepEnv(env_state, action)
        print(
            "turn bonus++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )
        for id in range(NUMBER_PLAYER):
            id_player = int(id + env_state[ENV_TEMP_BUTTON]) % NUMBER_PLAYER
            if env_state[ENV_ALL_PLAYER_STATUS + id_player] == 0:
                continue
            else:
                env_state[ENV_ID_ACTION] = id_player
                idx = int(env_state[ENV_ID_ACTION])
                player_state = getAgentState(env_state)
                if list_other[idx] == -1:
                    action, per_player = p0(player_state, per_player)
                    if getValidActions(player_state)[action] != 1:
                        raise Exception("bot dua ra action khong hop le")
                elif list_other[idx] == 1:
                    action, per1 = p1(player_state, per1)
                elif list_other[idx] == 2:
                    action, per2 = p2(player_state, per2)
                elif list_other[idx] == 3:
                    action, per3 = p3(player_state, per3)
                elif list_other[idx] == 4:
                    action, per4 = p4(player_state, per4)
                elif list_other[idx] == 5:
                    action, per5 = p5(player_state, per5)
                elif list_other[idx] == 6:
                    action, per6 = p6(player_state, per6)
                elif list_other[idx] == 7:
                    action, per7 = p7(player_state, per7)
                elif list_other[idx] == 8:
                    action, per8 = p8(player_state, per8)
                else:
                    raise Exception("Sai list_other.")
        print(
            "end turn bonus+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )
        env_state = reset_round(env_state)

    winner_real = check_winner(env_state)
    print("winner: ", winner_real, env_state[ENV_NUMBER_GAME_PLAYED])
    for p_idx in range(NUMBER_PLAYER):
        env_state[ENV_PHASE] = 1
        idx = int(env_state[ENV_ID_ACTION])
        player_state = getAgentState(env_state)
        if list_other[idx] == -1:
            action, per_player = p0(player_state, per_player)
        elif list_other[idx] == 1:
            action, per1 = p1(player_state, per1)
        elif list_other[idx] == 2:
            action, per2 = p2(player_state, per2)
        elif list_other[idx] == 3:
            action, per3 = p3(player_state, per3)
        elif list_other[idx] == 4:
            action, per4 = p4(player_state, per4)
        elif list_other[idx] == 5:
            action, per5 = p5(player_state, per5)
        elif list_other[idx] == 6:
            action, per6 = p6(player_state, per6)
        elif list_other[idx] == 7:
            action, per7 = p7(player_state, per7)
        elif list_other[idx] == 8:
            action, per8 = p8(player_state, per8)
        else:
            raise Exception("Sai list_other.")

        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4

    winner = False
    if np.where(list_other == -1)[0] == check_winner(env_state):
        winner = True
    else:
        winner = False
    return winner, per_player


def n_games_print_mode(
    p0,
    num_game,
    per_player,
    list_other,
    per1,
    per2,
    per3,
    per4,
    per5,
    per6,
    per7,
    per8,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_print_mode(
            p0,
            list_other,
            per_player,
            per1,
            per2,
            per3,
            per4,
            per5,
            per6,
            per7,
            per8,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            p8,
        )
        win += winner
    return win, per_player


@njit()
def one_game_numba(
    p0,
    list_other,
    per_player,
    per1,
    per2,
    per3,
    per4,
    per5,
    per6,
    per7,
    per8,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
):
    env_state = initEnv()
    env_state = reset_round(env_state)
    while not checkEnded(env_state):
        while env_state[ENV_STATUS_GAME] != 6:
            idx = int(env_state[ENV_ID_ACTION])
            player_state = getAgentState(env_state)
            if list_other[idx] == -1:
                action, per_player = p0(player_state, per_player)
                if getValidActions(player_state)[action] != 1:
                    raise Exception("bot dua ra action khong hop le")
            elif list_other[idx] == 1:
                action, per1 = p1(player_state, per1)
            elif list_other[idx] == 2:
                action, per2 = p2(player_state, per2)
            elif list_other[idx] == 3:
                action, per3 = p3(player_state, per3)
            elif list_other[idx] == 4:
                action, per4 = p4(player_state, per4)
            elif list_other[idx] == 5:
                action, per5 = p5(player_state, per5)
            elif list_other[idx] == 6:
                action, per6 = p6(player_state, per6)
            elif list_other[idx] == 7:
                action, per7 = p7(player_state, per7)
            elif list_other[idx] == 8:
                action, per8 = p8(player_state, per8)
            else:
                raise Exception("Sai list_other.")

            env_state = stepEnv(env_state, action)

        for id in range(NUMBER_PLAYER):
            id_player = int(id + env_state[ENV_TEMP_BUTTON]) % NUMBER_PLAYER
            if env_state[ENV_ALL_PLAYER_STATUS + id_player] == 0:
                continue
            else:
                env_state[ENV_ID_ACTION] = id_player
                idx = int(env_state[ENV_ID_ACTION])
                player_state = getAgentState(env_state)
                if list_other[idx] == -1:
                    action, per_player = p0(player_state, per_player)
                    if getValidActions(player_state)[action] != 1:
                        raise Exception("bot dua ra action khong hop le")
                elif list_other[idx] == 1:
                    action, per1 = p1(player_state, per1)
                elif list_other[idx] == 2:
                    action, per2 = p2(player_state, per2)
                elif list_other[idx] == 3:
                    action, per3 = p3(player_state, per3)
                elif list_other[idx] == 4:
                    action, per4 = p4(player_state, per4)
                elif list_other[idx] == 5:
                    action, per5 = p5(player_state, per5)
                elif list_other[idx] == 6:
                    action, per6 = p6(player_state, per6)
                elif list_other[idx] == 7:
                    action, per7 = p7(player_state, per7)
                elif list_other[idx] == 8:
                    action, per8 = p8(player_state, per8)
                else:
                    raise Exception("Sai list_other.")

        env_state = reset_round(env_state)

    env_state[ENV_CHECK_END] = 1
    win = check_winner(env_state)

    for p_idx in range(NUMBER_PLAYER):
        env_state[ENV_PHASE] = 1
        idx = int(env_state[ENV_ID_ACTION])
        player_state = getAgentState(env_state)
        if list_other[idx] == -1:
            action, per_player = p0(player_state, per_player)
        elif list_other[idx] == 1:
            action, per1 = p1(player_state, per1)
        elif list_other[idx] == 2:
            action, per2 = p2(player_state, per2)
        elif list_other[idx] == 3:
            action, per3 = p3(player_state, per3)
        elif list_other[idx] == 4:
            action, per4 = p4(player_state, per4)
        elif list_other[idx] == 5:
            action, per5 = p5(player_state, per5)
        elif list_other[idx] == 6:
            action, per6 = p6(player_state, per6)
        elif list_other[idx] == 7:
            action, per7 = p7(player_state, per7)
        elif list_other[idx] == 8:
            action, per8 = p8(player_state, per8)
        else:
            raise Exception("Sai list_other.")

        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER

    winner = False
    if np.where(list_other == -1)[0] == win:
        winner = True
    else:
        winner = False
    return winner, per_player


@njit()
def n_games_numba(
    p0,
    num_game,
    per_player,
    list_other,
    per1,
    per2,
    per3,
    per4,
    per5,
    per6,
    per7,
    per8,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_numba(
            p0,
            list_other,
            per_player,
            per1,
            per2,
            per3,
            per4,
            per5,
            per6,
            per7,
            per8,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            p8,
        )
        win += winner
    return win, per_player


def one_game_normal(
    p0,
    list_other,
    per_player,
    per1,
    per2,
    per3,
    per4,
    per5,
    per6,
    per7,
    per8,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
):
    env_state = initEnv()
    env_state = reset_round(env_state)
    while not checkEnded(env_state):
        while env_state[ENV_STATUS_GAME] != 6:
            idx = int(env_state[ENV_ID_ACTION])
            player_state = getAgentState(env_state)
            if list_other[idx] == -1:
                action, per_player = p0(player_state, per_player)
                if getValidActions(player_state)[action] != 1:
                    raise Exception("bot dua ra action khong hop le")
            elif list_other[idx] == 1:
                action, per1 = p1(player_state, per1)
            elif list_other[idx] == 2:
                action, per2 = p2(player_state, per2)
            elif list_other[idx] == 3:
                action, per3 = p3(player_state, per3)
            elif list_other[idx] == 4:
                action, per4 = p4(player_state, per4)
            elif list_other[idx] == 5:
                action, per5 = p5(player_state, per5)
            elif list_other[idx] == 6:
                action, per6 = p6(player_state, per6)
            elif list_other[idx] == 7:
                action, per7 = p7(player_state, per7)
            elif list_other[idx] == 8:
                action, per8 = p8(player_state, per8)
            else:
                raise Exception("Sai list_other.")

            env_state = stepEnv(env_state, action)
        for id in range(NUMBER_PLAYER):
            id_player = int(id + env_state[ENV_TEMP_BUTTON]) % NUMBER_PLAYER
            if env_state[ENV_ALL_PLAYER_STATUS + id_player] == 0:
                continue
            else:
                env_state[ENV_ID_ACTION] = id_player
                idx = int(env_state[ENV_ID_ACTION])
                player_state = getAgentState(env_state)
                if list_other[idx] == -1:
                    action, per_player = p0(player_state, per_player)
                    if getValidActions(player_state)[action] != 1:
                        raise Exception("bot dua ra action khong hop le")
                elif list_other[idx] == 1:
                    action, per1 = p1(player_state, per1)
                elif list_other[idx] == 2:
                    action, per2 = p2(player_state, per2)
                elif list_other[idx] == 3:
                    action, per3 = p3(player_state, per3)
                elif list_other[idx] == 4:
                    action, per4 = p4(player_state, per4)
                elif list_other[idx] == 5:
                    action, per5 = p5(player_state, per5)
                elif list_other[idx] == 6:
                    action, per6 = p6(player_state, per6)
                elif list_other[idx] == 7:
                    action, per7 = p7(player_state, per7)
                elif list_other[idx] == 8:
                    action, per8 = p8(player_state, per8)
                else:
                    raise Exception("Sai list_other.")
        env_state = reset_round(env_state)

    env_state[ENV_CHECK_END] = 1
    win = check_winner(env_state)

    for p_idx in range(NUMBER_PLAYER):
        env_state[ENV_PHASE] = 1
        idx = int(env_state[ENV_ID_ACTION])
        player_state = getAgentState(env_state)
        if list_other[idx] == -1:
            if getReward(player_state) == -1:
                print("game sai")
                print(list(env_state))
            action, per_player = p0(player_state, per_player)
        elif list_other[idx] == 1:
            action, per1 = p1(player_state, per1)
        elif list_other[idx] == 2:
            action, per2 = p2(player_state, per2)
        elif list_other[idx] == 3:
            action, per3 = p3(player_state, per3)
        elif list_other[idx] == 4:
            action, per4 = p4(player_state, per4)
        elif list_other[idx] == 5:
            action, per5 = p5(player_state, per5)
        elif list_other[idx] == 6:
            action, per6 = p6(player_state, per6)
        elif list_other[idx] == 7:
            action, per7 = p7(player_state, per7)
        elif list_other[idx] == 8:
            action, per8 = p8(player_state, per8)
        else:
            raise Exception("Sai list_other.")

        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER

    winner = False
    if np.where(list_other == -1)[0] == win:
        winner = True
    else:
        winner = False
    return winner, per_player


def n_games_normal(
    p0,
    num_game,
    per_player,
    list_other,
    per1,
    per2,
    per3,
    per4,
    per5,
    per6,
    per7,
    per8,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_normal(
            p0,
            list_other,
            per_player,
            per1,
            per2,
            per3,
            per4,
            per5,
            per6,
            per7,
            per8,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
            p8,
        )
        win += winner
    return win, per_player


import importlib.util
import json, sys

try:
    from env import SHORT_PATH
except:
    pass


def load_module_player(player, game_name=None):
    if game_name == None:
        spec = importlib.util.spec_from_file_location(
            "Agent_player", f"{SHORT_PATH}src/Agent/{player}/Agent_player.py"
        )
    else:
        spec = importlib.util.spec_from_file_location(
            "Agent_player", f"{SHORT_PATH}src/Agent/ifelse/{game_name}/{player}.py"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@njit()
def check_run_under_njit(agent, perData):
    return True


def load_agent(level, *args):
    num_bot = getAgentSize() - 1

    if "_level_" not in globals():
        global _level_
        _level_ = level
        init = True
    else:
        if _level_ != level:
            _level_ = level
            init = True
        else:
            init = False

    if init:
        global _list_per_level_
        global _list_bot_level_
        _list_per_level_ = []
        _list_bot_level_ = []

        if _level_ == 0:
            _list_per_level_ = [
                np.array([[0.0]], dtype=np.float64) for _ in range(num_bot)
            ]
            _list_bot_level_ = [bot_lv0 for _ in range(num_bot)]
        else:
            env_name = sys.argv[1]
            if len(args) > 0:
                dict_level = json.load(
                    open(f"{SHORT_PATH}src/Log/check_system_about_level.json")
                )
            else:
                dict_level = json.load(open(f"{SHORT_PATH}src/Log/level_game.json"))

            if str(_level_) not in dict_level[env_name]:
                raise Exception("Hiện tại không có level này")

            lst_agent_level = dict_level[env_name][str(level)][2]

            for i in range(num_bot):
                if level == -1:
                    module_agent = load_module_player(
                        lst_agent_level[i], game_name=env_name
                    )
                    _list_per_level_.append(module_agent.DataAgent())
                else:
                    data_agent_level = np.load(
                        f"{SHORT_PATH}src/Agent/{lst_agent_level[i]}/Data/{env_name}_{level}/Train.npy",
                        allow_pickle=True,
                    )
                    module_agent = load_module_player(lst_agent_level[i])
                    _list_per_level_.append(
                        module_agent.convert_to_test(data_agent_level)
                    )
                _list_bot_level_.append(module_agent.Test)

    return _list_bot_level_, _list_per_level_


def run(
    p0: any = bot_lv0,
    num_game: int = 100,
    per_player: np.ndarray = np.array([[0.0]]),
    level: int = 0,
    *args,
):
    num_bot = getAgentSize() - 1
    list_other = np.array([-1] + [i + 1 for i in range(num_bot)])
    try:
        check_njit = check_run_under_njit(p0, per_player)
    except:
        check_njit = False

    load_agent(level, *args)

    if check_njit:
        return n_games_numba(
            p0,
            num_game,
            per_player,
            list_other,
            _list_per_level_[0],
            _list_per_level_[1],
            _list_per_level_[2],
            _list_per_level_[3],
            _list_per_level_[4],
            _list_per_level_[5],
            _list_per_level_[6],
            _list_per_level_[7],
            _list_bot_level_[0],
            _list_bot_level_[1],
            _list_bot_level_[2],
            _list_bot_level_[3],
            _list_bot_level_[4],
            _list_bot_level_[5],
            _list_bot_level_[6],
            _list_bot_level_[7],
        )
    else:
        return n_games_normal(
            p0,
            num_game,
            per_player,
            list_other,
            _list_per_level_[0],
            _list_per_level_[1],
            _list_per_level_[2],
            _list_per_level_[3],
            _list_per_level_[4],
            _list_per_level_[5],
            _list_per_level_[6],
            _list_per_level_[7],
            _list_bot_level_[0],
            _list_bot_level_[1],
            _list_bot_level_[2],
            _list_bot_level_[3],
            _list_bot_level_[4],
            _list_bot_level_[5],
            _list_bot_level_[6],
            _list_bot_level_[7],
        )
