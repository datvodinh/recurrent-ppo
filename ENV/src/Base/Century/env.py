import numba as nb
import numpy as np
from numba import njit

from src.Base.Century.docs.index import *


@njit()
def getAgentSize():
    return 5


@njit()
def getStateSize():
    return 277


@njit()
def getActionSize():
    return 65


# khởi tạo bàn chơi
@njit()
def initEnv():
    start_card_player = np.concatenate((np.array([1]), np.zeros(42), np.array([1, 0])))
    start_player_0 = np.concatenate((np.array([0, 0, 3, 0, 0, 0]), start_card_player))
    start_player_1 = np.concatenate((np.array([0, 0, 4, 0, 0, 0]), start_card_player))
    start_player_2 = np.concatenate((np.array([0, 0, 4, 0, 0, 0]), start_card_player))
    start_player_3 = np.concatenate((np.array([0, 0, 3, 1, 0, 0]), start_card_player))
    start_player_4 = np.concatenate((np.array([0, 0, 3, 1, 0, 0]), start_card_player))
    list_card = np.append(np.arange(1, 43), 44)
    list_card_point = np.arange(36)
    np.random.shuffle(list_card)
    np.random.shuffle(list_card_point)
    top_6_card = ALL_CARD_IN4[list_card[:NUMBER_OPEN_ACTION_CARD]].flatten()
    top_5_card_point = ALL_CARD_POINT_IN4[
        list_card_point[:NUMBER_OPEN_POINT_CARD]
    ].flatten()
    order_player = np.arange(NUMBER_PLAYER)
    env_state = np.concatenate(
        (
            start_player_0,
            start_player_1,
            start_player_2,
            start_player_3,
            start_player_4,
            top_6_card,
            np.zeros(20),
            top_5_card_point,
            list_card,
            list_card_point,
            np.array([0, -0.5, 0, 10, 10, -1, 1, 0, 0]),
            order_player,
        )
    )
    # 5 player_in4, 6card_in4,5 token free, 5 card_point_in4, list_card_shuffle, list_card_point_shuffle, [number_action_upgrade, card_will_buy/card_hand_used, token need drop, silver, gold, last_action, phase, check_end, id_action]
    return env_state


@njit()
def getAgentState(env_state):
    player_state = np.zeros(P_LENGTH, dtype=np.float64)
    player_action = int(env_state[ENV_ID_ACTION])  # xác định người chơi
    player_main_in4 = env_state[
        ATTRIBUTE_PLAYER * player_action : ATTRIBUTE_PLAYER * (player_action + 1)
    ]
    player_state[P_SCORE : P_SCORE + BASIC_ATTRIBUTE_PLAYER] = player_main_in4[
        :BASIC_ATTRIBUTE_PLAYER
    ]
    player_state[P_ACTION_CARD_PLAYER:P_ACTION_CARD_DOWN_PLAYER] = (
        player_main_in4[BASIC_ATTRIBUTE_PLAYER:] > 0
    ) * 1
    player_state[P_ACTION_CARD_DOWN_PLAYER:P_OTHER_PLAYER_IN4] = (
        player_main_in4[BASIC_ATTRIBUTE_PLAYER:] < 0
    ) * 1
    for idx in range(1, 5):
        id = int((player_action + idx) % NUMBER_PLAYER)
        all_other_player_in4 = env_state[
            ATTRIBUTE_PLAYER * id : ATTRIBUTE_PLAYER * id + BASIC_ATTRIBUTE_PLAYER
        ].copy()
        # all_other_player_card = np.where(all_other_player_in4[BASIC_ATTRIBUTE_PLAYER:] == -1, 1, 0)      #xác định thẻ người chơi có dựa trên các thẻ đã đánh
        player_state[
            P_OTHER_PLAYER_IN4
            + (idx - 1) * BASIC_ATTRIBUTE_PLAYER : P_OTHER_PLAYER_IN4
            + idx * BASIC_ATTRIBUTE_PLAYER
        ] = all_other_player_in4
    # player_state = np.concatenate((player_state, env_state[ENV_OPEN_ACTION_CARD:ENV_DOWN_ACTION_CARD], env_state[ENV_SILVER_COIN : ENV_ID_ACTION]))
    player_state[P_OPEN_ACTION_CARD:P_SILVER_COIN] = env_state[
        ENV_OPEN_ACTION_CARD:ENV_DOWN_ACTION_CARD
    ]
    player_state[P_SILVER_COIN] = env_state[ENV_SILVER_COIN]
    player_state[P_GOLD_COIN] = env_state[ENV_GOLD_COIN]
    # action gần nhất nếu dùng thẻ action đổi token
    if env_state[ENV_LAST_ACTION] != -1:
        player_state[
            P_LAST_ACTION + int(env_state[ENV_LAST_ACTION])
        ] = 1  # update 25/4/2023
    # kiểm tra end game
    player_state[P_CHECK_END] = env_state[ENV_CHECK_END]

    # THứ tự người chơi
    player_state[P_ORDER : P_ORDER + NUMBER_PLAYER] = np.concatenate(
        (
            env_state[
                ENV_ORDER_PLAYER + player_action : ENV_ORDER_PLAYER + NUMBER_PLAYER
            ],
            env_state[ENV_ORDER_PLAYER : ENV_ORDER_PLAYER + player_action],
        )
    )
    # phase
    player_state[P_PHASE + int(env_state[ENV_PHASE]) - 1] = 1
    # print(list(player_state))
    # print(np.sum(player_state[221:266]), player_state[221:266])
    return player_state


@njit()
def getValidActions(player_state_origin):
    list_action_return = np.zeros(65, dtype=np.float64)
    player_state = player_state_origin.copy()
    phase_env = -1
    if len(np.where(player_state[P_PHASE : P_PHASE + NUMBER_PHASE] == 1)[0]) != 0:
        phase_env = (
            np.where(player_state[P_PHASE : P_PHASE + NUMBER_PHASE] == 1)[0][0] + 1
        )
    player_state_own = player_state[:P_OTHER_PLAYER_IN4]
    """
        Quy ước phase: 
        phase1: chọn mua thẻ (6 thẻ top và 5 thẻ point) hay đánh thẻ (thẻ trên tay) hay nghỉ ngơi (11 action mua, 45 action đánh, 1 action nghỉ)
        phase2: nếu mua thẻ top, chọn token để vào các thẻ trước thẻ mình mua (4 action)
        phase3: nếu đánh thẻ, chọn xem có thực hiện action của thẻ tiếp ko (2action, 1 cái là ko, 1 cái trùng vs action dùng thẻ)
        phase4: trả token dư thừa (4 action) (trùng phase 2)
        phase5: chọn tài nguyên nâng cấp    
    """
    player_token = player_state_own[P_TOKEN:P_ACTION_CARD_PLAYER]
    if phase_env == 1:
        # chọn mua thẻ (6 thẻ top và 5 thẻ point) hay đánh thẻ (thẻ trên tay) hay nghỉ ngơi (11 action mua, 45 action đánh, 1 action nghỉ)
        list_action_return[0] = 1  # luôn có action nghỉ ngơi
        # check mua 6 thẻ top
        number_token = np.sum(player_token)
        card_on_board = player_state[P_OPEN_ACTION_CARD:P_TOKEN_ON_ACTION_CARD]
        for act in range(6):
            if (
                act <= number_token
                and np.sum(
                    card_on_board[
                        LENGTH_ACTION_CARD * act : LENGTH_ACTION_CARD * (act + 1)
                    ]
                )
                > 0
            ):  # kiểm tra đủ tài nguyên đặt không, bàn chơi có còn thẻ ko
                list_action_return[act + FIRST_ACTION_BUY_ACTION_CARD] = 1
            else:
                break
        # check mua 5 thẻ point
        all_card_point = player_state[
            P_OPEN_POINT_CARD : P_OPEN_POINT_CARD
            + NUMBER_OPEN_POINT_CARD * ATTRIBUTE_POINT_CARD
        ]
        for id in range(5):
            card_in4 = all_card_point[
                ATTRIBUTE_POINT_CARD * id : ATTRIBUTE_POINT_CARD * (id + 1)
            ][:NUMBBER_TYPE_TOKEN]
            if np.sum(card_in4 > player_token) == 0:
                list_action_return[id + FIRST_ACTION_BUY_POINT_CARD] = 1
        # check đánh thẻ trên tay
        data = ALL_CARD_IN4.copy()
        for card_hand in range(NUMBER_ACTION_CARD - NUMBER_UPGRADE_CARD):
            if player_state[P_ACTION_CARD_PLAYER + card_hand] == 1:
                give = data[card_hand][:NUMBBER_TYPE_TOKEN]
                if np.sum(give > player_token) == 0:
                    list_action_return[card_hand + FIRST_ACTION_USE_ACTION_CARD] = 1

        if player_state[ATTRIBUTE_PLAYER - NUMBER_UPGRADE_CARD] == 1:
            if np.sum(player_token[:3] > 0) != 0:
                list_action_return[55] = 1
        if player_state[ATTRIBUTE_PLAYER - NUMBER_UPGRADE_CARD + 1] == 1:
            if np.sum(player_token[:3] > 0) != 0:
                list_action_return[56] = 1

    elif phase_env == 2:
        # nếu mua thẻ top và cần bỏ token, chọn token để vào các thẻ trước thẻ mình mua (4 action)
        list_action = np.where(player_token > 0)[0] + FIRST_ACTION_DROP_TOKEN
        list_action_return[list_action] = 1

    elif phase_env == 3:
        # nếu đánh thẻ, chọn xem có thực hiện action của thẻ tiếp ko (2action, 1 cái là ko, 1 cái trùng vs action dùng thẻ)
        list_action_return[61] = 1
        list_action_return[
            FIRST_ACTION_USE_ACTION_CARD : FIRST_ACTION_USE_ACTION_CARD
            + NUMBER_ACTION_CARD
        ] = (player_state[P_LAST_ACTION : P_LAST_ACTION + NUMBER_ACTION_CARD] == 1) * 1

    elif phase_env == 4:
        # trả token dư thừa (4 action) sau khi đánh thẻ hoặc mua thẻ top
        list_action = np.where(player_token > 0)[0] + FIRST_ACTION_DROP_TOKEN
        list_action_return[list_action] = 1

    elif phase_env == 5:
        # chọn token để nâng cấp
        list_action = np.where(player_token[:3] > 0)[0] + FIRST_ACTION_UPGRADE_TOKEN
        list_action_return[list_action] = 1

    return list_action_return


@njit()
def getReward(player_state):
    value_return = -1
    end_true = 0
    # kiểm tra xem game kết thúc thật hay loop quá giới hạn
    if player_state[P_PLAYER_NUMBER_POINT_CARD] == 5:
        end_true = 1
    else:
        for id in range(NUMBER_PLAYER - 1):
            if (
                player_state[P_OTHER_PLAYER_IN4:][
                    id * BASIC_ATTRIBUTE_PLAYER : (id + 1) * BASIC_ATTRIBUTE_PLAYER
                ][1]
                == 5
            ):
                end_true = 1
                break

    if player_state[P_CHECK_END] != 1:
        return value_return
    else:
        if end_true == 0:  # nếu loop quá lâu thì auto thua
            return 0
        else:
            # all_player_point = np.zeros(NUMBER_PLAYER)
            all_player_point = player_state[P_ORDER : P_ORDER + NUMBER_PLAYER].copy()
            for id_player in range(5):
                player_in4 = np.zeros(BASIC_ATTRIBUTE_PLAYER)
                if id_player == 0:
                    player_in4 = player_state[:BASIC_ATTRIBUTE_PLAYER]
                else:
                    player_in4 = player_state[
                        P_OTHER_PLAYER_IN4
                        + (id_player - 1) * BASIC_ATTRIBUTE_PLAYER : P_OTHER_PLAYER_IN4
                        + id_player * BASIC_ATTRIBUTE_PLAYER
                    ]

                player_point = player_in4[0] + np.sum(player_in4[3:6]) * 10
                all_player_point[id_player] += player_point
            if np.argmax(all_player_point) == 0:
                return 1
            else:
                return 0


@njit()
def check_winner(env_state):
    winner = -1
    end = 0
    for id_player in range(5):
        player_number_card_point = env_state[ATTRIBUTE_PLAYER * id_player + 1]
        if player_number_card_point == 5:
            end = 1
            break
    if end == 1:
        # all_player_point = np.zeros(NUMBER_PLAYER)
        all_player_point = env_state[
            ENV_ORDER_PLAYER : ENV_ORDER_PLAYER + NUMBER_PLAYER
        ].copy()
        for id_player in range(5):
            player_in4 = env_state[
                ATTRIBUTE_PLAYER * id_player : ATTRIBUTE_PLAYER * (id_player + 1)
            ]
            player_point = player_in4[0] + np.sum(player_in4[3:6]) * 10
            # all_player_point[id_player] = player_point
            all_player_point[id_player] += player_point
        winner = np.argmax(all_player_point)
        return winner
    else:
        # print('một ván ko kết thúc')
        return winner


@njit()
def system_check_end(env_state):
    for id_player in range(5):
        if (
            env_state[
                ATTRIBUTE_PLAYER * id_player : ATTRIBUTE_PLAYER * (id_player + 1)
            ][1]
            == 5
            and env_state[ENV_ID_ACTION] == 0
        ):
            return False
    return True


@njit()
def stepEnv(env_state, action):
    phase_env = int(env_state[ENV_PHASE])
    id_action = int(env_state[ENV_ID_ACTION])
    player_in4 = env_state[
        ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
    ]
    if phase_env == 1:
        # nếu người chơi nghỉ
        if action == 0:
            card_hand_player = player_in4[BASIC_ATTRIBUTE_PLAYER:ATTRIBUTE_PLAYER]
            card_hand_player = np.where(card_hand_player == -1, 1, card_hand_player)
            player_in4[BASIC_ATTRIBUTE_PLAYER:ATTRIBUTE_PLAYER] = card_hand_player
            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4
            env_state[ENV_PHASE] = 1
            env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 5
        # nếu người chơi mua thẻ trên bàn
        elif action in range(1, 7):
            if action == 1:  # nếu mua thẻ đầu
                # lấy thông tin
                list_card_player = player_in4[BASIC_ATTRIBUTE_PLAYER:ATTRIBUTE_PLAYER]
                list_card_board = env_state[ENV_DOWN_ACTION_CARD:ENV_DOWN_POINT_CARD]
                idx_card_buy = action - 1
                card_buy = int(list_card_board[idx_card_buy])
                all_token_free = env_state[ENV_TOKEN_ON_ACTION_CARD:ENV_OPEN_POINT_CARD]
                token_free = all_token_free[
                    NUMBBER_TYPE_TOKEN
                    * idx_card_buy : NUMBBER_TYPE_TOKEN
                    * (idx_card_buy + 1)
                ]
                all_token_free = np.concatenate(
                    (all_token_free[NUMBBER_TYPE_TOKEN:], np.zeros(4))
                )
                # cập nhật giá trị
                list_card_player[card_buy] = 1
                list_card_board[idx_card_buy:] = np.append(
                    list_card_board[idx_card_buy + 1 :], -1
                )
                top_6_card = np.zeros((NUMBER_OPEN_ACTION_CARD, LENGTH_ACTION_CARD))
                for i in range(6):
                    id = list_card_board[:6][i]
                    top_6_card[i] = ALL_CARD_IN4[int(id)]
                top_6_card = top_6_card.flatten()
                player_in4[
                    BASIC_ATTRIBUTE_PLAYER:
                ] = list_card_player  # cập nhật thẻ mới mua
                player_in4[2:6] += token_free  # cập nhật token free nếu có
                env_state[
                    ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
                ] = player_in4
                env_state[
                    ENV_DOWN_ACTION_CARD:ENV_DOWN_POINT_CARD
                ] = list_card_board  # cập nhật danh sách thẻ trên bàn
                env_state[
                    ENV_OPEN_ACTION_CARD:ENV_TOKEN_ON_ACTION_CARD
                ] = top_6_card  # cập nhật 6 thẻ người chơi có thể mua
                env_state[
                    ENV_TOKEN_ON_ACTION_CARD:ENV_OPEN_POINT_CARD
                ] = all_token_free  # cập nhật giảm token free
                # kiểm tra có phải trả tài nguyên ko
                if np.sum(player_in4[2:BASIC_ATTRIBUTE_PLAYER]) > 10:
                    env_state[ENV_TOKEN_NEED_DROP] = (
                        np.sum(player_in4[2:BASIC_ATTRIBUTE_PLAYER]) - 10
                    )
                    env_state[ENV_PHASE] = 4
                else:
                    # chuyển người chơi
                    env_state[ENV_PHASE] = 1
                    env_state[ENV_ID_ACTION] = (
                        env_state[ENV_ID_ACTION] + 1
                    ) % NUMBER_PLAYER

            else:  # nếu mua thẻ cần đặt token
                # lấy thông tin
                idx_card_buy = action - 1
                # đẩy thông tin vào hệ thống, thông tin là index thẻ mua và số token cần đặt
                env_state[ENV_TOKEN_NEED_DROP] = idx_card_buy
                env_state[ENV_CARD_BUY_OR_USE] = idx_card_buy
                # chuyển phase
                env_state[ENV_PHASE] = 2

        elif action in range(7, 12):
            # lấy thông tin
            list_card_point_board = env_state[
                ENV_DOWN_POINT_CARD : ENV_DOWN_POINT_CARD + NUMBER_POINT_CARD
            ]
            idx_card_buy = action - 7
            card_buy = int(list_card_point_board[idx_card_buy])
            token_fee = ALL_CARD_POINT_IN4[card_buy][:4]
            free_score = 0
            if idx_card_buy < 2:
                if idx_card_buy == 0:
                    if env_state[ENV_GOLD_COIN] != 0:
                        free_score = FREE_SCORE_GOLD
                        env_state[ENV_GOLD_COIN] -= 1
                    else:
                        if env_state[ENV_SILVER_COIN] != 0:
                            free_score = FREE_SCORE_SILVER
                            env_state[ENV_SILVER_COIN] -= 1
                else:
                    if (
                        env_state[ENV_SILVER_COIN] != 0
                        and env_state[ENV_GOLD_COIN] != 0
                    ):
                        free_score = FREE_SCORE_SILVER
                        env_state[ENV_SILVER_COIN] -= 1
            # Cập nhật giá trị
            player_in4[0] += free_score + ALL_CARD_POINT_IN4[card_buy][-1]
            player_in4[1] += 1
            player_in4[2:BASIC_ATTRIBUTE_PLAYER] -= token_fee
            list_card_point_board[idx_card_buy:] = np.append(
                list_card_point_board[idx_card_buy + 1 :], -1
            )
            top_5_card_point = np.zeros((NUMBER_OPEN_POINT_CARD, ATTRIBUTE_POINT_CARD))
            for i in range(NUMBER_OPEN_POINT_CARD):
                id = list_card_point_board[:5][i]
                top_5_card_point[i] = ALL_CARD_POINT_IN4[int(id)]
            top_5_card_point = top_5_card_point.flatten()

            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4
            env_state[
                ENV_DOWN_POINT_CARD : ENV_DOWN_POINT_CARD + NUMBER_POINT_CARD
            ] = list_card_point_board
            env_state[
                ENV_OPEN_POINT_CARD : ENV_OPEN_POINT_CARD
                + NUMBER_OPEN_POINT_CARD * ATTRIBUTE_POINT_CARD
            ] = top_5_card_point
            # Chuyển người chơi
            env_state[ENV_PHASE] = 1
            env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER

        elif action in range(12, 57):
            # lấy thông tin
            card_hand_player = player_in4[BASIC_ATTRIBUTE_PLAYER:ATTRIBUTE_PLAYER]
            id_card_use = action - 12
            token_fee_get = ALL_CARD_IN4[id_card_use][:ATTRIBUTE_ACTION_CARD]
            card_hand_player[id_card_use] = -1

            if np.sum(token_fee_get) == 0:  # nếu là thẻ nâng cấp
                player_in4[BASIC_ATTRIBUTE_PLAYER:ATTRIBUTE_PLAYER] = card_hand_player
                env_state[
                    ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
                ] = player_in4
                env_state[ENV_NUMBER_ACTION_UPGRADE] = ALL_CARD_IN4[id_card_use][-1]
                env_state[ENV_PHASE] = 5
            else:
                # Cập nhật giá trị
                player_in4[2:BASIC_ATTRIBUTE_PLAYER] = (
                    player_in4[2:BASIC_ATTRIBUTE_PLAYER]
                    - token_fee_get[:4]
                    + token_fee_get[4:]
                )
                player_in4[6:51] = card_hand_player
                env_state[
                    ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
                ] = player_in4
                # nếu thẻ được dùng nhiều lần
                if (
                    np.sum(token_fee_get[:4]) > 0
                    and np.sum(token_fee_get[:4] > player_in4[2:6]) == 0
                ):
                    env_state[
                        ENV_CARD_BUY_OR_USE
                    ] = id_card_use  # lưu trữ thẻ dùng gần nhất
                    env_state[
                        ENV_LAST_ACTION
                    ] = id_card_use  # lưu trữ action_main gần nhất
                    env_state[ENV_PHASE] = 3  # chuyển phase
                else:  # dùng 1 lần rồi bỏ
                    if (
                        np.sum(player_in4[2:BASIC_ATTRIBUTE_PLAYER]) > 10
                    ):  # nếu thừa nguyên liệu thì đi lược bỏ
                        env_state[ENV_TOKEN_NEED_DROP] = (
                            np.sum(player_in4[2:BASIC_ATTRIBUTE_PLAYER]) - 10
                        )
                        env_state[ENV_PHASE] = 4
                    else:
                        env_state[ENV_PHASE] = 1
                        env_state[ENV_ID_ACTION] = (
                            env_state[ENV_ID_ACTION] + 1
                        ) % NUMBER_PLAYER

    elif phase_env == 2:
        # lấy thông tin
        stay_drop = int(env_state[ENV_TOKEN_NEED_DROP]) - 1
        all_token_free = env_state[ENV_TOKEN_ON_ACTION_CARD:ENV_OPEN_POINT_CARD]
        token_drop = action - 57
        # Cập nhật thông tin
        player_in4[2:BASIC_ATTRIBUTE_PLAYER][token_drop] -= 1
        all_token_free[4 * stay_drop + token_drop] += 1
        env_state[ENV_TOKEN_NEED_DROP] -= 1

        if env_state[ENV_TOKEN_NEED_DROP] == 0:  # Hoàn tất đặt nguyên liệu thì lấy thẻ
            # lấy thông tin
            list_card_player = player_in4[BASIC_ATTRIBUTE_PLAYER:]
            list_card_board = env_state[ENV_DOWN_ACTION_CARD:ENV_DOWN_POINT_CARD]
            idx_card_buy = int(env_state[ENV_CARD_BUY_OR_USE])
            card_buy = int(list_card_board[idx_card_buy])
            token_free = np.zeros(NUMBBER_TYPE_TOKEN)
            if idx_card_buy != 5:
                token_free = all_token_free[
                    NUMBBER_TYPE_TOKEN
                    * idx_card_buy : NUMBBER_TYPE_TOKEN
                    * (idx_card_buy + 1)
                ]
                all_token_free = np.concatenate(
                    (
                        all_token_free[: NUMBBER_TYPE_TOKEN * idx_card_buy],
                        all_token_free[NUMBBER_TYPE_TOKEN * (idx_card_buy + 1) :],
                        np.zeros(NUMBBER_TYPE_TOKEN),
                    )
                )  # cập nhật giảm token free
            # cập nhật giá trị
            list_card_player[card_buy] = 1
            list_card_board[idx_card_buy:] = np.append(
                list_card_board[idx_card_buy + 1 :], -1
            )
            # top_6_card = card_in4[np.array(list_card_board[:6], dtype = int)].flatten()
            top_6_card = np.zeros((NUMBER_OPEN_ACTION_CARD, LENGTH_ACTION_CARD))
            for i in range(6):
                id = list_card_board[:6][i]
                top_6_card[i] = ALL_CARD_IN4[int(id)]

            # top_6_card = np.array([card_in4[int(id)] for id in list_card_board[:6]]).flatten()
            top_6_card = top_6_card.flatten()
            player_in4[
                BASIC_ATTRIBUTE_PLAYER:
            ] = list_card_player  # cập nhật thẻ mới mua
            player_in4[2:BASIC_ATTRIBUTE_PLAYER] = (
                player_in4[2:BASIC_ATTRIBUTE_PLAYER] + token_free
            )  # cập nhật token free nếu có
            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4
            env_state[
                ENV_TOKEN_ON_ACTION_CARD:ENV_OPEN_POINT_CARD
            ] = all_token_free  # cập nhật token free
            env_state[
                ENV_DOWN_ACTION_CARD:ENV_DOWN_POINT_CARD
            ] = list_card_board  # cập nhật danh sách thẻ trên bàn
            env_state[
                ENV_OPEN_ACTION_CARD:ENV_TOKEN_ON_ACTION_CARD
            ] = top_6_card  # cập nhật 6 thẻ người chơi có thể mua
            # Khôi phục các giá trị lưu trữ
            env_state[ENV_CARD_BUY_OR_USE] = -0.5
            # kiểm tra có phải trả tài nguyên ko
            if np.sum(player_in4[2:BASIC_ATTRIBUTE_PLAYER]) > 10:
                env_state[ENV_TOKEN_NEED_DROP] = (
                    np.sum(player_in4[2:BASIC_ATTRIBUTE_PLAYER]) - 10
                )
                env_state[ENV_PHASE] = 4
            else:
                # chuyển người chơi
                env_state[ENV_PHASE] = 1
                env_state[ENV_ID_ACTION] = (
                    env_state[ENV_ID_ACTION] + 1
                ) % NUMBER_PLAYER
        else:
            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4
            env_state[
                ENV_TOKEN_ON_ACTION_CARD:ENV_OPEN_POINT_CARD
            ] = all_token_free  # cập nhật token free

    elif phase_env == 3:
        if action == 61:  # nếu ko action tiếp
            env_state[ENV_CARD_BUY_OR_USE] = -0.5  # lưu trữ thẻ dùng gần nhất
            env_state[ENV_LAST_ACTION] = -1  # lưu trữ action_main gần nhất
            if (
                np.sum(player_in4[2:BASIC_ATTRIBUTE_PLAYER]) > 10
            ):  # nếu thừa nguyên liệu thì đi lược bỏ
                env_state[ENV_TOKEN_NEED_DROP] = (
                    np.sum(player_in4[2:BASIC_ATTRIBUTE_PLAYER]) - 10
                )
                env_state[ENV_PHASE] = 4
            else:
                env_state[ENV_PHASE] = 1
                env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 5
        else:
            # Lấy thông tin
            id_card_use = int(env_state[ENV_CARD_BUY_OR_USE])
            token_fee_get = ALL_CARD_IN4[id_card_use]
            # Cập nhật thông tin
            player_in4[2:BASIC_ATTRIBUTE_PLAYER] = (
                player_in4[2:BASIC_ATTRIBUTE_PLAYER]
                - token_fee_get[:NUMBBER_TYPE_TOKEN]
                + token_fee_get[NUMBBER_TYPE_TOKEN:ATTRIBUTE_ACTION_CARD]
            )
            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4
            if (
                np.sum(
                    token_fee_get[:NUMBBER_TYPE_TOKEN]
                    > player_in4[2:BASIC_ATTRIBUTE_PLAYER]
                )
                == 0
            ):
                # env_state[-7] = id_card_use     #lưu trữ thẻ dùng gần nhất
                # env_state[-3] = action          #lưu trữ action_main gần nhất
                env_state[ENV_PHASE] = 3  # chuyển phase
            else:
                env_state[ENV_CARD_BUY_OR_USE] = -0.5
                env_state[ENV_LAST_ACTION] = -1
                if np.sum(player_in4[2:6]) > 10:  # nếu thừa nguyên liệu thì đi lược bỏ
                    env_state[ENV_TOKEN_NEED_DROP] = np.sum(player_in4[2:6]) - 10
                    env_state[ENV_PHASE] = 4
                else:
                    env_state[ENV_PHASE] = 1
                    env_state[ENV_ID_ACTION] = (
                        env_state[ENV_ID_ACTION] + 1
                    ) % NUMBER_PLAYER

    elif phase_env == 4:
        # lấy thông tin
        # stay_drop = env_state[-6]
        token_drop = action - 57
        # Cập nhật thông tin
        player_in4[2:6][token_drop] -= 1
        env_state[
            ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
        ] = player_in4
        env_state[ENV_TOKEN_NEED_DROP] -= 1
        if env_state[ENV_TOKEN_NEED_DROP] == 0:
            env_state[ENV_PHASE] = 1
            env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER
        else:
            return env_state

    elif phase_env == 5:
        id_update = action - 62
        # if id_update == 3:
        #     env_state[-8] = 0
        #     env_state[-2] = 1
        #     env_state[-1] = (env_state[-1] + 1)%5
        # else:
        player_in4[2:6][id_update] -= 1
        player_in4[2:6][id_update + 1] += 1
        env_state[
            ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
        ] = player_in4
        env_state[ENV_NUMBER_ACTION_UPGRADE] -= 1
        if (
            env_state[ENV_NUMBER_ACTION_UPGRADE] == 0
            or np.sum(player_in4[2:5] > 0) == 0
        ):
            env_state[ENV_PHASE] = 1
            env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER

    return env_state


@njit()
def bot_lv0(state, perData):
    validActions = getValidActions(state)
    arr_action = np.where(validActions == 1)[0]
    idx = np.random.randint(0, arr_action.shape[0])
    return arr_action[idx], perData


@njit()
def one_game_numba(p0, list_other, per_player, per1, per2, per3, per4, p1, p2, p3, p4):
    env_state = initEnv()
    count_turn = 0
    while system_check_end(env_state) and count_turn < 2000:
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
        else:
            raise Exception("Sai list_other.")

        env_state = stepEnv(env_state, action)
        count_turn += 1

    env_state[ENV_CHECK_END] = 1

    for p_idx in range(5):
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
        else:
            raise Exception("Sai list_other.")
        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER

    winner = False
    if np.where(list_other == -1)[0] == check_winner(env_state):
        winner = True
    else:
        winner = False

    return winner, per_player


@njit()
def n_games_numba(
    p0, num_game, per_player, list_other, per1, per2, per3, per4, p1, p2, p3, p4
):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_numba(
            p0, list_other, per_player, per1, per2, per3, per4, p1, p2, p3, p4
        )
        win += winner
    return win, per_player


def one_game_normal(p0, list_other, per_player, per1, per2, per3, per4, p1, p2, p3, p4):
    env_state = initEnv()
    count_turn = 0
    while system_check_end(env_state) and count_turn < 2000:
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
        else:
            raise Exception("Sai list_other.")

        env_state = stepEnv(env_state, action)
        count_turn += 1

    env_state[ENV_CHECK_END] = 1

    for p_idx in range(5):
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
        else:
            raise Exception("Sai list_other.")
        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % NUMBER_PLAYER

    winner = False
    if np.where(list_other == -1)[0] == check_winner(env_state):
        winner = True
    else:
        winner = False

    return winner, per_player


def n_games_normal(
    p0, num_game, per_player, list_other, per1, per2, per3, per4, p1, p2, p3, p4
):
    win = 0
    for _n in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_normal(
            p0, list_other, per_player, per1, per2, per3, per4, p1, p2, p3, p4
        )
        win += winner
    return win, per_player


import importlib.util
import json, sys

import importlib.util
import json, sys

try:
    from env import SHORT_PATH
except:
    pass


def load_module_player(player):
    spec = importlib.util.spec_from_file_location(
        "Agent_player", f"{SHORT_PATH}src/Agent/{player}/Agent_player.py"
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
            _list_bot_level_[0],
            _list_bot_level_[1],
            _list_bot_level_[2],
            _list_bot_level_[3],
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
            _list_bot_level_[0],
            _list_bot_level_[1],
            _list_bot_level_[2],
            _list_bot_level_[3],
        )
