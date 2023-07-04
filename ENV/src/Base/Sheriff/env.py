import numpy as np
from numba import njit

from src.Base.Sheriff.docs.index import *

# from index import*


@njit()
def getActionSize():
    return 82


@njit()
def getAgentSize():
    return 4


@njit()
def getStateSize():
    return 478


@njit()
def system_check_end(env_state):
    if env_state[ENV_ROUND] <= 8:
        return True
    else:
        return False


@njit()
def initEnv():
    env_state = np.zeros(ENV_LENGTH)
    normal_card = NORMAL_CARD.copy()
    royal_card = ROYAL_CARD.copy()

    np.random.shuffle(normal_card)
    for id in range(NUMBER_PLAYER):
        player_i = START_PLAYER.copy()
        if id == 0:
            player_i[2] = 1
            card_player = player_i[-15:]
            for card in normal_card[id * 5 : (id + 1) * 5]:
                card_player[card - 1] += 1
            player_i[-15:] = card_player
            env_state[ENV_PLAYER_IN4:ATTRIBUTE_PLAYER] = player_i
        else:
            card_player = player_i[-15:]
            for card in normal_card[id * 5 : (id + 1) * 5]:
                card_player[card - 1] += 1
            player_i[-15:] = card_player
            env_state[ATTRIBUTE_PLAYER * id : ATTRIBUTE_PLAYER * (id + 1)] = player_i
    # print(env_state[ENV_PLAYER_IN4 : ATTRIBUTE_PLAYER*4])
    all_card = np.concatenate((normal_card[20:], royal_card))
    np.random.shuffle(all_card)
    env_state[ENV_LEFT_CARD : ENV_LEFT_CARD + CARD_OPEN_START] = all_card[:5]
    env_state[ENV_RIGHT_CARD : ENV_RIGHT_CARD + CARD_OPEN_START] = all_card[5:10]
    env_state[ENV_DOWN_CARD : ENV_DOWN_CARD + NUMBER_CARD_USE] = all_card[10:]
    env_state[ENV_ID_ACTION] = 1
    env_state[ENV_ROUND] = 1
    env_state[ENV_PHASE] = 1
    return env_state


@njit()
def getAgentState(env_state):
    player_action = int(env_state[ENV_ID_ACTION])
    player_state = np.zeros(P_LENGTH, dtype=np.float64)
    player_state[P_PLAYER_IN4:P_OTHER_PLAYER_IN4] = env_state[
        ATTRIBUTE_PLAYER * player_action : ATTRIBUTE_PLAYER * (player_action + 1)
    ]
    number_card_in_bag = np.zeros(NUMBER_PLAYER)
    number_card_in_bag[0] = np.sum(
        env_state[
            ATTRIBUTE_PLAYER * player_action : ATTRIBUTE_PLAYER * (player_action + 1)
        ][-45:-30]
    )

    # thông tin các người chơi khác
    for idx in range(1, NUMBER_PLAYER):
        id = (player_action + idx) % NUMBER_PLAYER
        all_other_player_in4 = env_state[
            ATTRIBUTE_PLAYER * id : ATTRIBUTE_PLAYER * (id + 1)
        ]
        other_player_in4 = all_other_player_in4[:OTHER_ATTRIBUTE_PLAYER]
        other_player_drop_card = env_state[
            ENV_TEMP_DROP
            + NUMBER_TYPE_CARD * id : ENV_TEMP_DROP
            + NUMBER_TYPE_CARD * (id + 1)
        ]
        player_state[
            P_OTHER_PLAYER_IN4
            + OTHER_ATTRIBUTE_PLAYER * (idx - 1) : P_OTHER_PLAYER_IN4
            + OTHER_ATTRIBUTE_PLAYER * idx
        ] = other_player_in4
        # thẻ người chơi bỏ trong túi
        number_card_in_bag[idx] = np.sum(all_other_player_in4[-45:-30])
        # thẻ người chơi khác đã bỏ
        player_state[
            P_CARD_OTHER_PLAYER_DROP
            + NUMBER_TYPE_CARD * (idx - 1) : P_CARD_OTHER_PLAYER_DROP
            + NUMBER_TYPE_CARD * idx
        ] = other_player_drop_card
        # thẻ người chơi có(chỉ hiện khiheets game)
        if env_state[ENV_CHECK_END] == 1:
            player_state[
                P_OTHER_PLAYER_DONE_CARD
                + NUMBER_TYPE_CARD * (idx - 1) : P_OTHER_PLAYER_DONE_CARD
                + NUMBER_TYPE_CARD * idx
            ] = all_other_player_in4[-75:-60]

    # 6 thẻ đầu ở 2 chồng bài
    # trái
    for id in range(MAX_CARD_TAKE):
        card_type_index = int(env_state[ENV_LEFT_CARD + id]) - 1
        if card_type_index >= 0:
            player_state[P_LEFT_CARD + NUMBER_TYPE_CARD * id + card_type_index] = 1
    # Phải
    for id in range(MAX_CARD_TAKE):
        card_type_index = int(env_state[ENV_RIGHT_CARD + id]) - 1
        if card_type_index >= 0:
            player_state[P_RIGHT_CARD + NUMBER_TYPE_CARD * id + card_type_index] = 1
    player_state[P_ROUND] = env_state[ENV_ROUND]
    player_state[P_CHECK_END] = env_state[ENV_CHECK_END]
    player_state[P_PHASE + int(env_state[ENV_PHASE] - 1)] = 1
    player_state[P_ORDER : P_ORDER + NUMBER_PLAYER] = np.concatenate(
        (ALL_ORDER[player_action:], ALL_ORDER[:player_action])
    )

    player_state[
        P_NUMBER_CARD_IN_BAG : P_NUMBER_CARD_IN_BAG + NUMBER_PLAYER
    ] = number_card_in_bag
    # print(player_action, number_card_in_bag)
    return player_state


@njit()
def getValidActions(player_state_origin):
    player_state = player_state_origin.copy()
    list_action_return = np.zeros(82, dtype=np.float64)
    phase_env = np.where(player_state[P_PHASE : P_PHASE + NUMBER_PHASE] == 1)[0][0] + 1
    player_state_own = player_state[:ATTRIBUTE_PLAYER]
    """
        OWNER: 100
            0: player coin
            1: debt
            2: is_police
            [3 : 7]: type_in_bag
            8: coin_bride
            9: number_smuggle
            10: number_card_bride_bag
            -90:-75: card_done_bride
            -75:60: card_done
            -60:-45: card_in_bag_bride
            -45:-30: card_in_bag
            -30:-15: card_bag_drop
            -15: card_hand
    """
    if phase_env == 1:
        list_action_return[15] = 1
        player_card = player_state_own[-15:]
        for id in range(15):
            if player_card[id] == 0:
                continue
            else:
                list_action_return[id] = 1

    elif phase_env == 2:
        # lấy thẻ từ các chồng bài: 16-bài rút, 17-lật trái, 18-lật phải
        list_action_return[16] = 1
        if np.sum(player_state[P_LEFT_CARD : P_LEFT_CARD + NUMBER_TYPE_CARD]) > 0:
            list_action_return[17] = 1
        if np.sum(player_state[P_RIGHT_CARD : P_RIGHT_CARD + NUMBER_TYPE_CARD]) > 0:
            list_action_return[18] = 1

    elif phase_env == 3:
        # trả thẻ bỏ vào  chồng bài lật: 19: lật trái, 20: lật phải
        list_action_return[np.array([19, 20])] = 1

    elif phase_env == 4:
        # lựa chọn dừng bỏ thẻ vào túi đi chợ
        list_action_return[36] = 1
        player_card = player_state_own[-15:]
        for id in range(15):
            if player_card[id] == 0:
                continue
            else:
                list_action_return[id + 21] = 1
        if np.sum(player_state_own[-45:-30]) == 0:
            list_action_return[36] = 0

    elif phase_env == 5:
        # chọn 1 trong 4 loại hàng chính ngạch
        list_action_return[np.array([37, 38, 39, 40])] = 1

    elif phase_env == 6:
        # 41, 42, 43 lần lượt là check người ở vị trí 1,2,3 sau mình
        for id in range(3):
            # kiểm tra xem đã check hay chưa bằng cách xem mặt hàng khai báo
            type_bag_other_player = player_state[
                P_OTHER_PLAYER_IN4
                + OTHER_ATTRIBUTE_PLAYER * id : P_OTHER_PLAYER_IN4
                + OTHER_ATTRIBUTE_PLAYER * (id + 1)
            ][P_TYPE_IN_BAG:P_COIN_BRIBE]
            if np.sum(type_bag_other_player) != 0:
                list_action_return[41 + id] = 1

    elif phase_env == 7:
        # 44 là ko hối lộ nữa, 45 là thêm coin
        if player_state_own[P_COIN] - player_state_own[P_DEBT] > 0:
            list_action_return[np.array([44, 45])] = 1
        else:
            list_action_return[44] = 1

    elif phase_env == 8:
        list_action_return[61] = 1  # action dừng hối lộ thẻ done
        player_card_done = player_state_own[-75:-60]
        for id in range(15):
            if player_card_done[id] == 0:
                continue
            else:
                # bắt đầu duyệt từ việc hối lộ thẻ táo ứng vs action 46
                list_action_return[id + 46] = 1

    elif phase_env == 9:
        list_action_return[77] = 1  # action dừng hối lộ thẻ trong túi
        player_card_bag = player_state_own[-45:-30]
        for id in range(15):
            if player_card_bag[id] == 0:
                continue
            else:
                list_action_return[id + 62] = 1

    elif phase_env == 10:
        # 79 là có check hàng, 80 là cho thoát
        list_action_return[np.array([78, 79])] = 1

    elif phase_env == 11:
        list_action_return[np.array([80, 81])] = 1

    return list_action_return


@njit()
def stepEnv(env_state, action):
    phase_env = env_state[ENV_PHASE]
    id_action = int(env_state[ENV_ID_ACTION])
    player_in4 = env_state[
        ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
    ]

    if phase_env == 1:
        if action != 15:
            player_card = player_in4[-15:]
            temp_drop_card = env_state[
                ENV_TEMP_DROP
                + NUMBER_TYPE_CARD * id_action : ENV_TEMP_DROP
                + NUMBER_TYPE_CARD * (id_action + 1)
            ]
            player_card[action] -= 1
            temp_drop_card[action] += 1
            player_in4[-15:] = player_card
            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4
            env_state[
                ENV_TEMP_DROP
                + NUMBER_TYPE_CARD * id_action : ENV_TEMP_DROP
                + NUMBER_TYPE_CARD * (id_action + 1)
            ] = temp_drop_card
            if np.sum(player_card) == 0:
                env_state[ENV_PHASE] = 2
        else:
            env_state[ENV_PHASE] = 2

    elif phase_env == 2:
        if action == 16:
            player_card = player_in4[-15:]
            # print('check', player_card)
            card_down = env_state[ENV_DOWN_CARD : ENV_DOWN_CARD + NUMBER_CARD_USE]
            number_card_get = int(MAX_CARD_TAKE - np.sum(player_card))
            card_get = card_down[:number_card_get]
            if len(card_get) > 0:
                for card in card_get:
                    player_card[int(card) - 1] += 1
            card_down = np.concatenate(
                (card_down[number_card_get:], np.zeros(number_card_get))
            )
            env_state[ENV_DOWN_CARD : ENV_DOWN_CARD + NUMBER_CARD_USE] = card_down
            player_in4[-15:] = player_card
            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4

            if (
                np.sum(
                    env_state[
                        ENV_TEMP_DROP
                        + NUMBER_TYPE_CARD * id_action : ENV_TEMP_DROP
                        + NUMBER_TYPE_CARD * (id_action + 1)
                    ]
                )
                != 0
            ):
                env_state[ENV_PHASE] = 3
            else:
                # print('chuyển người chơi luôn vì không bỏ thẻ')
                env_state[ENV_PHASE] = 1
                env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
                if env_state[ENV_ID_ACTION] == (env_state[ENV_ROUND] - 1) % 4:
                    env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
                    env_state[ENV_PHASE] = 4

        else:
            player_card = player_in4[-15:]
            card_up = env_state[
                ENV_LEFT_CARD
                + NUMBER_CARD_OPEN * (action - 17) : ENV_LEFT_CARD
                + NUMBER_CARD_OPEN * (action - 16)
            ]
            card_get = card_up[0]
            player_card[int(card_get) - 1] += 1
            card_up = np.append(card_up[1:], 0)
            env_state[
                ENV_LEFT_CARD
                + NUMBER_CARD_OPEN * (action - 17) : ENV_LEFT_CARD
                + NUMBER_CARD_OPEN * (action - 16)
            ] = card_up
            player_in4[-15:] = player_card
            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4
            if np.sum(player_card) == 6:
                if (
                    np.sum(
                        env_state[
                            ENV_TEMP_DROP
                            + NUMBER_TYPE_CARD * id_action : ENV_TEMP_DROP
                            + NUMBER_TYPE_CARD * (id_action + 1)
                        ]
                    )
                    != 0
                ):
                    env_state[ENV_PHASE] = 3
                else:
                    # print('chuyển người chơi luôn vì không bỏ thẻ')
                    env_state[ENV_PHASE] = 1
                    env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
                    if env_state[ENV_ID_ACTION] == (env_state[ENV_ROUND] - 1) % 4:
                        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
                        env_state[ENV_PHASE] = 4

    elif phase_env == 3:
        temp_card_drop = env_state[
            ENV_TEMP_DROP
            + NUMBER_TYPE_CARD * id_action : ENV_TEMP_DROP
            + NUMBER_TYPE_CARD * (id_action + 1)
        ]
        card_up = env_state[
            ENV_LEFT_CARD
            + NUMBER_CARD_OPEN * (action - 19) : ENV_LEFT_CARD
            + NUMBER_CARD_OPEN * (action - 18)
        ]
        for i in range(len(temp_card_drop)):
            if temp_card_drop[i] > 0:
                card_up = np.append(np.array([i + 1] * int(temp_card_drop[i])), card_up)
        if np.sum(temp_card_drop) > 0:
            card_up = card_up[: int(-np.sum(temp_card_drop))]

        # env_state[ENV_TEMP_DROP + NUMBER_TYPE_CARD * id_action : ENV_TEMP_DROP + NUMBER_TYPE_CARD * (id_action + 1)] = np.zeros(NUMBER_TYPE_CARD)
        env_state[
            ENV_LEFT_CARD
            + NUMBER_CARD_OPEN * (action - 19) : ENV_LEFT_CARD
            + NUMBER_CARD_OPEN * (action - 18)
        ] = card_up
        env_state[ENV_PHASE] = 1
        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
        if env_state[ENV_ID_ACTION] == (env_state[ENV_ROUND] - 1) % 4:
            env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
            env_state[ENV_PHASE] = 4

    elif phase_env == 4:
        card_bag = player_in4[-45:-30]
        player_card = player_in4[-15:]
        if action == 36:
            if np.sum(card_bag) == 0:
                print(env_state)
                raise Exception("chưa bỏ thẻ vào túi")
            else:
                env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
                if env_state[ENV_ID_ACTION] == (env_state[ENV_ROUND] - 1) % 4:
                    env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
                    env_state[ENV_PHASE] = 5
        else:
            card_bag[action - 21] += 1
            player_card[action - 21] -= 1
            player_in4[-15:] = player_card
            player_in4[-45:-30] = card_bag
            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4
            if np.sum(card_bag) == 5:
                env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
                if env_state[ENV_ID_ACTION] == (env_state[ENV_ROUND] - 1) % 4:
                    env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
                    env_state[ENV_PHASE] = 5

    elif phase_env == 5:
        type_bag = action - 36
        player_in4[P_TYPE_IN_BAG:P_COIN_BRIBE][int(type_bag - 1)] = 1
        env_state[
            ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
        ] = player_in4
        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
        if env_state[ENV_ID_ACTION] == (env_state[ENV_ROUND] - 1) % 4:
            env_state[ENV_PHASE] = 6

    elif phase_env == 6:
        player_checked = action - 40
        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + player_checked) % 4
        env_state[ENV_NUMBER_CHECKED] += 1
        env_state[ENV_PHASE] = 7
        env_state[ENV_LAST_CHECKED + int(env_state[ENV_ID_ACTION])] = 1

    elif phase_env == 7:
        if action == 45:
            player_in4[0] -= 1
            # new: chỉnh vị trí coin hối lộ
            player_in4[P_COIN_BRIBE] += 1
            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4
        else:
            env_state[ENV_PHASE] = 8

    elif phase_env == 8:
        if action == 61:
            env_state[ENV_PHASE] = 9
        else:
            all_card_done = player_in4[-75:-60]
            all_card_bride = player_in4[-90:-75]
            card_bride = action - 46
            all_card_done[card_bride] -= 1
            all_card_bride[card_bride] += 1
            player_in4[-75:-60] = all_card_done
            player_in4[-90:-75] = all_card_bride
            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4

    elif phase_env == 9:
        if action == 77:
            env_state[ENV_PHASE] = 10
            env_state[ENV_ID_ACTION] = (env_state[ENV_ROUND] - 1) % 4
        else:
            # print(player_in4[-90:-60])
            all_card_bag = player_in4[-45:-30]
            all_card_bride_bag = player_in4[-60:-45]
            card_bride = action - 62
            all_card_bag[card_bride] -= 1
            all_card_bride_bag[card_bride] += 1
            # cập nhập số thẻ trong túi dùng hối lộ
            player_in4[P_NUMBER_CARD_BRIBE_BAG] += 1
            player_in4[-45:-30] = all_card_bag
            player_in4[-60:-45] = all_card_bride_bag
            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4

    elif phase_env == 10:
        id_checked = np.where(
            env_state[ENV_LAST_CHECKED : ENV_LAST_CHECKED + NUMBER_PLAYER] == 1
        )[0][0]
        env_state[ATTRIBUTE_PLAYER * id_checked : ATTRIBUTE_PLAYER * (id_checked + 1)]
        player_checked = env_state[
            ATTRIBUTE_PLAYER * id_checked : ATTRIBUTE_PLAYER * (id_checked + 1)
        ]
        type_bag = np.where(player_checked[P_TYPE_IN_BAG:P_COIN_BRIBE] == 1)[0][0] + 1
        if action == 78:
            # có check
            # B1: khôi phục coin
            player_checked[P_COIN] += player_checked[P_COIN_BRIBE]
            player_checked[P_COIN_BRIBE] = 0
            # B3 khôi phục thẻ done
            player_checked[-75:-60] = player_checked[-75:-60] + player_checked[-90:-75]
            player_checked[-90:-75] = np.zeros(15)
            # khôi phục thẻ trong túi
            player_checked[-45:-30] = player_checked[-45:-30] + player_checked[-60:-45]
            player_checked[-60:-45] = np.zeros(15)
            # tính toán cho sheriff
            player_checked_card_bag = player_checked[-45:-30]
            player_checked_card_done = player_checked[-75:-60]
            card_bag_drop = player_in4[-30:-15]
            # cập nhật coin
            if player_checked_card_bag[int(type_bag) - 1] == np.sum(
                player_checked_card_bag
            ):
                penalty = (
                    -player_checked_card_bag[int(type_bag) - 1]
                    * ALL_PENALTY[int(type_bag) - 1]
                )
            else:
                penalty = (
                    np.sum(np.multiply(player_checked_card_bag, ALL_PENALTY))
                    - player_checked_card_bag[int(type_bag) - 1]
                    * ALL_PENALTY[int(type_bag) - 1]
                )
            player_in4[P_COIN] += penalty
            player_checked[P_COIN] -= penalty
            # Kiểm tra xem có đủ tiền ko để ghi nợ
            if player_in4[P_COIN] < 0:
                player_in4[P_DEBT] -= player_in4[P_COIN]
                player_in4[P_COIN] = 0
            if player_checked[P_COIN] < 0:
                player_checked[P_DEBT] -= player_checked[P_COIN]
                player_checked[P_COIN] = 0
            # cập nhật thẻ
            for id in range(15):
                if id == type_bag - 1:
                    player_checked_card_done[id] += player_checked_card_bag[id]
                else:
                    card_bag_drop[id] += player_checked_card_bag[id]
            player_checked[-45:-30] = np.zeros(15)
            player_checked[-75:-60] = player_checked_card_done

            player_checked[P_NUMBER_CARD_BRIBE_BAG] = 0  # số thẻ trong túi dùng hối lộ
            player_checked[
                P_TYPE_IN_BAG:P_COIN_BRIBE
            ] = 0  # loại thẻ người chơi khai báo
            player_checked[P_NUMBER_SMUGGLE] = np.sum(
                player_checked[-75:-60][4:]
            )  # số thẻ buôn lậu thành công của người chơi
            env_state[
                ATTRIBUTE_PLAYER * id_checked : ATTRIBUTE_PLAYER * (id_checked + 1)
            ] = player_checked
            player_in4[-30:-15] = card_bag_drop
            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4
            # cập nhật trong hệ thống
            if np.sum(card_bag_drop) > 0:
                env_state[ENV_PHASE] = 11
                env_state[ENV_ID_ACTION] = (env_state[ENV_ROUND] - 1) % 4
            else:
                env_state[ENV_LAST_CHECKED : ENV_LAST_CHECKED + NUMBER_PLAYER] = 0
                if env_state[ENV_NUMBER_CHECKED] == 3:
                    old_sheriff = int((env_state[ENV_ROUND] - 1) % 4)
                    env_state[
                        ATTRIBUTE_PLAYER * old_sheriff + 2
                    ] = 0  # hủy tư cách sheriff
                    env_state[ENV_PHASE] = 1
                    env_state[ENV_ID_ACTION] = (env_state[ENV_ROUND]) % 4

                    env_state[ENV_ROUND] += 1
                    new_sheriff = int((env_state[ENV_ROUND] - 1) % 4)
                    env_state[
                        ATTRIBUTE_PLAYER * new_sheriff + 2
                    ] = 1  # gán tư cách sheriff

                    if env_state[ENV_ID_ACTION] == (env_state[ENV_ROUND] - 1) % 4:
                        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
                    env_state[ENV_NUMBER_CHECKED] = 0
                else:
                    env_state[ENV_PHASE] = 6
                    env_state[ENV_ID_ACTION] = (env_state[ENV_ROUND] - 1) % 4
        else:
            # không check
            env_state[ENV_LAST_CHECKED : ENV_LAST_CHECKED + NUMBER_PLAYER] = 0
            # B1: Cập nhật coin
            player_in4[P_COIN] += player_checked[P_COIN_BRIBE]
            player_checked[P_COIN_BRIBE] = 0
            # B3: cập nhật thẻ done sheriff and player_checked
            player_in4[-75:-60] = player_in4[-75:-60] + player_checked[-90:-75]
            player_checked[-90:-75] = np.zeros(NUMBER_TYPE_CARD)
            # cập nhật thẻ done từ hối lộ thẻ trong túi và cập nhật thẻ trong túi
            player_in4[-75:-60] = player_in4[-75:-60] + player_checked[-60:-45]
            player_checked[-60:-45] = np.zeros(15)
            # cập nhật thẻ done player_checked
            player_checked[-75:-60] = player_checked[-75:-60] + player_checked[-45:-30]
            player_checked[-45:-30] = np.zeros(NUMBER_TYPE_CARD)
            # cập nhật full thông tin
            player_checked[P_NUMBER_CARD_BRIBE_BAG] = 0
            player_checked[P_TYPE_IN_BAG:P_COIN_BRIBE] = 0
            player_checked[P_NUMBER_SMUGGLE] = np.sum(player_checked[-75:-60][4:])
            env_state[
                ATTRIBUTE_PLAYER * id_checked : ATTRIBUTE_PLAYER * (id_checked + 1)
            ] = player_checked
            env_state[
                ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
            ] = player_in4
            # cập nhật hệ thống lượt chơi
            if env_state[ENV_NUMBER_CHECKED] == 3:
                env_state[
                    ENV_TEMP_DROP : ENV_TEMP_DROP + NUMBER_TYPE_CARD * NUMBER_PLAYER
                ] = 0  # reset thẻ drop trong lượt
                env_state[ENV_PHASE] = 1

                old_sheriff = int((env_state[ENV_ROUND] - 1) % 4)
                env_state[ATTRIBUTE_PLAYER * old_sheriff + 2] = 0  # hủy tư cách sheriff

                env_state[ENV_ID_ACTION] = (env_state[ENV_ROUND]) % 4
                env_state[ENV_ROUND] += 1

                new_sheriff = int((env_state[ENV_ROUND] - 1) % 4)
                env_state[ATTRIBUTE_PLAYER * new_sheriff + 2] = 1  # gán tư cách sheriff

                if env_state[ENV_ID_ACTION] == (env_state[ENV_ROUND] - 1) % 4:
                    env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
                env_state[ENV_NUMBER_CHECKED] = 0
            else:
                env_state[ENV_PHASE] = 6
                env_state[ENV_ID_ACTION] = (env_state[ENV_ROUND] - 1) % 4

    elif phase_env == 11:
        temp_card_drop = player_in4[-30:-15]
        card_up = env_state[
            ENV_LEFT_CARD
            + NUMBER_CARD_OPEN * (action - 80) : ENV_LEFT_CARD
            + NUMBER_CARD_OPEN * (action - 79)
        ]
        for i in range(len(temp_card_drop)):
            if temp_card_drop[i] > 0:
                card_up = np.append(np.array([i + 1] * int(temp_card_drop[i])), card_up)
        # if np.sum(temp_card_drop) > 0:
        card_up = card_up[: int(-np.sum(temp_card_drop))]
        temp_card_drop = np.zeros(NUMBER_TYPE_CARD)
        player_in4[-30:-15] = temp_card_drop
        env_state[
            ENV_LEFT_CARD
            + NUMBER_CARD_OPEN * (action - 80) : ENV_LEFT_CARD
            + NUMBER_CARD_OPEN * (action - 79)
        ] = card_up
        env_state[
            ATTRIBUTE_PLAYER * id_action : ATTRIBUTE_PLAYER * (id_action + 1)
        ] = player_in4
        env_state[ENV_LAST_CHECKED : ENV_LAST_CHECKED + NUMBER_PLAYER] = 0
        if env_state[ENV_NUMBER_CHECKED] == 3:
            env_state[ENV_PHASE] = 1
            # print('check phase 11')

            old_sheriff = int((env_state[ENV_ROUND] - 1) % 4)
            env_state[ATTRIBUTE_PLAYER * old_sheriff + 2] = 0  # hủy tư cách sheriff

            env_state[ENV_ID_ACTION] = (env_state[ENV_ROUND]) % 4
            env_state[ENV_ROUND] += 1

            new_sheriff = int((env_state[ENV_ROUND] - 1) % 4)
            env_state[ATTRIBUTE_PLAYER * new_sheriff + 2] = 1  # gán tư cách sheriff

            if env_state[ENV_ID_ACTION] == (env_state[ENV_ROUND] - 1) % 4:
                env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4
            env_state[ENV_NUMBER_CHECKED] = 0
        else:
            env_state[ENV_PHASE] = 6
            env_state[ENV_ID_ACTION] = (env_state[ENV_ROUND] - 1) % 4
    return env_state


@njit()
def check_winner(env_state):
    # mảng đếm số loại tài nguyên chính ngạch của 4 người chơi
    all_number_type_card = np.zeros(16)
    # mảng đếm hàng hóa buôn thành công của người chơi
    all_done_card = np.zeros(60)
    all_player_coin = np.array(
        [
            env_state[ATTRIBUTE_PLAYER * id + P_COIN]
            - env_state[ATTRIBUTE_PLAYER * id + P_DEBT]
            for id in range(4)
        ]
    )

    for id_player in range(4):
        player_i = env_state[
            ATTRIBUTE_PLAYER * id_player : ATTRIBUTE_PLAYER * (id_player + 1)
        ]
        player_i_done = player_i[-75:-60]  # thẻ hàng hóa buôn thành công của người chơi
        all_done_card[15 * id_player : 15 * (id_player + 1)] = player_i_done
        all_player_coin[id_player] = all_player_coin[id_player] + np.sum(
            np.multiply(player_i_done, ALL_REWARD)
        )
        all_done_card_i = player_i_done * ALL_NUMBER_COUNT
        # ghi nhận từng loại tài nguyên chính ngạch của người chơi
        all_number_type_card[id_player] = (
            all_done_card_i[0] + all_done_card_i[8] + all_done_card_i[12]
        )
        all_number_type_card[id_player + 4] = (
            all_done_card_i[1] + all_done_card_i[9] + all_done_card_i[13]
        )
        all_number_type_card[id_player + 8] = (
            all_done_card_i[2] + all_done_card_i[10] + all_done_card_i[14]
        )
        all_number_type_card[id_player + 12] = all_done_card_i[3] + all_done_card_i[11]
    reward_King = np.array([20, 15, 15, 10])
    reward_Queen = np.array([10, 10, 10, 5])
    for type_i in range(4):
        count_type = all_number_type_card[4 * type_i : 4 * (type_i + 1)]
        top1 = np.max(count_type)
        if top1 == 0:
            continue
        count_top1 = len(np.where(count_type == top1)[0])
        if count_top1 > 1:
            reward_i = (reward_King[type_i] + reward_Queen[type_i]) // count_top1
            all_player_coin = np.where(
                count_type == top1, all_player_coin + reward_i, all_player_coin
            )
        else:
            reward_i = reward_King[type_i]
            all_player_coin = np.where(
                count_type == top1, all_player_coin + reward_i, all_player_coin
            )
            count_type = np.where(count_type == top1, 0, count_type)
            top2 = np.max(count_type)
            if top2 == 0:
                continue
            else:
                count_top2 = len(np.where(count_type == top2)[0])
                if count_top2 > 1:
                    reward_i = reward_Queen[type_i] // count_top2
                    all_player_coin = np.where(
                        count_type == top2, all_player_coin + reward_i, all_player_coin
                    )
                else:
                    reward_i = reward_Queen[type_i]
                    all_player_coin = np.where(
                        count_type == top2, all_player_coin + reward_i, all_player_coin
                    )

    all_player_coin_int = np.array(
        [int(all_player_coin[i]) for i in range(len(all_player_coin))]
    )
    players_compare = np.zeros(NUMBER_PLAYER)
    temp_win = np.max(all_player_coin_int)
    number_win = np.where(all_player_coin_int == temp_win)[0]
    players_compare[number_win] = 1
    if len(number_win) == 1:
        return number_win[0]
    else:
        all_legal = np.zeros(4)
        for id in number_win:
            all_legal[id] = np.sum(all_done_card[15 * id : 15 * (id + 1)][:4])
        all_legal = all_legal * players_compare
        temp_win = np.max(all_legal)
        number_win = np.where(all_legal == temp_win)[0]
        if len(number_win) == 1:
            return number_win[0]
        else:
            players_compare = np.where(all_legal == temp_win, 1, 0)
            all_unlegal = np.zeros(4)
            for id in number_win:
                all_unlegal[id] = np.sum(all_done_card[15 * id : 15 * (id + 1)][4:])
            all_unlegal = players_compare * all_unlegal
            temp_win = np.max(all_unlegal)
            number_win = np.where(all_unlegal == temp_win)[0]
            if len(number_win) == 1:
                return number_win[0]
            else:
                all_player_coin = all_player_coin + np.array([0, 0.1, 0.2, 0.3])
                players_compare = np.where(all_unlegal == temp_win, 1, 0)
                number_win = np.where(players_compare == 1)[0]
                all_player_coin_last = np.zeros(4)
                all_player_coin_last[number_win] = all_player_coin[number_win]
                return np.argmax(all_player_coin_last)


@njit()
def getReward(player_state):
    check_end = player_state[P_CHECK_END]
    if check_end == 0:
        return -1
    elif check_end == 1 and player_state[P_ROUND] > 8:
        all_player_coin = np.array(
            [player_state[P_COIN] - player_state[P_DEBT]]
            + [
                (
                    player_state[
                        P_OTHER_PLAYER_IN4 + OTHER_ATTRIBUTE_PLAYER * id + P_COIN
                    ]
                    - player_state[
                        P_OTHER_PLAYER_IN4 + OTHER_ATTRIBUTE_PLAYER * id + P_DEBT
                    ]
                )
                for id in range(3)
            ]
        )
        all_player_order = player_state[P_ORDER : P_ORDER + NUMBER_PLAYER]
        all_player_coin = all_player_coin + all_player_order / 10
        all_number_type_card = np.zeros(16)
        all_done_card = np.zeros(60)
        for id_player in range(4):
            player_i_done = np.zeros(NUMBER_TYPE_CARD)
            if id_player == 0:
                player_i = player_state[:P_OTHER_PLAYER_IN4]
                player_i_done = player_i[
                    -75:-60
                ]  # thẻ hàng hóa buôn thành công của người chơi

            else:
                player_i_done = player_state[
                    P_OTHER_PLAYER_DONE_CARD
                    + NUMBER_TYPE_CARD * (id_player - 1) : P_OTHER_PLAYER_DONE_CARD
                    + NUMBER_TYPE_CARD * id_player
                ]
            all_done_card[15 * id_player : 15 * (id_player + 1)] = player_i_done
            all_player_coin[id_player] = all_player_coin[id_player] + np.sum(
                np.multiply(player_i_done, ALL_REWARD)
            )
            all_done_card_i = player_i_done * ALL_NUMBER_COUNT
            # ghi nhận từng loại tài nguyên chính ngạch của người chơi
            all_number_type_card[id_player] = (
                all_done_card_i[0] + all_done_card_i[8] + all_done_card_i[12]
            )
            all_number_type_card[id_player + 4] = (
                all_done_card_i[1] + all_done_card_i[9] + all_done_card_i[13]
            )
            all_number_type_card[id_player + 8] = (
                all_done_card_i[2] + all_done_card_i[10] + all_done_card_i[14]
            )
            all_number_type_card[id_player + 12] = (
                all_done_card_i[3] + all_done_card_i[11]
            )
        reward_King = np.array([20, 15, 15, 10])
        reward_Queen = np.array([10, 10, 10, 5])
        for type_i in range(4):
            count_type = all_number_type_card[4 * type_i : 4 * (type_i + 1)]
            top1 = np.max(count_type)
            if top1 == 0:
                continue
            count_top1 = len(np.where(count_type == top1)[0])
            if count_top1 > 1:
                reward_i = (reward_King[type_i] + reward_Queen[type_i]) // count_top1
                all_player_coin = np.where(
                    count_type == top1, all_player_coin + reward_i, all_player_coin
                )
            else:
                reward_i = reward_King[type_i]
                all_player_coin = np.where(
                    count_type == top1, all_player_coin + reward_i, all_player_coin
                )
                count_type = np.where(count_type == top1, 0, count_type)
                top2 = np.max(count_type)
                if top2 == 0:
                    continue
                else:
                    count_top2 = len(np.where(count_type == top2)[0])
                    if count_top2 > 1:
                        reward_i = reward_Queen[type_i] // count_top2
                        all_player_coin = np.where(
                            count_type == top2,
                            all_player_coin + reward_i,
                            all_player_coin,
                        )
                    else:
                        reward_i = reward_Queen[type_i]
                        all_player_coin = np.where(
                            count_type == top2,
                            all_player_coin + reward_i,
                            all_player_coin,
                        )

        all_player_coin_int = np.array(
            [int(all_player_coin[i]) for i in range(len(all_player_coin))]
        )
        players_compare = np.zeros(NUMBER_PLAYER)
        temp_win = np.max(all_player_coin_int)
        number_win = np.where(all_player_coin_int == temp_win)[0]
        players_compare[number_win] = 1
        if 0 not in number_win:
            return 0
        else:
            if len(number_win) == 1:
                return 1
            else:
                all_legal = np.zeros(4)
                for id in number_win:
                    all_legal[id] = np.sum(all_done_card[15 * id : 15 * (id + 1)][:4])
                all_legal = all_legal * players_compare
                temp_win = np.max(all_legal)
                number_win = np.where(all_legal == temp_win)[0]
                if 0 not in number_win:
                    return 0
                else:
                    if len(number_win) == 1:
                        return 1
                    else:
                        players_compare = np.where(all_legal == temp_win, 1, 0)
                        all_unlegal = np.zeros(4)
                        for id in number_win:
                            all_unlegal[id] = np.sum(
                                all_done_card[15 * id : 15 * (id + 1)][4:]
                            )
                        all_unlegal = players_compare * all_unlegal
                        temp_win = np.max(all_unlegal)
                        number_win = np.where(all_unlegal == temp_win)[0]
                        if 0 not in number_win:
                            return 0
                        else:
                            if len(number_win) == 1:
                                return 1
                            else:
                                players_compare = np.where(
                                    all_unlegal == temp_win, 1, 0
                                )
                                number_win = np.where(players_compare == 1)[0]
                                all_player_coin_last = np.zeros(4)
                                all_player_coin_last[number_win] = all_player_coin[
                                    number_win
                                ]
                                if np.argmax(all_player_coin_last) == 0:
                                    return 1
                                else:
                                    return 0
    else:
        return -1


@njit()
def bot_lv0(state, perData):
    validActions = getValidActions(state)
    arr_action = np.where(validActions == 1)[0]
    idx = np.random.randint(0, arr_action.shape[0])
    return arr_action[idx], perData


@njit()
def one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env_state = initEnv()
    count_turn = 0
    while system_check_end(env_state) and count_turn < 1000:
        # police_check = env_state[2:305:100]
        # print(police_check)
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
        else:
            raise Exception("Sai list_other.")
        env_state = stepEnv(env_state, action)
        count_turn += 1

    env_state[ENV_CHECK_END] = 1
    win = check_winner(env_state)
    for p_idx in range(4):
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
        else:
            raise Exception("Sai list_other.")

        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4

    winner = False
    if np.where(list_other == -1)[0] == win:
        winner = True
    else:
        winner = False
    return winner, per_player


def one_game_normal(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env_state = initEnv()
    count_turn = 0
    while system_check_end(env_state) and count_turn < 1000:
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
        else:
            raise Exception("Sai list_other.")
        env_state = stepEnv(env_state, action)
        count_turn += 1

    env_state[ENV_CHECK_END] = 1

    for p_idx in range(4):
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
        else:
            raise Exception("Sai list_other.")

        env_state[ENV_ID_ACTION] = (env_state[ENV_ID_ACTION] + 1) % 4

    winner = False
    if np.where(list_other == -1)[0] == check_winner(env_state):
        winner = True
    else:
        winner = False
    return winner, per_player


@njit()
def n_games_numba(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_numba(
            p0, list_other, per_player, per1, per2, per3, p1, p2, p3
        )
        win += winner

    return win, per_player


def n_games_normal(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_normal(
            p0, list_other, per_player, per1, per2, per3, p1, p2, p3
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


@njit()
def check_run_under_njit(agent, perData):
    return True


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
            _list_bot_level_[0],
            _list_bot_level_[1],
            _list_bot_level_[2],
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
            _list_bot_level_[0],
            _list_bot_level_[1],
            _list_bot_level_[2],
        )
