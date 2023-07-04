import numpy as np
from numba import njit

from src.Base.TicketToRide.docs.index import *


@njit()
def getAgentSize():
    return NUMBER_PLAYER


#########################################################


@njit()
def getActionSize():
    return NUMBER_ACTIONS


#########################################################


@njit()
def getStateSize():
    return P_LENGTH


#########################################################


@njit()
def initEnv():
    # tạo env_state full 0
    env_state = np.zeros(ENV_LENGTH)
    # các đường vô chủ gán là -1
    env_state[ENV_ROAD_BOARD:ENV_ROUTE_CARD_BOARD] = -1
    # các thẻ route đã bị người chơi sở hữu gán là -1
    env_state[ENV_TRAIN_CAR_CARD:ENV_IN4_PLAYER] = -1

    all_special_route_card = np.arange(
        NUMBER_ROUTE - NUMBER_SPECIAL_ROUTE, NUMBER_ROUTE
    )
    # chọn 5 thẻ special route cho người chơi
    special_route_card = np.random.choice(
        all_special_route_card, NUMBER_PLAYER, replace=False
    )

    all_normal_route_card = np.arange(NUMBER_ROUTE - NUMBER_SPECIAL_ROUTE)
    np.random.shuffle(all_normal_route_card)
    normal_route_card = all_normal_route_card[: NUMBER_PLAYER * NUMBER_ROUTE_GET]
    env_state[ENV_ROUTE_CARD_BOARD:ENV_TRAIN_CAR_CARD] = np.concatenate(
        (
            all_normal_route_card[NUMBER_PLAYER * NUMBER_ROUTE_GET :],
            np.full(NUMBER_SPECIAL_ROUTE + NUMBER_PLAYER * NUMBER_ROUTE_GET, -1),
        )
    )
    all_train_car_card = np.concatenate(
        (
            np.zeros(14),
            np.ones(12),
            np.full(12, 2),
            np.full(12, 3),
            np.full(12, 4),
            np.full(12, 5),
            np.full(12, 6),
            np.full(12, 7),
            np.full(12, 8),
        )
    )
    # các thẻ xe lửa
    np.random.shuffle(all_train_car_card)
    train_car_card = all_train_car_card[
        : NUMBER_PLAYER * NUMBER_TRAIN_CAR_GET
    ]  # các thẻ xe lửa cho người chơi
    # các thẻ xe lửa trong chồng bài úp
    env_state[
        ENV_TRAIN_CAR_CARD : ENV_TRAIN_CAR_CARD
        + NUMBER_TRAIN_CAR_CARD
        - NUMBER_PLAYER * NUMBER_TRAIN_CAR_GET
        - NUMBER_TRAIN_CAR_CARD_OPEN
    ] = all_train_car_card[
        NUMBER_PLAYER * NUMBER_TRAIN_CAR_GET + NUMBER_TRAIN_CAR_CARD_OPEN :
    ]

    for id_pl in range(NUMBER_PLAYER):
        special_route_id = special_route_card[id_pl]
        normal_route_id = normal_route_card[
            id_pl * NUMBER_ROUTE_GET : NUMBER_ROUTE_GET * (id_pl + 1)
        ]
        train_car_card_id = train_car_card[
            id_pl * NUMBER_TRAIN_CAR_GET : NUMBER_TRAIN_CAR_GET * (id_pl + 1)
        ]

        score_train_car = np.array(
            [0, 0, NUMBER_TRAIN]
        )  # điểm, điểm trừ, số toa tàu còn lại
        train_card_id = np.zeros(9)
        for id_train in train_car_card_id:
            train_card_id[int(id_train)] += 1
        route_card_id = np.zeros(NUMBER_ROUTE)
        route_card_id[special_route_id] = 1
        route_card_id[normal_route_id] = 1
        env_state[
            ENV_IN4_PLAYER
            + ATTRIBUTE_PLAYER * id_pl : ENV_IN4_PLAYER
            + ATTRIBUTE_PLAYER * (id_pl + 1)
        ] = np.concatenate((score_train_car, train_card_id, route_card_id))
    # các thẻ xe lửa được mở
    env_state[
        ENV_TRAIN_CAR_OPEN : ENV_TRAIN_CAR_OPEN + NUMBER_TRAIN_CAR_CARD_OPEN
    ] = all_train_car_card[
        NUMBER_PLAYER * NUMBER_TRAIN_CAR_GET : NUMBER_PLAYER * NUMBER_TRAIN_CAR_GET
        + NUMBER_TRAIN_CAR_CARD_OPEN
    ]
    env_state[ENV_ROUTE_CARD_GET : ENV_ROUTE_CARD_GET + NUMBER_ROUTE_RECEIVE] = -1
    # Other in4
    env_state[ENV_PHASE] = 3  # đầu game thì đi loại thẻ route
    env_state[ENV_CHECK_END] = 0
    env_state[ENV_ID_PLAYER_END] = -1
    env_state[ENV_ROAD_BUILT] = -1
    env_state[ENV_TURN] = 1
    id_action = int(env_state[ENV_ID_ACTION])
    env_state[ENV_ROUTE_CARD_GET:ENV_TRAIN_CAR_DROP] = env_state[
        ENV_IN4_PLAYER:ENV_TRAIN_CAR_OPEN
    ][id_action * ATTRIBUTE_PLAYER : (id_action + 1) * ATTRIBUTE_PLAYER][
        ATT_NUMBER_TYPE_TRAIN_CARD + NUMBER_TYPE_TRAIN_CAR_CARD :
    ]
    env_state[ENV_IN4_PLAYER:ENV_TRAIN_CAR_OPEN][
        id_action * ATTRIBUTE_PLAYER : (id_action + 1) * ATTRIBUTE_PLAYER
    ][ATT_NUMBER_TYPE_TRAIN_CARD + NUMBER_TYPE_TRAIN_CAR_CARD :] = 0
    env_state[ENV_PHASE] = 3

    return env_state


#########################################################


@njit()
def getAgentState(env_state):
    player_state = np.zeros(P_LENGTH, dtype=np.float64)
    id_action = int(env_state[ENV_ID_ACTION])
    # điểm tất cả người chơi
    all_player_score = env_state[
        ENV_IN4_PLAYER + ATT_SCORE : ENV_TRAIN_CAR_OPEN : ATTRIBUTE_PLAYER
    ]

    player_state[P_SCORE:P_NEG_SCORE] = np.concatenate(
        (all_player_score[id_action:], all_player_score[:id_action])
    )
    # số thẻ train_car_card của người chơi
    player_state[P_TRAIN_CAR_CARD:P_PLAYER_ROAD] = env_state[
        ENV_IN4_PLAYER:ENV_TRAIN_CAR_OPEN
    ][id_action * ATTRIBUTE_PLAYER : (id_action + 1) * ATTRIBUTE_PLAYER][
        ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
        + NUMBER_TYPE_TRAIN_CAR_CARD
    ]
    # số tàu người chơi còn
    player_state[P_NUMBER_TRAIN] = env_state[ENV_IN4_PLAYER:ENV_TRAIN_CAR_OPEN][
        id_action * ATTRIBUTE_PLAYER + ATT_NUMBER_TRAIN
    ]
    # đường của các người chơi
    for id_ in range(NUMBER_PLAYER):
        list_road_id_ = np.where(env_state[ENV_ROAD_BOARD:ENV_ROUTE_CARD_BOARD] == id_)[
            0
        ]
        idx = (id_ - id_action) % NUMBER_PLAYER
        player_state[P_PLAYER_ROAD:P_ROUTE_CARD][
            idx * NUMBER_ROAD : (idx + 1) * NUMBER_ROAD
        ][list_road_id_] = 1
    # thẻ route của người chơi
    player_route_card = np.zeros(NUMBER_ROUTE)
    player_route_card[
        np.where(
            env_state[ENV_IN4_PLAYER:ENV_TRAIN_CAR_OPEN][
                id_action * ATTRIBUTE_PLAYER : (id_action + 1) * ATTRIBUTE_PLAYER
            ][ATT_NUMBER_TYPE_TRAIN_CARD + NUMBER_TYPE_TRAIN_CAR_CARD :]
            > 0
        )[0]
    ] = 1

    player_state[P_ROUTE_CARD:P_ROUTE_GET] = player_route_card
    # route card get
    route_card_get = env_state[ENV_ROUTE_CARD_GET:ENV_TRAIN_CAR_DROP].astype(np.int64)
    player_state[P_ROUTE_GET:P_TRAIN_CAR_CARD_BOARD] = route_card_get
    # thẻ train_car trên bàn chơi
    for type_car in env_state[ENV_TRAIN_CAR_OPEN:ENV_ROUTE_CARD_GET]:
        if type_car != -1:
            player_state[P_TRAIN_CAR_CARD_BOARD:P_CARD_BULD_TUNNEL][int(type_car)] += 1
    player_state[P_CARD_BULD_TUNNEL:P_CARD_TEST_TUNNEL] = env_state[
        ENV_CARD_BULD_TUNNEL:ENV_CARD_TEST_TUNNEL
    ]
    player_state[P_ID_ACTION] = 0  # dấu hiệu đánh giá mình hành động
    player_state[P_PHASE + int(env_state[ENV_PHASE]) - 1] = 1  # edit 13h 4/1/2023
    player_state[P_CHECK_ROUTE_CARD] = int(env_state[ENV_ROUTE_CARD_BOARD] > -1)
    player_state[P_CHEKC_END] = int(env_state[ENV_CHECK_END] / 2)
    if player_state[P_CHEKC_END] == 1:
        # điểm sẽ trừ của tất cả người chơi
        all_player_neg_score = env_state[
            ENV_IN4_PLAYER + ATT_NEG_SCORE : ENV_TRAIN_CAR_OPEN : ATTRIBUTE_PLAYER
        ]
        player_state[P_NEG_SCORE:P_TRAIN_CAR_CARD] = np.concatenate(
            (all_player_neg_score[id_action:], all_player_neg_score[:id_action])
        )
        all_check_most_route = env_state[ENV_CHECK_MOST_ROUTE:ENV_CHECK_LONGEST_ROAD]
        all_check_longest_route = env_state[
            ENV_CHECK_LONGEST_ROAD : ENV_CHECK_LONGEST_ROAD + NUMBER_PLAYER
        ]
        player_state[P_CHECK_MOST_ROUTE:P_CHECK_LONGEST_ROAD] = np.concatenate(
            (all_check_most_route[id_action:], all_check_most_route[:id_action])
        )
        player_state[
            P_CHECK_LONGEST_ROAD : P_CHECK_LONGEST_ROAD + NUMBER_PLAYER
        ] = np.concatenate(
            (all_check_longest_route[id_action:], all_check_longest_route[:id_action])
        )

    if env_state[ENV_ID_PLAYER_END] != -1:
        player_state[
            P_ID_PLAYER_END
            + int((env_state[ENV_ID_PLAYER_END] - id_action) % NUMBER_PLAYER)
        ] = 1
    player_state[P_NUMBER_TRAIN_CAR_GET] = env_state[ENV_NUMBER_TRAIN_CAR_GET]
    player_state[P_NUMBER_DROP_ROUTE_CARD] = env_state[ENV_NUMBER_DROP_ROUTE_CARD]
    player_state[P_CARD_TEST_TUNNEL:P_TYPE_TRAIN_CAR_BUILD_ROAD] = env_state[
        ENV_CARD_TEST_TUNNEL : ENV_CARD_TEST_TUNNEL + NUMBER_TYPE_TRAIN_CAR_CARD
    ]
    player_state[
        P_TYPE_TRAIN_CAR_BUILD_ROAD : P_TYPE_TRAIN_CAR_BUILD_ROAD
        + NUMBER_TYPE_TRAIN_CAR_CARD
    ] = env_state[
        ENV_TYPE_TRAIN_CAR_BUILD_ROAD : ENV_TYPE_TRAIN_CAR_BUILD_ROAD
        + NUMBER_TYPE_TRAIN_CAR_CARD
    ]
    if (
        env_state[ENV_TRAIN_CAR_CARD] != -1
        and np.min(env_state[ENV_TRAIN_CAR_OPEN:ENV_ROUTE_CARD_GET]) > -1
    ):
        player_state[P_ACTION_GET_TRAIN_CAR_DOWN] = 1
    return player_state


#########################################################


@njit()
def getReward(player_state):
    if player_state[P_CHEKC_END] == 1:
        list_score = (
            player_state[P_SCORE:P_NEG_SCORE]
            - player_state[P_NEG_SCORE:P_TRAIN_CAR_CARD]
        )
        check_highest_score = np.zeros(NUMBER_PLAYER)
        if np.argmax(list_score) == 0:
            list_highest_score = np.where(list_score == np.max(list_score))[0]
            if len(list_highest_score) == 1:
                return 1
            else:
                check_highest_score[list_highest_score] = 1
                check_most_route = (
                    player_state[P_CHECK_MOST_ROUTE:P_CHECK_LONGEST_ROAD]
                    * check_highest_score
                )
                if np.argmax(check_most_route) == 0:
                    if np.sum(check_most_route) == 1:
                        return 1
                    else:
                        check_longest_road = player_state[
                            P_CHECK_LONGEST_ROAD : P_CHECK_LONGEST_ROAD + NUMBER_PLAYER
                        ]

                        if np.sum(check_most_route) == 0:
                            check_longest_road = (
                                check_longest_road * check_highest_score
                            )
                        else:
                            check_longest_road = check_longest_road * check_most_route

                        if check_longest_road[0] == 1:
                            return 1
                        else:
                            if np.sum(check_longest_road) > 0:
                                return 0
                            else:
                                return 1
                else:
                    return 0
        else:
            return 0
    else:
        return -1


#########################################################


@njit()
def check_winner(env_state):
    # B1: tìm người chơi con đường dài nhất
    list_longest_path = np.zeros(NUMBER_PLAYER)
    all_road = env_state[ENV_ROAD_BOARD:ENV_ROUTE_CARD_BOARD]
    number_route_comp = np.zeros(NUMBER_PLAYER)
    for player in range(NUMBER_PLAYER):
        p_road = np.where(all_road == player)[0]
        all_player_route_card = np.where(
            env_state[
                ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * player : ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * (player + 1)
            ][ATT_NUMBER_TYPE_TRAIN_CARD + NUMBER_TYPE_TRAIN_CAR_CARD :]
            == 1
        )[0]
        # tính toán điểm của thẻ route
        for route_card in all_player_route_card:
            point_source = LIST_ALL_ROUTE_POINT[route_card][0]
            point_dest = LIST_ALL_ROUTE_POINT[route_card][1]
            check_done = check_done_route_card(p_road, point_source, point_dest)
            if check_done == 1:
                number_route_comp[player] += 1
                env_state[
                    ENV_IN4_PLAYER + ATTRIBUTE_PLAYER * player + ATT_SCORE
                ] += LIST_ALL_SCORE_ROUTE[route_card]
            else:
                env_state[
                    ENV_IN4_PLAYER + ATTRIBUTE_PLAYER * player + ATT_NEG_SCORE
                ] += LIST_ALL_SCORE_ROUTE[route_card]

        list_longest_path[player] = calculator_longest_road(p_road)
    # xác định đường dài nhất
    max_road = np.max(list_longest_path)
    player_longest_path = np.where(list_longest_path == max_road)[0]
    # gán đánh dấu người chơi có con đường dài nhất
    env_state[ENV_CHECK_LONGEST_ROAD : ENV_CHECK_LONGEST_ROAD + NUMBER_PLAYER][
        player_longest_path
    ] = 1
    # cộng 10đ cho những người có đường dài nhất
    for player in player_longest_path:
        env_state[ENV_IN4_PLAYER + ATTRIBUTE_PLAYER * player + ATT_SCORE] += 10
    # cập nhật số đường người chơi hoàn thành
    env_state[ENV_CHECK_MOST_ROUTE:ENV_CHECK_LONGEST_ROAD] = number_route_comp
    # xét người chiến thắng
    all_player_score = env_state[
        ENV_IN4_PLAYER : ENV_TRAIN_CAR_OPEN - 1 : ATTRIBUTE_PLAYER
    ]
    all_player_neg_score = env_state[
        ENV_IN4_PLAYER + ATT_NEG_SCORE : ENV_TRAIN_CAR_OPEN - 1 : ATTRIBUTE_PLAYER
    ]
    all_player_score = all_player_score - all_player_neg_score
    score_max = np.max(all_player_score)
    winner_1 = np.where(all_player_score == score_max)[0]
    if len(winner_1) == 1:
        return winner_1, env_state
    else:
        number_route_winner_comp = np.full(NUMBER_PLAYER, -1)
        number_route_winner_comp[winner_1] = number_route_comp[winner_1]
        route_max = np.max(number_route_winner_comp)
        winner_2 = np.where(number_route_winner_comp == route_max)[0]
        #  env_state[ENV_CHECK_MOST_ROUTE : ENV_CHECK_LONGEST_ROAD][winner_2] = 1
        if len(winner_2) == 1:
            #  player_win = winner_2[0]
            return winner_2, env_state
        else:
            winner_3 = np.array([-1])
            for player in winner_2:
                if player in player_longest_path:
                    winner_3 = np.append(winner_3, player)
            if len(winner_3) == 1:
                return winner_2, env_state
            else:
                winner_3 = winner_3[1:]
                #  env_state[ENV_CHECK_LONGEST_ROAD : ENV_CHECK_LONGEST_ROAD + NUMBER_PLAYER][winner_3] = 1
                return winner_3, env_state


#########################################################


@njit()
def system_check_end(env_state):
    if (
        env_state[ENV_CHECK_END] == 2
        and env_state[ENV_ID_PLAYER_END] == env_state[ENV_ID_ACTION]
    ):
        return False
    else:
        return True


#########################################################


@njit()
def getValidActions(player_state_origin):
    player_state = player_state_origin.copy()
    phase_game = np.where(player_state[P_PHASE : P_PHASE + NUMBER_PHASE] == 1)[0][0] + 1
    list_action_return = np.zeros(NUMBER_ACTIONS, dtype=np.float64)
    if phase_game == 1:
        # kiểm tra các hành động nhặt train_car làm được
        train_car_board = player_state[P_TRAIN_CAR_CARD_BOARD:P_CARD_BULD_TUNNEL]
        if player_state[P_ACTION_GET_TRAIN_CAR_DOWN] == 1:
            list_action_return[156] = 1  # action nhặt thẻ úp
        if player_state[P_NUMBER_TRAIN_CAR_GET] == 0:
            # kiểm tra nhặt được thẻ ko
            if player_state[P_CHECK_ROUTE_CARD] != 0:
                list_action_return[168] = 1
            # kiểm tra các đường xây được
            road_can_build = check_road_can_build(player_state)
            list_action_return[road_can_build] = 1
            train_car_can_get = np.where(train_car_board > 0)[0] + 147
            list_action_return[train_car_can_get] = 1
        else:  #  player_state[P_NUMBER_TRAIN_CAR_GET] != 0:
            # nếu đã nhặt thẻ train_car thì k được nhặt thẻ locomotive
            train_car_board[0] = 0
            train_car_can_get = np.where(train_car_board > 0)[0] + 147
            list_action_return[train_car_can_get] = 1
    elif phase_game == 2:
        # chọn tài nguyên để xây đường
        """
        các kiểu dùng tài nguyên để xây con đường muốn xây được lưu ở player_state, từ đó xác định các cách xây có thể
        """
        type_train_car_build_road = np.where(
            player_state[
                P_TYPE_TRAIN_CAR_BUILD_ROAD : P_TYPE_TRAIN_CAR_BUILD_ROAD
                + NUMBER_TYPE_TRAIN_CAR_CARD
            ]
            > 0
        )[0]
        list_action_return[type_train_car_build_road + 157] = 1

    elif phase_game == 3:
        route_card_can_drop = np.where(
            player_state[P_ROUTE_GET:P_TRAIN_CAR_CARD_BOARD] == 1
        )[0]
        if player_state[P_NUMBER_DROP_ROUTE_CARD] < 2:
            list_action_return[route_card_can_drop + NUMBER_ROAD] = 1
        list_action_return[169] = 1  # dừng drop thẻ route

    elif phase_game == 4:
        card_build_tunnel = np.where(
            player_state[P_CARD_BULD_TUNNEL:P_CARD_TEST_TUNNEL] > 0
        )[0]
        card_test_tunnel = player_state[P_CARD_TEST_TUNNEL:P_TYPE_TRAIN_CAR_BUILD_ROAD]
        player_train_car = player_state[P_TRAIN_CAR_CARD:P_PLAYER_ROAD]
        check = 1
        test_train_car = 0
        for type_car in card_build_tunnel:
            if card_test_tunnel[type_car] > 0:
                if type_car == 0:
                    test_train_car -= card_test_tunnel[type_car]
                else:
                    test_train_car -= max(
                        card_test_tunnel[type_car] - player_train_car[type_car], 0
                    )
            else:
                continue
        if player_train_car[0] + test_train_car < 0:
            check = 0
        list_action_return[166] = check
        list_action_return[167] = 1
    if np.sum(list_action_return) == 0:
        list_action_return[-1] = 1

    return list_action_return


#########################################################


@njit()
def stepEnv(env_state, action):
    phase_game = env_state[ENV_PHASE]
    id_action = int(env_state[ENV_ID_ACTION])

    if phase_game == 1:
        if action < NUMBER_ROAD:  # nếu action là xây đường, xử lí xây đường
            road = action
            player_train_car = env_state[
                ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * (id_action + 1)
            ][
                ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                + NUMBER_TYPE_TRAIN_CAR_CARD
            ]
            temp_train_car = player_train_car.copy()
            train_car_else = np.where(temp_train_car[1:] > 0)[0] + 1
            temp_train_car[train_car_else] += temp_train_car[0]
            # nếu đường ko cần đầu máy
            if LIST_ALL_TYPE_ROAD[road] != 2:
                road_color = LIST_ALL_COLOR_ROAD[road]
                # nếu là đường màu xám
                if road_color == -1:
                    type_build = np.where(temp_train_car >= LIST_ALL_LENGTH_ROAD[road])[
                        0
                    ]
                    env_state[
                        ENV_TYPE_TRAIN_CAR_BUILD_ROAD : ENV_TYPE_TRAIN_CAR_BUILD_ROAD
                        + NUMBER_TYPE_TRAIN_CAR_CARD
                    ][type_build] = 1
                else:
                    if player_train_car[0] >= LIST_ALL_LENGTH_ROAD[road]:
                        env_state[
                            ENV_TYPE_TRAIN_CAR_BUILD_ROAD : ENV_TYPE_TRAIN_CAR_BUILD_ROAD
                            + NUMBER_TYPE_TRAIN_CAR_CARD
                        ][0] = 1
                    env_state[
                        ENV_TYPE_TRAIN_CAR_BUILD_ROAD : ENV_TYPE_TRAIN_CAR_BUILD_ROAD
                        + NUMBER_TYPE_TRAIN_CAR_CARD
                    ][road_color] = 1

            # nếu là ferry cần locomotive
            else:
                type_build = np.where(temp_train_car >= LIST_ALL_LENGTH_ROAD[road])[0]
                env_state[
                    ENV_TYPE_TRAIN_CAR_BUILD_ROAD : ENV_TYPE_TRAIN_CAR_BUILD_ROAD
                    + NUMBER_TYPE_TRAIN_CAR_CARD
                ][type_build] = 1
            env_state[ENV_PHASE] = 2
            env_state[ENV_ROAD_BUILT] = action
        # nếu nhặt thẻ locomotive thì update thông tin bàn chơi rồi chuyển người
        elif action == 147:
            # cập nhật các thẻ mở

            new_train_car = env_state[ENV_TRAIN_CAR_CARD]
            index_drop = np.where(
                env_state[ENV_TRAIN_CAR_OPEN:ENV_ROUTE_CARD_GET] == 0
            )[0][0]
            env_state[ENV_TRAIN_CAR_OPEN:ENV_ROUTE_CARD_GET][index_drop] = new_train_car
            env_state[ENV_TRAIN_CAR_CARD:ENV_IN4_PLAYER] = np.concatenate(
                (env_state[ENV_TRAIN_CAR_CARD:ENV_IN4_PLAYER][1:], np.array([-1]))
            )

            env_state[
                ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * (id_action + 1)
            ][action - 147 + ATT_NUMBER_TYPE_TRAIN_CARD] += 1
            env_state[ENV_ID_ACTION] = (id_action + 1) % NUMBER_PLAYER
            if env_state[ENV_CHECK_END] >= 1:
                if env_state[ENV_ID_ACTION] == env_state[ENV_ID_PLAYER_END]:
                    env_state[ENV_CHECK_END] += 0.5  # edit by Hieu 05012023
            env_state = process_train_car_board(env_state)
            # xáo thẻ drop nếu chồng bài úp hết bài
            if env_state[ENV_TRAIN_CAR_CARD] == -1 and (
                np.sum(env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL])
                or env_state[ENV_TRAIN_CAR_CARD] != -1
            ):
                env_state = shuffle_drop_card(env_state)
            # Lật thêm thẻ ra khi thẻ mở đang không đủ 5 thẻ
            if (
                np.min(
                    env_state[
                        ENV_TRAIN_CAR_OPEN : ENV_TRAIN_CAR_OPEN
                        + NUMBER_TRAIN_CAR_CARD_OPEN
                    ]
                )
                == -1
                and env_state[ENV_TRAIN_CAR_CARD] != -1
            ):
                env_state = shuffle_drop_card(env_state)
                env_state = process_train_car_board(env_state)

        # nếu nhặt thẻ thường
        elif action < 156:
            type_car_get = action - 147
            new_train_car = env_state[ENV_TRAIN_CAR_CARD]
            index_drop = np.where(
                env_state[ENV_TRAIN_CAR_OPEN:ENV_ROUTE_CARD_GET] == type_car_get
            )[0][0]
            env_state[ENV_TRAIN_CAR_OPEN:ENV_ROUTE_CARD_GET][index_drop] = new_train_car
            env_state[ENV_TRAIN_CAR_CARD:ENV_IN4_PLAYER] = np.concatenate(
                (env_state[ENV_TRAIN_CAR_CARD:ENV_IN4_PLAYER][1:], np.array([-1]))
            )
            env_state[
                ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * (id_action + 1)
            ][type_car_get + ATT_NUMBER_TYPE_TRAIN_CARD] += 1
            env_state = process_train_car_board(env_state)

            # xáo thẻ drop nếu chồng bài úp hết bài
            if env_state[ENV_TRAIN_CAR_CARD] == -1 and (
                np.sum(env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL])
                or env_state[ENV_TRAIN_CAR_CARD] != -1
            ):
                env_state = shuffle_drop_card(env_state)
            # Lật thêm thẻ ra khi thẻ mở đang không đủ 5 thẻ
            if (
                np.min(
                    env_state[
                        ENV_TRAIN_CAR_OPEN : ENV_TRAIN_CAR_OPEN
                        + NUMBER_TRAIN_CAR_CARD_OPEN
                    ]
                )
                == -1
                and env_state[ENV_TRAIN_CAR_CARD] != -1
            ):
                env_state = shuffle_drop_card(env_state)
                env_state = process_train_car_board(env_state)
            env_state[ENV_NUMBER_TRAIN_CAR_GET] += 1
            # nếu đã nhặt đủ 2 thẻ
            if env_state[ENV_NUMBER_TRAIN_CAR_GET] == 2:
                # reset số thẻ đã nhặt, cập nhật người chơi mơi
                env_state[ENV_NUMBER_TRAIN_CAR_GET] = 0
                env_state[ENV_ID_ACTION] = (id_action + 1) % NUMBER_PLAYER
                if env_state[ENV_CHECK_END] >= 1:
                    if env_state[ENV_ID_ACTION] == env_state[ENV_ID_PLAYER_END]:
                        env_state[ENV_CHECK_END] += 0.5  # edit by Hieu 05012023
        # nếu nhặt thẻ từ chồng train_Car úp
        elif action == 156:
            new_train_car = int(env_state[ENV_TRAIN_CAR_CARD])
            #  print('màu của thẻ úp được nhặt là ', LIST_COLOR[int(new_train_car)])
            env_state[
                ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * (id_action + 1)
            ][new_train_car + ATT_NUMBER_TYPE_TRAIN_CARD] += 1
            env_state[ENV_TRAIN_CAR_CARD:ENV_IN4_PLAYER] = np.concatenate(
                (env_state[ENV_TRAIN_CAR_CARD:ENV_IN4_PLAYER][1:], np.array([-1]))
            )
            env_state[ENV_NUMBER_TRAIN_CAR_GET] += 1
            # xáo thẻ drop nếu chồng bài úp hết bài
            if env_state[ENV_TRAIN_CAR_CARD] == -1 and np.sum(
                env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL]
            ):
                env_state = shuffle_drop_card(env_state)
            # nếu đã nhặt đủ 2 thẻ
            if env_state[ENV_NUMBER_TRAIN_CAR_GET] == 2:
                # reset số thẻ đã nhặt, cập nhật người chơi mơi
                env_state[ENV_NUMBER_TRAIN_CAR_GET] = 0
                env_state[ENV_ID_ACTION] = (id_action + 1) % NUMBER_PLAYER
                if env_state[ENV_CHECK_END] >= 1:
                    if env_state[ENV_ID_ACTION] == env_state[ENV_ID_PLAYER_END]:
                        env_state[ENV_CHECK_END] += 0.5  # edit by Hieu 05012023
        # nếu nhặt thẻ routecard thì cập nhật rồi sang phase 3
        elif action == 168:
            route_card_get = env_state[
                ENV_ROUTE_CARD_BOARD : ENV_ROUTE_CARD_BOARD + NUMBER_ROUTE_GET
            ]
            route_card_get = route_card_get[route_card_get > -1].astype(np.int64)
            env_state[ENV_ROUTE_CARD_GET:ENV_TRAIN_CAR_DROP][route_card_get] = 1
            env_state[ENV_ROUTE_CARD_BOARD:ENV_TRAIN_CAR_CARD] = np.concatenate(
                (
                    env_state[ENV_ROUTE_CARD_BOARD:ENV_TRAIN_CAR_CARD][
                        NUMBER_ROUTE_GET:
                    ],
                    np.array([-1] * NUMBER_ROUTE_GET),
                )
            )
            env_state[ENV_PHASE] = 3
            if len(route_card_get) == 3:
                env_state[ENV_NUMBER_DROP_ROUTE_CARD] = 0
            else:
                env_state[ENV_NUMBER_DROP_ROUTE_CARD] = 2
        # nếu ko action được gì khác thì skip
        else:
            # action == 170
            # skip qua người chơi
            env_state[ENV_NUMBER_TRAIN_CAR_GET] = 0
            env_state[ENV_ID_ACTION] = (id_action + 1) % NUMBER_PLAYER
            if env_state[ENV_CHECK_END] >= 1:
                if env_state[ENV_ID_ACTION] == env_state[ENV_ID_PLAYER_END]:
                    env_state[ENV_CHECK_END] += 0.5  # edit by Hieu 05012023

    elif phase_game == 2:
        type_train_car_use = action - 157
        road = int(env_state[ENV_ROAD_BUILT])
        type_road = LIST_ALL_TYPE_ROAD[road]
        player_train_car = env_state[
            ENV_IN4_PLAYER
            + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
            + ATTRIBUTE_PLAYER * (id_action + 1)
        ][
            ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
            + NUMBER_TYPE_TRAIN_CAR_CARD
        ]
        # nếu xây road thường
        if type_road == 0:
            if player_train_car[type_train_car_use] >= LIST_ALL_LENGTH_ROAD[road]:
                # bỏ thẻ vào chồng bài drop
                env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL][
                    type_train_car_use
                ] += LIST_ALL_LENGTH_ROAD[road]
                # đủ train_Car nên chỉ trừ train_Car
                env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][
                    ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                    + NUMBER_TYPE_TRAIN_CAR_CARD
                ][
                    type_train_car_use
                ] -= LIST_ALL_LENGTH_ROAD[
                    road
                ]
            else:
                # nếu k đủ traincar thì tối thiểu việc dùng locomotive
                locomotive_use = (
                    LIST_ALL_LENGTH_ROAD[road] - player_train_car[type_train_car_use]
                )
                # bỏ thẻ vào chồng bài drop
                env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL][
                    type_train_car_use
                ] += env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][
                    ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                    + NUMBER_TYPE_TRAIN_CAR_CARD
                ][
                    type_train_car_use
                ]
                env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL][
                    type_train_car_use - type_train_car_use
                ] += locomotive_use
                # trừ tài nguyên người chơi
                env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][
                    ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                    + NUMBER_TYPE_TRAIN_CAR_CARD
                ][
                    type_train_car_use - type_train_car_use
                ] -= locomotive_use
                env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][
                    ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                    + NUMBER_TYPE_TRAIN_CAR_CARD
                ][
                    type_train_car_use
                ] = 0

            # cộng điểm cho người chơi, trừ số tàu của người chơi
            env_state[
                ENV_IN4_PLAYER + ATTRIBUTE_PLAYER * id_action + ATT_SCORE
            ] += LIST_SCORE_BUILD_ROAD[
                LIST_ALL_LENGTH_ROAD[int(env_state[ENV_ROAD_BUILT])]
            ]
            env_state[
                ENV_IN4_PLAYER + ATTRIBUTE_PLAYER * id_action + ATT_NUMBER_TRAIN
            ] -= LIST_ALL_LENGTH_ROAD[int(env_state[ENV_ROAD_BUILT])]
            if (
                env_state[
                    ENV_IN4_PLAYER + ATTRIBUTE_PLAYER * id_action + ATT_NUMBER_TRAIN
                ]
                <= 2
                and env_state[ENV_CHECK_END] == 0
            ):
                env_state[ENV_CHECK_END] = 1
                env_state[ENV_ID_PLAYER_END] = (id_action + 1) % NUMBER_PLAYER
            # update đường của người chơi
            env_state[road] = id_action
            env_state[ENV_ROAD_BUILT] = -1
            env_state[
                ENV_TYPE_TRAIN_CAR_BUILD_ROAD : ENV_TYPE_TRAIN_CAR_BUILD_ROAD
                + NUMBER_TYPE_TRAIN_CAR_CARD
            ] = 0
            env_state[ENV_PHASE] = 1
            env_state[ENV_ID_ACTION] = (id_action + 1) % NUMBER_PLAYER
            if env_state[ENV_CHECK_END] >= 1:
                if env_state[ENV_ID_ACTION] == env_state[ENV_ID_PLAYER_END]:
                    env_state[ENV_CHECK_END] += 0.5  # edit by Hieu 05012023
            if (
                np.min(env_state[ENV_TRAIN_CAR_OPEN:ENV_ROUTE_CARD_GET]) == -1
                or env_state[ENV_TRAIN_CAR_CARD] == -1
            ) and np.sum(env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL]) > 0:
                env_state = shuffle_drop_card(env_state)
            # Lật thêm thẻ ra khi thẻ mở đang không đủ 5 thẻ
            if (
                np.min(
                    env_state[
                        ENV_TRAIN_CAR_OPEN : ENV_TRAIN_CAR_OPEN
                        + NUMBER_TRAIN_CAR_CARD_OPEN
                    ]
                )
                == -1
                and env_state[ENV_TRAIN_CAR_CARD] != -1
            ):
                env_state = shuffle_drop_card(env_state)
                env_state = process_train_car_board(env_state)
        # nếu xây ferry
        elif type_road == 2:
            temp_train_car = player_train_car.copy()
            temp_train_car[1:] += temp_train_car[0]
            locomotive_need_use = LIST_ROAD_LOCOMOTIVES[road]
            # player_train_car
            if player_train_car[type_train_car_use] >= LIST_ALL_LENGTH_ROAD[road]:
                # nếu đủ train_Car thì chỉ trừ train_Car
                env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][
                    ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                    + NUMBER_TYPE_TRAIN_CAR_CARD
                ][
                    type_train_car_use - type_train_car_use
                ] -= locomotive_need_use
                env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][
                    ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                    + NUMBER_TYPE_TRAIN_CAR_CARD
                ][
                    type_train_car_use
                ] -= (
                    LIST_ALL_LENGTH_ROAD[road] - locomotive_need_use
                )
                env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL][
                    type_train_car_use - type_train_car_use
                ] += locomotive_need_use
                env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL][
                    type_train_car_use
                ] += (LIST_ALL_LENGTH_ROAD[road] - locomotive_need_use)

            else:
                locomotive_need_use += (
                    LIST_ALL_LENGTH_ROAD[road]
                    - locomotive_need_use
                    - player_train_car[type_train_car_use]
                )
                env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][
                    ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                    + NUMBER_TYPE_TRAIN_CAR_CARD
                ][
                    type_train_car_use - type_train_car_use
                ] -= locomotive_need_use
                env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][
                    ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                    + NUMBER_TYPE_TRAIN_CAR_CARD
                ][
                    type_train_car_use
                ] -= (
                    LIST_ALL_LENGTH_ROAD[road] - locomotive_need_use
                )
                env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL][
                    type_train_car_use - type_train_car_use
                ] += locomotive_need_use
                env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL][
                    type_train_car_use
                ] += (LIST_ALL_LENGTH_ROAD[road] - locomotive_need_use)
            env_state[
                ENV_IN4_PLAYER + ATTRIBUTE_PLAYER * id_action + ATT_SCORE
            ] += LIST_SCORE_BUILD_ROAD[
                LIST_ALL_LENGTH_ROAD[int(env_state[ENV_ROAD_BUILT])]
            ]
            env_state[
                ENV_IN4_PLAYER + ATTRIBUTE_PLAYER * id_action + ATT_NUMBER_TRAIN
            ] -= LIST_ALL_LENGTH_ROAD[int(env_state[ENV_ROAD_BUILT])]
            if (
                env_state[
                    ENV_IN4_PLAYER + ATTRIBUTE_PLAYER * id_action + ATT_NUMBER_TRAIN
                ]
                <= 2
            ):
                env_state[ENV_CHECK_END] = 1
                env_state[ENV_ID_PLAYER_END] = (id_action + 1) % NUMBER_PLAYER
            # update đường của người chơi
            env_state[road] = id_action
            env_state[ENV_ROAD_BUILT] = -1
            env_state[
                ENV_TYPE_TRAIN_CAR_BUILD_ROAD : ENV_TYPE_TRAIN_CAR_BUILD_ROAD
                + NUMBER_TYPE_TRAIN_CAR_CARD
            ] = 0
            env_state[ENV_PHASE] = 1
            env_state[ENV_ID_ACTION] = (id_action + 1) % NUMBER_PLAYER
            if env_state[ENV_CHECK_END] >= 1:
                if env_state[ENV_ID_ACTION] == env_state[ENV_ID_PLAYER_END]:
                    env_state[ENV_CHECK_END] += 0.5  # edit by Hieu 05012023
            if (
                np.min(env_state[ENV_TRAIN_CAR_OPEN:ENV_ROUTE_CARD_GET]) == -1
                or env_state[ENV_TRAIN_CAR_CARD] == -1
            ) and np.sum(env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL]) > 0:
                env_state = shuffle_drop_card(env_state)
            # Lật thêm thẻ ra khi thẻ mở đang không đủ 5 thẻ
            if (
                np.min(
                    env_state[
                        ENV_TRAIN_CAR_OPEN : ENV_TRAIN_CAR_OPEN
                        + NUMBER_TRAIN_CAR_CARD_OPEN
                    ]
                )
                == -1
                and env_state[ENV_TRAIN_CAR_CARD] != -1
            ):
                env_state = shuffle_drop_card(env_state)
                env_state = process_train_car_board(env_state)
        # nếu xây tunnel
        else:
            if player_train_car[type_train_car_use] >= LIST_ALL_LENGTH_ROAD[road]:
                # bỏ thẻ vào chồng bài build_tunnel
                env_state[ENV_CARD_BULD_TUNNEL:ENV_CARD_TEST_TUNNEL][
                    type_train_car_use
                ] += LIST_ALL_LENGTH_ROAD[road]
                # nếu đủ train_Car thì chỉ trừ train_Car
                env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][
                    ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                    + NUMBER_TYPE_TRAIN_CAR_CARD
                ][
                    type_train_car_use
                ] -= LIST_ALL_LENGTH_ROAD[
                    road
                ]
            else:
                # nếu k đủ traincar thì tối thiểu việc dùng locomotive
                locomotive_use = (
                    LIST_ALL_LENGTH_ROAD[road] - player_train_car[type_train_car_use]
                )
                # bỏ thẻ vào chồng bài build_tunnel
                env_state[ENV_CARD_BULD_TUNNEL:ENV_CARD_TEST_TUNNEL][
                    type_train_car_use
                ] += env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][
                    ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                    + NUMBER_TYPE_TRAIN_CAR_CARD
                ][
                    type_train_car_use
                ]
                env_state[ENV_CARD_BULD_TUNNEL:ENV_CARD_TEST_TUNNEL][
                    type_train_car_use - type_train_car_use
                ] += locomotive_use
                # trừ tài nguyên người chơi
                env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][
                    ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                    + NUMBER_TYPE_TRAIN_CAR_CARD
                ][
                    type_train_car_use - type_train_car_use
                ] -= locomotive_use
                env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][
                    ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                    + NUMBER_TYPE_TRAIN_CAR_CARD
                ][
                    type_train_car_use
                ] = 0

            # đi lật bài để test tunnel
            if env_state[ENV_TRAIN_CAR_CARD + 3] == -1:
                env_state = shuffle_drop_card(env_state)
            car_test_tunnel = env_state[
                ENV_TRAIN_CAR_CARD : ENV_TRAIN_CAR_CARD + NUMBER_CARD_TEST_TUNNEL
            ].astype(np.int64)
            #  print('check', car_test_tunnel)
            for car in car_test_tunnel:
                if car != -1:
                    env_state[ENV_CARD_TEST_TUNNEL + car] += 1
            # điều chỉnh chồng bài úp
            env_state[ENV_TRAIN_CAR_CARD:ENV_IN4_PLAYER] = np.concatenate(
                (
                    env_state[ENV_TRAIN_CAR_CARD:ENV_IN4_PLAYER][
                        NUMBER_CARD_TEST_TUNNEL:
                    ],
                    np.array([-1] * NUMBER_CARD_TEST_TUNNEL),
                )
            )
            env_state[
                ENV_TYPE_TRAIN_CAR_BUILD_ROAD : ENV_TYPE_TRAIN_CAR_BUILD_ROAD
                + NUMBER_TYPE_TRAIN_CAR_CARD
            ] = 0
            env_state[ENV_PHASE] = 4

    elif phase_game == 3:
        # nếu dừng bỏ thẻ:
        if action == 169:
            route_card_save = np.where(
                env_state[ENV_ROUTE_CARD_GET:ENV_TRAIN_CAR_DROP] == 1
            )[0]
            player_route_card = env_state[
                ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * (id_action + 1)
            ][ATT_NUMBER_TYPE_TRAIN_CARD + NUMBER_TYPE_TRAIN_CAR_CARD :]
            player_route_card[route_card_save] = 1
            env_state[
                ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * (id_action + 1)
            ][
                ATT_NUMBER_TYPE_TRAIN_CARD + NUMBER_TYPE_TRAIN_CAR_CARD :
            ] = player_route_card
            env_state[ENV_ROUTE_CARD_GET:ENV_TRAIN_CAR_DROP] = 0
            env_state[ENV_NUMBER_DROP_ROUTE_CARD] = 0
            env_state[ENV_PHASE] = 1
            env_state[ENV_ID_ACTION] = (id_action + 1) % NUMBER_PLAYER
            if env_state[ENV_CHECK_END] >= 1:
                if env_state[ENV_ID_ACTION] == env_state[ENV_ID_PLAYER_END]:
                    env_state[ENV_CHECK_END] += 0.5  # edit by Hieu 05012023

            if env_state[ENV_TURN] <= NUMBER_PLAYER:
                env_state[ENV_TURN] += 1
            if env_state[ENV_TURN] <= NUMBER_PLAYER:
                id_action = int(env_state[ENV_ID_ACTION])
                env_state[ENV_ROUTE_CARD_GET:ENV_TRAIN_CAR_DROP] = env_state[
                    ENV_IN4_PLAYER:ENV_TRAIN_CAR_OPEN
                ][id_action * ATTRIBUTE_PLAYER : (id_action + 1) * ATTRIBUTE_PLAYER][
                    3 + NUMBER_TYPE_TRAIN_CAR_CARD :
                ]
                env_state[ENV_IN4_PLAYER:ENV_TRAIN_CAR_OPEN][
                    id_action * ATTRIBUTE_PLAYER : (id_action + 1) * ATTRIBUTE_PLAYER
                ][2 + NUMBER_TYPE_TRAIN_CAR_CARD :] = 0
                env_state[ENV_PHASE] = 3
        else:
            drop_card = action - 101
            # lưu thẻ mình đã bỏ nếu ko phải lượt bỏ thẻ ban đầu
            if env_state[ENV_TURN] > NUMBER_PLAYER:
                route_card_board = env_state[ENV_ROUTE_CARD_BOARD:ENV_TRAIN_CAR_CARD]
                route_insert = np.where(route_card_board == -1)[0][0]
                route_card_board[route_insert] = drop_card
                env_state[ENV_ROUTE_CARD_BOARD:ENV_TRAIN_CAR_CARD] = route_card_board
            env_state[
                ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * (id_action + 1)
            ][ATT_NUMBER_TYPE_TRAIN_CARD + NUMBER_TYPE_TRAIN_CAR_CARD :][drop_card] = -1
            env_state[ENV_ROUTE_CARD_GET:ENV_TRAIN_CAR_DROP][drop_card] = 0
            env_state[ENV_NUMBER_DROP_ROUTE_CARD] += 1
            # nếu ko được bỏ thẻ nữa
            if env_state[ENV_NUMBER_DROP_ROUTE_CARD] == 2:
                route_card_save = np.where(
                    env_state[ENV_ROUTE_CARD_GET:ENV_TRAIN_CAR_DROP] == 1
                )[0]
                player_route_card = env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][ATT_NUMBER_TYPE_TRAIN_CARD + NUMBER_TYPE_TRAIN_CAR_CARD :]
                player_route_card[route_card_save] = 1
                env_state[
                    ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                    + ATTRIBUTE_PLAYER * (id_action + 1)
                ][
                    ATT_NUMBER_TYPE_TRAIN_CARD + NUMBER_TYPE_TRAIN_CAR_CARD :
                ] = player_route_card
                env_state[ENV_ROUTE_CARD_GET:ENV_TRAIN_CAR_DROP] = 0
                env_state[ENV_NUMBER_DROP_ROUTE_CARD] = 0
                env_state[ENV_PHASE] = 1
                env_state[ENV_ID_ACTION] = (id_action + 1) % NUMBER_PLAYER

                if env_state[ENV_CHECK_END] >= 1:
                    if env_state[ENV_ID_ACTION] == env_state[ENV_ID_PLAYER_END]:
                        env_state[ENV_CHECK_END] += 0.5  # edit by Hieu 05012023
                #  env_state[ENV_TURN] += 1
                if env_state[ENV_TURN] <= NUMBER_PLAYER:
                    env_state[ENV_TURN] += 1
                if env_state[ENV_TURN] <= NUMBER_PLAYER:
                    env_state[ENV_PHASE] = 3
                    id_action = int(env_state[ENV_ID_ACTION])
                    env_state[ENV_ROUTE_CARD_GET:ENV_TRAIN_CAR_DROP] = env_state[
                        ENV_IN4_PLAYER:ENV_TRAIN_CAR_OPEN
                    ][
                        id_action
                        * ATTRIBUTE_PLAYER : (id_action + 1)
                        * ATTRIBUTE_PLAYER
                    ][
                        3 + NUMBER_TYPE_TRAIN_CAR_CARD :
                    ]
                    env_state[ENV_IN4_PLAYER:ENV_TRAIN_CAR_OPEN][
                        id_action
                        * ATTRIBUTE_PLAYER : (id_action + 1)
                        * ATTRIBUTE_PLAYER
                    ][ATT_NUMBER_TYPE_TRAIN_CARD + NUMBER_TYPE_TRAIN_CAR_CARD :] = 0

    elif phase_game == 4:
        train_car_return = env_state[ENV_CARD_BULD_TUNNEL:ENV_CARD_TEST_TUNNEL]
        test_train_car = env_state[ENV_CARD_TEST_TUNNEL:ENV_PHASE]
        if action == 167:  # không xây hầm
            # trả tài nguyên cho người chơi đã bỏ ra để định xây hầm, nạp các thẻ test xây hầm vào chồng bài úp
            env_state[
                ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * (id_action + 1)
            ][
                ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                + NUMBER_TYPE_TRAIN_CAR_CARD
            ] += train_car_return
            env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL] += test_train_car

        elif action == 166:  # có xây hầm
            # kiểm tra xem trùng loại thẻ nào, trừ loại thẻ đấy rồi cho hết lá bài vào drop
            temp = train_car_return * test_train_car
            subtract_car = (temp >= 1) * test_train_car
            # trừ tài nguyên nếu trùng vs tài nguyên test tunnel
            train_car_player = env_state[
                ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * (id_action + 1)
            ][
                ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                + NUMBER_TYPE_TRAIN_CAR_CARD
            ]
            train_car_player -= subtract_car
            for type_train_car in range(1, NUMBER_TYPE_TRAIN_CAR_CARD):
                if train_car_player[type_train_car] < 0:
                    train_car_player[0] += train_car_player[type_train_car]
                    train_car_player[type_train_car] = 0
            env_state[
                ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * id_action : ENV_IN4_PLAYER
                + ATTRIBUTE_PLAYER * (id_action + 1)
            ][
                ATT_NUMBER_TYPE_TRAIN_CARD : ATT_NUMBER_TYPE_TRAIN_CARD
                + NUMBER_TYPE_TRAIN_CAR_CARD
            ] = train_car_player
            env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL] += (
                test_train_car + train_car_return + subtract_car
            )
            # cộng điểm cho người chơi
            env_state[
                ENV_IN4_PLAYER + ATTRIBUTE_PLAYER * id_action + ATT_SCORE
            ] += LIST_SCORE_BUILD_ROAD[
                LIST_ALL_LENGTH_ROAD[int(env_state[ENV_ROAD_BUILT])]
            ]
            env_state[
                ENV_IN4_PLAYER + ATTRIBUTE_PLAYER * id_action + ATT_NUMBER_TRAIN
            ] -= LIST_ALL_LENGTH_ROAD[int(env_state[ENV_ROAD_BUILT])]
            if (
                env_state[
                    ENV_IN4_PLAYER + ATTRIBUTE_PLAYER * id_action + ATT_NUMBER_TRAIN
                ]
                <= 2
                and env_state[ENV_CHECK_END] == 0
            ):
                env_state[ENV_CHECK_END] = 1
                env_state[ENV_ID_PLAYER_END] = (id_action + 1) % NUMBER_PLAYER
            # ghi nhận đường cho người chơi:
            env_state[int(env_state[ENV_ROAD_BUILT])] = id_action
        # update phase, chuyển người chơi
        env_state[ENV_PHASE] = 1
        env_state[ENV_CARD_BULD_TUNNEL:ENV_PHASE] = 0
        env_state[ENV_ID_ACTION] = (id_action + 1) % NUMBER_PLAYER
        if env_state[ENV_CHECK_END] >= 1:
            if env_state[ENV_ID_ACTION] == env_state[ENV_ID_PLAYER_END]:
                env_state[ENV_CHECK_END] += 0.5  # edit by Hieu 05012023
        if (
            env_state[ENV_TRAIN_CAR_CARD] == -1
            and np.sum(env_state[ENV_TRAIN_CAR_DROP:ENV_CARD_BULD_TUNNEL]) > 0
        ):
            env_state = shuffle_drop_card(env_state)
        # Lật thêm thẻ ra khi thẻ mở đang không đủ 5 thẻ
        if (
            np.min(
                env_state[
                    ENV_TRAIN_CAR_OPEN : ENV_TRAIN_CAR_OPEN + NUMBER_TRAIN_CAR_CARD_OPEN
                ]
            )
            == -1
            and env_state[ENV_TRAIN_CAR_CARD] != -1
        ):
            env_state = shuffle_drop_card(env_state)
            env_state = process_train_car_board(env_state)
    return env_state


#  #########################################################


@njit()
def bot_lv0(state, perData):
    validActions = getValidActions(state)
    arr_action = np.where(validActions == 1)[0]
    idx = np.random.randint(0, arr_action.shape[0])
    return arr_action[idx], perData


#########################################################


@njit()
def one_game_numba(p0, list_other, per_player, per1, per2, per3, per4, p1, p2, p3, p4):
    env_state = initEnv()
    count_turn = 0

    while system_check_end(env_state) and count_turn < 3000:
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

    winner, env_state = check_winner(env_state)
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

    if np.where(list_other == -1)[0][0] in winner:
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

    while system_check_end(env_state) and count_turn < 3000:
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

    winner, env_state = check_winner(env_state)

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
    if np.where(list_other == -1)[0][0] in winner:
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
