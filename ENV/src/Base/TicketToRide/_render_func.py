import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from src.Base.TicketToRide import env as _env
from env import SHORT_PATH

IMG_PATH = SHORT_PATH + "src/Base/TicketToRide/Images/"
BG_SIZE_BASE = np.array((1600, 1100))
CARD_SIZE_BASE = np.array((120, 80))
ROUTE_SIZE_BASE = np.array([145, 80])
RATIO = 1
BG_SIZE = BG_SIZE_BASE * RATIO
CARD_SIZE = CARD_SIZE_BASE * RATIO
ROUTE_SIZE = ROUTE_SIZE_BASE * RATIO


class Sprites:
    def __init__(self) -> None:
        self.background = Image.open(IMG_PATH + "back_ground.jpg").resize(BG_SIZE)
        self.myFont = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", size=40 * RATIO
        )
        self.myFont_end = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", size=20 * RATIO
        )

        self.font2 = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", size=60 * RATIO
        )
        self.PLAYER_TRAIN_CARD_COOR = (
            np.array(
                [
                    [1400, 0],
                    [1400, 80],
                    [1400, 160],
                    [1400, 240],
                    [1400, 320],
                    [1400, 400],
                    [1400, 480],
                    [1400, 560],
                    [1400, 640],
                ]
            )
            * RATIO
        )

        self.COUNT_PLAYER_TRAIN_CARD_COOR = (
            np.array(
                [
                    [1530, 20],
                    [1530, 100],
                    [1530, 180],
                    [1530, 260],
                    [1530, 340],
                    [1530, 420],
                    [1530, 500],
                    [1530, 580],
                    [1530, 660],
                ]
            )
            * RATIO
        )

        self.COUNT_PLAYER_TRAIN_CARD_TUNNEL_COOR = (
            np.array(
                [
                    [1380, 20],
                    [1380, 100],
                    [1380, 180],
                    [1380, 260],
                    [1380, 340],
                    [1380, 420],
                    [1380, 500],
                    [1380, 580],
                    [1380, 660],
                ]
            )
            * RATIO
        )

        self.CARD_TRAIN_CARD_BOARD = (
            np.array(
                [
                    [1200, 0],
                    [1200, 80],
                    [1200, 160],
                    [1200, 240],
                    [1200, 320],
                    [1200, 400],
                ]
            )
            * RATIO
        )

        self.CARD_TRAIN_CAR_BOARD_DARE = (
            np.array([[1200, 480], [1200, 560], [1200, 640]]) * RATIO
        )

        self.ROUTE_CARD_BOARD_COOR = (
            np.array([[1200, 480], [1200, 560], [1200, 640], [1200, 720]]) * RATIO
        )

        self.ROUTE_CARD_COOR = (
            np.array(
                [
                    [0, 800],
                    [80, 800],
                    [160, 800],
                    [240, 800],
                    [320, 800],
                    [400, 800],
                    [480, 800],
                    [560, 800],
                    [640, 800],
                    [720, 800],
                    [800, 800],
                    [880, 800],
                    [960, 800],
                    [1040, 800],
                    [1120, 800],
                    [1200, 800],
                    [1280, 800],
                    [1360, 800],
                    [0, 945],
                    [80, 945],
                    [160, 945],
                    [240, 945],
                    [320, 945],
                    [400, 945],
                    [480, 945],
                    [560, 945],
                    [640, 945],
                    [720, 945],
                    [800, 945],
                    [880, 945],
                    [960, 945],
                    [1040, 945],
                    [1120, 945],
                    [1200, 945],
                    [1280, 945],
                    [1360, 945],
                    [1440, 945],
                ]
            )
            * RATIO
        )

        self.IN4_END = np.array([[1000, 945], [1000, 985], [1000, 1025], [1000, 1065]])

        self.ROAD_COOR = [
            [7, 734, 40, 765],
            [-7, 673, 3, 641, 50, 657],
            [105, 765, 130, 740, 100, 700],
            [130, 686, 175, 685],
            [105, 660, 130, 625, 170, 595],
            [90, 651, 118, 616, 156, 583],
            [200, 609, 203, 655],
            [235, 663, 265, 635, 300, 600, 340, 580],
            [235, 590, 257, 565, 292, 539, 337, 548],
            [200, 540, 223, 502, 240, 457, 247, 413],
            [213, 547, 238, 506, 253, 464, 263, 419],
            [188, 506, 185, 465, 170, 417, 132, 387],
            [137, 368, 185, 370, 227, 374],
            [130, 337, 175, 325],
            [245, 349],
            [211, 245, 206, 290],
            [229, 245, 225, 292],
            [245, 315, 281, 287],
            [302, 303, 282, 340],
            [318, 316, 297, 350],
            [335, 248],
            [258, 212, 305, 215],
            [153, 63, 169, 100, 190, 146, 207, 188],
            [170, 56, 187, 98, 205, 138, 222, 183],
            [1071, 753, 1115, 760, 1145, 740],
            [1056, 564, 1060, 610, 1075, 657, 1108, 690],
            [1153, 586, 1147, 631, 1142, 675],
            [925, 773, 970, 773, 1010, 760],
            [972, 690, 1010, 715],
            [1085, 537, 1125, 543],
            [1163, 469, 1160, 515],
            [965, 640, 993, 627, 1022, 607, 1039, 564],
            [910, 490, 947, 468, 995, 470, 1030, 500],
            [1062, 498, 1069, 454, 1091, 425, 1132, 429],
            [1155, 373, 1166, 404],
            [1150, 240, 1163, 257, 1162, 302, 1140, 343],
            [1083, 383, 1038, 387, 990, 372, 960, 337],
            [1065, 215, 1109, 202],
            [1052, 43, 1090, 65, 1119, 102, 1132, 151],
            [984, 307, 1032, 289, 1038, 245],
            [933, 225, 948, 269],
            [793, 261, 828, 287, 875, 292, 918, 293],
            [717, 395, 747, 365, 786, 340, 828, 322, 873, 311, 918, 311],
            [935, 183, 968, 170, 1000, 195],
            [790, 216, 822, 184, 874, 192],
            [805, 73, 812, 116, 842, 152, 878, 174],
            [921, 173, 948, 134, 975, 100, 1000, 63],
            [841, 37, 885, 35, 930, 35, 975, 38],
            [683, 23, 714, 19, 763, 2, 807, 2, 843, 2, 895, 2, 937, 2, 985, 15],
            [905, 735, 924, 696],
            [630, 769, 675, 769, 720, 769, 765, 769, 810, 769, 855, 769],
            [818, 735, 860, 745],
            [838, 610, 873, 629, 915, 655],
            [892, 548, 913, 588, 932, 630],
            [840, 575, 870, 545],
            [730, 436, 769, 458, 808, 477, 848, 500],
            [887, 476, 901, 432, 915, 391, 930, 350],
            [720, 115, 735, 78, 773, 49],
            [553, 69, 583, 39, 624, 7],
            [561, 80, 592, 48, 630, 22],
            [457, 193, 477, 157, 502, 115],
            [470, 205, 495, 165, 518, 127],
            [573, 208, 589, 169, 629, 136, 673, 138],
            [735, 165, 760, 200],
            [597, 236, 647, 222, 688, 219, 740, 220],
            [605, 250, 643, 240, 688, 240, 735, 240],
            [661, 373, 700, 345, 730, 312, 758, 269],
            [672, 408],
            [659, 424],
            [652, 489, 676, 457],
            [705, 460, 712, 502, 717, 546],
            [758, 565, 791, 565],
            [785, 615, 765, 660, 775, 700],
            [625, 542, 651, 585, 695, 591],
            [725, 617, 720, 660, 716, 707, 749, 722],
            [660, 668, 673, 707, 706, 740, 756, 745],
            [642, 675, 630, 720, 605, 750],
            [569, 596, 615, 615],
            [528, 533, 535, 572],
            [418, 558, 448, 533, 483, 544, 507, 579],
            [555, 625, 592, 653, 603, 695, 587, 741],
            [455, 300, 496, 282, 540, 263],
            [460, 317, 500, 300, 545, 278],
            [578, 284, 594, 329, 620, 362],
            [350, 188, 378, 177, 420, 203],
            [450, 285, 470, 260],
            [493, 225, 535, 230],
            [375, 250, 405, 275],
            [350, 280, 394, 291],
            [318, 374, 358, 354, 398, 331],
            [325, 390, 365, 372, 405, 345],
            [437, 351, 461, 379],
            [443, 429, 473, 402],
            [505, 415, 510, 457],
            [522, 399, 560, 422, 607, 404],
            [628, 425, 626, 464],
            [550, 485, 595, 492],
            [449, 469, 490, 490],
            [412, 489, 400, 530],
            [308, 419, 341, 452, 385, 460],
            [282, 430, 297, 476, 333, 509, 367, 539],
        ]

        self.PLAYER_NUMBER_TRAIN = np.array([1400, 750]) * RATIO
        card_values = np.arange(9)
        self.cards = []
        for value in card_values:
            self.cards.append(
                Image.open(IMG_PATH + f"trainCard_{value}.png").resize(CARD_SIZE)
            )
        self.cards.append(
            Image.open(IMG_PATH + f"trainCard_down.png").resize(CARD_SIZE)
        )

        route_id = np.arange(46)
        self.route_card = []
        for route in route_id:
            self.route_card.append(
                Image.open(IMG_PATH + f"/images_route/{route}.png").resize(ROUTE_SIZE)
            )
        self.route_card.append(
            Image.open(IMG_PATH + f"/images_route/route_down.png").resize(ROUTE_SIZE)
        )

        #  self.text_phase = ['chọn xúc sắc để đổ', 'chọn đổ lại hay k', 'chọn lấy tiền của ai', 'chọn người để đổi', 'chọn lá bài để đổi', 'chọn lá bài muốn lấy', 'chọn mua thẻ']


sprites = Sprites()


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""
    else:
        action_mean = _env.ACTIONS_MEAN[action]

    return f"player {action_mean}"


class Env_components:
    def __init__(self, env, winner, list_other) -> None:
        self.env = env
        self.winner = winner
        self.list_other = list_other


def get_env_components():
    env = _env.initEnv()
    winner = -1
    list_other = np.array([-1, 1, 2, 3, 4])
    np.random.shuffle(list_other)
    env_components = Env_components(env, winner, list_other)
    return env_components


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    if not action is None:
        env_components.env = _env.stepEnv(env_components.env, action)

    my_idx = np.where(env_components.list_other == -1)[0][0]
    env_components.winner = -1

    while _env.system_check_end(env_components.env):
        p_idx = int(env_components.env[_env.ENV_ID_ACTION])
        state = _env.getAgentState(env_components.env)

        if env_components.list_other[p_idx] == -1:
            win = -1
            return win, state, env_components

        agent = list_agent[env_components.list_other[p_idx] - 1]
        data = list_data[env_components.list_other[p_idx] - 1]
        action, data = agent(state, data)

        env_state = env_components.env.copy()
        arr_action = _env.getValidActions(state)
        env_components.env = _env.stepEnv(env_components.env, action)
    env_components.winner, env_components.env = _env.check_winner(env_components.env)
    for p_idx in range(5):
        if p_idx != my_idx:
            env_components.env[_env.ENV_ID_ACTION] = p_idx
            state = _env.getAgentState(env_components.env)

            agent = list_agent[env_components.list_other[p_idx] - 1]
            data = list_data[env_components.list_other[p_idx] - 1]
            action, data = agent(state, data)

    env_components.env[_env.ENV_ID_ACTION] = my_idx
    state = _env.getAgentState(env_components.env)
    if my_idx in env_components.winner:
        win = 1
    else:
        win = 0

    return win, state, env_components


def get_state_image(state=None):
    background = sprites.background.copy()

    draw = ImageDraw.Draw(background)
    if state is None:
        return background
    # vẽ thẻ traincar của người chơi
    player_train_card_card = state[10:19]
    draw_train_car_cards_player(background, player_train_card_card, draw)

    # vẽ đường trên bàn chơi
    all_player_road = state[19:524]
    for id_player in range(5):
        player_road = np.where(
            all_player_road[101 * id_player : 101 * (id_player + 1)]
        )[0]
        if len(player_road) > 0:
            draw_player_road(player_road, id_player, draw)

    # vẽ thẻ route của người chơi
    player_route_card = np.where(state[524:570])[0]
    draw_route_cards_player(background, player_route_card)

    # vẽ thẻ train_car mở trên bàn chơi
    board_train_card = state[616:625]
    train_card_down = state[666]
    draw_board_train_car(background, board_train_card, down=train_card_down)
    # chồng thẻ route úp
    if state[657]:
        background.paste(
            sprites.route_card[-1], tuple(sprites.ROUTE_CARD_BOARD_COOR[-1])
        )

    draw_other(background, draw, state)

    return background


def draw_board_train_car(bg, board_train_car, down=1):
    index = 1
    for type_card in range(9):
        #  if index == 5:
        #      break
        if board_train_car[type_card] != 0:
            for count in range(int(board_train_car[type_card])):
                #  print(index)
                bg.paste(
                    sprites.cards[type_card],
                    tuple(sprites.CARD_TRAIN_CARD_BOARD[index]),
                )
                index += 1
                #  if index == 5:
                #      break
    if down:
        bg.paste(sprites.cards[-1], tuple(sprites.CARD_TRAIN_CARD_BOARD[0]))


def draw_route_cards_player(bg, route_cards):
    #  print(route_cards)
    n = route_cards.shape[0]
    for i in range(n):
        card = sprites.route_card[route_cards[i]].rotate(90, expand=True)
        bg.paste(
            card,
            (sprites.ROUTE_CARD_COOR[i][0], sprites.ROUTE_CARD_COOR[i][1]),
        )


def draw_train_car_cards_player(bg, train_car_card, draw):
    n = train_car_card.shape[0]
    for card in range(9):
        #  if train_car_card[card] > 0:
        bg.paste(sprites.cards[card], tuple(sprites.PLAYER_TRAIN_CARD_COOR[card]))
        draw.text(
            sprites.COUNT_PLAYER_TRAIN_CARD_COOR[card],
            f"{train_car_card[card]}",
            (100, 255, 100),
            font=sprites.myFont,
        )


def draw_player_road(roads, player_id, draw):
    list_color = [
        (100, 255, 100),
        (100, 255, 200),
        (0, 0, 0),
        (100, 60, 100),
        (0, 0, 255),
    ]
    color = list_color[player_id]
    for id_road in roads:
        road = sprites.ROAD_COOR[id_road]
        for i in range(0, len(road), 2):
            draw.text(tuple(road[i : i + 2]), "*", color, font=sprites.myFont)


def draw_other(bg, draw, state):
    player_score = state[0:5].astype(np.int64)
    last_player = state[660:665].astype(np.int64)
    phase = np.where(state[653:657])[0][0] + 1
    #  print('check;',state[659])
    if state[659]:
        route_complete = state[668:673].astype(np.int64)
        is_longest_road = state[673:678].astype(np.int64)
        player_sub_score = state[5:10].astype(np.int64)
        all_in4 = [player_score, player_sub_score, route_complete, is_longest_road]
        all_text = ["score: ", "sub_score: ", "route_done: ", "LG_road: "]
        for i in range(4):
            draw.text(
                tuple(sprites.IN4_END[i]),
                f"{all_text[i]}{all_in4[i]}",
                (0, 0, 0),
                font=sprites.myFont_end,
            )
    else:
        all_text = ["score: ", "Phase: ", "last_player: "]
        draw.text(
            tuple(sprites.IN4_END[0]),
            f"{all_text[0]}{player_score}",
            (0, 0, 0),
            font=sprites.myFont,
        )
        draw.text(
            tuple(sprites.IN4_END[1]),
            f"{all_text[1]}{phase}",
            (0, 0, 0),
            font=sprites.myFont,
        )
        if np.sum(last_player) > 0:
            draw.text(
                tuple(sprites.IN4_END[2]),
                f"{all_text[2]}{last_player}",
                (0, 0, 0),
                font=sprites.myFont,
            )
        if phase == 2:
            type_builds = state[643:652]
            for t_build in range(9):
                if type_builds[t_build] != 0:
                    draw.text(
                        sprites.COUNT_PLAYER_TRAIN_CARD_TUNNEL_COOR[t_build],
                        f"C",
                        (100, 255, 100),
                        font=sprites.myFont,
                    )

        elif phase == 3:
            number_drop = state[658]
            route_get = np.where(state[570:616])[0]
            for i in range(len(route_get)):
                route_card_get_id = route_get[i]
                bg.paste(
                    sprites.route_card[route_card_get_id],
                    tuple(sprites.ROUTE_CARD_BOARD_COOR[i]),
                )

            draw.text(
                tuple(sprites.ROUTE_CARD_BOARD_COOR[-1]),
                f"drop:{number_drop}",
                (0, 0, 0),
                font=sprites.myFont,
            )

        elif phase == 4:
            card_build_tunnel = state[625:634]
            card_test_tunnel = state[634:643]
            for t_build in range(9):
                if card_build_tunnel[t_build] != 0:
                    draw.text(
                        sprites.COUNT_PLAYER_TRAIN_CARD_TUNNEL_COOR[t_build],
                        f"{card_build_tunnel[t_build]}",
                        (100, 255, 100),
                        font=sprites.myFont,
                    )

            index = 0
            for type_card in range(9):
                #  if index == 3:
                #      break
                if card_test_tunnel[type_card] != 0:
                    for count in range(int(card_test_tunnel[type_card])):
                        #  print(index)
                        bg.paste(
                            sprites.cards[type_card],
                            tuple(sprites.ROUTE_CARD_BOARD_COOR[index]),
                        )
                        index += 1
                        #  if index == 3:
                        #      break

    # print number_train
    draw.text(
        tuple(sprites.PLAYER_NUMBER_TRAIN),
        f"Train:{int(state[667])}",
        (0, 0, 0),
        font=sprites.myFont,
    )
