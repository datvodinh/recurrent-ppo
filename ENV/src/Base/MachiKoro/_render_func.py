import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from src.Base.MachiKoro import env as _env
from env import SHORT_PATH

IMG_PATH = SHORT_PATH + "src/Base/MachiKoro/images/"
ROBBER_ICON = (40, 40)
BG_SIZE = (1200, 800)
CARD_SIZE = (60, 100)


class Sprites:
    def __init__(self) -> None:
        self.background = Image.open(IMG_PATH + "background.webp").resize(BG_SIZE)
        self.robber = Image.open(IMG_PATH + "robber.jpg").resize(ROBBER_ICON)

        card_values = np.arange(19)
        self.cards = []
        for value in card_values:
            self.cards.append(Image.open(IMG_PATH + f"{value}.jpg").resize(CARD_SIZE))
        for value in card_values[-4:]:
            self.cards.append(
                Image.open(IMG_PATH + f"{value}down.jpg").resize(CARD_SIZE)
            )

        self.text_phase = [
            "chọn xúc sắc để đổ",
            "chọn đổ lại hay k",
            "chọn lấy tiền của ai",
            "chọn người để đổi",
            "chọn lá bài để đổi",
            "chọn lá bài muốn lấy",
            "chọn mua thẻ",
        ]


sprites = Sprites()


class Params:
    def __init__(self) -> None:
        self.center_card_x = BG_SIZE[0] * 0.5
        self.center_card_y = (BG_SIZE[1] - CARD_SIZE[1]) * 0.5
        self.list_coords_0 = [
            (self.center_card_x, 0.92 * BG_SIZE[1] - CARD_SIZE[1]),
            (0.82 * BG_SIZE[0], self.center_card_y),
            (self.center_card_x, 0.08 * BG_SIZE[1]),
            (0.18 * BG_SIZE[0], self.center_card_y),
        ]

        x_0 = BG_SIZE[0] * 0.32
        x_1 = BG_SIZE[0] * 0.68
        y_0 = 0.2 * BG_SIZE[1] - 0.25 * CARD_SIZE[1]
        y_1 = 0.8 * BG_SIZE[1] - 0.75 * CARD_SIZE[1]
        self.list_coords_1 = [(x_0, y_1), (x_1, y_1), (x_1, y_0), (x_0, y_0)]
        self.myFont = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", size=20
        )


params = Params()


_d_ = CARD_SIZE[0]


def draw_cards(bg, cards, s, y, rotate=False):
    n = cards.shape[0]
    y = round(y)
    #  print(rotate)
    if rotate % 2:
        for i in range(n):
            card = sprites.cards[cards[i]].rotate((rotate - 4) * 90, expand=True)
            bg.paste(
                card,
                (s, round(y + _d_ * i)),
            )
    else:
        for i in range(n):
            bg.paste(
                sprites.cards[cards[i]],
                (round(s + _d_ * i), y),
            )


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
    list_other = np.array([-1, 1, 2, 3])
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

        env_components.env = _env.stepEnv(env_components.env, action)

    env_components.env[_env.ENV_CHECK_END] = 1
    env_components.winner = _env.check_winner(env_components.env)
    for p_idx in range(4):
        if p_idx != my_idx:
            env_components.env[_env.ENV_ID_ACTION] = p_idx
            state = _env.getAgentState(env_components.env)

            agent = list_agent[env_components.list_other[p_idx] - 1]
            data = list_data[env_components.list_other[p_idx] - 1]
            action, data = agent(state, data)

    env_components.env[_env.ENV_ID_ACTION] = my_idx
    state = _env.getAgentState(env_components.env)
    if my_idx == env_components.winner:
        win = 1
    else:
        win = 0

    return win, state, env_components


def get_state_image(state=None):
    background = sprites.background.copy()
    if state is None:
        return background

    draw = ImageDraw.Draw(background)
    myFont = params.myFont

    arr_player_card_coordinate = np.array(
        [
            [
                [250, 670],
                [310, 670],
                [370, 670],
                [430, 670],
                [490, 670],
                [550, 670],
                [610, 670],
                [670, 670],
                [730, 670],
                [790, 670],
                [850, 670],
                [910, 670],
                [370, 570],
                [430, 570],
                [490, 570],
                [550, 570],
                [610, 570],
                [670, 570],
                [730, 570],
            ],
            [
                [1080, 20],
                [1080, 80],
                [1080, 140],
                [1080, 200],
                [1080, 260],
                [1080, 320],
                [1080, 380],
                [1080, 440],
                [1080, 500],
                [1080, 560],
                [1080, 620],
                [1080, 680],
                [980, 140],
                [980, 200],
                [980, 260],
                [980, 320],
                [980, 380],
                [980, 440],
                [980, 500],
            ],
            [
                [250, 30],
                [310, 30],
                [370, 30],
                [430, 30],
                [490, 30],
                [550, 30],
                [610, 30],
                [670, 30],
                [730, 30],
                [790, 30],
                [850, 30],
                [910, 30],
                [370, 130],
                [430, 130],
                [490, 130],
                [550, 130],
                [610, 130],
                [670, 130],
                [730, 130],
            ],
            [
                [30, 20],
                [30, 80],
                [30, 140],
                [30, 200],
                [30, 260],
                [30, 320],
                [30, 380],
                [30, 440],
                [30, 500],
                [30, 560],
                [30, 620],
                [30, 680],
                [130, 140],
                [130, 200],
                [130, 260],
                [130, 320],
                [130, 380],
                [130, 440],
                [130, 500],
            ],
        ]
    )

    arr_count_card_coor = [
        [
            [270, 770],
            [330, 770],
            [390, 770],
            [450, 770],
            [510, 770],
            [570, 770],
            [630, 770],
            [690, 770],
            [750, 770],
            [810, 770],
            [870, 770],
            [930, 770],
        ],
        [
            [1180, 40],
            [1180, 100],
            [1180, 160],
            [1180, 220],
            [1180, 280],
            [1180, 340],
            [1180, 400],
            [1180, 460],
            [1180, 520],
            [1180, 580],
            [1180, 640],
            [1180, 700],
        ],
        [
            [270, 20],
            [330, 20],
            [390, 20],
            [450, 20],
            [510, 20],
            [570, 20],
            [630, 20],
            [690, 20],
            [750, 20],
            [810, 20],
            [870, 20],
            [930, 20],
        ],
        [
            [15, 40],
            [15, 100],
            [15, 160],
            [15, 220],
            [15, 280],
            [15, 340],
            [15, 400],
            [15, 460],
            [15, 520],
            [15, 580],
            [15, 640],
            [15, 700],
        ],
    ]
    for id in range(4):
        player_in4 = state[20 * id : 20 * (id + 1)]
        background = draw_card_player(
            player_in4,
            background,
            id,
            arr_player_card_coordinate,
            arr_count_card_coor,
            draw,
        )

    background = draw_card_on_board(state, background)

    background = draw_other(state, draw, background, myFont)

    return background


def draw_card_on_board(state, background):
    board_card = np.array(
        [
            [[370, 300], [430, 300], [490, 300], [550, 300], [610, 300], [670, 300]],
            [[370, 400], [430, 400], [490, 400], [550, 400], [610, 400], [670, 400]],
        ]
    )
    card_board = state[80:92] > 0
    for i in range(12):
        if card_board[i]:
            draw_cards(
                background,
                np.array([i]),
                board_card[i // 6][i % 6][0],
                board_card[i // 6][i % 6][1],
                rotate=0,
            )

    return background


def draw_card_player(
    player_in4,
    background,
    player_id,
    arr_player_card_coordinate,
    arr_count_card_coor,
    draw,
):
    card_player = player_in4[1:]
    # draw normal_card
    for i in range(19):
        if i < 15:
            if card_player[i] > 0:
                draw_cards(
                    background,
                    np.array([i]),
                    arr_player_card_coordinate[player_id][i][0],
                    arr_player_card_coordinate[player_id][i][1],
                    rotate=player_id,
                )
                if i < 12:
                    draw.text(arr_count_card_coor[player_id][i], f"{card_player[i]}")

        else:
            if card_player[i] > 0:
                draw_cards(
                    background,
                    np.array([i]),
                    arr_player_card_coordinate[player_id][i][0],
                    arr_player_card_coordinate[player_id][i][1],
                    rotate=player_id,
                )
            else:
                draw_cards(
                    background,
                    np.array([i + 4]),
                    arr_player_card_coordinate[player_id][i][0],
                    arr_player_card_coordinate[player_id][i][1],
                    rotate=player_id,
                )
    return background


def draw_other(state, draw, background, myFont):
    phase = np.where(state[122:129])[0][0] + 1
    last_dice_val = state[117]
    robber_coor = np.array([[550, 550], [940, 320], [550, 240], [240, 320]])
    coin_coor = np.array([[270, 640], [1000, 80], [270, 130], [130, 80]])
    player_coin = state[0:80:20]

    draw.text((750, 300), f"Phase:{phase}", (255, 0, 0), font=myFont)
    draw.text((750, 330), f"Add_turn:{state[116]}", (255, 0, 0), font=myFont)
    draw.text((750, 360), f"Val_dice:{last_dice_val}", (255, 0, 0), font=myFont)

    for id in range(4):
        draw.text(coin_coor[id], f"{int(player_coin[id])}đ", (255, 255, 0), font=myFont)

    if phase == 5:
        player_robbed = np.where(state[118:122])[0][0]
        background.paste(
            sprites.robber,
            (robber_coor[player_robbed][0], robber_coor[player_robbed][1]),
        )

    elif phase == 6:
        player_robbed = np.where(state[118:122])[0][0]
        background.paste(
            sprites.robber,
            (robber_coor[player_robbed][0], robber_coor[player_robbed][1]),
        )

        card_give = np.where(state[104:116])[0][0]
        card_give_coor = np.array(
            [
                [280, 720],
                [340, 720],
                [400, 720],
                [460, 720],
                [520, 720],
                [580, 720],
                [640, 720],
                [700, 720],
                [760, 720],
                [820, 720],
                [880, 720],
                [940, 720],
            ]
        )
        draw.text(card_give_coor[card_give], f"G", (255, 255, 255), font=myFont)

    elif phase == 7:
        card_buy_in_turn = state[92:104] > 0
        card_buy_coor = np.array(
            [
                [280, 720],
                [340, 720],
                [400, 720],
                [460, 720],
                [520, 720],
                [580, 720],
                [640, 720],
                [700, 720],
                [760, 720],
                [820, 720],
                [880, 720],
                [940, 720],
            ]
        )
        for i in range(len(card_buy_coor)):
            if card_buy_in_turn[i]:
                draw.text(card_buy_coor[i], f"V", (0, 255, 0), font=myFont)

    return background
