import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from src.Base.Splendor_v2 import env as _env
from env import SHORT_PATH

NORMAL_CARD, NOBLE_CARD = _env.normal_cards_infor, _env.noble_cards_infor
IMG_PATH = SHORT_PATH + "src/Base/Splendor_v2/images/"

tl = 3
BG_SIZE = (np.array([2048, 2384]) / tl).astype(np.int64)
BOARD_SIZE = (int(BG_SIZE[0] * 16 / 9), BG_SIZE[1])
CARD_SIZE = (np.array([300, 425]) / (tl * 0.81)).astype(np.int64)
CARD_NOBLE_SIZE = (np.array([300, 300]) / (tl * 0.81)).astype(np.int64)
TOKEN_SIZE = (int(CARD_SIZE[0] * 0.3), int(CARD_SIZE[1] * 0.3))

action_description = {
    0: "Bỏ lượt",
    1: "Lấy thẻ thứ 1",
    2: "Lấy thẻ thứ 2",
    3: "Lấy thẻ thứ 3",
    4: "Lấy thẻ thứ 4",
    5: "Lấy thẻ thứ 5",
    6: "Lấy thẻ thứ 6",
    7: "Lấy thẻ thứ 7",
    8: "Lấy thẻ thứ 8",
    9: "Lấy thẻ thứ 9",
    10: "Lấy thẻ thứ 10",
    11: "Lấy thẻ thứ 11",
    12: "Lấy thẻ thứ 12",
    13: "Mở thẻ đang úp thứ 1",
    14: "Mở thẻ đang úp thứ 2",
    15: "Mở thẻ đang úp thứ 3",
    16: "Úp thẻ thứ 1",
    17: "Úp thẻ thứ 2",
    18: "Úp thẻ thứ 3",
    19: "Úp thẻ thứ 4",
    20: "Úp thẻ thứ 5",
    21: "Úp thẻ thứ 6",
    22: "Úp thẻ thứ 7",
    23: "Úp thẻ thứ 8",
    24: "Úp thẻ thứ 9",
    25: "Úp thẻ thứ 10",
    26: "Úp thẻ thứ 11",
    27: "Úp thẻ thứ 12",
    28: "Úp thẻ ẩn loại 1",
    29: "Úp thẻ ẩn loại 2",
    30: "Úp thẻ ẩn loại 3",
    31: "Lấy nguyên liệu red",
    32: "Lấy nguyên liệu blue",
    33: "Lấy nguyên liệu green",
    34: "Lấy nguyên liệu black",
    35: "Lấy nguyên liệu white",
    36: "Trả nguyên liệu red",
    37: "Trả nguyên liệu blue",
    38: "Trả nguyên liệu green",
    39: "Trả nguyên liệu black",
    40: "Trả nguyên liệu white",
    41: "Trả nguyên liệu yellow",
}


class Env_components:
    def __init__(self, env, winner, list_other, lv1, lv2, lv3) -> None:
        self.env = env
        self.winner = winner
        self.list_other = list_other
        self.lv1 = lv1
        self.lv2 = lv2
        self.lv3 = lv3
        self.cc = 0


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""
    return f"{action_description[action]}"


def get_env_components():
    env, lv1, lv2, lv3 = _env.initEnv()
    winner = _env.checkEnded(env)
    list_other = np.array([-1, 1, 2, 3])
    np.random.shuffle(list_other)
    while list_other[-1] == -1:
        np.random.shuffle(list_other)
    env_components = Env_components(env, winner, list_other, lv1, lv2, lv3)
    return env_components


class Sprites:
    def __init__(self) -> None:
        self.font = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", int(120 / tl)
        )
        self.font_max = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", int(200 / tl)
        )
        self.font2 = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", int(60 / tl)
        )

        self.im = Image.new("RGB", BOARD_SIZE, "black")
        self.background = Image.open(IMG_PATH + "bg.webp").resize(BG_SIZE)
        self.list_token_name = ["red", "blue", "green", "black", "white", "yellow"]

        self.list_img_card = [
            Image.open(IMG_PATH + f"Cards/{card}.png").resize(CARD_SIZE)
            for card in range(90)
        ]
        self.list_img_card_noble = [
            Image.open(IMG_PATH + f"Cards/{card}.png").resize(CARD_NOBLE_SIZE)
            for card in range(90, 100)
        ]
        self.list_img_card_hide = [
            Image.open(IMG_PATH + f"Cards/hide_card_{card}.png").resize(CARD_SIZE)
            for card in range(1, 4)
        ]
        self.list_img_token = [
            Image.open(IMG_PATH + f"Tokens/{token}.png").resize(TOKEN_SIZE)
            for token in self.list_token_name
        ]


class Params:
    def __init__(self) -> None:
        pass


_d_ = int(BG_SIZE[0] * 0.02)
_d2_ = int(BG_SIZE[0] * 0.05)
params = Params()
sprites = Sprites()


def draw_cards_image(
    background,
    list_id_noble=np.full(5, 0),
    list_id_normal=np.full((4, 3), 0),
    list_hide_card=np.full(3, 0),
):
    #  Draw cards image on background image
    y = int(BG_SIZE[1] * 0.77)
    for d_y in range(4):
        x = int(BG_SIZE[0] * 0.01)
        for d_x in range(5):
            if d_y == 3:
                if d_x < len(list_id_noble):
                    background.paste(
                        sprites.list_img_card_noble[list_id_noble[d_x]],
                        (x, int(y * 2.1)),
                    )
            else:
                if d_x == 0:
                    if list_hide_card[d_y] == 1:
                        background.paste(sprites.list_img_card_hide[d_y], (x, y))
                else:
                    id_card_in_list = 4 * d_y + d_x - 1
                    if list_id_normal[id_card_in_list] != -1:
                        background.paste(
                            sprites.list_img_card[list_id_normal[id_card_in_list]],
                            (x, y),
                        )
            x += CARD_SIZE[0] + _d_
        y -= CARD_SIZE[1] + _d_


def draw_tokens_image(
    im,
    list_token_board=np.full(6, 1),
    list_token_const=np.full(5, 1),
    list_token=np.full(6, 1),
    list_token_taken=np.full(5, 1),
):
    #  Draw tokens on board
    x = int(BG_SIZE[0] * 0.22)
    y = int(BOARD_SIZE[1] * 0.04)
    for i in range(6):
        im.paste(sprites.list_img_token[i], (x, y))
        ImageDraw.Draw(im).text(
            (x + TOKEN_SIZE[0], y + int(TOKEN_SIZE[1] / 3)),
            f"{list_token_board[i]}",
            fill="white",
            font=sprites.font,
        )
        x += TOKEN_SIZE[0] + _d2_

    #  Draw tokens const
    y = int(BOARD_SIZE[1] * 0.07)
    y_ = int(BOARD_SIZE[1] * 0.2)
    x = int(BG_SIZE[0] * 1.02)
    ImageDraw.Draw(im).text(
        (x, int(BOARD_SIZE[1] * 0.03)),
        f"Nguyên liệu mặc định:",
        fill="white",
        font=sprites.font2,
    )
    for i in range(5):
        im.paste(sprites.list_img_token[i], (x, y))
        ImageDraw.Draw(im).text(
            (x + TOKEN_SIZE[0], y + int(TOKEN_SIZE[1] / 3)),
            f"{list_token_const[i]}",
            fill="white",
            font=sprites.font,
        )
        x += TOKEN_SIZE[0] + _d2_

    #  Draw tokens on hand
    x = int(BG_SIZE[0] * 1.02)
    ImageDraw.Draw(im).text(
        (x, int(BOARD_SIZE[1] * 0.23)),
        f"Nguyên liệu của bản thân:",
        fill="white",
        font=sprites.font2,
    )
    for i in range(6):
        im.paste(sprites.list_img_token[i], (x, y + y_))
        ImageDraw.Draw(im).text(
            (x + TOKEN_SIZE[0], y + y_ + int(TOKEN_SIZE[1] / 3)),
            f"{list_token[i]}",
            fill="white",
            font=sprites.font,
        )
        x += TOKEN_SIZE[0] + _d2_

    #  Draw tokens taken
    x = int(BG_SIZE[0] * 1.02)
    y = int(BG_SIZE[1] * 0.8)
    ImageDraw.Draw(im).text(
        (x, y * 0.95),
        f"Nguyên liệu đã lấy trong lượt:",
        fill="white",
        font=sprites.font2,
    )
    for i in range(5):
        im.paste(sprites.list_img_token[i], (x, y))
        ImageDraw.Draw(im).text(
            (x + TOKEN_SIZE[0], y + int(TOKEN_SIZE[1] / 3)),
            f"{list_token_taken[i]}",
            fill="white",
            font=sprites.font,
        )
        x += TOKEN_SIZE[0] + _d2_


def draw_down_card(im, state):
    down_card = state[175:208].reshape(3, 11)
    list_down_card = []
    for card in down_card:
        for i in range(len(NORMAL_CARD)):
            if (card == NORMAL_CARD[i]).all():
                list_down_card.append(i)

    x = int(BOARD_SIZE[0] * 0.58)
    y = int(BOARD_SIZE[1] * 0.43)
    for i in range(len(list_down_card)):
        im.paste(sprites.list_img_card[list_down_card[i]], (x, y))
        x += CARD_SIZE[0] + _d_


def draw_score(im, state):
    scores = [state[17], state[213], state[214], state[215]]
    x = int(im.size[0] * 0.925)
    y = int(BOARD_SIZE[1] * 0.08)
    for agent in range(4):
        ImageDraw.Draw(im).text(
            (x, y - int(BOARD_SIZE[1] * 0.03)),
            f"Điểm:",
            fill="white",
            font=sprites.font2,
        )
        ImageDraw.Draw(im).text(
            (x, y), f"{scores[agent]}", fill="white", font=sprites.font_max
        )
        if agent == 0:
            ImageDraw.Draw(im).text(
                (x, y + int(BOARD_SIZE[1] * 0.11)),
                f"YOU",
                fill="white",
                font=sprites.font,
            )
        y += int(im.size[1] / 4)


def draw_line(im, color="White", width=3):
    w = int(im.size[0] * 0.92)
    for i in range(3):
        h = int(im.size[1] / 4) * (i + 1)
        shape = [(w, h), (im.size[0], h)]
        ImageDraw.Draw(im).line(shape, fill=color, width=width)

    ImageDraw.Draw(im).line([(w, 0), (w, im.size[1])], fill=color, width=width)


def get_id_card(background, state=None):
    cards_noble = state[150:175].reshape(5, 5)
    list_id_noble = []
    for card in cards_noble:
        for i in range(len(NOBLE_CARD)):
            if (card == NOBLE_CARD[i]).all():
                list_id_noble.append(i)

    card_normal = state[18:150].reshape(12, 11)
    list_id_normal = []
    for card in card_normal:
        for i in range(len(NORMAL_CARD)):
            if (card == NORMAL_CARD[i]).all():
                list_id_normal.append(i)
    list_card_normal_convert = np.full(12, -1)
    j = 0
    for i in range(len(list_id_normal)):
        if list_id_normal[i] < 40:
            list_card_normal_convert[j] = list_id_normal[i]
            j += 1
        elif list_id_normal[i] < 70:
            if j < 4:
                j = 4
            list_card_normal_convert[j] = list_id_normal[i]
            j += 1
        elif list_id_normal[i] < 90:
            if j < 8:
                j = 8
            list_card_normal_convert[j] = list_id_normal[i]
            j += 1

    list_hide_card = state[216:219]
    draw_cards_image(
        background,
        list_id_noble=list_id_noble,
        list_id_normal=list_card_normal_convert,
        list_hide_card=list_hide_card,
    )


def get_state_image(state=None):
    state = state.astype(np.int64)
    background = sprites.background.copy()
    im = sprites.im.copy()
    draw_line(im)

    get_id_card(background, state)
    im.paste(background, (0, 0))

    draw_tokens_image(
        im,
        list_token_board=state[0:6],
        list_token=state[6:12],
        list_token_const=state[12:17],
        list_token_taken=state[208:213],
    )

    draw_down_card(im, state)
    draw_score(im, state)
    return im


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    if not action is None:
        (
            env_components.env,
            env_components.lv1,
            env_components.lv2,
            env_components.lv3,
        ) = _env.stepEnv(
            action,
            env_components.env,
            env_components.lv1,
            env_components.lv2,
            env_components.lv3,
        )

    if env_components.winner == 0:  #  no ended
        while env_components.env[100] <= 400 and env_components.cc < 10000:
            idx = env_components.env[100] % 4
            state = _env.getAgentState(
                env_components.env,
                env_components.lv1,
                env_components.lv2,
                env_components.lv3,
            )
            if env_components.list_other[idx] == -1:
                break

            agent = list_agent[env_components.list_other[idx] - 1]
            data = list_data[env_components.list_other[idx] - 1]
            action, data = agent(state, data)
            (
                env_components.env,
                env_components.lv1,
                env_components.lv2,
                env_components.lv3,
            ) = _env.stepEnv(
                action,
                env_components.env,
                env_components.lv1,
                env_components.lv2,
                env_components.lv3,
            )
            env_components.winner = _env.checkEnded(env_components.env)
            if env_components.winner != 0:
                break
            env_components.cc += 1

    if env_components.winner == 0:  # no winner
        state = _env.getAgentState(
            env_components.env,
            env_components.lv1,
            env_components.lv2,
            env_components.lv3,
        )
        win = -1
    else:  # have winner
        my_idx = np.where(env_components.list_other == -1)[0][0]
        env = env_components.env.copy()
        env[100] = my_idx
        state[220] = 1
        state = _env.getAgentState(
            env, env_components.lv1, env_components.lv2, env_components.lv3
        )
        if my_idx == (env_components.winner - 1):
            win = 1
        else:
            win = 0

        #  Chạy turn cuối cho 3 bot hệ thống
        for p_idx in range(4):
            if p_idx != my_idx:
                env[100] = p_idx
                _state = _env.getAgentState(
                    env, env_components.lv1, env_components.lv2, env_components.lv3
                )
                _state[220] = 1
                agent = list_agent[env_components.list_other[p_idx] - 1]
                data = list_data[env_components.list_other[p_idx] - 1]
                action, data = agent(_state, data)

    return win, state, env_components
