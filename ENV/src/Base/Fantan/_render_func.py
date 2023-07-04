from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
from env import SHORT_PATH
from src.Base.Fantan import env as _env

IMG_PATH = SHORT_PATH + "src/Base/Fantan/images/"
BG_SIZE = (2100, 900)
CARD_SIZE = (100, 130)


class Sprites:
    def __init__(self) -> None:
        self.background = Image.open(IMG_PATH + "Fantan_background.png").resize(BG_SIZE)
        card_values = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        card_suits = ["Heart", "Diamond", "Club", "Spade"]
        self.cards = []
        self.action_name = []
        for suit in card_suits:
            for value in card_values:
                self.cards.append(
                    Image.open(IMG_PATH + f"{value}-{suit}.png").resize(CARD_SIZE)
                )
                self.action_name.append(f"{value}-{suit}")

        self.card_back = Image.open(IMG_PATH + "Card_back.png").resize(CARD_SIZE)
        self.faded_card_back = self.card_back.copy()
        br = ImageEnhance.Brightness(self.faded_card_back)
        self.faded_card_back = br.enhance(0.5)
        ct = ImageEnhance.Contrast(self.faded_card_back)
        self.faded_card_back = ct.enhance(0.5)
        self.gold = (
            Image.open(IMG_PATH + "poker_chip.png").resize((70, 70)).convert("RGBA")
        )


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
        self.font32 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 40)


params = Params()


_d_ = CARD_SIZE[0] * 0.2


def draw_cards(bg, cards, s, y, back=False, faded=False):
    n = cards.shape[0]
    y = round(y)
    if back:
        if faded:
            im = sprites.faded_card_back
        else:
            im = sprites.card_back

        for i in range(n):
            bg.paste(im, (round(s + _d_ * i), y))
    else:
        for i in range(n):
            bg.paste(sprites.cards[cards[i]], (round(s + _d_ * i), y))


def draw_outlined_text(draw, text, font, pos, color, opx):
    o_color = (255 - color[0], 255 - color[1], 255 - color[2])
    x = pos[0]
    y = pos[1]
    draw.text((x + opx, y + opx), text, o_color, font)
    draw.text((x - opx, y + opx), text, o_color, font)
    draw.text((x + opx, y - opx), text, o_color, font)
    draw.text((x - opx, y - opx), text, o_color, font)
    draw.text(pos, text, color, font)


def get_state_image(state=None):
    background = sprites.background.copy()
    if state is None:
        return background

    draw = ImageDraw.ImageDraw(background)
    my_cards = np.where(state[0:52])[0]
    n = my_cards.shape[0]
    w = CARD_SIZE[0] + _d_ * (n - 1)
    s = params.list_coords_0[0][0] - 0.5 * w
    draw_cards(background, my_cards, s, params.list_coords_0[0][1])

    for k in range(1, 4):
        faded = False
        n = state[104 + k]
        w = CARD_SIZE[0] + _d_ * (n - 1)
        if k == 1:
            s = params.list_coords_0[3][0]
        elif k == 2:
            s = params.list_coords_0[2][0] - 0.5 * w
        else:
            s = params.list_coords_0[1][0] - w

        draw_cards(
            background,
            np.full(int(n), 0),
            s,
            params.list_coords_0[2][1] + 120 if k != 2 else params.list_coords_0[2][1],
            True,
            faded,
        )

    cards_can_play = np.where(state[52:104] == 1)[0]
    n = cards_can_play.shape[0]
    w = CARD_SIZE[0] + _d_ * (n - 1)
    s = params.list_coords_0[0][0] - 0.5 * w
    draw_cards(background, cards_can_play, s, params.list_coords_0[1][1])

    my_chips = int(state[104])
    text = str(my_chips)
    bbox = draw.textbbox((0, 0), text, params.font32)
    draw_outlined_text(
        draw,
        text,
        params.font32,
        (1050 - bbox[2] // 2, 590 - bbox[3] // 2),
        (255, 255, 255),
        1,
    )

    my_chips = int(state[110])
    text = str(my_chips)
    bbox = draw.textbbox((0, 0), text, params.font32)
    draw_outlined_text(
        draw,
        text,
        params.font32,
        (1050 - bbox[2] // 2, 300 - bbox[3] // 2),
        (255, 255, 255),
        1,
    )

    my_chips = int(state[109])
    text = str(my_chips)
    bbox = draw.textbbox((0, 0), text, params.font32)
    draw_outlined_text(
        draw,
        text,
        params.font32,
        (435 - bbox[2] // 2, 425 - bbox[3] // 2),
        (255, 255, 255),
        1,
    )

    my_chips = int(state[111])
    text = str(my_chips)
    bbox = draw.textbbox((0, 0), text, params.font32)
    draw_outlined_text(
        draw,
        text,
        params.font32,
        (1670 - bbox[2] // 2, 425 - bbox[3] // 2),
        (255, 255, 255),
        1,
    )

    return background


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""

    if action == 52:
        return "Skip"

    return sprites.action_name[action]


class Env_components:
    def __init__(self) -> None:
        self.allGame = True
        self.saveStoreChip = np.array([50, 50, 50, 50])
        self.idxPlayerChip = np.array([21, 35, 49, 63])
        self.env = _env.initEnv()
        self.env[self.idxPlayerChip] = self.saveStoreChip
        self.oneGame = True
        self.i = 0
        self.list_other = np.array([-1, 1, 2, 3])
        np.random.shuffle(self.list_other)


def get_env_components():
    return Env_components()


def step_env(com: Env_components, action, list_agent, list_data):
    stepEnvReturn = _env.stepEnv(action, com.env)
    if stepEnvReturn == -1:
        com.oneGame = False
        com.env[8 + com.i * 14 + 13] += com.env[65]
        com.saveStoreChip = com.env[com.idxPlayerChip]
        com.env[65] = 0
    elif stepEnvReturn == -2:
        com.env[66] = 1
        player_chip = com.env[com.idxPlayerChip]
        player_id_not_0_chip = np.where(player_chip > 0)[0]
        arr_player_cards = np.zeros(13 * 3)
        for i in range(len(player_id_not_0_chip)):
            player_cards = com.env[
                8 + player_id_not_0_chip[i] * 13 : 8 + player_id_not_0_chip[i] * 13 + 13
            ]
            arr_player_cards[i * 13 : i * 13 + 13] = player_cards.astype(np.float64)

        arr_player_cards = np.reshape(arr_player_cards, (3, 13))
        player_card_len = np.array(
            [len(np.where(player_cards > -1)) for player_cards in arr_player_cards]
        )
        player_lowest_card = np.argmax(player_card_len)
        player_lowest_card_id = player_id_not_0_chip[player_lowest_card]
        com.env[com.idxPlayerChip[player_lowest_card_id]] += com.env[65]
        com.env[65] = 0

        env = com.env.copy()
        for pIdx in range(4):
            env[64] = pIdx
            state = _env.getAgentState(env)
            if com.list_other[pIdx] == -1:
                _state_ = state.copy()
                if _env.getReward(state) == 1:
                    win = 1
                else:
                    win = 0
            else:
                agent = list_agent[com.list_other[pIdx] - 1]
                data = list_data[com.list_other[pIdx] - 1]
                action, data = agent(state, data)

        com.allGame = False
        return win, _state_

    return -1, None


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    if not action is None:
        win, state = step_env(env_components, action, list_agent, list_data)
        if win != -1:
            return win, state, env_components

        env_components.i += 1

    while True:
        if env_components.i == 4:
            if not env_components.oneGame:
                env_components.env = _env.initEnv()
                env_components.env[
                    env_components.idxPlayerChip
                ] = env_components.saveStoreChip
                env_components.oneGame = True

            env_components.i = 0

        env_components.env[64] = env_components.i
        state = _env.getAgentState(env_components.env)
        if env_components.list_other[env_components.i] == -1:
            return -1, state, env_components

        agent = list_agent[env_components.list_other[env_components.i] - 1]
        data = list_data[env_components.list_other[env_components.i] - 1]
        action, data = agent(state, data)
        win, state = step_env(env_components, action, list_agent, list_data)
        if win != -1:
            return win, state, env_components

        env_components.i += 1
