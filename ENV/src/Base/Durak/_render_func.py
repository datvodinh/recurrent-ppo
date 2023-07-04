import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from src.Base.Durak import env as _env
from env import SHORT_PATH

IMG_PATH = SHORT_PATH + "src/Base/TLMN/images/"
tl = 3
BG_SIZE = (1680, 720)
CARD_SIZE = (80, 112)
SUIT_SIZE = (100, 100)


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

        self.background = Image.open(IMG_PATH + "background.png").resize(BG_SIZE)
        card_values = [
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "J",
            "Q",
            "K",
            "A",
        ]
        card_suits = ["Spade", "Club", "Diamond", "Heart"]
        self.cards = []
        self.card_name = []
        path = "src/Base/Durak/images/"
        for suit in card_suits:
            for value in card_values:
                self.cards.append(
                    Image.open(IMG_PATH + f"{value}-{suit}.png").resize(CARD_SIZE)
                )
                self.card_name.append(f"{value}-{suit}")

        self.img_suit = [
            Image.open(f"{path}{suit}.png").resize(SUIT_SIZE) for suit in card_suits
        ]

        self.card_back = Image.open(IMG_PATH + "Card_back.png").resize(CARD_SIZE)
        self.faded_card_back = self.card_back.copy()
        br = ImageEnhance.Brightness(self.faded_card_back)
        self.faded_card_back = br.enhance(0.5)
        ct = ImageEnhance.Contrast(self.faded_card_back)
        self.faded_card_back = ct.enhance(0.5)


class Params:
    def __init__(self) -> None:
        self.center_card_x = int(BG_SIZE[0] * 0.5)
        self.center_card_y = int((BG_SIZE[1] - CARD_SIZE[1]) * 0.5)
        self.list_coords_0 = [
            (self.center_card_x, int(0.92 * BG_SIZE[1] - CARD_SIZE[1])),
            (int(0.82 * BG_SIZE[0]), self.center_card_y),
            (self.center_card_x, int(0.08 * BG_SIZE[1])),
            (int(0.18 * BG_SIZE[0]), self.center_card_y),
        ]

        x_0 = int(BG_SIZE[0] * 0.32)
        x_1 = int(BG_SIZE[0] * 0.68)
        y_0 = int(0.2 * BG_SIZE[1] - 0.25 * CARD_SIZE[1])
        y_1 = int(0.8 * BG_SIZE[1] - 0.75 * CARD_SIZE[1])
        self.list_coords_1 = [(x_0, y_1), (x_1, y_1), (x_1, y_0), (x_0, y_0)]


sprites = Sprites()
params = Params()
_d_ = int(CARD_SIZE[0] * 0.2)


def draw_cards(bg, cards, s, y, back=False, faded=False, main_card=False):
    n = cards.shape[0]
    y = round(y)
    if back:
        if faded:
            im = sprites.faded_card_back
        else:
            im = sprites.card_back

        for i in range(n):
            bg.paste(im, (round(s + _d_ * i), y))
        ImageDraw.Draw(bg).text(
            (round(s + _d_ * (n + 1)), y), f"{n}", fill="white", font=sprites.font
        )

    else:
        for i in range(n):
            bg.paste(sprites.cards[cards[i]], (round(s + _d_ * i), y))
        if main_card == True:
            ImageDraw.Draw(bg).text(
                (round(s + _d_ * (n + 4)), y), f"{n}", fill="white", font=sprites.font
            )


def get_state_image(state=None):
    background = sprites.background.copy()
    state = state.astype(np.int64)

    #  Draw my card
    my_cards = np.where(state[0:52] == 1)[0]
    n = my_cards.shape[0]
    w = CARD_SIZE[0] + _d_ * (n - 1)
    x = int(params.list_coords_0[0][0] - 0.5 * w)
    draw_cards(
        background,
        my_cards,
        x,
        params.list_coords_0[0][1],
        back=False,
        faded=False,
        main_card=True,
    )

    #  Draw other cards
    for k in range(1, 4):
        n = state[162 + k]
        w = CARD_SIZE[0] + _d_ * (n - 1)
        if k == 1:
            s = params.list_coords_0[1][0] - w
        elif k == 2:
            s = params.list_coords_0[2][0] - 0.5 * w
        else:
            s = params.list_coords_0[3][0]

        draw_cards(background, np.full(int(n), 0), s, params.list_coords_0[k][1], True)

    #  card defend successful
    cur_cards = np.where(state[52:104] == 1)[0]
    n = cur_cards.shape[0]
    w = CARD_SIZE[0] + _d_ * (n - 1)
    s = params.center_card_x - 0.5 * w
    y = params.center_card_y + int(CARD_SIZE[1] / 1.9)
    draw_cards(background, cur_cards, s, y)

    #  card have to defend this round
    cur_cards = np.where(state[104:156] == 1)[0]
    n = cur_cards.shape[0]
    w = CARD_SIZE[0] + _d_ * (n - 1)
    s = params.center_card_x - 0.5 * w
    y = params.center_card_y - int(CARD_SIZE[1] / 1.9)
    draw_cards(background, cur_cards, s, y)

    # Draw trump card
    x = int(BG_SIZE[0] * 0.19)
    y = int(BG_SIZE[1] * 0.1)
    id_trump = np.where(state[158:162] == 1)[0][0]
    background.paste(sprites.img_suit[id_trump], (x, y))

    # attack or defense
    if state[156] == 1:
        mode_action = "attack"
    else:
        mode_action = "defense"
    ImageDraw.Draw(background).text(
        (int(x * 1.35), int(y * 1.65)),
        f"{mode_action}",
        fill="white",
        font=sprites.font,
    )

    # Draw count card on table
    x = int(BG_SIZE[0] * 0.75)
    y = int(BG_SIZE[1] * 0.1)
    background.paste(sprites.card_back, (x, y))
    ImageDraw.Draw(background).text(
        (x, y + int(CARD_SIZE[1] * 0.2)),
        f"{state[162]}",
        fill="black",
        font=sprites.font_max,
    )
    return background


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""
    if action == 52:
        return "pass"
    #  print(sprites.card_name)
    return sprites.card_name[action]


class Env_components:
    def __init__(self, env, winner, list_other) -> None:
        self.env = env
        self.winner = winner
        self.list_other = list_other


def get_env_components():
    env = _env.initEnv()
    winner = _env.checkEnded(env)
    list_other = np.array([-1, 1, 2, 3])
    np.random.shuffle(list_other)

    env_components = Env_components(env, winner, list_other)
    return env_components


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    if not action is None:
        _env.stepEnv(action, env_components.env)

    env_components.winner = _env.checkEnded(env_components.env)
    turn = 0
    if env_components.winner == -1:
        while True:
            if env_components.env[53] == 1:
                p_idx = int(env_components.env[58] - 1)
            else:
                p_idx = int(env_components.env[54:57][int(env_components.env[59])] - 1)
            if env_components.list_other[p_idx] == -1:
                break

            state = _env.getAgentState(env_components.env)
            agent = list_agent[env_components.list_other[p_idx] - 1]
            data = list_data[env_components.list_other[p_idx] - 1]
            action, data = agent(state, data)
            _env.stepEnv(action, env_components.env)

            env_components.winner = _env.checkEnded(env_components.env)
            if env_components.winner != -1:
                break

    if env_components.winner == -1:
        state = _env.getAgentState(env_components.env)
        win = -1
    else:
        my_idx = np.where(env_components.list_other == -1)[0][0]
        env = env_components.env.copy()
        env[80] = 1
        env[53] = 1
        env[58] = my_idx * 1.0 + 1.0
        state = _env.getAgentState(env)
        if my_idx == env_components.winner:
            win = 1
        else:
            win = 0

        #  Chạy turn cuối cho 3 bot hệ thống
        for p_idx in range(4):
            if p_idx != my_idx:
                env[53] = 1
                env[58] = p_idx * 1.0 + 1.0
                _state = _env.getAgentState(env)
                agent = list_agent[env_components.list_other[p_idx] - 1]
                data = list_data[env_components.list_other[p_idx] - 1]
                action, data = agent(_state, data)

    return win, state, env_components
