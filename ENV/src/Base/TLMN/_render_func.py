import numpy as np
from PIL import Image, ImageEnhance

from src.Base.TLMN import env as _env
from env import SHORT_PATH

IMG_PATH = SHORT_PATH + "src/Base/TLMN/images/"
BG_SIZE = (1680, 720)
CARD_SIZE = (80, 112)


class Sprites:
    def __init__(self) -> None:
        self.background = Image.open(IMG_PATH + "background.png").resize(BG_SIZE)
        card_values = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
        card_suits = ["Spade", "Club", "Diamond", "Heart"]
        self.cards = []
        for value in card_values:
            for suit in card_suits:
                self.cards.append(
                    Image.open(IMG_PATH + f"{value}-{suit}.png").resize(CARD_SIZE)
                )

        self.card_back = Image.open(IMG_PATH + "Card_back.png").resize(CARD_SIZE)
        self.faded_card_back = self.card_back.copy()
        br = ImageEnhance.Brightness(self.faded_card_back)
        self.faded_card_back = br.enhance(0.5)
        ct = ImageEnhance.Contrast(self.faded_card_back)
        self.faded_card_back = ct.enhance(0.5)


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


def get_state_image(state=None):
    background = sprites.background.copy()
    if state is None:
        return background

    my_cards = np.where(state[0:52])[0]
    n = my_cards.shape[0]
    w = CARD_SIZE[0] + _d_ * (n - 1)
    s = params.list_coords_0[0][0] - 0.5 * w
    draw_cards(background, my_cards, s, params.list_coords_0[0][1])

    for k in range(1, 4):
        faded = not state[103 + k]
        n = state[106 + k]
        w = CARD_SIZE[0] + _d_ * (n - 1)
        if k == 1:
            s = params.list_coords_0[1][0] - w
        elif k == 2:
            s = params.list_coords_0[2][0] - 0.5 * w
        else:
            s = params.list_coords_0[3][0]

        draw_cards(
            background, np.full(int(n), 0), s, params.list_coords_0[k][1], True, faded
        )

    cards_played = np.where(state[52:104])[0]
    cur_cards = np.where(state[113:165])[0]
    cards_played = np.setdiff1d(cards_played, cur_cards)

    n = cur_cards.shape[0]
    w = CARD_SIZE[0] + _d_ * (n - 1)
    s = params.center_card_x - 0.5 * w
    draw_cards(background, cur_cards, s, params.center_card_y)

    list_cards_played = np.array_split(cards_played, 4)
    for k in range(4):
        n = list_cards_played[k].shape[0]
        w = CARD_SIZE[0] + _d_ * (n - 1)
        s = params.list_coords_1[k][0] - 0.5 * w
        draw_cards(background, list_cards_played[k], s, params.list_coords_1[k][1])

    return background


hand_name = {
    1: "Single",
    2: "Pair",
    3: "Three of a kind",
    4: "Four of a kind",
    5: "3-card straight",
    6: "4-card straight",
    7: "5-card straight",
    8: "6-card straight",
    9: "7-card straight",
    10: "8-card straight",
    11: "9-card straight",
    12: "10-card straight",
    13: "11-card straight",
    14: "3-pair straight",
    15: "4-pair straight",
}

card_name = []
card_values = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
card_suits = ["Spade", "Club", "Diamond", "Heart"]
for value in card_values:
    for suit in card_suits:
        card_name.append(f"{value}-{suit}")


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""

    if action == 0:
        return "Skip"

    hand, card = _env._ACTIONS_[action]
    return f"{hand_name[hand]}, high card {card_name[card]}"


class Env_components:
    def __init__(self, env, cur_cards, winner, list_other) -> None:
        self.env = env
        self.cur_cards = cur_cards
        self.winner = winner
        self.list_other = list_other


def get_env_components():
    env = _env.initEnv()
    while not _env.check_player_hand(env):
        env = _env.initEnv()

    cur_cards = np.full(0, 0)
    winner = _env.checkEnded(env)
    list_other = np.array([-1, 1, 2, 3])
    np.random.shuffle(list_other)

    env_components = Env_components(env, cur_cards, winner, list_other)
    return env_components


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    if not action is None:
        if action != 0:
            env_components.cur_cards = _env.stepEnv(action, env_components.env)
        else:
            _env.stepEnv(action, env_components.env)

    env_components.winner = _env.checkEnded(env_components.env)
    if env_components.winner == -1:
        while True:
            p_idx = env_components.env[52]
            if env_components.list_other[p_idx] == -1:
                break

            state = _env.getAgentState(env_components.env, env_components.cur_cards)
            agent = list_agent[env_components.list_other[p_idx] - 1]
            data = list_data[env_components.list_other[p_idx] - 1]
            action, data = agent(state, data)
            if action != 0:
                env_components.cur_cards = _env.stepEnv(action, env_components.env)
            else:
                _env.stepEnv(action, env_components.env)

            env_components.winner = _env.checkEnded(env_components.env)
            if env_components.winner != -1:
                break

    if env_components.winner == -1:
        state = _env.getAgentState(env_components.env, env_components.cur_cards)
        win = -1
    else:
        my_idx = np.where(env_components.list_other == -1)[0][0]
        env = env_components.env.copy()
        env[52] = my_idx
        env[57] = my_idx
        cur_cards = np.full(0, 0)
        state = _env.getAgentState(env, cur_cards)
        if my_idx == env_components.winner:
            win = 1
        else:
            win = 0

        #  Chạy turn cuối cho 3 bot hệ thống
        for p_idx in range(4):
            if p_idx != my_idx:
                env[52] = p_idx
                env[57] = p_idx
                _state = _env.getAgentState(env, cur_cards)
                agent = list_agent[env_components.list_other[p_idx] - 1]
                data = list_data[env_components.list_other[p_idx] - 1]
                action, data = agent(_state, data)

    return win, state, env_components
