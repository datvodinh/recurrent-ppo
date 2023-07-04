import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.Base.Splendor import env as _env
from src.Base.Splendor.env import __NOBLE_CARD__, __NORMAL_CARD__
from env import SHORT_PATH

IMG_PATH = SHORT_PATH + "src/Base/Splendor/images/"


class Sprites:
    def __init__(self) -> None:
        self.background = Image.open(IMG_PATH + "background.png").resize((2100, 900))
        self._background_ = self.background.copy()

        self.cards = []
        self.small_cards = []
        for i in range(100):
            card = Image.open(IMG_PATH + f"{i}.png")
            if i < 90:
                card = card.resize((120, 168)).convert("RGBA")
            else:
                card = card.resize((120, 120)).convert("RGBA")

            self.cards.append(card)
            if i < 90:
                self.small_cards.append(card.resize((80, 112)))

        self.nob_cards = self.cards[90:]
        self.cards = self.cards[:90]

        self.cards_back = []
        self.small_cards_back = []
        for i in range(1, 4):
            card = (
                Image.open(IMG_PATH + f"hide_card_{i}.png")
                .resize((120, 168))
                .convert("RGBA")
            )
            self.cards_back.append(card)
            self.small_cards_back.append(card.resize((80, 112)))

        self.gems = []
        for name in ["red", "blue", "green", "black", "white", "gold"]:
            gem = Image.open(IMG_PATH + f"{name}.png").resize((50, 50)).convert("RGBA")
            self.gems.append(gem)


sprites = Sprites()


class Params:
    def __init__(self) -> None:
        self.font28 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 28)
        self.font32 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 32)
        self.font40 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 40)


params = Params()


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
    if state is None:
        return sprites.background

    bg = sprites.background.copy()

    #  Draw
    draw = ImageDraw.ImageDraw(bg)

    #  Noble
    for i in range(5):
        nob = state[6 + 6 * i : 12 + 6 * i]
        if (nob != 0).any():
            for k in range(10):
                if (__NOBLE_CARD__[k] == nob).all():
                    nob_id = k
                    break

            bg.paste(
                sprites.nob_cards[k], (1150 + 160 * i, 120), sprites.nob_cards[nob_id]
            )

    #  Normal card
    nm_cards_ = []
    for i in range(12):
        card = state[36 + 11 * i : 47 + 11 * i]
        if (card != 0).any():
            for k in range(90):
                if (__NORMAL_CARD__[k] == card).all():
                    nm_cards_.append(k)
                    break
        else:
            nm_cards_.append(-1)

    nm_cards = [-1] * 12
    nm_cards[0:4] = nm_cards_[8:12]
    nm_cards[4:8] = nm_cards_[4:8]
    nm_cards[8:12] = nm_cards_[0:4]

    for i in range(12):
        if nm_cards[i] != -1:
            a = i % 4
            b = i // 4
            bg.paste(
                sprites.cards[nm_cards[i]],
                (1310 + 160 * a, 270 + 200 * b),
                sprites.cards[nm_cards[i]],
            )
            text = str(nm_cards[i])
            if len(text) == 1:
                text = "0" + text

            draw_outlined_text(
                draw, text, params.font32, (1390 + 160 * a, 405 + 200 * b), (0, 0, 0), 1
            )

    for i in range(3):
        card = state[168 + 11 * i : 179 + 11 * i]
        if (card != 0).any():
            for k in range(90):
                if (__NORMAL_CARD__[k] == card).all():
                    card_id = k
                    break
            bg.paste(
                sprites.cards[card_id], (360 + 140 * i, 690), sprites.cards[card_id]
            )
            text = str(card_id)
            if len(text) == 1:
                text = "0" + text

            draw_outlined_text(
                draw, text, params.font32, (440 + 140 * i, 825), (0, 0, 0), 1
            )

    for p in range(3):
        temp = []
        for i in range(3):
            temp += [i] * int(state[249 + 3 * p + i])

        for k in range(len(temp)):
            bg.paste(
                sprites.small_cards_back[temp[k]],
                (380 + 140 * k, 90 + 200 * p),
                sprites.small_cards_back[temp[k]],
            )

    for i in range(3):
        if state[264 + i] == 1:
            k = 2 - i
            bg.paste(
                sprites.cards_back[i], (1150, 270 + 200 * k), sprites.cards_back[i]
            )

    for i in range(6):
        text = str(int(state[i]))
        draw_outlined_text(draw, text, params.font28, (1290 + 75 * i, 20), (0, 0, 0), 1)
        text = str(int(state[201 + i]))
        draw_outlined_text(draw, text, params.font28, (340 + 75 * i, 620), (0, 0, 0), 1)
        text = str(int(state[213 + i]))
        draw_outlined_text(draw, text, params.font28, (340 + 75 * i, 20), (0, 0, 0), 1)
        text = str(int(state[225 + i]))
        draw_outlined_text(draw, text, params.font28, (340 + 75 * i, 220), (0, 0, 0), 1)
        text = str(int(state[237 + i]))
        draw_outlined_text(draw, text, params.font28, (340 + 75 * i, 420), (0, 0, 0), 1)

        if i != 5:
            text = str(int(state[207 + i]))
            draw_outlined_text(
                draw, text, params.font28, (340 + 75 * i, 650), (0, 0, 0), 1
            )
            text = str(int(state[219 + i]))
            draw_outlined_text(
                draw, text, params.font28, (340 + 75 * i, 50), (0, 0, 0), 1
            )
            text = str(int(state[231 + i]))
            draw_outlined_text(
                draw, text, params.font28, (340 + 75 * i, 250), (0, 0, 0), 1
            )
            text = str(int(state[243 + i]))
            draw_outlined_text(
                draw, text, params.font28, (340 + 75 * i, 450), (0, 0, 0), 1
            )

    text = str(int(state[212]))
    bbox = draw.textbbox((0, 0), text, params.font40)
    draw_outlined_text(
        draw, text, params.font40, (200 - bbox[2] / 2, 700), (255, 255, 255), 1
    )
    text = str(int(state[224]))
    bbox = draw.textbbox((0, 0), text, params.font40)
    draw_outlined_text(
        draw, text, params.font40, (200 - bbox[2] / 2, 100), (255, 255, 255), 1
    )
    text = str(int(state[236]))
    bbox = draw.textbbox((0, 0), text, params.font40)
    draw_outlined_text(
        draw, text, params.font40, (200 - bbox[2] / 2, 300), (255, 255, 255), 1
    )
    text = str(int(state[248]))
    bbox = draw.textbbox((0, 0), text, params.font40)
    draw_outlined_text(
        draw, text, params.font40, (200 - bbox[2] / 2, 500), (255, 255, 255), 1
    )

    return bg


GEMS = ["ruby", "sapphire", "emerald", "onyx", "diamond", "gold"]


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""

    if action < 5:
        return f"Take one {GEMS[action]}"

    if action < 95:
        card_id = action - 5
        return f"Buy card, id: {card_id}"

    if action < 185:
        card_id = action - 95
        return f"Hold card, id: {card_id}"

    if action < 190:
        return f"Discard one {GEMS[action-185]}"

    if action < 193:
        return f"Hold hidden card level {action - 189}"

    if action == 193:
        return "Skip turn"


class Env_components:
    def __init__(self, env, lv1, lv2, lv3, winner, list_other) -> None:
        self.env = env
        self.lv1 = lv1
        self.lv2 = lv2
        self.lv3 = lv3
        self.winner = winner
        self.list_other = list_other


def get_env_components():
    env, lv1, lv2, lv3 = _env.initEnv()
    winner = _env.checkEnded(env)
    list_other = np.array([-1, 1, 2, 3])
    np.random.shuffle(list_other)
    return Env_components(env, lv1, lv2, lv3, winner, list_other)


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    env, lv1, lv2, lv3 = (
        env_components.env,
        env_components.lv1,
        env_components.lv2,
        env_components.lv3,
    )
    if not action is None:
        _env.stepEnv(action, env, lv1, lv2, lv3)
    env_components.winner = _env.checkEnded(env)
    if env_components.winner[0] == -1:
        while env[83] < 400:
            p_idx = env[83] % 4
            if env_components.list_other[p_idx] == -1:
                break

            state = _env.getAgentState(env, lv1, lv2, lv3)
            agent = list_agent[env_components.list_other[p_idx] - 1]
            data = list_data[env_components.list_other[p_idx] - 1]
            action, data = agent(state, data)
            _env.stepEnv(action, env, lv1, lv2, lv3)
            env_components.winner = _env.checkEnded(env)
            if env_components.winner[0] != -1:
                break

        if env_components.winner[0] != -1 or env[83] >= 400:
            env[89] = 1
    else:
        env[89] = 1

    if env_components.env[89] == 0:
        state = _env.getAgentState(env, lv1, lv2, lv3)
        win = -1
    else:
        my_idx = np.where(env_components.list_other == -1)[0][0]
        env_ = env.copy()
        env_[83] = my_idx
        state = _env.getAgentState(env_, lv1, lv2, lv3)
        if my_idx in env_components.winner:
            win = 1
        else:
            win = 0

        #  Chạy turn cuối cho 3 bot hệ thống
        for p_idx in range(4):
            if p_idx != my_idx:
                env_[83] = p_idx
                _state = _env.getAgentState(env_, lv1, lv2, lv3)
                agent = list_agent[env_components.list_other[p_idx] - 1]
                data = list_data[env_components.list_other[p_idx] - 1]
                action, data = agent(_state, data)

    return win, state, env_components
