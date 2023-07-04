import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.Base.Exploding_Kitten import env as _env
from env import SHORT_PATH

IMG_PATH = SHORT_PATH + "src/Base/Exploding_Kitten/images/"


class Params:
    def __init__(self) -> None:
        self.card_order = [
            "Nope",
            "Attack",
            "Skip",
            "Favor",
            "Shuffle",
            "Future",
            "Taco",
            "Rainbow",
            "Beard",
            "Potato",
            "Melon",
            "Defuse",
            "Exploding",
        ]
        self.font32 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 32)
        self.font24 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 24)
        self.coords = [(50, 430), (50, 30), (1930, 30), (1930, 430)]


params = Params()


class Sprites:
    def __init__(self) -> None:
        self.background = Image.open(IMG_PATH + "background.png").resize((2100, 900))
        self._background_ = self.background.copy()
        self.card_back = (
            Image.open(IMG_PATH + "Cardback.png").resize((120, 168)).convert("RGBA")
        )
        self.cards = []
        for name in params.card_order:
            self.cards.append(
                Image.open(IMG_PATH + f"{name}.png").resize((120, 168)).convert("RGBA")
            )


sprites = Sprites()


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

    #  Hold in hand
    temp = state[0:12].astype(int)
    for i in range(12):
        text = str(temp[i])
        draw_outlined_text(
            draw, text, params.font32, (153 + 140 * i, 835), (255, 255, 255), 1
        )

    #  Discard pile
    temp = state[12:25].astype(int)
    for i in range(13):
        text = str(temp[i])
        bbox = draw.textbbox((0, 0), text, params.font32)
        draw_outlined_text(
            draw,
            text,
            params.font32,
            (147 + 140 * i + 120 - bbox[2], 700),
            (255, 255, 255),
            1,
        )

    #  Draw pile
    text = str(int(state[25]))
    bbox = draw.textbbox((0, 0), text, params.font32)
    draw_outlined_text(
        draw, text, params.font32, (1050 - bbox[2] // 2, 260), (255, 255, 255), 1
    )

    #  Have to draw
    text = str(int(state[71]))
    bbox = draw.textbbox((0, 0), text, params.font32)
    draw_outlined_text(
        draw, text, params.font32, (1050 - bbox[2] // 2, 300), (0, 0, 0), 1
    )

    #  Others
    for i in range(4):
        if state[82 + i] == 0:
            text = "Exploded"
            font = params.font24
            b_h = 5
        else:
            text = str(int(state[87 + i]))
            font = params.font32
            b_h = 0

        bbox = draw.textbbox((0, 0), text, font)
        draw_outlined_text(
            draw,
            text,
            font,
            (params.coords[i][0] + 60 - bbox[2] // 2, params.coords[i][1] + 130 + b_h),
            (255, 255, 255),
            1,
        )

    #  Last action
    a = np.where(state[72:82] == 1)[0]
    if len(a) == 1:
        last_action = a[0]
        if last_action < 6:
            bg.paste(sprites.cards[last_action], (520, 130))
        else:
            if last_action == 6:
                text = "Draw card"
            elif last_action == 7:
                text = "Double"
            elif last_action == 8:
                text = "Triple"
            elif last_action == 9:
                text = "5 kinds of cards"
            else:
                text = ""

            bbox = draw.textbbox((0, 0), text, params.font32)
            draw_outlined_text(
                draw, text, params.font32, (580 - bbox[2] // 2, 180), (0, 0, 0), 1
            )

    #  Exploded
    if state[86] == 0:
        text = "Exploded"
        bbox = draw.textbbox((0, 0), text, params.font32)
        draw_outlined_text(
            draw, text, params.font32, (1050 - bbox[2] // 2, 600), (0, 0, 0), 1
        )

    #  See the future
    temp = []
    for i in range(3):
        try:
            temp.append(np.where(state[28 + 13 * i : 41 + 13 * i] == 1)[0][0])
        except:
            pass

    w_ = len(temp) * 140 - 20
    s_ = 1520 - w_ // 2
    for i in range(len(temp)):
        bg.paste(sprites.cards[temp[i]], (s_ + 140 * i, 130))

    #  Nope
    if state[27] == 1:
        text = "Nope"
        bbox = draw.textbbox((0, 0), text, params.font32)
        draw_outlined_text(
            draw, text, params.font32, (580 - bbox[2] // 2, 300), (0, 0, 0), 1
        )

    #  Cards have been discarded
    temp = []
    for i in range(11):
        for k in range(int(state[91 + i])):
            temp.append(i)

    w_ = len(temp) * 140 - 20
    s_ = 1050 - w_ // 2
    for i in range(len(temp)):
        bg.paste(sprites.cards[temp[i]], (s_ + 140 * i, 380))

    return bg


action_annotations = (
    [
        "Nope",
        "Attack",
        "Skip",
        "Favor",
        "Shuffle",
        "See the future",
        "Draw card",
        "Play doublecard",
        "Play triplecard",
        "Play five kinds of card",
        "Not Nope",
    ]
    + [f"Choose player {i+1} to rob" for i in range(4)]
    + [f'Choose "{params.card_order[i]}"-card to give' for i in range(12)]
    + [f'Ask about "{params.card_order[i]}"-card' for i in range(12)]
    + [f'Choose "{params.card_order[i]}"-card from Discard pile' for i in range(12)]
    + [f'Discard "{params.card_order[i]}"-card' for i in range(11)]
)


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""

    return action_annotations[action]


class Env_components:
    def __init__(self, env, draw, discard, winner, list_other, turn) -> None:
        self.env = env
        self.draw = draw
        self.discard = discard
        self.winner = winner
        self.list_other = list_other
        self.turn = turn


def get_env_components():
    env, draw, discard = _env.initEnv()
    winner = _env.checkEnded(env)
    list_other = np.array([-1, 1, 2, 3, 4])
    np.random.shuffle(list_other)
    turn = 0
    return Env_components(env, draw, discard, winner, list_other, turn)


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    if not action is None:
        env_components.env, env_components.draw, env_components.discard = _env.stepEnv(
            env_components.env, env_components.draw, env_components.discard, action
        )
        env_components.turn += 1

    env_components.winner = _env.checkEnded(env_components.env)
    while env_components.winner == -1 and env_components.turn < 500:
        phase = env_components.env[67]
        main_id = env_components.env[57]
        nope_id = env_components.env[73]
        last_action = env_components.env[72]
        if phase == 0:
            pIdx = int(main_id)
        elif phase == 1:
            pIdx = int(nope_id)
        elif phase == 2:
            pIdx = int(main_id)
        elif phase == 3:
            if last_action == 3:
                pIdx = int(env_components.env[74])
            else:
                pIdx = int(main_id)
        elif phase == 4:
            pIdx = int(main_id)

        if env_components.list_other[pIdx] == -1:
            break

        state = _env.getAgentState(
            env_components.env, env_components.draw, env_components.discard
        )
        agent = list_agent[env_components.list_other[pIdx] - 1]
        data = list_data[env_components.list_other[pIdx] - 1]
        action, data = agent(state, data)
        env_components.env, env_components.draw, env_components.discard = _env.stepEnv(
            env_components.env, env_components.draw, env_components.discard, action
        )
        env_components.turn += 1
        env_components.winner = _env.checkEnded(env_components.env)

    if env_components.winner == -1 and env_components.turn < 500:
        state = _env.getAgentState(
            env_components.env, env_components.draw, env_components.discard
        )
        win = -1
    else:
        env = env_components.env.copy()
        env[67] = 0
        my_idx = np.where(env_components.list_other == -1)[0][0]
        env[57] = my_idx
        env[58:62] = _env.nopeTurn(env[57])
        state = _env.getAgentState(env, env_components.draw, env_components.discard)
        if _env.checkEnded(env) == my_idx:
            win = 1
        else:
            win = 0

        for pIdx in range(5):
            env[57] = pIdx
            env[58:62] = _env.nopeTurn(env[57])
            if pIdx != my_idx:
                _state = _env.getAgentState(
                    env, env_components.draw, env_components.discard
                )
                agent = list_agent[env_components.list_other[pIdx] - 1]
                data = list_data[env_components.list_other[pIdx] - 1]
                action, data = agent(_state, data)

    return win, state, env_components
