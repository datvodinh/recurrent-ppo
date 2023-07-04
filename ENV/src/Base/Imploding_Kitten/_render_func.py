import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.Base.Imploding_Kitten import env as _env
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
        self.im_card_order = [
            "Reverse",
            "DrawFromBottom",
            "Feral",
            "AlterTheFuture",
            "TargetAttack",
            "Imploding",
        ]
        self.all_card_order = [
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
            "Reverse",
            "DrawFromBottom",
            "Feral",
            "AlterTheFuture",
            "TargetAttack",
            "Defuse",
        ]
        self.font32 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 32)
        self.font24 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 24)


params = Params()


class Sprites:
    def __init__(self) -> None:
        self.background = Image.open(IMG_PATH + "im_background.png").resize((2100, 900))
        self._background_ = self.background.copy()
        self.card_back = (
            Image.open(IMG_PATH + "Cardback.png").resize((120, 168)).convert("RGBA")
        )
        self.cards = []
        self.sm_cards = []
        for name in params.card_order:
            self.cards.append(
                Image.open(IMG_PATH + f"{name}.png").resize((120, 168)).convert("RGBA")
            )
            self.sm_cards.append(
                Image.open(IMG_PATH + f"{name}.png").resize((100, 140)).convert("RGBA")
            )

        self.im_cards = []
        self.sm_im_cards = []
        for name in params.im_card_order:
            self.im_cards.append(
                Image.open(IMG_PATH + f"{name}.png").resize((120, 168)).convert("RGBA")
            )
            self.sm_im_cards.append(
                Image.open(IMG_PATH + f"{name}.png").resize((100, 140)).convert("RGBA")
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
    temp = state[0:17].astype(int)
    for i in range(17):
        text = str(temp[i])
        draw_outlined_text(
            draw, text, params.font32, (68 + 110 * i, 827), (255, 255, 255), 1
        )

    #  Discard pile
    temp = state[17:35].astype(int)
    for i in range(18):
        text = str(temp[i])
        bbox = draw.textbbox((0, 0), text, params.font32)
        draw_outlined_text(
            draw,
            text,
            params.font32,
            (162 + 110 * i - bbox[2], 720),
            (255, 255, 255),
            1,
        )

    #  Draw pile
    text = str(int(state[35]))
    bbox = draw.textbbox((0, 0), text, params.font32)
    draw_outlined_text(
        draw, text, params.font32, (1050 - bbox[2] // 2, 380), (255, 255, 255), 1
    )

    #  Have to draw
    text = str(int(state[100]))
    bbox = draw.textbbox((0, 0), text, params.font32)
    draw_outlined_text(
        draw, text, params.font32, (1050 - bbox[2] // 2, 420), (0, 0, 0), 1
    )

    #  Others
    for i in range(5):
        if state[116 + i] == 0:
            text = "Exploded"
            font = params.font24
            b_h = 5
        else:
            text = str(int(state[122 + i]))
            font = params.font32
            b_h = 0

        bbox = draw.textbbox((0, 0), text, font)
        draw_outlined_text(
            draw,
            text,
            font,
            (110 + 470 * i - bbox[2] // 2, 160 + b_h),
            (255, 255, 255),
            1,
        )

    #  Last action
    a = np.where(state[101:116] == 1)[0]
    if len(a) == 1:
        last_action = a[0]
        if last_action < 6:
            bg.paste(sprites.cards[last_action], (300, 250))
        elif last_action == 11:
            bg.paste(sprites.im_cards[0], (300, 250))
        elif last_action == 12:
            bg.paste(sprites.im_cards[1], (300, 250))
        elif last_action == 13:
            bg.paste(sprites.im_cards[3], (300, 250))
        elif last_action == 14:
            bg.paste(sprites.im_cards[4], (300, 250))
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
                draw, text, params.font32, (360 - bbox[2] // 2, 300), (0, 0, 0), 1
            )

    #  Exploded
    if state[121] == 0:
        text = "Exploded"
        bbox = draw.textbbox((0, 0), text, params.font32)
        draw_outlined_text(
            draw, text, params.font32, (1050 - bbox[2] // 2, 630), (0, 0, 0), 1
        )

    #  See the future
    temp = []
    for i in range(3):
        try:
            temp.append(np.where(state[38 + 19 * i : 57 + 19 * i] == 1)[0][0])
        except:
            pass

    w_ = len(temp) * 140 - 20
    s_ = 1520 - w_ // 2
    for i in range(len(temp)):
        idx = temp[i]
        if idx < 11:
            card = sprites.cards[idx]
        elif idx < 16:
            card = sprites.im_cards[idx - 11]
        elif idx == 16:
            card = sprites.cards[-2]
        elif idx == 17:
            card = sprites.cards[-1]
        else:
            card = sprites.im_cards[-1]

        bg.paste(card, (s_ + 140 * i, 250))

    #  Nope
    if state[37] == 1:
        text = "Nope"
        bbox = draw.textbbox((0, 0), text, params.font32)
        draw_outlined_text(
            draw, text, params.font32, (360 - bbox[2] // 2, 420), (0, 0, 0), 1
        )

    #  Cards have been discarded
    temp = []
    for i in range(16):
        for k in range(int(state[128 + i])):
            temp.append(i)

    w_ = len(temp) * 110 - 10
    s_ = 1050 - w_ // 2
    for i in range(len(temp)):
        idx = temp[i]
        if idx < 11:
            card = sprites.sm_cards[idx]
        elif idx < 16:
            card = sprites.sm_im_cards[idx - 11]

        bg.paste(card, (s_ + 110 * i, 470))

    #  Imploded
    if state[127] == 1:
        bg.paste(sprites.im_cards[-1], (680, 250))

    return bg


temp = [
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
    "Reverse",
    "DrawFromBottom",
    "Feral",
    "AlterTheFuture",
    "TargetAttack",
    "Defuse",
]
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
        "Reverse",
        "Draw from bottom",
        "Alter the future",
        "Targeted Attack",
    ]
    + [f"Choose player {i+1} to rob" for i in range(5)]
    + [f'Choose "{temp[i]}"-card to give' for i in range(17)]
    + [f'Ask about "{temp[i]}"-card' for i in range(17)]
    + [f'Choose "{temp[i]}"-card from Discard pile' for i in range(17)]
    + [f"Choose the {i+1}-th combination to Alter" for i in range(6)]
    + [f'Discard "{temp[i]}"-card' for i in range(16)]
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
    list_other = np.array([-1, 1, 2, 3, 4, 5])
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
    while env_components.winner == -1 and env_components.turn < 300:
        phase = env_components.env[89]
        main_id = env_components.env[77]
        nope_id = env_components.env[95]
        last_action = env_components.env[94]
        if phase == 0:
            pIdx = int(main_id)
        elif phase == 1:
            pIdx = int(nope_id)
        elif phase == 2:
            pIdx = int(main_id)
        elif phase == 3:
            if last_action == 3:
                pIdx = int(env_components.env[96])
            else:
                pIdx = int(main_id)
        elif phase == 4:
            pIdx = int(main_id)
        elif phase == 5:
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

    if env_components.winner == -1 and env_components.turn < 300:
        state = _env.getAgentState(
            env_components.env, env_components.draw, env_components.discard
        )
        win = -1
    else:
        env = env_components.env.copy()
        env[89] = 0
        my_idx = np.where(env_components.list_other == -1)[0][0]
        env[77] = my_idx
        env[78:83] = _env.nopeTurn(my_idx)
        state = _env.getAgentState(env, env_components.draw, env_components.discard)
        if _env.checkEnded(env) == my_idx:
            win = 1
        else:
            win = 0

        for pIdx in range(5):
            env[77] = pIdx
            env[78:83] = _env.nopeTurn(pIdx)
            if pIdx != my_idx:
                _state = _env.getAgentState(
                    env, env_components.draw, env_components.discard
                )
                agent = list_agent[env_components.list_other[pIdx] - 1]
                data = list_data[env_components.list_other[pIdx] - 1]
                action, data = agent(_state, data)

    return win, state, env_components
