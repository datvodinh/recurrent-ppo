from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
from env import SHORT_PATH
from src.Base.RockPaperScissors import env as _env

IMG_PATH = SHORT_PATH + "src/Base/RockPaperScissors/images/"
BG_SIZE = (1680, 720)
SIZE = (300, 300)
SIZE_ = (150, 150)

values = ["0", "1", "2"]


class Sprites:
    def __init__(self) -> None:
        self.background = (
            Image.open(IMG_PATH + "backGround.png").resize(BG_SIZE).convert("RGBA")
        )

        self.cards = []
        for value in values:
            self.cards.append(
                Image.open(IMG_PATH + f"{value}.png").resize(SIZE).convert("RGBA")
            )

        self.cards_ = []
        for value in values:
            self.cards_.append(
                Image.open(IMG_PATH + f"{value}.png").resize(SIZE_).convert("RGBA")
            )

        self.cardsBr = []
        for i in self.cards_:
            br_ = ImageEnhance.Brightness(i)
            self.cardsBr.append(br_.enhance(0.5))


sprites = Sprites()


def get_state_image(
    state=None,
):  ###-------------------------------------------------------
    state = state.astype(int)
    font = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 35)
    font50 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 50)

    thor = Image.open(IMG_PATH + "thor.png").resize(SIZE_).convert("RGBA")
    widow = Image.open(IMG_PATH + "blackWidow.png").resize(SIZE_).convert("RGBA")

    bg = sprites.background.copy()
    bg.paste(thor, (800, 70))
    bg.paste(widow, (1300, 70))
    drawler = ImageDraw.ImageDraw(bg)
    drawler.text((820, 20), "YOU", (255, 0, 0), font)
    if state is None:
        return bg

    if state[6] == 0:
        for i in range(3):
            drawler.text((220, 150 + 200 * i), str(i) + ":", (0, 0, 0), font50)
            bg.paste(sprites.cards_[i], (300, 100 + 200 * i))
    else:
        v0 = np.where(state[:3])[0][0]
        v1 = np.where(state[3:6])[0][0]
        for i in range(3):
            if i == v0:
                drawler.text((220, 150 + 200 * i), str(i) + ":", (0, 0, 0), font50)
                bg.paste(sprites.cards_[i], (300, 100 + 200 * i))
            else:
                bg.paste(sprites.cardsBr[i], (300, 100 + 200 * i))
        bg.paste(sprites.cards[v0], (700, 220))
        bg.paste(sprites.cards[v1], (1200, 220))

        check = v0 - v1
        if check == 1 or check == -2:
            drawler.text((950, 550), "YOU WIN", (255, 0, 0), font50)
        elif check == -1 or check == 2:
            drawler.text((950, 550), "YOU LOSE", (0, 0, 0), font50)
        else:
            drawler.text((950, 550), "continue", (0, 0, 0), font50)
    return bg


def get_description(action):  # -----------------------------------------
    if action < 0 or action >= _env.getActionSize():
        return ""

    if action == 0:
        return "Scissors"
    elif action == 1:
        return "Rock"
    elif action == 2:
        return "Paper"
    else:
        return "Comfirm"


class Env_components:
    def __init__(self, env, winner, list_other) -> None:
        self.env = env
        self.winner = winner
        self.list_other = list_other


def get_env_components():  # -------------------------------------------------
    env = _env.initEnv()
    winner = _env.checkEnded(env)
    list_other = np.array([-1, 1])
    np.random.shuffle(list_other)

    env_components = Env_components(env, winner, list_other)
    return env_components


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):  # ---------------------------------
    if not action is None:
        _env.stepEnv(action, env_components.env)

    env_components.winner = _env.checkEnded(env_components.env)
    if env_components.winner == -1:
        env = env_components.env.copy()
        while env[2] < 100:
            p_idx = int(env_components.env[3])
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
        env[3] = my_idx
        state = _env.getAgentState(env)
        if my_idx == env_components.winner:
            win = 1
        else:
            win = 0

        # Chạy turn cuối cho 1 bot hệ thống
        for p_idx in range(2):
            if p_idx != my_idx:
                env[3] = p_idx
                _state = _env.getAgentState(env)
                agent = list_agent[env_components.list_other[p_idx] - 1]
                data = list_data[env_components.list_other[p_idx] - 1]
                action, data = agent(_state, data)

    return win, state, env_components
