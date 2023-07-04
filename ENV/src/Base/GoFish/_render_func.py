import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from src.Base.GoFish import env as _env
from env import SHORT_PATH

IMG_PATH = SHORT_PATH + "src/Base/GoFish/images/"
BG_SIZE = (1680, 720)
CARD_SIZE = (100, 140)

CARD_SIZE1 = (60, 84)

CARD_SIZE2 = (70, 98)

CARD_SIZE3 = (50, 70)
card_values = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


class Sprites:
    def __init__(self) -> None:
        self.background = (
            Image.open(IMG_PATH + "backGroundBlack.jpg").resize(BG_SIZE).convert("RGBA")
        )
        self.cards = []
        for value in card_values:
            self.cards.append(
                Image.open(IMG_PATH + f"{value}-Heart.png").convert("RGBA")
            )

        self.cards_ = []
        for i in self.cards:
            br_ = ImageEnhance.Brightness(i)
            self.cards_.append(br_.enhance(0.5))

        self.card_back = Image.open(IMG_PATH + "Card_back.png").convert("RGBA")
        br = ImageEnhance.Brightness(self.card_back)
        self.card_back_ = br.enhance(0.5)


sprites = Sprites()


def get_state_image(
    state=None,
):  ###-------------------------------------------------------
    state = state.astype(int)
    fontMin = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 20)
    font = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 35)
    font50 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 45)
    bg = sprites.background.copy()
    drawler = ImageDraw.ImageDraw(bg)
    if state is None:
        return bg

    pl = 0
    if np.sum(state[64:67]) == 1:
        pl = np.where(state[64:67])[0][0] + 1

    # Phần bài của tôi
    # thor = Image.open(IMG_PATH+"thor.png").resize((75, 64))
    # bg.paste(thor, (500, 600))

    myCards = state[:13]
    my_cards = np.where((myCards < 4) & (myCards > 0))[0]
    point = state[14]

    if point:
        _ = 840
        for i in np.where(myCards == 4)[0]:
            card = sprites.cards[i].resize(CARD_SIZE1)
            _ += 15
            bg.paste(card, (_, 620))

    drawler.text((600, 640), "Score: " + str(point), (255, 0, 0), font50)

    n = my_cards.size
    for i in range(n):
        id = my_cards[i]
        laBai = sprites.cards[id].resize(CARD_SIZE)
        x = 820 - (50 + (n - 1) * 25) + i * 50
        bg.paste(laBai, (x, 470))
        num = state[id]
        drawler.text((x + 10, 560), str(num), (0, 0, 255), font)

    # Lá còn lại để bốc
    card_back = sprites.card_back.resize((80, 112))
    bg.paste(card_back, (1180, 40))
    card_back = sprites.card_back.resize((80, 112))
    bg.paste(card_back, (1175, 45))
    drawler.text((1270, 60), str(state[60]), (255, 255, 255), font50)
    if np.where(state[61:64])[0].size:
        phase = np.where(state[61:64])[0][0]
        drawler.text((100, 60), "Phase: " + str(phase), (255, 255, 255), font50)

    # Phần bài người chơi khác
    ###player 2
    if pl != 0 and pl != 2:
        card_back = sprites.card_back_.resize(CARD_SIZE)
    else:
        card_back = sprites.card_back.resize(CARD_SIZE)
    bg.paste(card_back, (710, 40))
    # drawler.text((450, 100), "Player 2:", (255, 255, 255), font50)
    drawler.text((725, 125), str(state[43]), (255, 255, 31), font50)
    drawler.text((500, 110), "Score: " + str(state[44]), (255, 255, 255), font50)
    point2 = state[44]
    if point2:
        cards = state[30:43]
        _ = 810
        for i in np.where(cards)[0]:
            if pl != 0 and pl != 2:
                card = sprites.cards_[i].resize(CARD_SIZE3)
            else:
                card = sprites.cards[i].resize(CARD_SIZE3)
            _ += 10
            bg.paste(card, (_, 117))

    cardYc_ = state[80:93]
    cardYc = np.where(cardYc_)[0]
    __ = 800
    for i in range(cardYc.size):
        id = cardYc[i]
        if pl != 0 and pl != 2:
            laBai = sprites.cards_[id].resize(CARD_SIZE2)
        else:
            laBai = sprites.cards[id].resize(CARD_SIZE2)
        __ += 20
        bg.paste(laBai, (__, 10))

    ###player 1---------------
    if pl != 0 and pl != 1:
        card_back = sprites.card_back_.resize(CARD_SIZE)
    else:
        card_back = sprites.card_back.resize(CARD_SIZE)
    bg.paste(card_back, (220, 250))
    # drawler.text((450, 100), "Player 2:", (255, 255, 255), font50)
    drawler.text((235, 335), str(state[28]), (255, 255, 31), font50)
    drawler.text((15, 320), "Score: " + str(state[29]), (255, 255, 255), font50)
    point2 = state[29]
    if point2:
        cards = state[15:28]
        _ = 320
        for i in np.where(cards)[0]:
            if pl != 0 and pl != 2:
                card = sprites.cards_[i].resize(CARD_SIZE3)
            else:
                card = sprites.cards[i].resize(CARD_SIZE3)
            _ += 10
            bg.paste(card, (_, 327))

    cardYc_ = state[67:80]
    cardYc = np.where(cardYc_)[0]
    __ = 310
    for i in range(cardYc.size):
        id = cardYc[i]
        if pl != 0 and pl != 1:
            laBai = sprites.cards_[id].resize(CARD_SIZE2)
        else:
            laBai = sprites.cards[id].resize(CARD_SIZE2)
        __ += 20
        bg.paste(laBai, (__, 220))

    ###player 3---------------
    if pl != 0 and pl != 3:
        card_back = sprites.card_back_.resize(CARD_SIZE)
    else:
        card_back = sprites.card_back.resize(CARD_SIZE)
    bg.paste(card_back, (1680 - 330, 250))
    # drawler.text((450, 100), "Player 2:", (255, 255, 255), font50)
    drawler.text((1680 - 313, 335), str(state[58]), (255, 255, 31), font50)
    drawler.text((1680 - 215, 320), "Score: " + str(state[59]), (255, 255, 255), font50)
    point2 = state[59]
    if point2:
        cards = state[45:58]
        _ = 1680 - 380
        for i in np.where(cards)[0]:
            if pl != 0 and pl != 2:
                card = sprites.cards_[i].resize(CARD_SIZE3)
            else:
                card = sprites.cards[i].resize(CARD_SIZE3)
            _ -= 10
            bg.paste(card, (_, 327))

    cardYc_ = state[93:106]
    cardYc = np.where(cardYc_)[0]
    __ = 1680 - 390
    for i in range(cardYc.size):
        id = cardYc[i]
        if pl != 0 and pl != 3:
            laBai = sprites.cards_[id].resize(CARD_SIZE2)
        else:
            laBai = sprites.cards[id].resize(CARD_SIZE2)
        __ -= 20
        bg.paste(laBai, (__, 220))

    return bg


def get_description(action):  # -----------------------------------------
    if action < 0 or action >= _env.getActionSize():
        return ""

    if action == 0:
        return "Bốc"
    elif action < 4:
        return "Yêu cầu player: " + str(action)
    else:
        return "Yêu cầu lá bài: " + str(card_values[action - 4])


class Env_components:
    def __init__(self, env, winner, list_other) -> None:
        self.env = env
        self.winner = winner
        self.list_other = list_other


def get_env_components():  # -------------------------------------------------
    env = _env.initEnv()
    winner = _env.checkEnded(env)
    list_other = np.array([-1, 1, 2, 3])
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
        while env[113] < 400:
            p_idx = int(env_components.env[113]) % 4
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
        env[113] = my_idx
        state = _env.getAgentState(env)
        if my_idx == env_components.winner:
            win = 1
        else:
            win = 0

        # Chạy turn cuối cho 3 bot hệ thống
        for p_idx in range(4):
            if p_idx != my_idx:
                env[113] = p_idx
                _state = _env.getAgentState(env)
                agent = list_agent[env_components.list_other[p_idx] - 1]
                data = list_data[env_components.list_other[p_idx] - 1]
                action, data = agent(_state, data)

    return win, state, env_components
