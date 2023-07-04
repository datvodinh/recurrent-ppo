from PIL import Image, ImageDraw, ImageFont
import numpy as np
from src.Base.WelcomeToTheDungeon_v2 import env as _env

IMG_PATH = "src/Base/WelcomeToTheDungeon_v2/images/"
w_bg, h_bg = 1680, 720
BG_SIZE = (w_bg, h_bg)
c_w, c_h = 80, 112
CARD_SIZE = (c_w, c_h)


class Sprites:
    figures = ["barbarian", "mage"]
    barbarianEquips = [
        "leather_shield",
        "healing_potion",
        "torch",
        "war_hammer",
        "vorpal_axe",
        "chainmail",
    ]
    mageEquips = [
        "omnipotence",
        "holy_grail",
        "demonic_pact",
        "polymorph",
        "wall_of_fire",
        "bracelet_of_protection",
    ]
    monsters = [
        "golbin_1",
        "golbin_2",
        "skeleton_1",
        "skeleton_2",
        "orc_1",
        "orc_2",
        "vampire_1",
        "vampire_2",
        "golem_1",
        "golem_2",
        "lich",
        "demon",
        "dragon",
    ]

    def __init__(self) -> None:
        self.background = (
            Image.open(IMG_PATH + "bg.png").convert("RGBA").resize(BG_SIZE)
        )
        self.background.putalpha(80)

        self.aidcard = (
            Image.open(IMG_PATH + "aid_card.png").convert("RGBA").resize(CARD_SIZE)
        )

        self.figureImgs = []
        for fig in self.figures:
            imgPath = IMG_PATH + fig + ".png"
            self.figureImgs.append(
                Image.open(imgPath).convert("RGBA").resize(CARD_SIZE)
            )

        self.barbarianEquipImgs = []
        for equip in self.barbarianEquips:
            imgPath = IMG_PATH + "barbarian_" + equip + ".png"
            self.barbarianEquipImgs.append(
                Image.open(imgPath).convert("RGBA").resize(CARD_SIZE)
            )

        self.mageEquipImgs = []
        for equip in self.mageEquips:
            imgPath = IMG_PATH + "mage_" + equip + ".png"
            self.mageEquipImgs.append(
                Image.open(imgPath).convert("RGBA").resize(CARD_SIZE)
            )

        self.monsterImgs = []
        for monster in self.monsters:
            imgPath = IMG_PATH + "monster_" + monster + ".png"
            self.monsterImgs.append(
                Image.open(imgPath).convert("RGBA").resize(CARD_SIZE)
            )


sprites = Sprites()


def getFont(size):
    return ImageFont.truetype(r"src/ImageFonts/arial.ttf", size)


def addText(background, text, cor, color=None, fontSize=None):
    if fontSize is None:
        fontSize = 50
    font = getFont(fontSize)
    if color is None:
        color = (0, 0, 0)
    ImageDraw.Draw(background).text(cor, text, color, font)
    return background


def addEquips(background, state):
    state = state.astype("int16")
    figures = [sprites.figureImgs[0], sprites.figureImgs[1]]
    equips = [sprites.barbarianEquipImgs, sprites.mageEquipImgs]
    if state[53] == 0 and state[61] == 0:
        f_w, f_h = int(c_w * 1.7), int(c_h * 1.7)
        f_x, f_y = int(w_bg * 0.1), int(h_bg * 0.3)
        background.paste(figures[0].resize((f_w, f_h)), (f_x, f_y))
        i = 1
        for equip in equips[0]:
            background.paste(
                equip.resize((int(c_w * 1.5), int(c_h * 1.5))),
                (int(w_bg * 0.1) + i * 150, int(h_bg * 0.3)),
            )
            i += 1

        background.paste(figures[1].resize((f_w, f_h)), (f_x, f_y + f_h + 10))
        i = 1
        for equip in equips[1]:
            background.paste(
                equip.resize((int(c_w * 1.5), int(c_h * 1.5))),
                (int(w_bg * 0.1) + i * 150, int(h_bg * 0.6)),
            )
            i += 1

        background = addText(
            background, "Please choose a hero to start the round", (int(w_bg * 0.1), 10)
        )
    else:
        f_w, f_h = int(c_w * 1.7), int(c_h * 1.7)
        f_x, f_y = int(w_bg * 0.1), int(h_bg * 0.3)

        e_w, e_h = int(c_w * 1.2), int(c_h * 1.2)
        e_x, e_y = int(w_bg * 0.1), int(h_bg * 0.4)

        background.paste(figures[state[61]].resize((f_w, f_h)), (f_x, f_y))
        j = 1
        for i in range(len(equips[state[61]])):
            if state[i + 55 * state[53] + 63 * state[61]]:
                img = equips[state[61]][i]
            else:
                img = figures[state[61]]
            background.paste(img.resize((e_w, e_h)), (e_x + j * 150, e_y))
            j += 1

    return background


def addSeenMonster(background, state):
    monsters_ = state[14:53].reshape(13, -1)
    idx_monster = np.where(monsters_[:, 0] > 0)[0]
    if len(idx_monster) > 0:
        j = 0
        for idx in idx_monster:
            background.paste(sprites.monsterImgs[idx], (0 + j * 100, 0))
            if monsters_[idx][2] == 1:
                background = addText(
                    background, "Bị bỏ nha", (0 + j * 100, 120), fontSize=20
                )
            elif monsters_[idx][1] > 0:
                background = addText(
                    background,
                    f"{int(monsters_[idx][1])}",
                    (0 + j * 100, 120),
                    fontSize=20,
                )
            j += 1
    return background


def addDrewMonster(background, state):
    idx_monster = np.where(state[70:83] == 1)[0]
    if len(idx_monster) > 0:
        j = 0
        for idx in idx_monster:
            background.paste(sprites.monsterImgs[idx], (0 + j * 100, 200))
            j += 1
    return background


def addAidCard(background, state):
    if state[53] == 1 or state[61] == 1:
        background.paste(
            sprites.aidcard.resize((int(c_w * 2.0), int(c_h * 1.5))), (500, 600)
        )
    return background


def addScore(background, state):
    state = state.astype("int8")
    ignores = state[0:4]
    score_0 = state[4:6]
    score_1 = state[6:8]
    score_2 = state[8:10]
    score_3 = state[10:12]
    background = addText(
        background,
        f"win/lose/ignore: {score_0[0]} / {score_0[1]} / {ignores[0]}",
        (int(w_bg * 0.4), int(h_bg * 0.85)),
        fontSize=30,
    )

    background = addText(
        background,
        f"win/lose/ignore others",
        (int(w_bg * 0.8), int(h_bg * 0.75)),
        fontSize=30,
    )

    background = addText(
        background,
        f"{score_1[0]} / {score_1[1]} / {ignores[1]}",
        (int(w_bg * 0.8), int(h_bg * 0.75) + 35),
        fontSize=30,
    )

    background = addText(
        background,
        f"{score_2[0]} / {score_2[1]} / {ignores[2]}",
        (int(w_bg * 0.8), int(h_bg * 0.75) + 70),
        fontSize=30,
    )

    background = addText(
        background,
        f"{score_3[0]} / {score_3[1]} / {ignores[3]}",
        (int(w_bg * 0.8), int(h_bg * 0.75) + 105),
        fontSize=30,
    )
    return background


def get_state_image(state=None):
    background = sprites.background.copy()
    if state is None:
        return background
    if state[69] == 1:
        background = addText(
            background, "IN DUNGEON", (int(w_bg * 0.7), int(h_bg * 0.1)), fontSize=30
        )
    background = addScore(background, state)
    background = addEquips(background, state)
    background = addAidCard(background, state)
    background = addSeenMonster(background, state)
    background = addDrewMonster(background, state)
    background = addText(
        background,
        f"Số thẻ quái vật chưa mở: {int(state[12])}",
        (int(w_bg * 0.7), int(h_bg * 0.2)),
        fontSize=30,
    )
    background = addText(
        background,
        f"Số thẻ quái vật trong hang: {int(state[13])}",
        (int(w_bg * 0.7), int(h_bg * 0.3)),
        fontSize=30,
    )
    return background


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""
    if action == 0:
        return "Bỏ lượt"
    if action == 11:
        return "Xem bài"
    if action == 1:
        return "Rút thẻ quái vật và bỏ vào hang"
    if action == 8:
        return "Sử dụng Vorpal Axe anh hùng Barbarian"
    if action == 9:
        return "Sử dụng Polymorph của anh hùng Mage"
    if action == 2:
        return "Rút thẻ quái vật và bỏ trang bị thứ nhất"
    if action == 3:
        return "Rút thẻ quái vật và bỏ trang bị thứ hai"
    if action == 4:
        return "Rút thẻ quái vật và bỏ trang bị thứ ba"
    if action == 5:
        return "Rút thẻ quái vật và bỏ trang bị thứ bốn"
    if action == 6:
        return "Rút thẻ quái vật và bỏ trang bị thứ năm"
    if action == 7:
        return "Rút thẻ quái vật và bỏ trang bị thứ sáu"
    if action == 10:
        return "Đánh thẻ quái vật trên cùng"
    if action == 12:
        return "Chọn Barbarian"
    if action == 13:
        return "Chọn Mage"


class Env_components:
    def __init__(self, env, winner, list_other) -> None:
        self.env = env
        self.winner = winner
        self.list_other = list_other


def get_env_components():
    env = _env.initEnv()
    env = _env.resetRound(env)
    winner = _env.checkEnded(env)
    list_other = np.array([-1, 1, 2, 3])
    np.random.shuffle(list_other)

    env_components = Env_components(env, winner, list_other)
    return env_components


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    if action is not None:
        _env.stepEnv(action, env_components.env)

    env_components.winner = _env.checkEnded(env_components.env)
    if env_components.winner == -1:
        while True:
            p_idx = env_components.env[99] % 4
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
        env[99] = my_idx
        state = _env.getAgentState(env)
        if my_idx == env_components.winner:
            win = 1
        else:
            win = 0

        # Chạy turn cuối cho 3 bot hệ thống
        for p_idx in range(4):
            if p_idx != my_idx:
                env[99] = p_idx
                _state = _env.getAgentState(env)
                agent = list_agent[env_components.list_other[p_idx] - 1]
                data = list_data[env_components.list_other[p_idx] - 1]
                action, data = agent(_state, data)

    return win, state, env_components
