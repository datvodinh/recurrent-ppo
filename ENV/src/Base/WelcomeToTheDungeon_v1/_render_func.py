from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
from env import SHORT_PATH
from src.Base.WelcomeToTheDungeon_v1 import env as _env

IMG_PATH = SHORT_PATH + "src/Base/WelcomeToTheDungeon_v1/images/"
BG_SIZE = (1680, 720)
CARD_SIZE = (144, 202)
PLAYER_SIZE = (100, 120)
RESULT_SIZE = (25, 25)


class Sprites:
    def __init__(self) -> None:
        self.background = (
            Image.open(IMG_PATH + "bg.png").convert("RGBA").resize(BG_SIZE)
        )
        self.background.putalpha(80)
        _monster = [
            "goblin",
            "skeleton",
            "orc",
            "vampire",
            "golem",
            "lich",
            "demon",
            "dragon",
        ]
        self.cards = []
        for _mon in _monster:
            self.cards.append(
                Image.open(IMG_PATH + f"monster_{_mon}.png")
                .convert("RGBA")
                .resize(CARD_SIZE)
            )

        self.card_back = Image.open(IMG_PATH + "monster_back.png").resize(CARD_SIZE)
        self.faded_card_back = self.card_back.copy()
        br = ImageEnhance.Brightness(self.faded_card_back)
        self.faded_card_back = br.enhance(0.5)
        ct = ImageEnhance.Contrast(self.faded_card_back)
        self.faded_card_back = ct.enhance(0.5)

        self.players = []
        self.players.append(
            Image.open(IMG_PATH + f"player.jpg").convert("RGBA").resize(PLAYER_SIZE)
        )
        self.players.append(
            Image.open(IMG_PATH + f"player.jpg").convert("RGBA").resize(PLAYER_SIZE)
        )

        br = ImageEnhance.Brightness(self.players[1])
        self.players[1] = br.enhance(0.5)
        ct = ImageEnhance.Contrast(self.players[1])
        self.players[1] = ct.enhance(0.5)

        self.heroes = []
        self.heroes.append(
            Image.open(IMG_PATH + f"Warrior.png").convert("RGBA").resize(CARD_SIZE)
        )
        self.heroes.append(
            Image.open(IMG_PATH + f"Rogue.png").convert("RGBA").resize(CARD_SIZE)
        )

        self.Warrior = []
        for equip in [
            "knightshield",
            "platearmor",
            "torch",
            "holygrail",
            "dragonspear",
            "vorpalsword",
        ]:
            self.Warrior.append(
                Image.open(IMG_PATH + f"Warrior_{equip}.png")
                .convert("RGBA")
                .resize(CARD_SIZE)
            )

        self.Rogue = []
        for equip in [
            "buckler",
            "mithrilarmor",
            "ringofpower",
            "invisibilitycloak",
            "healingpotion",
            "vorpaldagger",
        ]:
            self.Rogue.append(
                Image.open(IMG_PATH + f"Rogue_{equip}.png")
                .convert("RGBA")
                .resize(CARD_SIZE)
            )

        self.result = []
        self.result.append(
            Image.open(IMG_PATH + f"green_tick.png").convert("RGBA").resize(RESULT_SIZE)
        )
        self.result.append(
            Image.open(IMG_PATH + f"lose_icon.png").convert("RGBA").resize(RESULT_SIZE)
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

        x_player0 = round(BG_SIZE[0] * 0.5 - PLAYER_SIZE[0] * 0.5)
        y_player0 = round(BG_SIZE[1] * 0.99 - PLAYER_SIZE[1])

        x_player1 = round(BG_SIZE[0] * 0.99 - PLAYER_SIZE[0])
        y_player1 = round(BG_SIZE[1] * 0.5 - PLAYER_SIZE[1] * 0.5)

        x_player2 = round(BG_SIZE[0] * 0.5 - PLAYER_SIZE[0] * 0.5)
        y_player2 = round(BG_SIZE[1] * 0.01)

        x_player3 = round(BG_SIZE[0] * 0.01)
        y_player3 = round(BG_SIZE[1] * 0.5 - PLAYER_SIZE[1] * 0.5)

        self.coords_players = [
            (x_player0, y_player0),
            (x_player1, y_player1),
            (x_player2, y_player2),
            (x_player3, y_player3),
        ]

        self.equip = []
        x_equip0 = round(BG_SIZE[0] * 0.15) + CARD_SIZE[0]
        y_equip0 = round(BG_SIZE[1] * 0.2)
        self.equip.append((x_equip0, y_equip0))
        self.equip.append((x_equip0 + CARD_SIZE[0], y_equip0))
        self.equip.append((x_equip0 + 2 * CARD_SIZE[0], y_equip0))
        self.equip.append((x_equip0, y_equip0 + CARD_SIZE[1]))
        self.equip.append((x_equip0 + CARD_SIZE[0], y_equip0 + CARD_SIZE[1]))
        self.equip.append((x_equip0 + 2 * CARD_SIZE[0], y_equip0 + CARD_SIZE[1]))


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

    # my_cards = np.array([0,1,3,4,6])
    # n = my_cards.shape[0]
    # w = CARD_SIZE[0] + _d_ * (n-1)
    # s = params.list_coords_0[0][0] - 0.5*w
    # draw_cards(background, my_cards, s, params.list_coords_0[0][1], True, False)

    for i in range(4):
        if state[8 + i] == 0:
            background.paste(sprites.players[0], params.coords_players[i])
        else:
            background.paste(sprites.players[1], params.coords_players[i])

    resultScore = state[0:8]
    for i in range(4):
        numWin = int(resultScore[2 * i])
        numLose = int(resultScore[2 * i + 1])
        if i % 2 == 0:
            for j in range(numWin):
                background.paste(
                    sprites.result[0],
                    (
                        params.coords_players[i][0] + PLAYER_SIZE[0],
                        params.coords_players[i][1] + j * RESULT_SIZE[1],
                    ),
                )
            for j in range(numLose):
                background.paste(
                    sprites.result[1],
                    (
                        params.coords_players[i][0] + PLAYER_SIZE[0],
                        params.coords_players[i][1] + (j + 2) * RESULT_SIZE[1],
                    ),
                )
        else:
            for j in range(numWin):
                background.paste(
                    sprites.result[0],
                    (
                        params.coords_players[i][0] + j * RESULT_SIZE[0],
                        params.coords_players[i][1] + PLAYER_SIZE[1],
                    ),
                )
            for j in range(numLose):
                background.paste(
                    sprites.result[1],
                    (
                        params.coords_players[i][0] + (j + 2) * RESULT_SIZE[0],
                        params.coords_players[i][1] + PLAYER_SIZE[1],
                    ),
                )

    if state[39] == 1:
        background.paste(
            sprites.heroes[0], (round(BG_SIZE[0] * 0.15), round(BG_SIZE[1] * 0.2))
        )
        for i in range(6):
            background.paste(sprites.Warrior[i], params.equip[i])

        background.paste(
            sprites.heroes[1], (round(BG_SIZE[0] * 0.55), round(BG_SIZE[1] * 0.2))
        )
        for i in range(6):
            background.paste(
                sprites.Rogue[i],
                (params.equip[i][0] + round(BG_SIZE[0] * 0.4), params.equip[i][1]),
            )

        return background

    if state[38] == 1:
        for i in range(8):
            background.paste(
                sprites.cards[i],
                (params.equip[0][0] + (i - 1) * CARD_SIZE[0], params.equip[0][1]),
            )

        return background

    if state[14] == 1:
        background.paste(
            sprites.heroes[0], (round(BG_SIZE[0] * 0.15), round(BG_SIZE[1] * 0.2))
        )

        for i in range(6):
            if state[15 + i] == 1:
                background.paste(sprites.Warrior[i], params.equip[i])
    elif state[21] == 1:
        background.paste(
            sprites.heroes[1], (round(BG_SIZE[0] * 0.15), round(BG_SIZE[1] * 0.2))
        )

        for i in range(6):
            if state[22 + i] == 1:
                background.paste(sprites.Rogue[i], params.equip[i])

    if state[37] == 0:
        background.paste(
            sprites.card_back, (round(BG_SIZE[0] * 0.6), round(BG_SIZE[1] * 0.35))
        )
    else:
        _mons = np.where(state[28:36] == 1)[0][0]
        background.paste(
            sprites.cards[_mons], (round(BG_SIZE[0] * 0.6), round(BG_SIZE[1] * 0.35))
        )
    cor = (round(BG_SIZE[0] * 0.6), round(BG_SIZE[1] * 0.65))
    text = f"Số lượng thẻ rút: {int(13 - state[12])}"
    color = (0, 0, 0)
    font = ImageFont.truetype(r"src/ImageFonts/arial.ttf", 20)
    ImageDraw.Draw(background).text(cor, text, color, font)

    background.paste(
        sprites.card_back, (round(BG_SIZE[0] * 0.75), round(BG_SIZE[1] * 0.35))
    )
    cor = (round(BG_SIZE[0] * 0.75), round(BG_SIZE[1] * 0.65))
    text = f"Số lượng thẻ trong hang: {int(state[13])}"
    color = (0, 0, 0)
    font = ImageFont.truetype(r"src/ImageFonts/arial.ttf", 20)
    ImageDraw.Draw(background).text(cor, text, color, font)
    return background


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""

    if action == 0:
        return "Bỏ lượt"

    elif action == 1:
        return "Xem monster"
    elif action == 2:
        return "Bỏ monster vào hang"

    if action in range(3, 9):
        Warrior = [
            "Knight Shield",
            "Plate Armor",
            "Torch",
            "Holy Grail",
            "Dragon Spear",
            "Vorpal Sword",
        ]
        card = Warrior[action - 3]
        return f"Bỏ monster cùng thẻ {card}"

    if action in range(9, 15):
        Rogue = [
            "Buckler",
            "Mithril Armor",
            "Ring Of Power",
            "Invisibility Cloak",
            "Healing Potion",
            "Vorpal Dagger",
        ]
        card = Rogue[action - 9]
        return f"Bỏ monster cùng thẻ {card}"

    if action in range(15, 23):
        Monster = [
            "Goblin",
            "Skeleton",
            "Orc",
            "Vampire",
            "Golem",
            "Lich",
            "Demon",
            "Dragon",
        ]

        monster = Monster[action - 15]
        return f"Sử dụng trang bị tiêu diệt {monster}"

    if action == 23:
        return "Chọn Warrior"
    if action == 24:
        return "Chọn Rogue"


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
    if not action is None:
        _env.stepEnv(action, env_components.env)

    env_components.winner = _env.checkEnded(env_components.env)
    if env_components.winner == -1:
        while True:
            p_idx = env_components.env[61] % 4
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
        env[61] = my_idx
        state = _env.getAgentState(env)
        if my_idx == env_components.winner:
            win = 1
        else:
            win = 0

        # Chạy turn cuối cho 3 bot hệ thống
        for p_idx in range(4):
            if p_idx != my_idx:
                env[61] = p_idx
                _state = _env.getAgentState(env)
                agent = list_agent[env_components.list_other[p_idx] - 1]
                data = list_data[env_components.list_other[p_idx] - 1]
                action, data = agent(_state, data)

    return win, state, env_components
