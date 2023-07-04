import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from src.Base.Century import env as _env
from env import SHORT_PATH

IMG_PATH = SHORT_PATH + "src/Base/Century/images/"
BG_SIZE_BASE = np.array((2400, 1600))
CARD_SIZE_BASE = np.array((150, 200))
RATIO = 1
TOKEN_SIZE = np.array([500, 50]) * RATIO
TOKEN_SIZE_BOARD = np.array([150, 25]) * RATIO


COIN_SIZE = np.array([50, 50]) * RATIO
BG_SIZE = BG_SIZE_BASE * RATIO
CARD_SIZE = CARD_SIZE_BASE * RATIO


class Sprites:
    def __init__(self) -> None:
        self.background = Image.open(IMG_PATH + "back_ground.jpg").resize(BG_SIZE)
        self.myFont = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", size=50 * RATIO
        )
        self.myFont_end = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", size=25 * RATIO
        )

        self.font2 = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", size=30 * RATIO
        )
        self.normal_card = []
        normal_card = np.arange(45)
        for i in range(len(normal_card)):
            self.normal_card.append(
                Image.open(f"./src/Base/Century/images/{i}.png").resize(CARD_SIZE)
            )

        self.victory_card = []
        victory_card = np.arange(36)
        for i in range(len(victory_card)):
            self.victory_card.append(
                Image.open(f"./src/Base/Century/images/victory_{i}.png").resize(
                    CARD_SIZE
                )
            )

        self.ALL_CARD_POINT_IN4 = np.array(
            [
                [0, 0, 0, 5, 200],
                [0, 0, 2, 3, 180],
                [0, 0, 3, 2, 170],
                [0, 0, 0, 4, 160],
                [0, 2, 0, 3, 160],
                [0, 0, 5, 0, 150],
                [0, 0, 2, 2, 140],
                [0, 3, 0, 2, 140],
                [2, 0, 0, 3, 140],
                [0, 2, 3, 0, 130],
                [0, 0, 4, 0, 120],
                [0, 2, 0, 2, 120],
                [0, 3, 2, 0, 120],
                [2, 2, 0, 0, 60],
                [3, 2, 0, 0, 70],
                [2, 3, 0, 0, 80],
                [2, 0, 2, 0, 80],
                [0, 4, 0, 0, 80],
                [3, 0, 2, 0, 90],
                [2, 0, 0, 2, 100],
                [0, 5, 0, 0, 100],
                [0, 2, 2, 0, 100],
                [2, 0, 3, 0, 110],
                [3, 0, 0, 2, 110],
                [1, 1, 1, 3, 200],
                [0, 2, 2, 2, 190],
                [1, 1, 3, 1, 180],
                [2, 0, 2, 2, 170],
                [1, 3, 1, 1, 160],
                [2, 2, 0, 2, 150],
                [3, 1, 1, 1, 140],
                [2, 2, 2, 0, 130],
                [0, 2, 1, 1, 120],
                [1, 0, 2, 1, 120],
                [1, 1, 1, 1, 120],
                [2, 1, 0, 1, 90],
            ]
        )

        self.ALL_CARD_IN4 = np.array(
            [
                [0, 0, 0, 0, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 2, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [2, 0, 0, 0, 0, 2, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 1, 0, 0],
                [3, 0, 0, 0, 0, 0, 0, 1, 0],
                [3, 0, 0, 0, 0, 3, 0, 0, 0],
                [3, 0, 0, 0, 0, 1, 1, 0, 0],
                [4, 0, 0, 0, 0, 0, 2, 0, 0],
                [4, 0, 0, 0, 0, 0, 1, 1, 0],
                [5, 0, 0, 0, 0, 0, 0, 2, 0],
                [5, 0, 0, 0, 0, 0, 3, 0, 0],
                [0, 1, 0, 0, 3, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 2, 0, 0],
                [0, 2, 0, 0, 3, 0, 1, 0, 0],
                [0, 2, 0, 0, 2, 0, 0, 1, 0],
                [0, 3, 0, 0, 0, 0, 3, 0, 0],
                [0, 3, 0, 0, 0, 0, 0, 2, 0],
                [0, 3, 0, 0, 1, 0, 1, 1, 0],
                [0, 3, 0, 0, 2, 0, 2, 0, 0],
                [0, 0, 1, 0, 4, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 2, 0, 0, 0],
                [0, 0, 1, 0, 0, 2, 0, 0, 0],
                [0, 0, 2, 0, 2, 1, 0, 1, 0],
                [0, 0, 2, 0, 0, 0, 0, 2, 0],
                [0, 0, 2, 0, 2, 3, 0, 0, 0],
                [0, 0, 2, 0, 0, 2, 0, 1, 0],
                [0, 0, 3, 0, 0, 0, 0, 3, 0],
                [0, 0, 0, 1, 0, 0, 2, 0, 0],
                [0, 0, 0, 1, 3, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 3, 0, 0, 0],
                [0, 0, 0, 1, 2, 2, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 2, 1, 1, 3, 0, 0],
                [0, 0, 0, 2, 0, 3, 2, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 1, 0],
                [2, 0, 1, 0, 0, 0, 0, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        self.coin_gold = Image.open(f"./src/Base/Century/images/coin-gold.png").resize(
            COIN_SIZE
        )
        self.coin_silver = Image.open(
            f"./src/Base/Century/images/coin-silver.png"
        ).resize(COIN_SIZE)
        self.token = Image.open(f"./src/Base/Century/images/token.png").resize(
            TOKEN_SIZE
        )
        self.token_board = Image.open(f"./src/Base/Century/images/token.png").resize(
            TOKEN_SIZE_BOARD
        )

        self.background.paste(self.token, (1600 * RATIO, 450 * RATIO))
        self.background.paste(self.token, (1600 * RATIO, 675 * RATIO))
        self.background.paste(self.token, (1600 * RATIO, 900 * RATIO))
        self.background.paste(self.token, (1600 * RATIO, 1125 * RATIO))
        self.background.paste(self.token, (1600 * RATIO, 1350 * RATIO))
        self.player_text_coor = [
            (1400 * RATIO, 500 * RATIO),
            (1400 * RATIO, 725 * RATIO),
            (1400 * RATIO, 950 * RATIO),
            (1400 * RATIO, 1175 * RATIO),
            (1400 * RATIO, 1400 * RATIO),
        ]


sprites = Sprites()


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""
    else:
        action_mean = _env.ACTIONS_MEAN[action]

    return f"player {action_mean}"


class Env_components:
    def __init__(self, env, winner, list_other) -> None:
        self.env = env
        self.winner = winner
        self.list_other = list_other


def get_env_components():
    env = _env.initEnv()
    winner = -1
    list_other = np.array([-1, 1, 2, 3, 4])
    np.random.shuffle(list_other)
    env_components = Env_components(env, winner, list_other)
    return env_components


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    if not action is None:
        env_components.env = _env.stepEnv(env_components.env, action)

    my_idx = np.where(env_components.list_other == -1)[0][0]
    env_components.winner = -1

    while _env.system_check_end(env_components.env):
        p_idx = int(env_components.env[_env.ENV_ID_ACTION])
        state = _env.getAgentState(env_components.env)

        if env_components.list_other[p_idx] == -1:
            win = -1
            return win, state, env_components

        agent = list_agent[env_components.list_other[p_idx] - 1]
        data = list_data[env_components.list_other[p_idx] - 1]
        # print('QUA')
        action, data = agent(state, data)

        env_components.env = _env.stepEnv(env_components.env, action)

    env_components.env[_env.ENV_CHECK_END] = 1
    env_components.winner = _env.check_winner(env_components.env)
    for p_idx in range(5):
        if p_idx != my_idx:
            env_components.env[_env.ENV_ID_ACTION] = p_idx
            state = _env.getAgentState(env_components.env)

            agent = list_agent[env_components.list_other[p_idx] - 1]
            data = list_data[env_components.list_other[p_idx] - 1]
            action, data = agent(state, data)

    env_components.env[_env.ENV_ID_ACTION] = my_idx
    state = _env.getAgentState(env_components.env)
    # print(env_components.winner)
    if my_idx == env_components.winner:
        win = 1
    else:
        win = 0

    return win, state, env_components


def get_state_image(state=None):
    background = sprites.background.copy()
    background = draw_state(state, background)
    return background


def draw_state(state, back_ground):
    phase = np.where(state[267:272])[0][0] + 1
    check_end = int(state[266])
    draw = ImageDraw.Draw(back_ground)

    # vẽ thẻ thường chưa dùng
    normal_card = np.where(state[6:51])[0]
    if len(normal_card) > 0:
        for id in range(len(normal_card)):
            back_ground.paste(
                sprites.normal_card[normal_card[id]], (40 * id * RATIO, 5 * RATIO)
            )

    # vẽ thẻ thường đã dùng
    normal_card_down = np.where(state[51:96])[0]
    if len(normal_card_down) > 0:
        for id in range(len(normal_card_down)):
            back_ground.paste(
                sprites.normal_card[normal_card_down[id]],
                (40 * id * RATIO, 225 * RATIO),
            )
    # vẽ thẻ thường trên bàn
    card_on_board = state[120:174]
    token_on_board = state[174:194].astype(np.int64)
    for id in range(6):
        card_id_in4 = card_on_board[9 * id : 9 * (id + 1)]
        if np.sum(card_id_in4) == 0:
            break
        id_img = -1
        for idd in range(len(sprites.ALL_CARD_IN4)):
            if np.sum(card_id_in4 == sprites.ALL_CARD_IN4[idd]) == 9:
                id_img = idd
                break
        back_ground.paste(
            sprites.normal_card[id_img], (175 * id * RATIO + 50 * RATIO, 825 * RATIO)
        )
        if id < 5:
            token_free = token_on_board[4 * id : 4 * (id + 1)]
            if np.sum(token_free) > 0:
                draw.text(
                    (175 * id * RATIO + 50 * RATIO, 750 * RATIO),
                    f"{token_free[0]}  {token_free[1]}  {token_free[2]}  {token_free[3]}",
                    (255, 250, 0),
                    font=sprites.myFont_end,
                )
                back_ground.paste(
                    sprites.token_board, (175 * id * RATIO + 50 * RATIO, 775 * RATIO)
                )

    card_point_on_board = state[194:219]
    for id in range(5):
        card_id_in4 = card_point_on_board[5 * id : 5 * (id + 1)]
        id_img = -1
        for idd in range(len(sprites.ALL_CARD_POINT_IN4)):
            if np.sum(card_id_in4 == sprites.ALL_CARD_POINT_IN4[idd]) == 5:
                id_img = idd
                break
        back_ground.paste(
            sprites.victory_card[id_img], (175 * id * RATIO + 50 * RATIO, 1100 * RATIO)
        )

    draw.text(
        (25 * RATIO, 500 * RATIO),
        f"Phase: {phase}\nCheck_end: {check_end}  ",
        (255, 0, 0),
        font=sprites.myFont,
    )
    if state[220]:
        back_ground.paste(
            sprites.coin_gold, (100 * RATIO, 1300 * RATIO), sprites.coin_gold
        )
        draw.text(
            (100 * RATIO, 1350 * RATIO),
            f"{int(state[220])}",
            (255, 255, 0),
            font=sprites.myFont,
        )
    if state[219]:
        back_ground.paste(
            sprites.coin_silver, (250 * RATIO, 1300 * RATIO), sprites.coin_silver
        )
        draw.text(
            (250 * RATIO, 1350 * RATIO),
            f"{int(state[219])}",
            (255, 255, 255),
            font=sprites.myFont,
        )

    player_in4 = state[:6].astype(np.int64)
    other_player_in4 = state[96:120].astype(np.int64)
    id_start = state[272:277].astype(np.int64)

    draw.text(
        sprites.player_text_coor[0],
        f"Token: {player_in4[2]}   {player_in4[3]}   {player_in4[4]}   {player_in4[5]}\nScore: {player_in4[0]}, {id_start[0]} \nNumber_point_card: {player_in4[1]} ",
        (255, 255, 0),
        font=sprites.myFont,
    )

    for i in range(4):
        player_in4_i = other_player_in4[6 * i : 6 * (i + 1)]
        draw.text(
            sprites.player_text_coor[i + 1],
            f"Token: {player_in4_i[2]}   {player_in4_i[3]}   {player_in4_i[4]}   {player_in4_i[5]}\nScore: {player_in4_i[0]}, {id_start[i+1]} \nNumber_point_card: {player_in4_i[1]} ",
            (255, 255, 0),
            font=sprites.myFont,
        )
    # state[240] = 1
    if np.sum(state[221:266]) > 0:
        last_action = np.where(state[221:266] == 1)[0][0]
        # print('check', last_action)
        back_ground.paste(sprites.normal_card[last_action], (500 * RATIO, 1400 * RATIO))

    return back_ground


def draw_card(card_back_ground, card_in4, list_img_source):
    give = card_in4[0:4]
    receive = card_in4[4:8]
    upgrade = card_in4[-1]
    # print(give, receive, upgrade)
    len_card = np.sum(card_in4)
    if upgrade != 0 or np.sum(give) != 0:
        len_card += 1
    img = Image.new("RGB", (50, 40 * len_card), color="white")
    if upgrade != 0:
        cube = list_img_source[-1]
        for i in range(upgrade):
            idx = np.array([5, 40 * i])
            img.paste(cube, tuple(idx))
        up_img = Image.open("./src/Base/Century/images/uparrow.png")
        img.paste(up_img, (5, 40 * upgrade))
    else:
        if np.sum(give) != 0:
            # print('vào đây')
            card_back_ground = Image.open(
                f"./src/Base/Century/images/img/resource-card-trading.png"
            ).resize((300, 400))
            down_img = Image.open("./src/Base/Century/images/downarrow.png")
            index_run = 0
            for type in range(4):
                if give[type] != 0:
                    cube = list_img_source[type]
                    for i in range(give[type]):
                        idx = np.array([5, 40 * index_run])
                        img.paste(cube, tuple(idx))
                        index_run += 1
            img.paste(down_img, (5, 40 * index_run))
            index_run += 1
            # print('check', index_run)
            for type in range(4):
                if receive[type] != 0:
                    cube = list_img_source[type]
                    for i in range(receive[type]):
                        idx = np.array([5, 40 * index_run])
                        img.paste(cube, tuple(idx))
                        index_run += 1
        else:
            # print('vào đây2')

            card_back_ground = Image.open(
                f"./src/Base/Century/images/img/resource-card-production.png"
            ).resize((300, 400))
            index_run = 0
            for type in range(4):
                if receive[type] != 0:
                    cube = list_img_source[type]
                    for i in range(receive[type]):
                        idx = np.array([5, 40 * index_run])
                        img.paste(cube, tuple(idx))
                        index_run += 1

    card_back_ground.paste(img, (30, 20))
    return card_back_ground


def draw_point_card(card_back_ground, card_in4, list_img_source):
    give = card_in4[:4]
    score = card_in4[-1]
    font_t = ImageFont.truetype("src/ImageFonts/FreeMonoBoldOblique.ttf", size=100)
    len_card = np.sum(give)
    img = Image.new("RGB", (40 * len_card, 50), color="white")
    index_run = 0
    # print(give)
    for type in range(4):
        if give[type] != 0:
            cube = list_img_source[type]
            for i in range(give[type]):
                idx = np.array([40 * index_run, 5])
                img.paste(cube, tuple(idx))
                index_run += 1
    draw = ImageDraw.Draw(card_back_ground)
    draw.text((50, 200), f"{score}", (255, 255, 0), font=font_t)
    card_back_ground.paste(img, (int((300 - 40 * len_card) / 2), 330))
    return card_back_ground


# type = ['yellow', 'red', 'green', 'brown', 'any']

# list_type = []
# for type_i in type:
#     list_type.append(Image.open(f'./src/Base/Century/images/img/cube-{type_i}.png'))

# list_type
