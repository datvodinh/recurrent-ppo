import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.Base.Poker import env as _env
from env import SHORT_PATH

IMG_PATH = SHORT_PATH + "src/Base/Poker/images/"
BG_SIZE = (1680, 900)
CARD_SIZE = (70, 100)


class Sprites:
    def __init__(self) -> None:
        self.background = Image.open(IMG_PATH + "background.png").resize(BG_SIZE)
        card_values = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        card_suits = ["Heart", "Diamond", "Club", "Spade"]
        self.cards = []
        for value in card_values:
            for suit in card_suits:
                self.cards.append(
                    Image.open(IMG_PATH + f"{value}-{suit}.png").resize(CARD_SIZE)
                )

        self.cards.append(Image.open(IMG_PATH + f"Card_back.png").resize(CARD_SIZE))

        #  self.other_in4 = []
        #  self.other_in4.append(Image.open(IMG_PATH+f"button_dealer.png").resize(CARD_SIZE))


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
        self.myFont = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", size=40
        )
        self.fontFold = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", size=60
        )


params = Params()


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""
    else:
        action_mean = _env.ACTIONS_MEAN[action]

    return f"player {action_mean}"


class Env_components:
    def __init__(self, env, winner, list_other, showdown_yet) -> None:
        self.env = env
        self.winner = winner
        self.list_other = list_other
        self.showdown_yet = True


def get_env_components():
    env = _env.initEnv()
    env = _env.reset_round(env)
    winner = -1
    showdown_yet = True
    list_other = np.array([-1, 1, 2, 3, 4, 5, 6, 7, 8])
    np.random.shuffle(list_other)
    env_components = Env_components(env, winner, list_other, showdown_yet)
    return env_components


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    if not action is None:
        #  print('check qua 1')
        if env_components.env[_env.ENV_STATUS_GAME] != 6:
            env_components.env = _env.stepEnv(env_components.env, action)
    my_idx = np.where(env_components.list_other == -1)[0][0]
    #  print(f'bot là người chơi thứ {my_idx}')
    #  if _env.checkEnded(env_components.env):
    #      print('đoạn này thừa')
    #      env_components.env[_env.ENV_CHECK_END] = 1
    #      env_components.winner = _env.check_winner(env_components.env)

    while not _env.checkEnded(env_components.env):
        while env_components.env[_env.ENV_STATUS_GAME] != 6:
            p_idx = int(env_components.env[_env.ENV_ID_ACTION])

            if env_components.list_other[p_idx] == -1:
                #  print('toang đây')
                state = _env.getAgentState(env_components.env)
                win = -1
                return win, state, env_components

            state = _env.getAgentState(env_components.env)
            agent = list_agent[env_components.list_other[p_idx] - 1]
            data = list_data[env_components.list_other[p_idx] - 1]
            action, data = agent(state, data)
            if _env.getValidActions(state)[action] != 1:
                raise Exception("bot dua ra action khong hop le")
            #  print(f'người chơi {p_idx} action {action}')

            env_components.env = _env.stepEnv(env_components.env, action)

        # chạy turn showdown
        if env_components.showdown_yet:
            #  print('chạy show down')
            my_idx = -1
            for id in range(_env.NUMBER_PLAYER):
                env_components.env[_env.ENV_ID_ACTION] = id
                p_idx = int(env_components.env[_env.ENV_ID_ACTION])
                if env_components.env[_env.ENV_ALL_PLAYER_STATUS + p_idx] == 0:
                    continue
                else:
                    if env_components.list_other[p_idx] == -1:
                        my_idx = p_idx
                        continue
                    state = _env.getAgentState(env_components.env)
                    agent = list_agent[env_components.list_other[p_idx] - 1]
                    data = list_data[env_components.list_other[p_idx] - 1]
                    action, data = agent(state, data)
            env_components.showdown_yet = False
            if env_components.env[_env.ENV_ALL_PLAYER_STATUS + my_idx] == 1:
                #  print('check đây', my_idx)
                env_components.env[_env.ENV_ID_ACTION] = my_idx
                state = _env.getAgentState(env_components.env)
                win = -1
                return win, state, env_components
        else:
            #  print('reset round')
            env_components.env = _env.reset_round(env_components.env)
            env_components.showdown_yet = True

    env_components.env[_env.ENV_CHECK_END] = 1
    env_components.winner = _env.check_winner(env_components.env)

    #  for id in range(_env.NUMBER_PLAYER):
    #      env_components.env[_env.ENV_PHASE] = 1
    #      p_idx = int(env_components.env[_env.ENV_ID_ACTION])
    #      state = _env.getAgentState(env_components.env)
    #      my_idx = np.where(env_components.list_other == -1)[0][0]
    #      if my_idx == env_components.winner:
    #          win = 1
    #      else:
    #          win = 0

    my_idx = np.where(env_components.list_other == -1)[0][0]
    env_components.env[_env.ENV_PHASE] = 1
    # chạy turn bonus cho bot hệ thống
    for p_idx in range(_env.NUMBER_PLAYER):
        if p_idx == my_idx:
            continue
        else:
            env_components.env[_env.ENV_ID_ACTION] = p_idx
            _state = _env.getAgentState(env_components.env)
            agent = list_agent[env_components.list_other[p_idx] - 1]
            data = list_data[env_components.list_other[p_idx] - 1]
            action, data = agent(_state, data)
    env_components.env[_env.ENV_ID_ACTION] = my_idx
    state = _env.getAgentState(env_components.env)
    if my_idx == env_components.winner:
        win = 1
    else:
        win = 0

    return win, state, env_components


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
    # mảng tọa độ in ảnh
    arr_coordinate_card = np.array(
        [
            [400, 720],
            [1000, 720],
            [1300, 550],
            [1300, 300],
            [1050, 100],
            [700, 100],
            [350, 100],
            [150, 300],
            [150, 550],
        ]
    )
    arr_coordinate_button_dealer = np.array(
        [
            [440, 680],
            [1040, 680],
            [1270, 570],
            [1270, 330],
            [1070, 200],
            [740, 200],
            [390, 200],
            [250, 330],
            [250, 570],
        ]
    )
    arr_coordinate_small = np.array(
        [
            [400, 680],
            [1000, 680],
            [1270, 540],
            [1270, 300],
            [1040, 200],
            [710, 200],
            [360, 200],
            [250, 300],
            [250, 540],
        ]
    )
    arr_coordinate_big = np.array(
        [
            [480, 680],
            [1080, 680],
            [1270, 600],
            [1270, 360],
            [1100, 200],
            [770, 200],
            [420, 200],
            [250, 360],
            [250, 600],
        ]
    )
    arr_coordinate_status = np.array(
        [
            [420, 750],
            [1020, 750],
            [1320, 560],
            [1320, 320],
            [1080, 120],
            [720, 120],
            [370, 120],
            [180, 320],
            [180, 560],
        ]
    )
    arr_coordinate_chip_else = np.array(
        [
            [420, 830],
            [1020, 830],
            [1380, 560],
            [1380, 330],
            [1080, 70],
            [720, 70],
            [370, 70],
            [80, 330],
            [80, 560],
        ]
    )
    arr_coordinate_chip_in_pot = np.array(
        [
            [320, 800],
            [920, 800],
            [1320, 520],
            [1320, 270],
            [980, 120],
            [620, 120],
            [270, 120],
            [180, 270],
            [180, 520],
        ]
    )

    background = sprites.background.copy()
    draw = ImageDraw.Draw(background)
    myFont = params.myFont
    fontFold = params.fontFold

    if state is None:
        return background
    # lấy thông tin (trạng thái người chơi, tiền còn, tiền đã bỏ ra trong turn)
    arr_player_status = state[486:495]
    arr_player_chip_in_pot = state[477:486]
    arr_player_chip_else = state[468:477]
    player_in = (arr_player_chip_in_pot + arr_player_chip_else + arr_player_status) > 0

    # Lấy card bàn chơi
    open_card = np.array([])
    check = 0
    for i in range(9):
        player_i_card = np.where(state[52 * i : 52 * (i + 1)])[0]
        if len(open_card) == 0:
            open_card = player_i_card.copy()
            check += 1
        elif len(player_i_card) > 0 and len(open_card) > 0:
            open_card = np.intersect1d(open_card, player_i_card)
            check += 1
            break
    if check < 2:
        open_card = np.array([])
    #  print(open_card,  'card chung', 'số ván đã chơi', state[513], state[508:513])
    # lấy card cho người chơi
    for i in range(9):
        player_i_card = np.where(state[52 * i : 52 * (i + 1)])[0]
        if len(player_i_card) < 1 or len(player_i_card) == len(open_card):
            if player_in[i]:
                player_i_card = np.array([-1, -1])
            else:
                player_i_card = np.array([])
        else:
            player_i_card = np.setdiff1d(player_i_card, open_card)
            if len(player_i_card) == 1:
                player_i_card = np.append(player_i_card, -1)

        draw_cards(
            background,
            player_i_card,
            arr_coordinate_card[i][0],
            arr_coordinate_card[i][1],
        )
        draw.text(
            (arr_coordinate_chip_else[i][0], arr_coordinate_chip_else[i][1]),
            f"{int(state[468+i])}",
            (255, 0, 0),
            font=myFont,
        )

    open_card_coodinate = np.array(
        [[640, 400], [720, 400], [800, 400], [880, 400], [960, 400]]
    )
    for i in range(len(open_card)):
        draw_cards(
            background,
            open_card[i : i + 1],
            open_card_coodinate[i][0],
            open_card_coodinate[i][1],
        )
    # DEALER, SMALL, BIG, STATUS
    if np.sum(player_in) > 1 and state[513] < 100:
        dealer = np.where(state[495:504])[0][0]
        draw.text(
            (
                arr_coordinate_button_dealer[dealer][0],
                arr_coordinate_button_dealer[dealer][1],
            ),
            f"D",
            (255, 255, 255),
            font=myFont,
        )
        check_BS = 0
        id_player = (dealer + 1) % 9
        while check_BS < 2:
            if player_in[id_player] == 1:
                if check_BS == 0:
                    draw.text(
                        (
                            arr_coordinate_small[id_player][0],
                            arr_coordinate_small[id_player][1],
                        ),
                        f"S",
                        (255, 0, 0),
                        font=myFont,
                    )
                else:
                    draw.text(
                        (
                            arr_coordinate_big[id_player][0],
                            arr_coordinate_big[id_player][1],
                        ),
                        f"B",
                        (255, 255, 0),
                        font=myFont,
                    )
                check_BS += 1
            id_player = (id_player + 1) % 9

        for i in range(9):
            if arr_player_status[i] == 0:
                draw.text(
                    (arr_coordinate_status[i][0], arr_coordinate_status[i][1]),
                    f"F",
                    (255, 0, 0),
                    font=fontFold,
                )
            else:
                draw.text(
                    (
                        arr_coordinate_chip_in_pot[i][0],
                        arr_coordinate_chip_in_pot[i][1],
                    ),
                    f"{int(state[477 + i])}",
                    (255, 165, 0),
                    font=myFont,
                )

    draw.text((700, 500), f"Pot:{int(state[506])}", (255, 0, 0), font=myFont)
    return background
