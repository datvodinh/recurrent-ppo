import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from src.Base.SushiGo import env as _env
from env import SHORT_PATH

IMG_PATH = SHORT_PATH + "src/Base/SushiGo/images/"
BG_SIZE = (1680, 720)
CARD_SIZE = (90, 130)


action_description = {
    0: "Tempura",
    1: "Sashimi",
    2: "Dumpling",
    3: "1 Maki Roll",
    4: "2 Maki Roll",
    5: "3 Maki Roll",
    6: "Salmon Nigiri",
    7: "Squid Nigiri",
    8: "Egg Nigiri",
    9: "Pudding",
    10: "Wasabi",
    11: "Chopsticks",
    12: "Use Chopsticks",
    13: "End Turn",
}


class Env_components:
    def __init__(self, env, winner, list_other, turn, round) -> None:
        self.env = env
        self.list_action = np.full((5, 3), 13)
        self.winner = winner
        self.list_other = list_other
        self.idx = -1
        self.turn = turn
        self.round = round
        self.count = 0
        self.state = None


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""

    return f"{action_description[action]}"


def get_env_components():
    env = _env.initEnv(5)
    winner = _env.winner_victory(env)
    list_other = np.array([-1, 1, 2, 3, 4])
    np.random.shuffle(list_other)
    while list_other[-1] == -1:
        #  print(list_other, 'change')
        np.random.shuffle(list_other)
    #  print(list_other)
    turn = env[1]
    round = env[0] - 1
    env_components = Env_components(env, winner, list_other, turn, round)
    return env_components


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    amount_player = 5
    if not action is None:
        env_components.list_action[env_components.idx][env_components.count] = action
        # print(env_components.turn, env_components.round, env_components.idx, env_components.count)
        # print(env_components.list_action)

    turn = env_components.env[1]
    check_end_game = False

    check_break = True
    while turn <= 7 * 3:
        turn = env_components.env[1]
        round = env_components.env[0] - 1
        env_components.turn = turn
        env_components.round = round
        if env_components.idx == 4:
            env_components.idx = -1
        #  if turn % 7 == 0:
        #      env_components.list_action = np.full((amount_player, 3), 13)

        check_use_chopsticks = False
        if env_components.list_other[env_components.idx] == -1:
            if (
                12 in env_components.list_action[env_components.idx]
                and env_components.count != 2
            ):
                check_use_chopsticks = True
                env_components.idx -= 1
            if env_components.count == 2:
                env_components.count = 0
                env_components.state = None

        for idx in range(env_components.idx + 1, amount_player):
            env_components.idx = idx
            player_state = _env.getAgentState(env_components.env, idx)
            count = env_components.count
            while player_state[-1] + player_state[-2] > 0:
                if env_components.list_other[idx] == -1:
                    check_break = False
                    if check_use_chopsticks:
                        # print('Use chopsticks')
                        # print(player_state[14:28])
                        if action == 12:
                            env_components.state = player_state
                        player_state = _env.test_action(env_components.state, action)
                        # print(player_state[14:28])
                        env_components.state = player_state
                        env_components.count += 1

                    break
                agent = list_agent[env_components.list_other[idx] - 1]
                data = list_data[env_components.list_other[idx] - 1]
                action, data = agent(player_state, data)
                env_components.list_action[idx][count] = action
                count += 1
                env_components.count = count
                player_state = _env.test_action(player_state, action)

            #  env_components.idx += 1
            if check_break == False:
                break
            env_components.count = 0
        if check_break == False:
            break
        env_components.env = _env.stepEnv(
            env_components.env, env_components.list_action, amount_player, turn, round
        )
        env_components.list_action = np.full((amount_player, 3), 13)
        if turn % 7 == 0:
            env_components.env = _env.calculator_score(
                env_components.env, amount_player
            )
            if env_components.env[0] < 3:
                env_components.env[0] += 1
                env_components.env = _env.reset_card_player(env_components.env)
        if turn == 7 * 3:
            #  print('Tính pudding')
            env_components.env = _env.calculator_pudding(
                env_components.env, amount_player
            )
        if turn <= 7 * 3:
            env_components.env[1] += 1
            #  print(env_components.env[1], turn)

    if check_break == True:
        check_end_game = True
    env_components.winner = _env.winner_victory(env_components.env)

    if check_end_game == False:
        win = -1
    else:
        my_idx = np.where(env_components.list_other == -1)[0][0]
        env = env_components.env.copy()

        player_state = _env.getAgentState(env_components.env, env_components.idx)
        #  print(player_state[:2], player_state[14:28])
        #  print(env_components.list_other, env_components.idx)
        if my_idx in env_components.winner:
            win = 1
        else:
            win = 0

        #  Chạy turn cuối cho 3 bot hệ thống
        for idx in range(amount_player):
            if idx != my_idx:
                _state = _env.getAgentState(env, idx)
                agent = list_agent[env_components.list_other[idx] - 1]
                data = list_data[env_components.list_other[idx] - 1]
                action, data = agent(_state, data)

    return win, player_state, env_components


class Sprites:
    def __init__(self) -> None:
        self.background = Image.open(IMG_PATH + "bg.jpg").resize(BG_SIZE)
        card_values = [
            "temura",
            "sashimi",
            "dumpling",
            "1maki",
            "2maki",
            "3maki",
            "salmon",
            "squid",
            "egg",
            "pudding",
            "wasabi",
            "chopsticks",
        ]
        self.cards = []
        for value in card_values:
            self.cards.append(Image.open(IMG_PATH + f"{value}.jpg").resize(CARD_SIZE))


class Params:
    def __init__(self) -> None:
        self.center_card_x = BG_SIZE[0] * 0.5
        self.center_card_y = (BG_SIZE[1] - CARD_SIZE[1]) * 0.35

        x_0 = BG_SIZE[0] * 0.12
        x_1 = BG_SIZE[0] * 0.8
        y_0 = 0.1 * BG_SIZE[1] - 0.25 * CARD_SIZE[1]
        y_1 = 0.9 * BG_SIZE[1] - 0.75 * CARD_SIZE[1]
        x_center = BG_SIZE[0] * 0.45
        y_center = 0.9 * BG_SIZE[1] - 0.75 * CARD_SIZE[1]
        self.list_coords_1 = [
            (x_center, y_center),
            (x_0, y_1),
            (x_1, y_1),
            (x_1, y_0),
            (x_0, y_0),
        ]

        self.score_coords = [
            (x_center, y_center * 0.9),
            (x_0, y_1 * 0.9),
            (x_1, y_1 * 0.9),
            (x_1, y_0 * 6),
            (x_0, y_0 * 6),
        ]


params = Params()
sprites = Sprites()


def draw_cards(bg, cards, s, y):
    y = round(y)
    id_card = 1
    for card in range(12):
        total_cards = cards[card]
        while total_cards > 0:
            bg.paste(sprites.cards[card], (round(s + _d_ * id_card), y))
            total_cards -= 1
            id_card += 1


_d_ = CARD_SIZE[0] * 0.8

font_ = ImageFont.truetype("src/ImageFonts/FreeMonoBoldOblique.ttf", 60)


def get_state_image(state=None):
    background = sprites.background.copy()
    if state is None:
        return background
    else:
        n = np.sum(state[2:14])
        w = CARD_SIZE[0] + _d_ * (n - 1)
        s = params.center_card_x - 0.5 * w
        draw_cards(background, state[2:14], s, params.center_card_y)

    for k in range(5):
        list_cards_played = state[14 * (k + 1) + 2 : 14 * (k + 2)]
        score = int(state[14 * (k + 1)])
        count_puding = int(state[14 * (k + 1) + 1])
        n = np.sum(list_cards_played)
        w = CARD_SIZE[0] + _d_ * (n - 1)
        s = params.list_coords_1[k][0] - 0.5 * w
        draw_cards(background, list_cards_played, s, params.list_coords_1[k][1])

        ImageDraw.Draw(background).text(
            params.score_coords[k],
            f"{score}|{count_puding}",
            fill=(255, 255, 255),
            anchor="mm",
            font=font_,
        )
    return background
