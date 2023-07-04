import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.Base.CatanNoExchange import env as _env
from src.Base.CatanNoExchange.env import MAX_TURN_IN_ONE_GAME
from env import SHORT_PATH

IMG_PATH = SHORT_PATH + "src/Base/CatanNoExchange/images/"

RES_NAME = ["lumber", "brick", "wool", "grain", "ore"]


class Sprites:
    def __init__(self) -> None:
        self.background = Image.open(IMG_PATH + "background.png")
        self._background_ = self.background.copy()

        self.tiles = []
        for tile in RES_NAME + ["desert"]:
            tile_img = (
                Image.open(IMG_PATH + f"tile_{tile}.png")
                .resize((133, 152))
                .convert("RGBA")
            )
            self.tiles.append(tile_img)

        self.probs = {}
        for i in range(2, 13):
            if i != 7:
                prob_img = (
                    Image.open(IMG_PATH + f"prob_{i}.png")
                    .resize((55, 55))
                    .convert("RGBA")
                )
                self.probs[i] = prob_img

        self.ports = []
        for port in RES_NAME:
            port_img = (
                Image.open(IMG_PATH + f"port_{port}.png")
                .resize((80, 80))
                .convert("RGBA")
            )
            self.ports.append(port_img)
        else:
            port_img = (
                Image.open(IMG_PATH + "port.png").resize((80, 80)).convert("RGBA")
            )
            self.ports.append(port_img)

        self.robber = (
            Image.open(IMG_PATH + "icon_robber.png").resize((55, 55)).convert("RGBA")
        )

        self.rds = []
        self.rds_60 = []
        self.rds_120 = []
        self.sts = []
        self.cts = []
        for color in ["mysticblue", "gold", "silver", "bronze"]:
            road_img = (
                Image.open(IMG_PATH + f"road_{color}.png")
                .resize((80, 80))
                .convert("RGBA")
            )
            sett_img = (
                Image.open(IMG_PATH + f"settlement_{color}.png")
                .resize((55, 55))
                .convert("RGBA")
            )
            city_img = (
                Image.open(IMG_PATH + f"city_{color}.png")
                .resize((55, 55))
                .convert("RGBA")
            )
            self.rds.append(road_img)
            self.rds_60.append(road_img.rotate(60, expand=True))
            self.rds_120.append(road_img.rotate(120, expand=True))
            self.sts.append(sett_img)
            self.cts.append(city_img)

        self.highlight_circle = (
            Image.open(IMG_PATH + "icon_highlight_circle_white.png")
            .resize((40, 40))
            .convert("RGBA")
        )
        self.icon_check = (
            Image.open(IMG_PATH + "icon_check.png").resize((50, 50)).convert("RGBA")
        )

        self.largest_army = (
            Image.open(IMG_PATH + "icon_largest_army_highlight.png")
            .resize((40, 40))
            .convert("RGBA")
        )
        self.longest_road = (
            Image.open(IMG_PATH + "icon_longest_road_highlight.png")
            .resize((40, 40))
            .convert("RGBA")
        )

        self.dice = {}
        for i in range(1, 7):
            self.dice[i] = (
                Image.open(IMG_PATH + f"dice_{i}.png").resize((60, 60)).convert("RGBA")
            )


sprites = Sprites()


class Params:
    def __init__(self) -> None:
        self.tile_center_pos = [
            (1397, 210),
            (1538, 210),
            (1609, 331),
            (1679, 453),
            (1609, 574),
            (1538, 696),
            (1397, 696),
            (1256, 696),
            (1186, 574),
            (1115, 453),
            (1186, 331),
            (1256, 210),
            (1327, 331),
            (1256, 453),
            (1327, 574),
            (1468, 574),
            (1538, 453),
            (1468, 331),
            (1397, 453),
        ]
        self.port_topleft_pos = [
            (1421, 31),
            (1632, 153),
            (1774, 396),
            (1632, 639),
            (1421, 759),
            (1139, 758),
            (999, 517),
            (999, 274),
            (1139, 31),
        ]
        self.font24 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 24)
        self.font28 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 28)
        self.font32 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 32)
        self.road_center_pos = [
            (1362, 149),
            (1433, 149),
            (1468, 210),
            (1503, 149),
            (1573, 149),
            (1609, 210),
            (1573, 270),
            (1644, 270),
            (1679, 331),
            (1644, 392),
            (1715, 392),
            (1750, 452),
            (1715, 513),
            (1644, 513),
            (1679, 574),
            (1644, 635),
            (1573, 635),
            (1609, 696),
            (1573, 751),
            (1503, 751),
            (1468, 696),
            (1433, 756),
            (1362, 756),
            (1327, 696),
            (1291, 756),
            (1221, 756),
            (1186, 696),
            (1221, 635),
            (1151, 635),
            (1115, 574),
            (1151, 513),
            (1080, 513),
            (1045, 452),
            (1080, 392),
            (1151, 392),
            (1115, 331),
            (1151, 270),
            (1221, 270),
            (1186, 210),
            (1221, 149),
            (1291, 149),
            (1327, 210),
            (1291, 270),
            (1256, 331),
            (1291, 392),
            (1221, 392),
            (1186, 452),
            (1221, 513),
            (1291, 513),
            (1256, 574),
            (1291, 635),
            (1362, 635),
            (1397, 574),
            (1433, 635),
            (1503, 635),
            (1538, 574),
            (1503, 513),
            (1573, 513),
            (1609, 452),
            (1573, 392),
            (1503, 392),
            (1538, 331),
            (1503, 270),
            (1433, 270),
            (1397, 331),
            (1362, 270),
            (1362, 392),
            (1327, 452),
            (1362, 513),
            (1433, 513),
            (1468, 452),
            (1433, 392),
        ]
        self.point_coords = [
            (1397, 129),
            (1468, 169),
            (1538, 129),
            (1609, 169),
            (1609, 250),
            (1679, 290),
            (1679, 372),
            (1750, 412),
            (1750, 493),
            (1679, 533),
            (1679, 615),
            (1609, 655),
            (1609, 736),
            (1538, 766),
            (1468, 736),
            (1397, 776),
            (1327, 736),
            (1256, 776),
            (1186, 736),
            (1186, 655),
            (1115, 615),
            (1115, 533),
            (1045, 493),
            (1045, 412),
            (1115, 372),
            (1115, 290),
            (1186, 250),
            (1186, 169),
            (1256, 129),
            (1327, 169),
            (1327, 250),
            (1256, 290),
            (1256, 372),
            (1186, 412),
            (1186, 493),
            (1256, 533),
            (1256, 615),
            (1327, 655),
            (1397, 615),
            (1468, 655),
            (1538, 615),
            (1538, 533),
            (1609, 493),
            (1609, 412),
            (1538, 372),
            (1538, 290),
            (1468, 250),
            (1397, 290),
            (1397, 372),
            (1327, 412),
            (1327, 493),
            (1397, 533),
            (1468, 493),
            (1468, 412),
        ]


params = Params()


class CheckInitMap:
    def __init__(self) -> None:
        self.initialized = False
        self.tile_states = None
        self.port_states = None
        self.prob_states = None


checkInitMap = CheckInitMap()


def check_init_map(state):
    if not checkInitMap.initialized:
        return True

    if (checkInitMap.tile_states != state[0:114]).any():
        return True

    if (checkInitMap.prob_states != state[133:361]).any():
        return True

    if (checkInitMap.port_states != state[361:415]).any():
        return True

    return False


def draw_outlined_text(draw, text, font, pos, color, opx):
    o_color = (255 - color[0], 255 - color[1], 255 - color[2])
    x = pos[0]
    y = pos[1]
    draw.text((x + opx, y + opx), text, o_color, font)
    draw.text((x - opx, y + opx), text, o_color, font)
    draw.text((x + opx, y - opx), text, o_color, font)
    draw.text((x - opx, y - opx), text, o_color, font)
    draw.text(pos, text, color, font)


def init_map(state):
    sprites.background = sprites._background_.copy()
    checkInitMap.initialized = True
    checkInitMap.tile_states = state[0:114].copy()
    checkInitMap.prob_states = state[133:361].copy()
    checkInitMap.port_states = state[361:415].copy()

    tiles = []
    for i in range(19):
        tiles.append(np.where(state[6 * i : 6 + 6 * i] == 1)[0][0])

    for i in range(19):
        sprites.background.paste(
            sprites.tiles[tiles[i]],
            (params.tile_center_pos[i][0] - 66, params.tile_center_pos[i][1] - 76),
            sprites.tiles[tiles[i]],
        )

    probs = []
    for i in range(19):
        probs.append(np.where(state[133 + 12 * i : 145 + 12 * i] == 1)[0][0] + 1)

    for i in range(19):
        if probs[i] != 1:
            sprites.background.paste(
                sprites.probs[probs[i]],
                (params.tile_center_pos[i][0] - 27, params.tile_center_pos[i][1] - 1),
                sprites.probs[probs[i]],
            )

    ports = []
    for i in range(9):
        ports.append(np.where(state[361 + 6 * i : 361 + 6 * (i + 1)] == 1)[0][0])

    for i in range(9):
        sprites.background.paste(
            sprites.ports[ports[i]], params.port_topleft_pos[i], sprites.ports[ports[i]]
        )


def get_state_image(state=None):
    if state is None:
        return sprites.background

    if check_init_map(state):
        init_map(state)

    bg = sprites.background.copy()

    #  Draw
    draw = ImageDraw.ImageDraw(bg)

    #  Vẽ Robber
    rob_pos = np.where(state[114:133] == 1)[0][0]
    bg.paste(
        sprites.robber,
        (
            params.tile_center_pos[rob_pos][0] - 55,
            params.tile_center_pos[rob_pos][1] - 27,
        ),
        sprites.robber,
    )

    #  Tài nguyên ngân hàng, thẻ phát triển
    bank_res = state[415:421]
    for i in range(6):
        if bank_res[i] == 0:
            text = "0"
        else:
            text = "?"

        draw_outlined_text(draw, text, params.font24, (455 + 50 * i, 182), (0, 0, 0), 1)

    #  Nguyên liệu của bản thân
    my_res = state[421:426]
    for i in range(5):
        text = str(int(my_res[i]))
        draw_outlined_text(draw, text, params.font24, (455 + 50 * i, 732), (0, 0, 0), 1)

    #  Nguyên liệu còn lại trong kho
    my_inv_res = state[1268:1273]
    for i in range(5):
        text = str(int(my_inv_res[i]))
        draw_outlined_text(draw, text, params.font24, (455 + 50 * i, 702), (0, 0, 0), 1)

    #  Thẻ phát triển của bản thân
    my_dev = state[426:431]
    for i in range(5):
        text = str(int(my_dev[i]))
        draw_outlined_text(draw, text, params.font24, (455 + 50 * i, 807), (0, 0, 0), 1)

    #  Điểm của bản thân
    my_score = str(int(state[431]))
    bbox = draw.textbbox((0, 0), my_score, params.font28)
    draw_outlined_text(
        draw,
        my_score,
        params.font28,
        (350 - bbox[2] / 2, 832 - bbox[3]),
        (255, 255, 255),
        1,
    )

    #  Điểm của 3 người chơi khác
    for i in range(3):
        score = str(int(state[631 + 185 * i]))
        bbox = draw.textbbox((0, 0), score, params.font28)
        draw_outlined_text(
            draw,
            score,
            params.font28,
            (350 - bbox[2] / 2, 382 + 150 * i - bbox[3]),
            (255, 255, 255),
            1,
        )

    #  Số thẻ knight đã dùng và con đường dài nhất của bản thân
    num_knight = str(int(state[612]))
    bbox = draw.textbbox((0, 0), num_knight, params.font24)
    draw_outlined_text(
        draw, num_knight, params.font24, (310 - bbox[2] / 2, 777), (0, 0, 0), 1
    )
    longest_road = str(int(state[613]))
    bbox = draw.textbbox((0, 0), longest_road, params.font24)
    draw_outlined_text(
        draw, longest_road, params.font24, (380 - bbox[2] / 2, 777), (0, 0, 0), 1
    )

    #  Số thẻ knight đã dùng và con đường dài nhất của 3 người chơi khác
    for i in range(3):
        num_knight = str(int(state[812 + 185 * i]))
        bbox = draw.textbbox((0, 0), num_knight, params.font24)
        draw_outlined_text(
            draw,
            num_knight,
            params.font24,
            (310 - bbox[2] / 2, 327 + 150 * i),
            (0, 0, 0),
            1,
        )
        longest_road = str(int(state[813 + 185 * i]))
        bbox = draw.textbbox((0, 0), longest_road, params.font24)
        draw_outlined_text(
            draw,
            longest_road,
            params.font24,
            (380 - bbox[2] / 2, 327 + 150 * i),
            (0, 0, 0),
            1,
        )

    #  Tổng số thẻ tài nguyên và tổng số thẻ phát triển của 3 người chơi khác
    for i in range(3):
        num_res = str(int(state[629 + 185 * i]))
        draw_outlined_text(
            draw, num_res, params.font24, (455, 282 + 150 * i), (0, 0, 0), 1
        )
        num_dev = str(int(state[630 + 185 * i]))
        draw_outlined_text(
            draw, num_dev, params.font24, (455, 357 + 150 * i), (0, 0, 0), 1
        )

    #  Đường của bản thân
    my_roads = np.where(state[432:504] == 1)[0]
    draw_roads(bg, my_roads, 0)

    #  Đường của 3 người chơi khác
    for i in range(3):
        roads = np.where(state[632 + 185 * i : 704 + 185 * i] == 1)[0]
        draw_roads(bg, roads, i + 1)

    #  Nhà và thành phố của bản thân
    setts = np.where(state[504:558] == 1)[0]
    draw_setts(bg, setts, 0, city=False)
    cities = np.where(state[558:612] == 1)[0]
    draw_setts(bg, cities, 0, city=True)

    #  Nhà và thành phố của 3 người chơi khác
    for i in range(3):
        setts = np.where(state[704 + 185 * i : 758 + 185 * i])[0]
        draw_setts(bg, setts, i + 1, city=False)
        cities = np.where(state[758 + 185 * i : 812 + 185 * i])[0]
        draw_setts(bg, cities, i + 1, city=True)

    #  Phase
    phase = np.where(state[1273:1286] == 1)[0][0]
    text = f"Phase {phase}: " + phase_annotations[phase]
    draw_outlined_text(draw, text, params.font28, (280, 65), (255, 255, 255), 1)

    #  Điểm đặt thứ nhất
    try:
        pos = np.where(state[1204:1258] == 1)[0][0]
        bg.paste(
            sprites.highlight_circle,
            (params.point_coords[pos][0] - 20, params.point_coords[pos][1] - 20),
            sprites.highlight_circle,
        )
    except:
        pass

    #  Trade offer
    for i in range(5):
        text = str(int(state[1286 + i]))
        draw_outlined_text(
            draw, text, params.font24, (455 + 300 + 50 * i, 732), (0, 0, 0), 1
        )

    #  Danh hiệu quân đội mạnh nhất
    lgar = np.where(state[1184:1188] == 1)[0]
    if len(lgar) > 0:
        idx = lgar[0]
        if idx == 0:
            y = 740
        else:
            y = 140 + 150 * idx

        bg.paste(sprites.largest_army, (290, y), sprites.largest_army)

    #  Danh hiệu con đường dài nhất
    lgrd = np.where(state[1188:1192] == 1)[0]
    if len(lgrd) > 0:
        idx = lgrd[0]
        if idx == 0:
            y = 740
        else:
            y = 140 + 150 * idx

        bg.paste(sprites.longest_road, (360, y), sprites.longest_road)

    #  Tổng xx
    try:
        xx = np.where(state[1192:1203] == 1)[0][0] + 2
    except:
        xx = -1

    if xx != -1:
        if xx <= 7:
            dice_1 = 1
            dice_2 = xx - dice_1
        else:
            dice_1 = 6
            dice_2 = xx - dice_1

        bg.paste(sprites.dice[dice_1], (770, 150), sprites.dice[dice_1])
        bg.paste(sprites.dice[dice_2], (840, 150), sprites.dice[dice_2])

    return bg


phase_annotations = [
    "Place initial settlements",
    "Choose endpoints of the road",
    "Roll the dice or use development card",
    "Discard resources",
    "Move the robber",
    "Take resources at early of match",
    "Choose a module",
    "Choose a type of resource when using development card",
    "Place a settlement",
    "Place a city",
    "Choose a type of resource to trade with bank",
    "Choose a type of resource to receive from bank",
    "Take a resource from inventory",
]


def draw_setts(bg, setts, idx, city=False):
    if city:
        img = sprites.cts[idx]
    else:
        img = sprites.sts[idx]

    for sett in setts:
        bg.paste(
            img,
            (params.point_coords[sett][0] - 27, params.point_coords[sett][1] - 27),
            img,
        )


def draw_roads(bg, roads, idx):
    for road in roads:
        pos_1 = params.point_coords[_env.ROAD_POINT[road][0]]
        pos_2 = params.point_coords[_env.ROAD_POINT[road][1]]
        x = pos_1[0] - pos_2[0]
        y = pos_1[1] - pos_2[1]
        z = x * y

        if z == 0:
            img = sprites.rds[idx]
        elif z > 0:
            img = sprites.rds_60[idx]
        else:
            img = sprites.rds_120[idx]

        pos = params.road_center_pos[road]
        bg.paste(img, (int(pos[0] - img.width / 2), int(pos[1] - img.height / 2)), img)


action_annotations = [
    "Choose the point with index 0",
    "Choose the point with index 1",
    "Choose the point with index 2",
    "Choose the point with index 3",
    "Choose the point with index 4",
    "Choose the point with index 5",
    "Choose the point with index 6",
    "Choose the point with index 7",
    "Choose the point with index 8",
    "Choose the point with index 9",
    "Choose the point with index 10",
    "Choose the point with index 11",
    "Choose the point with index 12",
    "Choose the point with index 13",
    "Choose the point with index 14",
    "Choose the point with index 15",
    "Choose the point with index 16",
    "Choose the point with index 17",
    "Choose the point with index 18",
    "Choose the point with index 19",
    "Choose the point with index 20",
    "Choose the point with index 21",
    "Choose the point with index 22",
    "Choose the point with index 23",
    "Choose the point with index 24",
    "Choose the point with index 25",
    "Choose the point with index 26",
    "Choose the point with index 27",
    "Choose the point with index 28",
    "Choose the point with index 29",
    "Choose the point with index 30",
    "Choose the point with index 31",
    "Choose the point with index 32",
    "Choose the point with index 33",
    "Choose the point with index 34",
    "Choose the point with index 35",
    "Choose the point with index 36",
    "Choose the point with index 37",
    "Choose the point with index 38",
    "Choose the point with index 39",
    "Choose the point with index 40",
    "Choose the point with index 41",
    "Choose the point with index 42",
    "Choose the point with index 43",
    "Choose the point with index 44",
    "Choose the point with index 45",
    "Choose the point with index 46",
    "Choose the point with index 47",
    "Choose the point with index 48",
    "Choose the point with index 49",
    "Choose the point with index 50",
    "Choose the point with index 51",
    "Choose the point with index 52",
    "Choose the point with index 53",
    "Roll the dice",
    "Use knight",
    "Use road building",
    "Use year of plenty",
    "Use monopoly",
    "Choose lumber to receive",
    "Choose brick to receive",
    "Choose wool to receive",
    "Choose grain to receive",
    "Choose ore to receive",
    "Choose the tile with index 0",
    "Choose the tile with index 1",
    "Choose the tile with index 2",
    "Choose the tile with index 3",
    "Choose the tile with index 4",
    "Choose the tile with index 5",
    "Choose the tile with index 6",
    "Choose the tile with index 7",
    "Choose the tile with index 8",
    "Choose the tile with index 9",
    "Choose the tile with index 10",
    "Choose the tile with index 11",
    "Choose the tile with index 12",
    "Choose the tile with index 13",
    "Choose the tile with index 14",
    "Choose the tile with index 15",
    "Choose the tile with index 16",
    "Choose the tile with index 17",
    "Choose the tile with index 18",
    "Buy a road",
    "Buy a settlement",
    "Buy a city",
    "Buy a development card",
    "Create a trade offer with bank",
    "End of turn",
    "Choose lumber to give (or discard)",
    "Choose brick to give (or discard)",
    "Choose wool to give (or discard)",
    "Choose grain to give (or discard)",
    "Choose ore to give (or discard)",
    "Take a resource from inventory",
]


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""

    return action_annotations[action]


class Env_components:
    def __init__(self, env, winner, list_other) -> None:
        self.env = env
        self.winner = winner
        self.list_other = list_other


def get_env_components():
    env = _env.initEnv()
    winner = _env.checkEnded(env)
    list_other = np.array([-1, 1, 2, 3])
    np.random.shuffle(list_other)

    env_components = Env_components(env, winner, list_other)
    return env_components


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    if not action is None:
        _env.stepEnv(env_components.env, action)

    env_components.winner = _env.checkEnded(env_components.env)
    if env_components.winner == -1:
        while env_components.env[230] < MAX_TURN_IN_ONE_GAME:
            p_idx = int(env_components.env[244])
            if env_components.list_other[p_idx] == -1:
                break

            state = _env.getAgentState(env_components.env)
            agent = list_agent[env_components.list_other[p_idx] - 1]
            data = list_data[env_components.list_other[p_idx] - 1]
            action, data = agent(state, data)
            _env.stepEnv(env_components.env, action)
            env_components.winner = _env.checkEnded(env_components.env)
            if env_components.winner != -1:
                break

        if (
            env_components.winner != -1
            or env_components.env[230] == MAX_TURN_IN_ONE_GAME
        ):
            env_components.env[280] = 1
    else:
        env_components.env[280] = 1

    if env_components.env[280] == 0:
        state = _env.getAgentState(env_components.env)
        win = -1
    else:
        env_components.env[np.array([68, 110, 152, 194])] += env_components.env[
            np.array([67, 109, 151, 193])
        ]
        my_idx = np.where(env_components.list_other == -1)[0][0]
        env = env_components.env.copy()
        env[229] = 2
        env[244] = my_idx
        state = _env.getAgentState(env)
        if my_idx == env_components.winner:
            win = 1
        else:
            win = 0

        #  Chạy turn cuối cho 3 bot hệ thống
        for p_idx in range(4):
            if p_idx != my_idx:
                env[244] = p_idx
                _state = _env.getAgentState(env)
                agent = list_agent[env_components.list_other[p_idx] - 1]
                data = list_data[env_components.list_other[p_idx] - 1]
                action, data = agent(_state, data)

    return win, state, env_components
