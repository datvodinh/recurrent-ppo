import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from src.Base.StoneAge import env as _env
from env import SHORT_PATH

BUILDING_CARDS, CIV_CARDS = _env.BUILDING_CARDS, _env.CIV_CARDS
IMG_PATH = SHORT_PATH + "src/Base/StoneAge/images/"
SIZE_BOARD = (1080, 607)

tl = 2
BG_SIZE = (np.array([2585, 1821]) / tl).astype(np.int64)
SIZE_BOARD = (int(BG_SIZE[0] * 16 / 9), BG_SIZE[1])
BUILDING_CARDS_SIZE = (np.array([263, 314]) / (tl + 0.1)).astype(np.int64)
CIV_CARDS_SIZE = (np.array([335, 515]) / (tl + 0.1)).astype(np.int64)
ICON_SIZE = (np.array([120, 120]) / tl).astype(np.int64)

action_description = {
    0: "Dừng lấy công cụ",
    1: "Đặt 1 người",
    2: "Đặt 2 người",
    3: "Đặt 3 người",
    4: "Đặt 4 người",
    5: "Đặt 5 người",
    6: "Đặt 6 người",
    7: "Đặt 7 người",
    8: "Đặt 8 người",
    9: "Đặt 9 người",
    11: "Đặt người vào lúa",
    12: "Đặt vào ô công cụ",
    13: "Đặt vào ô sinh sản",
    14: "Đặt vào khu gỗ",
    15: "Đặt vào khu gạch",
    16: "Đặt vào khu bạc",
    17: "Đặt vào khu vàng",
    18: "Đặt vào khu lương thực",
    19: "Đặt người vào ô thẻ civ 0",
    20: "Đặt người vào ô thẻ civ 1",
    21: "Đặt người vào ô thẻ civ 2",
    22: "Đặt người vào ô thẻ civ 3",
    23: "Đặt người vào ô thẻ building 0",
    24: "Đặt người vào ô thẻ building 1",
    25: "Đặt người vào ô thẻ building 2",
    26: "Đặt người vào ô thẻ building 3",
    27: "Chọn trừ nguyên liệu (Khi đến hết vòng không đủ thức ăn)",
    28: "Chọn trừ điểm (Khi đến hết vòng không đủ thức ăn)",
    29: "Lấy người từ lúa",
    30: "Lấy người từ công cụ",
    31: "Lấy người từ sinh sản",
    32: "Lấy người từ gỗ",
    33: "Lấy người từ gạch",
    34: "Lấy người từ bạc",
    35: "Lấy người từ vàng",
    36: "Lấy người từ lương thực",
    37: "Dùng công cụ ở ô 1",
    38: "Dùng công cụ ở ô 2",
    39: "Dùng công cụ ở ô 3",
    40: "Trả nguyên liêu gỗ",
    41: "Trả nguyên liêu gạch",
    42: "Trả nguyên liêu bạc",
    43: "Trả nguyên liêu vàng",
    44: "Dùng công cụ một lần 1",
    45: "Dùng công cụ một lần 2",
    46: "Dùng công cụ một lần 3",
    47: "Dừng trả nguyên liệu khi mua thẻ build 1-7 (Thẻ build trả 1 đến 7 người để đổi ra điểm)",
    48: "Lấy người từ ô thẻ civ 0",
    49: "Lấy người từ ô thẻ civ 1",
    50: "Lấy người từ ô thẻ civ 2",
    51: "Lấy người từ ô thẻ civ 3",
    52: "Lấy người từ ô thẻ building 0",
    53: "Lấy người từ ô thẻ building 1",
    54: "Lấy người từ ô thẻ building 2",
    55: "Lấy người từ ô thẻ building 3",
    57: "Chọn xúc xắc số 1: Thêm 1 gỗ",
    58: "Chọn xúc xắc số 2: Thêm 1 gạch",
    59: "Chọn xúc xắc số 3: Thêm 1 bạc",
    60: "Chọn xúc xắc số 4: Thêm 1 vàng",
    61: "Chọn xúc xắc số 5: Thêm 1 công cụ",
    62: "Chọn xúc xắc số 6: Thêm 1 lúa",
    63: "Chọn dùng thẻ lấy thêm 2 nguyên liệu từ thẻ civ",
    64: "Trả nguyên liêu gỗ",
    65: "Trả nguyên liêu gạch",
    66: "Trả nguyên liêu bạc",
    67: "Trả nguyên liêu vàng",
}


class Env_components:
    def __init__(self, env, winner, list_other, all_build_card, all_civ_card) -> None:
        #  print('Thứ tự các nền văn minh trong mô tả: \n Hình người, cây sáo, lá cây, dệt, đồng hồ, lu, xe, đá')
        self.env = env
        self.winner = winner
        self.list_other = list_other
        self.all_build_card = all_build_card
        self.all_civ_card = all_civ_card
        self.cc = 0


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""
    return f"{action_description[action]}"


def get_env_components():
    env, all_build_card, all_civ_card = _env.initEnv(BUILDING_CARDS, CIV_CARDS)
    winner = _env.checkEnded(env)
    list_other = np.array([-1, 1, 2, 3])
    np.random.shuffle(list_other)
    env_components = Env_components(
        env, winner, list_other, all_build_card, all_civ_card
    )
    return env_components


class Sprites:
    def __init__(self) -> None:
        self.im = Image.new("RGB", SIZE_BOARD, "black")
        self.background = Image.open(IMG_PATH + "board.webp").resize(BG_SIZE)

        self.cards_building = []
        for value in range(28):
            self.cards_building.append(
                Image.open(f"{IMG_PATH}building_card/{value}.png").resize(
                    tuple(BUILDING_CARDS_SIZE)
                )
            )

        self.card_civilization = []
        for value in range(36):
            self.card_civilization.append(
                Image.open(f"{IMG_PATH}civ_card/{value}.png").resize(
                    tuple(CIV_CARDS_SIZE)
                )
            )

        self.font = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", int(120 / tl)
        )
        self.font2 = ImageFont.truetype(
            "src/ImageFonts/FreeMonoBoldOblique.ttf", int(60 / tl)
        )

        self.list_color = ["blue", "yellow", "red", "green"]
        self.list_res = ["wood", "brick", "silver", "gold"]
        self.list_people_x = ["home", "tool", "people", "field"]
        self.people = []
        self.res = []
        self.tools = []
        self.tools_temp = []
        self.list_total_people_x = []
        for i in range(4):
            self.people.append(
                Image.open(f"{IMG_PATH}icon/people_{self.list_color[i]}.png").resize(
                    tuple(ICON_SIZE)
                )
            )
            self.res.append(
                Image.open(f"{IMG_PATH}icon/res_{self.list_res[i]}.png").resize(
                    tuple(ICON_SIZE)
                )
            )
            self.tools.append(
                Image.open(f"{IMG_PATH}icon/tool_{i+1}.png").resize(tuple(ICON_SIZE))
            )
            self.tools_temp.append(
                Image.open(f"{IMG_PATH}icon/tool_temp_{i+1}.png").resize(
                    tuple(ICON_SIZE)
                )
            )
            self.list_total_people_x.append(
                Image.open(
                    f"{IMG_PATH}icon/people_x_{self.list_people_x[i]}.png"
                ).resize(tuple(ICON_SIZE))
            )

        self.score = Image.open(f"{IMG_PATH}icon/score.png").resize(tuple(ICON_SIZE))
        self.type_civ = Image.open(f"{IMG_PATH}icon/type_civ.png").resize(
            tuple(ICON_SIZE)
        )
        self.food = Image.open(f"{IMG_PATH}icon/food.png").resize(tuple(ICON_SIZE))
        self.field = Image.open(f"{IMG_PATH}icon/field.png").resize(tuple(ICON_SIZE))
        self.building_icon = Image.open(f"{IMG_PATH}icon/home.png").resize(
            tuple(ICON_SIZE)
        )
        self.main_player = Image.open(f"{IMG_PATH}icon/main_player.png").resize(
            tuple(ICON_SIZE)
        )


class Draw_Agent:
    def __init__(self) -> None:
        x2 = int((SIZE_BOARD[0] - BG_SIZE[0]) / 2) + BG_SIZE[0]
        y2 = int((SIZE_BOARD[1]) / 2)
        self.coords = [(0, 0), (x2, 0), (x2, y2 + 3), (0, y2 + 3)]
        pass

    def draw_agent_block(
        self,
        im,
        res_array=np.full((4, 4), 1),
        tool_array=np.full((4, 3), 2),
        state_tool_array=np.full((4, 3), 1),
        tool_temp_array=np.full((4, 3), 2),
        people_x_array=np.full((4, 4), 3),
        all_type_civ=np.full((4, 8), 4),
        score=[0, 0, 0, 0],
        field=[0, 0, 0, 0],
        peoples=[0, 0, 0, 0],
        food=[0, 0, 0, 0],
        building=[0, 0, 0, 0],
        type_civ=[0, 0, 0, 0],
    ):
        for i in range(4):
            x = self.coords[i][0]
            y = self.coords[i][1]

            im.paste(sprites.score, (x, y))
            ImageDraw.Draw(im).text(
                (x + ICON_SIZE[0], y + int(ICON_SIZE[1] / 2)),
                str(score[i]),
                fill="white",
                font=sprites.font2,
            )
            im.paste(sprites.field, (x + 2 * ICON_SIZE[0], y))
            ImageDraw.Draw(im).text(
                (x + 3 * ICON_SIZE[0], y + int(ICON_SIZE[1] / 2)),
                str(field[i]),
                fill="white",
                font=sprites.font2,
            )
            im.paste(sprites.people[i], (x + 4 * ICON_SIZE[0], y))
            ImageDraw.Draw(im).text(
                (x + 5 * ICON_SIZE[0], y + int(ICON_SIZE[1] / 2)),
                str(peoples[i]),
                fill="white",
                font=sprites.font2,
            )

            if i == 0:
                im.paste(sprites.main_player, (x + 6 * ICON_SIZE[0], y))

            im.paste(sprites.food, (x + 0 * ICON_SIZE[0], y + ICON_SIZE[1]))
            ImageDraw.Draw(im).text(
                (x + 1 * ICON_SIZE[0], y + 1 * ICON_SIZE[1] + int(ICON_SIZE[1] / 2)),
                str(food[i]),
                fill="white",
                font=sprites.font2,
            )
            im.paste(sprites.building_icon, (x + 2 * ICON_SIZE[0], y + ICON_SIZE[1]))
            ImageDraw.Draw(im).text(
                (x + 3 * ICON_SIZE[0], y + 1 * ICON_SIZE[1] + int(ICON_SIZE[1] / 2)),
                str(building[i]),
                fill="white",
                font=sprites.font2,
            )
            im.paste(sprites.type_civ, (x + 4 * ICON_SIZE[0], y + ICON_SIZE[1]))
            ImageDraw.Draw(im).text(
                (x + 5 * ICON_SIZE[0], y + 1 * ICON_SIZE[1] + int(ICON_SIZE[1] / 2)),
                str(type_civ[i]),
                fill="white",
                font=sprites.font2,
            )

            for res in range(4):
                x_ = x + 2 * ICON_SIZE[0] * res
                y_ = y + 2 * ICON_SIZE[1]
                im.paste(sprites.res[res], (x_, y_))
                ImageDraw.Draw(im).text(
                    (x_ + ICON_SIZE[0], y_ + int(ICON_SIZE[1] / 2)),
                    str(res_array[i][res]),
                    fill="white",
                    font=sprites.font2,
                )

            for tool in range(3):
                x_ = x + 2 * ICON_SIZE[0] * tool
                y_ = y + 3 * ICON_SIZE[1]
                id_tool = tool_array[i][tool] - 1
                if id_tool != -1:
                    im.paste(sprites.tools[id_tool], (x_, y_))
                    ImageDraw.Draw(im).text(
                        (x_ + ICON_SIZE[0], y_ + int(ICON_SIZE[1] / 2)),
                        f"{state_tool_array[i][tool]}",
                        fill="white",
                        font=sprites.font2,
                    )
            for tool_temp in range(3):
                x_ = x + 2 * ICON_SIZE[0] * tool_temp
                y_ = y + 4 * ICON_SIZE[1]
                id_tool_temp = tool_temp_array[i][tool_temp] - 1
                if id_tool_temp != -1:
                    im.paste(sprites.tools_temp[tool_temp + 1], (x_, y_))
                    ImageDraw.Draw(im).text(
                        (x_ + ICON_SIZE[0], y_ + int(ICON_SIZE[1] / 2)),
                        f"1",
                        fill="white",
                        font=sprites.font2,
                    )
            for people_x in range(4):
                x_ = x + 2 * ICON_SIZE[0] * people_x
                y_ = y + 5 * ICON_SIZE[1]
                im.paste(sprites.list_total_people_x[people_x], (x_, y_))
                ImageDraw.Draw(im).text(
                    (x_ + ICON_SIZE[0], y_ + int(ICON_SIZE[1] / 2)),
                    str(people_x_array[i][people_x]),
                    fill="white",
                    font=sprites.font2,
                )

            ImageDraw.Draw(im).text(
                (x, y + 6.4 * ICON_SIZE[1]),
                "Các loại nền văn minh:",
                fill="white",
                font=sprites.font2,
            )
            for id_type_civ in range(8):
                x_ = x + ICON_SIZE[0] * id_type_civ
                y_ = y + 7 * ICON_SIZE[1]
                ImageDraw.Draw(im).text(
                    (x_, y_),
                    str(all_type_civ[i][id_type_civ]),
                    fill="white",
                    font=sprites.font2,
                )

    def draw_line(self, im, color="White", width=3):
        w, h = int(im.size[0] / 2), int(im.size[1] / 2)
        shape = [(w, 0), (w, int((im.size[1])))]
        ImageDraw.Draw(im).line(shape, fill=color, width=width)
        shape = [(0, h), (im.size[0], h)]
        ImageDraw.Draw(im).line(shape, fill=color, width=width)


class Params:
    def __init__(self) -> None:
        x = int(BG_SIZE[0] * 0.28)
        y = int(BG_SIZE[1] * 0.17)
        self.list_coords_forest = [
            (x, y),
            (x + _d_, y),
            (x + _d_, y + _d_),
            (x, y + _d_),
        ]

        x = int(BG_SIZE[0] * 0.50)
        y = int(BG_SIZE[1] * 0.15)
        self.list_coords_rock = [(x, y), (x + _d_, y), (x + _d_, y + _d_), (x, y + _d_)]

        x = int(BG_SIZE[0] * 0.88)
        y = int(BG_SIZE[1] * 0.14)
        self.list_coords_silver = [
            (x, y),
            (x + _d_, y),
            (x + _d_, y + _d_),
            (x, y + _d_),
        ]

        x = int(BG_SIZE[0] * 0.75)
        y = int(BG_SIZE[1] * 0.39)
        self.list_coords_gold = [(x, y), (x + _d_, y), (x + _d_, y + _d_), (x, y + _d_)]

        x = int(BG_SIZE[0] * 0.1)
        y = int(BG_SIZE[1] * 0.25)
        self.list_coords_food = [(x, y), (x + _d_, y), (x + _d_, y + _d_), (x, y + _d_)]

        r_x = BG_SIZE[0] * 0.040
        r_y = BG_SIZE[1] * 0.039
        x = BG_SIZE[0] * 0.25
        y = BG_SIZE[1] * 0.57
        self.field = (x, y, x + r_x, y + r_y)

        x = BG_SIZE[0] * 0.33
        y = BG_SIZE[1] * 0.72
        self.hut = [
            (x, y, x + r_x, y + r_y),
            (int(x * 1.1), y, int(x * 1.1) + r_x, y + r_y),
        ]

        x = BG_SIZE[0] * 0.51
        y = BG_SIZE[1] * 0.53
        self.tool_maker = (x, y, x + r_x, y + r_y)

        x = int(BG_SIZE[0] * 0.04)
        y = int(BG_SIZE[1] * 0.89)
        self.point_building = []
        for i in range(4):
            self.point_building.append((x, y, x + r_x, y + r_y))
            x += int(BG_SIZE[0] * 0.112)
        self.point_building = self.point_building[::-1]

        x = int(BG_SIZE[0] * 0.57)
        y = int(BG_SIZE[1] * 0.84)
        self.point_civ = []
        for i in range(4):
            self.point_civ.append((x, y, x + r_x, y + r_y))
            x += int(BG_SIZE[0] * 0.126)
        self.point_civ = self.point_civ[::-1]

        self.list_coords_dice_tool_maker = (
            int(BG_SIZE[0] * 0.1),
            int(BG_SIZE[1] * 0.05),
        )
        self.list_coords_card_choose_dice = (
            int(BG_SIZE[0] * 0.4),
            int(BG_SIZE[1] * 0.05),
        )

    def draw_field(self, bg, color):
        if color != "white":
            ImageDraw.Draw(bg).ellipse(params.field, fill=color)

    def draw_hut(self, bg, color):
        if color != "white":
            for i in range(2):
                ImageDraw.Draw(bg).ellipse(params.hut[i], fill=color)

    def draw_tool_maker(self, bg, color):
        if color != "white":
            ImageDraw.Draw(bg).ellipse(params.tool_maker, fill=color)

    def draw_building(self, bg, colors):
        for i in range(4):
            if colors[i] != "white":
                ImageDraw.Draw(bg).ellipse(params.point_building[i], fill=colors[i])

    def draw_civ(self, bg, colors):
        for i in range(4):
            if colors[i] != "white":
                ImageDraw.Draw(bg).ellipse(params.point_civ[i], fill=colors[i])

    def draw_forest(self, bg, list_count_res):
        for i in range(4):
            ImageDraw.Draw(bg).text(
                self.list_coords_forest[i],
                str(list_count_res[i]),
                fill=sprites.list_color[i],
                font=sprites.font,
            )

    def draw_rock(self, bg, list_count_res):
        for i in range(4):
            ImageDraw.Draw(bg).text(
                self.list_coords_rock[i],
                str(list_count_res[i]),
                fill=sprites.list_color[i],
                font=sprites.font,
            )

    def draw_silver(self, bg, list_count_res):
        for i in range(4):
            ImageDraw.Draw(bg).text(
                self.list_coords_silver[i],
                str(list_count_res[i]),
                fill=sprites.list_color[i],
                font=sprites.font,
            )

    def draw_gold(self, bg, list_count_res):
        for i in range(4):
            ImageDraw.Draw(bg).text(
                self.list_coords_gold[i],
                str(list_count_res[i]),
                fill=sprites.list_color[i],
                font=sprites.font,
            )

    def draw_food(self, bg, list_count_res):
        for i in range(4):
            ImageDraw.Draw(bg).text(
                self.list_coords_food[i],
                str(list_count_res[i]),
                fill=sprites.list_color[i],
                font=sprites.font,
            )

    def draw_dice_tool_maker(self, bg, dice=0, tool_used=0):
        ImageDraw.Draw(bg).text(
            self.list_coords_dice_tool_maker,
            f"Dice:{int(dice)}\nTool used:{int(tool_used)}",
            fill="white",
            font=sprites.font2,
        )

    def draw_card_choose_dice(self, bg, list_card_choose_dice=[0, 0, 0, 0]):
        dices = [0, 0, 0, 0]
        for i in range(4):
            dice_ = np.where(list_card_choose_dice[i] == 1)[0]
            if len(dice_) != 0:
                dices[i] = dice_[0] + 1
        ImageDraw.Draw(bg).text(
            self.list_coords_card_choose_dice,
            f"Card choose dice: \n {dices}",
            fill="white",
            font=sprites.font2,
        )


_d_ = BG_SIZE[0] * 0.04
params = Params()
sprites = Sprites()
_agent_ = Draw_Agent()


def draw_cards(
    bg,
    list_card_build_on_board,
    list_card_civ_on_board,
    list_count_building=[7, 7, 7, 7],
    count_civ=30,
):
    x = int(BG_SIZE[0] * 0.35)
    y = int(BG_SIZE[1] * 0.83)
    for i in range(len(list_card_build_on_board)):
        bg.paste(sprites.cards_building[list_card_build_on_board[i]], (x, y))
        ImageDraw.Draw(bg).text(
            (x + x * 0.1, y - y * 0.08),
            str(list_count_building[i]),
            fill="white",
            font=sprites.font,
        )
        x -= int(BG_SIZE[0] * 0.112)

    #  ImageDraw.Draw(bg).text((int(BG_SIZE[0]*0.73), int(BG_SIZE[1]*0.63)), str(count_civ), fill= 'black', font = sprites.font)

    x = int(BG_SIZE[0] * 0.87)
    y = int(BG_SIZE[1] * 0.735)
    for i in range(len(list_card_civ_on_board)):
        bg.paste(sprites.card_civilization[list_card_civ_on_board[i]], (x, y))
        x -= int(BG_SIZE[0] * 0.126)


def draw_state_card(state, background):
    list_state_card_civ = state[14:110].reshape(4, 24)
    list_state_card_building = state[110:142].reshape(4, 8)

    list_card_build_on_board = []
    list_card_civ_on_board = []
    for i in range(4):
        for j in range(len(BUILDING_CARDS)):
            if (list_state_card_building[i] == BUILDING_CARDS[j]).all():
                list_card_build_on_board.append(j)
                break
        for j in range(len(CIV_CARDS)):
            if (list_state_card_civ[i] == CIV_CARDS[j]).all():
                list_card_civ_on_board.append(j)
                break

    list_count_building = state[4:8]
    draw_cards(
        background,
        list_card_build_on_board,
        list_card_civ_on_board,
        list_count_building,
        30,
    )


def draw_specifications_agent(im, state):
    all_agent_state = state[142:318].reshape(4, 44)
    list_score = all_agent_state[:, 0]
    list_field = all_agent_state[:, 1]
    list_people = all_agent_state[:, 2]
    list_food = all_agent_state[:, 3]
    list_building = all_agent_state[:, 4]
    res_array = all_agent_state[:, 5:9]
    tool_array = all_agent_state[:, 9:12]
    tool_temp_array = all_agent_state[:, 12:15]
    state_tool_array = all_agent_state[:, 15:18]
    people_x_array = all_agent_state[:, 39:43]
    all_type_civ = all_agent_state[:, 22:30]
    type_civ = [len(np.where(all_agent_state[:, 22:30][i])[0]) for i in range(4)]
    #  print(all_type_civ)
    _agent_.draw_agent_block(
        im,
        res_array=res_array,
        tool_array=tool_array,
        tool_temp_array=tool_temp_array,
        state_tool_array=state_tool_array,
        people_x_array=people_x_array,
        all_type_civ=all_type_civ,
        score=list_score,
        field=list_field,
        peoples=list_people,
        food=list_food,
        building=list_building,
        type_civ=type_civ,
    )


def draw_res(background, state):
    all_agent_state = state[142:318].reshape(4, 44)
    all_res = all_agent_state[:, 34:39].T
    params.draw_forest(background, all_res[0])
    params.draw_rock(background, all_res[1])
    params.draw_silver(background, all_res[2])
    params.draw_gold(background, all_res[3])
    params.draw_food(background, all_res[4])
    params.draw_dice_tool_maker(background, state[411], state[386])
    params.draw_card_choose_dice(background, state[387:411].reshape(4, 6))


def draw_point(
    background,
    colors=["white", "white", "white"],
    color_building=["white", "white", "white", "white"],
    color_civ=["white", "white", "white", "white"],
):
    params.draw_field(background, colors[0])
    params.draw_tool_maker(background, colors[1])
    params.draw_hut(background, colors[2])
    params.draw_building(background, color_building)
    params.draw_civ(background, color_civ)


def make_point(background, state):
    all_agent_state = state[142:318].reshape(4, 44)
    all_agent_in_point = all_agent_state[:, 31:39]

    colors = ["white", "white", "white"]
    for i in range(3):
        for agent in range(4):
            if all_agent_in_point[agent][i] > 0:
                colors[i] = sprites.list_color[agent]
                break

    color_building = ["white", "white", "white", "white"]
    color_civ = ["white", "white", "white", "white"]
    all_agent_state_card = state[322:354].reshape(4, 8)
    for i in range(8):
        for agent in range(4):
            if all_agent_state_card[agent][i] > 0:
                if i < 4:
                    color_civ[i] = sprites.list_color[agent]
                else:
                    color_building[i - 4] = sprites.list_color[agent]
                break

    draw_point(background, colors, color_building, color_civ)


def get_state_image(state=None):
    state = state.astype(np.int64)
    background = sprites.background.copy()
    im = sprites.im.copy()
    _agent_.draw_line(im)

    if state is None:
        return background

    draw_state_card(state, background)
    make_point(background, state)
    draw_res(background, state)

    im.paste(
        background,
        (int((SIZE_BOARD[0] - BG_SIZE[0]) / 2), int((SIZE_BOARD[1] - BG_SIZE[1]) / 2)),
    )
    draw_specifications_agent(im, state)
    return im


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    if not action is None:
        (
            env_components.env,
            env_components.all_build_card,
            env_components.all_civ_card,
        ) = _env.stepEnv(
            action,
            env_components.env,
            env_components.all_build_card,
            env_components.all_civ_card,
        )

    if env_components.winner[0] == -1:
        while env_components.cc <= 1000:
            idx = np.where(env_components.env[0:4] == 1)[0][0]
            if env_components.list_other[idx] == -1:
                break

            state = _env.getAgentState(env_components.env)
            agent = list_agent[env_components.list_other[idx] - 1]
            data = list_data[env_components.list_other[idx] - 1]
            action, data = agent(state, data)
            (
                env_components.env,
                env_components.all_build_card,
                env_components.all_civ_card,
            ) = _env.stepEnv(
                action,
                env_components.env,
                env_components.all_build_card,
                env_components.all_civ_card,
            )

            env_components.winner = _env.checkEnded(env_components.env)
            if env_components.winner[0] != -1:
                break
            env_components.cc += 1

    if env_components.winner[0] == -1:
        state = _env.getAgentState(env_components.env)
        win = -1
    else:
        env = env_components.env.copy()

        env[82] = 1
        my_idx = np.where(env_components.list_other == -1)[0][0]

        env[83] = my_idx
        env[0:4] = 0
        env[my_idx] = 1

        state = _env.getAgentState(env)
        if my_idx in env_components.winner:
            win = 1
        else:
            win = 0

        #  Chạy turn cuối cho 3 bot hệ thống
        for p_idx in range(4):
            if p_idx != my_idx:
                env[83] = p_idx
                env[0:4] = 0
                env[p_idx] = 1
                _state = _env.getAgentState(env)
                agent = list_agent[env_components.list_other[p_idx] - 1]
                data = list_data[env_components.list_other[p_idx] - 1]
                action, data = agent(_state, data)

    return win, state, env_components


# Sửa số người đã đặt/tổng số người
# Thêm chỉ dẫn người chơi chính ok
# Thêm thẻ công cụ dùng 1 lần ok
# Thêm thông tin xúc xắc ok
# Sửa số nền văn minh khác nhau ok
# Thêm thông tin thẻ chọn giá trị xúc xắc ok
