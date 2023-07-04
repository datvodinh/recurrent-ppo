import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.Base.Sheriff import env as _env
from src.Base.Sheriff.docs import index
from env import SHORT_PATH

IMG_PATH = SHORT_PATH + "src/Base/Sheriff/images/"
import os


class Sprites:
    def __init__(self) -> None:
        self.background = Image.open(IMG_PATH + "background.png").resize((2100, 900))
        self._background_ = self.background.copy()
        self.sheriff = (
            Image.open(IMG_PATH + "sheriff.png").resize((100, 130)).convert("RGBA")
        )
        self.cards = {}
        card_names = os.listdir(IMG_PATH)
        for name in card_names:
            if name != "background.png":
                self.cards[name.split(".png")[0]] = (
                    Image.open(IMG_PATH + name).resize((100, 130)).convert("RGBA")
                )


sprites = Sprites()


class Params:
    def __init__(self) -> None:
        self.card_order = [
            "apples",
            "cheese",
            "bread",
            "chicken",
            "pepper",
            "mead",
            "silk",
            "crossbow",
            "2-apples",
            "2-cheese",
            "2-bread",
            "2-chicken",
            "3-apples",
            "3-cheese",
            "3-bread",
        ]
        self.font28 = ImageFont.FreeTypeFont("src/ImageFonts/arial.ttf", 28)


params = Params()


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

    # Draw
    draw = ImageDraw.ImageDraw(bg)

    # Số coin đang có, số coin nợ, ai là sheriff
    id_sheriff = np.where(state[[2, 102, 131, 160]] == 1)[0]
    if len(id_sheriff) == 1:
        id_sheriff = int(id_sheriff[0])
    else:
        id_sheriff = -1

    if id_sheriff != -1:
        bg.paste(sprites.sheriff, (415 + 500 * id_sheriff, 210), sprites.sheriff)

    s_0 = 380
    for p in range(4):
        s_ = 85 + 500 * p
        if p == 0:
            text = "coin: " + str(int(state[0]))
            text_1 = "debt: " + str(int(state[1]))
            text_2 = str(int(state[7]))
            text_3 = str(int(state[8]))
            text_4 = str(int(state[9]))
        else:
            text = "coin: " + str(int(state[100 + 29 * (p - 1)]))
            text_1 = "debt: " + str(int(state[101 + 29 * (p - 1)]))
            text_2 = str(int(state[107 + 29 * (p - 1)]))
            text_3 = str(int(state[108 + 29 * (p - 1)]))
            text_4 = str(int(state[109 + 29 * (p - 1)]))

        draw_outlined_text(draw, text, params.font28, (s_, 20), (0, 0, 0), 1)
        bbox = draw.textbbox((0, 0), text_1, params.font28)
        draw_outlined_text(
            draw, text_1, params.font28, (s_ + 430 - bbox[2], 20), (0, 0, 0), 1
        )
        bbox = draw.textbbox((0, 0), text_2, params.font28)
        draw_outlined_text(
            draw, text_2, params.font28, (s_ + 317 - bbox[2], 210), (0, 0, 0), 1
        )
        bbox = draw.textbbox((0, 0), text_3, params.font28)
        draw_outlined_text(
            draw, text_3, params.font28, (s_ + 317 - bbox[2], 260), (0, 0, 0), 1
        )
        bbox = draw.textbbox((0, 0), text_4, params.font28)
        draw_outlined_text(
            draw, text_4, params.font28, (s_ + 317 - bbox[2], 310), (0, 0, 0), 1
        )

        if p == 0:
            temp = state[3:7].astype(int)
        else:
            temp = state[103 + 29 * (p - 1) : 107 + 29 * (p - 1)].astype(int)

        for i in range(4):
            text = str(temp[i] * int(state[474 + p]))
            draw_outlined_text(
                draw, text, params.font28, (s_ + 3 + 110 * i, 70), (255, 255, 255), 1
            )

        if p != 0:
            temp = state[110 + 29 * (p - 1) : 125 + 29 * (p - 1)].astype(int)
            for i in range(15):
                text = str(temp[i])
                bbox = draw.textbbox((0, 0), text, params.font28)
                draw_outlined_text(
                    draw,
                    text,
                    params.font28,
                    (s_0 + 110 * i + 55 - bbox[2], 600 + 35 * (p - 1)),
                    (255, 255, 255),
                    1,
                )

            temp = state[410 + 15 * p : 425 + 15 * p].astype(int)
            for i in range(15):
                text = str(temp[i])
                draw_outlined_text(
                    draw,
                    text,
                    params.font28,
                    (s_0 + 110 * i - 15, 600 + 35 * (p - 1)),
                    (255, 255, 255),
                    1,
                )

            temp = state[125 + 29 * (p - 1) : 129 + 29 * (p - 1)].astype(int)
            for i in range(4):
                text = str(temp[i])
                draw_outlined_text(
                    draw,
                    text,
                    params.font28,
                    (s_ + 3 + 110 * i, 170),
                    (255, 255, 255),
                    1,
                )

    for k in range(3):
        inf_1 = state[10 + 30 * k : 25 + 30 * k].astype(int)
        inf_0 = state[25 + 30 * k : 40 + 30 * k].astype(int)
        for i in range(15):
            draw_outlined_text(
                draw,
                str(inf_0[i]),
                params.font28,
                (s_0 + 110 * i - 15, 775 + 35 * k),
                (255, 255, 255),
                1,
            )
            text = str(inf_1[i])
            bbox = draw.textbbox((0, 0), text, params.font28)
            draw_outlined_text(
                draw,
                text,
                params.font28,
                (s_0 + 110 * i + 55 - bbox[2], 775 + 35 * k),
                (255, 255, 255),
                1,
            )

    temp = state[367:382].astype(int)
    for i in range(15):
        text = str(temp[i])
        draw_outlined_text(
            draw, text, params.font28, (s_0 + 110 * i - 30, 702), (255, 255, 255), 1
        )

    temp = state[382:397].astype(int)
    for i in range(15):
        text = str(temp[i])
        bbox = draw.textbbox((0, 0), text, params.font28)
        draw_outlined_text(
            draw,
            text,
            params.font28,
            (s_0 + 110 * i + 50 - bbox[2] // 2 - 30, 710),
            (255, 255, 255),
            1,
        )

    temp = state[397:412].astype(int)
    for i in range(15):
        text = str(temp[i])
        bbox = draw.textbbox((0, 0), text, params.font28)
        draw_outlined_text(
            draw,
            text,
            params.font28,
            (s_0 + 110 * i + 100 - bbox[2] - 30, 718),
            (255, 255, 255),
            1,
        )

    left = []
    right = []
    for i in range(6):
        try:
            a = np.where(state[187 + 15 * i : 202 + 15 * i] == 1)[0][0]
            left.append(params.card_order[a])
        except:
            pass

        try:
            a = np.where(state[277 + 15 * i : 292 + 15 * i] == 1)[0][0]
            right.append(params.card_order[a])
        except:
            pass

    w_ = len(left) * 110 - 10
    s_ = 600 - w_ // 2
    for i in range(len(left)):
        bg.paste(sprites.cards[left[i]], (s_ + 110 * i, 400))

    w_ = len(right) * 110 - 10
    s_ = 1500 - w_ // 2
    for i in range(len(right)):
        bg.paste(sprites.cards[right[i]], (s_ + 110 * i, 400))

    phase = np.where(state[414:425] == 1)[0][0]
    text = str(phase)
    bbox = draw.textbbox((0, 0), text, params.font28)
    draw_outlined_text(
        draw, text, params.font28, (1050 - bbox[2] // 2, 450), (0, 0, 0), 1
    )
    return bg


action_annotations = [
    "bỏ thẻ apple",
    "bỏ thẻ cheese",
    "bỏ thẻ bread",
    "bỏ thẻ chicken",
    "bỏ thẻ peper",
    "bỏ thẻ mead",
    "bỏ thẻ silk",
    "bỏ thẻ crossbow",
    "bỏ thẻ green_apple",
    "bỏ thẻ gouda_cheese",
    "bỏ thẻ rye_bread",
    "bỏ thẻ royal_rooster",
    "bỏ thẻ golden_apple",
    "bỏ thẻ bleu_cheese",
    "bỏ thẻ pump_bread",
    "Không bỏ thẻ nữa",
    "Lấy thẻ chồng bài rút",
    "Lấy thẻ chồng bài lật trái",
    "Lấy thẻ chồng bài lật phải",
    "Trả thẻ vào chồng bài lật trái",
    "Trả thẻ vào chồng bài lật phải",
    "bỏ thẻ apple vào túi",
    "bỏ thẻ cheese vào túi",
    "bỏ thẻ bread vào túi",
    "bỏ thẻ chicken vào túi",
    "bỏ thẻ peper vào túi",
    "bỏ thẻ mead vào túi",
    "bỏ thẻ silk vào túi",
    "bỏ thẻ crossbow vào túi",
    "bỏ thẻ green_apple vào túi",
    "bỏ thẻ gouda_cheese vào túi",
    "bỏ thẻ rye_bread vào túi",
    "bỏ thẻ royal_rooster vào túi",
    "bỏ thẻ golden_apple vào túi",
    "bỏ thẻ bleu_cheese vào túi",
    "bỏ thẻ pump_bread vào túi",
    "Không bỏ thẻ vào túi nữa",
    "Khai báo hàng là apple",
    "Khai báo hàng là cheese",
    "Khai báo hàng là bread",
    "Khai báo hàng là chicken",
    "Kiểm tra người đầu tiên cạnh mình",
    "Kiểm tra người thứ 2 cạnh mình",
    "Kiểm tra người thứ 3 cạnh mình",
    "Không hối lộ coin nữa",
    "Hối lộ thêm 1 coin",
    "hối lộ thẻ apple done",
    "hối lộ thẻ cheese done",
    "hối lộ thẻ bread done",
    "hối lộ thẻ chicken done",
    "hối lộ thẻ peper done",
    "hối lộ thẻ mead done",
    "hối lộ thẻ silk done",
    "hối lộ thẻ crossbow done",
    "hối lộ thẻ green_apple done",
    "hối lộ thẻ gouda_cheese done",
    "hối lộ thẻ rye_bread done",
    "hối lộ thẻ royal_rooster done",
    "hối lộ thẻ golden_apple done",
    "hối lộ thẻ bleu_cheese done",
    "hối lộ thẻ pump_bread done",
    "Không hối lộ thẻ done nữa",
    "hối lộ thẻ apple trong túi",
    "hối lộ thẻ cheese trong túi",
    "hối lộ thẻ bread trong túi",
    "hối lộ thẻ chicken trong túi",
    "hối lộ thẻ peper trong túi",
    "hối lộ thẻ mead trong túi",
    "hối lộ thẻ silk trong túi",
    "hối lộ thẻ crossbow trong túi",
    "hối lộ thẻ green_apple trong túi",
    "hối lộ thẻ gouda_cheese trong túi",
    "hối lộ thẻ rye_bread trong túi",
    "hối lộ thẻ royal_rooster trong túi",
    "hối lộ thẻ golden_apple trong túi",
    "hối lộ thẻ bleu_cheese trong túi",
    "hối lộ thẻ pump_bread trong túi",
    "Không hối lộ thẻ trong túi nữa",
    "Kiểm tra hàng",
    "Cho qua",
    "Bỏ thẻ tịch thu vào bên trái",
    "Bỏ thẻ tịch thu vào bên phải",
]


def get_description(action):
    if action < 0 or action >= _env.getActionSize():
        return ""

    return action_annotations[action]


class Env_components:
    def __init__(self, env, count_turn, list_other) -> None:
        self.env = env
        self.count_turn = count_turn
        self.list_other = list_other


def get_env_components():
    env = _env.initEnv()
    count_turn = 0
    list_other = np.array([-1, 1, 2, 3])
    np.random.shuffle(list_other)
    return Env_components(env, count_turn, list_other)


def get_main_player_state(
    env_components: Env_components, list_agent, list_data, action=None
):
    if not action is None:
        env_components.env = _env.stepEnv(env_components.env, action)
        env_components.count_turn += 1

    while (
        _env.system_check_end(env_components.env) and env_components.count_turn < 1000
    ):
        p_idx = int(env_components.env[index.ENV_ID_ACTION])
        if env_components.list_other[p_idx] == -1:
            break

        state = _env.getAgentState(env_components.env)
        agent = list_agent[env_components.list_other[p_idx] - 1]
        data = list_data[env_components.list_other[p_idx] - 1]
        action, data = agent(state, data)
        env_components.env = _env.stepEnv(env_components.env, action)
        env_components.count_turn += 1

    if _env.system_check_end(env_components.env) and env_components.count_turn < 1000:
        state = _env.getAgentState(env_components.env)
        win = -1
    else:
        env = env_components.env.copy()
        env[index.ENV_PHASE] = 1
        env[index.ENV_CHECK_END] = 1
        my_idx = np.where(env_components.list_other == -1)[0][0]
        env[index.ENV_ID_ACTION] = my_idx
        state = _env.getAgentState(env)
        if _env.check_winner(env_components.env) == my_idx:
            win = 1
        else:
            win = 0

        for p_idx in range(4):
            if p_idx != my_idx:
                env[index.ENV_ID_ACTION] = p_idx
                _state = _env.getAgentState(env)
                agent = list_agent[env_components.list_other[p_idx] - 1]
                data = list_data[env_components.list_other[p_idx] - 1]
                action, data = agent(_state, data)

    return win, state, env_components
