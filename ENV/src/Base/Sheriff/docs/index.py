import numpy as np
import pandas as pd
from numba import njit

ATTRIBUTE_PLAYER = 100
"""
    OWNER: 100
        0: player coin
        1: debt
        2: is_police
        [3 : 7]: type_in_bag
        8: coin_bride
        9: number_smuggle
        10: number_card_bride_bag
        -90:-75: card_done_bride
        -75:60: card_done
        -60:-45: card_in_bag_bride
        -45:-30: card_in_bag
        -30:-15: card_bag_drop
        -15: card_hand
"""
# (coin, debt, is_police, typy_bag(4), coin_bribe, number_smuggle_card, number_bribe_card_in_bag) => 10
# 15*6: thẻ hối lộ done, thẻ done, thẻ hối lộ trong túi, thẻ trong túi, thẻ bỏ đi, thẻ trên tay => 90
OTHER_ATTRIBUTE_PLAYER = 29
# (coin, debt, is_police, typy_bag(4), coin_bribe, number_smuggle_card, number_bribe_card_in_bag) => 10 vị trí
# thẻ done hối lộ: 15 vị trí, thẻ done chính ngạch của người chơi: 4 vị trí => 19 vị trí

NUMBER_TYPE_CARD = 15
CARD_OPEN_START = 5
MAX_CARD_TAKE = 6
NUMBER_PLAYER = 4
NUMBER_CARD = 216
NUMBER_CARD_USE = 186  # thẻ sau khi chia cho người chơi và bỏ ở 2 chồng bài
NUMBER_CARD_OPEN = 125
NUMBER_PHASE = 11

INDEX = 0
# Thông tin các người chơi
ENV_PLAYER_IN4 = INDEX
INDEX += ATTRIBUTE_PLAYER * NUMBER_PLAYER
# chồng bài úp
ENV_DOWN_CARD = INDEX
INDEX += NUMBER_CARD_USE
# chồng bài lật trái
ENV_LEFT_CARD = INDEX
INDEX += NUMBER_CARD_OPEN
# chồng bài lật phải
ENV_RIGHT_CARD = INDEX
INDEX += NUMBER_CARD_OPEN
# thẻ người chơi bỏ trong lượt
ENV_TEMP_DROP = INDEX
INDEX += NUMBER_TYPE_CARD * NUMBER_PLAYER
# Người chơi đang bị kiểm tra, chỉ sheriff nhìn
ENV_LAST_CHECKED = INDEX
INDEX += NUMBER_PLAYER
# người chơi đang hành động
ENV_ID_ACTION = INDEX
INDEX += 1
# Số lượt đã done
ENV_ROUND = INDEX
INDEX += 1
# số người đã check
ENV_NUMBER_CHECKED = INDEX
INDEX += 1
# check end game
ENV_CHECK_END = INDEX
INDEX += 1
# Phase
ENV_PHASE = INDEX
INDEX += 1

ENV_LENGTH = INDEX

P_INDEX = 0
# thông tin người chơi
P_PLAYER_IN4 = P_INDEX
P_COIN = 0
P_DEBT = 1
P_IS_POLICE = 2
P_TYPE_IN_BAG = 3
P_COIN_BRIBE = 7
P_NUMBER_SMUGGLE = 8
P_NUMBER_CARD_BRIBE_BAG = 9

P_INDEX += ATTRIBUTE_PLAYER
# thông tin các người chơi khác
P_OTHER_PLAYER_IN4 = P_INDEX
P_INDEX += OTHER_ATTRIBUTE_PLAYER * (NUMBER_PLAYER - 1)
# 6 thẻ đầu ở chồng thẻ trái
P_LEFT_CARD = P_INDEX
P_INDEX += NUMBER_TYPE_CARD * MAX_CARD_TAKE
# 6 thẻ đầu ở chồng thẻ phải
P_RIGHT_CARD = P_INDEX
P_INDEX += NUMBER_TYPE_CARD * MAX_CARD_TAKE
# thẻ các người chơi khác đã bỏ
P_CARD_OTHER_PLAYER_DROP = P_INDEX
P_INDEX += NUMBER_TYPE_CARD * (NUMBER_PLAYER - 1)
# Round
P_ROUND = P_INDEX
P_INDEX += 1
P_CHECK_END = P_INDEX
P_INDEX += 1
P_PHASE = P_INDEX
P_INDEX += NUMBER_PHASE

P_OTHER_PLAYER_DONE_CARD = P_INDEX
P_INDEX += NUMBER_TYPE_CARD * (NUMBER_PLAYER - 1)

P_ORDER = P_INDEX
P_INDEX += NUMBER_PLAYER

P_NUMBER_CARD_IN_BAG = P_INDEX
P_INDEX += NUMBER_PLAYER

P_LENGTH = P_INDEX


ALL_PENALTY = np.array([2, 2, 2, 2, 4, 4, 4, 4, 3, 4, 4, 4, 4, 5, 5])

ALL_ORDER = np.arange(NUMBER_PLAYER)

ALL_REWARD = np.array([2, 3, 3, 4, 6, 7, 8, 9, 4, 6, 6, 8, 6, 9, 9])

ALL_NUMBER_COUNT = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])

NORMAL_CARD = np.array(
    [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        8,
        8,
        8,
        8,
        8,
    ]
)

ROYAL_CARD = np.array([9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 15])

START_PLAYER = np.array(
    [
        50.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)


"""
Action	Mean
0	bỏ thẻ apple
1	bỏ thẻ cheese
2	bỏ thẻ bread
3	bỏ thẻ chicken
4	bỏ thẻ peper
5	bỏ thẻ mead
6	bỏ thẻ silk
7	bỏ thẻ crossbow
8	bỏ thẻ green_apple
9	bỏ thẻ gouda_cheese
10	bỏ thẻ rye_bread
11	bỏ thẻ royal_rooster
12	bỏ thẻ golden_apple
13	bỏ thẻ bleu_cheese
14	bỏ thẻ pump_bread
15	Không bỏ thẻ nữa
16	Lấy thẻ chồng bài rút
17	Lấy thẻ chồng bài lật trái
18	Lấy thẻ chồng bài lật phải
19	Trả thẻ vào chồng bài lật trái
20	Trả thẻ vào chồng bài lật phải
21	bỏ thẻ apple vào túi
22	bỏ thẻ cheese vào túi
23	bỏ thẻ bread vào túi
24	bỏ thẻ chicken vào túi
25	bỏ thẻ peper vào túi
26	bỏ thẻ mead vào túi
27	bỏ thẻ silk vào túi
28	bỏ thẻ crossbow vào túi
29	bỏ thẻ green_apple vào túi
30	bỏ thẻ gouda_cheese vào túi
31	bỏ thẻ rye_bread vào túi
32	bỏ thẻ royal_rooster vào túi
33	bỏ thẻ golden_apple vào túi
34	bỏ thẻ bleu_cheese vào túi
35	bỏ thẻ pump_bread vào túi
36	Không bỏ thẻ vào túi nữa
37	Khai báo hàng là apple
38	Khai báo hàng là cheese
39	Khai báo hàng là bread
40	Khai báo hàng là chicken
41	Kiểm tra người đầu tiên cạnh mình
42	Kiểm tra người thứ 2 cạnh mình
43	Kiểm tra người thứ 3 cạnh mình
44	Không hối lộ coin nữa
45	Hối lộ thêm 1 coin
46	hối lộ thẻ apple done
47	hối lộ thẻ cheese done
48	hối lộ thẻ bread done
49	hối lộ thẻ chicken done
50	hối lộ thẻ peper done
51	hối lộ thẻ mead done
52	hối lộ thẻ silk done
53	hối lộ thẻ crossbow done
54	hối lộ thẻ green_apple done
55	hối lộ thẻ gouda_cheese done
56	hối lộ thẻ rye_bread done
57	hối lộ thẻ royal_rooster done
58	hối lộ thẻ golden_apple done
59	hối lộ thẻ bleu_cheese done
60	hối lộ thẻ pump_bread done
61	Không hối lộ thẻ done nữa
62	hối lộ thẻ apple trong túi
63	hối lộ thẻ cheese trong túi
64	hối lộ thẻ bread trong túi
65	hối lộ thẻ chicken trong túi
66	hối lộ thẻ peper trong túi
67	hối lộ thẻ mead trong túi
68	hối lộ thẻ silk trong túi
69	hối lộ thẻ crossbow trong túi
70	hối lộ thẻ green_apple trong túi
71	hối lộ thẻ gouda_cheese trong túi
72	hối lộ thẻ rye_bread trong túi
73	hối lộ thẻ royal_rooster trong túi
74	hối lộ thẻ golden_apple trong túi
75	hối lộ thẻ bleu_cheese trong túi
76	hối lộ thẻ pump_bread trong túi
77	Không hối lộ thẻ trong túi nữa
78	Kiểm tra hàng
79	Cho qua
80	Bỏ thẻ tịch thu vào bên trái
81	Bỏ thẻ tịch thu vào bên phải
"""
