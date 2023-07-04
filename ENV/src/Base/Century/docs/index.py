import numpy as np

NUMBER_PLAYER = 5
NUMBER_PHASE = 5
ATTRIBUTE_POINT_CARD = 5
ATTRIBUTE_ACTION_CARD = 8
LENGTH_ACTION_CARD = 9  # (số lượng tài nguyên bỏ ra(4 vị trí), số lượng tài nguyên nhận(4 vị trí), số lần nâng cấp (1 vị trí))
ATTRIBUTE_PLAYER = 51  # (điểm, số thẻ đã mua, 4 vị trí thể hiện số lượng 4 loại token, 45 vị trí cho 45 loại thẻ hành động)
BASIC_ATTRIBUTE_PLAYER = 6  # (điểm, số thẻ đã mua, 4 loại token)
NUMBER_OPEN_ACTION_CARD = 6
NUMBER_OPEN_POINT_CARD = 5
NUMBBER_TYPE_TOKEN = 4
NUMBER_ACTION_CARD = 45
NUMBER_POINT_CARD = 36
NUMBER_UPGRADE_CARD = 2
NUMBER_BASIC_ACTION_CARD = 2
FREE_SCORE_GOLD = 30
FREE_SCORE_SILVER = 10

FIRST_ACTION_BUY_ACTION_CARD = 1
FIRST_ACTION_BUY_POINT_CARD = 7
FIRST_ACTION_USE_ACTION_CARD = 12
FIRST_ACTION_DROP_TOKEN = 57
FIRST_ACTION_UPGRADE_TOKEN = 62


INDEX = 0

# thông tin người chơi
ENV_IN4_PLAYER = INDEX
INDEX += ATTRIBUTE_PLAYER * NUMBER_PLAYER

# 6 thẻ hành động trên bàn
ENV_OPEN_ACTION_CARD = INDEX
INDEX += NUMBER_OPEN_ACTION_CARD * LENGTH_ACTION_CARD

# Nguyên liệu đặt trên mỗi thẻ hành động (chỉ xét 5 thẻ đầu)
ENV_TOKEN_ON_ACTION_CARD = INDEX
INDEX += NUMBBER_TYPE_TOKEN * (NUMBER_OPEN_ACTION_CARD - 1)

# thông tin 5 thẻ khu chợ (thẻ điểm ), mỗi thẻ có (điểm, số lượng 4 token cần để mua)
ENV_OPEN_POINT_CARD = INDEX
INDEX += NUMBER_OPEN_POINT_CARD * ATTRIBUTE_POINT_CARD

# chuỗi index các thẻ action đang úp
ENV_DOWN_ACTION_CARD = INDEX
INDEX += NUMBER_ACTION_CARD - NUMBER_BASIC_ACTION_CARD

# chuỗi index các thẻ khu chợ (thẻ điểm) đang úp
ENV_DOWN_POINT_CARD = INDEX
INDEX += NUMBER_POINT_CARD

# Other_in4
ENV_NUMBER_ACTION_UPGRADE = INDEX
INDEX += 1

ENV_CARD_BUY_OR_USE = INDEX  # card hand used
INDEX += 1

# số lượng token cần bỏ lại ngân hàng
ENV_TOKEN_NEED_DROP = INDEX
INDEX += 1

# số đồng bạc còn lại
ENV_SILVER_COIN = INDEX
INDEX += 1

# số đồng vàng còn lại
ENV_GOLD_COIN = INDEX
INDEX += 1

# action vừa thực hiện (khi mà dùng action_card)
ENV_LAST_ACTION = INDEX
INDEX += 1

# Phase game
ENV_PHASE = INDEX
INDEX += 1

# kiểm tra hết game
ENV_CHECK_END = INDEX
INDEX += 1

# người chơi hành động
ENV_ID_ACTION = INDEX
INDEX += 1

# Thứ tự người chơi hành động
ENV_ORDER_PLAYER = INDEX
INDEX += NUMBER_PLAYER

ENV_LENGTH = INDEX


P_INDEX = 0
# điểm người chơi
P_SCORE = P_INDEX
P_INDEX += 1

# số thẻ điểm của người chơi
P_PLAYER_NUMBER_POINT_CARD = P_INDEX
P_INDEX += 1

# Token của người chơi
P_TOKEN = P_INDEX
P_INDEX += NUMBBER_TYPE_TOKEN

# Action card của người chơi
P_ACTION_CARD_PLAYER = P_INDEX
P_INDEX += NUMBER_ACTION_CARD

# Action card down của người chơi
P_ACTION_CARD_DOWN_PLAYER = P_INDEX
P_INDEX += NUMBER_ACTION_CARD

# Thông tin các người chơi khác
P_OTHER_PLAYER_IN4 = P_INDEX
P_INDEX += (NUMBER_PLAYER - 1) * BASIC_ATTRIBUTE_PLAYER

# Thông tin các thẻ action trên bàn
P_OPEN_ACTION_CARD = P_INDEX
P_INDEX += NUMBER_OPEN_ACTION_CARD * LENGTH_ACTION_CARD

# Token đặt trên các thẻ action trên bàn
P_TOKEN_ON_ACTION_CARD = P_INDEX
P_INDEX += NUMBBER_TYPE_TOKEN * (NUMBER_OPEN_ACTION_CARD - 1)

# thông tin các thẻ điểm trên bàn
P_OPEN_POINT_CARD = P_INDEX
P_INDEX += NUMBER_OPEN_POINT_CARD * ATTRIBUTE_POINT_CARD

# Other in4
# số lượng đồng bạc còn lại
P_SILVER_COIN = P_INDEX
P_INDEX += 1

# số lượng đồng vàng còn lại
P_GOLD_COIN = P_INDEX
P_INDEX += 1

# action vừa thực hiện (dùng khi dùng action card)
P_LAST_ACTION = P_INDEX
P_INDEX += NUMBER_ACTION_CARD

# Kiểm tra dừng game chưa
P_CHECK_END = P_INDEX
P_INDEX += 1

# Phase
P_PHASE = P_INDEX
P_INDEX += NUMBER_PHASE

# Thứ tự bắt đầu game
P_ORDER = P_INDEX
P_INDEX += NUMBER_PLAYER

P_LENGTH = P_INDEX


ACTIONS_MEAN = np.array(
    [
        "Nghỉ ngơi",
        "mua thẻ thường 1",
        "mua thẻ thường 2",
        "mua thẻ thường 3",
        "mua thẻ thường 4",
        "mua thẻ thường 5",
        "mua thẻ thường 6",
        "mua thẻ điểm 1",
        "mua thẻ điểm 2",
        "mua thẻ điểm 3",
        "mua thẻ điểm 4",
        "mua thẻ điểm 5",
        "dùng thẻ 000020000",
        "dùng thẻ 000030000",
        "dùng thẻ 000040000",
        "dùng thẻ 000011000",
        "dùng thẻ 000000100",
        "dùng thẻ 000021000",
        "dùng thẻ 000002000",
        "dùng thẻ 000010100",
        "dùng thẻ 000000010",
        "dùng thẻ 200002000",
        "dùng thẻ 200000100",
        "dùng thẻ 300000010",
        "dùng thẻ 300003000",
        "dùng thẻ 300001100",
        "dùng thẻ 400000200",
        "dùng thẻ 400000110",
        "dùng thẻ 500000020",
        "dùng thẻ 500000300",
        "dùng thẻ 010030000",
        "dùng thẻ 020000200",
        "dùng thẻ 020030100",
        "dùng thẻ 020020010",
        "dùng thẻ 030000300",
        "dùng thẻ 030000020",
        "dùng thẻ 030010110",
        "dùng thẻ 030020200",
        "dùng thẻ 001041000",
        "dùng thẻ 001012000",
        "dùng thẻ 001002000",
        "dùng thẻ 002021010",
        "dùng thẻ 002000020",
        "dùng thẻ 002023000",
        "dùng thẻ 002002010",
        "dùng thẻ 003000030",
        "dùng thẻ 000100200",
        "dùng thẻ 000130100",
        "dùng thẻ 000103000",
        "dùng thẻ 000122000",
        "dùng thẻ 000111100",
        "dùng thẻ 000211300",
        "dùng thẻ 000203200",
        "dùng thẻ 110000010",
        "dùng thẻ 201000020",
        "dùng thẻ 000000002",
        "dùng thẻ 000000003",
        "bỏ token vàng",
        "bỏ token đỏ",
        "bỏ token xanh",
        "bỏ token nâu",
        "không dùng tiếp thẻ action",
        "nâng cấp vàng",
        "nâng cấp đỏ",
        "nâng cấp xanh",
    ]
)


# ALL_CARD_POINT_IN4 = np.array([[0, 0, 0, 5, 20], [0, 0, 2, 3, 18], [0, 0, 3, 2, 17], [0, 0, 0, 4, 16], [0, 2, 0, 3, 16], [0, 0, 5, 0, 15], [0, 0, 2, 2, 14],
#                     [0, 3, 0, 2, 14], [2, 0, 0, 3, 14], [0, 2, 3, 0, 13], [0, 0, 4, 0, 12], [0, 2, 0, 2, 12], [0, 3, 2, 0, 12], [2, 2, 0, 0, 6], [3, 2, 0, 0, 7],
#                     [2, 3, 0, 0, 8], [2, 0, 2, 0, 8], [0, 4, 0, 0, 8], [3, 0, 2, 0, 9], [2, 0, 0, 2, 10], [0, 5, 0, 0, 10], [0, 2, 2, 0, 10], [2, 0, 3, 0, 11],
#                     [3, 0, 0, 2, 11], [1, 1, 1, 3, 20], [0, 2, 2, 2, 19], [1, 1, 3, 1, 18], [2, 0, 2, 2, 17], [1, 3, 1, 1, 16], [2, 2, 0, 2, 15], [3, 1, 1, 1, 14],
#                     [2, 2, 2, 0, 13], [0, 2, 1, 1, 12], [1, 0, 2, 1, 12], [1, 1, 1, 1, 12], [2, 1, 0, 1, 9]])

# ALL_CARD_IN4 = np.array([[0, 0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 3, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 2, 1, 0, 0], [0, 0, 0, 0, 0, 2, 0, 0],
#                         [0, 0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 2, 0, 0], [2, 0, 0, 0, 0, 0, 1, 0], [3, 0, 0, 0, 0, 0, 0, 1], [3, 0, 0, 0, 0, 3, 0, 0], [3, 0, 0, 0, 0, 1, 1, 0], [4, 0, 0, 0, 0, 0, 2, 0],
#                         [4, 0, 0, 0, 0, 0, 1, 1], [5, 0, 0, 0, 0, 0, 0, 2], [5, 0, 0, 0, 0, 0, 3, 0], [0, 1, 0, 0, 3, 0, 0, 0], [0, 2, 0, 0, 0, 0, 2, 0], [0, 2, 0, 0, 3, 0, 1, 0], [0, 2, 0, 0, 2, 0, 0, 1], [0, 3, 0, 0, 0, 0, 3, 0],
#                         [0, 3, 0, 0, 0, 0, 0, 2], [0, 3, 0, 0, 1, 0, 1, 1], [0, 3, 0, 0, 2, 0, 2, 0], [0, 0, 1, 0, 4, 1, 0, 0], [0, 0, 1, 0, 1, 2, 0, 0], [0, 0, 1, 0, 0, 2, 0, 0], [0, 0, 2, 0, 2, 1, 0, 1], [0, 0, 2, 0, 0, 0, 0, 2],
#                         [0, 0, 2, 0, 2, 3, 0, 0], [0, 0, 2, 0, 0, 2, 0, 1], [0, 0, 3, 0, 0, 0, 0, 3], [0, 0, 0, 1, 0, 0, 2, 0], [0, 0, 0, 1, 3, 0, 1, 0], [0, 0, 0, 1, 0, 3, 0, 0], [0, 0, 0, 1, 2, 2, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0],
#                         [0, 0, 0, 2, 1, 1, 3, 0], [0, 0, 0, 2, 0, 3, 2, 0], [1, 1, 0, 0, 0, 0, 0, 1], [2, 0, 1, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])

ALL_CARD_POINT_IN4 = np.array(
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


ALL_CARD_IN4 = np.array(
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
