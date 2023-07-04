import numpy as np

NUMBER_PLAYER = 4
NUMBER_PHASE = 7
ATTRIBUTE_PLAYER = 20  # (Coin, số lượng mỗi loại thẻ (12 thẻ thường, 3 thẻ special, 4 thẻ target (22-16-10-4)))
NUMBER_TYPE_NORMAL_CARD = 12
NUMBER_TYPE_SPECIAL_CARD = 3
NUMBER_TARGET_CARD = 4
NUMBER_PER_NORMAL_CARD = 6


INDEX = 0
# thông tin 4 người chơi
ENV_PLAYER_IN4 = INDEX
INDEX += NUMBER_PLAYER * ATTRIBUTE_PLAYER

# số lượng mỗi loại thẻ trên bàn chơi
ENV_NORMAL_CARD = INDEX
INDEX += NUMBER_TYPE_NORMAL_CARD

# thẻ người chơi đã mua trong turn
ENV_CARD_BUY_IN_TURN = INDEX
INDEX += NUMBER_TYPE_NORMAL_CARD

# các thông tin khác
#  ENV_OTHER_IN4 = INDEX

# thẻ người chơi đổi đi
ENV_CARD_SELL = INDEX
INDEX += 1

# người chơi được đổ xúc sắc tiếp hay không
ENV_PLAYER_CONTINUE = INDEX
INDEX += 1

# Giá trị xúc sắc gần nhất
ENV_LAST_DICE = INDEX
INDEX += 1

# Người chơi bị chọn
ENV_PICKED_PLAYER = INDEX
INDEX += 1

# Người chơi hành động
ENV_ID_ACTION = INDEX
INDEX += 1

# Phase
ENV_PHASE = INDEX
INDEX += 1

# Check_end_game
ENV_CHECK_END = INDEX
INDEX += 1

ENV_LENGTH = INDEX


P_INDEX = 0

# thông tin 4 người chơi
P_PLAYER_IN4 = P_INDEX
P_INDEX += NUMBER_PLAYER * ATTRIBUTE_PLAYER

# số lượng mỗi loại thẻ trên bàn chơi
P_NORMAL_CARD = P_INDEX
P_INDEX += NUMBER_TYPE_NORMAL_CARD

# thẻ người chơi đã mua trong turn
P_CARD_BUY_IN_TURN = P_INDEX
P_INDEX += NUMBER_TYPE_NORMAL_CARD

# các thông tin khác
# thẻ người chơi đổi đi
P_CARD_SELL = P_INDEX
P_INDEX += NUMBER_TYPE_NORMAL_CARD

# người chơi được đổ xúc sắc tiếp hay không
P_PLAYER_CONTINUE = P_INDEX
P_INDEX += 1

# Giá trị xúc sắc gần nhất
P_LAST_DICE = P_INDEX
P_INDEX += 1

# Người chơi bị chọn
P_PICKED_PLAYER = P_INDEX
P_INDEX += NUMBER_PLAYER


# Phase
P_PHASE = P_INDEX
P_INDEX += NUMBER_PHASE

# Check end
P_CHECK_END = P_INDEX
P_INDEX += 1

P_LENGTH = P_INDEX


ALL_CARD_FEE = np.array([1, 1, 1, 2, 2, 3, 5, 3, 6, 3, 3, 2, 6, 7, 8, 22, 16, 10, 4])


ACTIONS_MEAN = np.array(
    [
        "Không đổ lại xúc sắc",
        "Đổ 1 xúc sắc",
        "Đổ 2 xúc sắc",
        "Lấy tiền người đầu tiên sau mình",
        "Lấy tiền người thứ 2 sau mình",
        "Lấy tiền người thứ 3 sau mình",
        "Đối thẻ người đầu tiên sau mình",
        "Đổi thẻ người thứ 2 sau mình",
        "Đổi thẻ người thứ 3 sau mình",
        "Không đổi thẻ",
        "Đổi thẻ lúa mì",
        "Đổi thẻ nông trại",
        "Đổi thẻ tiệm bánh",
        "Đối thẻ quán cà phê",
        "Đổi thẻ cửa hàng tiện lợi",
        "Đổi thẻ rừng",
        "Đổi thẻ nhà máy pho mát",
        "Đổi thẻ nhà máy nội thất",
        "Đổi thẻ mỏ quặng",
        "Đổi thẻ quán ăn gia đình",
        "Đổi thẻ vườn táo",
        "Đổi thẻ chợ trái cây",
        "Chọn lấy thẻ lúa mì",
        "Chọn lấy thẻ nông trại",
        "Chọn lấy thẻ tiệm bánh",
        "Chọn lấy quán cà phê",
        "Chọn lấy thẻ cửa hàng tiện lợi",
        "Chọn lấy thẻ rừng",
        "Chọn lấy thẻ nhà máy pho mát",
        "Chọn lấy thẻ nhà máy nội thất",
        "Chọn lấy thẻ mỏ quặng",
        "Chọn lấy thẻ quán ăn gia đình",
        "Chọn lấy thẻ vườn táo",
        "Chọn lấy thẻ chợ trái cây",
        "Mua thẻ lúa mì",
        "Mua thẻ nông trại",
        "Mua thẻ tiệm bánh",
        "Mua thẻ quán cà phê",
        "Mua thẻ cửa hàng tiện lợi",
        "Mua thẻ rừng",
        "Mua thẻ nhà máy pho mát",
        "Mua thẻ nhà máy nội thất",
        "Mua thẻ mỏ quặng",
        "Mua thẻ quán ăn gia đình",
        "Mua thẻ vườn táo",
        "Mua thẻ chợ trái cây",
        "Mua thẻ sân vận động",
        "Mua thẻ đài truyền hình",
        "Mua thẻ trung tâm thương mại",
        "Mua thẻ 22đ",
        "Mua thẻ 16đ",
        "Mua thẻ 10đ",
        "Mua thẻ 4đ",
        "Ko mua thẻ nữa",
    ]
)
