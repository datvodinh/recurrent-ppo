import numpy as np
from numba import njit

#############################

__NORMAL_CARD__ = np.array(
    [
        [0, 0, 0, 1, 0, 0, 2, 2, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 2, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 2],
        [0, 0, 0, 1, 0, 0, 0, 3, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 1, 0, 0, 2, 1, 0, 2, 0],
        [0, 0, 1, 0, 0, 0, 2, 0, 2, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 2, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 1, 1, 3, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0],
        [1, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 2],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 3, 1],
        [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 2, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 3],
        [1, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 2],
        [0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 2],
        [0, 0, 0, 0, 1, 0, 3, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 2, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 3, 0, 2, 2],
        [2, 0, 0, 1, 0, 0, 0, 2, 0, 1, 4],
        [1, 0, 0, 1, 0, 0, 3, 0, 2, 0, 3],
        [2, 0, 0, 1, 0, 0, 0, 5, 3, 0, 0],
        [2, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0],
        [3, 0, 0, 1, 0, 0, 0, 0, 6, 0, 0],
        [3, 0, 1, 0, 0, 0, 0, 6, 0, 0, 0],
        [2, 0, 1, 0, 0, 0, 1, 0, 0, 4, 2],
        [2, 0, 1, 0, 0, 0, 0, 5, 0, 0, 0],
        [2, 0, 1, 0, 0, 0, 0, 3, 0, 0, 5],
        [1, 0, 1, 0, 0, 0, 0, 2, 3, 3, 0],
        [1, 0, 1, 0, 0, 0, 3, 2, 2, 0, 0],
        [3, 1, 0, 0, 0, 0, 6, 0, 0, 0, 0],
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 5, 3],
        [2, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0],
        [2, 1, 0, 0, 0, 0, 0, 4, 2, 0, 1],
        [1, 1, 0, 0, 0, 0, 2, 3, 0, 3, 0],
        [1, 1, 0, 0, 0, 0, 2, 0, 0, 3, 2],
        [3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 6],
        [2, 0, 0, 0, 0, 1, 5, 0, 0, 3, 0],
        [2, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 3, 3, 0, 0, 2],
        [1, 0, 0, 0, 0, 1, 2, 0, 3, 2, 0],
        [2, 0, 0, 0, 0, 1, 4, 0, 1, 2, 0],
        [1, 0, 0, 0, 1, 0, 0, 2, 2, 0, 3],
        [1, 0, 0, 0, 1, 0, 0, 0, 3, 2, 3],
        [2, 0, 0, 0, 1, 0, 2, 1, 4, 0, 0],
        [2, 0, 0, 0, 1, 0, 3, 0, 5, 0, 0],
        [2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 5],
        [3, 0, 0, 0, 1, 0, 0, 0, 0, 6, 0],
        [4, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0],
        [4, 0, 0, 1, 0, 0, 0, 6, 3, 0, 3],
        [5, 0, 0, 1, 0, 0, 0, 7, 3, 0, 0],
        [3, 0, 0, 1, 0, 0, 3, 3, 0, 3, 5],
        [3, 0, 1, 0, 0, 0, 3, 0, 3, 5, 3],
        [4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 7],
        [5, 0, 1, 0, 0, 0, 0, 3, 0, 0, 7],
        [4, 0, 1, 0, 0, 0, 0, 3, 0, 3, 6],
        [3, 1, 0, 0, 0, 0, 0, 5, 3, 3, 3],
        [4, 1, 0, 0, 0, 0, 0, 0, 7, 0, 0],
        [5, 1, 0, 0, 0, 0, 3, 0, 7, 0, 0],
        [4, 1, 0, 0, 0, 0, 3, 3, 6, 0, 0],
        [5, 0, 0, 0, 0, 1, 0, 0, 0, 7, 3],
        [3, 0, 0, 0, 0, 1, 5, 3, 3, 3, 0],
        [4, 0, 0, 0, 0, 1, 0, 0, 0, 7, 0],
        [4, 0, 0, 0, 0, 1, 3, 0, 0, 6, 3],
        [3, 0, 0, 0, 1, 0, 3, 3, 5, 0, 3],
        [5, 0, 0, 0, 1, 0, 7, 0, 0, 3, 0],
        [4, 0, 0, 0, 1, 0, 6, 0, 3, 3, 0],
        [4, 0, 0, 0, 1, 0, 7, 0, 0, 0, 0],
    ],
    dtype=np.int64,
)
__NOBLE_CARD__ = np.array(
    [
        [3, 0, 4, 4, 0, 0],
        [3, 3, 0, 3, 3, 0],
        [3, 3, 3, 3, 0, 0],
        [3, 3, 0, 0, 3, 3],
        [3, 0, 3, 0, 3, 3],
        [3, 4, 0, 4, 0, 0],
        [3, 4, 0, 0, 4, 0],
        [3, 0, 3, 3, 0, 3],
        [3, 0, 4, 0, 0, 4],
        [3, 0, 0, 0, 4, 4],
    ],
    dtype=np.int64,
)


__ENV_SIZE__ = 90


@njit()
def initEnv():
    lv1 = np.arange(41, dtype=np.int64)
    lv2 = np.arange(40, 71, dtype=np.int64)
    lv3 = np.arange(70, 91, dtype=np.int64)

    np.random.shuffle(lv1[:-1])
    np.random.shuffle(lv2[:-1])
    np.random.shuffle(lv3[:-1])

    lv1[-1] = 4
    lv2[-1] = 4
    lv3[-1] = 4

    env = np.full(__ENV_SIZE__, 0, dtype=np.int64)

    env[0:6] = np.array([7, 7, 7, 7, 7, 5])

    noble_ = np.arange(10)
    np.random.shuffle(noble_)
    env[6:11] = noble_[:5]

    env[11:15] = lv1[:4]
    env[15:19] = lv2[:4]
    env[19:23] = lv3[:4]

    #  23:38:53:68:83
    for pIdx in range(4):
        temp_ = 15 * pIdx
        #  env[23+temp_:35+temp_] = 0
        env[35 + temp_ : 38 + temp_] = -1

    #  env[83] == 0 #  Turn
    #  env[84:89] = 0 #  Dùng khi lấy nguyên liệu
    #  env[89] = 0 #  1 khi game kết thúc

    return env, lv1, lv2, lv3


__STATE_SIZE__ = 267


@njit()
def getAgentState(env, lv1, lv2, lv3):
    state = np.zeros(__STATE_SIZE__, dtype=np.float64)

    state[0:6] = env[0:6]

    #  6:12:18:24:30:36 #  Thẻ Noble
    for i in range(5):
        nobleId = env[6 + i]
        if nobleId != -1:
            temp_ = 6 * i
            state[6 + temp_ : 12 + temp_] = __NOBLE_CARD__[nobleId]

    #  36:47:58:69:80:91:102:113:124:135:146:157:168 #  Thẻ normal
    for i in range(12):
        cardId = env[11 + i]
        if cardId != -1:
            temp_ = 11 * i
            state[36 + temp_ : 47 + temp_] = __NORMAL_CARD__[cardId]

    pIdx = env[83] % 4
    for i in range(4):
        pEnvIdx = (pIdx + i) % 4
        temp1 = 12 * i
        temp2 = 15 * pEnvIdx

        #  201:213:225:237:249 #  Player infor
        state[201 + temp1 : 213 + temp1] = env[23 + temp2 : 35 + temp2]

        if i == 0:
            #  168:179:190:201 #  Thẻ úp
            for j in range(3):
                cardId = env[35 + temp2 + j]
                if cardId != -1:
                    temp_ = 11 * j
                    state[168 + temp_ : 179 + temp_] = __NORMAL_CARD__[cardId]

        else:
            #  249:252:255:258 #  Đếm cấp thẻ úp
            temp_ = 3 * (i - 1)
            for j in range(3):
                cardId = env[35 + temp2 + j]
                if cardId != -1:
                    if cardId < 40:
                        state[249 + temp_] += 1
                    elif cardId < 70:
                        state[250 + temp_] += 1
                    else:
                        state[251 + temp_] += 1

    #  [258:263] #  Nguyên liệu đã lấy
    state[258:263] = env[84:89]

    #  [263]
    state[263] = env[89]

    if lv1[-1] < 40:  #  Còn thẻ ẩn cấp 1
        state[264] = 1
    if lv2[-1] < 30:  #  Còn thẻ ẩn cấp 2
        state[265] = 1
    if lv3[-1] < 20:  #  Còn thẻ ẩn cấp 3
        state[266] = 1

    return state


@njit()
def checkBuyCard(gems, perGems, price):
    temp_ = gems[0:5] + perGems
    if np.sum((price > temp_) * (price - temp_)) <= gems[5]:
        return True

    return False


@njit()
def get_card_index(arr):
    for i in range(90):
        if (__NORMAL_CARD__[i] == arr).all():
            return i

    return -1


__ACTION_SIZE__ = 194


@njit()
def getValidActions(state):
    validActions = np.zeros(__ACTION_SIZE__, dtype=np.float64)
    boardStocks = state[0:6]

    takenStocks = state[258:263]
    if (takenStocks > 0).any():  #  Đang lấy nguyên liệu
        temp_ = np.where(boardStocks[0:5] > 0)[0]
        validActions[temp_] = 1

        s_ = np.sum(takenStocks)
        if s_ == 1:
            t_ = np.where(takenStocks == 1)[0]
            if t_.shape[0] > 0:
                t_ = t_[0]
                if boardStocks[t_] < 3:
                    validActions[t_] = 0
        else:
            t_ = np.where(takenStocks == 1)[0]
            validActions[t_] = 0

        return validActions

    if np.sum(state[201:207]) > 10:  #  Thừa nguyên liệu, cần trả nguyên liệu
        temp_ = np.where(state[201:206] > 0)[0] + 185
        validActions[temp_] = 1
        return validActions

    #  Lấy nguyên liệu
    temp_ = np.where(boardStocks[0:5] > 0)[0]
    validActions[temp_] = 1

    checkReserveCard = False
    for i in range(3):
        temp_ = 11 * i
        if (state[174 + temp_ : 179 + temp_] == 0).all():
            checkReserveCard = True
            break

    #  Các action mua thẻ (và úp thẻ)
    for i in range(15):
        temp_ = 11 * i
        cardPrice = state[42 + temp_ : 47 + temp_]
        if (cardPrice > 0).any():
            card_id = get_card_index(state[36 + temp_ : 47 + temp_])
            if card_id != -1:
                if checkReserveCard and i < 12:
                    validActions[95 + card_id] = 1

                if checkBuyCard(state[201:207], state[207:212], cardPrice):
                    validActions[5 + card_id] = 1

    #  Check úp thẻ ẩn
    if checkReserveCard:
        for i in range(3):
            if state[264 + i] == 1:
                validActions[190 + i] = 1

    #  Check nếu không có action nào có thể thực hiện (bị kẹt) thì cho action bỏ lượt
    if (validActions > 0).any():
        return validActions

    validActions[193] = 1
    return validActions


@njit()
def bot_lv0(state, perData):
    validActions = getValidActions(state)
    arr_action = np.where(validActions == 1)[0]
    idx = np.random.randint(0, arr_action.shape[0])
    return arr_action[idx], perData


@njit()
def checkEnded(env):
    scoreArr = env[np.array([34, 49, 64, 79])]
    maxScore = np.max(scoreArr)
    if maxScore >= 15 and env[83] % 4 == 0:
        maxScorePlayers = np.where(scoreArr == maxScore)[0]
        if len(maxScorePlayers) == 1:
            return maxScorePlayers
        else:
            playerBoughtCards = maxScorePlayers.copy()
            for i in range(maxScorePlayers.shape[0]):
                p_idx = maxScorePlayers[i]
                temp_ = 15 * p_idx
                playerBoughtCards[i] = np.sum(env[29 + temp_ : 34 + temp_])

            min_ = np.min(playerBoughtCards)
            winnerIdx = np.where(playerBoughtCards == min_)[0]
            #  if winnerIdx.shape[0] == 1:
            return maxScorePlayers[winnerIdx]
            #  else:
            #      return 4
    else:
        return np.array([-1])


@njit()
def openCard(env, lv1, lv2, lv3, cardId, posE):
    if cardId < 40:
        if lv1[-1] < 40:
            env[posE] = lv1[lv1[-1]]
            lv1[-1] += 1
        else:
            env[posE] = -1
    elif cardId < 70:
        if lv2[-1] < 30:
            env[posE] = lv2[lv2[-1]]
            lv2[-1] += 1
        else:
            env[posE] = -1
    else:
        if lv3[-1] < 20:
            env[posE] = lv3[lv3[-1]]
            lv3[-1] += 1
        else:
            env[posE] = -1


@njit()
def stepEnv(action, env, lv1, lv2, lv3):
    pIdx = env[83] % 4
    temp_ = 15 * pIdx
    pStocks = env[23 + temp_ : 29 + temp_]
    bStocks = env[0:6]
    pPerStocks = env[29 + temp_ : 34 + temp_]
    takenStocks = env[84:89]

    #  Lấy nguyên liệu
    if action < 5:
        takenStocks[action] += 1
        pStocks[action] += 1
        bStocks[action] -= 1

        check_ = False
        s_ = np.sum(takenStocks)
        if s_ == 1:
            if bStocks[action] < 3 and (np.sum(bStocks[0:5]) - bStocks[action]) == 0:
                check_ = True
        elif s_ == 2:
            if (
                np.max(takenStocks) == 2
                or (
                    np.sum(bStocks[0:5])
                    - np.sum(bStocks[np.where(takenStocks == 1)[0]])
                )
                == 0
            ):
                check_ = True
        else:
            check_ = True

        if check_:
            takenStocks[:] = 0

            #  Nếu không thừa nguyên liệu thì next turn
            if np.sum(pStocks) <= 10:
                env[83] += 1

    #  Trả nguyên liệu
    elif action >= 185 and action < 190:
        gem = action - 185
        pStocks[gem] -= 1
        bStocks[gem] += 1

        #  Nếu không thừa nguyên liệu thì next turn
        if np.sum(pStocks) <= 10:
            env[83] += 1

    #  Úp thẻ
    elif (action >= 95 and action < 185) or (action >= 190 and action < 193):
        temp_hideCard = 35 + temp_
        posP = (
            np.where(env[temp_hideCard : temp_hideCard + 3] == -1)[0][0] + temp_hideCard
        )

        if bStocks[5] > 0:
            pStocks[5] += 1
            bStocks[5] -= 1

        if action == 190:  #  Úp thẻ ẩn cấp 1
            env[posP] = lv1[lv1[-1]]
            lv1[-1] += 1
        elif action == 191:  #  Úp thẻ ẩn cấp 2
            env[posP] = lv2[lv2[-1]]
            lv2[-1] += 1
        elif action == 192:  #  Úp thẻ ẩn cấp 3
            env[posP] = lv3[lv3[-1]]
            lv3[-1] += 1
        else:  #  Úp thẻ trên bàn
            cardId = action - 95
            posE = np.where(env[11:23] == cardId)[0][0] + 11
            env[posP] = cardId

            #  Mở thẻ từ chồng úp lên trên bàn chơi
            openCard(env, lv1, lv2, lv3, cardId, posE)

        #  Nếu không thừa nguyên liệu thì next turn
        if np.sum(pStocks) <= 10:
            env[83] += 1

    #  Mua thẻ
    elif action >= 5 and action < 95:
        cardId = action - 5
        if cardId in env[11:23]:
            posE = np.where(env[11:23] == cardId)[0][0] + 11
            inboard = True
        else:
            posE = np.where(env[35 + temp_ : 38 + temp_] == cardId)[0][0] + 35 + temp_
            inboard = False

        cardIn4 = __NORMAL_CARD__[cardId]
        price = cardIn4[6:11]

        nlMat = (price > pPerStocks) * (price - pPerStocks)
        nlBt = np.minimum(nlMat, pStocks[0:5])
        nlG = np.sum(nlMat - nlBt)

        #  Trả nguyên liệu
        pStocks[0:5] -= nlBt  #  Trả nguyên liệu
        pStocks[5] -= nlG
        bStocks[0:5] += nlBt
        bStocks[5] += nlG

        #  Nhận các phần thưởng từ thẻ
        if inboard:
            openCard(env, lv1, lv2, lv3, cardId, posE)
        else:
            env[posE] = -1

        env[34 + temp_] += cardIn4[0]  #  Cộng điểm
        pPerStocks[:] += cardIn4[1:6]

        #  Next turn
        env[83] += 1

    #  40: Bỏ qua lượt (trường hợp đặc biệt khi không thể thực hiện action nào)
    else:
        env[83] += 1

    if (takenStocks == 0).all() and (pPerStocks >= 3).any():
        pos_nobles = np.full(5, 0)
        for i in range(5):
            nobleId = env[6 + i]
            if nobleId != -1:
                nobleIn4 = __NOBLE_CARD__[nobleId]
                price = nobleIn4[1:6]
                if (price <= pPerStocks).all():
                    pos_nobles[i] = 1

        if (pos_nobles == 1).any():
            arr_noble = np.where(pos_nobles == 1)[0]
            if arr_noble.shape[0] == 1:
                choose = 0
            else:
                choose = np.random.randint(0, arr_noble.shape[0])

            noble_idx = arr_noble[choose]
            env[6 + noble_idx] = -1
            env[34 + temp_] += 3


@njit()
def getAgentSize():
    return 4


@njit()
def getStateSize():
    return __STATE_SIZE__


@njit()
def getActionSize():
    return __ACTION_SIZE__


@njit()
def getReward(state):
    if state[263] == 0:
        return -1
    else:
        scoreArr = state[np.array([212, 224, 236, 248])]
        maxScore = np.max(scoreArr)
        if (
            scoreArr[0] < maxScore or scoreArr[0] < 15
        ):  #  Điểm của bản thân không cao nhất
            return 0
        else:
            maxScorePlayers = np.where(scoreArr == maxScore)[0]
            if (
                maxScorePlayers.shape[0] == 1
            ):  #  Bản thân là người duy nhất đạt điểm cao nhất
                return 1
            else:
                playerBoughtCards = maxScorePlayers.copy()
                for i in range(maxScorePlayers.shape[0]):
                    p_idx = maxScorePlayers[i]
                    temp_ = 12 * p_idx
                    playerBoughtCards[i] = np.sum(state[207 + temp_ : 212 + temp_])

                min_ = np.min(playerBoughtCards)
                if playerBoughtCards[0] > min_:  #  Số thẻ của bản thân nhiều hơn
                    return 0
                else:  #  Tất cả đều thắng
                    return 1


@njit()
def one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env, lv1, lv2, lv3 = initEnv()

    winner = np.array([-1])
    while env[83] < 400:
        p_idx = env[83] % 4
        state = getAgentState(env, lv1, lv2, lv3)
        if list_other[p_idx] == -1:
            action, per_player = p0(state, per_player)
            validActions = getValidActions(state)
            if validActions[action] != 1:
                raise Exception("Action không hợp lệ.")
        elif list_other[p_idx] == 1:
            action, per1 = p1(state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(state, per3)
        else:
            raise Exception("Sai list_other.")

        stepEnv(action, env, lv1, lv2, lv3)
        winner = checkEnded(env)
        if winner[0] != -1:
            break

    env[89] = 1
    p0_idx = np.where(list_other == -1)[0][0]
    for p_idx in range(4):
        env[83] = p_idx
        state = getAgentState(env, lv1, lv2, lv3)
        if list_other[p_idx] == -1:
            action, per_player = p0(state, per_player)
        elif list_other[p_idx] == 1:
            action, per1 = p1(state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(state, per3)
        else:
            raise Exception("Sai list_other.")

    if p0_idx in winner:
        result = 1
    else:
        result = 0
    return result, per_player


@njit()
def n_games_numba(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_numba(
            p0, list_other, per_player, per1, per2, per3, p1, p2, p3
        )
        win += winner

    return win, per_player


def one_game_normal(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env, lv1, lv2, lv3 = initEnv()

    winner = np.array([-1])
    while env[83] < 400:
        p_idx = env[83] % 4
        state = getAgentState(env, lv1, lv2, lv3)
        if list_other[p_idx] == -1:
            action, per_player = p0(state, per_player)
            validActions = getValidActions(state)
            if validActions[action] != 1:
                raise Exception("Action không hợp lệ.")
        elif list_other[p_idx] == 1:
            action, per1 = p1(state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(state, per3)
        else:
            raise Exception("Sai list_other.")

        stepEnv(action, env, lv1, lv2, lv3)
        winner = checkEnded(env)
        if winner[0] != -1:
            break

    env[89] = 1
    p0_idx = np.where(list_other == -1)[0][0]
    for p_idx in range(4):
        env[83] = p_idx
        state = getAgentState(env, lv1, lv2, lv3)
        if list_other[p_idx] == -1:
            action, per_player = p0(state, per_player)
        elif list_other[p_idx] == 1:
            action, per1 = p1(state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(state, per3)
        else:
            raise Exception("Sai list_other.")

    if p0_idx in winner:
        result = 1
    else:
        result = 0
    return result, per_player


def n_games_normal(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_normal(
            p0, list_other, per_player, per1, per2, per3, p1, p2, p3
        )
        win += winner

    return win, per_player


import importlib.util
import json, sys

try:
    from env import SHORT_PATH
except:
    pass


def load_module_player(player, game_name=None):
    if game_name == None:
        spec = importlib.util.spec_from_file_location(
            "Agent_player", f"{SHORT_PATH}src/Agent/{player}/Agent_player.py"
        )
    else:
        spec = importlib.util.spec_from_file_location(
            "Agent_player", f"{SHORT_PATH}src/Agent/ifelse/{game_name}/{player}.py"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_agent(level, *args):
    num_bot = getAgentSize() - 1

    if "_level_" not in globals():
        global _level_
        _level_ = level
        init = True
    else:
        if _level_ != level:
            _level_ = level
            init = True
        else:
            init = False

    if init:
        global _list_per_level_
        global _list_bot_level_
        _list_per_level_ = []
        _list_bot_level_ = []

        if _level_ == 0:
            _list_per_level_ = [
                np.array([[0.0]], dtype=np.float64) for _ in range(num_bot)
            ]
            _list_bot_level_ = [bot_lv0 for _ in range(num_bot)]
        else:
            env_name = sys.argv[1]
            if len(args) > 0:
                dict_level = json.load(
                    open(f"{SHORT_PATH}src/Log/check_system_about_level.json")
                )
            else:
                dict_level = json.load(open(f"{SHORT_PATH}src/Log/level_game.json"))

            if str(_level_) not in dict_level[env_name]:
                raise Exception("Hiện tại không có level này")

            lst_agent_level = dict_level[env_name][str(level)][2]

            for i in range(num_bot):
                if level == -1:
                    module_agent = load_module_player(
                        lst_agent_level[i], game_name=env_name
                    )
                    _list_per_level_.append(module_agent.DataAgent())
                else:
                    data_agent_level = np.load(
                        f"{SHORT_PATH}src/Agent/{lst_agent_level[i]}/Data/{env_name}_{level}/Train.npy",
                        allow_pickle=True,
                    )
                    module_agent = load_module_player(lst_agent_level[i])
                    _list_per_level_.append(
                        module_agent.convert_to_test(data_agent_level)
                    )
                _list_bot_level_.append(module_agent.Test)

    return _list_bot_level_, _list_per_level_


@njit()
def check_run_under_njit(agent, perData):
    return True


def run(
    p0: any = bot_lv0,
    num_game: int = 100,
    per_player: np.ndarray = np.array([[0.0]]),
    level: int = 0,
    *args,
):
    num_bot = getAgentSize() - 1
    list_other = np.array([-1] + [i + 1 for i in range(num_bot)])
    try:
        check_njit = check_run_under_njit(p0, per_player)
    except:
        check_njit = False

    load_agent(level, *args)

    if check_njit:
        return n_games_numba(
            p0,
            num_game,
            per_player,
            list_other,
            _list_per_level_[0],
            _list_per_level_[1],
            _list_per_level_[2],
            _list_bot_level_[0],
            _list_bot_level_[1],
            _list_bot_level_[2],
        )
    else:
        return n_games_normal(
            p0,
            num_game,
            per_player,
            list_other,
            _list_per_level_[0],
            _list_per_level_[1],
            _list_per_level_[2],
            _list_bot_level_[0],
            _list_bot_level_[1],
            _list_bot_level_[2],
        )
