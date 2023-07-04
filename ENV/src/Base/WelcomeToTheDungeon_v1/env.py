import numpy as np
from numba import njit, jit

from numba.typed import List
import importlib.util, json, sys


__STATE_SIZE__ = 40
__ACTION_SIZE__ = 25
__AGENT_SIZE__ = 4


@njit
def initEnv():
    env = np.full(63, 0)  # Khởi tạo env

    # env[0:8] = 0
    #     env[0:2]: Thắng,Thua của người chơi 1
    #     env[2:4]: Thắng,Thua của người chơi 2
    #     env[4:6]: Thắng,Thua của người chơi 3
    #     env[6:8]: Thắng,Thua của người chơi 4

    # env[8:21]: 13 monsters
    # env[21]: Sô monsters

    # env[22]: Hero: 0: Warrior   1: Rogue
    # env[23:29]: Hero equipments

    # env[29:42] = 0: quái vật trong hang
    # env[42]: Số lượng quái vật trong hang

    # env[43:56] = 0: Người xem quái vật: -1: chưa lật; 0,1,2,3: người chơi

    # env[56:60] = 0: Trạng thái trong round của 4 người chơi: 0: chơi, 1: bỏ lượt
    # env[61] = 0: Turn
    # env[62] = 0: Phase
    return env


@njit
def resetRound(env):
    monsters = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 9])
    np.random.shuffle(monsters)
    env[8:21] = monsters
    env[21] = 0

    if env[60] == 0:
        character = np.random.randint(2)  # Random Hero: 0: Warrior, 1: Rogue
        env[22] = character
    elif env[60] == 1:
        env[22] = 0  # Nếu người chơi chọn Warrior thì gán Hero là Warrior
    elif env[60] == 2:
        env[22] = 1  # Nếu người chơi chọn Rogue thì gán tướng là Rogue

    env[23:29] = 1  # Đặt lại các trang bị

    env[29:43] = 0  # Đặt lại quái vật trong hang
    env[43:56] = -1  # Đặt lại người xem quái vật
    env[56:60] = 0  # Đặt lại trạng thái người chơi
    for i in range(4):
        if env[i * 2 + 1] == 2:
            env[56 + i] = 1
    env[62] = 0  # Phase = 0 bắt đầu lượt mới

    return env


@njit
def getStateSize():
    return __STATE_SIZE__


@njit
def getAgentState(env):
    state = np.zeros(__STATE_SIZE__)
    pIdx = env[61] % 4  # Index của người chơi nhận state
    score = env[0:8]
    passArr = env[56:60]
    for i in range(4):
        pIdxEnv = (pIdx + i) % 4
        state[i * 2 : i * 2 + 2] = score[pIdxEnv * 2 : pIdxEnv * 2 + 2]
        state[8 + i] = passArr[pIdxEnv]

    state[12] = env[21]  # Số thẻ monsters đã lật
    state[13] = env[42]  # Số monsters trong hang
    if env[22] == 0:
        state[14] = 1
        state[15:21] = env[23:29]
    else:
        state[21] = 1
        state[22:28] = env[23:29]
    state[28:36] = 0  # Monster đã xem trong phase
    if env[62] == 1:
        monster = env[7 + env[21]]
        list_monster = np.array([1, 2, 3, 4, 5, 6, 7, 9])
        monIdx = np.where(list_monster == monster)[0][0]
        state[28 + monIdx] = 1
    state[36:40] = 0  # Phase
    state[36 + env[62]] = 1

    return state


@njit
def getActionSize():
    return __ACTION_SIZE__


@njit
def getValidActions(state):
    validActions = np.zeros(__ACTION_SIZE__)
    phaseS = np.where(state[36:40] == 1)[0]
    if len(phaseS) != 0:
        phase = phaseS[0]
    else:
        return validActions

    if phase == 0:  # Lựa chọn bỏ lượt hay lật thẻ
        validActions[0] = 1  # Có thẻ bỏ lượt
        if (
            state[8] == 0 and state[12] < 13
        ):  # Nếu chưa bỏ lượt và số lượng quái vật đã mở < 13
            validActions[1] = 1  # Có thể lật thẻ

    elif phase == 1:  # Đã xem monster
        validActions[2] = 1  # Có thể bỏ vào hang

        equipment1 = state[15:21]
        equipment2 = state[22:28]
        idx1 = np.where(equipment1 == 1)[0]
        validActions[idx1 + 3] = 1
        idx2 = np.where(equipment2 == 1)[0]
        validActions[idx2 + 9] = 1

    elif phase == 2:
        validActions[15:23] = 1

    elif phase == 3:
        validActions[23:25] = 1

    return validActions


@njit
def stepEnv(action, env):
    phase = env[62]
    playerIdx = env[61] % 4
    if phase == 0:  # Bỏ qua hoặc xem monster
        if action == 0:
            env[56 + playerIdx] = 1  # Gán người chơi bỏ qua
            if checkEndRound(env) == 1:  # Nếu hết round
                while env[56 + env[61] % 4] == 1:
                    env[61] += 1
                if env[28] == 1:  # Nếu có trang bị Vorpal sword / Vorpal dagger
                    env[62] = 2  # Gán phase = 2 (Vào phase chọn quái vật tiêu diệt)

                else:  # Vào hang đánh
                    getRewardRound(env)
                    env[62] = 3

            else:
                env[61] += 1
                while env[56 + env[61] % 4] == 1:
                    env[61] += 1  # sang turn tiếp
        elif action == 1:
            env[21] += 1  # Nếu xem monster thì số thẻ monster lật tăng thêm 1
            env[42 + env[21]] = (
                env[61] % 4
            )  # Gán người xem quái vật là người chơi hiện tại
            env[62] = 1  # Sang phase lên 1

    elif phase == 1:  # Đã xem monster
        numMons = env[21]
        monster = env[7 + numMons]
        if action == 2:
            env[29 + env[42]] = monster  # env[42] là số monster trong hang
            env[42] += 1  # action 2: cho monster vào hang và tăng số monster trong hang
            env[62] = 0
            env[61] += 1
            while env[56 + env[61] % 4] == 1:
                env[61] += 1  # Sang turn tiếp theo
        else:
            idxeq = 0
            if action in range(3, 9):
                idxeq = action - 3
            else:
                idxeq = action - 9
            env[23 + idxeq] = 0  # Bỏ trang bị tương ứng
            env[62] = 0  # Gán phase = 0
            env[61] += 1
            while env[56 + env[61] % 4] == 1:
                env[61] += 1  # Sang turn tiếp theo

    elif phase == 2:  # Tiêu diệt quái vật
        env[62] = 0
        monsterUniqe = np.array([1, 2, 3, 4, 5, 6, 7, 9])
        monsteridx = action - 15
        monsterdelete = monsterUniqe[monsteridx]
        for i in range(13):
            if env[i + 29] == monsterdelete:
                env[i + 29] = 0

        getRewardRound(env)
        env[62] = 3

    elif phase == 3:  # Chọn Hero
        env[62] = 0
        env[60] = action - 22
        resetRound(env)
        while env[56 + env[61] % 4] == 1:
            env[61] += 1


@njit
def getReward(state):
    score = state[0:8]
    winArr = score[np.array([0, 2, 4, 6])]
    loseArr = score[np.array([1, 3, 5, 7])]
    if len(np.where(winArr == 2)[0]) == 1:
        if np.where(winArr == 2)[0][0] == 0:
            return 1
        else:
            return 0

    elif len(np.where(loseArr == 2)[0]) == 3:
        winner = np.where(loseArr < 2)[0][0]
        if winner == 0:
            return 1
        else:
            return 0
    else:
        return -1


@njit
def getAgentSize():
    return __AGENT_SIZE__


@njit
def checkEndRound(env):
    passArr = env[56:60]
    if np.sum(passArr) == 3:
        # Kết thúc round, đi vào hang
        return 1

    else:  # Chưa kết thúc game
        return -1


@njit
def checkEnded(env):
    scoreArr = env[0:8]
    winArr = scoreArr[np.array([0, 2, 4, 6])]
    loseArr = scoreArr[np.array([1, 3, 5, 7])]
    if len(np.where(winArr == 2)[0]) == 1:
        winner = np.where(winArr == 2)[0][0]
        return winner
    elif len(np.where(loseArr == 2)[0]) == 3:
        winner = np.where(loseArr < 2)[0][0]
        return winner
    else:
        return -1


@njit
def getRewardRound(env):
    # Đi vào hang
    playerIdx = env[61] % 4

    hero = env[22]
    equip = env[23:29]
    monsterDungeon = env[29:42].copy()
    monsterDungeon = monsterDungeon[::-1]
    numMons = 13
    win = 0
    if hero == 0:  # Tính điểm Warrior
        HpHero = 3
        HpHero += 3 * equip[0] + 5 * equip[1]
        if equip[2] == 1:  # Có Torch
            for i in range(numMons):
                if monsterDungeon[i] <= 3:
                    monsterDungeon[i] = 0
        if equip[3] == 1:  # Có Holy Grail
            for i in range(numMons):
                if monsterDungeon[i] % 2 == 0:
                    monsterDungeon[i] = 0
        if equip[4] == 1:  # Có Dragon Spear
            for i in range(numMons):
                if monsterDungeon[i] == 9:
                    monsterDungeon[i] = 0
        HpHero = HpHero - np.sum(monsterDungeon)
        if HpHero > 0:
            win = 1

    elif hero == 1:  # Tính điểm Rogue
        HpHero = 3
        HpHero += 3 * equip[0] + 5 * equip[1]

        if equip[3] == 1:  # Có Invisibility Cloak
            for i in range(numMons):
                if monsterDungeon[i] >= 6:
                    monsterDungeon[i] = 0
        if equip[4] == 1:  # Có Healing Potion
            if equip[2] == 1:  # Có Ring Of Power
                for i in range(numMons):
                    if monsterDungeon[i] <= 2:
                        HpHero += monsterDungeon[i]
                        monsterDungeon[i] = 0
                    else:
                        HpHero -= monsterDungeon[i]
                        monsterDungeon[i] = 0
                    if HpHero <= 0:
                        HpHero = 3
                        break
            else:
                for i in range(numMons):
                    HpHero -= monsterDungeon[i]
                    monsterDungeon[i] = 0
                    if HpHero <= 0:
                        HpHero = 3
                        break
        if equip[2] == 1:  # Có Ring Of Power
            for i in range(numMons):
                if HpHero <= 0:
                    break
                if monsterDungeon[i] <= 2:
                    HpHero += monsterDungeon[i]
                    monsterDungeon[i] = 0
                else:
                    HpHero -= monsterDungeon[i]
                    monsterDungeon[i] = 0
        HpHero = HpHero - np.sum(monsterDungeon)
        if HpHero > 0:
            win = 1

    if win == 0:
        env[playerIdx * 2 + 1] += 1
    elif win == 1:
        env[playerIdx * 2] += 1
    return win


@njit()
def one_game_numba(p0, list_other, per_player, per1, per2, per3, p1, p2, p3):
    env = initEnv()
    resetRound(env)
    _cc = 0

    while env[61] <= 1000 and _cc <= 10000:
        idx = env[61] % 4
        player_state = getAgentState(env)
        if list_other[idx] == -1:
            action, per_player = p0(player_state, per_player)
            list_action = getValidActions(player_state)
            if list_action[action] != 1:
                raise Exception("Action không hợp lệ")
        elif list_other[idx] == 1:
            action, per1 = p1(player_state, per1)
        elif list_other[idx] == 2:
            action, per2 = p2(player_state, per2)
        elif list_other[idx] == 3:
            action, per3 = p3(player_state, per3)

        stepEnv(action, env)
        winner = checkEnded(env)
        if winner != -1:
            break

        _cc += 1

    for p_idx in range(4):
        env[61] = p_idx
        p_state = getAgentState(env)
        if list_other[p_idx] == -1:
            act, per_player = p0(p_state, per_player)
        elif list_other[p_idx] == 1:
            action, per1 = p1(p_state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(p_state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(p_state, per3)
        else:
            raise Exception("Sai list_other.")

    winner__ = False
    if np.where(list_other == -1)[0][0] == winner:
        winner__ = True
    else:
        winner__ = False
    return winner__, per_player


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
    env = initEnv()
    resetRound(env)
    _cc = 0

    while env[61] <= 1000 and _cc <= 10000:
        idx = env[61] % 4
        player_state = getAgentState(env)
        if list_other[idx] == -1:
            action, per_player = p0(player_state, per_player)
            list_action = getValidActions(player_state)
            if list_action[action] != 1:
                raise Exception("Action không hợp lệ")
        elif list_other[idx] == 1:
            action, per1 = p1(player_state, per1)
        elif list_other[idx] == 2:
            action, per2 = p2(player_state, per2)
        elif list_other[idx] == 3:
            action, per3 = p3(player_state, per3)

        stepEnv(action, env)
        winner = checkEnded(env)
        if winner != -1:
            break

        _cc += 1

    for p_idx in range(4):
        env[61] = p_idx
        p_state = getAgentState(env)
        if list_other[p_idx] == -1:
            act, per_player = p0(p_state, per_player)

        elif list_other[p_idx] == 1:
            action, per1 = p1(p_state, per1)
        elif list_other[p_idx] == 2:
            action, per2 = p2(p_state, per2)
        elif list_other[p_idx] == 3:
            action, per3 = p3(p_state, per3)
        else:
            raise Exception("Sai list_other.")

    winner__ = False
    if np.where(list_other == -1)[0][0] == winner:
        winner__ = True
    else:
        winner__ = False
    return winner__, per_player


def n_games_normal(p0, num_game, per_player, list_other, per1, per2, per3, p1, p2, p3):
    win = 0
    for _ in range(num_game):
        np.random.shuffle(list_other)
        winner, per_player = one_game_normal(
            p0, list_other, per_player, per1, per2, per3, p1, p2, p3
        )
        win += winner

    return win, per_player


@njit()
def bot_lv0(state, perData):
    validActions = getValidActions(state)
    arr_action = np.where(validActions == 1)[0]
    idx = np.random.randint(0, arr_action.shape[0])
    return arr_action[idx], perData


import importlib.util, json, sys

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


# def stupid(state, perData):
#     validActions = getValidActions(state)
#     arr_action = np.where(validActions==1)[0]
#     idx = np.random.randint(0, arr_action.shape[0])
#     return arr_action[idx], perData

# run(stupid,1,np.zeros(1),0)
