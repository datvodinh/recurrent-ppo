import warnings
import env
from src.Utils import load_module_player

warnings.filterwarnings("ignore")
from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaExperimentalFeatureWarning,
    NumbaPendingDeprecationWarning,
    NumbaWarning,
)

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
warnings.simplefilter("ignore", category=NumbaWarning)

COUNT_TEST = 1000


# check hết hệ thống
def CheckAllFunc(Agent_name, BOOL_CHECK_ENV, msg):
    env.make("SushiGo")
    Agent = load_module_player(Agent_name)
    for func in ["DataAgent", "Train", "Test", "convert_to_save", "convert_to_test"]:
        try:
            getattr(Agent, func)
        except:
            msg.append(f"Không có hàm: {func}")
            BOOL_CHECK_ENV = False
    return BOOL_CHECK_ENV, msg


def CheckRunGame(Agent_name, BOOL_CHECK_ENV, msg):
    for game_name in [
        "Splendor_v3",
        "MachiKoro",
        "SushiGo",
    ]:
        env.make(game_name)
        Agent = load_module_player(Agent_name)
        try:
            per = Agent.DataAgent()
            win, per = env.run(Agent.Train, COUNT_TEST, per, 0)
        except:
            msg.append(f"Train đang bị lỗi {game_name}")
            BOOL_CHECK_ENV = False
            break

        try:
            per = Agent.convert_to_test(Agent.convert_to_save(Agent.DataAgent()))
            win, per = env.run(Agent.Test, COUNT_TEST, per, 0)
        except:
            msg.append(f"Test đang bị lỗi {game_name}")
            BOOL_CHECK_ENV = False
            break

    return BOOL_CHECK_ENV, msg


def check_agent(Agent_name):
    BOOL_CHECK_ENV = True
    msg = []
    print(
        Agent_name,
        "| Function checking ...",
    )
    BOOL_CHECK_ENV, msg = CheckAllFunc(Agent_name, BOOL_CHECK_ENV, msg)
    print(
        Agent_name,
        "| Run game checking ...",
    )
    BOOL_CHECK_ENV, msg = CheckRunGame(Agent_name, BOOL_CHECK_ENV, msg)
    return BOOL_CHECK_ENV, msg
