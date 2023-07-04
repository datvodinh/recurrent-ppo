game_name = "Splendor"
time_run_game = 100
N_AGENT = 5
N_GAME = 15
PASS_LEVEL = 2500
COUNT_TRAIN = 10000
COUNT_TEST = 1000

#  path = "C:\AutomaticTrain\State.xlsx"
#  SHOT_PATH = 'A:/AutoTrain/GAME/'
DRIVE_FOLDER = "G:/My Drive/AutomaticColab/"


SHORT_PATH = ""
#  DRIVE_FOLDER = 'H:/Drive của tôi/AutomaticColab/'

import importlib.util
import sys


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
