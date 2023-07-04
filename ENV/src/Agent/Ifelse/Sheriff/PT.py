import sys
import numpy as np
from numba import njit
from numba.typed import List
import env

game_name = sys.argv[1]
env.make(game_name)


def DataAgent():
    return np.array([])


# ----------------------#----------------------#----------------------#----------------------#----------------------#----------------------
@njit()
def Test(state, per):
    actions = env.getValidActions(state)
    actions = np.where(actions == 1)[0]
    if 2 in actions:
        return 2, per
    elif 3 in actions:
        return 3, per
    elif 6 in actions:
        return 6, per
    elif 8 in actions:
        return 8, per
    elif 18 in actions:
        return 18, per
    elif 19 in actions:
        return 19, per
    elif 26 in actions:
        return 26, per
    elif 28 in actions:
        return 28, per
    elif 31 in actions:
        return 31, per
    elif 32 in actions:
        return 32, per
    elif 49 in actions:
        return 49, per
    elif 61 in actions:
        return 61, per
    elif 65 in actions:
        return 65, per
    elif 69 in actions:
        return 69, per
    elif 77 in actions:
        return 77, per
    elif 78 in actions:
        return 78, per
    elif 81 in actions:
        return 81, per
    else:
        return np.random.choice(actions), per
