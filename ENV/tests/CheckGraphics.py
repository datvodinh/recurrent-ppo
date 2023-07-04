import warnings


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

import os


def check_graphics(game_name):
    BOOL_CHECK_GRAPHICS = True
    msg = []
    if os.path.exists(f"src/Base/{game_name}/_render_func.py") == False:
        BOOL_CHECK_GRAPHICS = False
        msg.append("_render_func.py not found")
    elif os.path.exists(f"src/Base/{game_name}/env.py") == False:
        BOOL_CHECK_GRAPHICS = False
        msg.append("env.py not found")
    return BOOL_CHECK_GRAPHICS, msg
