SHORT_PATH = ""
import sys
from numba import njit
from src.render_template import Render as __Render, import_files

from numba.core.errors import (
    NumbaPendingDeprecationWarning as __NumbaPendingDeprecationWarning,
)
import warnings as __warnings

__warnings.simplefilter("ignore", __NumbaPendingDeprecationWarning)
_game_name_ = "RockPaperScissors"


def make(game_name: str = "RockPaperScissors") -> any:
    """
    make the environment

    Parameters
    ----------
    game_name : str, optional
        name of the game, by default "RockPaperScissors"
        List game:
            'Catan',
            'CatanNoExchange',
            'Century',
            'Durak',
            'Exploding_Kitten',
            'Fantan',
            'GoFish',
            'Imploding_Kitten',
            'MachiKoro',
            'Poker',
            'RockPaperScissors',
            'Sheriff',
            'Splendor',
            'Splendor_v2',
            'Splendor_v3',
            'StoneAge',
            'SushiGo',
            'TLMN',
            'TicketToRide',
            'WelcomeToTheDungeon_v1',
            'WelcomeToTheDungeon_v2',
    """

    global _game_name_, __env__, agent_random
    _game_name_ = game_name
    add_game_to_syspath()
    __env__ = __import__(f"src.Base.{_game_name_}.env", fromlist=["*"])
    import_files(_game_name_)
    agent_random = __env__.bot_lv0

def add_game_to_syspath():
    if len(sys.argv) >= 2:
        sys.argv = [sys.argv[0]]
    sys.argv.append(_game_name_)


#'------------------------------------------------------------------------------------------------------'
#'------------------------------------------------------------------------------------------------------'
#'------------------------------------------------------------------------------------------------------'


@njit()
def getValidActions(state):
    """
    return a array of valid actions

    Parameters
    ----------
    state : any
        current state of the game

    Returns
    -------
    ValidActions : np.array 1D
        np.array of valid actions
    """

    return __env__.getValidActions(state)


@njit()
def getActionSize():
    """
    return the size of action space

    Returns
    -------
    ActionSize : int
    """

    return __env__.getActionSize()


@njit()
def getAgentSize():
    """
    return the size of agent space

    Returns
    -------
    AgentSize : int
    """

    return __env__.getAgentSize()


@njit()
def getStateSize():
    """
    return the size of state space
    
    Returns
    -------
    StateSize : int
    """

    return __env__.getStateSize()


@njit()
def getReward(state):
    """
    return the reward of the state

    Parameters
    ----------
    state : any
        current state of the game
    Returns
    -------
    Reward : int
        0 if the game is not end
        1 if the game is end and the player win
        -1 if the game is end and the player lose
    """
    return __env__.getReward(state)


# @njit()
def run(
    agent,
    num_game: int = 100,
    agent_model: any = 1,
    level: int = 0,
    *args,
):
    """
    run the game

    Parameters
    ----------
    p0 : any, optional
        the agent of environment, this is function,
        by default 'bot_lv0'
    num_game : int, optional
        number of game, by default 100
    agent_model : any, optional
        model of agent, by default 1
    level : int, optional
        level of the game, by default 0
        0: random mode
        1: easy mode
        -1: hard mode

    Returns
    -------
    result :
        result of the game
    agent_model :
        model of agent
    """
    return __env__.run(agent, num_game, agent_model, level, *args)


def render(
    Agent="human",
    agent_model: any = [0],
    level: int = 0,
    *args,
    max_temp_frame=100,
):
    """
    render the game

    Parameters
    ----------
    Agent : str, optional
        the agent of environment, by default "human"
    agent_model : any, optional
        data of agent, by default [0]
    level : int, optional
        level of the game, by default 0
        0: random mode
        1: easy mode
        -1: hard mode

    Returns
    -------
    result :
        result of the game
    agent_model :
        model of agent
    """
    list_agent, list_data = __env__.load_agent(level, *args)

    if "__render" not in globals():
        global __render
        __render = __Render(Agent, agent_model, list_agent, list_data, max_temp_frame)
    else:
        __render.__init__(Agent, agent_model, list_agent, list_data, max_temp_frame)

    return __render.render()


def get_data_from_visualized_match():
    if "__render" not in globals():
        print("Nothing to get, visualize the match before running this function")
        return None

    return {
        "history_state": __render.history_state,
        "history_action": __render.history_action,
    }
