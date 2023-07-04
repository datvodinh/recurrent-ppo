# from setup import _game_name

# base_module = __import__(f"src.Base.{_game_name}.env", fromlist=["*"])
# from src.render_template import Render as __Render, import_files as __import_files
# from numba.core.errors import NumbaPendingDeprecationWarning as __NumbaPendingDeprecationWarning
# import warnings as __warnings
# __warnings.simplefilter("ignore", __NumbaPendingDeprecationWarning)


# __import_files(_game_name)

# __env = base_module

# bot_lv0 = __env.bot_lv0
# getValidActions = __env.getValidActions
# getActionSize = __env.getActionSize
# getAgentSize = __env.getAgentSize
# getStateSize = __env.getStateSize
# getReward = __env.getReward
# run = __env.run

# def render(Agent = 'human',
#            per_data: any = [0],
#            level: int = 0,
#             *args,
#             max_temp_frame=100):
#     list_agent, list_data = __env.load_agent(level, *args)

#     if "__render" not in globals():
#         global __render
#         __render = __Render(Agent, per_data, list_agent, list_data, max_temp_frame)
#     else:
#         __render.__init__(Agent, per_data, list_agent, list_data, max_temp_frame)

#     return __render.render()


# def get_data_from_visualized_match():
#     if "__render" not in globals():
#         print("Nothing to get, visualize the match before running this function")
#         return None

#     return {
#         "history_state": __render.history_state,
#         "history_action": __render.history_action,
#     }
