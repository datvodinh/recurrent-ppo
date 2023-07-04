import importlib.util
import math

import ipywidgets as widgets
import numpy as np
from IPython.display import clear_output, display

from env import SHORT_PATH


def import_files(game_name):
    spec = importlib.util.spec_from_file_location(
        "_env", f"{SHORT_PATH}src/Base/{game_name}/env.py"
    )
    global _env
    _env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_env)

    spec_1 = importlib.util.spec_from_file_location(
        "_render_func", f"{SHORT_PATH}src/Base/{game_name}/_render_func.py"
    )
    global _render_func
    _render_func = importlib.util.module_from_spec(spec_1)
    spec_1.loader.exec_module(_render_func)


class Render:
    def __init__(self, Agent, Data, list_agent, list_data, max_temp_frame) -> None:
        self.list_agent = list_agent
        self.list_data = list_data
        self.Agent = Agent
        self.Data = Data
        self.max_temp_frame = max_temp_frame

        self.output_image = widgets.Output()

        self.take = widgets.Dropdown(
            options=["Take an action"],
            value="Take an action",
            disabled=True,
            layout=widgets.Layout(width="20%"),
        )
        self.take.observe(self.handle_take, "value")

        self.explain = widgets.Dropdown(
            options=["Explain an action"],
            value="Explain an action",
            disabled=True,
            layout=widgets.Layout(width="20%"),
        )
        self.explain.observe(self.handle_explain, "value")

        self.slider = widgets.IntSlider(
            value=-1,
            min=-1,
            max=0,
            step=1,
            description="State:",
            disabled=True,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format="d",
            layout=widgets.Layout(width="50%"),
        )
        self.slider.observe(self.handle_slider, "value")

        self.previous = widgets.Button(
            disabled=True, icon="fa-chevron-left", layout=widgets.Layout(width="5%")
        )
        self.previous.on_click(self.handle_previous)

        self.next = widgets.Button(
            disabled=True, icon="fa-chevron-right", layout=widgets.Layout(width="5%")
        )
        self.next.on_click(self.handle_next)

        self.output_text = widgets.Output()

        self.hbox = widgets.HBox(
            [self.take, self.previous, self.slider, self.next, self.explain],
            layout=widgets.Layout(width="100%"),
        )
        self.vbox = widgets.VBox(
            [self.output_image, self.hbox, self.output_text],
            layout=widgets.Layout(display="flex", align_items="center"),
        )

        self.start()

    def start(self):
        self.env_components = _render_func.get_env_components()
        self.history_state = []
        self.images = {}
        self.max_state_idx = -1
        self.history_action = []
        self.history_valid_actions = []

        if type(self.Agent) == str and self.Agent.lower() == "human":
            self.system_mode = "play"
            self.step()
        else:
            action = None
            win, state, self.env_components = _render_func.get_main_player_state(
                self.env_components, self.list_agent, self.list_data, action
            )
            self.history_state.append(state)
            while win == -1:
                valid_actions = np.where(_env.getValidActions(state) == 1)[0]
                self.history_valid_actions.append(valid_actions)
                action, self.Data = self.Agent(state, self.Data)
                if action not in valid_actions:
                    raise Exception("Invalid action")

                self.history_action.append(action)
                win, state, self.env_components = _render_func.get_main_player_state(
                    self.env_components, self.list_agent, self.list_data, action
                )
                self.history_state.append(state)

            if win == 0:
                self.system_mode = "You lose"
            else:
                self.system_mode = "You win"

            self.max_state_idx = len(self.history_state) - 1
            self.slider.max = self.max_state_idx
            self.slider.value = self.slider.max
            self.slider.min = 0

    def step(self, action=None):
        self.disable_all()

        if not action is None:
            self.history_action.append(action)

        win, state, self.env_components = _render_func.get_main_player_state(
            self.env_components, self.list_agent, self.list_data, action
        )
        self.history_state.append(state)
        self.max_state_idx += 1

        if win == 0:
            self.system_mode = "You lose"
        elif win == 1:
            self.system_mode = "You win"

        if len(self.images) == self.max_temp_frame:
            min_ = min(self.images.keys())
            self.images.pop(min_)

        self.images[self.max_state_idx] = _render_func.get_state_image(state)
        self.slider.max = self.max_state_idx
        self.slider.value = self.slider.max
        if self.max_state_idx == 0:
            self.slider.min = 0

    def render(self):
        return display(self.vbox)

    def handle_take(self, p):
        if p.new != "Take an action":
            self.step(p.new)

        self.take.value = "Take an action"

    def handle_explain(self, p):
        if p.new != "Explain an action":
            self.show_text(f"{p.new}: {_render_func.get_description(p.new)}")

        self.explain.value = "Explain an action"

    def handle_previous(self, p):
        self.slider.value -= 1

    def handle_next(self, p):
        self.slider.value += 1

    def handle_slider(self, p):
        self.disable_all()

        if p.new not in self.images.keys():
            if p.new >= self.max_state_idx / 2.0:
                up = int(p.new + self.max_temp_frame / 2.0)
                upbound = min(self.max_state_idx, up)
                l_ = self.max_temp_frame - (upbound - p.new + 1)
                lowbound = p.new - l_
                lowbound = max(0, lowbound)
            else:
                low = math.ceil(p.new - self.max_temp_frame / 2.0)
                lowbound = max(0, low)
                u_ = self.max_temp_frame - (p.new - lowbound + 1)
                upbound = p.new + u_
                upbound = min(self.max_state_idx, upbound)

            keys = list(self.images.keys())
            for key in keys:
                if key < lowbound or key > upbound:
                    self.images.pop(key)

            keys = list(self.images.keys())
            for key in range(lowbound, upbound + 1):
                if key not in keys:
                    self.images[key] = _render_func.get_state_image(
                        self.history_state[key]
                    )

        self.show_image(p.new)

        if self.system_mode == "play" and p.new == self.max_state_idx:
            state = self.history_state[p.new]
            if len(self.history_valid_actions) != self.max_state_idx + 1:
                valid_actions = np.where(_env.getValidActions(state) == 1)[0]
                self.history_valid_actions.append(valid_actions)
            else:
                valid_actions = self.history_valid_actions[p.new]

            self.take.options = ["Take an action"] + list(valid_actions)
            self.explain.options = ["Explain an action"] + list(valid_actions)
            self.show_text("")
            self.take.disabled = False
            self.explain.disabled = False
            if p.new != 0:
                self.previous.disabled = False
        else:
            if p.new != self.max_state_idx:
                valid_actions = self.history_valid_actions[p.new]
                self.explain.options = ["Explain an action"] + list(valid_actions)
                self.show_text(f"Action has been taken: {self.history_action[p.new]}")
                self.next.disabled = False
                self.explain.disabled = False
            else:
                self.show_text(self.system_mode)

            if p.new != 0:
                self.previous.disabled = False

        self.slider.disabled = False

    def show_image(self, idx):
        with self.output_image:
            clear_output(wait=True)
            display(self.images[idx])

    def show_text(self, txt):
        with self.output_text:
            clear_output(wait=True)
            print(txt)

    def disable_all(self):
        self.take.disabled = True
        self.explain.disabled = True
        self.slider.disabled = True
        self.previous.disabled = True
        self.next.disabled = True
