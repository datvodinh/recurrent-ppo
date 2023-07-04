# 1. Overview of the graphics system:
**1. Graphics interface:**
> * It is a collection of images representing the game.
> * Each image depicts the **(state)** that the player observes from a personal perspective.
> * It is designed based on the state, meaning it visually represents the parameters of the state for the player to easily track the game's progress.

**2. Interaction:**
> * At each state, the player receives appropriate **(actions)** to use.
> * When selecting one of those actions, the graphics interface records it and transitions to the corresponding next state. In this new state, there will be another set of appropriate actions... and so on until the end of the game.

--> All **(states)** and actions used by the player are stored for later observation when a game ends.
***

# 2. Designing the graphics system:
* Use the available template for the system.
* Use the PIL library to process images.
## 1. Create the `_render_func.py` file and build the functions in the system:
1. `get_state_image` function:
> * Input: state ( np.array 1D)
> * Output: Image representing the corresponding state

2. `get_description` function:
> * Input: action ( float64)
> * Output: String describing the meaning of that action

3. `get_env_components` function:
> * Input: ( None)
> * Output: An `env_components` object containing all components and information about the environment's state.

4. `get_main_player_state` function:
> * Input: `env_components, list_agent, list_data, action = None`
> * Output: `win, state, new_env_components`
>    * win: Does the player win or not?
>    * state: The next state received by the **main player**
>    * new_env_components: contains changes to the environment's components.

# How to use the graphics system:
![GraphicsGuide](https://github.com/tandat17z/ENV/assets/126872123/f9f02500-c57f-46e8-a0a9-905f0d4b47c3)

***

1. **Observing a match played by an Agent:**

![GraphicsGuide](https://github.com/tandat17z/ENV/assets/126872123/15e55747-eadd-4fb9-9de8-1daa7f0bcb54)

1. **Playing directly:**
* Set Agent = "human"

![image](https://github.com/tandat17z/ENV/assets/126872123/76ec52b7-67a9-4bbf-b484-75ed57abc465)
