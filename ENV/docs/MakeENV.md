# Guide to code Game System
Ex: RockPaperScissors.

## I. Environment, action and state design.
### 1. Environment(env)
- Contains information about all the elements present in the game. The `env` can be understood as the perspective of the game master (referee) standing outside, observing the game, and being able to see the state of the board and the state of all players.
- Simulate the `env` as one or multiple arrays to optimize runtime using the `numba` library.
- How should the `env` be designed? Study and observe other players in the game to initially determine the information that needs to be simulated in the `env`."

### 2. Actions
- Consist of all the actions that an Agent can perform in the game.
- Denote: $\Lambda = \{0,1,...,n-1\}$ where $\Lambda$ is the action space and n is the total number of actions in the game.

### 3. State
- The state of the environment that the *Agent* can observe.
- Simulated using a single numpy array, one-dimensional, containing data with categorical properties when passed from the environment to the state.
- *States* of the agent at different positions must have a common representation.

## II.Function design in ENV

|Function       |Input          |Input Description|Ouput            |Output Description|
|:-------       |:-----------   |:-----------     |:----------      |:----------       |
|initEnv        |None           | np.array 1D     | np.array 1D     |initialize env|
|getAgentState  |np.array 1D    | current env    | np.array 1D     |State agent at current time step|
|stepEnv        |np.array 1D    | current env    | np.array 1D     |Change the env after applied action|
|getValidActions|np.float64     | state           |np.float64       |Valid actions that agent can take |
|getReward      |np.float64     | state           |int              |1: win, 0: loss, -1: not done|
|getActionSize  |None           |                 |int              |Shape of action space|
|getStateSize   |None           |                 |int              |Shape of the state|
|getAgentSize   |None           |                 |int              |Number of agent in the game|
|checkEnded     |np.array 1D    |env              |int              |-1: not done, $0 \rightarrow n$: winner's index|
|one_game       |               |                 |                 |Play one game|
|n_games        |               |                 |                 |Play n games, n defined by user|
|Run            |Agent, int, any, int|Agent, number_of_games, file_data_agent, level|                 |The main function to use Env|

## III. Environment testing
### 3.1 Testing with ENV's built in function
This is a mandatory procedure to verify after creating a system that returns the correct information about the system's rules, and if there are any errors, it will return the existing errors. The following test cases are available in the system:

- Check for changes in *State* length during runtime.
- Check for changes in *Actions* length during runtime.
- Check if the output conforms to the standards of 0 or 1.
- Check for negative *State* values.
- Check for running with agent `numba` and without `numba`.
- Check if the number of completed matches differs from the total number of games played.
- Check if the number of winning matches using getReward matches the number of wins returned by `env.run()`.
- Check if `env.getValidActions()` doesn't produce errors when given any state."

#### 3.1.1 Import `check_env`

```python
from setup import make
from tests.CheckEnv import check_env

env = make("RockPaperScissors")
print(check_env(env))
```

#### 3.1.2 Test with pytest

In terminal:
```
pip install pytest
pytest
```

### 3.2 Testing with live game.
When coding a new environment, it can be test by play some games in real life and check the rule.

### 3.3 Testing with Graphic system
- Create an graphic env with guide
- Test a game.

## IV. The process of building a system

(Particularly with RockPaperScissors aka RPS)

### 4.1 Learn the rules and understand the gameplay.
- It is necessary to grasp all the possible situations that can occur in the game,from basic to complex.
> In RPS, the rules are very simple. Players make choices (rock, paper, scissors) and compare them to each other: Rock beats scissors, scissors beats paper, and paper beats rock. If both players make the same choice, it's a tie, and the game continues with a new round until a winner is found.

### 4.2 Design env, state and actions.

- `env`: The components of the game from the perspective of an observer (someone who knows everything about the game but is not a player).
- `state`: The components of the game from the perspective of the current player (this information is derived from the env).
- `actions`: All the possible actions that a player can take while playing.
- Depending on the nature of the game, we can define the components in env, state, and actions in a way that allows for easy transitioning between them, facilitates code writing, and accurately represents the essence of the game. It is especially important to recreate the gameplay through the graphics system. You can create a block diagram to illustrate the functioning of the game system you want to design, providing an overview before starting the coding process.

> **Specifically, for the RPS game**:
> `env`: Observers of the match will know the choices of all players, the number of rounds played, whose turn it is, the winner, and whether the game has ended or not.
> `state`: Players will know their own choices and when it comes to the comparison phase, they will know the opponent's choice as well.
> `action`: Players can make one of three choices: Rock, Paper, or Scissors.
> These are the most basic elements. Next, we need to summarize them in the form of arrays and add a few more elements to make the system function easily. (Refer to the [README](https://github.com/ngoxuanphong/ENV/tree/main/Base/RockPaperScissors/README.md))
> *Why is there an additional **action confirm** (action = 3)?:* When a player receives a state that only allows them to perform the **confirm** action, it allows the player to know everyone's choices and the outcome of that round. Furthermore, this state will help the graphical system display that information, making the observation of the match more complete. Without this action, players would only receive the state in the phase of choosing Rock, Paper, or Scissors (without being allowed to know the opponent's choice).

- ### 4.3 Writing code according to the designed format:
Sample system git link: [RockPaperScissors](https://github.com/ngoxuanphong/ENV/tree/main/Base/RockPaperScissors)
- `initEnv()`: Initializes the initial state (env) of the game.
- `getAgentState(env)`: Returns an array State, which contains the state information retrieved from the game's current state (env) at the current time step.
> - **Note:** State represents the game state from the player's perspective.
> - **Env and state are different**: Env always knows the choices made by the player, while the state can only be determined after comparing to know the opponent's choice.
- `getValidActions(state)`: Returns an array validActions (containing only values 0 and 1 – with validActions[k] = 1 if action k can be performed, otherwise 0), representing the actions that can be taken by the player at the given state.
> Specifically, in the choice phase, the player can choose one of the actions: Rock, Paper, Scissors... (validActions = [1, 1, 1, 0]); while in the confirmation phase, the player can only perform the confirm action (validActions = [0, 0, 0, 1]).
- `stepEnv(action, env)`: Given an action, this method performs the game based on that action (i.e., modifies the env) to generate a new state. This is the most complex function in the system, requiring a deep understanding of the game to handle all the cases and ensure the game operates correctly.
> Specifically, when the action is 0, 1, or 2 (Rock, Paper, Scissors), we need to determine whose turn it currently is (or find the player who made the action) and remember the player's choice. After that, we need to switch to the other player...
- `checkEnded(env)`: Returns a number indicating the winning player's index (-1 if the game has not ended, 0...n for the winning player).
- `getReward(state)`: Returns a value indicating whether the player wins or loses based on the given state (-1 if the game has not ended, 0 for loss, 1 for win).
- `one_game(...)`: Using the built-in functions, this function executes one round of the game.
> Specifically, it first initializes the game state using initEnv(), then enters a while loop with a condition to stop the game after reaching a certain limit (in RPS, the limit is the number of plays < 100). For each player, they receive the state, make an action, and then the environment receives the action and modifies it... This continues until a winning player is found, at which point the loop is exited. Finally, the system asks each player to perform a final turn, and this function returns the winning player.
- `n_games(...)`: Executes n number of the games and returns the number of victories for agent p0.
> `List_other` is an array that rearranges the positions of the agents, where agent p0 has a value of -1, and other agents have values from 1 to n (e.g., List_other = [-1, 1, 2, 3] or [1, 3, -1, 2] ... for a 4-player game system, in RPS, List_other = [-1, 1] or [1, -1]).
- `The remaining part` is mostly the same for most systems, but it is important to check for any differences that may exist in certain points (e.g., the number of agents and per_file).

### For better understanding, readers should carefully review the sample system, and if there are any questions, please send them to us.