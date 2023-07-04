##  FUNCTION


|Function       |Input          |Input Description|Ouput            |Output Description|
|:-------       |:-----------   |:-----------     |:----------      |:----------       |
|Agent          |np.float64, any| state,data agent|int, any         ||
|getValidActions|np.float64     | state           |np.float64       |Valids Actions in current turn|
|getReward      |np.float64     | state           |int              |1: win, 0:lose, -1: not done|
|getActionSize  |None           |                 |int              |action size |
|getStateSize   |None           |                 |int              |amount agent state size|

## Run environment
### Basic agent
```python
from numba import njit
import numpy as np

@njit()
def Agent(state, agent_data):
    validActions = env.getValidActions(state)
    actions = np.where(validActions==1)[0]
    action = np.random.choice(actions)
    return arr_action[idx], agent_data
```
[Example](https://github.com/ngoxuanphong/ENV/tree/main/Agent)
### Import one env
```python
from setup import make
env = make('SushiGo')
```
More env please read [Environments](https://github.com/ngoxuanphong/ENV/wiki/Environments)
### Run multiple matches of a environment
```python
env = make('SushiGo)
count_win, agent_data = env.numba_main_2(Agent, count_game_train = 1, agent_data = [0], level = 0)
```

|Var|Type|Description|
|:-------|:-----------|:-----------|
|count_game_train| int     |matches of a environment|
|agent_data      | any     | data train of agent|
|level           | 0, 1, -1| level of environment (update more later)|
### Render one game
```python
env.render(Agent=Agent, per_data=[0], level=0, max_temp_frame=100)
```
