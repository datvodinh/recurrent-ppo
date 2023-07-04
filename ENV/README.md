# VIS - Environment for Reinforcement Learning
vis is an open source Python library for developing and comparing reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments, as well as a standard set of environments compliant with that API

![Python package](https://github.com/ngoxuanphong/ENV/workflows/Python%20package/badge.svg) 
<!-- ![Upload Python Package](https://github.com/ngoxuanphong/ENV/workflows/Upload%20Python%20Package/badge.svg) -->
<!-- [![Downloads](https://pepy.tech/badge/ma-gym)](https://pepy.tech/project/ma-gym) -->
[![Wiki Docs](https://img.shields.io/badge/-Wiki%20Docs-informational?style=flat)](https://github.com/ngoxuanphong/ENV/wiki)

##  Installation
We are support Python 3.7, 3.8, 3.9, 3.10 on Linux and Windows  
To install the base **vis** library, use:
- Using pip:
    ```python
    Update later
    ```

- Directly from source (recommended):
    ```python
    git clone https://github.com/ngoxuanphong/vis.git
    cd vis
    pip install -r requirements.txt
    ```

##  API
```python
import env

env.make()
env.run(env.agent_random)
```

```python
import env
from numba import njit
import numpy as np

@njit()
def Agent(state, agent_data):
    validActions = env.getValidActions(state)
    actions = np.where(validActions==1)[0]
    action = np.random.choice(actions)
    return action, agent_data
    
env.make('SushiGo')
env.run(Agent, num_game = 1000, agent_model = [0], level = 0)

```
[Example](https://github.com/ngoxuanphong/ENV/blob/main/src/Log/Example.ipynb)

Please refer to [Wiki](https://github.com/ngoxuanphong/ENV/wiki/Using) for complete usage details

##  Environment
**vis** includes 20 games, which are listed below along with the number of actions, observations, time run, win rate they have:

|Game        |Win-lv0       |win-lv1        |win-lv2        |Time-lv0       |Time-lv1       |Time-lv2       | Graphics      | Link|
|:-----------|:-----------  |:-----------   |:-----------   |:-----------   |:-----------   |:-----------   |:-----------   |:-----------   |
|Catan      |2440           | 354           | 7             |252            | 230           | 183           |True           |[Catan](https://github.com/ngoxuanphong/ENV/tree/main/Base/Catan)|
|CatanNoExchange|2491| 424| 22|168| 1021| 96|True|[CatanNoExchange](https://github.com/ngoxuanphong/ENV/tree/main/Base/CatanNoExchange)|
|Century|1970| 7| |36| 61| |True|[Century](https://github.com/ngoxuanphong/ENV/tree/main/Base/Century)|
|Durak|2540| 460| |15| 21| |True|[Durak](https://github.com/ngoxuanphong/ENV/tree/main/Base/Durak)|
|Exploding_Kitten|2044| 1695| |24| 35| |True|[Exploding_Kitten](https://github.com/ngoxuanphong/ENV/tree/main/Base/Exploding_Kitten)|
|Fantan|2464| 114| |17| 31| ||[Fantan](https://github.com/ngoxuanphong/ENV/tree/main/Base/Fantan)|
|GoFish|2466| 2517| 497|9| 34| 13|True|[GoFish](https://github.com/ngoxuanphong/ENV/tree/main/Base/GoFish)|
|Imploding_Kitten|1646| 1274| |62| 51| |True|[Imploding_Kitten](https://github.com/ngoxuanphong/ENV/tree/main/Base/Imploding_Kitten)|
|MachiKoro|2468| 31| 3|11| 9| 8|True|[MachiKoro](https://github.com/ngoxuanphong/ENV/tree/main/Base/MachiKoro)|
|Poker|1100| 1112| |81| 127| |True|[Poker](https://github.com/ngoxuanphong/ENV/tree/main/Base/Poker)|
|Sheriff|2481| 4| 365|29| 29| 19|True|[Sheriff](https://github.com/ngoxuanphong/ENV/tree/main/Base/Sheriff)|
|Splendor|2517| 21| 4|105| 154| 85|True|[Splendor](https://github.com/ngoxuanphong/ENV/tree/main/Base/Splendor)|
|Splendor_v2|2632| 9| 4|50| 180| 123|True|[Splendor_v2](https://github.com/ngoxuanphong/ENV/tree/main/Base/Splendor_v2)|
|Splendor_v3|2700| 681| 28|34| 33| 41|True|[Splendor_v3](https://github.com/ngoxuanphong/ENV/tree/main/Base/Splendor_v3)|
|StoneAge|2485| 40| 0|65| 229| 100|True|[StoneAge](https://github.com/ngoxuanphong/ENV/tree/main/Base/StoneAge)|
|SushiGo|2096| 144| 187|8| 14| 14|True|[SushiGo](https://github.com/ngoxuanphong/ENV/tree/main/Base/SushiGo)|
|TicketToRide|2017| 0| |84| 313| |True|[TicketToRide](https://github.com/ngoxuanphong/ENV/tree/main/Base/TicketToRide)|
|TLMN|2532| 766| 279|23| 25| 31|True|[TLMN](https://github.com/ngoxuanphong/ENV/tree/main/Base/TLMN)|
|WelcomeToTheDungeon_v1|2512| 830| |5| 11| ||[WelcomeToTheDungeon_v1](https://github.com/ngoxuanphong/ENV/tree/main/Base/WelcomeToTheDungeon_v1)|
|WelcomeToTheDungeon_v2|2458| 889| 443|8| 11| 23||[WelcomeToTheDungeon_v2](https://github.com/ngoxuanphong/ENV/tree/main/Base/WelcomeToTheDungeon_v2)|

Please refer to [Wiki](https://github.com/ngoxuanphong/ENV/wiki/Environments) for more details.
