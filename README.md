
<a href="https://colab.research.google.com/github/datvodinh10/recurrent-ppo/blob/main/main.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# PPO + LSTM Project

## Introduction
This project explores the combination of Proximal Policy Optimization (PPO) and Long Short-Term Memory (LSTM) networks in reinforcement learning tasks. PPO is a popular policy optimization algorithm, while LSTM is a type of recurrent neural network that is capable of capturing temporal dependencies in sequential data. The goal of this project is to leverage the benefits of both PPO and LSTM to enhance the performance of reinforcement learning agents.

## Graph

`Forward`
![](img/PPO-LSTM%20Graph.svg)
`Backward`
![](img/Backward%20Graph.svg)
## Installation

1. Clone the repository:

```
git clone https://github.com/datvodinh10/recurrent-ppo.git
```

2. Install requirement:
```
pip install -r requirements.txt
```


## Run

```
Open main.ipynb in Colab -> Run All
```

## Update
- model_v2: split network, normalize state (running mean and var), use GRU (instead of LSTM)
- model: shared network, LSTM. 