# EARL
### Extensible Attention-based Rocket League model

Uses a transformer-like architecture to support any number of in-game entities (balls, boosts, players) simultaneously.

Provided models:
- `EARL`
  The standard model, used to model interactions between all the in-game entities equally.
- `EARLPerceiver`
  A compressed version of EARL inspired by the [Perceiver](https://arxiv.org/abs/2103.03206) which uses a small number of entities to attend to all entities, reducing complexity and improving performance.
  Particularly good for predicting actions, or reinforcement learning through [RLGym](https://github.com/lucas-emery/rocket-league-gym)
- `EPRL`
  An experimental alternative to EARL which does a simple max pooling operation instead of attention.
  
An example of use for the standard EARL model:
![Actor Critic](EARLActorCritic.svg?raw=true)


### Installation
```
pip install EARL-pytorch
```

### Example
```python
from earl_pytorch import EARL
from earl_pytorch.model import EARL, NextGoalPredictor
from earl_pytorch.dataset.create_dataset import replay_to_dfs, convert_dfs
from torch.nn import Sequential

earl = EARL()

# Option 1: Open some replay file
model = Sequential(earl, NextGoalPredictor(earl.n_dims))
dfs = replay_to_dfs("2627e02a-aa46-4e13-b66b-b76a32069a07.replay", )
x_data, y_data = convert_dfs(dfs, tensors=True)

for epoch in range(n_epochs):
    ...  # Train the model
```

<details>
  <summary>Details</summary>
  
Currently, the input is 21 features for each entity. For the non-relevant entities, the values are set to zero by default.
  
| Feature      | Type  | Entities            |
|--------------|-------|---------------------|
| cls          | bool  | None                |
| is_ball      | bool  | Ball                |
| is_boost     | bool  | Boost               |
| is_blue      | bool  | Player              |
| is_orange    | bool  | Player              |
| pos_x        | float | Ball, Boost, Player |
| pos_y        | float | Ball, Boost, Player |
| pos_z        | float | Ball, Boost, Player |
| forward_x    | float | Player              |
| forward_y    | float | Player              |
| forward_z    | float | Player              |
| up_x         | float | Player              |
| up_y         | float | Player              |
| up_z         | float | Player              |
| vel_x        | float | Ball, Player        |
| vel_y        | float | Ball, Player        |
| vel_z        | float | Ball, Player        |
| ang_vel_x    | float | Ball, Player        |
| ang_vel_y    | float | Ball, Player        |
| ang_vel_z    | float | Ball, Player        |
| boost_amount | float | Boost, Player       |
| is_demoed    | bool  | Boost, Player       |
| on_ground    | bool  | Player              |
| has_flip     | bool  | Player              |

Note: pos and vel are divided by 2300, ang_vel by 5.5 and boost by 100.

These values are fed into a linear layer to produce a kind of embedding for the entity state, which is then fed into a transformer, producing final embeddings for each entity depending on all the other entities.

The CLS "entity" is to be used to summarize the game state, like predicting which team is likely to score next.

</details>
