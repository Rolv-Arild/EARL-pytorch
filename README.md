# EARL
## Extensible Attention-based Rocket League model

Uses a transformer-like architecture to support any number of in-game entities (balls, boosts, players) simultaneously.

Provided models:
- `EARLReplayModel`
  In-game event prediction, e.g. goals, touches, boost grabs and demos.
- `EARLActorCritic`
  Action prediction for playing the game, or reinforcement learning through [RLGym](https://github.com/lucas-emery/rocket-league-gym)
- `EARLRankModel`
  Rank prediction of players from replay data.

### Installation
```
pip install EARL-pytorch
```

### Example
```python
from earl_pytorch import EARL
from earl_pytorch.model import EARLReplayModel
from earl_pytorch.dataset.create_dataset import replay_to_dfs, convert_dfs

earl = EARL()

# Option 1: Open some replay file
model = EARLReplayModel(earl)
dfs = replay_to_dfs("2627e02a-aa46-4e13-b66b-b76a32069a07.replay", )
x_data, y_data = convert_dfs(dfs, tensors=True)

for epoch in range(n_epochs):
    ...  # Train the model
```
