# EARL
EARL - Extensible Attention-based Rocket League model

Uses a transformer-like architecture to support any number of in-game entities (balls, boosts, players).
Can be used for in-game event prediction (`EARLReplayModel`), action prediction (`EARLActorCritic`), rank prediction (`EARLRankModel`) etc.

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
