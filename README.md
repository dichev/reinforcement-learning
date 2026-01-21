# Reinforcement Learning

### Setup local env
```
uv sync
uv run AutoROM --accept-license
```

### Train an RL agent
```
source .venv/Scripts/activate

python -m agents.values.categorical_dqn__atari
python -m agents.values.dqn_attention__minigrid
python -m agents.values.dqn__atari
python -m agents.values.dqn__grid_world

python -m agents.policy.a2c__cart_pole
python -m agents.policy.cross-entropy-policy__cart_pole
python -m agents.policy.reinforce_baseline__cart_pole
python -m agents.policy.reinforce__cart_pole

python -m agents.tabular.q_learning__multi_arm_bandit
python -m agents.tabular.q_learning__multi_arm_contextual_bandit
python -m agents.tabular.td_learning__cliff_walking_grid
python -m agents.tabular.v_learning__frozen_lake

python -m agents.planning.dyna_q_planning__maze
python -m agents.planning.dyna_q_plus_planning__shortcut_maze
python -m agents.planning.mcts__play_go
python -m agents.planning.rollout__play_go
```

### Run tests
```
MPLBACKEND=Agg  pytest ./test
```