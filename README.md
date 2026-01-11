# Reinforcement Learning

### Setup local env
```
uv sync
uv run AutoROM --accept-license
```

### Train an RL agent
```
source .venv/Scripts/activate

python -m learn-values.categorical_dqn__atari
python -m learn-values.dqn_attention__minigrid
python -m learn-values.dqn__atari
python -m learn-values.dqn__grid_world

python -m learn-policy.a2c__cart_pole
python -m learn-policy.cross-entropy-policy__cart_pole
python -m learn-policy.reinforce_baseline__cart_pole
python -m learn-policy.reinforce__cart_pole

python -m learn-tabular.dyna_q_planning__maze
python -m learn-tabular.dyna_q_plus_planning__shortcut_maze
python -m learn-tabular.q_learning__multi_arm_bandit
python -m learn-tabular.q_learning__multi_arm_contextual_bandit
python -m learn-tabular.td_learning__cliff_walking_grid
python -m learn-tabular.v_learning__frozen_lake
```

### Run tests
```
MPLBACKEND=Agg  pytest ./test
```