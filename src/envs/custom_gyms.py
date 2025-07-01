from gymnasium.envs.registration import register

register(
    id='custom/FrozenLake_OneHot_DiscountedReward',
    entry_point='src.envs.FrozenLake_Custom:make__FrozenLake_OneHot_DiscountedReward'
)

register(
    id='custom/MiniGrid',
    entry_point='src.envs.MiniGrid:make__MiniGrid'
)

register(
    id='custom/GridWorldGym',
    entry_point='src.envs.GridWorldGym:GridWorldGym'
)

register(
    id='custom/Pong',
    entry_point='src.envs.external.atari_wrappers_openai:make_pong'
)
