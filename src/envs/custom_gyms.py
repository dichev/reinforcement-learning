from gymnasium.envs.registration import register

register(
    id='custom/FrozenLake_OneHot_DiscountedReward',
    entry_point='src.envs.FrozenLake:make__FrozenLake_OneHot_DiscountedReward'
)

register(
    id='custom/MiniGrid',
    entry_point='src.envs.MiniGrid:make__MiniGrid',
    kwargs={'fully_observable': False}
)

register(
    id='custom/TinyGrid',
    entry_point='src.envs.TinyGrid:make_TinyGrid',
    kwargs={'template': 'static_9x6', 'noise': 0}
)

register(
    id='custom/TinyRandomGridNoisy',
    entry_point='src.envs.TinyGrid:make_TinyGrid',
    kwargs={'template': 'random_4x4', 'noise': 0.01}
)


register(
    id='custom/Pong',
    entry_point='src.envs.atari:make_atari',
    kwargs={'id': 'ALE/Pong-v5'}
)

register(
    id='custom/Freeway',
    entry_point='src.envs.atari:make_atari',
    kwargs={'id': 'ALE/Freeway-v5', 'fire_reset': False}
)

register(
    id='custom/SpaceInvaders',
    entry_point='src.envs.atari:make_atari',
    kwargs={'id': 'ALE/SpaceInvaders-v5'}
)

register(
    id="custom/DoomMaze",
    entry_point="src.envs.Doom:make_doom"
)