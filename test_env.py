from gym_puyopuyo import register
from gym import make

register()

small_env = make("PuyoPuyoEndlessSmall-v2")

for i in range(10):
    small_env.step(small_env.action_space.sample())

small_env.render()