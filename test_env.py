import gym
import gym_puyopuyo

# Create the Puyo Puyo environment
env = gym.make('PuyoPuyoEndlessNormal-v2')

# Reset the environment to the initial state
state = env.reset()

# Take a random action
action = env.action_space.sample()

# Step through the environment with the chosen action
next_state, reward, done, info = env.step(action)

print(f"Next State: {next_state}")
print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Info: {info}")

env.close()
