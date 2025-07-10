import gym
from gym_puyopuyo import register
from stable_baselines3 import PPO, DQN
import flatten

register()

# Load the trained model
model = DQN.load("sqn_puyo_final")

# Create the environment
env = gym.make("PuyoPuyoEndlessLarge-v2")
env = flatten.FlattenObservationEndless(env)

# Run a few episodes
for episode in range(5):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        # Get action from the model
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
