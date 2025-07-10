from gym_puyopuyo import register
import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import flatten

register()

# Set load address
load_address = ""

# Create the Puyo Puyo environment
env = gym.make("PuyoPuyoEndlessLarge-v2")
env = flatten.FlattenObservationEndless(env)

# Print the shape of a single observation after flattening
observation = env.reset()
print("Flattened observation shape:", observation.shape)

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

# Define the model
if load_address:
    model = DQN.load(load_address)
else:
    model = DQN("MlpPolicy", env, verbose=2)

# Create a checkpoint callback to save the model periodically
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='./models/', name_prefix='dqn_puyo')

def callback(_locals, _globals):
    # Log the current step
    #print("Step:", _locals['self'].num_timesteps)
    return True


print("Training model...")

# Train the model
model.learn(total_timesteps=50000, callback=callback)
print("Training complete.")

# Save the final model
model.save("dqn_puyo_final")

# Close the environment
env.close()
