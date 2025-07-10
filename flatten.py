import gym
from gym.spaces import Box
import numpy as np

class FlattenObservationEndless(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Calculate the total size of the flattened observation space
        flattened_shape = sum([np.prod(space.shape) for space in env.observation_space.spaces])
        self.observation_space = Box(low=0.0, high=1.0, shape=(flattened_shape,), dtype=np.float32)

    def observation(self, obs):
        # Flatten and concatenate all parts of the observation tuple, and convert to float32
        return np.concatenate([np.ravel(o).astype(np.float32) for o in obs])

class FlattenObservationVersus(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Get the observation space for both players
        self.player1_space = env.observation_space.spaces[0]
        self.player2_space = env.observation_space.spaces[1]
        
        # Calculate the flattened shape for both players
        flattened_shape1 = self._calculate_flattened_shape(self.player1_space)
        flattened_shape2 = self._calculate_flattened_shape(self.player2_space)
        
        # Set the new observation space as a single flattened Box
        self.observation_space = Box(
            low=0.0, 
            high=1.0, 
            shape=(flattened_shape1 + flattened_shape2,), 
            dtype=np.float32
        )

    def _calculate_flattened_shape(self, space_dict):
        """Helper function to calculate the flattened shape of a Dict space."""
        shape = 0
        for key, space in space_dict.spaces.items():
            if isinstance(space, Box):
                shape += np.prod(space.shape)
            elif isinstance(space, gym.spaces.Discrete):
                shape += 1
        return shape

    def _flatten_dict(self, obs_dict):
        """Helper function to flatten a dictionary of observations."""
        flattened_obs = []
        for key, value in obs_dict.items():
            if isinstance(value, np.ndarray):
                flattened_obs.append(value.ravel().astype(np.float32))
            else:  # Handle Discrete spaces
                flattened_obs.append(np.array([value], dtype=np.float32))
        return np.concatenate(flattened_obs)

    def observation(self, obs):
        # Flatten both players' observations
        flattened_obs1 = self._flatten_dict(obs[0])
        flattened_obs2 = self._flatten_dict(obs[1])
        
        # Concatenate both flattened observations into a single array
        return np.concatenate([flattened_obs1, flattened_obs2])
