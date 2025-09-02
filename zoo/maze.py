from flax import struct
import jax.numpy as jnp
from jax import random, vmap
import matplotlib.pyplot as plt
import os

from rl import Env, EnvParams, EnvState

@struct.dataclass
class MazeParams(EnvParams):
    num_agents: int         # Number of agents in the environment
    num_particles: int      # Number of particles that each agent controls
    grid: jnp.ndarray       # The binary occupancy map of the env (0 means obstacle, 1 is empty cell)
    grid_shape: jnp.ndarray # The shape of the 2D occupancy grid, Shape (2), dtype=int
    goal_point: jnp.ndarray # The index of goal grid cell, Shape (2,), dtype=int
    num_steps: int          # Time limit for the episode in steps
    available_positions: jnp.ndarray # (Required to make JIT possible) Precomputed list of available positions in the grid
    dt: float               # Used for rendering only
    
    def __hash__(self):
        return hash((
            self.num_agents,
            self.num_particles,
            # self.grid.tobytes(), # Don't include arrays in the hash
            # self.grid_shape.tobytes(), # Don't include arrays in the hash
            # self.goal_point.tobytes(), # Don't include arrays in the hash
            # self.num_steps,
            # self.available_positions.tobytes(), # Don't include arrays in the hash
            self.dt,
        ))
    
    def __eq__(self, other):
        return (
            self.num_agents == other.num_agents and
            self.num_particles == other.num_particles and
            jnp.array_equal(self.grid_shape, other.grid_shape) and
            jnp.array_equal(self.goal_point, other.goal_point) and
            jnp.isclose(self.dt, other.dt)
        )

@struct.dataclass
class MazeState(EnvState):
    timestep: jnp.ndarray   # Current timestep, Shape: (num_agents,), dtype=int
    positions: jnp.ndarray  # (row, col) indices within the grid, Shape: (num_agents, num_particles, 2), dtype=int

class MazeEnv(Env):
    @classmethod
    def reset(cls, key: jnp.ndarray, params: MazeParams):
        # Randomly sample positions for each agent's particles
        keys = random.split(key, params.num_agents)
        positions = vmap(
            lambda k: random.choice(
                k, params.available_positions, shape=(params.num_particles,), replace=False
            )
        )(keys)

        state = MazeState(positions=positions, timestep=jnp.zeros(shape=params.num_agents, dtype=jnp.int32))
        obs = cls.observation(state, params)

        return obs, state

    @classmethod
    def step(cls, key: jnp.ndarray, state: MazeState, action: jnp.ndarray, params: MazeParams):
        next_state = cls.maze_step(state, action, params)
        obs = cls.observation(next_state, params)
        terminated = cls.terminated(next_state, params)
        reward = cls.reward(next_state, params)
        return obs, next_state, reward, terminated, {'t': next_state.timestep, 'reward': reward, 'done': terminated}

    @staticmethod
    def action_space(params: MazeParams):
        return jnp.array([0, 0, 0, 0]), jnp.array([1, 1, 1, 1])  # Four discrete actions: up, down, left, right

    @staticmethod
    def observation_space(params: MazeParams):
        low =  jnp.zeros(shape=params.grid_shape + (1,), dtype=jnp.float32) # Single channel grid
        high = jnp.ones(shape=params.grid_shape + (1,), dtype=jnp.float32) # Single channel grid
        return low, high

    @staticmethod
    def maze_step(state: MazeState, action: jnp.ndarray, params: MazeParams):
        # Actions: 0=left, 1=right, 2=up, 3=down in (row, col)
        deltas = jnp.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        move = deltas[action]  # Shape: (num_agents, 2)
        candidate = state.positions + move[:, None, :]        # broadcast to all particles
        candidate = jnp.clip(candidate, 0, jnp.array(params.grid_shape) - 1)

        # Check target cell is free (grid==1)
        free = params.grid[candidate[..., 0], candidate[..., 1]] == 1  # (num_agents, num_particles)

        new_positions = jnp.where(free[..., None], candidate, state.positions)

        return MazeState(positions=new_positions, timestep=state.timestep + 1)

    @staticmethod
    def reward(state: MazeState, params: MazeParams) -> jnp.ndarray:
        diagonal = jnp.linalg.norm(jnp.array(params.grid_shape)) # Max possible distance in the grid
        return (
            0.5 - jnp.linalg.norm(state.positions - params.goal_point, axis=-1).mean(axis=-1) / diagonal
        )

    @staticmethod
    def terminated(state: MazeState, params: MazeParams) -> jnp.ndarray:
        return jnp.full((state.positions.shape[0],), False)

    @staticmethod
    def observation(state: MazeState, params: MazeParams) -> jnp.ndarray:
        def mark_positions(positions):
            return jnp.zeros(params.grid_shape).reshape(*params.grid_shape, 1).at[positions[:, 0], positions[:, 1], 0].set(1.0)
        obs = vmap(mark_positions)(state.positions)
        return obs

    @staticmethod
    def render(state: MazeState, params: MazeParams):
        # Convert grid to RGB: black for obstacles, white for background
        base = params.grid.astype(jnp.float32)
        img = jnp.stack([base, base, base], axis=-1)  # (H, W, 3)

        # Add green particles for each agent
        for i, agent_positions in enumerate(state.positions):
            cmap = plt.cm.get_cmap('hsv', len(state.positions) + 1)  # Get a colormap
            color = cmap(i)[:3]  # Extract RGB values for the agent
            img = img.at[agent_positions[:, 0], agent_positions[:, 1]].set(color)  # Set to agent-specific color

        # Add red goal point
        img = img.at[params.goal_point[0], params.goal_point[1], 0].set(1.0)  # Red channel
        
        # Add text info
        plt.text(0, -2, f"Timestep: {state.timestep[0]}", color='black', fontsize=12)
        
        plt.imshow(img)
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.title('Maze')

    @staticmethod
    def make_params(
        num_agents: int = 1,
        num_particles: int = 16,
        maze_path: str = os.path.join(os.path.dirname(__file__), 'assets/maze32.png'),
        goal_point = jnp.array([0,0]),
        num_steps: int = 200,
    ) -> MazeParams:
        image = plt.imread(maze_path)
        grayscale = jnp.mean(image, axis=-1)
        grid = (grayscale > 0.5).astype(jnp.int32)  # Threshold at 0.5
        grid_shape = grid.shape
        available_positions = jnp.argwhere(grid == 1)

        return MazeParams(
            num_agents=num_agents,
            num_particles=num_particles,
            grid=grid,
            grid_shape=grid_shape,
            goal_point=goal_point,
            num_steps=num_steps,
            available_positions=available_positions,
            dt=0.001,
        )
