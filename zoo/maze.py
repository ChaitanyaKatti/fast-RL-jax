from flax import struct
import jax
import jax.numpy as jnp
from jax import random, vmap
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

from rl import Env, EnvParams, EnvState

@struct.dataclass
class MazeParams(EnvParams):
    num_agents: int         # Number of agents in the environment
    num_particles: int      # Number of particles that each agent controls
    grid: jnp.ndarray       # The binary occupancy map of the env (0 means obstacle, 1 is empty cell), Shape (*grid_shape), dtype=bool
    grid_shape: jnp.ndarray # The shape of the 2D occupancy grid, Shape (2), dtype=int
    goal_point: jnp.ndarray # The index of goal grid cell, Shape (2,), dtype=int
    num_steps: int          # Time limit for the episode in steps
    available_positions: jnp.ndarray # (Required to make JIT possible) Precomputed list of available positions in the grid
    dt: float               # Used for rendering only
    max_dist: float         # Maximum possible distance in the grid, used for normalizing the reward
    dist_grid: jnp.ndarray  # Precomputed distance grid from each cell to the goal, used for reward calculation

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
    occupancy: jnp.ndarray  # Occupancy Map of particles, Shape: (num_agents, *grid_shape), dtype=bool

class MazeEnv(Env):
    @classmethod
    def reset(cls, key: jnp.ndarray, params: MazeParams):
        # Randomly sample positions for each agent's particles
        keys = random.split(key, params.num_agents)
        occupancy = jnp.full((params.grid_shape), False)

        def random_occupancy(k):
            indices = random.choice(
                k, params.available_positions, shape=(params.num_particles,), replace=False
            )
            return occupancy.at[indices[:, 0], indices[:, 1]].set(True)

        occupancy = vmap(random_occupancy)(keys)
        state = MazeState(
            occupancy=occupancy,
            timestep=jnp.zeros(shape=params.num_agents, dtype=jnp.int32),
        )
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
        low =  jnp.zeros(shape=params.grid_shape + (2,), dtype=jnp.float32) # 1st channel grid, 2nd channel occupancy
        high = jnp.ones(shape=params.grid_shape + (2,), dtype=jnp.float32) # 1st channel grid, 2nd channel occupancy
        return low, high

    @staticmethod
    def maze_step(state: MazeState, action: jnp.ndarray, params: MazeParams):
        H, W = params.grid_shape

        def step_dir(i, occ_grid):
            occ, grid = occ_grid
            free = (~occ[i - 1]) & grid[i - 1]
            occ = occ.at[i - 1].set((occ[i] & free) | occ[i - 1])
            occ = occ.at[i].set(occ[i] & ~free)
            return (occ, grid)

        def move_with_scan(occ, grid, size, flip_axis=None, transpose=False):
            if transpose:
                occ, grid = occ.T, grid.T
            if flip_axis is not None:
                occ, grid = jnp.flip(occ, axis=flip_axis), jnp.flip(grid, axis=flip_axis)

            occ, _ = jax.lax.fori_loop(1, size, step_dir, (occ, grid))

            if flip_axis is not None:
                occ = jnp.flip(occ, axis=flip_axis)
            if transpose:
                occ = occ.T
            return occ

        def move_particles(occ, act):
            return jax.lax.switch(
                act,
                [
                    lambda occ: move_with_scan(occ, params.grid, W, transpose=True),          # left
                    lambda occ: move_with_scan(occ, params.grid, W, flip_axis=0, transpose=True),  # right
                    lambda occ: move_with_scan(occ, params.grid, H),                          # up
                    lambda occ: move_with_scan(occ, params.grid, H, flip_axis=0),             # down
                ],
                occ,
            )

        occupancy = vmap(move_particles)(state.occupancy, action)
        return MazeState(occupancy=occupancy, timestep=state.timestep + 1)

    @staticmethod
    def reward(state: MazeState, params: MazeParams) -> jnp.ndarray:
        return 0.5 - jnp.max(state.occupancy * params.dist_grid, axis=(-1,-2)) / (params.max_dist)

    @staticmethod
    def terminated(state: MazeState, params: MazeParams) -> jnp.ndarray:
        return jnp.full(params.num_agents, False) # No terminal state, episodes end after fixed number of steps

    @staticmethod
    def observation(state: MazeState, params: MazeParams) -> jnp.ndarray:
        # Duplicate params.grid for each agent and stack with occupancy
        grid = jnp.broadcast_to(params.grid, (params.num_agents,) + params.grid.shape) # Duplicate grid for each agent, Shape: (num_agents, H, W)
        grid = grid[..., jnp.newaxis].astype(jnp.float32) # Add channel dim, Shape: (num_agents, H, W, 1)
        occupancy = state.occupancy[..., jnp.newaxis].astype(jnp.float32) # Add channel dim, Shape: (num_agents, H, W, 1)
        return jnp.concatenate([grid, occupancy], axis=-1)  # Shape: (num_agents, H, W, 2)

    @staticmethod
    def make_renderer():
        window_name = "MazeEnv"
        closed = {"flag": False}

        def renderer(state: MazeState, params: MazeParams) -> bool:
            if closed["flag"]:
                return False

            base = params.grid.astype(jnp.float32)

            @jax.jit
            def render_single(occ):
                img = jnp.stack([base, base, base], axis=-1)  # RGB background
                occ = occ.astype(jnp.float32).reshape(params.grid_shape + (1,))
                img = img * (1 - occ) + occ * jnp.array([0.0, 1.0, 0.0])  # mark agent green
                img = img.at[params.goal_point[0], params.goal_point[1]].set(jnp.array([1.0, 0.0, 0.0]))  # goal red
                return img

            plot_grid_size = int(jnp.sqrt(params.num_agents))
            images = vmap(render_single)(state.occupancy[:plot_grid_size**2])
            images = jnp.clip(images, 0.0, 1.0)

            img_size = params.grid_shape[0] * plot_grid_size
            final = jnp.ones((img_size, img_size, 3), dtype=jnp.float32)

            for i, img in enumerate(images):
                row = i // plot_grid_size
                col = i % plot_grid_size
                final = final.at[
                    row * params.grid_shape[0]:(row + 1) * params.grid_shape[0],
                    col * params.grid_shape[1]:(col + 1) * params.grid_shape[1],
                    :
                ].set(img)

            final = np.array((final * 255).astype(jnp.uint8))
            final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
            final = cv2.resize(final, (720, 720), interpolation=cv2.INTER_NEAREST)
            cv2.putText(final, f"Step: {int(state.timestep[0])}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

            cv2.imshow(window_name, final)

            # check if user closed the window
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                closed["flag"] = True
                return False

            # keep UI responsive
            if cv2.waitKey(int(1000 * params.dt)) & 0xFF == 27:  # ESC closes
                closed["flag"] = True
                return False

            return True

        return renderer


    @staticmethod
    def make_params(
        maze_path: str,
        num_agents: int = 1,
        num_particles: int = 16,
        goal_point: tuple = (0,0),
        num_steps: int = 200,
        dt=0.001
    ) -> MazeParams:
        image = plt.imread(maze_path)
        grayscale = jnp.mean(image, axis=-1)
        grid = (grayscale > 0.5) # Threshold at 0.5
        grid_shape = grid.shape
        goal_point = jnp.array(goal_point, dtype=jnp.int32)
        available_positions = jnp.argwhere(grid)

        H, W = grid.shape
        rows = jnp.arange(H)[:, None] # Shape (H, 1)
        cols = jnp.arange(W)[None, :] # Shape (1, W)
        dist_grid = jnp.sqrt((rows - goal_point[0])**2 + (cols - goal_point[1])**2) # Shape (H, W)
        max_dist = jnp.linalg.norm(jnp.array(grid.shape))

        return MazeParams(
            num_agents=num_agents,
            num_particles=num_particles,
            grid=grid,
            grid_shape=grid_shape,
            goal_point=goal_point,
            num_steps=num_steps,
            available_positions=available_positions,
            dist_grid=dist_grid,
            max_dist=max_dist,
            dt=dt,
        )
