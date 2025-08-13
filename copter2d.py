from flax import struct
import jax.numpy as jnp
from jax import random, vmap
from env import Env, EnvParams, EnvState
import matplotlib.pyplot as plt

@struct.dataclass
class Copter2DParams(EnvParams):
    num_agents: int     # Number of agents in the environment
    mass: float         # Mass of the drone
    inertia: float      # Inertia of the drone
    arm_length: float   # length of drone arm
    g: float            # Acceleration due to gravity
    thurst_mag: float   # Magnitude of the thrust applied to the drone
    torque_mag: float   # Magnitude of the thrust applied to the drone
    dt: float           # Time step for integration
    x_threshold: float  # Position at which the episode terminates
    num_steps: int      # Time limit for the episode in steps

@struct.dataclass
class Copter2DState(EnvState):
    physics: jnp.ndarray        # shape: (num_agents, 7) # [x, y, x_dot, y_dot, theta, theta_dot, t]

class Copter2DEnv(Env):
    @classmethod
    def reset(cls, key: jnp.ndarray, params: Copter2DParams):
        low =  jnp.array([-params.x_threshold/2, -params.x_threshold/2, -0.5, -0.5, -jnp.pi/2, -0.5, 0.0])
        high = jnp.array([ params.x_threshold/2,  params.x_threshold/2,  0.5,  0.5,  jnp.pi/2,  0.5, 0.0])
        
        keys = random.split(key, params.num_agents)
        physics = vmap(
            lambda k: random.uniform(k, shape=(7), minval=low, maxval=high)
        )(keys)

        state = Copter2DState(physics=physics)
        obs = cls.observation(state, params)

        return obs, state

    @classmethod
    def step(cls, key: jnp.ndarray, state: Copter2DState, action: jnp.ndarray, params: Copter2DParams):
        next_state = cls.copter2D_step(state, action, params)
        obs = cls.observation(next_state, params)
        terminated = cls.terminated(next_state, params)
        reward = cls.reward(next_state, params)
        return obs, next_state, reward, terminated, {'t': next_state.physics[:, -1], 'reward': reward, 'done': terminated}

    @staticmethod
    def action_space(params: Copter2DParams):
        return jnp.array([0.0, -1.0]), jnp.array([1.0, 1.0])

    @staticmethod
    def observation_space(params: Copter2DParams):
        # Observation space:
        #                [                  x,                   y,    x_dot,    y_dot,   theta, theta_dot]
        low =  jnp.array([-params.x_threshold, -params.x_threshold, -jnp.inf, -jnp.inf, -jnp.pi, -jnp.inf])
        high = jnp.array([ params.x_threshold,  params.x_threshold,  jnp.inf,  jnp.inf,  jnp.pi,  jnp.inf])
        return low, high

    @staticmethod
    def copter2D_step(state: Copter2DState, action: jnp.ndarray, params: Copter2DParams):
        x, y, x_dot, y_dot, theta, theta_dot, t = state.physics.T
        action = jnp.clip(action, -1.0, 1.0)
        thrust = (0.5*action[:, 0] + 0.5) * params.thurst_mag
        torque = action[:, 1] * params.torque_mag

        # Calculate forces
        x_dot_dot = -thrust * jnp.sin(theta) / params.mass
        y_dot_dot = thrust * jnp.cos(theta) / params.mass - params.g

        theta_dot_dot = torque / params.inertia
        x_dot += x_dot_dot * params.dt
        y_dot += y_dot_dot * params.dt
        theta_dot += theta_dot_dot * params.dt

        # Euler Update
        dx = jnp.stack([x_dot, y_dot, x_dot_dot, y_dot_dot, theta_dot, theta_dot_dot, jnp.ones_like(x)], axis=-1)
        next_physics = state.physics + dx * params.dt

        # Bound the angles to [-pi, pi]
        next_physics = next_physics.at[:, 4].set(jnp.mod(next_physics[:, 4] + jnp.pi, 2 * jnp.pi) - jnp.pi)

        # Clip the position, Zero the velocity, if out of bounds
        # out_of_bounds = jnp.logical_or(
        #     jnp.abs(state.physics[:, 0]) > params.x_threshold,  # x position out of bounds
        #     jnp.abs(state.physics[:, 1]) > params.x_threshold,  # y position out of bounds
        # )
        # x = jnp.clip(next_physics[:, 0], -params.x_threshold, params.x_threshold)
        # y = jnp.clip(next_physics[:, 1], -params.x_threshold, params.x_threshold)
        # vx = jnp.where(out_of_bounds, 0.0, next_physics[:, 2])
        # vy = jnp.where(out_of_bounds, 0.0, next_physics[:, 3])
        # next_physics = next_physics.at[:, 0].set(x)
        # next_physics = next_physics.at[:, 1].set(y)
        # next_physics = next_physics.at[:, 2].set(vx)
        # next_physics = next_physics.at[:, 3].set(vy)

        return Copter2DState(physics=next_physics)

    @staticmethod
    def reward(state: Copter2DState, params: Copter2DParams) -> jnp.ndarray:
        return (
            1.0 # base reward for each step
            - (jnp.abs(state.physics[:, 0]) / params.x_threshold)   # x position based reward
            - (jnp.abs(state.physics[:, 1]) / params.x_threshold)   # y position based reward
            - (jnp.abs(state.physics[:, 4]) / jnp.pi / 2)           # angle based reward
        )

    @staticmethod
    def terminated(state: Copter2DState, params: Copter2DParams) -> jnp.ndarray:
        return jnp.logical_or(
            jnp.abs(state.physics[:, 0]) > params.x_threshold,  # x position out of bounds
            jnp.abs(state.physics[:, 1]) > params.x_threshold,  # y position out of bounds
        )

    @staticmethod
    def observation(state: Copter2DState, params: Copter2DParams) -> jnp.ndarray:
        return state.physics[:, :-1]  # Exclude the time step from the observation

    @staticmethod
    def render(state: Copter2DState, params: Copter2DParams):
        xs, ys, thetas = state.physics[:, 0], state.physics[:, 1], state.physics[:, 4]

        # Plot the drone position
        for x, y, theta in zip(xs, ys, thetas):
            cosl = params.arm_length* jnp.cos(theta)
            sinl = params.arm_length* jnp.sin(theta)
            # Drone body
            plt.plot(x, y, 'ro', markersize=15, alpha=0.6)
            # Arms
            plt.plot([x - cosl, x + cosl], [y - sinl, y + sinl], 'b-', linewidth=2, alpha=0.6)
            # Motors
            plt.plot(x - cosl, y - sinl, "bo", markersize=5, alpha=0.6)
            plt.plot(x + cosl, y + sinl, "bo", markersize=5, alpha=0.6)
            # Vertial
            plt.plot([x, x - sinl], [y, y + cosl], 'g-', linewidth=2, alpha=0.6)

        plt.xlim(-params.x_threshold, params.x_threshold)
        plt.ylim(-params.x_threshold, params.x_threshold)
        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Copter2D')
        plt.grid(alpha=0.3)

    @staticmethod
    def make_params(
        num_agents: int = 1,
        mass: float = 1.0,
        inertia: float = 1.0,
        arm_length: float = 0.5,
        g: float = 9.8,
        thurst_mag: float = 20.0,
        torque_mag: float = 10.0,
        dt: float = 0.05,
        x_threshold: float = 3.0,
        num_steps: int = 100,
    ) -> Copter2DParams:
        return Copter2DParams(
            num_agents=num_agents,
            mass=mass,
            inertia=inertia,
            arm_length=arm_length,
            g=g,
            thurst_mag=thurst_mag,
            torque_mag=torque_mag,
            dt=dt,
            x_threshold=x_threshold,
            num_steps=num_steps,
        )
