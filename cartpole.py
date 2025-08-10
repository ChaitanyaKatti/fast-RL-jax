from flax import struct
import jax.numpy as jnp
from jax import random, vmap
from env import Env, EnvParams, EnvState
import matplotlib.pyplot as plt

@struct.dataclass
class CartPoleParams(EnvParams):
    num_agents: int     # Number of agents in the environment
    m_c: float          # Mass of the cart
    m_p: float          # Mass of the pole
    half_length: float  # Half the length of the pole
    g: float            # Acceleration due to gravity
    mu_c: float         # Coefficient of friction for the cart
    mu_p: float         # Coefficient of friction for the pole
    force_mag: float    # Magnitude of the force applied to the cart
    dt: float           # Time step for integration
    x_threshold: float  # Position at which the episode terminates
    num_steps: int      # Time limit for the episode in steps

@struct.dataclass
class CartPoleState(EnvState):
    physics: jnp.ndarray        # shape: (num_agents, 5) # [x, x_dot, theta, theta_dot, t]

class CartPoleEnv(Env):
    @classmethod
    def reset(cls, key: jnp.ndarray, params: CartPoleParams):
        low =  jnp.array([-params.x_threshold/2, -0.5, -jnp.pi, -0.5, 0.0])
        high = jnp.array([ params.x_threshold/2,  0.5,  jnp.pi,  0.5, 0.0])

        physics = vmap(
            lambda k: random.uniform(k, shape=(5), minval=low, maxval=high)
        )(key)

        state = CartPoleState(physics=physics)
        obs = cls.observation(state, params)

        return obs, state

    @classmethod
    def step(cls, key: jnp.ndarray, state: CartPoleState, action: jnp.ndarray, params: CartPoleParams):
        next_state = cls.cartpole_step(state, action, params)
        obs = cls.observation(next_state, params)
        terminated = cls.terminated(next_state, params)
        reward = cls.reward(next_state, params)
        return obs, next_state, reward, terminated, {'t': next_state.physics[:, -1], 'reward': reward, 'done': terminated}

    @staticmethod
    def action_space(params: CartPoleParams):
        return jnp.array([-1.0]), jnp.array([1.0])

    @staticmethod
    def observation_space(params: CartPoleParams):
        # Observation space: [x, x_dot, theta, theta_dot, t]
        low =  jnp.array([-params.x_threshold, -jnp.inf, -jnp.pi, -jnp.inf])
        high = jnp.array([ params.x_threshold,  jnp.inf,  jnp.pi,  jnp.inf])
        return low, high

    @staticmethod
    def cartpole_step(state: CartPoleState, action: jnp.ndarray, params: CartPoleParams):
        x, x_dot, theta, theta_dot, t = state.physics.T
        action = jnp.clip(action, -1.0, 1.0)
        force = (params.force_mag * action).reshape(params.num_agents, )

        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        num1 = params.m_p * params.g * costheta * sintheta
        num2 = (7/3) * (force + params.m_p * params.half_length * jnp.square(theta_dot) * sintheta - params.mu_c * x_dot)
        num3 = params.mu_p * theta_dot / params.half_length
        denom = params.m_p * costheta**2 - (7/3) * (params.m_c + params.m_p)

        # Update equations
        x_dot_dot = (num1 - num2 - num3) / denom
        theta_dot_dot = (3 / (7 * params.half_length)) * (
            params.g * sintheta - x_dot_dot * costheta - params.mu_p * theta_dot / (params.m_p * params.half_length)
        )

        # Euler Update
        dx = jnp.stack([x_dot, x_dot_dot, theta_dot, theta_dot_dot, jnp.ones_like(x)], axis=-1)
        next_physics = state.physics + dx * params.dt

        # Bound the angles to [-pi, pi]
        next_physics = next_physics.at[:, 2].set(jnp.mod(next_physics[:, 2] + jnp.pi, 2 * jnp.pi) - jnp.pi)

        # Clip the position, Zero the velocity, if out of bounds
        # out_of_bounds = jnp.abs(next_physics[:, 0]) > params.x_threshold
        # x = jnp.clip(next_physics[:, 0], -params.x_threshold, params.x_threshold)
        # v = jnp.where(out_of_bounds, 0.0, next_physics[:, 1])
        # next_physics = next_physics.at[:, 0].set(x)
        # next_physics = next_physics.at[:, 1].set(v)

        return CartPoleState(physics=next_physics)

    @staticmethod
    def reward(state: CartPoleState, params: CartPoleParams) -> jnp.ndarray:
        return (
            1.0 # base reward for each step
            - (jnp.abs(state.physics[:, 0]) / params.x_threshold) #  position based reward
            - (jnp.abs(state.physics[:, 2]) / jnp.pi) # angle based reward
        )

    @staticmethod
    def terminated(state: CartPoleState, params: CartPoleParams) -> jnp.ndarray:
        return jnp.greater(jnp.abs(state.physics[:, 0]), params.x_threshold)

    @staticmethod
    def observation(state: CartPoleState, params: CartPoleParams) -> jnp.ndarray:
        return state.physics[:, :-1]  # Exclude the time step from the observation

    @staticmethod
    def render(state: CartPoleState, params: CartPoleParams) -> None:
        # Draw the cart
        xs, thetas = state.physics[:, 0], state.physics[:, 2]
        cart_width = 0.4
        cart_height = 0.2

        for i in range(params.num_agents):
            cart_x = xs[i]
            cart_y = 0.0  # y position of the cart
            # Draw cart as a box centered at (x, 0)
            plt.gca().add_patch(
                plt.Rectangle(
                    (cart_x - cart_width / 2, cart_y - cart_height / 2),
                    cart_width,
                    cart_height,
                    color="blue",
                    alpha=0.6,
                )
            )

            # Draw pole as a line from cart center at (x, 0) at angle theta
            pole_length = params.half_length * 2
            pole_x_end = cart_x + pole_length * jnp.sin(thetas[i])
            pole_y_end = cart_y + pole_length * jnp.cos(thetas[i])
            plt.plot(
                [cart_x, pole_x_end],
                [cart_y, pole_y_end],
                color="red",
                linewidth=3,
                alpha=0.6,
            )

            # Draw axle
            plt.plot(cart_x, cart_y, 'ko', markersize=6, alpha=0.6)

        plt.xlim(-4, 4)
        plt.ylim(-1.5, 1.5)
        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('CartPole')
        plt.grid(alpha=0.3)

    @staticmethod
    def makeParams(
        num_agents: int = 1,
        m_c: float = 1.0,
        m_p: float = 0.1,
        half_length: float = 0.5,
        g: float = 9.8,
        mu_c: float = 0.01,
        mu_p: float = 0.001,
        force_mag: float = 5.0,
        dt: float = 0.05,
        x_threshold: float = 3.0,
        num_steps: int = 100,
    ) -> CartPoleParams:
        """
        Factory function to create CartPoleParams with a specified arguments.
        Returns:
            CartPoleParams instance with the specified number of agents.
        """
        return CartPoleParams(
            num_agents=num_agents,
            m_c=m_c,
            m_p=m_p,
            half_length=half_length,
            g=g,
            mu_c=mu_c,
            mu_p=mu_p,
            force_mag=force_mag,
            dt=dt,
            x_threshold=x_threshold,
            num_steps=num_steps,
        )
