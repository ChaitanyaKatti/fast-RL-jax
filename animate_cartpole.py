import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from cartpole import CartPoleEnv, CartPoleParams
from matplotlib.animation import FuncAnimation
from pynput.keyboard import Key, Listener, Controller 
from pynput import keyboard
from network import ActorCritic

def on_press(key, injected):
    global action
    try:
        if key == keyboard.Key.left:
            action = -1  # Move left
        elif key == keyboard.Key.right:
            action = 1   # Move right
        else:
            action = 0   # No action
    except Exception as e:
        print(f"Error: {e}")


def plot(xs, thetas, params):
    num_agents = xs.shape[0]
    
    for i in range(num_agents):
        # Draw cart as a box centered at (x, 0)
        cart_width, cart_height = 0.4, 0.2
        cart_y = 0.0
        plt.gca().add_patch(
            plt.Rectangle((xs[i] - cart_width/2, cart_y - cart_height/2), cart_width, cart_height, color='blue', alpha=0.6)
        )

        # Draw pole as a line from cart center at (x, 0) at angle theta
        pole_length = params.half_length * 2
        pole_x_end = xs[i] + pole_length * jnp.sin(thetas[i])
        pole_y_end = cart_y + pole_length * jnp.cos(thetas[i])
        plt.plot([xs[i], pole_x_end], [cart_y, pole_y_end], color='red', linewidth=3, alpha=0.6)

        # Draw axle
        plt.plot(xs[i], cart_y, 'ko', markersize=6)

    plt.xlim(-4, 4)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('CartPole')
    plt.show()
    
if __name__ == "__main__":
    jax.default_device(jax.devices('cpu')[0])

    listener = keyboard.Listener(on_press=on_press)
    action = 0  # Default action
    listener.start()

    env = CartPoleEnv()
    env_params = CartPoleParams(num_agents=10)
    network = ActorCritic(action_dim=env.action_space(env_params)[0].shape[0])
    rng = jax.random.PRNGKey(0)
    init_x = jnp.zeros(env.observation_space(env_params)[0].shape[0])
    network_params = network.init(rng, init_x)

    fig, ax = plt.subplots(figsize=(16, 6))
    key = random.PRNGKey(0)
    keys = random.split(key, env_params.num_agents)
    obs, state = env.reset(keys, env_params)
    total_reward = jnp.zeros(env_params.num_agents)

    def animate(i):
        global key, state, action, obs, network, network_params, total_reward
        # actions = jnp.array([action] * env_params.num_agents)
        pi, _ = network.apply(network_params, obs)
        key, _key = random.split(key)
        actions = pi.sample(_key)
        
        step_keys = random.split(key, env_params.num_agents)
        obs, state, reward, terminated, info = env.step(step_keys, state, actions, env_params)
        total_reward += reward
        
        ax.clear()
        xs, thetas = state.physics[:, 0], state.physics[:, 2]
        ax.set_xlim(-4, 4)
        ax.set_ylim(-1.5, 1.5)
        plot(xs, thetas, params=env_params)

        if i % env_params.num_steps == 0:
            reset_keys = random.split(key, env_params.num_agents)
            obs, state = env.reset(reset_keys, env_params)
            print(f"Step {i}, Reward: {total_reward.mean()}, Terminated: {terminated}")
            total_reward = jnp.zeros(env_params.num_agents)

    ani = FuncAnimation(fig, animate, frames=None, interval=1, cache_frame_data=False)
    plt.show()