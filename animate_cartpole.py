import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from cartpole import CartPoleEnv
from matplotlib.animation import FuncAnimation
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

if __name__ == "__main__":
    jax.default_device(jax.devices('cpu')[0])

    listener = keyboard.Listener(on_press=on_press)
    action = 0  # Default action
    listener.start()

    env = CartPoleEnv()
    env_params = env.make_params(num_agents=10)
    network = ActorCritic(action_dim=env.action_space(env_params)[0].shape[0])
    rng = jax.random.PRNGKey(0)
    init_x = jnp.zeros(env.observation_space(env_params)[0].shape[0])
    network_params = network.init(rng, init_x)

    fig, ax = plt.subplots(figsize=(16, 6))
    key = random.PRNGKey(0)
    obs, state = env.reset(key, env_params)
    total_reward = jnp.zeros(env_params.num_agents)

    def animate(i):
        global key, state, action, obs, network, network_params, total_reward
        # pi, _ = network.apply(network_params, obs)
        # key, _key = random.split(key)
        # actions = pi.sample(_key)
        actions = jnp.array([action] * env_params.num_agents)

        step_keys = random.split(key, env_params.num_agents)
        obs, state, reward, terminated, info = env.step(step_keys, state, actions, env_params)
        total_reward += reward

        ax.clear()
        env.render(state, params=env_params)

        if i % env_params.num_steps == 0:
            key, sub_key = random.split(key, 2)
            obs, state = env.reset(sub_key, env_params)
            print(f"Step {i}, Reward: {total_reward.mean()}, Terminated: {terminated}")
            total_reward = jnp.zeros(env_params.num_agents)

    ani = FuncAnimation(fig, animate, frames=None, interval=1, cache_frame_data=False)
    plt.show()
