import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from crazyflie import CrazyflieEnv, CrazyflieParams, CrazyflieState
from matplotlib.animation import FuncAnimation
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
from network import ActorCritic

def on_press(key):
    global action
    try:
        if key == Key.up:
            action[0] = 1
        elif key == Key.down:
            action[0] = -1
        if key == Key.left:
            action[3] = 1
        elif key == Key.right:
            action[3] = -1
        if key == KeyCode.from_char('w'):
            action[2] = 1
        elif key == KeyCode.from_char('s'):
            action[2] = -1
        if key == KeyCode.from_char('a'):
            action[1] = -1
        elif key == KeyCode.from_char('d'):
            action[1] = 1
        print(f"Action set to: {action}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    jax.default_device(jax.devices('cpu')[0])

    listener = keyboard.Listener(on_press=on_press)
    action = [0, 0, 0, 0]  # Default action
    listener.start()

    env = CrazyflieEnv()
    env_params = env.make_params(num_agents=2,phy_freq=50, ctrl_freq=50)
    network = ActorCritic(action_dim=env.action_space(env_params)[0].shape[0])
    rng = jax.random.PRNGKey(0)
    init_x = jnp.zeros(env.observation_space(env_params)[0].shape[0])
    network_params = network.init(rng, init_x)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_title('Crazyflie Animation')
    ax.view_init(elev=20, azim=-60)

    key = random.PRNGKey(0)
    obs, state = env.reset(key, env_params)
    total_reward = jnp.zeros(env_params.num_agents)

    def animate(i):
        global key, state, action, obs, network, network_params, total_reward
        # pi, _ = network.apply(network_params, obs)
        # key, _key = random.split(key)
        # actions = pi.sample(_key)
        actions = jnp.array([action] * env_params.num_agents, dtype=jnp.float32)

        step_keys = random.split(key, env_params.num_agents)
        obs, state, reward, terminated, info = env.step(step_keys, state, actions, env_params)
        total_reward += reward

        # ax.clear()
        # env.render(ax, state, params=env_params)

        if i % env_params.num_steps == 0 or terminated.any():
            key, sub_key = random.split(key, 2)
            obs, state = env.reset(sub_key, env_params)
            print(f"Step {i}, Reward: {total_reward.mean()}, Terminated: {terminated}")
            total_reward = jnp.zeros(env_params.num_agents)

    ani = FuncAnimation(fig, animate, frames=None, interval=1, cache_frame_data=False)
    plt.show()