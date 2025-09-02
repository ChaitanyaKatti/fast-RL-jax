import jax
import jax.numpy as jnp
from jax import random
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from zoo import CrazyflieEnv
from rl import ActorCritic

def on_press(key):
    global action
    try:
        if key == Key.up:
            action[2] = 1
        elif key == Key.down:
            action[2] = -1
        if key == Key.left:
            action[1] = -1
        elif key == Key.right:
            action[1] = 1
        if key == KeyCode.from_char('w'):
            action[0] = 1
        elif key == KeyCode.from_char('s'):
            action[0] = -1
        if key == KeyCode.from_char('a'):
            action[3] = 1
        elif key == KeyCode.from_char('d'):
            action[3] = -1
    except Exception as e:
        print(f"Error: {e}")

def on_release(key):
    global action
    if key in [Key.up, Key.down]:
        action[2] = 0
    if key in [Key.left, Key.right]:
        action[1] = 0
    if key in [KeyCode.from_char('w'), KeyCode.from_char('s')]:
        action[0] = 0
    if key in [KeyCode.from_char('a'), KeyCode.from_char('d')]:
        action[3] = 0
    if key == Key.esc: # Stop listener
        return False

if __name__ == "__main__":
    jax.default_device(jax.devices('cpu')[0])

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    action = [0, 0, 0, 0]  # Default action
    listener.start()

    env = CrazyflieEnv()
    env_params = env.make_params(num_agents=1, phy_freq=50, ctrl_freq=50)
    network = ActorCritic(action_dim=env.action_space(env_params)[0].shape[0])
    rng = jax.random.PRNGKey(0)
    init_x = jnp.zeros(env.observation_space(env_params)[0].shape[0])
    network_params = network.init(rng, init_x)

    key = random.PRNGKey(0)
    obs, state = env.reset(key, env_params)
    total_reward = jnp.zeros(env_params.num_agents)

    for i in range(10000):
        # pi, _ = network.apply(network_params, obs)
        # key, _key = random.split(key)
        # actions = pi.sample(_key)
        actions = jnp.array([action] * env_params.num_agents, dtype=jnp.float32)

        key, sub_key = random.split(key, 2)
        obs, state, reward, terminated, info = env.step(sub_key, state, actions, env_params)
        env.render(state, env_params)
        total_reward += reward

        if terminated.any():
            key, sub_key = random.split(key, 2)
            obs, state = env.reset(sub_key, env_params)
            print(f"Step {i}, Reward: {total_reward.mean()}, Terminated: {terminated}")
            total_reward = jnp.zeros(env_params.num_agents)
