import jax
import jax.numpy as jnp
from jax import random
from pynput import keyboard
import cv2

from zoo import MazeEnv
from rl import CNNActorCritic

def on_press(key):
    global action
    try:
        if key == keyboard.Key.left:
            action = 0
        elif key == keyboard.Key.right:
            action = 1
        elif key == keyboard.Key.up:
            action = 2
        elif key == keyboard.Key.down:
            action = 3
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    jax.default_device(jax.devices('cpu')[0])

    listener = keyboard.Listener(on_press=on_press)
    action = 3  # Default action
    listener.start()

    env = MazeEnv()
    env_params = env.make_params(num_agents=1, num_steps=100, num_particles=8, dt=0.001, maze_path="./zoo/assets/maze16.png")
    network = CNNActorCritic(action_dim=env.action_space(env_params)[0].shape[0])
    rng = jax.random.PRNGKey(0)
    init_x = jnp.zeros(env.observation_space(env_params)[0].shape)
    network_params = network.init(rng, init_x)

    key = random.PRNGKey(0)
    obs, state = env.reset(key, env_params)
    total_reward = jnp.zeros(env_params.num_agents)
    
    step_jit  = jax.jit(env.step, static_argnames=['params'])
    
    renderer = env.make_renderer()

    i = 1
    while True:
        pi, _ = network.apply(network_params, obs)
        key, _key = random.split(key)
        actions = pi.sample(_key)
        # actions = pi.sample_deterministic()  # Deterministic actions
        # actions = jnp.array([action] * env_params.num_agents)

        step_keys = random.split(key, env_params.num_agents)
        obs, state, reward, terminated, info = step_jit(step_keys, state, actions, env_params)
        total_reward += reward

        renderer(state, params=env_params)

        if i % env_params.num_steps == 0:
            key, sub_key = random.split(key, 2)
            obs, state = env.reset(sub_key, env_params)
            print(f"Step {i}, Reward: {total_reward}, Terminated: {terminated}")
            total_reward = jnp.zeros(env_params.num_agents)

        i += 1
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
