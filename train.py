from rl import ppo

# from rl import ActorCritic as Agent
from rl import CNNActorCritic as Agent

# from zoo import CartPoleEnv as Env
# from zoo import Copter2DEnv as Env
# from zoo import CrazyflieEnv as Env
from zoo import MazeEnv as Env

# from hyperparams import cartpole as hyperparams
# from hyperparams import copter2d as hyperparams
# from hyperparams import crazyflie as hyperparams
from hyperparams import maze as hyperparams

import jax
import jax.numpy as jnp
import colorama
from colorama import Fore, Style, Back
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

jax.config.update("jax_debug_nans", True)
jax.default_device(jax.devices('cuda')[0])
colorama.init(autoreset=True)

def test(network_params, max_frames=500):
    cpu = jax.devices('cpu')[0]
    jax.default_device(cpu)
    network_params = jax.tree.map(lambda x: jax.device_put(x, cpu), network_params)

    # Initialize the environment and parameters
    test_env = Env()
    test_env_params = test_env.make_params(**{**hyperparams["env_params"], 'num_agents': 4})  # Same as training environment

    # Reset the environment
    key = jax.random.PRNGKey(0)
    obs, state = test_env.reset(key, test_env_params)

    # Initialize the network
    network = Agent(action_dim=test_env.action_space(test_env_params)[0].shape[0])

    # Initialize total reward
    total_reward = jnp.zeros(test_env_params.num_agents)

    renderer = test_env.make_renderer()

    for frame in range(max_frames):
        # Reset episode at boundaries
        if frame % test_env_params.num_steps == 0 and frame > 0:
            key, sub_key = jax.random.split(key, 2)
            obs, state = test_env.reset(sub_key, test_env_params)
            print(f"Episode {frame // test_env_params.num_steps}, Reward: {total_reward.mean()}")
            total_reward = jnp.zeros(test_env_params.num_agents)

        # Forward pass through network
        pi, _ = network.apply(network_params, obs)
        key, _key = jax.random.split(key)
        actions = pi.sample_deterministic()

        # Step environment
        step_keys = jax.random.split(key, test_env_params.num_agents)
        obs, state, reward, terminated, info = test_env.step(step_keys, state, actions, test_env_params)
        total_reward += reward

        # Render
        if not renderer(state, params=test_env_params):
                print("Exiting render loop.")
                break


if __name__ == "__main__":
    env = Env()
    env_params = env.make_params(**hyperparams['env_params'])
    ppo_params = ppo.make_params(**hyperparams['ppo_params'])

    rng = jax.random.PRNGKey(0)
    train = jax.jit(ppo.make_train(Agent, env, env_params, ppo_params))
    out = train(rng)
    network_params = out['train_state'].params

    print("Training complete. Testing the agent...")

    # Save as GIF
    # test(network_params, save_gif=True, gif_filename="./misc/agent_test.gif", max_frames=300)

    # Or display live (comment out the line above and uncomment below)
    test(network_params)
