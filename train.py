from agent import ActorCritic
from ppo import make_train, PPOParams
from cartpole import CartPoleEnv, CartPoleParams

import jax
import jax.numpy as jnp
from colorama import Fore, init
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from animate_cartpole import plot

jax.default_device(jax.devices('cpu')[0])
init(autoreset=True)

def test(network_params):
    fig, ax = plt.subplots(figsize=(16, 6))

    # Initialize the environment and parameters
    test_env = CartPoleEnv()
    test_env_params = CartPoleParams(num_agents=6)

    # Reset the environment
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, test_env_params.num_agents)
    obs, state = test_env.reset(keys, test_env_params)
    
    # Initialize the network
    network = ActorCritic(action_dim=test_env.action_space(test_env_params)[0].shape[0])
    
    # Initialize total reward
    total_reward = jnp.zeros(test_env_params.num_agents)
    
    def animate(i):
        nonlocal key, state, obs, network, network_params, total_reward
        if i % test_env_params.num_steps == 0:
            reset_keys = jax.random.split(key, test_env_params.num_agents)
            obs, state = test_env.reset(reset_keys, test_env_params)
            print(f"Episode {i // test_env_params.num_steps}, Reward: {total_reward.mean()}")
            total_reward = jnp.zeros(test_env_params.num_agents)

        pi, _ = network.apply(network_params, obs)
        key, _key = jax.random.split(key)
        actions = pi.loc # Deterministic actions

        step_keys = jax.random.split(key, test_env_params.num_agents)
        obs, state, reward, terminated, info = test_env.step(step_keys, state, actions, test_env_params)
        total_reward += reward
        
        ax.clear()
        xs, thetas = state.physics[:, 0], state.physics[:, 2]
        ax.set_xlim(-4, 4)
        ax.set_ylim(-1.5, 1.5)
        plot(xs, thetas, params=test_env_params)

    ani = FuncAnimation(fig, animate, interval=1, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    env = CartPoleEnv()
    env_params = CartPoleParams(num_agents=100)
    ppo_params = PPOParams(
        LR=1e-3,
        TOTAL_TIMESTEPS=3_000_000,
        NUM_AGENTS=env_params.num_agents,
        NUM_STEPS=env_params.num_steps,
        NUM_MINIBATCHES=200,
        ANNEAL_LR=False,
        MAX_GRAD_NORM=1.0,
        CLIP_VALUE=1.5,
        CLIP_EPS=0.2,
        EPOCHS=5,
        ENT_COEF=0.01,
        DEBUG=True)

    network = ActorCritic(action_dim=env.action_space(env_params)[0].shape[0])
    rng = jax.random.PRNGKey(0)
    train = jax.jit(make_train(network, env, env_params, ppo_params))
    out = train(rng)
    # out.block_until_ready()

    print(Fore.GREEN + "Training complete. Testing the agent...")
    network_params = out['train_state'].params
    test(network_params)  