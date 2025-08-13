import ppo
from network import ActorCritic
from cartpole import CartPoleEnv as Env
# from copter2d import Copter2DEnv as Env
# from crazyflie import CrazyflieEnv as Env

import jax
import jax.numpy as jnp
import colorama
from colorama import Fore, Style, Back
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

jax.default_device(jax.devices('cuda')[0])
colorama.init(autoreset=True)

def test(network_params):
    cpu = jax.devices('cpu')[0]
    jax.default_device(cpu)
    network_params = jax.tree.map(lambda x: jax.device_put(x, cpu), network_params)

    # Initialize the environment and parameters
    test_env = Env()
    test_env_params = test_env.make_params(num_agents=5, num_steps=100) # Same as training environment

    # Reset the environment
    key = jax.random.PRNGKey(0)
    obs, state = test_env.reset(key, test_env_params)

    # Initialize the network
    network = ActorCritic(action_dim=test_env.action_space(test_env_params)[0].shape[0])

    # Initialize total reward
    total_reward = jnp.zeros(test_env_params.num_agents)

    fig, ax = plt.subplots(figsize=(16, 6))

    def animate(i):
        nonlocal key, state, obs, network, network_params, total_reward
        if i % test_env_params.num_steps == 0:
            key, sub_key = jax.random.split(key, 2)
            obs, state = test_env.reset(sub_key, test_env_params)
            print(f"Episode {i // test_env_params.num_steps}, Reward: {total_reward.mean()}")
            total_reward = jnp.zeros(test_env_params.num_agents)

        pi, _ = network.apply(network_params, obs)
        key, _key = jax.random.split(key)
        actions = pi.sample(key) # Sample actions from the policy
        actions = pi.loc # Deterministic actions

        step_keys = jax.random.split(key, test_env_params.num_agents)
        obs, state, reward, terminated, info = test_env.step(step_keys, state, actions, test_env_params)
        total_reward += reward

        ax.clear()
        test_env.render(state, test_env_params)

    ani = FuncAnimation(fig, animate, interval=1, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    env = Env()
    env_params = env.make_params(num_agents=1024, num_steps=100)
    ppo_params = ppo.make_params(
        LR=6e-4,
        TOTAL_TIMESTEPS=50_000_000,
        NUM_AGENTS=env_params.num_agents,
        NUM_STEPS=env_params.num_steps,  # 100
        NUM_MINIBATCHES=64,
        ANNEAL_LR=False,
        MAX_GRAD_NORM=1.0,
        CLIP_VALUE=1.0,
        CLIP_EPS=0.2,
        EPOCHS=4,
        ENT_COEF=0.01,
        DEBUG=True,
    )

    network = ActorCritic(action_dim=env.action_space(env_params)[0].shape[0])
    rng = jax.random.PRNGKey(0)
    train = jax.jit(ppo.make_train(network, env, env_params, ppo_params))
    out = train(rng)

    print(Fore.GREEN + Style.BRIGHT + Back.WHITE + "Training complete. Testing the agent...")

    network_params = out['train_state'].params
    test(network_params)
