from agent import ActorCritic
from ppo import make_train, PPOParams
from cartpole import CartPoleEnv, CartPoleParams
import jax
jax.default_device(jax.devices('cpu')[0])
from colorama import Fore, init
init(autoreset=True)

if __name__ == "__main__":
    env = CartPoleEnv()
    env_params = CartPoleParams(num_agents=100)
    ppo_params = PPOParams(
        NUM_AGENTS=env_params.num_agents,
        NUM_STEPS=env_params.time_limit,
        NUM_MINIBATCHES=100,
        EPOCHS=10)
    # 100 agents, 100 steps = 10000 per rollout, 100 minibatches of size 100, 10 epochs per minibatch
    network = ActorCritic(action_dim=env.action_space(env_params)[0].shape[0])
    rng = jax.random.PRNGKey(0)
    train_jit = (make_train(network, env, env_params, ppo_params))
    out = train_jit(rng)
    
    print(Fore.GREEN + "Training completed.")
    # print(out['runner_state'][0].params)
