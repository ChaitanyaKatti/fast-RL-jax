import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.struct import dataclass
from flax.training.train_state import TrainState
from env import Env, EnvParams
from typing import Dict
from colorama import Fore

@dataclass
class PPOParams:
    LR: float = 3e-4
    NUM_AGENTS: int = 1
    NUM_STEPS: int = 128
    TOTAL_TIMESTEPS: int = 1e6
    EPOCHS: int = 4
    NUM_MINIBATCHES: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    CLIP_VALUE: float = 0.5
    ENT_COEF: float = 0.0
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    ANNEAL_LR: bool = False
    DEBUG: bool = True

    # Derived parameters
    BATCH_SIZE: int = 0 # To be calculated based on NUM_AGENTS and NUM_STEPS
    NUM_UPDATES: int = 0 # To be calculated based on TOTAL_TIMESTEPS, NUM_STEPS, and NUM_AGENTS
    MINIBATCH_SIZE: int = 0 # To be calculated based on NUM_AGENTS, NUM_STEPS, and NUM_MINIBATCHES

@dataclass
class Transition:
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def make_train(network: nn.Module, env: Env, env_params: EnvParams, params: PPOParams):
    assert (
        params.NUM_AGENTS * params.NUM_STEPS % params.NUM_MINIBATCHES == 0
    ), "Number of agents * number of steps must be divisible by number of minibatches"

    params = params.replace(
        NUM_UPDATES=int(params.TOTAL_TIMESTEPS // params.NUM_STEPS // params.NUM_AGENTS),
        MINIBATCH_SIZE=int(params.NUM_AGENTS * params.NUM_STEPS // params.NUM_MINIBATCHES),
        BATCH_SIZE=int(params.NUM_AGENTS * params.NUM_STEPS),
    )

    # Define a linear learning rate schedule
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (params.NUM_MINIBATCHES * params.EPOCHS))
            / params.NUM_UPDATES
        )
        return params.LR * frac

    # Define the training function
    def train(rng):
        # NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params)[0].shape[0])
        network_params = network.init(_rng, init_x)

        # OPTIMIZER
        if params.ANNEAL_LR:
            tx = optax.chain(
                optax.clip_by_global_norm(params.MAX_GRAD_NORM),
                optax.adam(learning_rate=linear_schedule),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(params.MAX_GRAD_NORM),
                optax.adam(params.LR),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # TRAIN LOOP
        def _update_step(train_state, rng):
            def _env_step(runner_state, rng):
                train_state, env_state, last_obs = runner_state

                # SELECT ACTION
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng_step = jax.random.split(rng, params.NUM_AGENTS)
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                # STORE TRANSITION
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv)
                return runner_state, transition

            # RESET ENV
            reset_rng = jax.random.split(rng, params.NUM_AGENTS)
            obsv, env_state = env.reset(reset_rng, env_params)
            runner_state = (train_state, env_state, obsv)

            # COLLECT TRAJECTORIES
            step_rng = jax.random.split(rng, params.NUM_STEPS)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, step_rng, length=params.NUM_STEPS
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + params.GAMMA * next_value * (1 - done) - value
                    gae = (
                        delta
                        + params.GAMMA * params.GAE_LAMBDA * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            # NORMALIZE ADVANTAGES per ROLLOUT
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # BATCH PREPARATION
            batch = (traj_batch, advantages, targets)
            batch = jax.tree.map(
                lambda x: x.reshape((params.BATCH_SIZE,) + x.shape[2:]), batch
            )
            # SHUFFLE THE BATCH
            permutation = jax.random.permutation(rng, params.BATCH_SIZE)
            shuffled_batch = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            # SPLIT INTO MINIBATCHES
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(
                    x, [params.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )

            # UPDATE EPOCH
            def _update_epoch(train_state, unused):
                # UPDATE MINIBATCH
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(network_params: Dict, ppo_params:PPOParams, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(network_params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-ppo_params.CLIP_VALUE, ppo_params.CLIP_VALUE)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        # ADVANTAGE NORMALIZATION per MINIBATCH
                        # gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - ppo_params.CLIP_EPS,
                                1.0 + ppo_params.CLIP_EPS,
                            )
                            * gae
                        )
                        # loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = -(ratio * gae - jnp.abs(gae) * jnp.square(ratio - 1.0)/(2.0 * ppo_params.CLIP_EPS)) # Simple Policy Optimization
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + ppo_params.VF_COEF * value_loss
                            - ppo_params.ENT_COEF * entropy
                        )
                        return total_loss, (loss_actor, value_loss, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    return train_state, total_loss

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches, length=params.NUM_MINIBATCHES
                )
                return train_state, total_loss

            train_state, loss_info = jax.lax.scan(
                _update_epoch, train_state, None, length=params.EPOCHS
            )
            metric = (traj_batch.info, loss_info)

            if params.DEBUG:
                def callback(metric):
                    info, loss_info = metric
                    total_loss, (actor_loss_arr, value_loss_arr, entropy_arr) = loss_info
                    done_until = jnp.cumsum(info['done'], axis=0) # shape: (num_steps, num_agents)
                    done_until = jnp.where(done_until == 0, 1, 0)
                    total_rewards = jnp.mean(jnp.sum(info['reward'] * done_until, axis=0))
                    avg_length = jnp.mean(jnp.sum(done_until, axis=0))
                    print(
                        Fore.GREEN
                        + f"\tAverage episode length: {avg_length:.2f}"
                        + "\t|\t"
                        + Fore.BLUE
                        + f"Total rewards: {total_rewards:.2f}"
                        + "\t|\t",
                        end="",
                    )
                    print(
                        Fore.CYAN
                        + f"Actor Loss: {jnp.mean(actor_loss_arr):e}"
                        + "\t|\t"
                        + Fore.MAGENTA
                        + f"Value Loss: {jnp.mean(value_loss_arr):e}"
                        + "\t|\t"
                        + Fore.YELLOW
                        + f"Entropy: {jnp.mean(entropy_arr):.4f}"
                    )

                jax.debug.callback(callback, metric)

            return train_state, metric

        # ENV
        rngs = jax.random.split(rng, params.NUM_UPDATES)
        train_state, metric = jax.lax.scan(
            _update_step, train_state, rngs, length=params.NUM_UPDATES
        )
        return {"train_state": train_state, "metrics": metric}

    return train
