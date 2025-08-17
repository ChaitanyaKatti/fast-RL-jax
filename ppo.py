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
    LR: float
    NUM_AGENTS: int
    NUM_STEPS: int
    TOTAL_TIMESTEPS: int
    EPOCHS: int
    NUM_MINIBATCHES: int
    GAMMA: float
    GAE_LAMBDA: float
    CLIP_EPS: float
    CLIP_VALUE: float
    ENT_COEF: float
    VF_COEF: float
    MAX_GRAD_NORM: float
    ANNEAL_LR: bool
    DEBUG: bool

    # Derived parameters
    BATCH_SIZE: int     # To be calculated based on NUM_AGENTS and NUM_STEPS
    NUM_UPDATES: int    # To be calculated based on TOTAL_TIMESTEPS, NUM_STEPS, and NUM_AGENTS
    MINIBATCH_SIZE: int # To be calculated based on NUM_AGENTS, NUM_STEPS, and NUM_MINIBATCHES


@dataclass
class Transition:
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_params(
    LR: float = 3e-4,
    NUM_AGENTS: int = 1,
    NUM_STEPS: int = 128,
    TOTAL_TIMESTEPS: int = 1e6,
    EPOCHS: int = 4,
    NUM_MINIBATCHES: int = 4,
    GAMMA: float = 0.99,
    GAE_LAMBDA: float = 0.95,
    CLIP_EPS: float = 0.2,
    CLIP_VALUE: float = 0.5,
    ENT_COEF: float = 0.0,
    VF_COEF: float = 0.5,
    MAX_GRAD_NORM: float = 0.5,
    ANNEAL_LR: bool = False,
    DEBUG: bool = True,
) -> PPOParams:
    assert (
        NUM_AGENTS * NUM_STEPS % NUM_MINIBATCHES == 0
    ), "Number of agents * number of steps must be divisible by number of minibatches"

    BATCH_SIZE = NUM_AGENTS * NUM_STEPS
    NUM_UPDATES = int(TOTAL_TIMESTEPS // NUM_STEPS // NUM_AGENTS)
    MINIBATCH_SIZE = int(NUM_AGENTS * NUM_STEPS // NUM_MINIBATCHES)

    return PPOParams(
        LR=LR,
        NUM_AGENTS=NUM_AGENTS,
        NUM_STEPS=NUM_STEPS,
        TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
        EPOCHS=EPOCHS,
        NUM_MINIBATCHES=NUM_MINIBATCHES,
        GAMMA=GAMMA,
        GAE_LAMBDA=GAE_LAMBDA,
        CLIP_EPS=CLIP_EPS,
        CLIP_VALUE=CLIP_VALUE,
        ENT_COEF=ENT_COEF,
        VF_COEF=VF_COEF,
        MAX_GRAD_NORM=MAX_GRAD_NORM,
        ANNEAL_LR=ANNEAL_LR,
        DEBUG=DEBUG,
        BATCH_SIZE=BATCH_SIZE,
        NUM_UPDATES=NUM_UPDATES,
        MINIBATCH_SIZE=MINIBATCH_SIZE,
    )


def make_train(network: nn.Module, env: Env, env_params: EnvParams, params: PPOParams):

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
            obsv, env_state = env.reset(rng, env_params)
            runner_state = (train_state, env_state, obsv)

            # COLLECT TRAJECTORIES
            step_rngs = jax.random.split(rng, params.NUM_STEPS)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, step_rngs, length=params.NUM_STEPS
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

                    def _loss_fn(network_params: Dict, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(network_params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-params.CLIP_VALUE, params.CLIP_VALUE)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)

                        # PPO, Clipped Type L1 Loss
                        # loss_actor = -jnp.minimum(
                        #     ratio * gae,
                        #     jnp.clip(ratio, 1.0 - params.CLIP_EPS, 1.0 + params.CLIP_EPS,)* gae
                        # )

                        # Simple Policy Optimization, Type L2 Loss
                        loss_actor = -(ratio * gae - jnp.abs(gae) * jnp.square(ratio - 1.0)/(2.0 * params.CLIP_EPS))

                        # SPO, Type L1 Loss
                        # loss_actor = jnp.abs(gae*(ratio - 1) - jnp.abs(gae) * params.CLIP_EPS) # Type L1 loss

                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + params.VF_COEF * value_loss
                            - params.ENT_COEF * entropy
                        )
                        return total_loss, (loss_actor, value_loss, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
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
            metric = (traj_batch.info, loss_info, train_state.step // (params.NUM_MINIBATCHES * params.EPOCHS))

            if params.DEBUG:
                def callback(metric):
                    info, loss_info, i = metric
                    total_loss, (actor_loss_arr, value_loss_arr, entropy_arr) = loss_info
                    done_until = jnp.cumsum(info['done'], axis=0) # shape: (num_steps, num_agents)
                    done_until = jnp.where(done_until == 0, 1, 0)
                    total_rewards = jnp.mean(jnp.sum(info['reward'] * done_until, axis=0))
                    avg_length = jnp.mean(jnp.sum(done_until, axis=0))
                    print(
                        Fore.YELLOW
                        + f"Iteration: {i:>6d}"
                        + Fore.GREEN
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
