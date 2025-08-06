import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from cartpole import CartPoleEnv, CartPoleParams
from matplotlib.animation import FuncAnimation

jax.default_device(jax.devices('cpu')[0])

env = CartPoleEnv()
params = CartPoleParams(num_agents=5)

def plot(xs, thetas):
    num_agents = xs.shape[0]
    
    for i in range(num_agents):
        # Draw cart as a box centered at (x, 0)
        cart_width, cart_height = 0.4, 0.2
        cart_y = 0.0
        plt.gca().add_patch(
            plt.Rectangle((xs[i] - cart_width/2, cart_y - cart_height/2), cart_width, cart_height, color='blue', alpha=0.7)
        )

        # Draw pole as a line from cart center at (x, 0) at angle theta
        pole_length = params.half_length * 2
        pole_x_end = xs[i] + pole_length * jnp.sin(thetas[i])
        pole_y_end = cart_y + pole_length * jnp.cos(thetas[i])
        plt.plot([xs[i], pole_x_end], [cart_y, pole_y_end], color='red', linewidth=3)

        # Draw axle
        plt.plot(xs[i], cart_y, 'ko', markersize=6)

    plt.xlim(-4, 4)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('CartPole')
    plt.show()

@jax.jit
def simulate():
    key = random.PRNGKey(9)
    state = env.reset(key, params)
    traj = jnp.zeros((100, params.num_agents, 2))
    action = jnp.zeros((params.num_agents, ))  # Assuming action space is 1D
    for i in range(100):
        traj = traj.at[i, :, :].set(jnp.array([state.physics[:, 0], state.physics[:, 2]]).T)  # Store x and theta
        state = env.step(state, action, params)
    return traj

traj = simulate()
traj.block_until_ready()

fig, ax = plt.subplots(figsize=(16, 6))

def animate(i):
    ax.clear()
    xs, thetas = traj[i, :, :].T
    plot(xs, thetas)

ani = FuncAnimation(fig, animate, frames=len(traj), interval=1, repeat=True)
plt.show()