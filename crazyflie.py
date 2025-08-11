from flax import struct
import jax.numpy as jnp
from jax import lax, random
from env import Env, EnvParams, EnvState
from typing import Tuple


@struct.dataclass
class CrazyflieParams(EnvParams):
    num_agents: int             # Number of agents in the environment
    mass: float                 # Mass of the drone (in kg)
    inertia: jnp.ndarray        # Inertia of the drone (in kg*m^2)              # Shape(3,3)
    inertia_inv: jnp.ndarray    # Inverse of inertia                            # Shape(3,3)
    arm_length: float           # length of drone arm in (m)
    g: float                    # Acceleration due to gravity (m/s^2)

    PWM_MAX: int                # PWM command magnitude
    rate_center: jnp.ndarray    # Betaflight rates (in degrees/second)          # Shape (3,)
    rate_max: jnp.ndarray       # Betaflight rates (in degrees/second)          # Shape (3,)
    rate_expo: jnp.ndarray      # Betaflight rates expo                         # Shape (3,)
    P_GAIN: jnp.ndarray         # Proportional gain for the rate controller     # Shape (3,)
    I_GAIN: jnp.ndarray         # Integral gain for the rate controller         # Shape (3,)
    ERROR_SUM_MAX: jnp.ndarray  # Maximum integral term for the rate controller # Shape (3,)

    # Thurst coefficients
    thurst_coeff_a: float
    thurst_coeff_b: float
    thurst_coeff_c: float
    # Rotor velocity coefficient
    rotor_velocity_coeff_a: float
    rotor_velocity_coeff_b: float
    # Torque coefficient
    torque_coeff: float
    # Drag coefficients(in kg/rad)
    drag_coeff: jnp.ndarray     # Shape (3,)

    motor_settling_time: float  # Motor settling time
    motor_alpha: float          # Motor alpha for smoothing

    phy_freq: int                   # Physics loop frequency
    phy_dt: float                   # Physics loop time step
    ctrl_freq: int                  # Control loop frequency
    ctrl_dt: float                  # Control loop time step
    phy_steps_per_ctrl_step: int    # Number of physics steps per control step

    pos_threshold: jnp.ndarray  # Position at which the episode terminates      # Shape (3,)
    num_steps: int              # Time limit for the episode in steps

    # Mixer Matrix
    phy_mix: jnp.ndarray


@struct.dataclass
class CrazyflieState(EnvState):
    pos: jnp.ndarray            # shape: (num_agents, 3)    # [x, y, z]
    vel: jnp.ndarray            # shape: (num_agents, 3)    # [x_dot, y_dot, z_dot]
    quat: jnp.ndarray           # shape: (num_agents, 4)    # Quaternion representation of orientation
    rot_mat: jnp.ndarray        # shape: (num_agents, 3, 3) # Rotation matrix
    ang_vel: jnp.ndarray        # shape: (num_agents, 3)    # Angular velocity in body frame [p, q, r]
    pwm: jnp.ndarray            # shape: (num_agents, 4)    # PWM commands for the motors
    current_action: jnp.ndarray # shape: (num_agents, 4)    # Current action applied to the drone
    last_action: jnp.ndarray    # shape: (num_agents, 4)    # Last action applied to the drone
    rate_error_sum: jnp.ndarray # shape: (num_agents, 3)    # Integral term for the rate controller
    t: jnp.ndarray              # shape: (num_agents,)      # Time step for each agent


class CrazyflieEnv(Env):
    @classmethod
    def reset(cls, key: jnp.ndarray, params: CrazyflieParams):
        pos_rand_mag = params.pos_threshold             # Random position magnitude for initial state
        quat_rand_mag = jnp.array([0.1, 0.1, 0.1, 0.1]) # Small random quaternion for initial state
        vel_rand_mag = jnp.array([0.1, 0.1, 0.1])       # Small random velocity for initial state
        ang_vel_rand_mag = jnp.array([0.1, 0.1, 0.1])   # Small random angular velocity for initial state

        key1, key2, key3, key4 = random.split(key, 4)
        pos = random.uniform(key1, (params.num_agents, 3), minval=-pos_rand_mag, maxval=pos_rand_mag)
        vel = random.uniform(key2, (params.num_agents, 3), minval=-vel_rand_mag, maxval=vel_rand_mag)
        quat = random.uniform(key3, (params.num_agents, 4), minval=-quat_rand_mag, maxval=quat_rand_mag)
        ang_vel = random.uniform(key4, (params.num_agents, 3), minval=-ang_vel_rand_mag, maxval=ang_vel_rand_mag)

        quat = quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)  # Normalize quaternion
        rot_mat = cls.quat_to_rot_mat(quat) # Shape: (num_agents, 3, 3)

        pwm = 0.5 * params.PWM_MAX * jnp.ones((params.num_agents, 4))   # Set PWM to 50% of max
        current_action = jnp.zeros((params.num_agents, 4))              # 50% throttle and no roll, pitch, yaw
        last_action = jnp.zeros((params.num_agents, 4))                 # 50% throttle and no roll, pitch, yaw
        rate_error_sum = jnp.zeros((params.num_agents, 3))
        t = jnp.zeros(params.num_agents)

        state = CrazyflieState(
            pos=pos,
            vel=vel,
            quat=quat,
            rot_mat=rot_mat,
            ang_vel=ang_vel,
            pwm=pwm,
            current_action=current_action,
            last_action=last_action,
            rate_error_sum=rate_error_sum,
            t=t
        )
        obs = cls.observation(state, params)

        return obs, state

    @classmethod
    def step(cls, key: jnp.ndarray, state: CrazyflieState, action: jnp.ndarray, params: CrazyflieParams):
        next_state = cls.crazyflie_step(state, action, params)
        obs = cls.observation(next_state, params)
        terminated = cls.terminated(next_state, params)
        reward = cls.reward(next_state, params)
        return obs, next_state, reward, terminated, {'reward': reward, 'done': terminated}

    @staticmethod
    def action_space(params: CrazyflieParams):
        return jnp.array([-1.0, -1.0, -1.0, -1.0]), jnp.array([1.0, 1.0, 1.0, 1.0])

    @staticmethod
    def observation_space(params: CrazyflieParams):
        # Observation space:
        low = jnp.array(
            [
                -params.pos_threshold[0], -params.pos_threshold[1], -params.pos_threshold[2], # Position
                -1, -1, -1,                     # First column of R
                -1, -1, -1,                     # Third column of R
                -jnp.inf, -jnp.inf, -jnp.inf,   # Velocity
                -jnp.inf, -jnp.inf, -jnp.inf,   # Roll, Pitch, Yaw rates
                -1, -1, -1, -1,                 # Last Action
            ]
        )
        high = jnp.array(
            [
                params.pos_threshold[0], params.pos_threshold[1], params.pos_threshold[2], # Position
                1, 1, 1,                    # First column of R,
                1, 1, 1,                    # Third column of R,
                jnp.inf, jnp.inf, jnp.inf,  # Velocity
                jnp.inf, jnp.inf, jnp.inf,  # Roll, Pitch, Yaw rates
                1, 1, 1, 1,                 # Last Action
            ]
        )
        return low, high

    @classmethod
    def crazyflie_single_step(cls, state: CrazyflieState, action: jnp.ndarray, params: CrazyflieParams):
        """
        Perform a single step of the Crazyflie environment.
        """
        #                X axis
        #                ^
        #                |
        #                |
        #
        #         M4(CW)   M1(CCW)
        #            \      /
        #              \  /
        #  Y axis<------##
        #              /  \
        #            /      \
        #         M3(CCW)   M2(CW)

        # PWM from rate_controller
        pwm, rate_error_sum = cls.rate_controller(state, action, params)                                           # Shape (num_agents, 4)
        pwm = (1.0-params.motor_alpha) * state.pwm + params.motor_alpha * pwm # PWM Motor smoothing,    # Shape (num_agents, 4)

        # Thrust for each motor
        motor_thurst = (
            params.thurst_coeff_a * pwm**2
            + params.thurst_coeff_b * pwm
            + params.thurst_coeff_c
        )  # Shape (num_agents, 4)
        total_thrust, torque_x, torque_y, torque_z = (motor_thurst @ params.phy_mix.T).T
        total_thrust = total_thrust.reshape(-1, 1)  # Shape (num_agents, 1)
        torques = jnp.stack([torque_x, torque_y, torque_z], axis=-1)  # Shape (num_agents, 3)

        # Drag, in local frame, proportional to rotor angular velocity
        # R_inv is the inverse of the rotation matrix, which transforms global velocity to local frame
        R_inv = jnp.swapaxes(state.rot_mat, -1, -2)
        rotor_ang_vel = (params.rotor_velocity_coeff_a * pwm + params.rotor_velocity_coeff_b)           # Shape (num_agents, 4)
        rotor_sum = jnp.sum(rotor_ang_vel, axis=-1)  # (N,)
        body_vel = jnp.einsum('nij,nj->ni', R_inv, state.vel)  # (N,3)
        drags = (body_vel * params.drag_coeff) * rotor_sum[:, None]  # (N,3)

        # Equations of motion
        # Global Linear Acceleration
        up = jnp.array([0.0, 0.0, 1.0])
        acc = jnp.einsum('nij,nj->ni', state.rot_mat, (total_thrust * up - drags))/params.mass - params.g*up        # Shape (num_agents, 3)
        # Local Angular Acceleration
        I_omega = jnp.einsum('ij,nj->ni', params.inertia, state.ang_vel)                # Shape (num_agents,3)
        cross = jnp.cross(state.ang_vel, I_omega)                                       # Shape (num_agents,3)
        angular_acc = jnp.einsum('ij,nj->ni', params.inertia_inv, (torques - cross))    # Shape (num_agents,3)

        # Update state
        ang_vel = state.ang_vel + angular_acc * params.ctrl_dt                          # Shape (num_agents, 3)
        pos = state.pos + state.vel * params.ctrl_dt + 0.5 * acc * params.ctrl_dt**2    # Shape (num_agents, 3)
        vel = state.vel + acc * params.ctrl_dt                                          # Shape (num_agents, 3)
        quat = cls.quat_update(state.quat, ang_vel, params.ctrl_dt)                     # Shape (num_agents, 4)
        rot_mat = cls.quat_to_rot_mat(quat)                                             # Shape (num_agents, 3, 3)
        current_action = action             # Store the new action
        last_action = state.current_action  # Retrive the action from last step
        t = state.t + params.ctrl_dt  # Update time step for each agent

        return state.replace(
            pos=pos,
            vel=vel,
            quat=quat,
            rot_mat=rot_mat,
            ang_vel=ang_vel,
            pwm=pwm,
            current_action=current_action,
            last_action=last_action,
            rate_error_sum=rate_error_sum,
            t=t
        )

    @classmethod
    def crazyflie_step(cls, state: CrazyflieState, action: jnp.ndarray, params: CrazyflieParams):
        """ Perform multiple steps of the Crazyflie environment.
        Arguments:
            state: Current state of the environment
            action: Action taken by the agent, which is the desired thrust in the range [-1, 1]
            params: Parameters of the environment
        Returns:
            next_state: Next state of the environment after applying the action
        """
        # Save the last action since it will be update many times during substeps
        last_action = state.current_action

        # Perform params.phy_steps_per_ctrl_step physics steps
        state = lax.fori_loop(0, params.phy_steps_per_ctrl_step, lambda i, s: cls.crazyflie_single_step(s, action, params), state)

        return state.replace(
            current_action=action,  # Update the current action
            last_action=last_action,  # Keep the last action for reward calculation
        )    

    @staticmethod
    def reward(state: CrazyflieState, params: CrazyflieParams) -> jnp.ndarray:
        return (
            1.0                                                                     # Reward to stay alive
            - jnp.linalg.norm(state.pos, axis=1) / params.pos_threshold[0]          # Reward based on distance to origin
            - jnp.linalg.norm(state.current_action - state.last_action, axis=-1)    # Reward for smooth actions
        )

    @staticmethod
    def terminated(state: CrazyflieState, params: CrazyflieParams) -> jnp.ndarray:
        return jnp.any((jnp.abs(state.pos) > params.pos_threshold), axis=-1) # Terminate if any agent any of x,y,z bounds

    @staticmethod
    def observation(state: CrazyflieState, params: CrazyflieParams) -> jnp.ndarray:
        return jnp.concatenate([
            state.pos,              # Position in the world frame
            state.rot_mat[:, :, 0], # First column of R, which is the forward vector in the body frame
            state.rot_mat[:, :, 2], # Third column of R, which is the up vector in the body frame
            state.vel,              # Velocity in the world frame
            state.ang_vel,          # Roll, Pitch, Yaw rates in the body frame
            state.last_action,
        ], axis=-1)

    @staticmethod
    def render(state: CrazyflieState, params: CrazyflieParams):
        # This function is a placeholder for rendering the Crazyflie environment.
        # You can implement your own rendering logic here, such as using matplotlib or any other visualization library.
        pass

    @staticmethod
    def makeParams(
        num_agents: int = 1,
        mass: float = 0.028, # 28g
        inertia: jnp.ndarray = jnp.array(
            [
                [16.655602, -0.830806,   1.800197],
                [-0.830806, 16.571710,  -0.718277],
                [ 1.800197,  -0.718277, 29.261652],
            ]
        ),
        arm_length: float = 0.045, # 4.5cm
        g: float = 9.8,
        PWM_MAX: int = 65535,
        rate_center=jnp.array([200.0, 200.0, 200.0]),
        rate_max=jnp.array([670.0, 670.0, 670.0]),
        rate_expo=jnp.array([0.54, 0.54, 0.54]),
        P_GAIN=jnp.array([3800, 3800, 7600]),
        I_GAIN=jnp.array([1200, 1200, 2400]),
        ERROR_SUM_MAX=jnp.array([0.5, 0.5, 0.1]),
        thurst_coeff_a: float = 2.130295e-11,
        thurst_coeff_b: float = 1.032633e-6,
        thurst_coeff_c: float = 5.484560e-4,
        rotor_velocity_coeff_a: float = 0.04076521,
        rotor_velocity_coeff_b: float = 380.8359,
        torque_coeff: float = 0.005964552,
        drag_coeff: jnp.ndarray = jnp.array([-9.1785e-7, -9.1785e-7, -10.311e-7]),
        motor_settling_time: float = 0.01,
        phy_freq: int = 100,
        ctrl_freq: int = 20,
        pos_threshold: jnp.ndarray = jnp.array([2.0, 2.0, 2.0]),
        num_steps: int = 100,
    ) -> CrazyflieParams:
        assert phy_freq % ctrl_freq == 0, "phy_freq must be a multiple of ctrl_freq"
        phy_time_step = 1.0 / phy_freq
        ctrl_time_step = 1.0 / ctrl_freq
        phy_steps_per_ctrl_step = int(ctrl_time_step // phy_time_step)
        motor_alpha = 1.0 - jnp.exp(-4 * phy_time_step / motor_settling_time)

        # Inverse of inertia
        inertia_inv = jnp.linalg.inv(inertia)

        # Create the mixer matrix
        phy_mix = jnp.array(
            [
                [ 1.0,  1.0,  1.0,  1.0],   # total thrust
                [-1.0, -1.0,  1.0,  1.0],   # roll torque coeffs
                [-1.0,  1.0,  1.0, -1.0],   # pitch torque coeffs
                [-1.0,  1.0, -1.0,  1.0],   # yaw torque coeffs
            ]
        )
        phy_mix = phy_mix.at[1].multiply(arm_length)    # torque_x propotinal to arm length
        phy_mix = phy_mix.at[2].multiply(arm_length)    # torque_y propotinal to arm length
        phy_mix = phy_mix.at[3].multiply(torque_coeff)  # torque_z propotinal to torque_coeff

        return CrazyflieParams(
            num_agents=num_agents,
            mass=mass,
            inertia=inertia,
            inertia_inv=inertia_inv,
            arm_length=arm_length,
            g=g,
            PWM_MAX=PWM_MAX,
            rate_center=rate_center,
            rate_max=rate_max,
            rate_expo=rate_expo,
            P_GAIN=P_GAIN,
            I_GAIN=I_GAIN,
            ERROR_SUM_MAX=ERROR_SUM_MAX,
            thurst_coeff_a=thurst_coeff_a,
            thurst_coeff_b=thurst_coeff_b,
            thurst_coeff_c=thurst_coeff_c,
            rotor_velocity_coeff_a=rotor_velocity_coeff_a,
            rotor_velocity_coeff_b=rotor_velocity_coeff_b,
            torque_coeff=torque_coeff,
            drag_coeff=drag_coeff,
            motor_settling_time=motor_settling_time,
            motor_alpha=motor_alpha,
            phy_freq=phy_freq,
            phy_dt=phy_time_step,
            ctrl_freq=ctrl_freq,
            ctrl_dt=ctrl_time_step,
            phy_steps_per_ctrl_step=phy_steps_per_ctrl_step,
            pos_threshold=pos_threshold,
            num_steps=num_steps,
            phy_mix=phy_mix
        )

    @staticmethod
    def betaflight_rates(rates: jnp.ndarray, params: CrazyflieParams) -> jnp.ndarray:
        """
        Convert Betaflight rates to actual rates in radians per second.
        Arguments:
            rates: rates between -1 and 1. Shape (num_agents, 3)
        Returns:
            Actual rates in radians per second. Shape (num_agents, 3)
        """
        rates = rates * params.rate_center + (params.rate_max - params.rate_center) * jnp.abs(rates) * (jnp.power(rates, 5) * params.rate_expo + rates * (1 - params.rate_expo))
        rates = rates * (jnp.pi / 180)  # Convert degrees to radians
        return rates

    @classmethod
    def rate_controller(cls, state: CrazyflieState, action: jnp.ndarray, params: CrazyflieParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
        '''
        Rate controller that outputs desired PWMs value for each motor
        Arguments:
            state: Current state of the environment
            action: Action taken by the agent, which is the desired thrust in the range [-1, 1]
            params: Parameters of the environment
        Returns:
            pwm: PWM values for each motor, shape (num_agents, 4)
            rate_error_sum: Updated integral term for the rate controller, shape (num_agents, 3)
        '''
        # If thrust is <10% of max, then reset the rate error sum, need to slice action with :1 to maintain shape (num_agents, 1)
        rate_error_sum = jnp.where(action[..., :1] < -0.8, jnp.zeros_like(state.rate_error_sum), state.rate_error_sum)

        # Convert action to desired rates and thrust
        desired_rates = cls.betaflight_rates(action[...,1:4], params)   # Shape (num_agents, 3)
        desired_thrust = params.PWM_MAX * (0.5 * action[..., 0] + 0.5)  # Shape (num_agents, )

        # Calculate rate error
        rate_error = desired_rates - state.ang_vel                      # Shape (num_agents, 3)
        rate_error_sum = jnp.clip(
            state.rate_error_sum + rate_error * params.phy_dt,
            -params.ERROR_SUM_MAX,
            params.ERROR_SUM_MAX
        )

        # PI controller for rate control, output in units of PWM
        control = params.P_GAIN * rate_error + params.I_GAIN * rate_error_sum   # Shape (num_agents, 3)

        # PWM Mixer
        ctrl_mixer = jnp.array(
            [
                [1.0,   1.0,  1.0,  1.0],   # Total thrust
                [-1.0, -1.0,  1.0,  1.0],   # Roll torque
                [-1.0,  1.0,  1.0, -1.0],   # Pitch torque
                [-1.0,  1.0, -1.0,  1.0],   # Yaw torque
            ]
        )
        pwm = jnp.concatenate([desired_thrust[:, None], control], axis=-1) @ ctrl_mixer.T
        pwm = jnp.clip(pwm, 0, params.PWM_MAX)  # Shape (num_agents, 4)

        return pwm, rate_error_sum

    @staticmethod
    def quat_update(q, omega, dt):
        # q: (..., 4)  quaternion [qw, qx, qy, qz]
        # omega: (..., 3) angular velocity in body frame
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        wx, wy, wz = omega[..., 0], omega[..., 1], omega[..., 2]

        dq_w = -0.5 * (x*wx + y*wy + z*wz)
        dq_x =  0.5 * (w*wx + y*wz - z*wy)
        dq_y =  0.5 * (w*wy - x*wz + z*wx)
        dq_z =  0.5 * (w*wz + x*wy - y*wx)

        q_new = q + dt * jnp.stack([dq_w, dq_x, dq_y, dq_z], axis=-1)
        q_new /= jnp.linalg.norm(q_new, axis=-1, keepdims=True)  # Normalize
        return q_new

    @staticmethod
    def quat_to_rot_mat(q):
        # q: (..., 4) quaternion [qw, qx, qy, qz]
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        rot_mat = jnp.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
        ]).transpose(2, 0, 1) # Transpose to fix stacking since it was done along the first dimension which was num_agents
        return rot_mat
