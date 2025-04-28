"""
The LiAISON EKF using the ELFO and NRHO orbits

CR3BP model as ground truth
The unit of distance is normalised such that the distance between earth and moon is 1
Frame of reference is rotational centred at the barycentre of earth and moon
The unit of time is normalised such that the orbital period of the moon around the Earth is 2pi

state vector is [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2]
"""

import numpy as np
from pymatreader import read_mat
from scipy.integrate import solve_ivp
import spiceypy as sp
import matplotlib.pyplot as plt

mat_capstone = read_mat("MATLAB/HALO/output/ORBdataCapstone.mat")
mat_ELFO = read_mat("MATLAB/HALO/output/ORBdataClementine.mat")

seq_capstone = mat_capstone["orb"]["seq"]["a"]["XJ2000"]
seq_ELFO = mat_ELFO["orb"]["seq"]["a"]["XJ2000"]
time_seq_capstone = (
    mat_capstone["orb"]["seq"]["a"]["t"] - mat_capstone["orb"]["seq"]["a"]["t"][0]
)
time_seq_ELFO = mat_ELFO["orb"]["seq"]["a"]["t"] - mat_ELFO["orb"]["seq"]["a"]["t"][0]

d = 384400
t = 2*np.pi/2358720
# Using the same starting states of Capstone and Clementine but transformed into CR3BP propagator coordinates
NRHO_init = [1.03545,0,-0.19003,0,-0.13071,5.62991e-07]
ELFO_init = [
    0.98772 + seq_ELFO[0][0] / d,
    seq_ELFO[0][1] / d,
    seq_ELFO[0][2] / d,
    seq_ELFO[0][3],
    seq_ELFO[0][4],
    seq_ELFO[0][5],
]

def cr3bp_state_transition(t, x):
    mu = 0.012150585609624  # Mass ratio between Earth and Moon
    # Unpack the state vector
    x1, x2, x3, vx1, vx2, vx3 = x

    # Distances to primary and secondary bodies
    r1 = np.sqrt((x1 + mu) ** 2 + x2**2 + x3**2)
    r2 = np.sqrt((x1 - 1 + mu) ** 2 + x2**2 + x3**2)

    # Equations of motion
    ddx1 = 2 * vx2 + x1 - (1 - mu) * (x1 + mu) / r1**3 - mu * (x1 - 1 + mu) / r2**3
    ddx2 = -2 * vx1 + x2 - (1 - mu) * x2 / r1**3 - mu * x2 / r2**3
    ddx3 = -(1 - mu) * x3 / r1**3 - mu * x3 / r2**3

    # State transition function
    return np.array([vx1, vx2, vx3, ddx1, ddx2, ddx3])

# Creating ground truth
timestep = 30  # seconds
t_span = (0, 5) # 0.921 is 4 days in normalised time units
t_eval = np.linspace(t_span[0], t_span[1], 100000) # 2.663811e-6 is one second in the normalised time units
# Solve the system
solution_capstone = solve_ivp(
    cr3bp_state_transition, 
    t_span, 
    NRHO_init, 
    t_eval=t_eval, 
    method='LSODA', 
    rtol=1e-9, 
    atol=1e-9
)
solution_ELFO = solve_ivp(
    cr3bp_state_transition, 
    t_span, 
    ELFO_init, 
    t_eval=t_eval, 
    method='LSODA', 
    rtol=1e-9, 
    atol=1e-9
)
# Extracting state vectors
ground_truth = np.vstack((solution_capstone.y, solution_ELFO.y)).T

# Function for converting state vectors to measurements
def state_to_measurements(state):
    pos_diff = state[6:9] - state[0:3]
    vel_diff = state[9:12] - state[3:6]
    return np.array(
        [np.linalg.norm(pos_diff), (pos_diff @ vel_diff) / np.linalg.norm(pos_diff)]
    )

# Correct range and range rate measurements (Every 30 seconds)
ranges = []
range_rates = []
for i in range(len(ground_truth)):
    range_i, range_rate_i = state_to_measurements(
        ground_truth[i]
    )
    # range_i += np.random.normal(
    #     0, 0.1 / d
    # )  # Error is represented with a Gaussian variable with mean 0m and STD 1km
    # range_rate_i += np.random.normal(
    #     0, 0.00001
    # )  # Error is represented with a Gaussian variable with mean 0 and STD 1cm/s
    ranges.append(range_i)
    range_rates.append(range_rate_i)
measurements = np.column_stack((ranges, range_rates))

# Next 2 functions are used to calculate state transition Jacobian
def calculate_F_jacobian(states, f, epsilon=10e-7):
    individual_jacobians = []
    for i in range(len(states)):
        F = np.zeros((6, 6))
        #fx0 = f(states[i])  # Evaluate function at the nominal state

        # Perturb each state variable
        for j in range(6):
            x_perturbed_m = np.copy(states[i])
            x_perturbed_p = np.copy(states[i])
            x_perturbed_p[j] += epsilon
            x_perturbed_m[j] -= epsilon
            f_perturbed_p = f(timestep * 2.663811e-6, x_perturbed_p)
            f_perturbed_m = f(timestep * 2.663811e-6, x_perturbed_m)
            F[:, j] = (f_perturbed_p - f_perturbed_m) / (2*epsilon)  # Approximate the derivative
        individual_jacobians.append(F)

    combined_jacobian = np.block(
        [
            [individual_jacobians[0], np.zeros(individual_jacobians[0].shape)],
            [np.zeros(individual_jacobians[0].shape), individual_jacobians[1]],
        ]
    )

    return combined_jacobian


# Calculating Jacobian for relating states to measurements
def calculate_H_jacobian(state, f, epsilon=1e-7):
    H = np.zeros((2, 12))
    # #f0 = f(state)  # Evaluate the measurement function at the nominal state

    # Perturb each state variable
    for i in range(12):
        state_perturbed_m = np.copy(state)
        state_perturbed_p = np.copy(state)
        state_perturbed_p[i] += epsilon
        state_perturbed_m[i] -= epsilon
        f_perturbed_p = f(state_perturbed_p)
        f_perturbed_m = f(state_perturbed_m)
        H[:, i] = (f_perturbed_p - f_perturbed_m) / (2*epsilon)  # Approximate the derivative

    return H


measurement_covariance = np.diag([0.1 / d, 0.00001])
prev_state_covariance_matrix = np.diag(  # Initial covariance
    [
        1 / d,
        1 / d,
        1 / d,
        2e-5,
        2e-5,
        2e-5,
        1 / d,
        1 / d,
        1 / d,
        0.0003,
        0.0005,
        0.002,
    ]
)*10e-6
prediction_covariance = np.diag(
    [
        1 / d,
        1 / d,
        1 / d,
        0.0001,
        0.0001,
        0.0001,
        1 / d,
        1 / d,
        1 / d,
        0.0001,
        0.0001,
        0.0001,
    ]
)*10e-6

# Creating 'predictions' (initial position error, no prediction error)
NRHO_init_err = NRHO_init + np.array([np.random.normal(0, 1/d), np.random.normal(0, 1/d), np.random.normal(0, 1/d), np.random.normal(0, 2e-5), np.random.normal(0, 2e-5), np.random.normal(0, 2e-5)])/10000
ELFO_init_err = ELFO_init + np.array([np.random.normal(0, 1/d), np.random.normal(0, 1/d), np.random.normal(0, 1/d), np.random.normal(0, 0.0003), np.random.normal(0, 0.0005), np.random.normal(0, 0.002)])/10000
predictions_NRHO = solve_ivp(
    cr3bp_state_transition, 
    t_span, 
    NRHO_init_err, 
    t_eval=t_eval, 
    method='RK45', 
    rtol=1e-9, 
    atol=1e-9
)
predictions_ELFO = solve_ivp(
    cr3bp_state_transition, 
    t_span, 
    ELFO_init_err, 
    t_eval=t_eval, 
    method='RK45', 
    rtol=1e-9, 
    atol=1e-9
)
# Extracting state vectors
predictions = np.vstack((predictions_NRHO.y, predictions_ELFO.y)).T

# The actual Kalman Filter
prev_state_NRHO_EKF = NRHO_init
prev_state_ELFO_EKF = ELFO_init
bad_measurements = []
all_a_posterioris = []
residuals = []
for i in range(len(ground_truth) - 1):
    # Calculate Kalman Gain
    H_jacobian = calculate_H_jacobian(
        np.concatenate([prev_state_NRHO_EKF, prev_state_ELFO_EKF]),
        state_to_measurements,
    )

    F_jacobian = calculate_F_jacobian(
        [prev_state_NRHO_EKF, prev_state_ELFO_EKF], cr3bp_state_transition
    )

    state_covariance_matrix = (
        F_jacobian @ prev_state_covariance_matrix @ F_jacobian.T + prediction_covariance
    )
    kalman_gain = (
        state_covariance_matrix
        @ H_jacobian.T
        @ np.linalg.inv(
            H_jacobian @ state_covariance_matrix @ H_jacobian.T + measurement_covariance
        )
    )

    # Calculate predictions
    # t_span = (0, t_eval[1] - t_eval[0])
    # t_eval_ekf = np.linspace(t_span[0], t_span[1], 2)
    # prediction_capstone = solve_ivp(
    #     cr3bp_state_transition, 
    #     t_span, 
    #     prev_state_capstone_EKF, 
    #     t_eval=t_eval_ekf, 
    #     method='RK45', 
    #     rtol=1e-9, 
    #     atol=1e-9
    # )
    # prediction_ELFO = solve_ivp(
    #     cr3bp_state_transition, 
    #     t_span, 
    #     prev_state_ELFO_EKF, 
    #     t_eval=t_eval_ekf, 
    #     method='RK45', 
    #     rtol=1e-9, 
    #     atol=1e-9
    # )
    # prediction = np.vstack((prediction_capstone.y, prediction_ELFO.y)).T[-1]
    prediction = predictions[i]
    
    pred_range, pred_range_rate = state_to_measurements(prediction)
    pred_measurements = np.array([pred_range, pred_range_rate])
    bad_measurements.append(pred_measurements)
    a_posteriori = prediction + kalman_gain @ (
        measurements[i + 1] - pred_measurements
    )
    prev_state_NRHO_EKF = a_posteriori[0:6]
    prev_state_ELFO_EKF = a_posteriori[6:12]
    residuals.append(measurements[i] - pred_measurements)
    all_a_posterioris.append(a_posteriori)

    # Updating for next step
    prev_state_covariance_matrix = (np.identity(12) - kalman_gain @ H_jacobian) @ state_covariance_matrix @ (np.identity(12) - kalman_gain @ H_jacobian).T + kalman_gain @ measurement_covariance @ kalman_gain.T

print(len(all_a_posterioris))
'''
Plotting
'''
# Create 3D plot of the predicted trajectories
EKF_states_NRHO = np.array([arr[0:6] for arr in all_a_posterioris])
EKF_states_ELFO = np.array([arr[6:12] for arr in all_a_posterioris])
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the trajectory for cr3bp_states_NRHO
ax.plot(
    EKF_states_NRHO[:, 0],
    EKF_states_NRHO[:, 1],
    EKF_states_NRHO[:, 2],
    label="NRHO",
)

# Plot the trajectory for cr3bp_states_ELFO
ax.plot(
    EKF_states_ELFO[:, 0],
    EKF_states_ELFO[:, 1],
    EKF_states_ELFO[:, 2],
    label="ELFO",
)

# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Trajectory of NRHO and ELFO")

# Add legend
ax.legend()
plt.show()
fig.savefig("AWP/output/EKF_Trajectory_of_NRHO_and_ELFO.png")

# Plot range and range rate measurements
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(t_eval[1:len(t_eval)], [arr[0] for arr in bad_measurements][0:len(t_eval) - 1], "o")
ax1.set_xlabel('Time (Seconds)')
ax1.set_ylabel('Range (Earth-to-Moon distances)')
ax1.set_title("Ranges")

ax2.plot(t_eval[1:len(t_eval)], [arr[1] for arr in bad_measurements][0:len(t_eval) - 1], "o")
ax2.set_xlabel('Time (Seconds)')
ax2.set_ylabel('Range rate (km/s)')
ax2.set_title("Range Rates")

plt.tight_layout()
plt.show()
fig.savefig("AWP/output/EKF_Ranges_and_Range_Rates.png")

# Plot residuals
# Plot range and range rate measurements
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(t_eval[1:len(t_eval)], [arr[0] for arr in residuals][0:len(t_eval) - 1], "o")
ax1.set_title("Range residuals")

ax2.plot(t_eval[1:len(t_eval)], [arr[1] for arr in residuals][0:len(t_eval) - 1], "o")
ax2.set_title("Range rate residuals")

plt.tight_layout()
plt.show()
fig.savefig("AWP/output/Residuals.png")

# Plot each state vector value
# Create a 3x4 grid of subplots
fig, axs = plt.subplots(3, 4, figsize=(15, 10))
# Plotting on each subplot
axs[0, 0].plot(t_eval[1:len(t_eval)], np.array([np.array([arr[0] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[0] for arr in ground_truth][0:len(t_eval[1:len(t_eval)])])]).flatten(),'r')
axs[0, 0].set_xlabel('Time (seconds)')
axs[0, 0].set_ylabel('Distance (Earth-to-Moon distances)')
axs[0, 0].set_title('NRHO X')

axs[0, 1].plot(t_eval[1:len(t_eval)], np.array([np.array([arr[1] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[1] for arr in ground_truth][0:len(t_eval[1:len(t_eval)])])]).flatten(), 'r')
axs[0, 1].set_xlabel('Time (seconds)')
axs[0, 1].set_ylabel('Distance (Earth-to-Moon distances)')
axs[0, 1].set_title('NRHO Y')

axs[0, 2].plot(t_eval[1:len(t_eval)], np.array([np.array([arr[2] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[2] for arr in ground_truth][0:len(t_eval[1:len(t_eval)])])]).flatten(), 'r')
axs[0, 2].set_xlabel('Time (seconds)')
axs[0, 2].set_ylabel('Distance (Earth-to-Moon distances)')
axs[0, 2].set_title('NRHO Z')

axs[0, 3].plot(t_eval[1:len(t_eval)], np.array([np.array([arr[3] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[3] for arr in ground_truth][0:len(t_eval[1:len(t_eval)])])]).flatten(), 'r')
axs[0, 3].set_xlabel('Time (seconds)')
axs[0, 3].set_ylabel('Velocity (km/s)')
axs[0, 3].set_title('NRHO vX')

axs[1, 0].plot(t_eval[1:len(t_eval)], np.array([np.array([arr[4] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[4] for arr in ground_truth][0:len(t_eval[1:len(t_eval)])])]).flatten(), 'r')
axs[1, 0].set_xlabel('Time (seconds)')
axs[1, 0].set_ylabel('Velocity (km/s)')
axs[1, 0].set_title('NRHO vY')

axs[1, 1].plot(t_eval[1:len(t_eval)], np.array([np.array([arr[5] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[5] for arr in ground_truth][0:len(t_eval[1:len(t_eval)])])]).flatten(), 'r')
axs[1, 1].set_xlabel('Time (seconds)')
axs[1, 1].set_ylabel('Velocity (km/s)')
axs[1, 1].set_title('NRHO vZ')

axs[1, 2].plot(t_eval[1:len(t_eval)], np.array([np.array([arr[6] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[6] for arr in ground_truth][0:len(t_eval[1:len(t_eval)])])]).flatten(), 'r')
axs[1, 2].set_xlabel('Time (seconds)')
axs[1, 2].set_ylabel('Distance (Earth-to-Moon distances)')
axs[1, 2].set_title('ELFO X')

axs[1, 3].plot(t_eval[1:len(t_eval)], np.array([np.array([arr[7] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[7] for arr in ground_truth][0:len(t_eval[1:len(t_eval)])])]).flatten(), 'r')
axs[1, 3].set_xlabel('Time (seconds)')
axs[1, 3].set_ylabel('Distance (Earth-to-Moon distances)')
axs[1, 3].set_title('ELFO Y')

axs[2, 0].plot(t_eval[1:len(t_eval)], np.array([np.array([arr[8] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[8] for arr in ground_truth][0:len(t_eval[1:len(t_eval)])])]).flatten(), 'r')
axs[2, 0].set_xlabel('Time (seconds)')
axs[2, 0].set_ylabel('Distance (Earth-to-Moon distances)')
axs[2, 0].set_title('ELFO Z')

axs[2, 1].plot(t_eval[1:len(t_eval)], np.array([np.array([arr[9] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[9] for arr in ground_truth][0:len(t_eval[1:len(t_eval)])])]).flatten(), 'r')
axs[2, 1].set_xlabel('Time (seconds)')
axs[2, 1].set_ylabel('Velocity (km/s)')
axs[2, 1].set_title('ELFO vX')

axs[2, 2].plot(t_eval[1:len(t_eval)], np.array([np.array([arr[10] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[10] for arr in ground_truth][0:len(t_eval[1:len(t_eval)])])]).flatten(), 'r')
axs[2, 2].set_xlabel('Time (seconds)')
axs[2, 2].set_ylabel('Velocity (km/s)')
axs[2, 2].set_title('ELFO vY')

axs[2, 3].plot(t_eval[1:len(t_eval)], np.array([np.array([arr[11] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[11] for arr in ground_truth][0:len(t_eval[1:len(t_eval)])])]).flatten(), 'r')
axs[2, 3].set_xlabel('Time (seconds)')
axs[2, 3].set_ylabel('Velocity (km/s)')
axs[2, 3].set_title('ELFO vZ')

plt.tight_layout()
plt.show()
fig.savefig("AWP/output/errors.png")
