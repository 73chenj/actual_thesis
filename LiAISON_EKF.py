"""
The LiAISON EKF using the ELFO and NRHO orbits

CR3BP model as ground truth
The unit of distance is normalised such that the distance between earth and moon is 1
Frame of reference is rotational centred at the barycentre of earth and moon
The unit of time is normalised such that the orbital period of the moon around the Earth is 2pi
To convert velocity from km/s to this frame you multiply it by 0.977

state vector is [x1, y1, z1, vx1, vy1, vz1]

Estimating only the ELFO satellite, assume the NRHO orbit is known
"""

import numpy as np
from scipy.integrate import solve_ivp
import spiceypy as sp
import matplotlib.pyplot as plt
import sympy
import time

d = 384400
t = 2*np.pi/2358720
# Using the same starting states of NRHO and Clementine but transformed into CR3BP coordinates
NRHO_init = [1.03545,0,-0.19003,0,-0.13071,5.62991e-07]
ELFO_init = [0.9822609090260146, 0.0033780550309573357, -0.009090163078043704, 0.8948704814799997, 0.17693000307000004, -0.15601097269]

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

def propagate(x, dt):
    # Creating ground truth
    t_span = (0, dt)
    t_eval = np.linspace(t_span[0], t_span[1], 2)
    solution_ELFO = solve_ivp(
        cr3bp_state_transition, 
        t_span, 
        x, 
        t_eval=t_eval, 
        method='RK45', 
        rtol=1e-9, 
        atol=1e-12
    )
    # Extracting state vectors
    prediction = np.vstack(solution_ELFO.y).T[-1]
    return prediction

def state_to_measurements(state):
    pos_diff = state[0:3] - state[6:9]
    vel_diff = state[3:6] - state[9:12]
    return np.array(
        [np.linalg.norm(pos_diff), (pos_diff/np.linalg.norm(pos_diff)) @ vel_diff ]
    )

'''
Generate ground truth
'''
# Creating ground truth
n_steps = 50000
t_span = (0, 6) # 0.921 is 4 days in normalised time units
t_eval = np.linspace(t_span[0], t_span[1], n_steps)
dt = t_span[1]/n_steps
d = 384400
# Solve the system
solution_NRHO = solve_ivp(
    cr3bp_state_transition, 
    t_span, 
    NRHO_init, 
    t_eval=t_eval, 
    method='RK45', 
    rtol=1e-9, 
    atol=1e-12
)
solution_ELFO = solve_ivp(
    cr3bp_state_transition, 
    t_span, 
    ELFO_init, 
    t_eval=t_eval, 
    method='RK45', 
    rtol=1e-9, 
    atol=1e-12
)
# Extracting state vectors
ground_truth_ELFO = np.vstack(solution_ELFO.y).T
ground_truth_NRHO = np.vstack(solution_NRHO.y).T

# Next 2 functions are used to calculate state transition Jacobian
def calculate_F_jacobian(state, propagate_func, dt, epsilon=1e-5):
    n = len(state)
    F = np.zeros((n, n))

    for j in range(n):
        x_perturbed_m = np.copy(state)
        x_perturbed_p = np.copy(state)
        x_perturbed_p[j] += epsilon
        x_perturbed_m[j] -= epsilon

        f_perturbed_p = propagate_func(x_perturbed_p, dt)
        f_perturbed_m = propagate_func(x_perturbed_m, dt)

        F[:, j] = (f_perturbed_p - f_perturbed_m) / (2 * epsilon)

    return F

# Calculating Jacobian for relating states to measurements
def calculate_H_jacobian(state, f, epsilon=1e-5):
    H = np.zeros((2, 6))
    # #f0 = f(state)  # Evaluate the measurement function at the nominal state

    # Perturb each state variable
    for i in range(6):
        state_perturbed_m = np.copy(state)
        state_perturbed_p = np.copy(state)
        state_perturbed_p[i + 6] += epsilon
        state_perturbed_m[i + 6] -= epsilon
        f_perturbed_p = f(state_perturbed_p)
        f_perturbed_m = f(state_perturbed_m)
        H[:, i] = (f_perturbed_p - f_perturbed_m) / (2*epsilon)  # Approximate the derivative

    return H

R = np.diag(((1/d)**2, (0.977/1000)**2))
R_noise = np.diag(((1/d)**2, (0.977/1000)**2))
P = np.diag((1/(d**2), 1/(d**2), 1/(d**2), (0.977/1000)**2, (0.977/1000)**2, (0.977/1000)**2))
Q = np.diag((1/(d**2), 1/(d**2), 1/(d**2), (0.977/1000)**2, (0.977/1000)**2, (0.977/1000)**2))*1e-10
Q_noise = np.diag((1/(d**2), 1/(d**2), 1/(d**2), (0.977/1000)**2, (0.977/1000)**2, (0.977/1000)**2))*1e-10

# Creating 'predictions' (initial position error, no prediction error)
ELFO_init_err = ELFO_init + np.random.multivariate_normal([0, 0, 0, 0, 0, 0], P)
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
predictions = np.vstack(predictions_ELFO.y).T

# The actual Kalman Filter
prev_state_ELFO_EKF = ELFO_init
bad_measurements = []
all_a_posterioris = []
residuals = []
covariances = []
start_time = time.time()
for i in range(len(ground_truth_NRHO) - 1):
    covariances.append(np.diag(P))
    all_a_posterioris.append(prev_state_ELFO_EKF)
    # Calculate Kalman Gain
    H_jacobian = calculate_H_jacobian(
        np.concatenate([ground_truth_NRHO[i], prev_state_ELFO_EKF]),
        state_to_measurements,
    )

    F_jacobian = calculate_F_jacobian(
        prev_state_ELFO_EKF, propagate, dt
    )

    state_covariance_matrix = (
        F_jacobian @ P @ F_jacobian.T + Q
    )
    kalman_gain = (
        state_covariance_matrix
        @ H_jacobian.T
        @ np.linalg.inv(
            H_jacobian @ state_covariance_matrix @ H_jacobian.T + R
        )
    )

    # Calculate predictions
    prediction = propagate(prev_state_ELFO_EKF, dt) + np.random.multivariate_normal([0, 0, 0, 0, 0, 0], Q_noise)

    # Measurement Update
    pred_range, pred_range_rate = state_to_measurements(np.concatenate((ground_truth_NRHO[i + 1], prediction)))
    pred_measurements = np.array([pred_range, pred_range_rate])
    true_range, true_range_rate = state_to_measurements(np.concatenate((ground_truth_NRHO[i + 1], ground_truth_ELFO[i + 1])))
    true_measurements = np.array([true_range, true_range_rate]) + np.random.multivariate_normal([0, 0], R_noise)
    residual = true_measurements - pred_measurements
    a_posteriori = prediction #+ kalman_gain @ (
    #     residual
    # )
    prev_state_ELFO_EKF = a_posteriori
    residuals.append(true_measurements)
    
    # Updating for next step
    P = state_covariance_matrix#(np.identity(6) - kalman_gain @ H_jacobian) @ state_covariance_matrix @ (np.identity(6) - kalman_gain @ H_jacobian).T + kalman_gain @ R @ kalman_gain.T
    if i >= 45830:
        a = 1
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")
# print(ground_truth_ELFO[41000:41003])
# print(residuals[41000:41003])
# print(predictions[41000:41003])
# print(all_a_posterioris[41000:41003])
# plt.plot(t_eval[1:len(t_eval)], KG_value)
# plt.show()
# exit()
'''
Plotting
'''
# # Create 3D plot of the predicted trajectories
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# # Plot the trajectory for cr3bp_states_NRHO
# ax.scatter(
#     ground_truth_NRHO[:, 0],
#     ground_truth_NRHO[:, 1],
#     ground_truth_NRHO[:, 2],
#     label="NRHO",
#     s=3
# )

# EKF_states_ELFO = np.array([arr[0:6] for arr in all_a_posterioris])
# # Plot the trajectory for cr3bp_states_ELFO
# ax.scatter(
#     EKF_states_ELFO[:, 0],
#     EKF_states_ELFO[:, 1],
#     EKF_states_ELFO[:, 2],
#     label="ELFO",
#     s=3
# )

# # Set labels and title
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("3D Trajectory of NRHO and ELFO")

# # Add legend
# ax.legend()
# plt.show()
# fig.savefig("AWP/output/EKF_Trajectory_of_NRHO_and_ELFO.png")

# # Plot range and range rate measurements
# fig, (ax1, ax2) = plt.subplots(2, 1)

# ax1.plot(t_eval[1:len(t_eval)], [arr[0] for arr in bad_measurements][0:len(t_eval) - 1], "o")
# ax1.set_xlabel('Time (Seconds)')
# ax1.set_ylabel('Range (Earth-to-Moon distances)')
# ax1.set_title("Ranges")

# ax2.plot(t_eval[1:len(t_eval)], [arr[1] for arr in bad_measurements][0:len(t_eval) - 1], "o")
# ax2.set_xlabel('Time (Seconds)')
# ax2.set_ylabel('Range rate (km/s)')
# ax2.set_title("Range Rates")

# plt.tight_layout()
# plt.show()
# fig.savefig("AWP/output/EKF_Ranges_and_Range_Rates.png")

# Plot residuals
# Plot range and range rate measurements
# fig, (ax1, ax2) = plt.subplots(2, 1)

# ax1.plot(t_eval[1:len(t_eval)]*4.34, [arr[0]*d for arr in residuals][0:len(t_eval)], "o")
# ax1.set_title("Range")
# ax1.set_xlabel("Time (days)")  # Add label for x-axis
# ax1.set_ylabel("Range (km)")  # Add label for y-axis

# ax2.plot(t_eval[1:len(t_eval)]*4.34, [arr[1] for arr in residuals][0:len(t_eval)], "o")
# ax2.set_title("Range rate")
# ax2.set_xlabel("Time")  # Add label for x-axis
# ax2.set_ylabel("Range Rate (km/s)")  # Add label for y-axis

# plt.tight_layout()
# plt.show()
# fig.savefig("AWP/output/ground_truth_measurements.png")

# Plot each state vector value
# Create a 3x4 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
# Plotting on each subplot
axs[0, 0].plot(t_eval[1:len(t_eval)]*4.34, np.array([np.array([arr[0] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[0] for arr in ground_truth_ELFO][0:len(t_eval[1:len(t_eval)])])]).flatten()*d, 'r')
axs[0, 0].set_xlabel('Time (days)')
axs[0, 0].set_ylabel('Distance (km)')
axs[0, 0].set_title('x')

axs[0, 1].plot(t_eval[1:len(t_eval)]*4.34, np.array([np.array([arr[1] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[1] for arr in ground_truth_ELFO][0:len(t_eval[1:len(t_eval)])])]).flatten()*d, 'r')
axs[0, 1].set_xlabel('Time (days)')
axs[0, 1].set_ylabel('Distance (km)')
axs[0, 1].set_title('y')

axs[0, 2].plot(t_eval[1:len(t_eval)]*4.34, np.array([np.array([arr[2] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[2] for arr in ground_truth_ELFO][0:len(t_eval[1:len(t_eval)])])]).flatten()*d, 'r')
axs[0, 2].set_xlabel('Time (days)')
axs[0, 2].set_ylabel('Distance (km)')
axs[0, 2].set_title('z')

axs[1, 0].plot(t_eval[1:len(t_eval)]*4.34, np.array([np.array([arr[3] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[3] for arr in ground_truth_ELFO][0:len(t_eval[1:len(t_eval)])])]).flatten()*1000, 'r')
axs[1, 0].set_xlabel('Time (days)')
axs[1, 0].set_ylabel('Velocity (m/s)')
axs[1, 0].set_title('vX')

axs[1, 1].plot(t_eval[1:len(t_eval)]*4.34, np.array([np.array([arr[4] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[4] for arr in ground_truth_ELFO][0:len(t_eval[1:len(t_eval)])])]).flatten()*1000, 'r')
axs[1, 1].set_xlabel('Time (days)')
axs[1, 1].set_ylabel('Velocity (m/s)')
axs[1, 1].set_title('vY')

axs[1, 2].plot(t_eval[1:len(t_eval)]*4.34, np.array([np.array([arr[5] for arr in all_a_posterioris][0:len(t_eval[1:len(t_eval)])]) - np.array([arr[5] for arr in ground_truth_ELFO][0:len(t_eval[1:len(t_eval)])])]).flatten()*1000, 'r')
axs[1, 2].set_xlabel('Time (days)')
axs[1, 2].set_ylabel('Velocity (m/s)')
axs[1, 2].set_title('vZ')

plt.tight_layout()
plt.show()
fig.savefig("AWP/output/errors.png")

# Plot each state vector value
# Create a 3x4 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
# Plotting on each subplot
axs[0, 0].plot(t_eval[1:len(t_eval)]*4.34, np.array([arr[0] for arr in covariances][0:len(t_eval[1:len(t_eval)])])*d*d, 'b')
axs[0, 0].set_xlabel('Days')
axs[0, 0].set_ylabel('km$^2$')
axs[0, 0].set_title('x')

axs[0, 1].plot(t_eval[1:len(t_eval)]*4.34, np.array([arr[1] for arr in covariances][0:len(t_eval[1:len(t_eval)])])*d*d, 'b')
axs[0, 1].set_xlabel('Days')
axs[0, 1].set_ylabel('km$^2$')
axs[0, 1].set_title('y')

axs[0, 2].plot(t_eval[1:len(t_eval)]*4.34, np.array([arr[2] for arr in covariances][0:len(t_eval[1:len(t_eval)])])*d*d, 'b')
axs[0, 2].set_xlabel('Days')
axs[0, 2].set_ylabel('km$^2$')
axs[0, 2].set_title('z')

axs[1, 0].plot(t_eval[1:len(t_eval)]*4.34, np.array([arr[3] for arr in covariances][0:len(t_eval[1:len(t_eval)])])*((1000/0.977)**2), 'b')
axs[1, 0].set_xlabel('Days')
axs[1, 0].set_ylabel('(m/s)$^2$')
axs[1, 0].set_title('vX')

axs[1, 1].plot(t_eval[1:len(t_eval)]*4.34, np.array([arr[4] for arr in covariances][0:len(t_eval[1:len(t_eval)])])*((1000/0.977)**2), 'b')
axs[1, 1].set_xlabel('Days')
axs[1, 1].set_ylabel('(m/s)$^2$')
axs[1, 1].set_title('vY')

axs[1, 2].plot(t_eval[1:len(t_eval)]*4.34, np.array([arr[5] for arr in covariances][0:len(t_eval[1:len(t_eval)])])*((1000/0.977)**2), 'b')
axs[1, 2].set_xlabel('Days')
axs[1, 2].set_ylabel('(m/s)$^2$')
axs[1, 2].set_title('vZ')

plt.tight_layout()
plt.show()
fig.savefig("AWP/output/covariances.png")
