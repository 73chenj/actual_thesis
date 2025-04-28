'''
LiAISON UKF with one estimate
'''
import numpy as np
from scipy.integrate import solve_ivp
import spiceypy as sp
import matplotlib.pyplot as plt
from filterpy.kalman import MerweScaledSigmaPoints
import time

NRHO_init = [1.03545,0,-0.19003,0,-0.13071,5.62991e-07]
ELFO_init = [0.9822609090260146, 0.0033780550309573357, -0.009090163078043704, 0.8948704814799997, 0.17693000307000004, -0.15601097269]

'''
Functions
'''
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
        rtol=1e-12, 
        atol=1e-14
    )
    # Extracting state vectors
    prediction = np.vstack(solution_ELFO.y).T[-1]
    return prediction

def state_to_measurements(state, NRHO):
    pos_diff = state[0:3] - NRHO[0:3]
    vel_diff = state[3:6] - NRHO[3:6]
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
d = 384400
# Solve the system
solution_NRHO = solve_ivp(
    cr3bp_state_transition, 
    t_span, 
    NRHO_init, 
    t_eval=t_eval, 
    method='RK45', 
    rtol=1e-12, 
    atol=1e-14
)
solution_ELFO = solve_ivp(
    cr3bp_state_transition, 
    t_span, 
    ELFO_init, 
    t_eval=t_eval, 
    method='RK45', 
    rtol=1e-12, 
    atol=1e-14
)
# Extracting state vectors
ground_truth_ELFO = np.vstack(solution_ELFO.y).T
ground_truth_NRHO = np.vstack(solution_NRHO.y).T

ranges = []
range_rates = []
ground_truth_total = np.vstack((solution_NRHO.y, solution_ELFO.y)).T
for i in range(len(ground_truth_NRHO)):
    range_i, range_rate_i = state_to_measurements(
        ground_truth_ELFO[i], ground_truth_NRHO[i]
    )
    ranges.append(range_i)
    range_rates.append(range_rate_i)
measurements = np.column_stack((ranges, range_rates))

'''
UKF algorithm
'''
# UKF Parameters
dt = t_span[1]/n_steps  # Time step
dim_x = 6  # State dimension [x, y, z, vx, vy, vz]
dim_z = 2  # Measurement dimension [range, range rate]

# Create sigma points generator
points = MerweScaledSigmaPoints(n=dim_x, alpha=0.001, beta=2, kappa=-3)

# Initial state and covariance
P = np.diag((1/(d**2), 1/(d**2), 1/(d**2), (0.977/1000)**2, (0.977/1000)**2, (0.977/1000)**2))  # Small initial uncertainty
x = np.array([0.9822609090260146, 0.0033780550309573357, -0.009090163078043704, 0.8948704814799997, 0.17693000307000004, -0.15601097269]) + np.random.multivariate_normal([0, 0, 0, 0, 0, 0], P)

# Process and measurement noise
Q = np.diag((1/(d**2), 1/(d**2), 1/(d**2), (0.977/1000)**2, (0.977/1000)**2, (0.977/1000)**2))*1e-10 # Process noise
R = np.diag(((1/d)**2, (0.977/1000)**2))
R_noise = np.diag(((0.0001/d)**2, (0.977*0.003/1000)**2)) # Measurement noise

# Store results for analysis
state_history = []
measurement_history = []
residuals = []
covariances = []
spread_history = []

# Run UKF over multiple time steps
start_time = time.time()
for i in range(len(ground_truth_ELFO) - 1):
    covariances.append(np.diag(P))
    # Generate sigma points and weights
    try:
        sigma_pts = points.sigma_points(x, P)
        Wm, Wc = points.Wm, points.Wc  # Mean and covariance weights
    except np.linalg.LinAlgError:
        P += np.eye(P.shape[0])
        print(P)
        sigma_pts = points.sigma_points(x, P)
        Wm, Wc = points.Wm, points.Wc  # Mean and covariance weights

    # Store spread of sigma points
    spread = np.linalg.norm(sigma_pts - x, axis=1)
    spread_history.append(np.mean(spread))
    
    # --- Prediction Step ---
    sigma_pts_pred = np.array([propagate(pt, dt) for pt in sigma_pts])  # Propagate through CR3BP

    # Predicted mean
    x_pred = np.sum(Wm[:, None] * sigma_pts_pred, axis=0) + np.random.multivariate_normal([0, 0, 0, 0, 0, 0], Q)

    # Predicted covariance
    P_pred = Q + np.sum(Wc[:, None, None] * (sigma_pts_pred - x_pred)[..., None] @ (sigma_pts_pred - x_pred)[:, None, :], axis=0)

    # --- Update Step ---
    sigma_pts_meas = np.array([state_to_measurements(pt, ground_truth_NRHO[1]) for pt in sigma_pts_pred])  # Transform to measurement space

    # Predicted measurement mean
    z_pred = np.sum(Wm[:, None] * sigma_pts_meas, axis=0)

    # Measurement covariance
    P_zz = R + np.sum(Wc[:, None, None] * (sigma_pts_meas - z_pred)[..., None] @ (sigma_pts_meas - z_pred)[:, None, :], axis=0)

    # Cross covariance
    P_xz = np.sum(Wc[:, None, None] * (sigma_pts_pred - x_pred)[..., None] @ (sigma_pts_meas - z_pred)[:, None, :], axis=0)

    # Kalman Gain
    K = P_xz @ np.linalg.inv(P_zz)

    # Generate synthetic measurement (for now, assume perfect measurement + small noise)
    true_z = state_to_measurements(ground_truth_ELFO[i + 1], ground_truth_NRHO[1])  # Ideal measurement
    z = true_z + np.random.multivariate_normal([0, 0], R)  # Add measurement noise

    # State update
    x = x_pred #+ K @ (z - z_pred)
    P = P_pred #- K @ P_zz @ K.T
    residuals.append(state_to_measurements(ground_truth_ELFO[i + 1], x))
    # Store history
    state_history.append(x.copy())
    measurement_history.append(z.copy())
    
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")
'''
Plotting
'''
# Plot sigma point spreads
plt.plot(spread_history)
plt.xlabel("Time Step")
plt.ylabel("Mean Sigma Point Spread")
plt.title("Sigma Point Spread Over Time")
plt.show()

# Create 3D plot of the predicted trajectories
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the trajectory for cr3bp_states_NRHO
ax.scatter(
    ground_truth_ELFO[:, 0],
    ground_truth_ELFO[:, 1],
    ground_truth_ELFO[:, 2],
    label="Satellite B",
    s=1
)

UKF_states_ELFO = np.array([arr[0:6] for arr in state_history])
# Plot the trajectory for cr3bp_states_ELFO
ax.scatter(
    UKF_states_ELFO[:, 0],
    UKF_states_ELFO[:, 1],
    UKF_states_ELFO[:, 2],
    label="Satellite A",
    s=1
)

# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Trajectory of NRHO and ELFO")

# Add legend
ax.legend()
plt.show()

# Plot residuals
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(t_eval[0:len(UKF_states_ELFO)], [arr[0]*d for arr in residuals][0:len(t_eval)], "o")
ax1.set_title("Range residuals")

ax2.plot(t_eval[0:len(UKF_states_ELFO)], [arr[1] for arr in residuals][0:len(t_eval)], "o")
ax2.set_title("Range rate residuals")

plt.tight_layout()
plt.show()

# Plot each state vector value
# Create a 3x4 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Error Evolution of UKF State Estimates')
ground_truth_ELFO = ground_truth_ELFO[1:len(ground_truth_ELFO)]

# Plotting on each subplot
axs[0, 0].plot(t_eval[0:len(UKF_states_ELFO)]*4.34, np.array([np.array([arr[0] for arr in UKF_states_ELFO])] - np.array([arr[0] for arr in ground_truth_ELFO])).flatten()*d, 'r')
axs[0, 0].set_xlabel('Time (days)')
axs[0, 0].set_ylabel('Distance (km)')
axs[0, 0].set_title('x')

axs[0, 1].plot(t_eval[0:len(UKF_states_ELFO)]*4.34, np.array([np.array([arr[1] for arr in UKF_states_ELFO]) - np.array([arr[1] for arr in ground_truth_ELFO])]).flatten()*d, 'r')
axs[0, 1].set_xlabel('Time (days)')
axs[0, 1].set_ylabel('Distance (km)')
axs[0, 1].set_title('y')

axs[0, 2].plot(t_eval[0:len(UKF_states_ELFO)]*4.34, np.array([np.array([arr[2] for arr in UKF_states_ELFO]) - np.array([arr[2] for arr in ground_truth_ELFO])]).flatten()*d, 'r')
axs[0, 2].set_xlabel('Time (days)')
axs[0, 2].set_ylabel('Distance (km)')
axs[0, 2].set_title('z')

axs[1, 0].plot(t_eval[0:len(UKF_states_ELFO)]*4.34, np.array([np.array([arr[3] for arr in UKF_states_ELFO]) - np.array([arr[3] for arr in ground_truth_ELFO])]).flatten()*1000, 'r')
axs[1, 0].set_xlabel('Time (days)')
axs[1, 0].set_ylabel('Velocity (m/s)')
axs[1, 0].set_title('Vx')

axs[1, 1].plot(t_eval[0:len(UKF_states_ELFO)]*4.34, np.array([np.array([arr[4] for arr in UKF_states_ELFO]) - np.array([arr[4] for arr in ground_truth_ELFO])]).flatten()*1000, 'r')
axs[1, 1].set_xlabel('Time (days)')
axs[1, 1].set_ylabel('Velocity (m/s)')
axs[1, 1].set_title('Vy')

axs[1, 2].plot(t_eval[0:len(UKF_states_ELFO)]*4.34, np.array([np.array([arr[5] for arr in UKF_states_ELFO]) - np.array([arr[5] for arr in ground_truth_ELFO])]).flatten()*1000, 'r')
axs[1, 2].set_xlabel('Time (days)')
axs[1, 2].set_ylabel('Velocity (m/s)')
axs[1, 2].set_title('Vz')

plt.tight_layout()
plt.show()

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
# TODO: Try plotting the absolute errors for distance and velocity instead of splitting them into components.
