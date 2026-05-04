import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid

def extract_global_acceleration(file_path, sensor_id, cutoff_freq=0.5):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None

    header_idx = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.startswith("Time"):
                header_idx = i
                break

    df_raw = pd.read_csv(file_path, sep='\t', skiprows=header_idx, low_memory=False)
    df_raw.columns = df_raw.columns.str.strip()

    time_col = next(col for col in df_raw.columns if "Time" in col)
    time = df_raw[time_col].values
    fs = 1.0 / np.mean(np.diff(time))

    def get_sensor_col(data_type, axis):
        matches = [c for c in df_raw.columns if f"Imu_{sensor_id}" in c and data_type in c and axis in c]
        if matches: return df_raw[matches[0]].values
        return np.zeros(len(time))

    # --- Extract Local Data ---
    local_x = get_sensor_col('Acc', 'X')
    local_y = get_sensor_col('Acc', 'Y')
    local_z = get_sensor_col('Acc', 'Z')
    local_acc = np.column_stack((local_x, local_y, local_z))

    gyro_x = get_sensor_col('Gyro', 'X')
    gyro_y = get_sensor_col('Gyro', 'Y')
    gyro_z = get_sensor_col('Gyro', 'Z')
    local_gyro = np.column_stack((gyro_x, gyro_y, gyro_z))

    # --- Gravity Alignment ---
    nyq = 0.5 * fs
    b, a = butter(2, cutoff_freq / nyq, btype='low', analog=False)
    gx = filtfilt(b, a, local_x)
    gy = filtfilt(b, a, local_y)
    gz = filtfilt(b, a, local_z)

    grav_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    Z_g = np.column_stack((gx / grav_mag, gy / grav_mag, gz / grav_mag))

    ref = np.zeros_like(Z_g)
    use_y = np.abs(Z_g[:, 0]) > 0.9
    ref[~use_y, 0] = 1.0
    ref[use_y, 1] = 1.0

    dot_ref_Z = np.sum(ref * Z_g, axis=1, keepdims=True)
    X_g = ref - dot_ref_Z * Z_g
    X_g_norm = np.linalg.norm(X_g, axis=1, keepdims=True)
    X_g = X_g / X_g_norm
    Y_g = np.cross(Z_g, X_g)

    # --- Project to Global Axes ---
    global_acc_x = np.sum(local_acc * X_g, axis=1)
    global_acc_y = np.sum(local_acc * Y_g, axis=1)
    global_acc_z = np.sum(local_acc * Z_g, axis=1) - 1.0  # Subtract 1g to center around 0

    global_gyro_z_raw = np.sum(local_gyro * Z_g, axis=1)

    # --- GYRO PROCESSING & INTEGRATION ---
    # 1. Low-Pass Filter (remove impacts)
    b_gyro, a_gyro = butter(2, 5.0 / nyq, btype='low', analog=False)
    global_gyro_z_clean = filtfilt(b_gyro, a_gyro, global_gyro_z_raw)

    # 2. Dynamic Baseline Subtraction (The Median Trick)
    gyro_bias = np.median(global_gyro_z_clean)
    gyro_z_unbiased = global_gyro_z_clean - gyro_bias

    # 3. Integrate to get Angle (Degrees)
    yaw_angle_clean = cumulative_trapezoid(gyro_z_unbiased, time, initial=0)

    # --- THE MISSING PART: Total Acceleration Magnitude ---
    # The magnitude of a vector is invariant to rotation, so we can just use the local sensors directly
    total_acc_mag = np.linalg.norm(local_acc, axis=1)

    result_df = pd.DataFrame({
        'Time': time,
        'Global_X': global_acc_x,
        'Global_Y': global_acc_y,
        'Global_Z': global_acc_z,
        'Global_Gyro_Z': global_gyro_z_clean,
        'Yaw_Angle': yaw_angle_clean,
        'Total_Acc_Mag': total_acc_mag  # <-- Added this back!
    })

    return result_df