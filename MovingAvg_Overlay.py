import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Default downsampling factor
DOWN_SAMPLE_DEFAULT = 175

# Function to calculate moving average
def calculate_moving_average(data, window_size):
    avg = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    # Set the edges to NaN to avoid spikes
    avg[:window_size // 2] = np.nan
    avg[-window_size // 2:] = np.nan
    return avg

def process_dataset(df, position_col, torque_col, downsample_factor):
    
    start_index = df[df[position_col].apply(lambda x: int(x) == 2)].index.min() # Identify where ActualPosition [rad] starts with 2
    if pd.isna(start_index):
        return None  

    df = df.iloc[start_index:].reset_index(drop=True)

    rotations = []
    last_position = df[position_col].iloc[0]  
    rotation_data = []

    for _, row in df.iterrows():
        position = row[position_col]
        torque = row[torque_col]

        if position - last_position >= 6.28318531:  
            rotations.append(rotation_data)
            rotation_data = []
            last_position = position
        rotation_data.append((position, torque))

    if rotation_data:  
        rotations.append(rotation_data)

    downsampled_rotations = {}
    for i, rotation in enumerate(rotations):
        positions = [p for p, _ in rotation]
        torques = [t for _, t in rotation]

        position_start = positions[0]
        relative_positions = [p - position_start for p in positions]

        degrees_positions = np.degrees([p % (2 * np.pi) for p in relative_positions])

        downsampled_torques = [
            np.mean(torques[j:j + downsample_factor]) for j in range(0, len(torques), downsample_factor)
        ]

        max_degrees = degrees_positions[-1]
        downsampled_positions = np.linspace(0, max_degrees, len(downsampled_torques))

        if max_degrees < 360:
            extra_positions = np.linspace(max_degrees, 360, 10)  
            extra_torques = [downsampled_torques[-1]] * len(extra_positions)
            downsampled_positions = np.concatenate((downsampled_positions, extra_positions))
            downsampled_torques = np.concatenate((downsampled_torques, extra_torques))

        downsampled_rotations[f"Rotation {i+1}"] = {
            "positions": downsampled_positions,
            "torques": downsampled_torques
        }

    return downsampled_rotations

# Function to plot moving averages of each rotation on separate subplots
def plot_moving_averages_per_rotation(rotations, moving_avg_window, file_name):
    
    rotations = {label: data for i, (label, data) in enumerate(rotations.items()) if i < len(rotations) - 1}

    all_moving_avgs = []
    for data in rotations.values():
        torques = data["torques"]
        moving_avg = calculate_moving_average(torques, moving_avg_window)
        all_moving_avgs.extend(moving_avg[~np.isnan(moving_avg)])  # Exclude NaN values

    y_min, y_max = min(all_moving_avgs), max(all_moving_avgs)

    fig, ax = plt.subplots(len(rotations), 1, figsize=(8, 2 * len(rotations)), sharex=True, constrained_layout=True)

    fig.suptitle(f"Torque and Position Analysis - {file_name}", fontsize=14)

    if len(rotations) == 1:
        ax = [ax]

    for i, (label, data) in enumerate(rotations.items()):
        positions = data["positions"]
        torques = data["torques"]

        moving_avg = calculate_moving_average(torques, moving_avg_window)

        ax[i].plot(positions, moving_avg, color="darkblue", linewidth=2, label=label)

        ax[i].set_ylabel("Torque", fontsize=9)
        ax[i].grid(True)
        ax[i].set_ylim(y_min, y_max) 
        ax[i].set_xticks(np.arange(0, 361, 20))  
        ax[i].tick_params(axis='x', labelsize=8)
        ax[i].tick_params(axis='y', labelsize=8)

        ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8, frameon=False)

    ax[-1].set_xlabel("Position [°]", fontsize=10)  # Set x-axis label for the last subplot
    st.pyplot(fig)

# Streamlit App UI
st.title("Torque and Position Analyzer")
st.write("Upload datasets to visualize moving averages of each rotation on separate diagrams.")

downsample_factor = st.sidebar.slider("Downsampling Size", min_value=10, max_value=400, value=DOWN_SAMPLE_DEFAULT, step=5)

moving_avg_window = st.sidebar.slider("Moving Average Window Size", min_value=1, max_value=100, value=20, step=1)

uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

if uploaded_file:
    st.subheader(f"File: {uploaded_file.name}")
    df = pd.read_csv(uploaded_file, delimiter='\t')

    df = df.drop(columns=['Unnamed: 3'], errors='ignore')

    position_col = "ActualPosition [rad]"
    torque_col = "Actual Torque [of nominal]"

    if position_col not in df.columns or torque_col not in df.columns:
        st.error(f"File {uploaded_file.name} does not contain the required columns: {position_col}, {torque_col}")
    else:
        
        timeframes = process_dataset(df, position_col, torque_col, downsample_factor)

        if timeframes is None:
            st.warning(f"File {uploaded_file.name} does not contain valid data starting with a minimum {position_col} value of 2.")
        else:
            plot_moving_averages_per_rotation(timeframes, moving_avg_window, uploaded_file.name)
else:
    st.info("Upload a CSV file to begin.")
