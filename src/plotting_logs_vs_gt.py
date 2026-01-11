import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

def load_data(est_path, gt_path):

    # Load CSV files
    est = pd.read_csv(est_path, sep=",")
    gt = pd.read_csv(gt_path, sep=";", decimal=",")

    # Normalize column names
    est = normalize_columns(est)
    gt = normalize_columns(gt)

    # Ensure required columns exist
    for col in ["time", "x-pos", "y-pos", "z-pos"]:
        est[col] = pd.to_numeric(est[col], errors="coerce")
        gt[col] = pd.to_numeric(gt[col], errors="coerce")

    # Convert positions from meters to centimeters
    est[["x-pos", "y-pos", "z-pos"]] *= 100.0

    return est.sort_values("time"), gt.sort_values("time")

def normalize_columns(df):

    # Rename columns for consistency
    rename_map = {
        "finger": "finger_idx",
        "x": "x-pos",
        "y": "y-pos",
        "z": "z-pos"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def interpolate_gt(gt_df, target_times, axis):

    # Create interpolation function
    interp_fn = interp1d(
        gt_df["time"],
        gt_df[axis],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan
    )
    return interp_fn(target_times)

def apply_time_window(df, t_start=None, t_stop=None):

    # Apply time window filtering
    if t_start is not None:
        df = df[df["time"] >= t_start]
    if t_stop is not None:
        df = df[df["time"] <= t_stop]
    return df


def plot_all_axes(est, gt, test_name, t_start=None, t_stop=None):

    # Plot estimated vs ground truth for all axes
    axes = ["x-pos", "y-pos", "z-pos"]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    est_plot = apply_time_window(est, t_start, t_stop)
    gt_plot = apply_time_window(gt, t_start, t_stop)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 2. Plotting Each Axis
    for i, axis in enumerate(axes):
        ax = axs[i]
        all_fingers = sorted(set(est_plot["finger_idx"].unique()))

        gt_is_constant_across_fingers = gt_plot.groupby("time")[axis].nunique().max() <= 1

        for idx, finger in enumerate(all_fingers):
            color = colors[idx % len(colors)]

            est_f = est_plot[est_plot["finger_idx"] == finger].sort_values("time")
            if not est_f.empty:
                ax.plot(est_f["time"], est_f[axis],
                        label=f"Est f{finger}", linestyle="-", color=color)

            gt_f = gt_plot[gt_plot["finger_idx"] == finger].sort_values("time")
            if not gt_f.empty:
                if gt_is_constant_across_fingers:

                    label_text = "GT (all fingers)" if idx == 0 else None

                    ax.plot(gt_f["time"], gt_f[axis],
                            label=label_text, linestyle="--", color="red", zorder=5)
                else:
                    ax.plot(gt_f["time"], gt_f[axis],
                            label=f"GT f{finger}", linestyle="--", color=color, zorder=5)


        ax.legend(loc='lower right', ncol=2, fontsize=8, framealpha=0.9)
        ax.set_ylabel(f"{axis} [cm]")
        ax.grid(True, linestyle=':')

    # Final Adjustments
    axs[-1].set_xlabel("Time [s]")
    fig.suptitle(test_name, fontsize=14)
    plt.tight_layout()
    plt.show()

#=========================main===============================

project_root = Path(__file__).resolve().parents[1]
recordings_dir = project_root / "recordings"

#Test 1-> x-axis
#est_path = recordings_dir / "finger_coords_2026-01-10_15-06-03.csv"
#gt_path = recordings_dir / "finger_coords_2026-01-10_15-06-03 - Ground_Truth.csv"

#Test 2 -> y-axis
#est_path = recordings_dir / "finger_coords_2026-01-10_15-05-13.csv"
#gt_path = recordings_dir / "finger_coords_2026-01-10_15-05-13 - Ground_Truth.csv"

#Test 3 -> z-axis
est_path = recordings_dir / "finger_coords_2026-01-10_15-21-32.csv"
gt_path = recordings_dir / "finger_coords_2026-01-10_15-21-32 - Ground_Truth.csv"

est, gt = load_data(est_path, gt_path)

# Define time window (timeframe of test)
# Test 1:
#t_start, t_stop = 14, 36.5
# Test 2:
#t_start, t_stop = 20, 34.5
# Test 3:
t_start, t_stop = 29, 38

plot_all_axes(est, gt, "Test 2", t_start, t_stop)