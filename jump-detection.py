import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import our helper function from the helpers file
from helpersgood import extract_global_acceleration


class VolleyballJumpDetectorGlobal:
    """
    A robust algorithm for detecting volleyball jumps using a SINGLE IMU sensor.
    Uses True Global Z Acceleration (Absolute Value) for precise flight time measurement and impact detection.
    """

    def __init__(self,
                 sensor_id=15,
                 freefall_thresh=0.3,  # NEW: The 'epsilon' representing true freefall (0G)
                 impact_thresh=6.0,  # Minimum 6 G for landing impact
                 min_flight_time=0.2,  # Minimum 200ms flight time
                 gap_tolerance=0.1,
                 smoothing_window=5,
                 cooldown=1.0):  # Adjustable cooldown period in seconds

        self.sensor_id = sensor_id
        self.freefall_thresh = freefall_thresh
        self.impact_thresh = impact_thresh
        self.min_flight_time = min_flight_time
        self.gap_tolerance = gap_tolerance
        self.smoothing_window = smoothing_window
        self.cooldown = cooldown

        self.results = []
        self.all_candidate_phases = []
        self.data_loaded = False
        self.fs = 148.15
        self.times = None
        self.global_z = None
        self.filename = "Unknown"

    def load_and_process(self, file_path):
        """Loads data using the helper function and applies absolute value + smoothing to Global Z."""

        self.filename = os.path.basename(file_path)
        print(f"--- Loading and Transforming: {self.filename} (Sensor {self.sensor_id}) ---")

        # Use the external helper function to get the structured DataFrame
        df = extract_global_acceleration(file_path, sensor_id=self.sensor_id)

        if df is None:
            print(f"Error: Could not process file for sensor {self.sensor_id}.")
            return False

        self.times = df['Time'].values

        if len(self.times) > 1:
            self.fs = 1.0 / np.mean(np.diff(self.times))

        # Take Absolute Value before smoothing to ensure resting gravity is ~1.0g
        raw_global_z = np.abs(df['Global_Z'].values)

        self.global_z = pd.Series(raw_global_z).rolling(
            self.smoothing_window, center=True).mean().bfill().ffill().values

        self.data_loaded = True
        return True

    def detect(self):
        """Main Algorithm Pipeline using Global Z"""
        if not self.data_loaded:
            return []

        # A. Detection (Strictly based on the new freefall_thresh/epsilon)
        is_flight = self.global_z < self.freefall_thresh

        # B. Segmentation
        segments = []
        i = 0
        while i < len(is_flight):
            if is_flight[i]:
                start = i
                while i < len(is_flight) and is_flight[i]:
                    i += 1
                end = i
                segments.append({'start': start, 'end': end})
            else:
                i += 1

        # C. Gap Filling (Do not merge if there is a strong impact in the gap)
        max_gap_samples = int(self.gap_tolerance * self.fs)
        merged_segments = []
        if segments:
            curr = segments[0]
            for next_seg in segments[1:]:
                gap_start = curr['end']
                gap_end = next_seg['start']
                gap_length = gap_end - gap_start

                if gap_length <= max_gap_samples:
                    # Look inside the gap for a strong impact
                    max_in_gap = np.max(self.global_z[gap_start:gap_end]) if gap_end > gap_start else 0

                    if max_in_gap <= self.impact_thresh:
                        # Safe to merge
                        curr['end'] = next_seg['end']
                    else:
                        # DO NOT MERGE! A valid landing impact occurred
                        merged_segments.append(curr)
                        curr = next_seg
                else:
                    merged_segments.append(curr)
                    curr = next_seg
            merged_segments.append(curr)

        # D. Processing
        self.results = []
        self.all_candidate_phases = []
        search_window = int(0.4 * self.fs)

        for seg in merged_segments:
            # We use the strict boundaries found by the threshold
            start_idx = seg['start']
            end_idx = seg['end']
            duration_sec = (end_idx - start_idx) / self.fs

            # Evaluate Criteria 1: Flight Time
            passed_time = duration_sec >= self.min_flight_time

            passed_impact = False
            impact_val = 0
            land_time = self.times[end_idx] if end_idx < len(self.times) else 0

            # Evaluate Criteria 2: Impact Peak
            if end_idx < len(self.global_z):
                window = self.global_z[end_idx: min(end_idx + search_window, len(self.global_z))]
                if len(window) > 0:
                    impact_val = np.max(window)
                    if impact_val > self.impact_thresh:
                        passed_impact = True

                    land_time_idx = end_idx + np.argmax(window)
                    land_time = self.times[land_time_idx]

            # Save detailed candidate info for plotting
            self.all_candidate_phases.append({
                'indices': (start_idx, end_idx),
                'duration_sec': duration_sec,
                'passed_time': passed_time,
                'passed_impact': passed_impact,
                'impact_g': impact_val,
                'impact_time': land_time
            })

            # If it passed BOTH criteria, it's a valid jump
            if passed_time and passed_impact:
                height_cm = 0.125 * 9.81 * (duration_sec ** 2) * 100

                # Check cooldown dynamically using self.cooldown
                if not self.results or (land_time - self.results[-1]['time'] > self.cooldown):
                    self.results.append({
                        'time': round(land_time, 2),
                        'height_cm': round(height_cm, 1),
                        'flight_s': round(duration_sec, 3),
                        'impact_g': round(impact_val, 1),
                        'indices': (start_idx, end_idx)
                    })

        return self.results

    def plot(self, ax=None):
        """
        Visualizes results with detailed tags for all candidates.
        """
        if not self.data_loaded: return

        show_plot = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            show_plot = True

        # Plot the absolute Global Z axis
        line_color = 'green' if self.sensor_id == 16 else 'blue'
        ax.plot(self.times, self.global_z, label=f'|Global Z| (Sensor {self.sensor_id})', color=line_color, alpha=0.8)

        # Updated line to plot the new freefall threshold
        ax.axhline(self.freefall_thresh, color='orange', linestyle=':', label='Freefall Thresh')
        ax.axhline(self.impact_thresh, color='purple', linestyle='--', label='Impact Thresh')
        ax.grid(True, alpha=0.3)

        # Highlight ALL flight phase candidates in ORANGE
        for i, cand in enumerate(self.all_candidate_phases):
            s, e = cand['indices']

            # --- FIX: Ensure indices stay within the bounds of the array ---
            s_safe = min(s, len(self.times) - 1)
            e_safe = min(e, len(self.times) - 1)

            label = 'Flight Phase' if i == 0 else None
            ax.axvspan(self.times[s_safe], self.times[e_safe], color='orange', alpha=0.3, label=label)

            # Build the text label
            tags = []
            if cand['passed_time']:
                tags.append(f"{cand['duration_sec'] * 1000:.0f}ms")
            if cand['passed_impact']:
                tags.append("Peak")

            # Check if this candidate made it to the final results
            is_confirmed = any(res['indices'] == cand['indices'] for res in self.results)

            if is_confirmed:
                res = next(r for r in self.results if r['indices'] == cand['indices'])
                tags.append(f"{res['height_cm']}cm")
                # Draw the red dot exactly on the impact peak
                ax.scatter(cand['impact_time'], cand['impact_g'], color='red', zorder=5)

            # Draw the label if there are any tags
            if tags:
                label_text = "\n".join(tags)
                end_of_flight_time = self.times[e_safe]

                # Plot the text right after the orange background
                ax.text(end_of_flight_time, 2.0, label_text,
                        ha='left', color='darkred', fontweight='bold', fontsize=9,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))

        ax.set_title(f"Sensor {self.sensor_id} | Confirmed Jumps: {len(self.results)}", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_ylabel("Absolute Accel (g)")

        if show_plot:
            ax.set_xlabel("Time [s]")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # Use a relative path to the 'data' folder
    base_dir = 'data'
    target_file = 'beach-volleyball.txt'
    path = os.path.join(base_dir, target_file)

    if os.path.exists(path):

        # --- 1. Initialize and Process Sensor 15 ---
        det15 = VolleyballJumpDetectorGlobal(sensor_id=15, freefall_thresh=0.3, impact_thresh=7.0, min_flight_time=0.36)
        success15 = det15.load_and_process(path)
        if success15:
            jumps15 = det15.detect()

        # --- 2. Initialize and Process Sensor 16 ---
        det16 = VolleyballJumpDetectorGlobal(sensor_id=16, freefall_thresh=0.3)
        success16 = det16.load_and_process(path)
        if success16:
            jumps16 = det16.detect()

        # --- 3. Plot Both Sensors in One Figure ---
        if success15 and success16:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            det15.plot(ax=ax1)
            det16.plot(ax=ax2)

            ax2.set_xlabel("Time [s]", fontsize=12)
            fig.suptitle(f"Dual Sensor Jump Detection Comparison\nFile: {target_file}", fontsize=16, y=0.98)

            plt.tight_layout()
            plt.show()

    else:
        print(f"CRITICAL ERROR: Could not find '{path}'.")