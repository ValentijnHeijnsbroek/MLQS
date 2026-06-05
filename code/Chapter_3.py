"""
Chapter 3: Handling Noise and Missing Values
Uses the course-provided OutlierDetection, ImputationMissingValues, and DataTransformation classes.

Steps:
    0. Chauvenet C analysis + combined figure (elbow + visual inspection)
    1. Outlier detection           — Chauvenet per session, C=3
    2. Missing value imputation    — linear interpolation
    3. Low-pass filter             — Butterworth, cutoff=8 Hz
    4. Save cleaned dataset
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

try:
    LIB  = Path(__file__).parent / 'lib'
    BASE = Path(__file__).parent.parent
except NameError:
    LIB  = Path().resolve() / 'lib'
    BASE = Path().resolve().parent

sys.path.insert(0, str(LIB))

from OutlierDetection import DistributionBasedOutlierDetection
from ImputationMissingValues import ImputationMissingValues


# ==============================================================================
#  PATHS & SETTINGS
# ==============================================================================

INTERMEDIATE = BASE / 'intermediate_datafiles'
INPUT_FILE   = INTERMEDIATE / 'dataset_50ms.csv'
OUTPUT_FILE  = INTERMEDIATE / 'dataset_50ms_chapter3.csv'
FIGURES      = BASE / 'figures'
FIGURES.mkdir(exist_ok=True)

SENSOR_COLS = [
    'acc_x', 'acc_y', 'acc_z',
    'gyr_x', 'gyr_y', 'gyr_z',
    'mag_x', 'mag_y', 'mag_z',
]
SENSOR_GROUPS = {
    'Accelerometer': ['acc_x', 'acc_y', 'acc_z'],
    'Gyroscope':     ['gyr_x', 'gyr_y', 'gyr_z'],
    'Magnetometer':  ['mag_x', 'mag_y', 'mag_z'],
}

CHAUVENET_C   = 3
SAMPLING_FREQ = 20
CUTOFF_FREQ   = 8
C_VALUES      = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


# ==============================================================================
#  STEP 0 — CHAUVENET C ANALYSIS + COMBINED FIGURE
#  Computes outlier % per C analytically (one pass over z-scores).
#  Generates a combined figure:
#    Left:  elbow plot to justify C=3
#    Right: time series with flagged outliers for visual verification
#           as recommended by Hoogendoorn & Funk (2018)
# ==============================================================================

print('=' * 60)
print('STEP 0  Chauvenet C analysis + combined figure')
print('=' * 60)

data     = pd.read_csv(INPUT_FILE)
sessions = data['session'].unique()
print(f'  Loaded {len(data)} rows from {len(sessions)} sessions')

# One pass: compute two-tailed p-values per session/column
session_probs = {}
for session in sessions:
    seg = data[data['session'] == session]
    for grp, cols in SENSOR_GROUPS.items():
        for col in cols:
            s = seg[col].dropna()
            N, mean, std = len(s), s.mean(), s.std()
            if std == 0 or N == 0:
                continue
            probs = 2 * norm.sf((s - mean).abs() / std)
            session_probs[(session, col)] = (N, probs)

# Derive outlier % for each C value
results = {grp: [] for grp in SENSOR_GROUPS}
for C in C_VALUES:
    grp_counts = {grp: {'out': 0, 'total': 0} for grp in SENSOR_GROUPS}
    for (session, col), (N, probs) in session_probs.items():
        n_out = (probs < 1.0 / (C * N)).sum()
        for grp, cols in SENSOR_GROUPS.items():
            if col in cols:
                grp_counts[grp]['out']   += n_out
                grp_counts[grp]['total'] += N
    for grp in SENSOR_GROUPS:
        results[grp].append(100 * grp_counts[grp]['out'] / grp_counts[grp]['total'])

# Heatmap data: outlier % per activity per channel
ACTIVITY_MAP = {'Rennen':'Running','Lopen':'Walking','Jumping jacks':'Jumping jacks',
                'Traplopen':'Stair climbing','Standing':'Standing'}
hm_results = {act: {col: 0 for col in SENSOR_COLS} for act in ACTIVITY_MAP}
hm_totals  = {act: 0 for act in ACTIVITY_MAP}
for session in sessions:
    seg = data[data['session'] == session]
    act = seg['activity'].iloc[0]
    if act not in ACTIVITY_MAP: continue
    hm_totals[act] += len(seg)
    for col in SENSOR_COLS:
        s = seg[col].dropna()
        N2, mean2, std2 = len(s), s.mean(), s.std()
        if std2 == 0 or N2 == 0: continue
        from scipy.stats import norm as norm2
        probs2 = 2 * norm2.sf((s - mean2).abs() / std2)
        hm_results[act][col] += (probs2 < 1.0/(CHAUVENET_C*N2)).sum()

matrix = pd.DataFrame(
    {act: {col: 100*hm_results[act][col]/hm_totals[act] for col in SENSOR_COLS}
     for act in ACTIVITY_MAP}).T
matrix.index = [ACTIVITY_MAP[a] for a in matrix.index]

# Combined figure
colors_grp = {'Accelerometer': '#4878CF', 'Gyroscope': '#6ACC65', 'Magnetometer': '#D65F5F'}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left: elbow plot
for grp, pcts in results.items():
    ax1.plot(C_VALUES, pcts, marker='o', label=grp, color=colors_grp[grp], lw=2)
ax1.axvline(CHAUVENET_C, color='gray', lw=1.2, linestyle='--', label=f'Chosen C={CHAUVENET_C}')
ax1.set_xlabel('Chauvenet parameter C')
ax1.set_ylabel('Outliers detected (%)')
ax1.set_title('Outlier rate vs. C')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(C_VALUES)

# Right: heatmap per activity per channel
im = ax2.imshow(matrix.values, cmap='YlOrRd', aspect='auto')
ax2.set_xticks(range(len(SENSOR_COLS)))
ax2.set_xticklabels(SENSOR_COLS, rotation=45, ha='right', fontsize=8)
ax2.set_yticks(range(len(matrix.index)))
ax2.set_yticklabels(matrix.index, fontsize=8)
ax2.set_title(f'Outlier rate (%) per channel and activity (C={CHAUVENET_C})')
for i in range(len(matrix.index)):
    for j in range(len(SENSOR_COLS)):
        ax2.text(j, i, f'{matrix.values[i,j]:.2f}', ha='center', va='center', fontsize=7)
plt.colorbar(im, ax=ax2, label='%')

fig.tight_layout()
fig.savefig(FIGURES / 'fig_outlier_combined.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved fig_outlier_combined')

# Low-pass filter motivation figure
# Single panel: raw 50Hz, aggregated 20Hz, and low-pass 8Hz overlaid.
# Orange and red dashed overlap perfectly, showing aggregation already smoothed the signal.
from DataTransformation import LowPassFilter as LPF
raw_csv = BASE / 'data/Morris/Rennen 1/Accelerometer.csv'
raw_50  = pd.read_csv(raw_csv).rename(columns={'X (m/s^2)': 'x', 'Time (s)': 't'})
mid_r   = len(raw_50) // 2
raw_50  = raw_50.iloc[mid_r-100:mid_r+100].reset_index(drop=True)
t_raw   = (raw_50['t'] - raw_50['t'].iloc[0]).values

agg     = data[data['session'] == 'Morris_Rennen 1'].reset_index(drop=True)
mid_a   = len(agg) // 2
agg     = agg.iloc[mid_a-40:mid_a+40].reset_index(drop=True)
lpf_mot = LPF()
agg     = lpf_mot.low_pass_filter(agg, 'acc_x', 20, 8, order=5)
t_agg   = np.arange(len(agg)) * 0.05

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(t_raw, raw_50['x'],           color='#6baed6', lw=1.5, alpha=0.9, label='Raw 50 Hz',           zorder=1)
ax.plot(t_agg, agg['acc_x'],          color='#e6550d', lw=2.5,            label='Aggregated 20 Hz',    zorder=2)
ax.plot(t_agg, agg['acc_x_lowpass'],  color='#31a354', lw=2.0, linestyle='--', label='+ Low-pass 8 Hz', zorder=3)
ax.set_xlabel('Time (s)')
ax.set_ylabel('acc_x (m/s$^2$)')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-55, 55)
fig.tight_layout()
fig.savefig(FIGURES / 'fig_lowpass_filter.pdf', bbox_inches='tight')
fig.savefig(FIGURES / 'fig_lowpass_filter.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved fig_lowpass_filter')


# ==============================================================================
#  STEP 1 — OUTLIER DETECTION
#  Chauvenet's criterion applied per session (not on the full dataset) so that
#  each segment has an approximately normal distribution, satisfying the
#  normality assumption that Chauvenet requires.
# ==============================================================================

print()
print('=' * 60)
print(f'STEP 1  Outlier detection — Chauvenet per session (C={CHAUVENET_C})')
print('=' * 60)

outlier_detector = DistributionBasedOutlierDetection()

for col in SENSOR_COLS:
    data[col + '_outlier'] = False

total_outliers = {col: 0 for col in SENSOR_COLS}
for session in sessions:
    idx = data['session'] == session
    seg = data.loc[idx].copy().reset_index(drop=True)
    for col in SENSOR_COLS:
        seg = outlier_detector.chauvenet(seg, col, CHAUVENET_C)
        data.loc[idx, col + '_outlier'] = seg[col + '_outlier'].values
        total_outliers[col] += seg[col + '_outlier'].sum()

for col in SENSOR_COLS:
    data.loc[data[col + '_outlier'], col] = None
    pct = 100 * total_outliers[col] / len(data)
    print(f'  {col}: {total_outliers[col]} outliers ({pct:.2f}%)')


# ==============================================================================
#  STEP 2 — MISSING VALUE IMPUTATION
#  Linear interpolation between neighbouring time points.
#  Preferred over mean/median because it respects the temporal structure of
#  the signal. Backward fill is used as fallback for missing values at the
#  start/end of a recording.
# ==============================================================================

print()
print('=' * 60)
print('STEP 2  Missing value imputation — linear interpolation')
print('=' * 60)

imputer = ImputationMissingValues()

for col in SENSOR_COLS:
    n_before = data[col].isna().sum()
    data = imputer.impute_interpolate(data, col)
    print(f'  {col}: {n_before} missing -> {data[col].isna().sum()} remaining')


# ==============================================================================
#  STEP 3 — LOW-PASS FILTER
#  Butterworth low-pass filter (order=5, cutoff=8 Hz).
#  Cutoff must be strictly below the Nyquist frequency (10 Hz at 20 Hz sampling).
#  Filtered signals are stored as new _lowpass columns alongside the originals.
# ==============================================================================

print()
print('=' * 60)
print(f'STEP 3  Low-pass filter (cutoff={CUTOFF_FREQ} Hz, sampling={SAMPLING_FREQ} Hz)')
print('=' * 60)

lpf = LowPassFilter()
for col in SENSOR_COLS:
    data = lpf.low_pass_filter(data, col, SAMPLING_FREQ, CUTOFF_FREQ, order=5)
    print(f'  {col} -> {col}_lowpass added')


# ==============================================================================
#  STEP 4 — SAVE CLEANED DATASET
# ==============================================================================

print()
print('=' * 60)
print('STEP 3  Saving cleaned dataset')
print('=' * 60)

outlier_flag_cols = [col + '_outlier' for col in SENSOR_COLS]
data.drop(columns=outlier_flag_cols, inplace=True)
data.to_csv(OUTPUT_FILE, index=False)
print(f'  Saved {len(data)} rows to {OUTPUT_FILE}')
print(f'\nAll figures saved to {FIGURES}')
