"""
Chapter 2: From Raw Signals to a Dataset
Builds the dataset at three granularities and generates EDA figures.

Steps:
    0. EDA figures  — summary statistics, boxplot, per-activity time series
    1. Build dataset — aggregate raw phyphox CSVs into dataset_Xms.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

try:
    BASE = Path(__file__).parent.parent
    LIB  = Path(__file__).parent / 'lib'
except NameError:
    BASE = Path().resolve().parent
    LIB  = Path().resolve() / 'lib'

sys.path.insert(0, str(LIB))
from CreateDataset import CreateDataset


# ==============================================================================
#  PATHS & SETTINGS
# ==============================================================================

DATA_PATH    = BASE / 'data'
RESULT_PATH  = BASE / 'intermediate_datafiles'
FIGURES      = BASE / 'figures'
RESULT_PATH.mkdir(exist_ok=True, parents=True)
FIGURES.mkdir(exist_ok=True, parents=True)

PEOPLE        = ['Valentijn', 'Lars', 'Morris']
GRANULARITIES = [50, 250, 1000]
ACC           = ['acc_x', 'acc_y', 'acc_z']
CHANNELS      = ACC + ['gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z']
ACTIVITIES    = [('Standing', 'Standing'), ('Lopen', 'Walking'),
                 ('Traplopen', 'Stair climbing'), ('Jumping jacks', 'Jumping jacks'),
                 ('Rennen', 'Running')]
SENSORS = [
    ('Accelerometer.csv', {'X (m/s^2)': 'x', 'Y (m/s^2)': 'y', 'Z (m/s^2)': 'z'}, 'acc_'),
    ('Gyroscope.csv',     {'X (rad/s)':  'x', 'Y (rad/s)':  'y', 'Z (rad/s)':  'z'}, 'gyr_'),
    ('Magnetometer.csv',  {'X (µT)':     'x', 'Y (µT)':     'y', 'Z (µT)':     'z'}, 'mag_'),
]


# ==============================================================================
#  STEP 0 — EDA FIGURES
#  Requires the datasets to already exist (run Step 1 first if they don't).
#  Generates:
#    - Summary statistics table (printed to console)
#    - Boxplot of accelerometer spread per granularity (Fig 1)
#    - Per-activity 8-second time series of accelerometer (Fig 2)
# ==============================================================================

print('=' * 60)
print('STEP 0  EDA figures')
print('=' * 60)

datasets = {g: pd.read_csv(RESULT_PATH / f'dataset_{g}ms.csv') for g in GRANULARITIES}
fine = datasets[50]

# Summary statistics (printed, used for Table 3 in report)
print('\nSummary statistics (50 ms dataset):')
for c in CHANNELS:
    s = fine[c]
    print(f'  {c:6s}  miss%={s.isna().mean()*100:5.2f}  mean={s.mean():8.3f}  '
          f'sd={s.std():7.3f}  min={s.min():8.2f}  max={s.max():8.2f}')

# Accelerometer std per activity per granularity (Table 1 in report)
print('\nAccelerometer std per activity (mean over x, y, z):')
for act, name in ACTIVITIES:
    vals = [datasets[g][datasets[g]['activity'] == act][ACC].std().mean()
            for g in GRANULARITIES]
    print(f'  {name:14s}  ' + '  '.join(f'{v:5.2f}' for v in vals))

# Boxplot of accelerometer spread per granularity (Fig 1)
fig, ax = plt.subplots(figsize=(6, 4))
data_box = [datasets[g][ACC].stack().dropna().values for g in GRANULARITIES]
ax.boxplot(data_box, flierprops={'marker': '.', 'markersize': 2, 'alpha': 0.3})
ax.set_xticks(range(1, len(GRANULARITIES) + 1))
ax.set_xticklabels([f'{g} ms' for g in GRANULARITIES])
ax.set_ylabel('accelerometer (m/s$^2$)')
ax.set_xlabel('granularity $\\Delta t$')
fig.tight_layout()
fig.savefig(FIGURES / 'boxplot_granularity.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n  Saved boxplot_granularity')

# Per-activity 8-second time series (Fig 2) — 2x3 grid layout
fig, axes = plt.subplots(2, 3, figsize=(11, 4.5), sharey=True)
axes_flat = axes.flatten()
WIN = 160  # 8 s at 50 ms
for i, (act, name) in enumerate(ACTIVITIES):
    ax   = axes_flat[i]
    sess = sorted(fine[fine['activity'] == act]['session'].unique())[0]
    seg  = fine[fine['session'] == sess].reset_index(drop=True)
    mid  = len(seg) // 2
    seg  = seg.iloc[mid - WIN // 2: mid + WIN // 2].reset_index(drop=True)
    t    = np.arange(len(seg)) * 0.05
    for c, lab in zip(ACC, ['x', 'y', 'z']):
        ax.plot(t, seg[c], linewidth=0.8, label=lab)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel('time (s)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(-55, 55)
    ax.grid(True, alpha=0.3)
    if i % 3 == 0:
        ax.set_ylabel('m/s$^2$', fontsize=8)
axes_flat[0].legend(ncol=3, fontsize=7, loc='upper right')
axes_flat[5].set_visible(False)
fig.tight_layout(pad=0.8)
fig.savefig(FIGURES / 'timeseries_activities.png', dpi=150, bbox_inches='tight')
fig.savefig(FIGURES / 'timeseries_activities.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved timeseries_activities')

print(f'\nAll figures saved to {FIGURES}')


# ==============================================================================
#  STEP 1 — BUILD DATASET
#  Aggregates raw phyphox CSVs into dataset_50ms.csv, dataset_250ms.csv,
#  and dataset_1000ms.csv using the CreateDataset class (Chapter 2 of book).
# ==============================================================================

print()
print('=' * 60)
print('STEP 1  Building datasets')
print('=' * 60)

for granularity in GRANULARITIES:
    recordings = []
    for person in PEOPLE:
        for folder in sorted((DATA_PATH / person).iterdir()):
            if not folder.is_dir():
                continue
            print(f'  {granularity}ms | {person} - {folder.name}')
            dataset = CreateDataset(folder, granularity)
            for file, rename, prefix in SENSORS:
                dataset.add_numerical_dataset(file, 'Time (s)', ['x', 'y', 'z'], 'avg', prefix,
                                              time_unit='s', rename=rename)
            table = dataset.data_table.reset_index().rename(columns={'index': 'time'})
            table['activity'] = folder.name.rsplit(' ', 1)[0]
            table['person']   = person
            table['session']  = f'{person}_{folder.name}'
            recordings.append(table)

    data = pd.concat(recordings, ignore_index=True)
    labels = pd.get_dummies(data['activity'].str.replace(r'[^0-9a-zA-Z]+', '', regex=True),
                            prefix='label', prefix_sep='').astype(int)
    data = pd.concat([data, labels], axis=1)
    data.to_csv(RESULT_PATH / f'dataset_{granularity}ms.csv', index=False)
    print(f'  {len(data)} rows from {len(recordings)} recordings -> dataset_{granularity}ms.csv')
