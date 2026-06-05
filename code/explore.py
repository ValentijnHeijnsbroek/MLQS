from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_PATH = Path(__file__).parent.parent / 'intermediate_datafiles'
FIG_PATH = Path(__file__).parent.parent / 'figures'
FIG_PATH.mkdir(exist_ok=True, parents=True)

GRANULARITIES = [50, 250, 1000]
ACC = ['acc_x', 'acc_y', 'acc_z']
CHANNELS = ACC + ['gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z']

# least -> most dynamic, with the Dutch label mapped to an English display name
ACTIVITIES = [('Standing', 'Standing'), ('Lopen', 'Walking'),
              ('Traplopen', 'Stair climbing'), ('Jumping jacks', 'Jumping jacks'),
              ('Rennen', 'Running')]

datasets = {g: pd.read_csv(RESULT_PATH / f'dataset_{g}ms.csv') for g in GRANULARITIES}
fine = datasets[50]

# --- summary statistics over the full 50 ms dataset (Table 3 in the report) ---
print('Summary statistics (50 ms dataset)')
for c in CHANNELS:
    s = fine[c]
    print(f'{c:6s} miss%={s.isna().mean()*100:5.2f} mean={s.mean():8.3f} '
          f'sd={s.std():7.3f} min={s.min():8.2f} max={s.max():8.2f}')

# --- accelerometer std per activity at each granularity (Table 1) ---
print('\nAccelerometer std per activity (mean over x,y,z)')
for act, name in ACTIVITIES:
    vals = [datasets[g][datasets[g]['activity'] == act][ACC].std().mean() for g in GRANULARITIES]
    print(f'{name:14s} ' + ' '.join(f'{v:5.2f}' for v in vals))

# --- boxplot of accelerometer spread per granularity (Fig 1) ---
fig, ax = plt.subplots(figsize=(6, 4))
data = [datasets[g][ACC].stack().dropna().values for g in GRANULARITIES]
ax.boxplot(data, flierprops={'marker': '.', 'markersize': 2, 'alpha': 0.3})
ax.set_xticks(range(1, len(GRANULARITIES) + 1))
ax.set_xticklabels([f'{g} ms' for g in GRANULARITIES])
ax.set_ylabel('accelerometer (m/s$^2$)')
ax.set_xlabel('granularity $\\Delta t$')
fig.tight_layout()
fig.savefig(FIG_PATH / 'boxplot_granularity.png', dpi=150, bbox_inches='tight')

# --- per-activity time series over an 8 s window (Fig 2) ---
fig, axes = plt.subplots(len(ACTIVITIES), 1, figsize=(7, 8.5), sharex=True)
WIN = 160  # 8 s at 50 ms
for ax, (act, name) in zip(axes, ACTIVITIES):
    sess = sorted(fine[fine['activity'] == act]['session'].unique())[0]
    seg = fine[fine['session'] == sess].reset_index(drop=True)
    mid = len(seg) // 2
    seg = seg.iloc[mid - WIN // 2: mid + WIN // 2].reset_index(drop=True)
    t = np.arange(len(seg)) * 0.05
    for c, lab in zip(ACC, ['x', 'y', 'z']):
        ax.plot(t, seg[c], linewidth=0.8, label=lab)
    ax.set_ylabel('m/s$^2$', fontsize=8)
    ax.set_title(name, fontsize=9, loc='left')
    ax.tick_params(labelsize=7)
    ax.set_ylim(-50, 50)
axes[0].legend(ncol=3, fontsize=7, loc='upper right')
axes[-1].set_xlabel('time within recording (s)', fontsize=8)
fig.tight_layout()
fig.savefig(FIG_PATH / 'timeseries_activities.png', dpi=150, bbox_inches='tight')

print(f'\nFigures written to {FIG_PATH}')
