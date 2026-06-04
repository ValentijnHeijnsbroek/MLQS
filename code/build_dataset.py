from pathlib import Path
import pandas as pd
from CreateDataset import CreateDataset

DATA_PATH = Path(__file__).parent.parent / 'data'
RESULT_PATH = Path(__file__).parent.parent / 'intermediate_datafiles'
PEOPLE = ['Valentijn', 'Lars', 'Morris']
GRANULARITIES = [50, 250, 1000]

# phyphox file, column rename to x/y/z, and output prefix per sensor (barometer left out)
SENSORS = [
    ('Accelerometer.csv', {'X (m/s^2)': 'x', 'Y (m/s^2)': 'y', 'Z (m/s^2)': 'z'}, 'acc_'),
    ('Gyroscope.csv', {'X (rad/s)': 'x', 'Y (rad/s)': 'y', 'Z (rad/s)': 'z'}, 'gyr_'),
    ('Magnetometer.csv', {'X (µT)': 'x', 'Y (µT)': 'y', 'Z (µT)': 'z'}, 'mag_'),
]

RESULT_PATH.mkdir(exist_ok=True, parents=True)

for granularity in GRANULARITIES:
    recordings = []
    for person in PEOPLE:
        for folder in sorted((DATA_PATH / person).iterdir()):
            if not folder.is_dir():
                continue
            print(f'{granularity}ms | {person} - {folder.name}')
            dataset = CreateDataset(folder, granularity)
            for file, rename, prefix in SENSORS:
                dataset.add_numerical_dataset(file, 'Time (s)', ['x', 'y', 'z'], 'avg', prefix,
                                              time_unit='s', rename=rename)

            table = dataset.data_table.reset_index().rename(columns={'index': 'time'})
            table['activity'] = folder.name.rsplit(' ', 1)[0]
            table['person'] = person
            table['session'] = f'{person}_{folder.name}'
            recordings.append(table)

    data = pd.concat(recordings, ignore_index=True)

    # one binary column per activity (book format)
    labels = pd.get_dummies(data['activity'].str.replace(r'[^0-9a-zA-Z]+', '', regex=True),
                            prefix='label', prefix_sep='').astype(int)
    data = pd.concat([data, labels], axis=1)

    data.to_csv(RESULT_PATH / f'dataset_{granularity}ms.csv', index=False)
    print(f'{len(data)} rows from {len(recordings)} recordings -> dataset_{granularity}ms.csv')
