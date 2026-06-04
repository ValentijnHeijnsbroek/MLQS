# ML4QS Project — Human Activity Recognition

Machine Learning for the Quantified Self (VU Amsterdam) practical.
Classifying physical activities from smartphone sensor data (phyphox).

## Activities (Dutch labels)
Rennen (running), Lopen (walking), Traplopen (stairs), Standing, Jumping jacks.

## Data (`data/`)
One folder per recording, named `<Activity> <session>`, for three people
(Valentijn, Lars, Morris). Each folder holds phyphox raw CSVs with a relative
`Time (s)` column: `Accelerometer.csv`, `Gyroscope.csv`, `Magnetometer.csv`
(and `Barometer.csv` where recorded). **The barometer is not used.**

## Code (`code/`)
Our working pipeline, adapted from the course code by Hoogendoorn & Funk.
- `CreateDataset.py` — builds the aggregated dataset at a chosen granularity (Δt).
