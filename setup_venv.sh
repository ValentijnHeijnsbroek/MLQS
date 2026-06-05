#!/bin/bash
cd "$(dirname "$0")"
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn scipy pandas matplotlib
echo ""
echo "✓ Venv klaar. Activeer met: source venv/bin/activate"
echo "  Dan run je: python code/chapter3.py"
