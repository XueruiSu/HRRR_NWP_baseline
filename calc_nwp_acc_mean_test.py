import numpy as np
import pandas as pd
import pickle
import os
import sys

# open:
# normalize_mean2.pkl


normalize_mean2 = pickle.load(open("./normalize_mean2.pkl", "rb"))

print(normalize_mean2.keys())

print(normalize_mean2['2m_temperature'])

print(normalize_mean2['surface_pressure'])
print(len(normalize_mean2.keys()))