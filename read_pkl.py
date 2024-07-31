import pandas as pd
from pathlib import Path

prediction = pd.read_pickle("./outputs_srdgru_metr-la_6/checkpoints/test_pre.pkl")

print(prediction)