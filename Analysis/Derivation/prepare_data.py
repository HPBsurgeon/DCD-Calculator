import pandas as pd
import numpy as np

def load_derivation_data(path):
    df = pd.read_csv(path)
    # Define outcome variable
    df['CSTATUS_30'] = np.select([
        df['Survival_time'] <= 30,
        df['Survival_time'] > 30,
    ], [1, 0], default=np.nan)

    # Define feature set
    features = [
        'GCS','pupil','gag','corneal','cough','motor','OBV',
        'MAP','Na','Plt','initial_PF_ratio','end_PF_ratio',
        'end_ph','arrest_his','Mechanism_of_injury','BMI'
    ]
    x = df[features].copy()
    t = np.array(df['CSTATUS_30'].tolist())
    return x, t
