# utils/prepare_data.py

import pandas as pd
import numpy as np

def load_and_prepare_data(path):
    # Load dataset
    unet = pd.read_csv(path)

    # Select relevant features
    ML_df = unet[[
        'Validation', 'UNET_ID',
        'CSTATUS_60','CSTATUS_45','CSTATUS_30',
        'GCS','pupil','gag','corneal','cough','motor','OBV',
        'end_MAP_category','end_Na_category','end_Plt_category',
        'initial_PF_ratio_category','end_PF_ratio_category',
        'end_ph_category','arrest_his','Mechanism_of_injury3','BMI_category'
    ]]

    # --- Training Data (Validation == 'd') ---
    df1 = ML_df[ML_df['Validation'] == 'd'].dropna(
        subset=ML_df.columns.difference(['GCS','pupil','gag','corneal','cough','motor','OBV'])
    )
    df1 = df1.dropna(subset=['pupil','gag','corneal','cough','motor','OBV'], thresh=3)

    x = df1.drop(['CSTATUS_60','CSTATUS_45','CSTATUS_30','UNET_ID','Validation'], axis=1)
    t = np.array(df1['CSTATUS_30'])

    # --- Validation Data (Validation == 'v') ---
    df2 = ML_df[ML_df['Validation'] == 'v'].dropna(
        subset=ML_df.columns.difference(['GCS','pupil','gag','corneal','cough','motor','OBV'])
    )
    df2 = df2.dropna(subset=['pupil','gag','corneal','cough','motor','OBV'], thresh=3)

    x_vali = df2.drop(['CSTATUS_60','CSTATUS_45','CSTATUS_30','UNET_ID','Validation'], axis=1)
    t_vali = np.array(df2['CSTATUS_30'])

    return x, t, x_vali, t_vali
