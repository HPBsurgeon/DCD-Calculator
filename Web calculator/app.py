import gradio as gr
import pandas as pd
import joblib
import sqlite3
import os
import numpy as np

# モデルのロード
model_total = joblib.load('model_total.joblib')
model_total2 = joblib.load('model_total_45.joblib')
model_total3 = joblib.load('model_total_60.joblib')
model_total4 = joblib.load('model_80_80_30.joblib')
model_total5 = joblib.load('model_60_60_30.joblib')
model_total6 = joblib.load('model_50_30.joblib')
model_total7 = joblib.load('model_80_80_45.joblib')
model_total8 = joblib.load('model_60_60_45.joblib')
model_total9 = joblib.load('model_50_45.joblib')

# SQLiteデータベースにスレッドセーフな接続を作成
def get_db_connection():
    conn = sqlite3.connect('predictions.db', check_same_thread=False)
    # テーブルが存在しない場合に作成
    conn.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        name TEXT,
        donor_id TEXT,
        feature1 REAL,
        feature2 TEXT,
        feature3 TEXT,
        feature4 TEXT,
        feature5 TEXT,
        feature6 TEXT,
        feature6_1 TEXT,
        feature7_1 REAL,
        feature7_2 REAL,
        feature8 TEXT,
        feature9 TEXT,
        feature10 TEXT,
        feature11 TEXT,
        feature12 TEXT,
        feature14 TEXT,
        feature15 TEXT,
        feature16 TEXT,
        prediction REAL
    )
    ''')
    conn.commit()
    return conn

def prediction(name, donor_id, feature1, feature2, feature3, feature4, feature5, feature6, feature6_1, feature7_1, feature7_2, feature8, feature9, feature10, feature11, feature12, feature14, feature15, feature16):
    data = {
        'name': name,
        'donor_id': donor_id,
        'feature1': feature1 if feature1 != '' else np.nan,  # GCSが未入力の場合はNaNを設定
        'feature2': feature2, 'feature3': feature3,
        'feature4': feature4, 'feature5': feature5, 'feature6': feature6, 'feature6_1': feature6_1,
        'feature7_1': feature7_1, 'feature7_2': feature7_2, 'feature8': feature8, 'feature9': feature9,
        'feature10': feature10, 'feature11': feature11, 'feature12': feature12,
        'feature14': feature14, 'feature15': feature15, 'feature16': feature16
    }

    # マッピング
    pupil_mapping = {"negative": 1, "positive": 0}
    gag_mapping = {"negative": 1, "positive": 0}
    corneal_mapping = {"negative": 1, "positive": 0}
    cough_mapping = {"negative": 1, "positive": 0}
    motor_mapping = {"negative": 1, "positive": 0}
    OBV_mapping = {"negative": 1, "positive": 0}
    initial_PF_ratio_mapping = {"400~": 0, "300~400": 1, "200~300": 2, "100~200": 3, "~100": 4}
    end_PF_ratio_mapping = {"400~": 0, "300~400": 1, "200~300": 2, "100~200": 3, "~100": 4}
    end_Na_mapping = {"~135": 0, "135~146": 1, "146~156": 2, "156~": 3}
    end_Plt_mapping = {"150~": 0, "100~150": 1, "~100": 2}
    end_ph_mapping = {"7.35~7.45": 0, "~7.35": 1, "7.45~": 2}
    arrest_mapping = {"Yes": 1, "No": 0}
    mechanism_mapping = {"BLUNT": 0, "CARDIOVASCULAR": 1, "DRUG": 2, "INTRACRANIAL": 3, "OTHERS": 4}
    BMI_mapping = {"~30": 0, "30~": 1}

    data['feature2'] = pupil_mapping[data['feature2']]
    data['feature3'] = gag_mapping[data['feature3']]
    data['feature4'] = corneal_mapping[data['feature4']]
    data['feature5'] = cough_mapping[data['feature5']]
    data['feature6'] = motor_mapping[data['feature6']]
    data['feature6_1'] = OBV_mapping[data['feature6_1']]
    data['feature8'] = initial_PF_ratio_mapping[data['feature8']]
    data['feature9'] = end_PF_ratio_mapping[data['feature9']]
    data['feature10'] = end_Na_mapping[data['feature10']]
    data['feature11'] = end_Plt_mapping[data['feature11']]
    data['feature12'] = end_ph_mapping[data['feature12']]
    data['feature14'] = arrest_mapping[data['feature14']]
    data['feature15'] = mechanism_mapping[data['feature15']]
    data['feature16'] = BMI_mapping[data['feature16']]

    input_data = pd.DataFrame([data])
    input_data['feature1'] = input_data['feature1'].astype(float)
    input_data['feature2'] = pd.Categorical(input_data['feature2'], categories=[1, 0], ordered=True)
    input_data['feature3'] = pd.Categorical(input_data['feature3'], categories=[1, 0], ordered=True)
    input_data['feature4'] = pd.Categorical(input_data['feature4'], categories=[1, 0], ordered=True)
    input_data['feature5'] = pd.Categorical(input_data['feature5'], categories=[1, 0], ordered=True)
    input_data['feature6'] = pd.Categorical(input_data['feature6'], categories=[1, 0], ordered=True)
    input_data['feature6_1'] = pd.Categorical(input_data['feature6_1'], categories=[1, 0], ordered=True)
    input_data['feature8'] = input_data['feature8'].astype(int)
    input_data['feature9'] = input_data['feature9'].astype(int)
    input_data['feature10'] = input_data['feature10'].astype(int)
    input_data['feature11'] = input_data['feature11'].astype(int)
    input_data['feature12'] = input_data['feature12'].astype(int)
    input_data['feature14'] = pd.Categorical(input_data['feature14'], categories=[1, 0], ordered=True)
    input_data['feature15'] = pd.Categorical(input_data['feature15'], categories=[1, 2, 3, 4, 0], ordered=True)
    input_data['feature16'] = input_data['feature16'].astype(int)

    input_data['feature7'] = (input_data['feature7_1'] - input_data['feature7_2']) / 3 + input_data['feature7_2']
    end_MAP = []
    for i in range(len(input_data)):
        a = input_data.iat[i, input_data.columns.get_loc('feature7')]
        if a < 75:
            x = int(1)
        elif 75 <= a:
            x = int(0)
        end_MAP.append(x)

    input_data['feature7'] = end_MAP
    input_data['feature7'] = input_data['feature7'].astype(int)

    # 予測のためのデータのみを抽出
    prediction_data = input_data[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature6_1', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature14', 'feature15', 'feature16']]

    prediction = model_total.predict(prediction_data)
    prediction2 = model_total2.predict(prediction_data)
    prediction3 = model_total3.predict(prediction_data)
    prediction4 = model_total4.predict(prediction_data)
    prediction5 = model_total5.predict(prediction_data)
    prediction6 = model_total6.predict(prediction_data)
    prediction7 = model_total7.predict(prediction_data)
    prediction8 = model_total8.predict(prediction_data)
    prediction9 = model_total9.predict(prediction_data)

    if prediction2<prediction:
        prediction2 = prediction

    if prediction3<prediction2:
        prediction3 = prediction2

    if prediction5<prediction4:
        prediction5 = prediction4

    if prediction6<prediction5:
        prediction6 = prediction5

    if prediction8<prediction7:
        prediction8 = prediction7

    if prediction9<prediction8:
        prediction9 = prediction8
    
    if prediction7<prediction4:
        prediction7 = prediction4

    if prediction8<prediction5:
        prediction8 = prediction5

    if prediction9<prediction6:
        prediction9 = prediction6

    # SQLiteデータベースに接続
    conn = get_db_connection()
    cur = conn.cursor()

    # データをデータベースに保存
    cur.execute('''
        INSERT INTO predictions (name, donor_id, feature1, feature2, feature3, feature4, feature5, feature6, feature6_1, feature7_1, feature7_2, feature10, feature11, feature8, feature9, feature12, feature14, feature15, feature16, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        name, donor_id, feature1, feature2, feature3, feature4, feature5, feature6, feature6_1,
        feature7_1, feature7_2, feature8, feature9, feature10, feature11, feature12,
        feature14, feature15, feature16, prediction[0]
    ))

    conn.commit()
    conn.close()

    return f"From extubation\nExpire rate within 30min: {round(prediction[0], 2) * 100}%, Expire rate within 45min: {round(prediction2[0], 2) * 100}%, Expire rate within 60min: {round(prediction3[0], 2) * 100}% (cutoff=50%)\n\nFrom 80%/80mmHg\nExpire rate within 30min: {round(prediction4[0], 2) * 100}%, Expire rate within 45min: {round(prediction7[0], 2) * 100}% (cutoff=50%)\n\nFrom 60%/60mmHg\nExpire rate within 30min: {round(prediction5[0], 2) * 100}%, Expire rate within 45min: {round(prediction8[0], 2) * 100}% (cutoff=50%)\n\nFrom 50mmHg\nExpire rate within 30min: {round(prediction6[0], 2) * 100}%, Expire rate within 45min: {round(prediction9[0], 2) * 100}% (cutoff=50%)"

# SQLiteデータベースからExcelファイルを作成
def export_to_excel():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    output_path = "predictions_output.xlsx"
    
    if os.path.exists(output_path):
        existing_df = pd.read_excel(output_path)
        df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates()

    df.to_excel(output_path, index=False)
    conn.close()
    return output_path

# Gradioインターフェースの設定
iface = gr.Interface(
    fn=prediction,
    inputs=[
        gr.Dropdown(choices=["TransMedics Staff", "Stanford", "Rochester", "Cleveland", "VCU", "Florida", "Columbia", 
                             # "Rintaro Yanagawa(Developer)",
                             "Others"], label="User Name"),
        gr.Textbox(label="Donor ID"),
        gr.Textbox(label="GCS", placeholder="Optional, leave empty for NaN"),  # GCS入力ボックス
        gr.Dropdown(choices=["positive", "negative"], label="Pupil"),
        gr.Dropdown(choices=["positive", "negative"], label="Gag"),
        gr.Dropdown(choices=["positive", "negative"], label="Corneal"),
        gr.Dropdown(choices=["positive", "negative"], label="Cough"),
        gr.Dropdown(choices=["positive", "negative"], label="Motor"),
        gr.Dropdown(choices=["positive", "negative"], label="OBV"),
        gr.Number(label="End systolic BP"), 
        gr.Number(label="End diastolic BP"), 
        gr.Dropdown(choices=["400~", "300~400", "200~300", "100~200", "~100"], label="Initial PF Ratio"),
        gr.Dropdown(choices=["400~", "300~400", "200~300", "100~200", "~100"], label="End PF Ratio"),
        gr.Dropdown(choices=["~135", "135~146", "146~156", "156~"], label="End Na"),
        gr.Dropdown(choices=["150~", "100~150", "~100"], label="End Plt"),
        gr.Dropdown(choices=["7.35~7.45", "~7.35", "7.45~"], label="End pH"),
        gr.Dropdown(choices=["No", "Yes"], label="Arrest History"),
        gr.Dropdown(choices=["BLUNT", "CARDIOVASCULAR", "DRUG", "INTRACRANIAL", "OTHERS"], label="Mechanism of Injury"),
        gr.Dropdown(choices=["~30", "30~"], label="BMI")
    ],
    outputs="text",
    title="DCD Prediction Calculator"
)

# エクスポート用インターフェース（ボタンをクリックしてダウンロード）
iface_export = gr.Interface(
    fn=export_to_excel,
    inputs=[],
    outputs=gr.File(label="Download Excel file"),
    title="Export Predictions to Excel",
)

# 2つのインターフェースを統合
with gr.Blocks() as demo:
    gr.Markdown("### DCD Prediction Calculator")
    iface.render()
    gr.Markdown("### Export Predictions to Excel")
    iface_export.render()

demo.launch(share=True)
