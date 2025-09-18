import gradio as gr
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import traceback

df = pd.read_csv("Churn_Modelling_3000_Top8.csv")
if "Exited" not in df.columns:
    raise SystemExit("CSV must contain 'Exited' column")
features = df.drop(columns=["Exited"]).columns.tolist()
X = df[features]
y = df["Exited"]

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in features if c not in num_cols]

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

pre = ColumnTransformer([("num", StandardScaler(), num_cols), ("cat", ohe, cat_cols)], remainder="drop")
model = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
model.fit(X, y)

defaults = {}
for c in features:
    if c in cat_cols:
        defaults[c] = str(df[c].astype(str).mode()[0])
    else:
        defaults[c] = float(df[c].median())

def predict(row_index, *args):
    try:
        use_index = False
        if row_index is not None:
            try:
                idx = int(row_index)
                if 0 <= idx < len(df):
                    use_index = True
            except Exception:
                use_index = False
        if use_index:
            idx = int(row_index)
            row = df.iloc[idx][features]
            input_df = pd.DataFrame([row])
            for c in cat_cols:
                input_df[c] = input_df[c].astype(str)
            proba = model.predict_proba(input_df)[0][1]
            pred = int(proba >= 0.5)
            actual = int(df.iloc[idx]["Exited"])
            return ("Will Churn" if pred == 1 else "Will Not Churn"), float(proba), f"Actual: {actual}"
        else:
            input_data = {f: (args[i] if args[i] is not None else defaults[f]) for i, f in enumerate(features)}
            input_df = pd.DataFrame([input_data])
            for c in cat_cols:
                input_df[c] = input_df[c].astype(str)
            proba = model.predict_proba(input_df)[0][1]
            pred = int(proba >= 0.5)
            return ("Will Churn" if pred == 1 else "Will Not Churn"), float(proba), ""
    except Exception as e:
        return f"Error: {str(e)}", None, traceback.format_exc()

inputs = [gr.Number(label="Row Index (-1 for manual)", value=-1, precision=0)]
for c in features:
    if c in cat_cols:
        choices = [str(x) for x in df[c].astype(str).unique().tolist()]
        inputs.append(gr.Dropdown(choices=choices, label=c, value=defaults[c]))
    else:
        inputs.append(gr.Number(label=c, value=defaults[c]))

outputs = [gr.Textbox(label="Prediction"), gr.Number(label="Probability", precision=4), gr.Textbox(label="Notes")]

app = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Churn Prediction (trained on CSV)")
if __name__ == "__main__":
    app.launch()




