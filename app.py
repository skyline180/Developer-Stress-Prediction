import gradio as gr
import pandas as pd
import numpy as np
import pickle

# load model
with open("rf_stress_model.pkl", "rb") as file:
    model = pickle.load(file)


# prediction function
def predict_stress(
    hours_worked,
    sleep_hours,
    bugs,
    deadline_days,
    coffee_cups,
    meetings,
    interruptions,
    experience_years,
    code_complexity,
    remote_work
):
    input_df = pd.DataFrame([{
        "Hours_Worked": hours_worked,
        "Sleep_Hours": sleep_hours,
        "Bugs": bugs,
        "Deadline_Days": deadline_days,
        "Coffee_Cups": coffee_cups,
        "Meetings": meetings,
        "Interruptions": interruptions,
        "Experience_Years": experience_years,
        "Code_Complexity": code_complexity,
        "Remote_Work": int(remote_work)
    }])

    prediction = model.predict(input_df)
    return f"Predicted Stress Level: {prediction[0]:.2f}"

# Gradio interface
app = gr.Interface(
    fn=predict_stress,
    inputs=[
        gr.Number(label="Hours Worked per Week"),
        gr.Number(label="Sleep Hours per Night"),
        gr.Number(label="Number of Bugs"),
        gr.Number(label="Deadline Days Remaining"),
        gr.Number(label="Coffee Cups per Day"),
        gr.Number(label="Meetings per Week"),
        gr.Number(label="Interruptions per Day"),
        gr.Number(label="Years of Experience"),
        gr.Number(label="Code Complexity (1-10)"),
        gr.Radio([0, 1], label="Remote Work (0 = No, 1 = Yes)")
    ],
    outputs=gr.Textbox(label="Stress Level Prediction"),
    title="Developer Stress Level Predictor",
    description="Predict developer stress using work, lifestyle, and code metrics."
)

# launch
app.launch(share=True) 