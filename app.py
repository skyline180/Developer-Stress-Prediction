import gradio as gr
import pandas as pd
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

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
        "Remote_Work": remote_work
    }])

    stress = float(model.predict(input_df)[0])
    stress = max(0, min(100, stress))

    # Stress category + advice
    if stress < 34:
        label = "Low Stress"
        color = "#2ecc71"
        advice = "You're doing great! Maintain your current work-life balance and keep prioritizing rest and focus."
        emoji = "😌"
    elif stress < 67:
        label = "Moderate Stress"
        color = "#f39c12"
        advice = "You're managing, but stress is building up. Consider taking short breaks, reducing distractions, and improving sleep."
        emoji = "😐"
    else:
        label = "High Stress"
        color = "#e74c3c"
        advice = (
            "Your stress level is high. It's strongly recommended to slow down, "
            "take breaks, reduce workload if possible, and prioritize sleep. "
            "Prolonged stress can impact health and performance."
        )
        emoji = "😫"

    # HTML output
    html = f"""
    <div style="width:100%; max-width:520px; font-family: Arial, sans-serif;">
    
        <div style="font-size:20px; font-weight:bold; margin-bottom:6px; color:#000;">
            Stress Level: {stress:.1f}/100 {emoji}
        </div>
    
        <div style="font-size:16px; margin-bottom:8px; color:#000;">
            Status:
            <span style="color:{color}; font-weight:bold;">
                {label}
            </span>
        </div>
    
        <div style="
            height:28px;
            background:#e0e0e0;
            border-radius:14px;
            overflow:hidden;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
            margin-bottom:12px;
        ">
            <div style="
                height:100%;
                width:{stress}%;
                background: linear-gradient(
                    90deg,
                    #2ecc71 0%,
                    #f39c12 50%,
                    #e74c3c 100%
                );
                transition: width 0.6s ease-in-out;
            "></div>
        </div>
    
        <div style="
            padding:12px;
            border-left:5px solid {color};
            background:#f9f9f9;
            border-radius:6px;
            font-size:15px;
            color:#222;
            line-height:1.5;
        ">
            <strong style="color:#000;">Advice:</strong><br>
            <span style="color:#222;">
                {advice}
            </span>
        </div>
    
    </div>
    """
    return html


# Interface
app = gr.Interface(
    fn=predict_stress,
    inputs=[
        gr.Slider(0, 16, value=8, step=1, label="Hours Worked per Day"),
        gr.Slider(0, 12, value=7, step=1, label="Sleep Hours per Night"),
        gr.Slider(0, 50, value=5, step=1, label="Number of Bugs"),
        gr.Slider(0, 60, value=10, step=1, label="Deadline Days Remaining"),
        gr.Slider(0, 10, value=2, step=1, label="Coffee Cups per Day"),
        gr.Slider(0, 20, value=3, step=1, label="Meetings per Week"),
        gr.Slider(0, 10, value=1, step=1, label="Interruptions per Hour"),
        gr.Dropdown(
            ["Junior", "Mid", "Senior"],
            label="Experience Level",
            value="Mid"
        ),
        gr.Dropdown(
            ["Low", "Medium", "High"],
            label="Code Complexity",
            value="Medium"
        ),
        gr.Radio(
            ["Yes", "No"],
            label="Remote Work",
            value="Yes"
        )
    ],
    outputs=gr.HTML(label="Stress Assessment"),
    title="Developer Stress Level Predictor",
    description=(
        "Predict developer stress using work habits, lifestyle factors, "
        "and project complexity. Includes a visual stress meter and "
        "personalized advice."
    )
)

# Launch
if __name__ == "__main__":
    app.launch()
