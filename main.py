
import gradio as gr
import pandas as pd
import pickle


# ==========================================
# 1. Load Trained Model
# ==========================================

with open("breast_cancer_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]
target_names = model_data["target_names"]


# ==========================================
# 2. Prediction Function
# ==========================================

def predict(
    mean_radius,
    mean_texture,
    mean_smoothness,
    mean_compactness,
    mean_concavity
):

    # Create DataFrame
    input_data = pd.DataFrame([[
        mean_radius,
        mean_texture,
        mean_smoothness,
        mean_compactness,
        mean_concavity
    ]], columns=feature_names)

    # Apply the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get probabilities
    probabilities = model.predict_proba(input_scaled)[0]

    predicted_class = target_names[prediction]

    malignant_probability = probabilities[0]
    benign_probability = probabilities[1]

    return (
        predicted_class.capitalize(),
        f"{malignant_probability:.2%}",
        f"{benign_probability:.2%}"
    )


# ==========================================
# 3. Gradio UI
# ==========================================

with gr.Blocks(title="Breast Cancer Classification") as demo:

    gr.Markdown(
        """
        # 🩺 Breast Cancer Classification

        Enter the five selected features below and click
        **Predict** to classify the sample.

        > This is an educational machine-learning demonstration,
        > not a medical diagnostic tool.
        """
    )

    with gr.Row():

        with gr.Column():

            mean_radius = gr.Slider(
                minimum=6.0,
                maximum=30.0,
                value=14.0,
                step=0.1,
                label="Mean Radius",
                info="Typical range: approximately 6–30"
            )

            mean_texture = gr.Slider(
                minimum=9.0,
                maximum=40.0,
                value=19.0,
                step=0.1,
                label="Mean Texture",
                info="Typical range: approximately 9–40"
            )

            mean_smoothness = gr.Slider(
                minimum=0.05,
                maximum=0.16,
                value=0.10,
                step=0.001,
                label="Mean Smoothness",
                info="Typical range: approximately 0.05–0.16"
            )

            mean_compactness = gr.Slider(
                minimum=0.02,
                maximum=0.35,
                value=0.10,
                step=0.001,
                label="Mean Compactness",
                info="Typical range: approximately 0.02–0.35"
            )

            mean_concavity = gr.Slider(
                minimum=0.0,
                maximum=0.43,
                value=0.09,
                step=0.001,
                label="Mean Concavity",
                info="Typical range: approximately 0–0.43"
            )

            predict_button = gr.Button(
                "🔍 Predict",
                variant="primary"
            )

        with gr.Column():

            prediction_output = gr.Textbox(
                label="Prediction"
            )

            malignant_output = gr.Textbox(
                label="Malignant Probability"
            )

            benign_output = gr.Textbox(
                label="Benign Probability"
            )


    # ==========================================
    # 4. Button Action
    # ==========================================

    predict_button.click(
        fn=predict,
        inputs=[
            mean_radius,
            mean_texture,
            mean_smoothness,
            mean_compactness,
            mean_concavity
        ],
        outputs=[
            prediction_output,
            malignant_output,
            benign_output
        ]
    )


# ==========================================
# 5. Launch
# ==========================================

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True
)

