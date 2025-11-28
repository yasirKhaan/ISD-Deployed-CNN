import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="MURA X-Ray Analyzer",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .normal {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .abnormal {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Model Architecture (same as training)
class MURACNNModel(nn.Module):
    def __init__(self, pretrained=False):
        super(MURACNNModel, self).__init__()
        
        # Load pretrained ResNet34
        self.backbone = models.resnet34(pretrained=pretrained)
        
        # Modify the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MURACNNModel(pretrained=False).to(device)
    
    try:
        checkpoint = torch.load('best_mura_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(model, image, device):
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0
        confidence = probability if prediction == 1 else (1 - probability)
    
    return prediction, probability, confidence

# Main app
def main():
    # Header
    st.title("🏥 MURA X-Ray Classification System")
    st.markdown("### Musculoskeletal Radiograph Abnormality Detection")
    st.markdown("Upload an X-ray image to detect abnormalities in elbow or finger radiographs.")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            """
            This AI-powered system analyzes musculoskeletal X-ray images 
            to detect potential abnormalities.
            
            **Supported Body Parts:**
            - Elbow (XR_ELBOW)
            - Finger (XR_FINGER)
            
            **Model:** ResNet34-based CNN
            
            **Note:** This tool is for educational purposes only 
            and should not replace professional medical diagnosis.
            """
        )
        
        st.header("📊 Model Info")
        model, device = load_model()
        if model is not None:
            st.success("✅ Model loaded successfully")
            st.write(f"**Device:** {device}")
            st.write(f"**Parameters:** {sum(p.numel() for p in model.parameters()):,}")
        else:
            st.error("❌ Failed to load model")
            return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a musculoskeletal X-ray image (ELBOW or FINGER)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded X-Ray', use_container_width=True)
            
            # Add analyze button
            if st.button("🔍 Analyze X-Ray"):
                with st.spinner('Analyzing image...'):
                    # Preprocess and predict
                    processed_image = preprocess_image(image)
                    prediction, probability, confidence = predict(model, processed_image, device)
                    
                    # Store results in session state
                    st.session_state['prediction'] = prediction
                    st.session_state['probability'] = probability
                    st.session_state['confidence'] = confidence
    
    with col2:
        st.header("📋 Analysis Results")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            probability = st.session_state['probability']
            confidence = st.session_state['confidence']
            
            # Display prediction
            if prediction == 0:
                st.markdown(
                    f"""
                    <div class="prediction-box normal">
                        <h2>✅ NORMAL</h2>
                        <p style="font-size: 20px;">No abnormalities detected</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="prediction-box abnormal">
                        <h2>⚠️ ABNORMAL</h2>
                        <p style="font-size: 20px;">Potential abnormality detected</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Confidence metrics
            st.markdown("---")
            st.subheader("📊 Confidence Metrics")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Prediction Confidence", f"{confidence*100:.2f}%")
            with col_b:
                st.metric("Abnormality Score", f"{probability*100:.2f}%")
            
            # Confidence gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence Level"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#28a745" if prediction == 0 else "#dc3545"},
                    'steps': [
                        {'range': [0, 50], 'color': "#f8d7da"},
                        {'range': [50, 75], 'color': "#fff3cd"},
                        {'range': [75, 100], 'color': "#d4edda"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Probability breakdown
            st.markdown("---")
            st.subheader("📈 Probability Breakdown")
            
            prob_data = {
                'Classification': ['Normal', 'Abnormal'],
                'Probability': [(1-probability)*100, probability*100]
            }
            
            fig2 = go.Figure(data=[
                go.Bar(
                    x=prob_data['Classification'],
                    y=prob_data['Probability'],
                    marker_color=['#28a745', '#dc3545'],
                    text=[f"{p:.2f}%" for p in prob_data['Probability']],
                    textposition='auto',
                )
            ])
            fig2.update_layout(
                yaxis_title="Probability (%)",
                xaxis_title="Classification",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Medical disclaimer
            st.markdown("---")
            st.warning(
                """
                ⚠️ **Medical Disclaimer:** This is an AI-based screening tool 
                and should NOT be used as the sole basis for medical decisions. 
                Always consult with qualified healthcare professionals for proper 
                diagnosis and treatment.
                """
            )
        else:
            st.info("👆 Upload an X-ray image and click 'Analyze' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>MURA X-Ray Classification System | Powered by Deep Learning</p>
            <p>Built with PyTorch & Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()