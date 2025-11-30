# MURA X-Ray Classification System 🏥

An AI-powered web application for detecting abnormalities in musculoskeletal X-ray images using deep learning.

## 🎯 Features

- **Real-time X-ray Analysis**: Upload elbow or finger X-rays for instant abnormality detection
- **Deep Learning Model**: ResNet34-based CNN trained on the MURA dataset
- **Interactive Visualizations**: Confidence metrics, gauge charts, and probability breakdowns
- **User-Friendly Interface**: Clean, medical-themed UI built with Streamlit

## 🚀 Live Demo

[Add your Streamlit Cloud URL here after deployment]

## 📊 Model Performance

- **Architecture**: ResNet34 with custom classification head
- **Training Dataset**: MURA (MUsculoskeletal RAdiographs) from Stanford
- **Supported Body Parts**: Elbow (XR_ELBOW), Finger (XR_FINGER)
- **Model Parameters**: 21.5M

## 🛠️ Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mura-xray-classifier.git
cd mura-xray-classifier

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📦 Requirements

- Python 3.8+
- PyTorch
- Streamlit
- Plotly
- Pillow

## 🎓 About MURA Dataset

MURA (MUsculoskeletal RAdiographs) is one of the largest public radiographic image datasets from Stanford ML Group, containing 40,561 multi-view radiographic images.

## ⚠️ Medical Disclaimer

This application is for educational and research purposes only. It should NOT be used as the sole basis for medical decisions. Always consult qualified healthcare professionals for proper diagnosis and treatment.

## 📄 License

MIT License

## 🙏 Acknowledgments

- Stanford ML Group for the MURA dataset
- PyTorch and Streamlit communities
