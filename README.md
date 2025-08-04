# 🧠 Carbon Fiber Defect Detection and Classification App


A **Streamlit-based interactive app** to detect, classify, and analyze defects in **carbon fiber composites**, perform **SEM image evaluations**, and simulate **3D carbon fiber weave structures**.

Powered by:
- **TensorFlow** (defect detection)
- **Google Gemini Pro (Generative AI)** for intelligent classification & root cause analysis
- **Plotly** for 3D visualization
- **Streamlit** for intuitive interaction

---

## 🚀 Features

### 🛠️ Defect Detection
- Upload a carbon fiber image
- TensorFlow CNN detects presence of visual surface defects
- Returns probability/confidence score

### 🔍 Defect Classification + Root Cause (via Gemini)
- Upload defect image
- Gemini classifies defect type and predicts likely root cause (e.g., poor curing, pressure mismanagement)
- Lightweight generative model (Gemini 2 Flash)

### 🔬 SEM Image Analysis
- Analyze SEM images of composites
- Gemini returns:
  - Observed microstructural features (voids, cracks, fiber pull-out)
  - Potential material issues
  - Recommendations

### 🎥 3D Weave Simulation
- Interactive 3D carbon fiber weave visualization
- Simulates a sinusoidal weave structure using Plotly
- Explore weave geometry and interlacing patterns

### 📚 Carbon Fiber Knowledge Hub
- Overview of carbon fiber materials
- Interactive breakdown of **manufacturing steps**:
  - Raw material → spinning → stabilization → carbonization → surface treatment → sizing → weaving
- Linked visuals + flowcharts

---
