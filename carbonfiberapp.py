import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import google.generativeai as genai
import io
import plotly.graph_objs as go

# Set your Gemini API key
genai.configure(api_key="-----------")

st.set_page_config(page_title="Carbon Fiber Explorer", layout="wide")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"carbon_fiber_defect_detection_model2.h5", compile=False)
    return model

model = load_model()

# Preprocess uploaded image
def preprocess_image(uploaded_file, target_size=(128,128)):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Gemini defect classification
def classify_with_gemini(image):
    model = genai.GenerativeModel("gemini-2.0-flash")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    prompt = (
        "Analyze this carbon fiber defect image and respond ONLY with two things:\n"
        "1. The type of defect (Defect: ...)\n"
        "2. The likely root cause (Possible Root Cause: ...)\n"
        "No extra explanation."
    )

    response = model.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": img_byte_arr}
    ])
    return response.text

# Gemini SEM image analysis
def analyze_sem_image(image):
    model = genai.GenerativeModel("gemini-2.0-flash")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    prompt = (
        "Analyze this SEM image of a material. Provide:\n"
        "1. Observed features (grain, defects, voids, cracks, roughness).\n"
        "2. Possible material issues.\n"
        "3. Recommendations to improve material quality.\n"
        "Keep it concise."
    )

    response = model.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": img_byte_arr}
    ])
    return response.text

# Streamlit interface
st.title("🚀 Carbon Fiber Explorer & Defect Analysis App")

# Sidebar
page = st.sidebar.radio("Choose Section", [
    "📖 Carbon Fiber Overview",
    "🏭 Manufacturing Process (Interactive)",
    "🛠️ Defect Detection",
    "🔍 Defect Classification",
    "🔬 SEM Image Analysis",
    "🎥 3D Carbon Fiber Weave Model"
])

# Tab 1: Overview
if page == "📖 Carbon Fiber Overview":
    st.header("📖 Carbon Fiber Overview")
    st.markdown("""
    **Carbon Fiber** is a material consisting of extremely thin fibers (~5-10 μm diameter) composed mostly of carbon atoms.  
    It is known for:
    - High tensile strength
    - Low weight
    - High stiffness
    - High chemical resistance
    - High temperature tolerance

    Common applications include:
    - Aerospace and aviation
    - Automotive (F1, supercars)
    - Sports equipment
    - Civil engineering
    - Wind turbines

    ### Why carbon fiber is special:
    - Strength-to-weight ratio is among the highest of all materials.
    - Customizable: can be layered in different orientations.
    - Electrically conductive.
    """)
    st.image(r"C:\Users\prana\Downloads\download (4).jpeg", caption="Microscopic View of Carbon Fiber")

# Tab 2: Manufacturing Process
elif page == "🏭 Manufacturing Process (Interactive)":
    st.header("🏭 Carbon Fiber Manufacturing Process")
    st.markdown("""
    The manufacturing of carbon fiber involves multiple precise steps:
    """)
    
    steps = [
        "1️⃣ Raw Material Preparation",
        "2️⃣ Spinning (Fiber Formation)",
        "3️⃣ Stabilization (Oxidation)",
        "4️⃣ Carbonization (High-Temperature Treatment)",
        "5️⃣ Surface Treatment",
        "6️⃣ Sizing (Protective Coating)",
        "7️⃣ Weaving/Composite Manufacturing"
    ]
    
    choice = st.selectbox("Select a manufacturing step to learn more:", steps)
    
    images_urls = {
        "Raw Material Preparation": "https://www.researchgate.net/profile/Yongdong-Xie/publication/308726623/figure/fig1/AS:668755340939264@1536453052234/Schematic-of-polyacrylonitrile-PAN-based-carbon-fiber-preparation-process.png",
        "Spinning (Fiber Formation)": "https://www.researchgate.net/profile/Yongdong-Xie/publication/308726623/figure/fig3/AS:669263656906752@1536426747095/Spinning-process-of-PAN-based-carbon-fibers.png",
        "Stabilization (Oxidation)": "https://ars.els-cdn.com/content/image/1-s2.0-S0264127517302748-gr2.jpg",
        "Carbonization": "https://www.researchgate.net/publication/316536232/figure/fig3/AS:613672498212869@1523387410365/The-carbonization-process-of-PAN-based-carbon-fiber.png",
        "Surface Treatment": "https://www.researchgate.net/publication/341684790/figure/fig2/AS:894967927926785@1590133939542/Surface-treatment-of-carbon-fibers.png",
        "Sizing": "https://www.researchgate.net/publication/319264131/figure/fig1/AS:614121658920960@1523444380945/Schematic-representation-of-carbon-fiber-sizing-and-matrix-interaction.png",
        "Weaving/Composite Manufacturing": "https://www.researchgate.net/publication/341497952/figure/fig2/AS:897229235482626@1590516618979/Process-of-manufacturing-carbon-fiber-reinforced-polymer-CFRP-composites.png",
    }

    if choice.startswith("1️⃣"):
        st.info("Polyacrylonitrile (PAN) or Pitch is used as the precursor for most carbon fibers. PAN is dissolved and spun into fibers.")
        st.image(images_urls["Raw Material Preparation"], caption="Raw Material Preparation")
    elif choice.startswith("2️⃣"):
        st.info("PAN fibers are spun into long filaments using wet or dry jet wet spinning processes.")
        st.image(images_urls["Spinning (Fiber Formation)"], caption="Spinning Process")
    elif choice.startswith("3️⃣"):
        st.info("Fibers are heated (~200-300°C) in air to stabilize and cross-link polymer chains, forming a heat-resistant structure.")
        st.image(images_urls["Stabilization (Oxidation)"], caption="Stabilization (Oxidation)")
    elif choice.startswith("4️⃣"):
        st.info("Fibers are heated to 1000-3000°C in an inert atmosphere to convert them into nearly pure carbon.")
        st.image(images_urls["Carbonization"], caption="Carbonization")
    elif choice.startswith("5️⃣"):
        st.info("Surface is treated to improve bonding with resins by adding functional groups to the fiber surface.")
        st.image(images_urls["Surface Treatment"], caption="Surface Treatment")
    elif choice.startswith("6️⃣"):
        st.info("Fibers are coated with sizing (epoxy or polymer) to protect them during handling and improve resin compatibility.")
        st.image(images_urls["Sizing"], caption="Sizing")
    elif choice.startswith("7️⃣"):
        st.info("Fibers are woven or used in composites (resin transfer molding, filament winding, autoclave curing).")
        st.image(images_urls["Weaving/Composite Manufacturing"], caption="Weaving/Composite Manufacturing")
    
    st.markdown("### Overall Manufacturing Flowchart:")
    st.image("https://www.researchgate.net/profile/Ahmed-Saba/publication/316536232/figure/fig2/AS:613672498208773@1523387410362/Manufacturing-process-of-carbon-fiber-fabric-composites.png", caption="Complete Manufacturing Process Flowchart")

# Tab 3: Defect Detection
elif page == "🛠️ Defect Detection":
    st.header("🛠️ Defect Detection")
    uploaded_file = st.file_uploader("Upload an image for defect detection", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        input_arr = preprocess_image(uploaded_file)
        pred = model.predict(input_arr)[0][0]

        if pred > 0.5:
            st.error(f"⚠️ Defect Detected! Confidence: {pred:.2f}")
        else:
            st.success(f"✅ No Defect Detected. Confidence: {(1 - pred):.2f}")

# Tab 4: Defect Classification
elif page == "🔍 Defect Classification":
    st.header("🔍 Defect Classification")
    uploaded_file = st.file_uploader("Upload an image for defect classification", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Defect Image", use_column_width=True)
        st.info("Classifying defect type and root cause...")

        try:
            result = classify_with_gemini(image)
            st.write("**Classification Result:**")
            st.success(result)
        except Exception as e:
            st.error(f"❌ failed: {e}")

# Tab 5: SEM Image Analysis
elif page == "🔬 SEM Image Analysis":
    st.header("🔬 SEM Image Analysis")
    uploaded_file = st.file_uploader("Upload an SEM image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded SEM Image", use_column_width=True)
        st.info("Analyzing SEM image...")

        try:
            result = analyze_sem_image(image)
            st.write("**SEM Analysis Result:**")
            st.success(result)
        except Exception as e:
            st.error(f"❌ failed: {e}")

# Tab 6: 3D Model Tab
elif page == "🎥 3D Carbon Fiber Weave Model":
    st.header("🎥 3D Interactive Carbon Fiber Weave Model")
    
    st.markdown("""
    This is a **simulated 3D model** of a Carbon Fiber Weave pattern.  
    You can **rotate**, **zoom**, and explore the weave structure interactively.

    *Note*: This is a plain weave simulation.
    """)

    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    x, y = np.meshgrid(x, y)
    z = 0.2 * np.sin(2 * np.pi * x / 2) * np.cos(2 * np.pi * y / 2)

    surface = go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale='Greys',
        showscale=False,
        opacity=0.9
    )

    layout = go.Layout(
        title='3D Carbon Fiber Weave Simulation',
        autosize=True,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Weave Depth',
            aspectratio=dict(x=1, y=1, z=0.3)
        )
    )

    fig = go.Figure(data=[surface], layout=layout)
    st.plotly_chart(fig, use_container_width=True)

