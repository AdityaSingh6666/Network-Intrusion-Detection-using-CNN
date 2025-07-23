# 🛡️ Network Intrusion Detection Using CNN

A **Streamlit-based web app** for detecting network intrusions using a Convolutional Neural Network (CNN) trained on the NSL-KDD dataset.

---

## 🚀 Features

* Detects intrusions in network data using a CNN model
* Simple, clean Streamlit interface for testing samples
* Lightweight and easy to deploy
* Great for learning ML deployment pipelines

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/NID-Using-CNN.git
cd NID-Using-CNN
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv .venv
```

### 3️⃣ Activate the environment

**Windows:**

```cmd
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/Scripts/activate
```

### 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🩻 Usage

### 1️⃣ Ensure you have your trained model

Place your trained CNN model in the following path:

```
models/cnn_nids_model.keras
```

### 2️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

The app will launch in your browser, allowing you to upload or input data for real-time network intrusion detection.

---


