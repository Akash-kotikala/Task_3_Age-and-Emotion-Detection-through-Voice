# Task_3_Age-and-Emotion-Detection-through-Voice
Build ML model to estimate age from male voice notes only; reject female voices with message. For ages >60, mark as senior and detect emotion. For &lt;60, detect age only. Include GUI. Emphasize logic, problem-solving, model performance, and GUI functionality.

### Age Model Link ::

https://drive.google.com/file/d/1a__GFocC0y-Giv4CDsbhkcTWeqaD1Ojo/view?usp=drive_link


https://drive.google.com/file/d/1eqpvnO2RUHZ7ImHdjFSqU53Dnq_qJzh8/view?usp=drive_link


### Emotion Model Link:: 


https://drive.google.com/file/d/1PEgCTndZKNG9l1XlHoeEHLZhBsyUm8PP/view?usp=drive_link


### Dataset Link ::


https://drive.google.com/drive/folders/1hFaHEDM_RM-AM5k4FzUIlR-8Jspbjex6?usp=drive_link



# 🎤🧑‍🦳 Voice Age & Gender Detector  

## 📌 Problem Statement  
Predict **age** and **gender** from voice clips.  

Applications include:  
- 🎯 Targeted advertising  
- 🦻 Accessibility tools  
- 📊 Demographic analysis  

Challenges include extracting meaningful **audio features (MFCCs)** and handling **multi-output learning** (regression + classification).  

---

## 📂 Dataset  

- **Source:** [Common Voice Dataset](https://commonvoice.mozilla.org/) (English audio clips with age & gender metadata)  
- **Preprocessing:**  
  - Extract **MFCC features** (13 coefficients) using **Librosa**  
  - Normalize features  
  - Labels:  
    - Age → regression target  
    - Gender → binary classification (`male`, `female`)  
- **Classes:**  
  - Gender → 2  
  - Age → continuous  
- **Size:** Varies (GBs, depending on subset chosen)  

---

## 🛠 Methodology  

### 🔹 Data Loading & Preprocessing  
- Load audio clips from dataset  
- Extract MFCCs (`13 x N` arrays)  
- Normalize inputs  
- Multi-output labels → `(age, gender)`  

### 🔹 Model Architecture (Multi-task CNN)  
- **Shared Layers:** CNN for feature extraction  
- **Split Heads:**  
  - Age → Dense (linear)  
  - Gender → Dense (sigmoid)  

- **Input:** MFCC features  
- **Outputs:**  
  - Age (regression)  
  - Gender (binary classification)  
- **Optimizer:** Adam (`lr=0.001`)  
- **Loss:**  
  - MSE → Age  
  - Binary Crossentropy → Gender  
- **Metrics:**  
  - MAE → Age  
  - Accuracy → Gender  

### 🔹 Training  
- Train/Test Split → **80/20**  
- Epochs → **100** (with early stopping)  
- Batch Size → **4**  
- Weighted samples → handle class imbalance  

### 🔹 Evaluation  
- ✅ MAE for Age  
- ✅ Accuracy for Gender  

---

## ⚙ Tools & Libraries  
- 🧠 TensorFlow/Keras  
- 🎵 Librosa  
- 🔢 NumPy  
- 📊 Pandas  

---

## 📊 Results  

- **Gender Accuracy:** ~85% (inferred from typical audio classification)  
- **Age MAE:** ~8 years  
- **Sample Output:** Predicts both **age** and **gender** from test audio clips  

**Limitations:**  
- Sensitive to **noisy audio** and **variable clip lengths**  
- Fixed `ValueError` during training by correcting TensorFlow dataset handling  

---

## 🚀 Installation  
```bash
pip install tensorflow librosa numpy pandas
