# 🚗 Real-Time Driver Drowsiness Detection System

## 📌 Project Overview
Fatigue-related driving accidents are a major global safety issue. This project is a real-time computer vision system designed to detect early signs of driver drowsiness (eye closure and yawning) and trigger an alert to prevent accidents.

The system uses **Deep Learning (CNNs)** to analyze facial features and **Computer Vision** to track the driver's state via a live webcam feed.

## 🚀 Key Features
* **Real-Time Detection:** Runs purely on CPU with low latency.
* **Hybrid Detection Logic:** * **Eye Closure:** Triggers alert if eyes are closed for >2 seconds (configurable) to distinguish blinking from sleeping.
    * **Yawning:** Detects persistent mouth opening indicative of fatigue.
* **Driver-Centric:** Automatically detects and isolates the **driver's face** (largest face in frame) to ignore passengers.
* **Audio Alerts:** Uses system sound to wake the driver.
* **Works with Spectacles:** The model is trained on diverse images including glasses and reflections.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Computer Vision:** OpenCV (Haar Cascades for face/eye detection)
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** NumPy, SciPy

## 📊 Model Performance
We trained two custom Convolutional Neural Networks (CNNs):

| Model | Class 1 | Class 2 | Validation Accuracy |
| :--- | :--- | :--- | :--- |
| **Eye Model** | Open | Closed | **~89.0%** |
| **Mouth Model** | Yawn | No Yawn | **~96.3%** |

## 📂 Dataset
1.  **MRL Eye Dataset:** Used for training the eye state detection (Infrared/Grayscale images).
2.  **Custom Mouth Dataset:** Curated specifically for yawning detection.

## ⚙️ Installation & Setup

1.  **Clone or Download** this repository.
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Project Structure:**
    Ensure your folders are organized as follows:
    ```text
    /Project_Root
      ├── models/
      │     ├── eye_model.h5
      │     └── mouth_model.h5
      ├── main_system.py  (Run this for the app)
      ├── train_eye_model.py
      ├── train_mouth_model.py
      └── requirements.txt
    ```

## 🖥️ How to Run
1.  Connect a webcam.
2.  Run the main script:
    ```bash
    python main_system.py
    ```
3.  **To Exit:** Press `q` on your keyboard.

## 🧠 Logic & Thresholds
The system uses a frame-based counter to prevent false alarms:
* **Drowsy Threshold:** Eyes must be detected as "Closed" for **15 consecutive frames** (approx. 0.5 - 1.0 seconds depending on FPS) to trigger a warning.
* **Yawn Threshold:** Mouth must be detected as "Yawning" for **15 consecutive frames**.

## 🔄 Retraining the Models
If you want to improve accuracy with new data:
1.  Place images in `mrl_data/Train` (for eyes) or `mouth_data/train` (for mouth).
2.  Run the training scripts:
    ```bash
    python train_eye_model.py
    python train_mouth_model.py
    ```

## ⚠️ Note for Non-Windows Users
This project uses `winsound` for alerts, which is specific to Windows.
* **Mac/Linux Users:** Please replace `winsound.Beep()` in `main_system.py` with an alternative like `os.system('say "Wake up"')` (Mac) or a library like `playsound`.

Note on Data: Due to file size limits, the dataset is not included in this repository.

Download the MRL Eye Dataset from Kaggle (or the source you used).

Create a folder named mrl_data and place the images there

**Author**
Vasudev Jinnagara Guruprasad
Linkedin: "www.linkedin.com/in/vasudev-jinnagara-guruprasad-29511a398"
Github: "https://github.com/Vgowda-autotech"