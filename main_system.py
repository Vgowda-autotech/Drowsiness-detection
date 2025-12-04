import cv2
import numpy as np
from tensorflow.keras.models import load_model
import winsound
import os

# --- CONFIGURATION ---
EYE_MODEL_PATH = 'models/eye_model.h5'
MOUTH_MODEL_PATH = 'models/mouth_model.h5'
ALARM_FREQUENCY = 2500
ALARM_DURATION = 1000

# --- LOAD MODELS ---
print("Loading models...")
eye_model = load_model(EYE_MODEL_PATH)
mouth_model = load_model(MOUTH_MODEL_PATH)
print("Models loaded!")

# --- LOAD HAAR CASCADES (Built-in OpenCV face detectors) ---
# OpenCV usually keeps these in a specific data folder
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# --- HELPER FUNCTION ---
def preprocess_for_model(img_crop):
    """Resize to 64x64, Grayscale, Normalize"""
    try:
        if img_crop is None or img_crop.size == 0:
            return None
        img_crop = cv2.resize(img_crop, (64, 64))
        if len(img_crop.shape) == 3:
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        img_crop = img_crop / 255.0
        img_crop = np.reshape(img_crop, (1, 64, 64, 1))
        return img_crop
    except Exception as e:
        return None

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
drowsy_counter = 0
yawn_counter = 0

# Adjust these sensitivities if needed
DROWSY_LIMIT = 15  # Number of frames eyes are closed before alert
YAWN_LIMIT = 15    # Number of frames mouth is yawning before alert

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect Faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    status_text = "Active"
    color = (0, 255, 0)
    
    # Logic: Only process the largest face (The Driver)
    largest_face = None
    max_area = 0
    
    for (x, y, w, h) in faces:
        if w * h > max_area:
            max_area = w * h
            largest_face = (x, y, w, h)
            
    if largest_face is not None:
        (x, y, w, h) = largest_face
        
        # Draw box around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # --- ROI: EYES ---
        # Eyes are usually in the upper half of the face
        roi_gray_upper = gray[y:y + h//2, x:x + w]
        roi_color_upper = frame[y:y + h//2, x:x + w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray_upper)
        
        eyes_closed_prediction = True # Assume closed until we see open eyes
        
        if len(eyes) > 0:
            # Check every detected eye
            open_eye_count = 0
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color_upper, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 1)
                
                # Extract and Predict
                eye_img = roi_color_upper[ey:ey+eh, ex:ex+ew]
                input_data = preprocess_for_model(eye_img)
                
                if input_data is not None:
                    pred = eye_model.predict(input_data, verbose=0)[0][0]
                    # Check your model logic! 
                    # Usually: 0=Closed, 1=Open.
                    # If pred > 0.5, it is Open.
                    if pred > 0.5: 
                        open_eye_count += 1
            
            # If we found at least one OPEN eye, we are awake
            if open_eye_count > 0:
                eyes_closed_prediction = False
        else:
            # No eyes detected? Could be blinking or looking away.
            # We treat this as "Potential Closed" but be careful.
            pass

        # Update Counters
        if eyes_closed_prediction:
            drowsy_counter += 1
            status_text = f"Sleepy? {drowsy_counter}"
        else:
            drowsy_counter -= 1 if drowsy_counter > 0 else 0

        # --- ROI: MOUTH ---
        # Mouth is usually in the lower third of the face
        # We manually crop that area instead of using a cascade (more stable)
        mouth_y_start = y + int(h * 0.6)
        mouth_y_end = y + h
        mouth_x_start = x + int(w * 0.2)
        mouth_x_end = x + int(w * 0.8)
        
        mouth_img = frame[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
        cv2.rectangle(frame, (mouth_x_start, mouth_y_start), (mouth_x_end, mouth_y_end), (0, 0, 255), 1)
        
        input_mouth = preprocess_for_model(mouth_img)
        if input_mouth is not None:
            m_pred = mouth_model.predict(input_mouth, verbose=0)[0][0]
            
            # Logic: If m_pred > 0.5, it is Yawn (depending on folder order)
            # If your 'Yawn' folder was read second by Keras, it is class 1.
            is_yawning = m_pred > 0.5 
            
            if is_yawning:
                yawn_counter += 1
                status_text = f"Yawning {yawn_counter}"
            else:
                yawn_counter -= 1 if yawn_counter > 0 else 0

        # --- ALERTS ---
        if drowsy_counter > DROWSY_LIMIT:
            status_text = "WAKE UP !!!"
            color = (0, 0, 255)
            winsound.Beep(ALARM_FREQUENCY, 200)
            
        if yawn_counter > YAWN_LIMIT:
            status_text = "Take a Break"
            color = (0, 165, 255)
            winsound.Beep(1000, 200)

    # Display
    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Drowsiness Detection (Haar)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()