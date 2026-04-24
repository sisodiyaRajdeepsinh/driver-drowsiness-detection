import cv2
import torch
import numpy as np
import os
from collections import deque
from model import DrowsinessCNN
import time
import urllib.request
from torchvision import transforms
import pygame
import bz2
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import config

# --- Config ---
MODEL_PATH = 'drowsines_model.pth'
ALARM_SOUND_PATH = 'alarm.wav' 
SCORE_THRESHOLD = 45
SHAPE_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# --- Helpers ---
def download_predictor():
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        print(f"Downloading {SHAPE_PREDICTOR_PATH}.bz2...")
        bz2_path = SHAPE_PREDICTOR_PATH + ".bz2"
        urllib.request.urlretrieve(SHAPE_PREDICTOR_URL, bz2_path)
        print("Extracting...")
        with bz2.BZ2File(bz2_path, 'rb') as f_in:
            with open(SHAPE_PREDICTOR_PATH, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(bz2_path)
        print("Done downloading shape predictor.")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """Calculate the Mouth Aspect Ratio (MAR) using dlib 68-point landmarks.
    
    Uses the inner lip landmarks:
      - Vertical: points 50-58, 52-56 (0-indexed from mouth slice: 2-10, 4-8)
      - Horizontal: points 48-54 (0-indexed from mouth slice: 0-6)
    """
    # Vertical mouth distances
    A = distance.euclidean(mouth[2], mouth[10])   # 50 - 58
    B = distance.euclidean(mouth[4], mouth[8])     # 52 - 56
    # Horizontal mouth distance
    C = distance.euclidean(mouth[0], mouth[6])     # 48 - 54
    mar = (A + B) / (2.0 * C)
    return mar

class DrowsinessDetector:
    def __init__(self):
        # 1. Setup Audio at startup
        self.init_audio()
        
        # 2. Load Model Efficiently
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = DrowsinessCNN().to(self.device)
        self.load_weights()
        self.model.eval()

        # 3. Precompile Transforms (CLAHE applied before this)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((24, 24)),
            transforms.ToTensor(),
        ])
        
        # 4. Load Models & Dlib Predictor
        download_predictor()
        print("Initializing Dlib...")
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

        # 5. CLAHE for contrast normalization (handles varying lighting)
        # More aggressive CLAHE for tough lighting
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # State
        self.score = 0
        self.alarm_active = False
        self.yawn_counter = 0  # consecutive yawn frames

        # --- Temporal smoothing buffers ---
        self.ear_history = deque(maxlen=5)         # Rolling EAR average (5 frames)
        self.cnn_history = deque(maxlen=5)          # CNN vote buffer (majority over 5 frames)
        self.face_lost_frames = 0                   # Consecutive frames with no face
        self.FACE_LOST_GRACE = 15                   # Ignore up to 15 face-lost frames

        # Calibration state
        self.calibrated = False
        self.ear_baseline = None
        self.ear_thresh = 0.25  # fallback default, will be overwritten by calibration

    def init_audio(self):
        try:
            pygame.mixer.init()
            if os.path.exists(ALARM_SOUND_PATH):
                pygame.mixer.music.load(ALARM_SOUND_PATH)
            else:
                print("Warning: Alarm file not found.")
        except Exception as e:
            print(f"Audio Init Error: {e}")

    def load_weights(self):
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            print("Model weights loaded.")
        else:
            print("WARNING: No model weights found. Predictions will be random.")

    def play_alarm(self):
        if not self.alarm_active:
            try:
                if pygame.mixer.get_init():
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play(-1) # Loop
                    self.alarm_active = True
            except Exception as e:
                print(f"Play Alarm Error: {e}")

    def stop_alarm(self):
        if self.alarm_active:
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
                self.alarm_active = False
            except Exception as e:
                print(f"Stop Alarm Error: {e}")

    def adaptive_preprocess(self, gray_frame):
        """Apply brightness-adaptive preprocessing for robust detection under any lighting.
        
        Steps:
        1. Measure average brightness of the frame
        2. Apply adaptive gamma correction to normalize brightness
        3. Apply bilateral filter to reduce noise while keeping edges
        4. Apply aggressive CLAHE for local contrast enhancement
        """
        # Measure average brightness
        mean_brightness = np.mean(gray_frame)
        
        # Adaptive gamma: darken if too bright, brighten if too dark
        # Target brightness ~120 (mid-range)
        if mean_brightness < 10:
            # Extremely dark — aggressive brightening
            gamma = 0.3
        elif mean_brightness < 50:
            gamma = 0.5
        elif mean_brightness < 80:
            gamma = 0.7
        elif mean_brightness > 200:
            # Very bright / washed out — darken
            gamma = 1.8
        elif mean_brightness > 160:
            gamma = 1.3
        else:
            gamma = 1.0  # Good lighting, no correction needed
        
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
            ]).astype("uint8")
            gray_frame = cv2.LUT(gray_frame, table)
        
        # Bilateral filter: reduce noise but preserve edges (important for landmarks)
        gray_frame = cv2.bilateralFilter(gray_frame, d=5, sigmaColor=50, sigmaSpace=50)
        
        # CLAHE for local contrast normalization
        gray_frame = self.clahe.apply(gray_frame)
        
        return gray_frame

    def preprocess_eyes(self, eye_images):
        """Batch preprocess eye images with CLAHE normalization for better accuracy."""
        tensors = []
        for img in eye_images:
            # Apply CLAHE to normalize contrast/lighting
            enhanced = self.clahe.apply(img)
            # Also apply histogram equalization as an extra step for eye crops
            enhanced = cv2.equalizeHist(enhanced)
            tensors.append(self.transform(enhanced))
        if not tensors:
            return None
        return torch.stack(tensors).to(self.device)

    def calibrate(self, cap):
        """Run a calibration phase to determine the user's personal EAR baseline.
        
        The user should look at the camera with eyes open naturally.
        The system averages EAR over CALIBRATION_FRAMES to set a personalized threshold.
        Uses adaptive preprocessing so calibration works under any lighting.
        """
        print("=" * 50)
        print("  CALIBRATION: Look at the camera normally.")
        print("  Keep your eyes open for ~5 seconds.")
        print("=" * 50)

        ear_values = []
        frame_count = 0
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        target_frames = config.CALIBRATION_FRAMES

        while frame_count < target_frames:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Use full adaptive preprocessing during calibration too
            gray = self.adaptive_preprocess(gray)

            # Upsample=1 for better detection in poor lighting
            subjects = self.detect(gray, 1)

            for subject in subjects:
                shape = self.predict(gray, subject)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                ear_values.append(ear)

                # Draw eye contours during calibration
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Show progress bar
            progress = frame_count / target_frames
            bar_w = int(progress * 300)
            cv2.rectangle(frame, (50, 30), (350, 60), (50, 50, 50), -1)
            cv2.rectangle(frame, (50, 30), (50 + bar_w, 60), (0, 255, 128), -1)
            cv2.rectangle(frame, (50, 30), (350, 60), (255, 255, 255), 2)
            cv2.putText(frame, "CALIBRATING... Look at camera", (50, 90),
                        font, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{int(progress * 100)}%", (170, 55),
                        font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('Drowsiness Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return False

            frame_count += 1

        if ear_values:
            # Remove outliers (bottom/top 10%) for robust baseline
            sorted_ears = sorted(ear_values)
            trim = max(1, len(sorted_ears) // 10)
            trimmed = sorted_ears[trim:-trim] if len(sorted_ears) > 2 * trim else sorted_ears
            
            self.ear_baseline = np.mean(trimmed)
            # Threshold = 75% of baseline — generous to handle lighting noise
            self.ear_thresh = self.ear_baseline * 0.75
            self.calibrated = True
            print(f"  Calibration complete!")
            print(f"  Baseline EAR: {self.ear_baseline:.4f}")
            print(f"  Adaptive Threshold: {self.ear_thresh:.4f}")
            print("=" * 50)
        else:
            print("  WARNING: No face detected during calibration. Using default threshold.")
            self.ear_thresh = 0.25
            self.calibrated = True

        return True
        
    def run(self):
        cap = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        # --- Calibration Phase ---
        if not self.calibrate(cap):
            return  # user pressed 'q' during calibration

        print("Starting video loop... Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # === ADAPTIVE PREPROCESSING for lighting robustness ===
            gray = self.adaptive_preprocess(gray)
            
            # Draw score background
            cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
            
            # Dlib Face Detection (upsample=1 for better small/dark face detection)
            subjects = self.detect(gray, 1)
            
            detected_eyes_imgs = []
            eye_rects = [] # To draw later
            ear_closed = False
            yawn_detected = False
            face_found = False
            current_ear = None
            
            for subject in subjects:
                face_found = True
                shape = self.predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                
                # --- Eye Aspect Ratio ---
                leftEye = shape[self.lStart:self.lEnd]
                rightEye = shape[self.rStart:self.rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                current_ear = ear
                
                # Add to rolling EAR history for temporal smoothing
                self.ear_history.append(ear)
                smoothed_ear = np.mean(self.ear_history)
                
                # Use the SMOOTHED EAR against the calibrated threshold
                if smoothed_ear < self.ear_thresh:
                    ear_closed = True

                # --- Mouth Aspect Ratio (Yawn Detection) ---
                mouth = shape[self.mStart:self.mEnd]
                mar = mouth_aspect_ratio(mouth)
                if mar > config.MAR_THRESH:
                    self.yawn_counter += 1
                else:
                    self.yawn_counter = 0

                if self.yawn_counter >= config.YAWN_CONSEC_FRAMES:
                    yawn_detected = True
                
                # Visualize Dlib Contours
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

                # Show EAR and MAR values on frame (show both raw and smoothed)
                cv2.putText(frame, f"EAR: {ear:.2f} smooth: {smoothed_ear:.2f} (thr: {self.ear_thresh:.2f})",
                            (10, 30), font, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, f"MAR: {mar:.2f}",
                            (10, 60), font, 1, (255, 255, 0), 1, cv2.LINE_AA)
                
                # Extract Eyes using Bounding Rect for CNN evaluation
                for eye_landmarks in [leftEye, rightEye]:
                    (x, y, w, h) = cv2.boundingRect(np.array(eye_landmarks))
                    
                    # Add padding for Model input
                    padding = 5
                    x_start, y_start = max(0, x - padding), max(0, y - padding)
                    x_end, y_end = min(width, x + w + padding), min(height, y + h + padding)
                    
                    eye_img_gray = gray[y_start:y_end, x_start:x_end]
                    if eye_img_gray.size > 0:
                        detected_eyes_imgs.append(eye_img_gray)
                        eye_rects.append((x_start, y_start, x_end - x_start, y_end - y_start))

            # --- Face-lost grace period ---
            if not face_found:
                self.face_lost_frames += 1
                if self.face_lost_frames <= self.FACE_LOST_GRACE:
                    # Don't change the score — assume face is still there (lighting glitch)
                    cv2.putText(frame, "Face lost (grace)", (10, 30),
                                font, 0.8, (0, 128, 255), 1, cv2.LINE_AA)
                else:
                    # Face genuinely lost for too long — slowly decrease score
                    self.score -= 1
                    cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                                font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                self.face_lost_frames = 0

            # Batch Inference via PyTorch Model
            cnn_closed = False
            if detected_eyes_imgs:
                try:
                    with torch.no_grad():
                        input_batch = self.preprocess_eyes(detected_eyes_imgs)
                        outputs = self.model(input_batch)
                        _, predictions = torch.max(outputs, 1) # 0=Closed, 1=Open
                        
                        closed_count = (predictions == 0).sum().item()
                        open_count = (predictions == 1).sum().item()
                        
                        # Add to CNN vote history for temporal smoothing
                        frame_cnn_closed = closed_count > open_count
                        self.cnn_history.append(frame_cnn_closed)
                        
                        # Majority vote over last N frames
                        if sum(self.cnn_history) > len(self.cnn_history) / 2:
                            cnn_closed = True
                        
                        # Show prediction rects
                        for i, rect in enumerate(eye_rects):
                            ex, ey, ew, eh = rect
                            state = predictions[i].item() # 0 or 1
                            color = (0, 0, 255) if state == 0 else (255, 0, 0)
                            cv2.rectangle(frame, (ex,ey), (ex+ew,ey+eh), color, 1)

                except Exception as e:
                    print(f"Inference Error: {e}")

            # --- Refined Scoring: weighted ensemble + yawn ---
            # Only update score based on eye state if a face was found (or within grace)
            if face_found:
                if ear_closed and cnn_closed:
                    # Both agree → high confidence
                    self.score += config.SCORE_INC_EYES_CLOSED
                elif ear_closed:
                    # EAR says closed (reliable geometric measure)
                    self.score += config.SCORE_INC_EAR_ONLY
                elif cnn_closed:
                    # CNN says closed
                    self.score += config.SCORE_INC_CNN_ONLY
                else:
                    # Eyes open — decrease score towards recovery
                    self.score -= config.SCORE_DEC_NORMAL

            # Yawn adds to drowsiness score independently
            if yawn_detected:
                self.score += config.SCORE_INC_YAWN
                cv2.putText(frame, "YAWNING!", (width - 250, 60),
                            font, 2, (0, 165, 255), 2, cv2.LINE_AA)

            # Score Constraints
            if self.score < 0: self.score = 0
            
            # UI Updates
            bar_pct = min(self.score / SCORE_THRESHOLD, 1.0)
            bar_width = int(bar_pct * 200)
            
            # Determine color
            if self.score < SCORE_THRESHOLD * 0.5: bar_color = (0, 255, 0)
            elif self.score < SCORE_THRESHOLD * 0.8: bar_color = (0, 255, 255)
            else: bar_color = (0, 0, 255)
                
            cv2.rectangle(frame, (10, height-40), (10+bar_width, height-10), bar_color, -1)
            cv2.rectangle(frame, (10, height-40), (10+200, height-10), (255,255,255), 2)
            cv2.putText(frame, f'Score: {self.score}', (10,height-50), font, 1, (255,255,255), 1, cv2.LINE_AA)
            
            # Hysteresis: need higher score to ENTER drowsy, lower to EXIT
            # This prevents rapid on/off cycling at the boundary
            if not self.alarm_active:
                is_drowsy = self.score > SCORE_THRESHOLD
            else:
                # Once alarming, require score to drop significantly before stopping
                is_drowsy = self.score > (SCORE_THRESHOLD * 0.6)

            if is_drowsy:
                cv2.putText(frame, "DROWSY!", (100, height-100), font, 3, (0,0,255), 3, cv2.LINE_AA)
                
                if self.score % 10 == 0: # Save snapshot
                     cv2.imwrite(os.path.join(os.getcwd(), 'drowsy_event.jpg'), frame)
                
                self.play_alarm()
            else:
                self.stop_alarm()
            
            # check for notifications — pass is_drowsy (not alarm_active)
            # so that brief score dips don't reset the 5-second alert timer
            from alert_system import notifier
            notifier.check_and_alert(is_drowsy, frame)

            # Show brightness info for debugging
            mean_b = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            cv2.putText(frame, f"Brightness: {mean_b:.0f}", (width - 200, height - 10),
                        font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow('Drowsiness Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.stop_alarm()

if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()
