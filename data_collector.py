import cv2
import os
import time

# --- Config ---
DATA_DIR = './dataset'
OPEN_DIR = os.path.join(DATA_DIR, 'Open')
CLOSED_DIR = os.path.join(DATA_DIR, 'Closed')
HAAR_FACE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
HAAR_EYE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"

# Ensure directories exist
os.makedirs(OPEN_DIR, exist_ok=True)
os.makedirs(CLOSED_DIR, exist_ok=True)

def main():
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    if face_classifier.empty() or eye_classifier.empty():
        print("Error: XML files not found. Run main.py once to download them automatically.")
        return

    cap = cv2.VideoCapture(0)
    count_open = len(os.listdir(OPEN_DIR))
    count_closed = len(os.listdir(CLOSED_DIR))
    
    print("--- DATA COLLECTOR ---")
    print("Instructions:")
    print("1. Look at the camera.")
    print("2. Hold 'o' to capture OPEN eyes.")
    print("3. Hold 'c' to capture CLOSED eyes.")
    print("4. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        target_dir = None
        
        keys = cv2.waitKey(1)
        if keys & 0xFF == ord('q'):
            break
        elif keys & 0xFF == ord('o'):
            target_dir = OPEN_DIR
            label = "CAPTURING OPEN"
            color = (0, 255, 0)
        elif keys & 0xFF == ord('c'):
            target_dir = CLOSED_DIR
            label = "CAPTURING CLOSED"
            color = (0, 0, 255)
        else:
            label = "Press 'o' or 'c'"
            color = (255, 255, 255)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = eye_classifier.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), color, 1)
                
                if target_dir:
                    eye_img = roi_color[ey:ey+eh, ex:ex+ew]
                    timestamp = int(time.time() * 1000)
                    if target_dir == OPEN_DIR:
                        count_open += 1
                        filename = f"{count_open}_{timestamp}.jpg"
                    else:
                        count_closed += 1
                        filename = f"{count_closed}_{timestamp}.jpg"
                        
                    cv2.imwrite(os.path.join(target_dir, filename), eye_img)
                    
        # Stats
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Open: {count_open}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Closed: {count_closed}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Data Collector', frame)
        
    cap.release()
    cv2.destroyAllWindows()
    print(f"Collection Finished. Total Open: {count_open}, Total Closed: {count_closed}")

if __name__ == "__main__":
    main()
