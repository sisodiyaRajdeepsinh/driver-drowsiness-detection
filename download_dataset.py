import kagglehub
import os
import shutil
import glob

def setup_dataset():
    print("Downloading MRL Eye Dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("prasadvpatil/mrl-dataset")
    print("Dataset downloaded to:", path)

    # Destination paths
    base_dir = './dataset'
    open_dir = os.path.join(base_dir, 'Open')
    closed_dir = os.path.join(base_dir, 'Closed')

    # Create directories if they don't exist
    os.makedirs(open_dir, exist_ok=True)
    os.makedirs(closed_dir, exist_ok=True)

    print("Organizing dataset into Open/Closed folders...")
    
    # MRL dataset structure is often: /mrl_eye_2018_01/subject_id/*.png
    # But usually Kaggle datasets might be flat or slightly different.
    # We will search recursively for all .png files.
    
    # File naming convention: subjectID_gender_glasses_eyeState_reflections_lighting.png
    # eyeState: 0 = closed, 1 = open
    
    all_images = glob.glob(os.path.join(path, '**', '*.png'), recursive=True)
    print(f"Found {len(all_images)} images total. Processing...")

    count_open = 0
    count_closed = 0

    for img_path in all_images:
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        
        # Safety check for filename format
        if len(parts) >= 4:
            eye_state = parts[4] # Index 4 is usually the state in MRL dataset convention
            
            # 0 = Closed, 1 = Open
            if eye_state == '0':
                shutil.copy(img_path, os.path.join(closed_dir, filename))
                count_closed += 1
            elif eye_state == '1':
                shutil.copy(img_path, os.path.join(open_dir, filename))
                count_open += 1
                
    print(f"Done! Organized {count_open} Open eyes and {count_closed} Closed eyes.")
    print(f"Data is ready in '{base_dir}' for training.")

if __name__ == "__main__":
    setup_dataset()
