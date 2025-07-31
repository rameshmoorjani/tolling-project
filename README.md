# Tolling Project

This project is a full-stack application for vehicle classification and toll management using machine learning and cloud infrastructure.

## Table of Contents

- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Step-by-Step Guide](#step-by-step-guide)
- [KITTI Dataset Info](#kitti-dataset-info)
- [Summary of What You'll Learn](#summary-of-what-youll-learn)

- ## Project Structure

- <pre>tolling-project/
      │── backend/ # FastAPI backend server
      │ ├── data/ # Empty placeholder folder (.gitkeep)
      │ ├── data_object_image_2/ # KITTI images (excluded from Git)
      │ ├── data_object_label_2/ # KITTI labels (excluded from Git)
      │ └── main.py # FastAPI main app
      ├── frontend/ # React TypeScript frontend
      ├── data/
      │ ├── raw/
      │ │ ├── image_2/ # KITTI original left images
      │ │ ├── label_2/ # KITTI original label files
      │ ├── processed/
      │ │ ├── crops/ # Cropped vehicle images (Car, Truck, Cyclist)
      │ │ ├── classification/
      │ │ │ ├── train/ # Train images
      │ │ │ └── val/ # Validation images
      ├── parse_kitti_labels.py # Script to parse labels
      ├── split_dataset.py # Script to split dataset</pre>

---

## How to Run

1. Setup backend environment
2. Start FastAPI server
3. Run frontend app

---

## Step-by-Step Guide

### Step 1: Setup Your Development Environment

1. **Install Python and Node.js**
   - Install Python 3.8+ from: https://www.python.org/downloads/
   - Install Node.js (with npm) from: https://nodejs.org/

2. **Create project structure**
   ```bash
   mkdir tolling-project && cd tolling-project
   mkdir backend frontend data

### Step 2: Prepare Python Backend Environment
Create and activate virtual environment

cd backend
python -m venv venv
# Activate (Windows):
venv\Scripts\activate
# or macOS/Linux:
source venv/bin/activate
Install dependencies

pip install fastapi uvicorn tensorflow numpy pydantic python-multipart opencv-python
Create FastAPI app

python
# backend/main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Tolling API is running"}
Run backend


uvicorn main:app --reload
# Open http://localhost:8000/
### Step 3: Prepare Frontend Environment
Initialize React app


cd ../frontend
npx create-react-app . --template typescript
Start frontend


npm start
# Open http://localhost:3000/
### Step 4: Download and Explore KITTI Dataset
How to Download KITTI Dataset Images and Labels
Visit: http://www.cvlibs.net/datasets/kitti/

Go to the Object Detection dataset section.

Download:

data_object_image_2.zip

data_object_label_2.zip

Extract and place contents:


backend/data_object_image_2/   # contains images
backend/data_object_label_2/   # contains label files
These files are ignored in Git using .gitignore.

### Step 5: Prepare Data for Model Training

**Parse KITTI Labels**  
Use the script [`parse_kitti_labels.py`](./parse_kitti_labels.py) to extract bounding boxes and filter vehicle types (Car, Truck, Cyclist). This script will crop the images and save them for training.

```bash
python parse_kitti_labels.py
```

**Output directory:**

```bash
data/processed/crops/
```

---

**Split Dataset into Train and Validation Sets**  
Use the script [`split_dataset.py`](./split_dataset.py) to split the cropped vehicle images into training and validation sets (80/20 split).

```bash
python split_dataset.py
```

**Output directory structure:**

```
data/processed/classification/
├── train/
└── val/
```

---

### Step 6: Train Vehicle Classification Model

**Train the Model**  
Use the script [`train_model.py`](./train_model.py) to train a CNN model using TensorFlow. You can use MobileNetV2 or a custom CNN architecture.

**Example command to run:**

```bash
python train_model.py
```

**What the script does:**

- Loads images from `data/processed/classification/train/` and `val/`
- Resizes and normalizes images
- Applies data augmentation *(optional)*
- Trains a model and evaluates on the validation set
- Saves the trained model to a directory

**Model output directory:**

```
vehicle_classifier/
├── saved_model.pb
└── variables/
```

### Step 7: Integrate Model into FastAPI
Load model in main.py

Add /predict POST endpoint

Accept image upload or base64 string

Preprocess and predict

Return JSON with prediction result

### Step 8: Connect Frontend to Backend
Add image upload form in React

Use Axios/Fetch to call /predict API

Display results on screen
