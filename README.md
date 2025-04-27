# Facial Classification Project

This repository contains a PyTorch-based facial classification project.

---

## How to Run

1. **Download the Dataset**  
   - Download the two files (`labels.npy` and `faces.npy`) from the provided link.
   - Place these files under `app/data`.

2. **Install Requirements**  
   ```bash
   pip install -r requirements.txt

3. **Dataloader (if needed to create dataset from the raw data, the provided dataset already been preprocessed)**  
   - put all classes all raw data classes into 'app/data' folder
   ```bash
   python app/data/preprocessing/preprocessing.py

4. **Train the Model**  
   ```bash
   python app/train.py

5. **Check Your Model & Artifacts**  
   - The trained model is saved under app/models.
   - Training plots (loss and accuracy) are saved under app/artifacts.

6. **Hyperparameter Tuning**  
   - Refer to notebooks/model_building.ipynb for any custom hyperparameter tuning workflows.

7. **Use the Latest Trained Model**
   - The script app/predict.py automatically locates the most recently modified .pkl file in app/models.
   ```bash
   python app/predict.py
   ```
   - It will prompt for an image filepath and display the predicted label.

### Project Structure
 ```bash
.
├── app
│   ├── data            # Contains faces.npy and labels.npy
│   ├── models          # Model pickle files (trained models)
│   ├── artifacts       # Training artifacts (plots, logs, etc.)
│   ├── train.py        # Main training script
│   └── predict.py      # Loads the latest model for inference 
├── notebooks
│   ├── EDA.ipynb  
│   └── model_building.ipynb   # Hyperparameter tuning
├── requirements.txt
└── README.md

