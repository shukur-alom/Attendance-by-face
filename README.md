# Face Detection and Data Addition System

This system is designed to detect faces in video, check if they match a predefined dataset, and add corresponding data to a CSV file. To use the system, follow these steps:

## Prerequisites

- Python
- Install the required libraries by running:

```
  pip install -r requirements.txt
```


## Image Naming Convention
Ensure your image files follow this naming convention: "Name Number.jpg" (e.g., "Shukur 1.jpg").

1. Run train_model.py to train the face detection model:

```
python train_model.py
```
2. Run main_work.py to detect faces, check for matches in the dataset, and add data to the CSV file:
   
```
python main_work.py
```

The system will update the CSV file with corresponding data if a match is found.
