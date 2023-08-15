# Fire Detection AI using YOLO and Python

## Description

This project implements an AI-based fire detection system using YOLO (You Only Look Once) object detection model in Python. The project aims to enhance safety by detecting fires in LIVE STREAM security photage or images and videos.

## Table of Contents


- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Inference](#inference)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the environment and install dependencies:

```bash
# Create a virtual environment (optional but recommended)
python -m venv fire_detection_env
source fire_detection_env/bin/activate

# Install required packages
pip install ultralitics
pip install Roboflow
pip install supervision
```

## Usage

To use the fire detection AI, follow these steps in your terminal:

1. **Clone the Repository:**

   Clone this repository to your local machine:

   ```bash
   git clone https://github.com/AI-ForestFireDetection/AI-Yolo.git
   cd AI-Yolo
   ```
    Install Dependencies:

    Install the required dependencies using the following command:
   ```bash
    pip install -r requirements.txt
    ```
Run the CLI Menu:

Run the provided Python script to access the CLI menu:

```bash
  python -m App
```
The CLI menu will be displayed with the following options:
   ```bash
        1. Download default training dataset
        2. Train
        3. Validation
        4. Live Test
        5. Quit
   ```
        

Choose an Option:

Choose an option based on your needs by entering the corresponding number and pressing "Enter." 
<br><br>For example, to train the model, choose option "2." To perform a live test, choose option "4."

## Follow the Instructions:

Depending on your chosen option, follow any further instructions provided by the script. For instance, if you choose to train the model, the script will initiate the training process. If you choose a live test, the script will display real-time detections from your webcam.

    Exit the CLI Menu:

    To exit the CLI menu, choose option "5" or press "Ctrl+C."

