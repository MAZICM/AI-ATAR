




# Real-Time Forest Fire Detection
   ## Introduction

The Real-Time Forest Fire Detection project employs cutting-edge deep learning techniques to detect and respond to forest fires promptly. Leveraging YOLO (You Only Look Once) models and efficient object detection algorithms, this project aims to contribute to early fire detection, reducing the risk of catastrophic damage to our natural landscapes.

   ## Project in Action
   ![Fire Detection Demo](2.mp4_out.gif)
   This GIF demonstrates how the fire detection system detects fire in a real-time video stream.
 

## Features

- **Video-Based Detection**: Detect fires in real-time from video streams, enabling swift intervention.
- **Efficient Algorithms**: Utilize YOLO models for accurate and rapid fire detection.
- **Customization**: Easily adapt the models and configurations to suit specific detection requirements.
- **Live Streaming**: Enable live fire detection from webcams or video sources for immediate monitoring.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Installation

1. Clone this repository:
   ```sh
   $ git clone https://github.com/AI-ForestFireDetection/AI-Yolo.git
   ```
2. Access the Repo:
   ```sh
   $ cd AI-Yolo/
   ```
3. Create your Python env :
   ```sh
   $ python3 -m venv venv
   ```
4. Activate your python env:
   ```sh
   $ source  ./venv/bin/activate
   ```
5. Install the required dependencies:
   ```sh
   pip install ultralytics opencv-python
   ```

## Usage
### Video Detection 
Run the `videoDetect()` function to detect fires in a video file.

```sh
python -c 'from Utilities.vDetect import video_Detect; video_Detect()'
```

### Live Stream Detection

Run the `Stream()` function to start a live stream for fire detection.

```sh
python -c 'from Utilities.sDetect import Stream; Stream()'
```

### Model Training

Run the `train()` function to train your own YOLO model.

```sh
python -c 'from Utilities.modelTrain import mTrain; mTrain()'
```

### Validation

Run the `valid()` function to validate your YOLO model.


```sh
python -c 'from Utilities.modelValid import mValid; mValid()'
```


## Configuration

Modify the parameters in the script to adapt the detection to your needs.

## Examples

- To run live stream detection:
  ```sh
  python your_script_name.py
  ```
 
   ### Example Output

   Here's an example of the fire detection output from the live stream:

   ```python
   [Output Example Here]
   ```
  This output demonstrates how the system detects fire in real-time from the video stream.


- To train a YOLO model:
  ```sh
  python your_script_name.py
  ```
  ### Example Output

   Here's an example of the fire detection output from the live stream:

   ```python
   [Output Example Here]
   ```

## Contributing

Contributions are welcome! Fork the repository, create a new branch, and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, feel free to reach out to [MOUHIB Otman](mailto:mouhib.otm@gmail.com) or [BACCARI Rihab](mailto:mouhib.otm@gmail.com).

## Acknowledgements

- Ultralytics for YOLO wrapper
- Roboflow for dataset



/**********************************/



   ## User Reviews

   > "I've been using this forest fire detection system for a while now, and it has significantly improved our response time to potential fire outbreaks. The accuracy and real-time detection capabilities are impressive."
   >
   > — John Doe, Forest Ranger

   ## Note


   > "I've been using this forest fire detection system for a while now, and it has significantly improved our response time to potential fire outbreaks. The accuracy and real-time detection capabilities are impressive."
   
