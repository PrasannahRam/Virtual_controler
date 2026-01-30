✋ Hand Motion Detection using MediaPipe & Neural Network
This repository implements a hand motion (dynamic gesture) detection system using MediaPipe for hand landmark extraction and a custom neural network for motion classification.
MediaPipe provides hand positions, not motion directly. This project solves that limitation by learning temporal changes of hand landmarks across multiple frames.
🔑 Core Concept
•	MediaPipe detects 21 hand landmarks per frame
•	Motion is derived using frame-to-frame landmark deviations
•	Each gesture is represented as a fixed-length sequence of frames
•	A neural network is trained to classify motion patterns
•	Real-time prediction is done via a Flask server
📁 Project Structure
. ├── Hand_detector.py ├── MotionDetectorServer.py ├── NN_trainer.py ├── testModel.py ├── visualizeData.py ├── ControlCursor.py ├── gesture_dataset_NN.json └── README.md
📄 File Descriptions
Hand_detector.py
•	MediaPipe hand tracking
•	Landmark normalization (relative to wrist)
•	Motion extraction (dx, dy between frames)
•	Gesture frame interpolation
•	Training data collection
•	Holding-pose detection (indexing, closed palm)
MotionDetectorServer.py
•	Runs as a Flask server
•	Loads the trained neural network
•	Receives motion sequences via HTTP
•	Returns predicted gesture labels
•	Reason for server: MediaPipe and TensorFlow run in separate environments.
NN_trainer.py
•	Loads motion dataset (gesture_dataset_NN.json)
•	Prepares data for training
•	Trains the neural network
•	Saves the trained model
testModel.py
•	Runs real-time MediaPipe detection
•	Captures motion sequences
•	Sends data to MotionDetectorServer.py
•	Receives predicted motion
•	Triggers actions (e.g., cursor control)
visualizeData.py
•	Visualizes landmark motion paths
•	Verifies dataset quality
•	Helps detect noisy or invalid samples before training
gesture_dataset_NN.json
•	Stores motion training samples.
•	Structure: { “label”: “gesture_name”, “sequence”: [ [[dx, dy], [dx, dy], …] ] }
🧠 Motion Processing Pipeline
1.	Capture hand landmarks
2.	Normalize landmarks using wrist as origin
3.	Compute frame-to-frame deviations
4.	Interpolate motion to fixed length (15 frames)
5.	Feed sequence into neural network
🖥 Requirements
•	Python 3.8+
•	OpenCV
•	MediaPipe
•	NumPy
•	TensorFlow
•	Flask
•	PyAutoGUI
▶️ Usage
Start Motion Detection Server python MotionDetectorServer.py
Run Real-Time Gesture Detection python testModel.py
Collect Training Data rec = Recorder(trainingMode=True) rec.scan() save_sample(rec.data, “gesture_label”)
Train the Neural Network python NN_trainer.py
🔮 Future Enhancements
•	LSTM / GRU based temporal models
•	Z-axis (depth) motion support
•	Multi-hand gesture detection
•	TensorFlow Lite deployment
•	Gesture confidence scoring
👤 Author
Developed as a custom hand motion recognition system combining:
•	Computer Vision
•	Signal Processing
•	Machine Learning

