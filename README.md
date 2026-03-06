### American Sign Language (ASL) Recognition with Deep Learning

The project presents a real-time classification system that is capable of recognizing 36 distinct ASL alphabetical and numeral gestures using deep learning and computer vision. While not capturing the full depth and richness of ASL itself, this solution was built as a stepping stone towards more accessible and inclusive communication technology.

The workflow follows a three-stage pipeline for training (stage 1) and real-time inference (stage 2 and 3). 

In Addition to traditional end-to-end CNN methods that process raw images directly, this system also explores **landmark-based preprocessing** using Google's MediaPipe framework to isolate hand regions with high precision before classification.

#### a) Dataset : `asl_dataset`

The training dataset comprises **7,200 RGB images** with balanced distribution across all classes:
- **36 classes:** 26 letters (A-Z) + 10 digits (0-9)
- **200 samples per class** ensuring equal representation
- **dataset split:** 80% training (5,760 images), 10% each (720 images) for validation and testing

The dataset was compiled from publicly available ASL image collections and manually curated for quality and consistency.

_**Data challenges**_: there are some issues with the collected dataset:

- **visually similar gestures:** pairs like (M, N, S, T) or (V, 2) can cause misclassification
- **background noise:** inconsistent lighting, varying skin tones, motion blur
- **framing variability:** different hand sizes, positions, and orientations

These challenges, however, motivated the mentioned-above mediaPipe-based preprocessing approach to reduce background content and standardize hand positioning.

#### b) Three-Stage Pipeline

#### 🥚 **Stage 1: Training**

This phase explores three different models:
- CNN trained directly on raw images
- SVM using HOG features
- CNN combined with mediapipe-based hand localization

The initial dataset `asl_dataset` was used to train with our first two experimental models: CNN [`asl_cnn_training.ipynb`], and SVM [`asl_svm_training.ipynb`].Despite achieving reasonable accuracy, both showed signs of overfitting. 
- CNN : test accuracy of 88.47% but with moderate overfitting gap of 6.86%
- SVM : test accuracy of 84.17% but with severe overfitting gap of 15.15%

To address this issue, the `asl_dataset` was reannotated using MediaPipe framework (pls refer to `dataset_reannotate.py`). For each image, the pipeline: 

1. detects 21 hand anatomical keypoints in normalized [0,1] coordinate space
2. calculates an axis-aligned bounding box with 15% padding on each side
3. converts coordinates to pixel space
4. crops and extracts the hand region
5. resizes it to 64×64 pixels
6. converts the image back to BGR for OpenCV file saving
7. finally, saves to a new dataset `asl_dataset_mediapipe`

_Note: if you have a much larger dataset with high-quality images, do not `use_fallback` as it will save the original images even when mediapipe fails, which would only add more noise to what should be a well-reannotated dataset._

The same CNN architecture was then retrained on this new dataset [`asl_cnn_mediapipe_training.ipynb`], achieving _97.92% test accuracy_  with validation accuracy (98.47%) exceeding training (97.90%), which is a sign of good generalization.

**Final model:** `asl_final_cnn_mediapipe_model.keras` is used in the next two phases.


#### 🐣 **Stage 2: Real-Time Recognition**

This phase implements live webcam inference with stability filtering for robust predictions.

The process starts by capturing webcam input at 640×480 resolution at 30 FPS via OpenCV's videocapture interface. Each frame is converted from BGR to RGB for MediaPipe compatibility. Detected hand landmarks are used to compute a bounding box (identical to the preprocessing pipeline in stage 1), and the cropped hand image is fed into the trained CNN model.

Raw CNN outputs (softmax probabilities) undergo a two-layer stability filtering system: (1) confidence threshold of 0.5, and (2) a temporal voting mechanism (that requires the same prediction to appear ≥2 times within the last 3 frames).

Only stable predictions are visualized, along with the detected landmarks, bbox, predicted class, and confidence score.


#### 🐥 **Stage 3: Interactive Application**

This stage is built on top of the previous one, but includes some additional steps to accumulate stable predictions into a sequential text buffer, representing the user’s intended messages.

_**Text Buffer Features:**_
- automatic letter accumulation from stable predictions
- duplicate prevention (rejects identical letters within 0.5 seconds)
- word stats: word counts, letter counts, and last added character
- optional saving the words as a file

#### c) CNN Architecture

The model inputs 64×64×3 RGB images.

The CNN architecture consists of 3 convolutional blocks with progressively increasing feature depths: 
   - Block 1 has 2 Conv2D layers (32 filters each)
   - Block 2 has 2 Conv2D layers (64 filters each), and 
   - Block 3 has 1 Conv2D layer (128 filters). 

Each block is followed by 2×2 max pooling and dropout (0.25 for blocks 1-2, 0.4 for block 3). After flattening, 2 fully connected layers (256 & 128 neurons) with 0.5 dropout between them refine features before the final 36-class softmax output layer.


#### d) Usage

Please install all required packages (including key dependencies like Tensorflow, Keras, MediaPipe, and OpenCV) as listed in `requirements.txt` within the "Side file" folder. 

GPU is recommended for training in stage 1, and CPU alone is sufficient to run stage 2 and 3.

_**Note:** whether you would like to re-train the model or use the trained one for real time inference, please kindly ensure that the file structure/path is correct._

#### e) Contributing

Your contributions are always welcome!

Should you have any suggestions for improving model performance and enhancing classification accuracy -- whether by adding new features, experimenting with altenative hand detection method or testing different CNN configs or mechanism -- please share them by opening an issue or a pull request. 


#### f) License

This project is licensed under the **MIT License**.

#### g) Acknowledgments

- Google MediaPipe for robust hand landmark detection framework
- TensorFlow/Keras for deep learning infrastructure
- OpenCV for computer vision tools
- ASL Community for inspiring accessible communication technology
- Open-source dataset contributors for making ASL research possible

Thank you!

🍔