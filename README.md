# AS-PCI
Analytic system for parent-child interactions based on nonverbal communication using computer vision and machine learning in family behavior therapy

## Abstract
Parent-child interactions (PCIs) are critical for the development of a child across various domains, including language, social-emotional, cognitive, and emotional growth. In the context of family behavioral therapy, nonverbal communication (NVC) between parents and children is essential for evaluating the quality of PCIs. This study introduces a novel system that employs computer vision and deep learning techniques to analyze NVC-based PCI behaviors. The proposed system comprises three core modules: video stream input and preprocessing, NVC context extraction, and NVC-based PCI behavior analysis. The proposed system detects and tracks parents and children using multi-camera inputs, extracts NVC contexts such as distance and gaze, and classifies interactions based on these contexts. The key contributions of this study include the design of the system, automated extraction of NVC contexts, and analysis of the interaction categories. This approach facilitates a more objective and quantitative evaluation of PCI, offering valuable insights for enhancing the quality of PCIs. We validated the feasibility and applicability of the proposed analytical system for NVC context-based PCI behavior by implementing a prototype solution and applying it to actual video data obtained from our testbed.

This repo contains demo code for implementing the system. The original dataset is not publicly available due to subject privacy concerns, but the extracted NVC contexts can be used to implement it.

## System Requirements
- **Operating System**: Ubuntu 22.04.3 LTS
- **CPU**: 12th Gen Intel(R) Core(TM) i7-12700F
- **GPU**: NVIDIA GeForce RTX 4070 (12GB VRAM)
- **System Memory (RAM)**: 32GB

## Pre-trained Models
We trained a YOLOv8-based Head Detector and an PC Detector model to detect Parents and Children using data built from a private dataset.

The pre-trained model weights can be found in [here](https://github.com/Kim-Byeong-Hun/AS-PCI/tree/main/demo/weights), and the weights of the pre-trained RepVGG model can be downloaded [here](https://drive.google.com/drive/folders/1Du7GPb3Xf2eb5ZbWnXhbSFQxC1B3K7fG?usp=sharing).

## How to run it

### Preparation
First, clone the repository
```bash
git clone https://github.com/Kim-Byeong-Hun/AS-PCI.git
```
Then download the required packages.
```bash
pip install -r requirements.txt
```

### NVC Context Extraction Process
Upload the video to /demo/videos, and run the following to extract the NVC context from therecorded footage
```bash
# Front-angle video
python /demo/main_Front.py --video_name 240712_007_C2

# Video from bird's eyes view
python /demo/main_Top.py --video_name 240712_007_C7
```
The extracted text file and the resulting video are stored in demo/outputs/, where the data extracted in this paper is already stored.

### PCI analysis process
The PCI analysis is done via /demo/k_means.ipynb, and the code includes both preprocessing and analysis.

## Reference
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [6DoFHPE](https://github.com/Redhwan-A/6DoFHPE)
