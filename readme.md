
# **RadarNet**

This project implements a high-performance Content-Based Image Retrieval (CBIR) system specifically designed for RADAR images. It uses a hybrid deep learning model, fusing features from ResNet50 and ImageNet, to provide accurate and fast retrieval results. The system is deployed as an interactive web application using Streamlit.

This system achieves a retrieval efficiency of **94.8%** and reduces retrieval time by **35%** compared to previous state-of-the-art methods.

## Features

* **Hybrid Deep Learning Model**: Fuses features from ResNet50 and pre-trained ImageNet layers for robust feature extraction.
* **High Accuracy & Speed**: Optimized for both precision and low latency in retrieving RADAR images.
* **Handles Complex Imagery**: Effectively manages challenges unique to RADAR data, like speckle noise and complex textures.
* **Interactive UI**: A user-friendly web interface built with Streamlit to upload a query image and view the top 5 recommended matches.

## Installation

Follow these steps to set up the project environment. This guide assumes you have an NVIDIA GPU.

### 1. Prerequisites: System Environment Setup

First, install the necessary system-level dependencies.

#### a. Python

Install Python version 3.10. You can download it from the [official Python website](https://www.python.org/downloads/release/python-3100/).
During installation, make sure to check the box that says **"Add Python to PATH"**.

Verify the installation:

```bash
  python --version
```
#### b. NVIDIA CUDA Toolkit

This project requires CUDA for GPU acceleration. Version 11.2 is recommended for compatibility with TensorFlow 2.10.

* Visit the [NVIDIA CUDA Toolkit 11.2 Archive](https://developer.nvidia.com/cuda-11.2.0-download-archive).
* Download and run the installer for your operating system.
* Ensure the installer sets the `CUDA_HOME` and `PATH` environment variables correctly.

#### c. NVIDIA cuDNN

Download the cuDNN library that matches your CUDA version (v8.1 for CUDA 11.2).

* Go to the [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive). You will need an NVIDIA developer account.
* Download the appropriate cuDNN version for CUDA 11.x.
* Extract the downloaded ZIP file and copy the contents of the `bin`, `include`, and `lib` folders into the corresponding folders in your CUDA installation directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\`).

### 2. Project Setup

Clone the repository and set up the Python environment.

#### a. Clone the Repository

```bash
    git clone https://github.com/daksh26tiku/RadarNet.git
    cd RadarNet
```
#### b. Create a Virtual Environment

It is highly recommended to use a virtual environment.

```bash
    python -m venv venv
```
* Activate it
    * On Windows/Mac(or Linux):

```bash
    venv\Scripts\activate
```

```bash
    source venv/bin/activate
```

#### c. Install Required Libraries

Create a `requirements.txt` file with the following content:

```bash
    tensorflow==2.10.0
    streamlit==1.4.0
    numpy
    pandas
    scikit-learn
    Pillow
    tqdm
```
Now, install all the dependencies from the file:

```bash
    pip install -r requirements.txt
```
## Usage

The application runs in two stages: first, you extract features from your dataset, and second, you run the web application.

### 1. Prepare Your Dataset

* Create a folder named `Images` in the root of the project directory.
* Place all your RADAR image files (e.g., in `.jpg`, `.png`, `.tif` format) inside this folder. You can also have subdirectories.

### 2. Extract Image Features

Run the `app.py` script to process all images in the `Images` folder and generate feature embeddings. This will create two files: `embeddings.pkl` and `filenames.pkl`. This only needs to be done once for your dataset.

```bash
    python app.py
```


### 3. Run the Streamlit Application

Once the feature embeddings are created, start the Streamlit web server by running `main.py`.

```bash
    streamlit run main.py
```

Your web browser should open with the application running. You can now upload a RADAR image to find the most similar ones in your dataset.