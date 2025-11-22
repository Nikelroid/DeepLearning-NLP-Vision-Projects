

# Deep Learning Assignments: NLP & Computer Vision

![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.26%2B-yellow?logo=huggingface&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-0.14%2B-orange?logo=pytorch&logoColor=white)
![Hazm](https://img.shields.io/badge/Hazm-NLP-green)


## Description
This repository contains a collection of three deep learning projects developed as part of an advanced assignment series. The projects utilize **PyTorch** to solve distinct challenges in Natural Language Processing (NLP) and Computer Vision. The focus ranges from generating Persian poetry using Recurrent Neural Networks to captioning images using Encoder-Decoder architectures, and finally, classifying literary styles using modern Transformers (BERT).

The code is structured into three Jupyter Notebooks, designed to be run in environments like Google Colab or a local Jupyter server with GPU support.

## Projects Overview

### 1. Persian Poem Generation (RNNs)
* **File:** `DL2022-HW4-P1.ipynb`
* **Goal:** Train generative models to compose Persian poems in the style of Ferdousi.
* **Architecture:** Implements and compares two Recurrent Neural Network architectures: **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit).
* **Key Features:**
    * Text preprocessing using `Hazm` (normalization, tokenization).
    * N-gram sequence creation for training.
    * Embedding layer followed by RNN layers and a fully connected output.
    * Loss and Accuracy visualization over epochs.
    * **Inference:** A generation loop that predicts the next word based on a seed phrase.

### 2. Image Captioning (CNN + RNN)
* **File:** `DL2022_HW4_P2.ipynb`
* **Goal:** Generate descriptive captions for input images automatically.
* **Dataset:** COCO 2014 Dataset (Train/Test).
* **Architecture:** An Encoder-Decoder model.
    * **Encoder:** Uses a pre-trained **ResNet50** (with the final classification layer removed) to extract feature vectors from images.
    * **Decoder:** Uses an **LSTM** network to translate feature vectors into natural language captions.
* **Key Features:**
    * Custom Dataset class for loading COCO images and captions.
    * Image preprocessing (Resize, Crop, Normalize).
    * Vocabulary building using `nltk`.
    * A `captionize` function to generate text for any given input image.

### 3. Poet Classification (Transformers/BERT)
* **File:** `DL2022_HW4_P3.ipynb`
* **Goal:** Classify Persian poems by identifying the poet (10 classes: Jami, Moulavi, Bahar, etc.).
* **Architecture:** Fine-tuning **ParsBERT** (`HooshvareLab/bert-fa-base-uncased`).
* **Key Features:**
    * Data handling using `pandas` and stratification for unbalanced classes.
    * Tokenization using `BertTokenizer`.
    * Comparison of optimization strategies: **AdamW** vs. **SGD**.
    * Evaluation metrics: F1 Score, Accuracy, Confusion Matrix, and **Perplexity**.
    * Layer freezing/unfreezing strategies for fine-tuning.

## Installation

To run these projects locally, ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nikelroid/deeplearning_hw4.git](https://github.com/nikelroid/deeplearning_hw4.git)
    cd deeplearning_hw4
    ```

2.  **Install dependencies:**
    The projects rely on several specific libraries. You can install them via pip:
    ```bash
    pip install torch torchvision torchtext
    pip install transformers numpy pandas matplotlib
    pip install nltk pycocotools
    pip install hazm plotly scikit-learn
    ```

    *Note: For `pycocotools`, you might need a C++ compiler installed on your system.*

## Usage

These projects are provided as Jupyter Notebooks.

1.  **Start Jupyter Lab or Notebook:**
    ```bash
    jupyter lab
    ```
2.  **Open a specific notebook:**
    * Open `DL2022-HW4-P1.ipynb` for text generation.
    * Open `DL2022_HW4_P2.ipynb` for image captioning.
    * Open `DL2022_HW4_P3.ipynb` for BERT classification.

3.  **Data Setup:**
    * **P1:** Ensure `ferdousi.txt` is available in the working directory.
    * **P2:** The notebook attempts to download COCO data. If you have it locally, map the paths in the `get_loader` function.
    * **P3:** The notebook clones the `Persian_poems_corpus` from GitHub automatically.

4.  **Google Colab:**
    If running on Colab, ensure you mount your Google Drive if you wish to save checkpoints, as indicated in the notebook headers:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

## Contributing

Contributions are welcome! If you find a bug or want to improve the model architectures:

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

## Contact

For questions regarding the code or the models used:
* **Maintainer:** Nikelroid
* **GitHub:** [https://github.com/nikelroid](https://github.com/nikelroid)


## Materials
Models of the question are in the drive folder: [GOOGLE DRIVE LINK](https://drive.google.com/drive/folders/1v5fVXVsjR9cD7hVL0kK69N68E-Traw5Q?usp=share_link) </br>
Data For Train HW4-P2 : [Train.zip](https://drive.google.com/file/d/1-CAe_tjXUA-lzlPDuaxyZXcG22yayaip/view?usp=share_link)
