# Chamanti OCR చామంతి

# Mission
This project aims to build an ambitious state-of-the-art OCR framework, that should work on any language.
It will not rely on segmentation algorithms (at the glyph level), making it ideal for highly
agglutinative scripts like Arabic, Devanagari etc. We will be starting with Telugu however.
We use the technology of `Convolutional Recurrent Neural Networks` from `Keras` in `TensorFlow 2.0`.
`CRNN` with `CTC` (Connectionist Temporal Classification) loss function is the main work-horse.

# Dependencies
1. tensorflow
1. [Lekhaka](https://github.com/rakeshvar/Lekhaka) - My 'scribing' package for generating complex text, including Indian languages like Telugu, on the fly

# Setup
1. Install TensorFlow
1. Download IndicScribe and place in a parallel dicrectory

# Files
1. `model.py` The TensorFlow CRNN model with CTC loss
1. `train.py` Main file to run
1. `utils.py` Utilities to print images and Probabilities to terminal, etc.

## Training the CRNN
You can now train a CRNN to read Telugu text! 
```sh
python3 train.py
```
