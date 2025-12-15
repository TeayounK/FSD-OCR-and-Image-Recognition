# FSD-OCR-and-Image-Recognition
Developing text-based image recognition specialized in genetic records

## Generating Synthetic images
Before running `synthetic_data.py`, please install `imgaug`.
`pip install pillow numpy opencv-python imgaug`


# Dount: Text-Based Image Processing Model

Dount is a deep learning model designed to process and understand text in images without relying on traditional OCR (Optical Character Recognition). The model focuses on detecting and interpreting textual information in images by leveraging contextual understanding, attention mechanisms, and end-to-end vision-language models. Unlike conventional OCR techniques, Dount aims to process text in images directly without extracting individual characters, providing a more holistic approach to text-based image processing. But in this case, we need OCR labels to fine tune the model for specific type of images.

## Features

Text Detection: Detects areas of text within images, identifying text regions without extracting individual characters.

Contextual Understanding: Understands the context and relationships between text and visual content, without relying on explicit character recognition.

End-to-End Processing: Processes both text and image features in an integrated manner using deep learning models like transformers, allowing for rich semantic understanding.

No OCR Required: Unlike traditional OCR systems, Dount avoids character-level recognition, focusing instead on understanding text in images in context.


## Model Architecture

The Dount model leverages a combination of advanced deep learning techniques for text detection and contextual understanding:

Text Detection: The model uses a convolutional neural network (CNN) or transformer-based architecture to detect text regions in images. These regions are not processed for character recognition, but rather are understood as text blocks within the visual context.

Contextual Representation: Dount processes both text and image data in a joint learning setup. Using a Vision Transformer (ViT) or multimodal transformer architecture, the model learns to understand relationships between text and other objects or scenes in the image. This enables it to process images without needing traditional OCR-based character extraction.

End-to-End Vision-Language Model: The model integrates visual and textual features through attention mechanisms that allow for direct interaction between the text (in image form) and the overall scene. This allows Dount to process text content as part of the broader image context.


Document Scanning: Process scanned documents or receipts and extract contextual meaning (e.g., identifying categories like "price," "address," "product type") without traditional OCR methods.
