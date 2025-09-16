# Image Captioning with CNN + RNN

This project implements an **Image Captioning system** using a **pretrained CNN (ResNet50) encoder** and an **LSTM-based RNN decoder**. It takes an input image and generates a descriptive textual caption.

---

## Features

- **Pretrained ResNet50 encoder** for robust image feature extraction.
- **LSTM decoder** with:
  - Initial hidden and cell states projected from image features.
  - Beam search and greedy decoding for caption generation.
- **Flexible vocabulary handling** using `vocab.pkl`.
- **Training and inference pipelines** fully implemented in PyTorch.
- Supports both **CPU** and **GPU** inference.

---

## Project Structure

```

ImageCaptioning/
├─ checkpoints/          # Saved model weights and vocab
│  ├─ best\_encoder.pth
│  ├─ best\_decoder.pth
│  └─ vocab.pkl
├─ flickr8k/images/      # Example images (Flickr8k dataset)
├─ model.py              # EncoderCNN and DecoderRNN models
├─ training.py           # Training pipeline
├─ inference.py          # Generate captions for new images
├─ utils.py              # Helper functions (dataset, preprocessing)
└─ README.md

````

---

## Installation

1. Clone the repository:

```bash
git clone git@github.com:Sahahahil/Captioning-Image.git
cd ImageCaptioning
````

2. Install dependencies:

```bash
pip install torch torchvision pillow numpy
```

3. Optional: If using GPU, make sure PyTorch is installed with CUDA support.

---

## Usage

### Training

Edit `training.py` to set your dataset paths, batch size, learning rate, etc. Then run:

```bash
python training.py
```

This will:

* Train the encoder and decoder models.
* Save the best model checkpoints in `checkpoints/`.
* Save the vocabulary as `vocab.pkl`.

---

### Inference

Update `inference.py` with:

* `image_path` → Path to your input image.
* `checkpoint_dir` → Directory where model checkpoints are stored.

Run the script:

```bash
python inference.py
```

You will see:

```
🖼️ Image: example.jpg
📝 Generated Caption: A dog is running in the park
```

**Decoding Options:**

* `use_beam=True` → Beam search decoding.
* `use_beam=False` → Greedy decoding.

---

## Dataset

* Designed for **Flickr8k** dataset.
* Place images in `flickr8k/images/` and captions in a compatible format for training.

---

## Notes

* The decoder expects the **vocabulary** used during training. Make sure `vocab.pkl` matches the trained model.
* Model weights for encoder and decoder are required for inference.
* GPU is recommended for faster training and inference.

---

## Future Improvements

* Switch to **Transformer-based decoder** for improved caption quality.
* Fine-tune the ResNet encoder on your dataset.
* Add **attention mechanism** for better image-context alignment.
* Support **larger datasets** like Flickr30k or COCO.

---
