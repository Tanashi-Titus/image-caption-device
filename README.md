# 📸 Image Caption Device

## 🚀 Introduction

**Image Caption Device** is an AI system that automatically generates natural language descriptions (captions) for input images.

This project combines:

* **Computer Vision** → to extract visual features from images
* **Natural Language Processing (NLP)** → to generate meaningful textual descriptions

👉 **Goal:**

* Build an end-to-end pipeline from image → caption
* Optimize for deployment on lightweight or edge devices

---

## 🧠 System Architecture

The model follows an **Encoder–Decoder architecture**:

```
Image → CNN Encoder → Feature Vector → Decoder (LSTM / Transformer) → Caption
```

### 🔹 Main Components

1. **Encoder (Vision Model)**

   * Pretrained CNN (ViT)
   * Extracts high-level image features

2. **Decoder (Language Model)**

   * Transformer-based model
   * Generates captions token by token

3. **Training Objective**

   * Cross-Entropy Loss
   * Teacher Forcing

---

## 📊 Dataset

Supported datasets:
* Flickr8k 

### Example Format:

```json
{
  "image_1.jpg": ["a dog running on grass", "a brown dog playing outside"],
  "image_2.jpg": ["a man riding a bike"]
}
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Tanashi-Titus/image-caption-device.git
cd image-caption-device

pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
python train.py
```

### Key Hyperparameters:

* `batch_size`
* `learning_rate`
* `embedding_dim`
* `hidden_dim`
* `num_epochs`

---

## 🔍 Inference

Generate a caption for a new image:

```bash
python inference.py --image path/to/image.jpg
```

### Example Output:

```
"a man riding a bicycle on the street"
```

---

## 📈 Evaluation

Common evaluation metrics:

* BLEU
* ROUGE

---

## 🧪 Demo

| Input Image | Output Caption              |
| ----------- | --------------------------- |
| 🖼️         | "a dog playing in the park" |

---

## 🛠️ Technologies Used

* Python
* PyTorch / TensorFlow
* OpenCV
* NumPy
* NLP Tokenizers

---

## 🔥 Key Features

* End-to-end pipeline (training → inference)
* Modular and extensible architecture
* Suitable for edge deployment

---

## 📌 Roadmap

* [ ] Improve caption quality with Attention mechanisms
* [ ] Upgrade to Transformer-based models (ViT + GPT-like decoder)
* [ ] Deploy API (FastAPI / Flask)
* [ ] Optimize for real-time inference on edge devices

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 👨‍💻 Author

**Titus (Tanashi)**
AI Engineer (Computer Vision + NLP)

---
