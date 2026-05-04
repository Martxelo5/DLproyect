# Image Classification with AlexNet (CIFAR-10) and Transfer Learning (EuroSAT)

1. **CIFAR‑10** – classifying 32×32 natural images using a custom AlexNet with hyperparameter optimisation.
2. **EuroSAT** – classifying 64×64 satellite images using transfer learning with state‑of‑the‑art architectures.

Both parts are implemented in a single Jupyter Notebook (`main.ipynb`) using PyTorch, with thorough exploration, training, evaluation, and explainability.

---

## 📂 Repository Structure

```
├── main.ipynb                 # Main notebook with all code and documentation
├── models/
│   ├── AlexNet.py             # AlexNet adapted for CIFAR-10 (Part 1)
│   └── AlexNetEuroSAT.py      # AlexNet adapted for EuroSAT (Part 2)
├── saved/                     # (not tracked, download from Drive) trained models
├── data/                      # CIFAR-10 data (not included, automatically downloaded)
├── README.md
└── requirements.txt           # Python dependencies
```

---

## 🔗 Pre‑trained Model Weights

Due to GitHub’s file size limits, the trained `.pth` files are **not** included in this repository.  
You can download them from Google Drive:

📎 **[Download model weights (Google Drive)]([https://drive.google.com/your-shared-link-here](https://drive.google.com/drive/folders/10wIzrx6zCwm6DvhvDYEaTfSbs1wkddWL?usp=sharing))**

Place the downloaded files inside the `saved/` folder before running the notebook:
```
saved/
├── best_alexnet_cifar_2.pth  
├── alexnet_scratch_eurosat.pth
├── vgg16_eurosat.pth
├── resnet50_eurosat.pth
└── effnetv2_eurosat.pth

```

---
