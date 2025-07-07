# 🧠 Bone Age Prediction from Hand X-Rays using Deep Learning

This project applies deep learning models to perform **regression** on pediatric hand X-ray images for **age prediction**. The task is crucial in pediatric medicine to assess skeletal maturity in an automatic, scalable, and objective way.

We evaluate three convolutional neural network (CNN) architectures:

- **Inception-v4**
- **ResNet50**
- **SqueezeNet v1.1**

The models were implemented in TensorFlow / Keras framework.

---

## 📁 Project Structure


```text
bone-age-prediction-from-xray
├── models/                   
│   ├── inception-f.ipynb   # Inception implementation
│   ├── resnet-f_.ipynb     # ResNet implementation
│   └── squeezenet-f.ipynb  # SqueezeNet implementation
│
├── report/
│   └── HDA_project.pdf     # Project report
│
├── images/                  
│
├── data/                    
│   └── .gitignore            
├── README.md                
├── requirements.txt         # Python dependencies
└── LICENSE                  # Licenza del progetto
```
---

## 📊 Dataset

- Dataset source: Stanford Medicine Box (open access at: https://www.rsna.org/rsnai/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017)
- Total images: **14,236**
- Split:
  - **Train**: 12,611
  - **Validation**: 1,425
  - **Test**: 200
- Target: Patient's age in **months**
- Format: Each image is associated with a CSV row containing image ID and age.

*Note*: Gender information was not used in the regression task.

---

## 🧹 Preprocessing Pipeline

1. **Resizing** to 256x256
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for enhanced contrast
3. **Normalization** to [0, 1]
4. **Random Cropping**
5. **Channel reduction** (RGB → grayscale)


---

## 📈 Results (Test Set)

| Model         | MAE (months) | MSE   | Size     |
|---------------|--------------|-------|----------|
| SqueezeNet v1.1 | **13.11**     | 270.83 | 5.14 MB  |
| ResNet50      | **11.94**     | 219.53 | 90.08 MB |
| Inception-v4  | **12.31**     | 228.37 | 157.31 MB |

🟢 **ResNet50** achieved the best MAE.  
🟡 **SqueezeNet** offers an excellent trade-off between size and performance.

---

