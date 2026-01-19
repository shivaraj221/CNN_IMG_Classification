# **CIFAR-10 Image Classification Using CNN From Scratch**

---

## 📌 Project Overview

This project implements an **image classification system using a Convolutional Neural Network (CNN) built completely from scratch** with TensorFlow and Keras.

The model is trained on the **CIFAR-10 dataset**, a standard benchmark dataset in computer vision containing small RGB images from 10 object categories.

The objective of this project is to understand how CNNs learn visual features such as edges, textures, and shapes, and how they outperform traditional neural networks for image-based tasks.

---

## 🎯 Objectives

* Load and explore the CIFAR-10 dataset
* Preprocess and normalize image data
* Build a CNN architecture from scratch
* Train the model on image data
* Evaluate performance on unseen test images
* Predict object classes from new inputs

---

## 🗂 Dataset Information

**CIFAR-10 Dataset**

* Total images: **60,000**
* Training images: **50,000**
* Testing images: **10,000**
* Image size: **32 × 32 × 3 (RGB)**
* Number of classes: **10**

### Class Labels

```
airplane   automobile   bird   cat   deer
dog        frog         horse  ship  truck
```

---

## ⚙️ Technologies Used

* Python 3.x
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn

---

## 🔄 Project Workflow

1. Import required libraries
2. Load CIFAR-10 dataset
3. Analyze image shapes and labels
4. Convert labels into 1D format
5. Visualize training samples
6. Normalize pixel values (0–255 → 0–1)
7. Build CNN model from scratch
8. Train the CNN model
9. Evaluate accuracy on test data
10. Perform predictions

---

## 🧪 Data Preprocessing

### Label Reshaping

Original label format:

```
(50000, 1)
```

Converted to:

```
(50000,)
```

Required for sparse categorical cross-entropy.

---

### Image Normalization

Pixel values range from **0 to 255**.

They are normalized to **0–1** using:

```python
X_train = X_train / 255.0
X_test  = X_test / 255.0
```

This improves training speed and model stability.

---

## 🧠 CNN Architecture (From Scratch)

```
Input Image (32 × 32 × 3)

↓ Conv2D (32 filters, 3×3, ReLU)
↓ MaxPooling (2×2)

↓ Conv2D (64 filters, 3×3, ReLU)
↓ MaxPooling (2×2)

↓ Flatten
↓ Dense (64 neurons, ReLU)

↓ Dense (10 neurons, Softmax)
```

---

## 🏋️ Training Configuration

* Optimizer: **Adam**
* Loss Function: **Sparse Categorical Cross-Entropy**
* Epochs: **10**
* Batch Size: **32**

---

## 📊 Model Performance

| Metric            | Result   |
| ----------------- | -------- |
| Training Accuracy | ~78%     |
| Test Accuracy     | **~70%** |

The model generalizes well on unseen images and produces stable predictions across all classes.

---

## 🔍 Sample Prediction

```
Actual Class    : Ship
Predicted Class : Ship
```

Predictions are obtained using:

```python
np.argmax(cnn.predict(image))
```

---

## ✅ Key Highlights

* CNN built completely from scratch
* No pretrained models used
* No transfer learning
* End-to-end training
* Clear improvement over traditional methods
* Industry-standard dataset

---

## ⚠️ Limitations

* No data augmentation
* No dropout regularization
* No batch normalization
* Shallow network architecture

---

## 🚀 Future Improvements

* Add data augmentation
* Use deeper CNN layers
* Apply batch normalization
* Add dropout to reduce overfitting
* Train for more epochs
* Use pretrained architectures for comparison

---

## ▶️ How to Run

### Install Dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Run the Program

```bash
python main.py
```

---

## 📌 Conclusion

This project demonstrates the practical implementation of a **Convolutional Neural Network from scratch** for image classification.

By training directly on raw image data, the CNN successfully learns meaningful visual patterns and achieves approximately **70% accuracy on the CIFAR-10 dataset**.

This project serves as a strong foundation for understanding deep learning and computer vision concepts.


