# Eardrum Image Classification Using Deep Learning

This project focuses on designing and implementing an image classification system for eardrum conditions using various Deep Learning algorithms. The goal is to accurately classify eardrum images into categories such as normal, middle ear effusion, and tympanostomy tube conditions, providing a valuable tool for timely diagnosis.

---

## 👂 Project Objective

The primary objective of this project is to:
* Design an image classification system for eardrum images.
* Classify images into three conditions: **normal**, **middle ear effusion**, and **tympanostomy tube**.
* Compare the performance (accuracy and F1 score) of different Convolutional Neural Network (CNN) algorithms, specifically **Basic CNN**, **ResNet-18**, and **AlexNet**.

---

## 📝 Introduction

Acute infections of the middle ear are among the most commonly treated childhood diseases. Timely and accurate diagnosis is crucial, as complications can affect children's language learning and cognitive processes. Middle ear diseases are typically diagnosed using a patient's history and otoscopic findings of the tympanic membrane (TM).

Artificial intelligence, particularly deep learning, is increasingly employed for high-level image analysis tasks such as classification, segmentation, and matching in medical imaging. Convolutional Neural Networks (CNNs) have shown promising performance in the automatic classification of various medical images, including skin lesions, eye lesions, and radiological images. This project leverages the power of CNNs to automate the classification of eardrum images.

---

## 💡 Proposed System Overview

The proposed system is an image classification pipeline designed to categorize eardrum images. It involves:
* **Image Input**: Eardrum images are fed into the system.
* **Feature Extraction**: Deep learning models (CNN, ResNet-18, AlexNet) are used to automatically extract relevant features from the images.
* **Classification**: Based on the extracted features, the system classifies the eardrum image into one of three conditions: Normal, Middle Ear Effusion, or Tympanostomy Tube.
* **Performance Comparison**: The system evaluates and compares the accuracy and F1 score of the different algorithms to determine the most effective model for this task.

*(Note: Visual representations of the proposed system, including input/output examples and general system flow, would be included here if image assets were available.)*

---

## 🤖 Algorithms Used

This project explores and compares the performance of three prominent Convolutional Neural Network architectures for eardrum image classification:

* **Basic Convolutional Neural Network (CNN)**: A foundational deep learning model for image analysis, consisting of convolutional, pooling, and fully connected layers.
    * *(A block diagram of the Basic CNN, showing input, convolution, pooling, and fully connected layers leading to a Softmax output, would be placed here.)*

* **AlexNet**: A pioneering deep CNN architecture known for its performance in image recognition tasks, featuring multiple convolutional and fully connected layers.
    * *(A block diagram of AlexNet, detailing its convolutional and pooling stages, followed by fully connected layers and Softmax output, would be placed here.)*

* **Residual Neural Network (ResNet-18)**: A variant of ResNet, which utilizes residual connections to enable the training of much deeper networks, combating the vanishing gradient problem.
    * *(A block diagram of ResNet-18, illustrating its residual blocks and overall architecture, would be placed here.)*

---

## 📊 Dataset

A database comprising **454 labeled eardrum images** is used to train and test the system. The dataset is distributed as follows:
* **179 Normal** eardrum images
* **179 Effusion** (middle ear effusion) images
* **96 Tube** (tympanostomy tube) images

The dataset used is based on the "OtoMatch: Content-based Eardrum Image Retrieval using Deep Learning" research.

---

## ⚙️ Training and Testing Process

The project involves a rigorous training and testing process for each of the selected algorithms:

* **Training**: Each CNN model (Basic CNN, ResNet-18, AlexNet) is trained on the labeled eardrum image dataset. The training progress, including accuracy and loss curves over epochs, is monitored and analyzed for each algorithm.
    * *(Graphs showing training progress (Accuracy vs. Iteration, Loss vs. Iteration) for CNN, ResNet-18, and AlexNet would be included here if image assets were available.)*

* **Testing**: After training, the models are evaluated on unseen eardrum images to assess their classification performance. The testing process involves:
    * Inputting an image.
    * Resizing the image for model compatibility.
    * Obtaining the classification output (Effusion, Normal, or Tube).
    * *(Screenshots demonstrating the input image, resized image, and the classification output for Effusion, Normal, and Tube cases would be included here if image assets were available.)*

---

## 📈 Evaluation Metrics

The performance of the image classification system is rigorously evaluated using standard metrics derived from the **Confusion Matrix**.

### Confusion Matrix Explained
A Confusion Matrix is a table used to describe the performance of a classification model on a set of test data for which the true values are known.

* **Rows**: Correspond to the **predicted class** (Output Class).
* **Columns**: Correspond to the **true class** (Target Class).
* **Diagonal Cells**: Represent observations that are **correctly classified**.
* **Off-diagonal Cells**: Represent **incorrectly classified** observations.
* Each cell shows both the number of observations and the percentage of the total number of observations.
* The far right column typically shows **Precision**.
* The bottom row typically shows **Recall**.
* The bottom-right cell shows the **overall accuracy**.

*(Confusion matrices for CNN, ResNet-18, and AlexNet would be included here if image assets were available.)*

### Key Metrics
The four fundamental metrics derived from the Confusion Matrix are:

* **True Negative (TN)**: The model predicted 'No', and the actual value was also 'No'.
* **True Positive (TP)**: The model predicted 'Yes', and the actual value was also 'Yes'.
* **False Negative (FN)**: The model predicted 'No', but the actual value was 'Yes' (also known as Type-II error).
* **False Positive (FP)**: The model predicted 'Yes', but the actual value was 'No' (also known as Type-I error).

From these, the following evaluation parameters are calculated:

* **Accuracy**: $$(TP+TN)/(TP+TN+FP+FN)$$
* **Error Rate**: $$(FP+FN)/(TP+TN+FP+FN)$$
* **Precision**: $$TP/(TP+FP)$$
* **Recall**: $$(TP)/(TP+FN)$$
* **F-Measure (F1 Score)**: $$2* (Recall*Precision)/(Recall+Precision)$$

---

## 📊 Result Analysis & Conclusion

Since the F1 score is often considered a more robust measure for evaluating classification models, especially when dealing with imbalanced datasets or when both precision and recall are important, it was a key performance evaluation metric in this proposed system.

Based on the analysis of various algorithms, it was concluded that the **AlexNet algorithm performed the best** in terms of both F1 score and accuracy when compared to the Basic CNN and ResNet-18 algorithms. Therefore, AlexNet is the recommended algorithm for this eardrum image classification system.

*(A graph comparing Accuracy vs. F1 Score for all algorithms would be included here if image assets were available.)*

---

## 📚 References

* **Base Paper**:
    * OtoMatch: Content-based eardrum image retrieval using deep learning (plos.org)
    * OtoMatch: Content-based eardrum image retrieval using deep learning - PubMed (nih.gov)

* **Dataset**:
    * OtoMatch: Content-based Eardrum Image Retrieval using Deep Learning | Zenodo

---

## ✨ Acknowledgements

This project was developed by the following batch members from Panimalar Engineering College, Department of Electronics and Instrumentation Engineering:

* DEBORAH DEVA KIRUBAI.M (211418107019)
* JASMINE .C (211418107039)
* RESHMA .V (211418107076)

Under the guidance of:
* Mr. V.VASUDEVAN, M.E.,
    Assistant Professor,
    Department of EIE,
    Panimalar Engineering College.

---

## 🤝 Contributing

We welcome any contributions to this project! If you have suggestions for improvements, new features, or find any issues, please feel free to open an issue or submit a pull request.
