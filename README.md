# IT2011 - Artificial Intelligence and Machine Learning
## Final Project: Automated Road Issue Detection

---

### 1. Project Overview & Problem Statement
Rapid urbanization requires efficient municipal management to maintain road safety, public cleanliness, and public property integrity. Traditional citizen complaint mechanisms suffer from slow reporting, manual triage delays, and subjective severity assessments. 

This project develops an end-to-end Machine Learning and Deep Learning system capable of automatically classifying municipal and environmental issues from photographic imagery into 7 critical urban categories:
1. **Broken Road Sign Issues**
2. **Damaged Road Issues**
3. **Illegal Parking Issues**
4. **Littering Garbage on Public Places Issues**
5. **Mixed Issues**
6. **Pothole Issues**
7. **Vandalism Issues**

By benchmarking six distinct machine learning algorithms — ranging from linear baselines and non-linear kernel/tree models to deep neural architectures — the optimal solution for municipal automated dispatch is identified.

---

### 2. Dataset Description
- **Domain:** Urban Infrastructure, Municipal Cleanliness, and Public Safety.
- **Classes:** 7 Balanced Classes.
- **Total Images in Full Processed Dataset:** 23,317 images (3,331 images per class, augmented & balanced during Milestone 1).
- **Standardized Image Dimensions:** 128 x 128 RGB (3 channels).
- **Feature Extraction (for Classical ML):** Principal Component Analysis (PCA) with 50 components, preserving over 81% of explained variance.

---

### 3. Group Member Roles and Individual Models (Final Evaluation)

| Member Name | Student ID | Milestone 1 Role | Assigned Model | Test Accuracy | Macro F1 |
| :--- | :--- | :--- | :--- | :---: | :---: |
| Nawarathna M.D.B | IT25101587 | Data Integrity & EDA | Logistic Regression (Softmax) | **38.92%** | **0.3835** |
| Disanayaka D.M.M | IT25102576 | Image Standardization | K-Nearest Neighbors (KNN) | **74.68%** | **0.7352** |
| Edirisooriya P.W.J | IT25103642 | Pixel Normalization | Support Vector Machine (SVM) | **62.86%** | **0.6210** |
| Gajasingha P.B | IT25100675 | Feature Engineering | Random Forest Ensemble | **72.26%** | **0.7188** |
| Chandralal R.K.R | IT25103457 | Outlier Removal | Artificial Neural Network (MLP) | **70.03%** | **0.6974** |
| Janandith P.D.V | IT25101738 | Data Augmentation | Convolutional Neural Network (CNN) | **85.96%** | **0.8572** |

---

### 4. Key Comparative Findings & Justification

1. **Undisputed Winner — CNN (Janandith P.D.V, 85.96% Accuracy, 0.8572 Macro F1):**
   - Convolutional layers preserve spatial grid structures (pothole borders, road sign geometry, spray graffiti edges), allowing the model to learn localized visual filters that resist viewpoint variations.
2. **Best Classical Model — KNN (Disanayaka D.M.M, 74.68% Accuracy):**
   - Distance-weighted KNN with Manhattan metric in PCA space effectively leverages local neighborhood patterns for classification.
3. **Strong Ensemble — Random Forest (Gajasingha P.B, 72.26% Accuracy):**
   - Bagging across 200 randomized trees smoothed decision boundaries and mitigated overfitting compared to single trees.
4. **Deep Learning Baseline — ANN/MLP (Chandralal R.K.R, 70.03% Accuracy):**
   - Feedforward neural network with dropout regularization bridges the gap between linear methods and spatial deep learning.
5. **Kernel Methods — SVM (Edirisooriya P.W.J, 62.86% Accuracy):**
   - RBF kernel demonstrated strong non-linear separation capability, though computational constraints required subset training.
6. **Linear Inseparability — Logistic Regression (Nawarathna M.D.B, 38.92% Accuracy):**
   - Confirms that urban infrastructure imagery cannot be accurately separated using linear hyperplanes.

---

### 5. Repository Layout
```
├── README.md                                # Project documentation & overview
├── 2026-Y2-S1-MLB-B6G1-02.pdf              # Final report (PDF)
├── data/                                    # Dataset directories
├── notebooks/                               # Individual model notebooks & group comparison
│   ├── Member_1_Logistic_Regression.ipynb
│   ├── Member_2_KNN.ipynb
│   ├── Member_3_SVM.ipynb
│   ├── Member_4_Decision_Trees_Random_Forest.ipynb
│   ├── Member_5_ANN_MLP.ipynb
│   ├── Member_6_CNN.ipynb
│   └── group_model_comparison.ipynb
├── preprocessing files/                     # Milestone 1: Preprocessing & Data Cleaning notebooks
└── results/
    ├── model_evaluations/                   # Comparison charts (PNG)
    ├── logs/                                # Benchmark metrics CSV & JSON
    └── outputs/                             # 23,317 processed images (7 class folders)
```

---

### 6. How to Run the Code

1. **Install Prerequisites:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pillow
   ```
2. **Open and Run Individual Notebooks:**
   Launch Jupyter Notebook or VS Code and open any notebook in `notebooks/`. All cells are organized step-by-step with clear beginner-friendly comments. Each notebook loads images directly from `results/outputs/`.
3. **View Group Comparison:**
   Open `notebooks/group_model_comparison.ipynb` to see all six models compared side-by-side with accuracy, F1, and overfitting analysis charts.
