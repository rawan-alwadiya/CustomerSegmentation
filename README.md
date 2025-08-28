# **CustomerSegmentation: End-to-End ML Pipeline**

**CustomerSegmentation** is an end-to-end machine learning project designed to segment credit card customers into meaningful groups based on their financial behavior.  
The workflow includes **exploratory data analysis (EDA)**, **preprocessing** (handling missing values, scaling), **dimensionality reduction (PCA)**, **clustering (K-Means)**, and finally **classification using Support Vector Machine (Polynomial Kernel)**.  

The project concludes with deployment as an interactive **Streamlit web app**, allowing users to input customer details and instantly view their predicted segment.  

---

## **Demo**

- 🎥 [View LinkedIn Demo Post](https://www.linkedin.com/posts/rawan-alwadeya-17948a305_machinelearning-customersegmentation-clustering-activity-7366848369752801280-4srU?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3YzG0BAZw48kimDDr_guvq8zXgSjDgk_I)  
- 🌐 [Try the App Live on Streamlit](https://customersegmentation-xt7vbctp6slquksrnvdk9t.streamlit.app/)  

![App Demo](https://github.com/rawan-alwadiya/CustomerSegmentation/blob/main/CustomerSegmentation.png)

---

## **Project Overview**

The workflow includes:  
- **Exploratory Data Analysis (EDA):** Visualization of customer spending patterns  
- **Preprocessing:** Handling missing values, scaling numerical features  
- **Dimensionality Reduction:** `PCA` to preserve variance and simplify high-dimensional data  
- **Clustering:** `K-Means` to group customers into 3 meaningful clusters  
- **Handling Imbalance:** `RandomOverSampler` for fair representation of each cluster  
- **Classification:** `SVM (Polynomial Kernel)` to classify customers into their segment  
- **Deployment:** Interactive Streamlit web app for real-time predictions  

---

## **Objective**

Develop a robust and interpretable pipeline for **credit card customer segmentation** to help businesses understand customer behaviors and tailor marketing strategies toward different spending profiles.  

---

## **Dataset**

- **Source:** [Customer Segmentation Dataset (Kaggle)](https://www.kaggle.com/datasets/mahnazarjmand/customer-segmentation)  
- **Samples:** 8,950  
- **Features:** 17 financial attributes (balance, purchases, transactions, payments, etc.)  
- **Target:** Cluster-based customer segments (Low / Moderate / High spending)  

---

## **Project Workflow**

- **EDA & Visualization:** Identified customer spending distributions & patterns  
- **Preprocessing:**  
  - Missing values filled using `KNN Imputer`  
  - Standardized numerical features with `StandardScaler`  
- **Dimensionality Reduction:** `PCA` applied to reduce feature space  
- **Clustering (K-Means):**  
  - Cluster 0 → Low-spending / Inactive customers  
  - Cluster 1 → Moderate-spending customers  
  - Cluster 2 → High-spending active customers  
- **Handling Imbalance:** Applied `RandomOverSampler` to balance cluster representation during training  
- **Modeling (SVM - Polynomial Kernel):**  
  - Trained classifier on PCA-transformed data  
  - Chosen due to non-linear separability of the clusters  

---

## **Performance Results**

**Support Vector Machine Classifier (Polynomial Kernel):**  
- Accuracy: **97.95%**  
- Precision: **97.96%**  
- Recall: **97.95%**  
- F1-score: **97.94%**  

✅ The model achieved **balanced and reliable performance** across all three customer segments.  

---

## **Project Links**

- [Live Streamlit App](https://customersegmentation-xt7vbctp6slquksrnvdk9t.streamlit.app/)  
- [Kaggle Notebook](https://www.kaggle.com/code/rawanalwadeya/customersegmentation-end-to-end-ml-pipeline)  
- [Dataset](https://www.kaggle.com/datasets/mahnazarjmand/customer-segmentation)  

---

## **Tech Stack**

**Languages & Libraries:**  
- Python, Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Imbalanced-learn (`RandomOverSampler`)  
- Streamlit (Deployment)  

**Techniques:**  
- `PCA` for dimensionality reduction  
- `K-Means` clustering  
- `SVM (Polynomial Kernel)` classification  
- Oversampling for cluster balance  
- Real-time web deployment with Streamlit  
