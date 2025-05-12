
# Customer Churn Prediction Project

This project predicts customer churn using the Telco Customer Churn dataset from Kaggle, implementing multiple machine learning models (Logistic Regression, Decision Tree, k-NN) to compare performance. It showcases skills in data preprocessing, model building, evaluation, and visualization using Python and essential data science libraries.

## 📁 Project Structure

```
Churn_Prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── churn_analysis.ipynb
├── scripts/
├── results/
├── requirements.txt
└── README.md
```

## ⚙️ Step 1: Environment Setup

### Tools Used
- VS Code
- Anaconda (Python 3.9)
- Jupyter Notebook

### Setup Instructions
1. **Create and activate environment**:
   ```bash
   conda create -n churn_project python=3.9
   conda activate churn_project
   ```

2. **Install required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Open Notebook**:
   ```bash
   jupyter notebook notebooks/churn_analysis.ipynb
   ```

4. **Dataset**:
   - Download [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   - Place it in the `data/` folder.

## 📊 Project Workflow

1. **Exploratory Data Analysis (EDA)**:
   - Understand dataset characteristics and churn distribution.

2. **Data Preprocessing**:
   - Handle missing values, encode categories, scale features.

3. **Model Training**:
   - Logistic Regression
   - Decision Tree
   - k-NN

4. **Model Evaluation**:
   - Accuracy scores, classification reports, confusion matrices.

5. **Improvements**:
   - Hyperparameter tuning
   - SMOTE for imbalance correction

6. **Visualization**:
   - Churn distribution, feature importance, model performance comparison.

## 🧪 Requirements

- Python 3.9
- Jupyter Notebook
- See `requirements.txt` for full dependency list.

## 📌 Usage

```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

## 📜 License

This project is licensed under the MIT License.

## 👤 Author

**Didar Ali**  
📧 didaralideed@gmail.com
