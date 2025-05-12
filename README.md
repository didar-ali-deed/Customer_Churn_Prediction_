# Customer Churn Prediction Project

This project predicts customer churn using the Telco Customer Churn dataset from Kaggle, implementing multiple machine learning models (Logistic Regression, Decision Tree, k-NN) to compare performance. The project demonstrates skills in data preprocessing, model building, evaluation, and visualization using Python, Pandas, Scikit-learn, Matplotlib, and Seaborn.

## Step 1: Set Up Your Environment

### Objective
Set up the Python environment in VS Code with Anaconda, install required libraries, download the dataset, and organize the project structure.

### Tools
- VS Code
- Anaconda
- Python 3.9
- Kaggle Telco Customer Churn dataset

### Steps Completed
1. **Created Anaconda Environment**:
   - Created and activated a dedicated environment: `conda create -n churn_project python=3.9` and `conda activate churn_project`.
2. **Installed Libraries**:
   - Installed required libraries: `pip install pandas numpy scikit-learn matplotlib seaborn`.
3. **Configured VS Code**:
   - Selected `churn_project` as the Python interpreter in VS Code.
   - Installed Python extension for enhanced coding support.
4. **Downloaded Dataset**:
   - Downloaded `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
   - Saved to `data/` folder.
5. **Set Up Project Structure**:
   - Created folders: `data/`, `notebooks/`, `scripts/`, `results/`.
   - Structure:

   Churn_Prediction/
        ├── data/
        │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
        ├── notebooks/
        │   └── churn_analysis.ipynb
        ├── scripts/
        ├── results/
6. **Created Jupyter Notebook**:
- Set up `churn_analysis.ipynb` in `notebooks/` using VS Code’s Jupyter support.
- Tested setup with a sample Pandas import.

### Next Steps
Proceed to Step 2: Load and explore the dataset using Pandas and visualize churn distribution with Seaborn/Matplotlib.

### Technical Skills
- Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- Jupyter Notebook, VS Code, Anaconda