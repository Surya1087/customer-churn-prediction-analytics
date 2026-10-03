# Notebook-first version

This folder is the learning-oriented notebook version of the project. Open notebooks directly in VS Code and run cells from top to bottom.

Recommended order:

1. `Customer_Churn_Complete.ipynb` — complete end-to-end workflow in one file.
2. `01_Data_SQL_EDA.ipynb` — data, SQL, and exploratory analysis.
3. `02_Model_Training.ipynb` — preprocessing, five models, tuning, and evaluation.
4. `03_Prediction_Segmentation.ipynb` — prediction and K-Means segmentation.

Individual model notebooks are also available under `notebooks/models/`:

- `01_DummyClassifier.ipynb`
- `02_LogisticRegression.ipynb`
- `03_DecisionTreeClassifier.ipynb`
- `04_RandomForestClassifier.ipynb`
- `05_XGBClassifier.ipynb`

The complete notebook now includes EDA charts, SQL outputs, model comparison, confusion matrices, ROC curves, permutation importance, optional real SHAP output, prediction, and K-Means segmentation.

The notebooks use the real dataset at `../data/raw/` and save artifacts to `../models/`. They do not replace the existing modular application.
