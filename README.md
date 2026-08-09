# Customer Segmentation

A modular rebuild of `Unsupervised_Clustering_Solution.ipynb`. 

The project groups mall customers into segments using K-means, evaluating candidate cluster counts with inertia and silhouette score, and assigns new customers to the fitted segments.

## Setup

Run every command below from the project root (the folder containing `app.py`).

The virtual environment is not included in the repository, so this creates a
fresh one:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
### Run test verifications (pytest)

```powershell
python -m pytest -v
```

### Run the application

```powershell
streamlit run app.py
```

## Project structure

```text
3_customer_clustering/
|-- app.py
|-- README.md
|-- requirements.txt
|-- data/
|   `-- mall_customers.csv
|-- original_notebook/
|   `-- Unsupervised_Clustering_Solution.ipynb
|-- src/
|   |-- __init__.py
|   |-- data.py
|   |-- preprocessing.py
|   |-- modeling.py
|   `-- logging_config.py
`-- tests/
    `-- test_project.py
```

## How it works

The bundled dataset contains 200 hypothetical mall customers. `Customer_ID` is excluded because an identifier does not describe customer behavior. `Gender` is retained for the overview but excluded from K-means because its categories do not have a meaningful numeric distance.

The original notebook's two-feature view uses Annual Income and Spending Score on their original units. It evaluates `k=3` through `k=8`; the highest silhouette score selects `k=5`.

The final model adds Age. Because age, income, and spending score use different units and numeric ranges, `StandardScaler` gives them comparable influence on distance. The same fitted scaler transforms new customer input. The final candidate comparison selects `k=6`, also by highest silhouette score.

Every K-means fit uses the notebook exercise's `k-means++` initialization and `random_state=42`. Inertia is reported as supporting elbow evidence, but it does not select the model because it always decreases as clusters are added.

## Limitations

- Educational demonstration only, using a small hypothetical dataset. 
- No known correct segment labels, so silhouette measures separation rather than real-world correctness. 
- Cluster numbers are arbitrary labels, and K-means assigns any in-range customer to a group even when the fit is weak.

## Links

- GitHub repository: https://github.com/ihyrda/3-customer-clustering
- Deployed Streamlit app: https://3-customer-clustering-ihyrda.streamlit.app/
