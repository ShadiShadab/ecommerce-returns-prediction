# E-commerce Returns Prediction: ASOS and TheLook

Predicting whether an online fashion purchase will be returned, using two public e-commerce datasets and two modelling perspectives.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-2ea44f)

Team project, **MIT Emerging Talent Program, Computer & Data Science Project (CDSP)**, Group 17 "Mission Impostible", summer 2025.
This repository is my copy of the team repository [MIT-Emerging-Talent/ET6-CDSP-group-17-repo-Mission_Impostible](https://github.com/MIT-Emerging-Talent/ET6-CDSP-group-17-repo-Mission_Impostible), with its full commit history.

---

## At a glance

| | ASOS track (my lead) | TheLook track (teammates) |
|---|---|---|
| **Data** | [ASOS GraphReturns](https://osf.io/c793h/): 1.3M+ purchase events linking customers and products | [TheLook E-commerce](https://www.kaggle.com/code/aniqohhanahaura/thelook-dataset) (synthetic): 180,952 order items |
| **Approach** | Customer–product graph construction, feature engineering, classification | Tabular feature engineering, classification |
| **Models** | Logistic Regression, Random Forest, XGBoost | Logistic Regression (baseline), XGBoost |
| **Best result** | **XGBoost: ROC-AUC 0.825, accuracy 74.2%**, precision 75.1%, recall 79.8% (273,827 test rows) | XGBoost: ROC-AUC 0.655 ± 0.004 (cross-validated) |
| **Strongest predictors** | Customer's historical return rate, product return rate, shipping country | Product category, discount, basket size, customer tenure |

---

## The business problem

Returns are one of the largest hidden costs in online fashion retail: reverse logistics, re-stocking, lost margin and environmental waste. If a retailer can estimate the probability of a return at purchase time, it can act early, for example with better size guidance, targeted policies or adjusted inventory planning.

**Question:** how well can customer, product and transaction features predict whether an item will be returned? Does a graph view of customer–product relationships add useful signal?

---

## My contribution

I led the **ASOS track** end to end and made 154 of the repository's 254 commits (53 of the 68 commits that touched ASOS files):

- **Data acquisition and preparation:** loaded the ASOS GraphReturns data, merged event, customer and product tables, handled missing values and encoded categorical features
- **Graph construction:** built a bipartite customer–product graph with NetworkX, where purchases are edges labelled returned or kept
- **Exploratory analysis:** return rates by product type, brand, country, age and price, plus PCA and t-SNE projections
- **Modelling and evaluation:** trained and compared Logistic Regression, Random Forest and XGBoost. Evaluated them with accuracy, precision, recall and ROC-AUC, and explained them with feature importance and SHAP.
- **Communication:** the ASOS communication strategy for business stakeholders

The TheLook track was built mainly by Pyae Linn and Derek Karungani. Pyae Linn built the Streamlit app and Derek Karungani wrote the stakeholder analysis. Project setup and collaboration documents were shared work.

---

## Results: ASOS

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| **XGBoost** | **0.742** | **0.751** | **0.798** | **0.825** |
| Logistic Regression | 0.739 | 0.749 | 0.793 | 0.820 |
| Random Forest | 0.725 | 0.742 | 0.771 | 0.801 |

Source: model comparison output in [`ASOS_Data_Analysis.ipynb`](4_data_analysis/ASOS_dataset/ASOS_Data_Analysis.ipynb).

<p align="center">
  <img src="4_data_analysis/ASOS_dataset/Models/model_roc_comparison.png" width="48%" alt="ROC-AUC comparison of ASOS models">
  <img src="4_data_analysis/ASOS_dataset/Models/xgboost_feature_importance.png" width="48%" alt="XGBoost feature importance">
</p>

**Interpretation:** past behaviour dominates. Customers who returned items before, and products that are often returned, are the strongest signals. Demographics add little.

## Results: TheLook

XGBoost reached a cross-validated ROC-AUC of 0.655, compared with 0.499 for the logistic baseline. This shows that numeric order features alone carry limited signal in this synthetic dataset. See [`4_data_analysis/theLook_Ecommerce/`](4_data_analysis/theLook_Ecommerce/README.md).

---

## Reproduce

```bash
git clone https://github.com/ShadiShadab/ecommerce-returns-prediction.git
cd ecommerce-returns-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- **TheLook:** the raw CSVs are included in `1_datasets/thelook_Ecommerce/`. Run the notebooks in `2_data_preparation/theLook_Ecommerce/`, `3_data_exploration/theLook_Ecommerce/` and `4_data_analysis/theLook_Ecommerce/` in that order.
- **ASOS:** download the GraphReturns files from [OSF](https://osf.io/c793h/) (too large for GitHub). The ASOS notebooks were developed in Google Colab and read their inputs from `/content/`. Upload the files there, or change the paths at the top of each notebook.
- **Streamlit app:**

  ```bash
  streamlit run 5_communication_strategy/strweb.py
  ```

---

## Repository structure

```text
├── 0_domain_study/              Background research on e-commerce returns
├── 1_datasets/                  Dataset documentation; TheLook raw CSVs
├── 2_data_preparation/          Cleaning and feature engineering (ASOS, TheLook)
├── 3_data_exploration/          EDA notebooks and visualisations
├── 4_data_analysis/             Modelling notebooks, model plots, SHAP output
├── 5_communication_strategy/    Stakeholder analysis, communication plan, Streamlit app (strweb.py)
├── 6_final_presentation/        Final presentation (PDF)
├── collaboration/, notes/, Retrospective/   Team process documents
└── requirements.txt
```

Final presentation: [Group 17 – Final CDSP Presentation (PDF)](6_final_presentation/Group%2017%20-%20Final%20CDSP%20Presentation.pdf)

---

## Limitations and next steps

- **Possible leakage in historical return rates:** if customer and product return rates were computed over the full dataset, they include the purchase being predicted. The next version should compute them only from earlier purchases or from the training split.
- **Graph modelling:** the customer–product graph was built, but the models are tabular classifiers on graph-derived and merged features. A graph neural network is a natural next step.
- **No timestamps in ASOS:** a time-based validation split is not possible with this dataset.
- **TheLook is synthetic,** which limits how far its findings carry over to real retail.

---

## Team

- **Shadi Shadabshoar**: ASOS track and ASOS communication strategy. [LinkedIn](https://www.linkedin.com/in/shadi-shadabshoar/)
- **Pyae Linn**: TheLook track, Streamlit app. [LinkedIn](https://www.linkedin.com/in/pyae-linn-2419a62a4/)
- **Derek Karungani**: TheLook track, stakeholder analysis

Additional contributions to the domain study and retrospectives came from other cohort members.

## License

[MIT](LICENSE). Datasets remain under their original licences (ASOS GraphReturns on OSF; TheLook on Kaggle/BigQuery public data).
