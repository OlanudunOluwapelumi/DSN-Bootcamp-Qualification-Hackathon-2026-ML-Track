# DSN Bootcamp Qualification Hackathon 2026 · ML Track

## DSN Mart Sales Prediction | Olanudun Oluwapelumi

This project predicts `total_sales` for each product-store record in the DSN Mart test set. The notebook explores the data, prepares comparable DSN and original Big Mart records, estimates sales with a three-neighbour model, compares a Random Forest, and creates a submission file.

**Competition metric:** Root mean squared error (RMSE). Lower is better.

## Files and data

| File | Role |
| --- | --- |
| `DSN_Bootcamp_Qualification_Hackathon_2026_ML_Track_Olanudun_Oluwapelumi.ipynb` | Analysis, models, evaluation, and submission code. |
| `train.csv` | 6,818 DSN rows with the `total_sales` target. |
| `test.csv` | 1,705 DSN rows for prediction. |
| `train (14).csv` | Original Big Mart training data, 8,523 rows with `Item_Outlet_Sales`. Required by the KNN workflow; supply this file separately. |

The notebook's first code cell expects the original data at `train (14).csv` and the DSN files at `train.csv` and `test.csv` (with an alternate upload path for train). Change those path settings if your filenames differ. The notebook points to the [Big Mart source dataset](https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets?select=train.csv) in its code.

## 1. Read the data

The notebook loads all three CSV files and checks their shapes. DSN train and test contain 6,818 and 1,705 rows, adding up to the original data's 8,523 rows. The original file serves as the sales reference for the nearest-neighbour predictions.

## 2. Make the two datasets comparable

Original outlet establishment year becomes store age using the notebook's 2032 reference year. Location tiers and product categories are brought into a common text format. Missing weights are filled using each product's median across stores, then the overall median when necessary. Neither prepared weight column has missing values in the recorded run.

The code groups source and DSN rows by **store age, location tier, and cleaned category**. Every DSN row finds a source group. Groups have at least four candidates, so the chosen three-neighbour setting can be applied throughout.

## 3. Audit the target and missing fields

The labeled data contains 1,555 products and ten stores, with no repeated row IDs. Weight is missing in 1,225 training rows and store size in 1,919. Sales range from **32.70** to **12,996.82**, with a mean of **2,174.76** and median of **1,790.89**. RMSE gives large misses more influence, so the upper end of sales matters during evaluation.

## 4. Engineer features for the comparison model

The notebook creates `price_per_kg` from product price and prepared weight, plus `relative_visibility` from shelf visibility divided by the category median. These are used by the Random Forest comparison. The KNN model uses price, prepared weight, and visibility directly.

## 5. Explore sales across products and stores

Charts show the sales distribution, average sales by store format, and the price-sales relationship. The analysis also summarizes categories and correlations. Corner Shops average **336.4** per row, while Flagship Hypermarkets average **3,660.2**. Price and sales have a correlation of **0.57** in the training data; visibility and sales have a correlation of **-0.118**. These are descriptive associations.

## 6. Define the nearest-neighbour model

For each DSN record, the model looks only at original Big Mart records with the same store age, location tier, and cleaned category. It compares scaled product price, prepared weight, and shelf visibility within that group. `KNeighborsRegressor` uses **three neighbours** and distance weighting, giving closer records more influence. The reference targets are the original `Item_Outlet_Sales` values.

| KNN setting | Value |
| --- | ---: |
| Price scale | 3.5 |
| Weight scale | 0.25 |
| Visibility scale | 0.006 |
| Neighbours | 3 |
| Weighting | Distance |

## 7. Evaluate the KNN predictions

The notebook uses a fixed 20% sample of labeled DSN rows for a local RMSE check and also reports error across the full labeled set. It then breaks the sample's errors down by store format and price quartile.

| Check | Recorded RMSE |
| --- | ---: |
| 20% labeled-row check | **378.56** |
| All labeled DSN rows | **377.72** |
| Lowest price quartile in the check | 130.77 |
| Highest price quartile in the check | 524.14 |

The sampled rows are withheld from the DSN Random Forest fit, but the KNN reference is the separately loaded original Big Mart dataset, which contains sales values for source records. The **378.56** result measures this source-neighbour workflow, not performance on entirely new products or outlets without that reference data.

## 8. Compare a feature-based model

A `RandomForestRegressor` is trained on 80% of labeled DSN rows. It uses numeric product and store information, the two engineered features, and one-hot encoded categories. On the same 20% row indices, its RMSE is **1,107.22**, compared with **378.56** for the source-based KNN. This comparison reflects both the algorithms and their different information sources: KNN consults original sales records, while the forest fits DSN train only.

In the fitted forest, product price and the Corner Shop indicator have the largest reported feature importances. The notebook selects KNN for its submission.

## 9. Generate the submission

The notebook takes KNN predictions for the 1,705 test rows, floors any negative prediction at zero, and saves `primrose_dsn.csv` with two columns in the original test order:

```csv
id,total_sales
row_00009,2186.423752
row_00015,4456.300205
```

The code checks the output row count, ID order, and missing predictions before writing the file.

## Run the notebook

Use Python 3.10+ in Jupyter or Google Colab. Install the required packages:

```bash
python -m pip install numpy pandas scikit-learn matplotlib notebook
python -m notebook DSN_Bootcamp_Qualification_Hackathon_2026_ML_Track_Olanudun_Oluwapelumi.ipynb
```

Put `train (14).csv`, `train.csv`, and `test.csv` in the notebook's working directory, or edit the file paths in the first code cell. Run the cells in order. The final CSV, `primrose_dsn.csv`, is saved in that working directory.
