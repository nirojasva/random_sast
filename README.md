# RSAST: Random Scalable and Accurate Subsequence Transform for Time Series Classification

RSAST is a shapelet-based time series classification method based on SAST. But, aiming to overcome scalability issues. 



### Results RSAST

- [Results Default Split](./ExperimentationRSAST/results_default_split.csv)

- [Results 10 Resamplings](./ExperimentationRSAST/results_default_split.csv)

- [Execution time regarding the number of series](./ExperimentationRSAST/results_comparison_accuracy/df_overall_comparison_scalability_number_of_seriesLR.csv)

- [Execution time regarding series length](./ExperimentationRSAST/results_comparison_accuracy/df_overall_comparison_scalability_series_length.csv)



### RSAST, SAST and STC

#### Pairwise accuracy comparison

| ![](./ExperimentationRSAST/images_one_vs_one_comparison/RSASTvsSAST.png) | ![](./ExperimentationRSAST/images_one_vs_one_comparison/RSASTvsSTC.png) |
| -------------------------------------------------- | ---------------------------------------------------- |

#### Critical difference diagram

![SAST-models CDD](./ExperimentationRSAST/images_cd_diagram/comparison_rsast_sast_stc.png)

### STC-k vs STC

#### Pairwise accuracy comparison

| ![STC vs STC-1](images/scatter-stc-vs-stck1.png)    | ![STC vs STC-0.25](images/scatter-stc-vs-stck025.png) |
| --------------------------------------------------- | ----------------------------------------------------- |
| ![STC vs STC-0.5](images/scatter-stc-vs-stck05.png) | ![STC vs STC-0.75](images/scatter-stc-vs-stck075.png) |

#### Critical difference diagram

![SCT vs STC-k CDD](images/cdd-stck.png)

### SAST vs STC

| ![SAST vs STC-1](images/scatter-sast-stc1.png) | ![SAST vs STC-1](images/scatter-sast-stc.png) |
| ---------------------------------------------- | --------------------------------------------- |

#### Critical difference diagram

![CDD SAST vs STC](images/cdd-sast-stck.png)

#### Percentage of wins per problem type

![win-per-dataset-type-stck](images/win-per-dataset-type-stck.png)

### SAST vs others shapelets methods

#### Pairwise accuracy comparison

| ![SAST vs ELIS++](images/scatter-sast-elis++.png) | ![SAST vs LS](images/scatter-sast-ls.png) |
| ------------------------------------------------- | ----------------------------------------- |

![SAST vs FS](images/scatter-sast-fs.png)

#### Critical difference diagram

![SAST vs other shapelets CDD](images/cdd-sast-vs-others-shapelet.png)

#### Percentage of wins per problem types

![win-per-dataset-type-shapelet](images/win-per-dataset-type-shapelet.png)

### SAST vs SOTA

#### Pairwise accuracy comparison

| ![scatter-sast-vs-rocket](images/scatter-sast-vs-rocket.jpg) | ![scatter-sast-ridge-vs-hive-cote](images/scatter-sast-ridge-vs-hive-cote.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |

![scatter-sast-ridge-vs-chief](images/scatter-sast-ridge-vs-chief.jpg)

#### Percentage of wins per problem type


![win-per-dataset-type-sota](./images/win-per-dataset-type-sota.png)


### Scalability plots

- Regarding the length of time series

![](images/line-scalability-series-length.jpg)

- Regarding the number of time series in the dataset

![](images/line-scalability-nb-series.jpg)

## Usage

```python
import numpy as np
from sast.utils_sast import *
from sast.sast import *
from sklearn.linear_model import RidgeClassifierCV
from sktime.datasets import load_UCR_UEA_dataset

ds='Chinatown' # Chosing a dataset from # Number of classes to consider
rtype="numpy2D"

X_train, y_train = load_UCR_UEA_dataset(name=ds, extract_path='data', split="train", return_type=rtype)

X_test, y_test = load_UCR_UEA_dataset(name=ds, extract_path='data', split="test", return_type=rtype)

clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
rsast_ridge = RSAST(n_random_points=10, nb_inst_per_class=10, len_method="both")
rsast_ridge.fit(X_train, y_train)



prediction = rsast_ridge.predict(X_test)
```

### Dependencies

- numpy == 1.18.5
- numba == 0.50.1
- scikit-learn == 0.23.1
- sktime == 0.5.3

