# RSAST: Random Scalable and Accurate Subsequence Transform for Time Series Classification

RSAST is a shapelet-based time series classification method based on SAST. But, aiming to overcome scalability issues. 



## Results RSAST

- [Results Default Split](./ExperimentationRSAST/results_default_split.csv)

- [Results 10 Resamplings](./ExperimentationRSAST/results_10resampling.csv)

- [Results Comparison RSAST](./ExperimentationRSAST/results_comparison_rsast.csv)

- [Execution time regarding the number of series](./ExperimentationRSAST/results_comparison_accuracy/df_overall_comparison_scalability_number_of_seriesLR.csv)

- [Execution time regarding series length](./ExperimentationRSAST/results_comparison_accuracy/df_overall_comparison_scalability_series_length.csv)



## RSAST, SAST and STC

### Critical difference diagram

![](./ExperimentationRSAST/images_cd_diagram/comparison_rsast_sast_st.png)

### Pairwise accuracy comparison

| ![](./ExperimentationRSAST/images_one_vs_one_comparison/RSASTvsSAST.png) | ![](./ExperimentationRSAST/images_one_vs_one_comparison/RSASTvsSTC.png) |
| -------------------------------------------------- | ---------------------------------------------------- |


## Shapelet Approaches

### Critical difference diagram

![](./ExperimentationRSAST/images_cd_diagram/comparison_shapelet_method.png)

### Pairwise accuracy comparison

| ![](./ExperimentationRSAST/images_one_vs_one_comparison/RSASTvsFS.png) | ![](./ExperimentationRSAST/images_one_vs_one_comparison/RSASTvsLS.png) | ![](./ExperimentationRSAST/images_one_vs_one_comparison/RSASTvsRDST.png) |
| ----------------------------------------- | ------------------------------------------- | ------------------------------------------- |

## Alternative Length Methods

In order to explore another alternatives for the default length method of the shapelets (ACF&PACF) some supplementary length methods are examined: Max PACF and None.

### Critical difference diagram per Length method

| ![](./ExperimentationRSAST/images_cd_diagram/cd-diagram_ACF&PACF.png) | ![](./ExperimentationRSAST/images_cd_diagram/cd-diagram_Max_PACF.png) | ![](./ExperimentationRSAST/images_cd_diagram/cd-diagram_None.png) |


### Critical difference diagram best performance

![](./ExperimentationRSAST/images_cd_diagram/cd-diagram_best_com.png)