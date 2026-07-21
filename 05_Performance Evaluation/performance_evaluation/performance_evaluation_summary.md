# Performance Evaluation Summary

## Metric Definitions

- **MAE**: Average absolute prediction error in RUL cycles. Lower is better.
- **MSE**: Average squared error. Penalizes larger mistakes more strongly. Lower is better.
- **RMSE**: Square root of MSE, in cycles. Easier to interpret than MSE. Lower is better.
- **R2**: Variance explained by the model. Higher is better.

## Results Table

| Model          |     MAE |     MSE |    RMSE |       R2 |   Rank |
|:---------------|--------:|--------:|--------:|---------:|-------:|
| Neural Network | 17.4117 | 696.273 | 26.387  | 0.5968   |      1 |
| XGBoost        | 19.5924 | 709.892 | 26.6438 | 0.588914 |      2 |
| Random Forest  | 20.0495 | 727.52  | 26.9726 | 0.578706 |      3 |

## Model Ranking

| Model          |     MAE |    RMSE |       R2 |
|:---------------|--------:|--------:|---------:|
| Neural Network | 17.4117 | 26.387  | 0.5968   |
| XGBoost        | 19.5924 | 26.6438 | 0.588914 |
| Random Forest  | 20.0495 | 26.9726 | 0.578706 |

**Best-performing model:** Neural Network

## Dissertation Discussion

The models demonstrate that machine learning can predict RUL from engineered sensor features with measurable accuracy. 
The best model is the one with the lowest MAE and RMSE and the highest R2, indicating it captures degradation patterns most effectively. 
This supports the research objective of using NASA turbofan data for predictive maintenance and answers the research question on prediction accuracy.
