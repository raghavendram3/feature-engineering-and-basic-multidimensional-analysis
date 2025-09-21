# Feature Engineering and Regression Analysis Documentation

## Overview

This code performs automated feature engineering by systematically combining input variables using mathematical operations to predict activation energy (Ea). It explores all possible combinations of three variables with two sequential mathematical operations to find the best predictive features. It can be modified to take combinations of two variables.

## Purpose

The script aims to:
- **Discover non-linear relationships** between input variables and activation energy
- **Automate feature engineering** by testing mathematical combinations
- **Identify optimal predictive features** using linear regression and RMSE evaluation
- **Visualize the best relationships** between engineered features and target variable

## Methodology

### Feature Engineering Process
1. **Triplet Selection**: Choose all possible combinations of 3 variables from input features
2. **Operation Application**: Apply two sequential mathematical operations:
   - First operation: Combine variables 1 and 2
   - Second operation: Combine result with variable 3
3. **Expression Format**: `((Variable1 OP1 Variable2) OP2 Variable3)`

### Mathematical Operations
- **Addition (+)**: Linear combination of variables
- **Subtraction (-)**: Difference-based features
- **Multiplication (*)**: Product interactions
- **Division (/)**: Ratio-based features (with zero-division protection)

### Evaluation Metrics
- **RMSE (Root Mean Square Error)**: Primary metric for model performance
- **Linear Regression**: Simple model to test feature predictive power
- **Data Validation**: Automatic removal of NaN and infinite values

## Input Data Format

- **File**: Tab-separated text file (`train.txt`)
- **Structure**: 
  - Column 1: Target variable `$E_a$` (activation energy)
  - Columns 2+: Input features/descriptors
- **Requirements**: Numerical data only

## Output

### Numerical Results
- Top 10 feature combinations ranked by RMSE
- Mathematical expressions for best features
- Performance metrics for each combination

### Visualizations
- 5×2 subplot grid showing best 10 combinations
- Scatter plots of engineered features vs. activation energy
- Linear regression fit lines
- RMSE values displayed on each plot

## Mathematical Complexity

The code explores **N³ × 16** possible combinations where:
- **N**: Number of input variables
- **16**: Total operation combinations (4 operations × 4 operations)
- **Computational Complexity**: O(N³) for variable selection + O(M) for regression fitting

## Applications

This approach is particularly useful for:
- **Materials Science**: Discovering structure-property relationships
- **Chemical Engineering**: Finding optimal process parameter combinations
- **Catalysis Research**: Identifying descriptors for reaction energetics
- **Machine Learning**: Automated feature discovery for regression problems

## Limitations

1. **Linear Relationships Only**: Uses linear regression for evaluation
2. **Limited Operations**: Only basic arithmetic operations
3. **No Interaction Terms**: Doesn't consider higher-order interactions
4. **Computational Cost**: Scales as O(N³) with number of variables
5. **Overfitting Risk**: May find spurious correlations in small datasets

## Performance Considerations

- **Memory Usage**: Stores all valid combinations in memory
- **Processing Time**: Depends on number of variables and data size
- **Error Handling**: Robust to division by zero and invalid operations
- **Data Quality**: Automatic filtering of problematic values

## Potential Extensions

- Add more mathematical operations (log, exp, power, etc.)
- Implement cross-validation for more robust evaluation
- Include polynomial features or interaction terms
- Add regularization techniques (Ridge, Lasso)
- Implement parallel processing for large datasets
