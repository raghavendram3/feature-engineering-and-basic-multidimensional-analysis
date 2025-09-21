#!/usr/bin/env python3
"""
Automated Feature Engineering for Activation Energy Prediction

This script performs systematic feature engineering by combining input variables
with mathematical operations to discover optimal predictive features for
activation energy (Ea) using linear regression analysis.

Author: Raghavendra Meena
Dependencies: pandas, numpy, matplotlib, scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Tuple, Dict, Callable
import warnings
import time

warnings.filterwarnings('ignore')


class FeatureEngineeringAnalyzer:
    """
    Automated feature engineering analyzer for discovering optimal variable combinations.
    """
    
    def __init__(self, data_file_path: str, target_column: str = "$E_a$"):
        """
        Initialize the analyzer with data.
        
        Parameters:
        -----------
        data_file_path : str
            Path to the tab-separated data file
        target_column : str
            Name of the target variable column
        """
        self.data_file_path = data_file_path
        self.target_column = target_column
        self.data = None
        self.target_values = None
        self.feature_columns = None
        self.results = []
        
        # Define mathematical operations with safe implementations
        self.mathematical_operations = {
            "+": {
                "function": np.add,
                "symbol": "+",
                "description": "Addition"
            },
            "-": {
                "function": np.subtract,
                "symbol": "-",
                "description": "Subtraction"
            },
            "*": {
                "function": np.multiply,
                "symbol": "*",
                "description": "Multiplication"
            },
            "/": {
                "function": self._safe_divide,
                "symbol": "/",
                "description": "Division (safe)"
            }
        }
    
    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """
        Perform safe division avoiding division by zero.
        
        Parameters:
        -----------
        numerator : np.ndarray
            Numerator values
        denominator : np.ndarray
            Denominator values
            
        Returns:
        --------
        np.ndarray: Result of division with NaN for zero denominators
        """
        return np.divide(numerator, np.where(np.abs(denominator) < 1e-10, np.nan, denominator))
    
    def load_data(self) -> bool:
        """
        Load and validate input data.
        
        Returns:
        --------
        bool: True if data loaded successfully, False otherwise
        """
        try:
            print(f"Loading data from: {self.data_file_path}")
            self.data = pd.read_csv(self.data_file_path, sep="\t")
            
            # Extract target variable and features
            if self.target_column not in self.data.columns:
                print(f"Error: Target column '{self.target_column}' not found in data")
                return False
            
            self.target_values = self.data[self.target_column]
            self.feature_columns = [col for col in self.data.columns if col != self.target_column]
            
            print(f"Data loaded successfully:")
            print(f"  • Data shape: {self.data.shape}")
            print(f"  • Target variable: {self.target_column}")
            print(f"  • Number of features: {len(self.feature_columns)}")
            print(f"  • Feature names: {self.feature_columns}")
            
            # Validate data types
            non_numeric_cols = self.data.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                print(f"Warning: Non-numeric columns detected: {non_numeric_cols}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def generate_feature_combinations(self, num_variables: int = 3) -> List[Tuple]:
        """
        Generate all combinations of variables for feature engineering.
        
        Parameters:
        -----------
        num_variables : int
            Number of variables to combine (default: 3)
            
        Returns:
        --------
        List[Tuple]: List of variable combinations
        """
        if len(self.feature_columns) < num_variables:
            print(f"Error: Need at least {num_variables} features, but only {len(self.feature_columns)} available")
            return []
        
        variable_combinations = list(combinations(self.feature_columns, num_variables))
        
        print(f"Generated {len(variable_combinations)} variable combinations")
        print(f"Total expressions to evaluate: {len(variable_combinations) * len(self.mathematical_operations) ** 2}")
        
        return variable_combinations
    
    def create_engineered_feature(self, var1: str, var2: str, var3: str, 
                                 op1_key: str, op2_key: str) -> Tuple[np.ndarray, str]:
        """
        Create an engineered feature using two sequential operations.
        
        Parameters:
        -----------
        var1, var2, var3 : str
            Variable names to combine
        op1_key, op2_key : str
            Operation keys for first and second operations
            
        Returns:
        --------
        Tuple[np.ndarray, str]: Engineered feature values and expression string
        """
        op1_func = self.mathematical_operations[op1_key]["function"]
        op2_func = self.mathematical_operations[op2_key]["function"]
        op1_symbol = self.mathematical_operations[op1_key]["symbol"]
        op2_symbol = self.mathematical_operations[op2_key]["symbol"]
        
        # Apply first operation: var1 op1 var2
        intermediate_result = op1_func(self.data[var1].values, self.data[var2].values)
        
        # Apply second operation: (var1 op1 var2) op2 var3
        final_feature = op2_func(intermediate_result, self.data[var3].values)
        
        # Create expression string
        expression = f"(({var1} {op1_symbol} {var2}) {op2_symbol} {var3})"
        
        return final_feature, expression
    
    def evaluate_feature_quality(self, engineered_feature: np.ndarray, 
                                expression: str) -> Dict:
        """
        Evaluate the quality of an engineered feature using linear regression.
        
        Parameters:
        -----------
        engineered_feature : np.ndarray
            Values of the engineered feature
        expression : str
            Mathematical expression string
            
        Returns:
        --------
        Dict: Evaluation metrics and data
        """
        # Remove invalid values (NaN, Inf, -Inf)
        valid_mask = (
            ~np.isnan(engineered_feature) & 
            ~np.isinf(engineered_feature) & 
            np.isfinite(engineered_feature)
        )
        
        valid_feature_values = engineered_feature[valid_mask]
        valid_target_values = self.target_values[valid_mask]
        
        # Skip if too few valid values
        if len(valid_feature_values) < 5:
            return None
        
        # Fit linear regression model
        feature_reshaped = valid_feature_values.reshape(-1, 1)
        regression_model = LinearRegression()
        regression_model.fit(feature_reshaped, valid_target_values)
        
        # Make predictions
        predictions = regression_model.predict(feature_reshaped)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(valid_target_values, predictions))
        r2 = r2_score(valid_target_values, predictions)
        coefficient = regression_model.coef_[0]
        intercept = regression_model.intercept_
        
        return {
            'rmse': rmse,
            'r2': r2,
            'coefficient': coefficient,
            'intercept': intercept,
            'expression': expression,
            'feature_values': valid_feature_values,
            'target_values': valid_target_values,
            'predictions': predictions,
            'num_valid_points': len(valid_feature_values),
            'data_coverage': len(valid_feature_values) / len(self.target_values)
        }
    
    def run_comprehensive_analysis(self, num_variables: int = 3, 
                                 max_results: int = 10) -> List[Dict]:
        """
        Run comprehensive feature engineering analysis.
        
        Parameters:
        -----------
        num_variables : int
            Number of variables to combine
        max_results : int
            Maximum number of top results to return
            
        Returns:
        --------
        List[Dict]: Top feature combinations with evaluation metrics
        """
        print("Starting comprehensive feature engineering analysis...")
        start_time = time.time()
        
        # Generate variable combinations
        variable_combinations = self.generate_feature_combinations(num_variables)
        if not variable_combinations:
            return []
        
        # Evaluate all combinations
        self.results = []
        total_combinations = len(variable_combinations) * len(self.mathematical_operations) ** 2
        processed = 0
        
        print("\nProcessing combinations...")
        
        for var_combo in variable_combinations:
            var1, var2, var3 = var_combo
            
            for op1_key in self.mathematical_operations.keys():
                for op2_key in self.mathematical_operations.keys():
                    try:
                        # Create engineered feature
                        engineered_feature, expression = self.create_engineered_feature(
                            var1, var2, var3, op1_key, op2_key
                        )
                        
                        # Evaluate feature quality
                        evaluation_result = self.evaluate_feature_quality(
                            engineered_feature, expression
                        )
                        
                        if evaluation_result is not None:
                            self.results.append(evaluation_result)
                        
                    except Exception as e:
                        # Skip problematic combinations
                        continue
                    
                    processed += 1
                    if processed % 1000 == 0:
                        progress = (processed / total_combinations) * 100
                        print(f"Progress: {progress:.1f}% ({processed}/{total_combinations})")
        
        # Sort results by RMSE (ascending)
        self.results.sort(key=lambda x: x['rmse'])
        
        # Get top results
        top_results = self.results[:max_results]
        
        elapsed_time = time.time() - start_time
        print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")
        print(f"Evaluated {len(self.results)} valid combinations")
        print(f"Returning top {len(top_results)} results")
        
        return top_results
    
    def visualize_top_results(self, top_results: List[Dict], 
                            figure_size: Tuple[int, int] = (15, 12)):
        """
        Create visualization of top feature combinations.
        
        Parameters:
        -----------
        top_results : List[Dict]
            Top evaluation results to visualize
        figure_size : Tuple[int, int]
            Figure size for the plot
        """
        if not top_results:
            print("No results to visualize")
            return
        
        num_results = len(top_results)
        num_rows = (num_results + 1) // 2
        
        plt.figure(figsize=figure_size)
        plt.suptitle("Top Feature Engineering Results: Engineered Features vs Activation Energy", 
                    fontsize=16, fontweight='bold')
        
        for i, result in enumerate(top_results, 1):
            plt.subplot(num_rows, 2, i)
            
            # Scatter plot of data points
            plt.scatter(result['feature_values'], result['target_values'], 
                       alpha=0.6, color='blue', s=30, label='Data Points')
            
            # Linear regression fit line
            plt.plot(result['feature_values'], result['predictions'], 
                    color='red', linewidth=2, label='Linear Fit')
            
            # Formatting
            plt.title(f"Rank {i}: {result['expression']}\n"
                     f"RMSE: {result['rmse']:.4f} | R²: {result['r2']:.4f} | "
                     f"Coverage: {result['data_coverage']:.1%}", 
                     fontsize=10, pad=10)
            
            plt.xlabel("Engineered Feature Value", fontsize=9)
            plt.ylabel("Activation Energy ($E_a$)", fontsize=9)
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            
            # Add equation to plot
            equation_text = f"y = {result['coefficient']:.3f}x + {result['intercept']:.3f}"
            plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, 
                    fontsize=8, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def print_detailed_results(self, top_results: List[Dict]):
        """
        Print detailed analysis results.
        
        Parameters:
        -----------
        top_results : List[Dict]
            Top evaluation results to display
        """
        print("\n" + "="*80)
        print("DETAILED FEATURE ENGINEERING RESULTS")
        print("="*80)
        
        if not top_results:
            print("No valid results found.")
            return
        
        print(f"\nTop {len(top_results)} Feature Combinations (Ranked by RMSE):")
        print("-" * 80)
        
        for rank, result in enumerate(top_results, 1):
            print(f"\nRank {rank}:")
            print(f"  Expression: {result['expression']}")
            print(f"  RMSE: {result['rmse']:.6f}")
            print(f"  R² Score: {result['r2']:.6f}")
            print(f"  Coefficient: {result['coefficient']:.6f}")
            print(f"  Intercept: {result['intercept']:.6f}")
            print(f"  Data Coverage: {result['data_coverage']:.1%} "
                  f"({result['num_valid_points']}/{len(self.target_values)} points)")
            
            # Regression equation
            sign = '+' if result['intercept'] >= 0 else ''
            print(f"  Equation: Ea = {result['coefficient']:.6f} × (feature) {sign}{result['intercept']:.6f}")
        
        # Summary statistics
        print(f"\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        rmse_values = [r['rmse'] for r in top_results]
        r2_values = [r['r2'] for r in top_results]
        
        print(f"Best RMSE: {min(rmse_values):.6f}")
        print(f"Worst RMSE (in top {len(top_results)}): {max(rmse_values):.6f}")
        print(f"Average RMSE: {np.mean(rmse_values):.6f}")
        print(f"Best R² Score: {max(r2_values):.6f}")
        print(f"Average R² Score: {np.mean(r2_values):.6f}")


def main():
    """
    Main analysis function demonstrating the feature engineering workflow.
    """
    # Configuration parameters
    DATA_FILE_PATH = "train.txt"
    TARGET_COLUMN = "$E_a$"
    NUM_VARIABLES_TO_COMBINE = 3
    MAX_RESULTS_TO_SHOW = 10
    
    print("="*80)
    print("AUTOMATED FEATURE ENGINEERING FOR ACTIVATION ENERGY PREDICTION")
    print("="*80)
    
    # Initialize analyzer
    analyzer = FeatureEngineeringAnalyzer(
        data_file_path=DATA_FILE_PATH,
        target_column=TARGET_COLUMN
    )
    
    # Load and validate data
    if not analyzer.load_data():
        print("Failed to load data. Exiting.")
        return
    
    # Run comprehensive analysis
    top_results = analyzer.run_comprehensive_analysis(
        num_variables=NUM_VARIABLES_TO_COMBINE,
        max_results=MAX_RESULTS_TO_SHOW
    )
    
    if not top_results:
        print("No valid results found. Please check your data.")
        return
    
    # Display results
    analyzer.print_detailed_results(top_results)
    
    # Create visualizations
    analyzer.visualize_top_results(top_results)
    
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
