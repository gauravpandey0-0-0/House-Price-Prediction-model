"""
House Price Prediction - Machine Learning Project
================================================

This project demonstrates a complete ML workflow for predicting house prices using:
1. Linear Regression
2. Random Forest Regressor  
3. Gradient Boosting Regressor

Author: ML Learning Project
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HousePricePredictor:
    """
    A comprehensive class for house price prediction using multiple ML algorithms
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_and_explore_data(self):
        """
        Load and explore the dataset
        """
        print("🏠 Loading and exploring house price data...")
        
        # Create synthetic house price data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic house features
        data = {
            'area': np.random.normal(1500, 500, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'age': np.random.randint(0, 50, n_samples),
            'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
            'garage': np.random.choice([0, 1, 2], n_samples),
            'pool': np.random.choice([0, 1], n_samples),
            'school_rating': np.random.uniform(1, 10, n_samples)
        }
        
        # Create realistic price based on features
        price = (
            data['area'] * 100 +  # Base price per sq ft
            data['bedrooms'] * 10000 +
            data['bathrooms'] * 15000 -
            data['age'] * 1000 +
            np.where(data['location'] == 'Urban', 50000, 
                    np.where(data['location'] == 'Suburban', 30000, 0)) +
            data['garage'] * 15000 +
            data['pool'] * 25000 +
            data['school_rating'] * 5000 +
            np.random.normal(0, 20000, n_samples)  # Random noise
        )
        
        data['price'] = np.maximum(price, 50000)  # Ensure minimum price
        
        self.df = pd.DataFrame(data)
        
        print(f"Dataset shape: {self.df.shape}")
        print("\n📊 Dataset Info:")
        print(self.df.info())
        print("\n📈 Dataset Statistics:")
        print(self.df.describe())
        
        return self.df
    
    def visualize_data(self):
        """
        Create visualizations to understand the data
        """
        print("\n📊 Creating data visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('House Price Dataset Analysis', fontsize=16, fontweight='bold')
        
        # Price distribution
        axes[0, 0].hist(self.df['price'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Area vs Price
        axes[0, 1].scatter(self.df['area'], self.df['price'], alpha=0.6, color='green')
        axes[0, 1].set_title('Area vs Price')
        axes[0, 1].set_xlabel('Area (sq ft)')
        axes[0, 1].set_ylabel('Price ($)')
        
        # Bedrooms vs Price
        bedroom_avg = self.df.groupby('bedrooms')['price'].mean()
        axes[0, 2].bar(bedroom_avg.index, bedroom_avg.values, color='orange')
        axes[0, 2].set_title('Average Price by Bedrooms')
        axes[0, 2].set_xlabel('Bedrooms')
        axes[0, 2].set_ylabel('Average Price ($)')
        
        # Location vs Price
        location_avg = self.df.groupby('location')['price'].mean()
        axes[1, 0].bar(location_avg.index, location_avg.values, color='red')
        axes[1, 0].set_title('Average Price by Location')
        axes[1, 0].set_xlabel('Location')
        axes[1, 0].set_ylabel('Average Price ($)')
        
        # Age vs Price
        axes[1, 1].scatter(self.df['age'], self.df['price'], alpha=0.6, color='purple')
        axes[1, 1].set_title('Age vs Price')
        axes[1, 1].set_xlabel('Age (years)')
        axes[1, 1].set_ylabel('Price ($)')
        
        # Correlation heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
        axes[1, 2].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('house_price_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def preprocess_data(self):
        """
        Preprocess the data for machine learning
        """
        print("\n🔧 Preprocessing data...")
        
        # Handle categorical variables
        categorical_cols = ['location']
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        
        # Separate features and target
        X = self.df.drop('price', axis=1)
        y = self.df['price']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.feature_names = X.columns.tolist()
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Features: {self.feature_names}")
        
    def train_models(self):
        """
        Train multiple machine learning models
        """
        print("\n🤖 Training machine learning models...")
        
        # Initialize models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train models
        for name, model in self.models.items():
            print(f"Training {name}...")
            if name == 'Linear Regression':
                model.fit(self.X_train_scaled, self.y_train)
            else:
                model.fit(self.X_train, self.y_train)
        
        print("✅ All models trained successfully!")
        
    def evaluate_models(self):
        """
        Evaluate all models using various metrics
        """
        print("\n📊 Evaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            # Make predictions
            if name == 'Linear Regression':
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }
            
            print(f"\n{name} Performance:")
            print(f"  MSE: {mse:,.2f}")
            print(f"  RMSE: {rmse:,.2f}")
            print(f"  MAE: {mae:,.2f}")
            print(f"  R²: {r2:.4f}")
        
        return results
    
    def cross_validate_models(self):
        """
        Perform cross-validation for all models
        """
        print("\n🔄 Performing cross-validation...")
        
        cv_results = {}
        
        for name, model in self.models.items():
            if name == 'Linear Regression':
                scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=5, scoring='neg_mean_squared_error')
            else:
                scores = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=5, scoring='neg_mean_squared_error')
            
            cv_results[name] = {
                'mean_cv_score': -scores.mean(),
                'std_cv_score': scores.std()
            }
            
            print(f"{name} CV RMSE: {-scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        
        return cv_results
    
    def optimize_models(self):
        """
        Optimize models using hyperparameter tuning
        """
        print("\n⚡ Optimizing models with hyperparameter tuning...")
        
        # Random Forest optimization
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_params,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rf_grid.fit(self.X_train, self.y_train)
        
        # Gradient Boosting optimization
        gb_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
        gb_grid = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            gb_params,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        gb_grid.fit(self.X_train, self.y_train)
        
        # Update models with optimized parameters
        self.models['Random Forest (Optimized)'] = rf_grid.best_estimator_
        self.models['Gradient Boosting (Optimized)'] = gb_grid.best_estimator_
        
        print(f"Random Forest best params: {rf_grid.best_params_}")
        print(f"Gradient Boosting best params: {gb_grid.best_params_}")
        
        return rf_grid, gb_grid
    
    def visualize_results(self, results):
        """
        Create visualizations comparing model performance
        """
        print("\n📈 Creating performance visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract metrics for plotting
        model_names = list(results.keys())
        mse_values = [results[name]['MSE'] for name in model_names]
        rmse_values = [results[name]['RMSE'] for name in model_names]
        mae_values = [results[name]['MAE'] for name in model_names]
        r2_values = [results[name]['R²'] for name in model_names]
        
        # MSE comparison
        axes[0, 0].bar(model_names, mse_values, color='lightcoral')
        axes[0, 0].set_title('Mean Squared Error Comparison')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[0, 1].bar(model_names, rmse_values, color='lightblue')
        axes[0, 1].set_title('Root Mean Squared Error Comparison')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[1, 0].bar(model_names, mae_values, color='lightgreen')
        axes[1, 0].set_title('Mean Absolute Error Comparison')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1, 1].bar(model_names, r2_values, color='gold')
        axes[1, 1].set_title('R² Score Comparison')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def feature_importance_analysis(self):
        """
        Analyze feature importance for tree-based models
        """
        print("\n🔍 Analyzing feature importance...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # Random Forest feature importance
        if 'Random Forest' in self.models:
            rf_importance = self.models['Random Forest'].feature_importances_
            axes[0].barh(self.feature_names, rf_importance, color='skyblue')
            axes[0].set_title('Random Forest Feature Importance')
            axes[0].set_xlabel('Importance')
        
        # Gradient Boosting feature importance
        if 'Gradient Boosting' in self.models:
            gb_importance = self.models['Gradient Boosting'].feature_importances_
            axes[1].barh(self.feature_names, gb_importance, color='lightcoral')
            axes[1].set_title('Gradient Boosting Feature Importance')
            axes[1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def predict_new_house(self, house_features):
        """
        Predict price for a new house
        """
        print("\n🏠 Predicting price for new house...")
        
        # Convert to DataFrame
        new_house = pd.DataFrame([house_features])
        
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            if col in new_house.columns:
                new_house[col] = le.transform(new_house[col])
        
        # Make predictions with all models
        predictions = {}
        
        for name, model in self.models.items():
            if name == 'Linear Regression':
                pred = model.predict(self.scaler.transform(new_house))
            else:
                pred = model.predict(new_house)
            predictions[name] = pred[0]
        
        print("Predicted prices:")
        for model, price in predictions.items():
            print(f"  {model}: ${price:,.2f}")
        
        return predictions
    
    def run_complete_analysis(self):
        """
        Run the complete machine learning pipeline
        """
        print("🚀 Starting Complete House Price Prediction Analysis")
        print("=" * 60)
        
        # Step 1: Load and explore data
        self.load_and_explore_data()
        
        # Step 2: Visualize data
        self.visualize_data()
        
        # Step 3: Preprocess data
        self.preprocess_data()
        
        # Step 4: Train models
        self.train_models()
        
        # Step 5: Evaluate models
        results = self.evaluate_models()
        
        # Step 6: Cross-validation
        cv_results = self.cross_validate_models()
        
        # Step 7: Optimize models
        self.optimize_models()
        
        # Step 8: Re-evaluate optimized models
        print("\n📊 Evaluating optimized models...")
        optimized_results = self.evaluate_models()
        
        # Step 9: Visualize results
        self.visualize_results(optimized_results)
        
        # Step 10: Feature importance analysis
        self.feature_importance_analysis()
        
        # Step 11: Example prediction
        example_house = {
            'area': 2000,
            'bedrooms': 3,
            'bathrooms': 2,
            'age': 5,
            'location': 'Urban',
            'garage': 2,
            'pool': 1,
            'school_rating': 8.5
        }
        self.predict_new_house(example_house)
        
        print("\n✅ Complete analysis finished!")
        print("=" * 60)
        
        return optimized_results

def main():
    """
    Main function to run the house price prediction project
    """
    # Create predictor instance
    predictor = HousePricePredictor()
    
    # Run complete analysis
    results = predictor.run_complete_analysis()
    
    # Print final summary
    print("\n📋 FINAL SUMMARY")
    print("=" * 40)
    print("Best performing models (by R² score):")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['R²'], reverse=True)
    for i, (model, metrics) in enumerate(sorted_results, 1):
        print(f"{i}. {model}: R² = {metrics['R²']:.4f}, RMSE = ${metrics['RMSE']:,.2f}")
    
    print(f"\n🎯 The best model is: {sorted_results[0][0]}")
    print(f"   R² Score: {sorted_results[0][1]['R²']:.4f}")
    print(f"   RMSE: ${sorted_results[0][1]['RMSE']:,.2f}")

if __name__ == "__main__":
    main()
