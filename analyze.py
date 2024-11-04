import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Data Loading & Initial Inspection
def load_and_inspect_data(filepath):
    # Load the dataset
    teams = pd.read_csv(filepath)
    print("Data Loaded Successfully.\n")
    
    # Display first few rows
    print("First 5 Rows of the Dataset:")
    print(teams.head(), "\n")
    
    # Summary of the dataset
    print("Dataset Information:")
    print(teams.info(), "\n")
    
    # Check for missing values
    print("Missing Values in Each Column:")
    print(teams.isnull().sum(), "\n")
    
    return teams

# 2. Descriptive Statistics
def descriptive_statistics(teams):
    print("Descriptive Statistics for Numerical Features:")
    print(teams.describe(), "\n")
    
    print("Descriptive Statistics for Categorical Features:")
    print(teams[['tmID', 'year', 'rank', 'playoff']].describe(), "\n")

# 3. Target Variable Analysis
def target_variable_analysis(teams):
    print("Distribution of Next Season Playoff:")
    playoff_counts = teams['next_season_playoff'].value_counts()
    print(playoff_counts, "\n")
    
    # Visualize the distribution
    sns.countplot(x='next_season_playoff', data=teams, palette='viridis')
    plt.title('Distribution of Next Season Playoff')
    plt.xlabel('Next Season Playoff (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.show()
    
    # Trend over years
    playoff_trend = teams.groupby('year')['next_season_playoff'].mean().reset_index()
    sns.lineplot(x='year', y='next_season_playoff', data=playoff_trend, marker='o')
    plt.title('Trend of Playoff Participation Over Years')
    plt.xlabel('Year')
    plt.ylabel('Proportion Making Playoffs')
    plt.ylim(0,1)
    plt.show()

# 4. Feature Distributions
def feature_distributions(teams, numerical_features):
    # Histograms
    teams[numerical_features].hist(bins=15, figsize=(20, 15), layout=(8, 6))
    plt.tight_layout()
    plt.suptitle('Histograms of Numerical Features', fontsize=20)
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    # Boxplots for selected features
    selected_features = ['won', 'lost', 'o_pts', 'd_pts', 'tmTRB', 'opptmTRB']
    teams_melted = teams.melt(id_vars='next_season_playoff', value_vars=selected_features)
    sns.boxplot(x='variable', y='value', hue='next_season_playoff', data=teams_melted)
    plt.title('Boxplots of Selected Features by Playoff Status')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.legend(title='Next Season Playoff')
    plt.show()

# 5. Correlation Analysis
def correlation_analysis(teams, numerical_features):
    # Compute correlation matrix
    corr_matrix = teams[numerical_features + ['next_season_playoff']].corr()
    
    # Heatmap of correlations
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()
    
    # Correlation with target variable
    target_corr = corr_matrix['next_season_playoff'].sort_values(ascending=False)
    print("Correlation of Features with Next Season Playoff:")
    print(target_corr, "\n")
    
    # Bar plot of correlations with target
    plt.figure(figsize=(12,8))
    sns.barplot(x=target_corr.index, y=target_corr.values, palette='viridis')
    plt.xticks(rotation=90)
    plt.title('Feature Correlation with Next Season Playoff')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.show()

# 6. Comparative Analysis
def comparative_analysis(teams, numerical_features):
    # Split data based on target
    playoff = teams[teams['next_season_playoff'] == 1]
    non_playoff = teams[teams['next_season_playoff'] == 0]
    
    # Compute means
    playoff_means = playoff[numerical_features].mean()
    non_playoff_means = non_playoff[numerical_features].mean()
    
    # Combine into a DataFrame
    comparison = pd.DataFrame({
        'Playoff': playoff_means,
        'Non-Playoff': non_playoff_means
    })
    
    # Plot the comparison
    comparison.plot(kind='bar', figsize=(15,7))
    plt.title('Comparison of Feature Means: Playoff vs Non-Playoff Teams')
    plt.xlabel('Features')
    plt.ylabel('Mean Values')
    plt.legend()
    plt.show()
    
    # Statistical Tests
    print("Statistical Tests (T-test) Between Playoff and Non-Playoff Teams:")
    for feature in numerical_features:
        stat, p = stats.ttest_ind(playoff[feature], non_playoff[feature], equal_var=False)
        print(f"{feature}: t-statistic = {stat:.2f}, p-value = {p:.4f}")
    print("\n")

# 7. Time-Series Analysis
def time_series_analysis(teams, features):
    # Aggregate by year
    yearly_stats = teams.groupby('year')[features].mean().reset_index()
    
    # Plot trends
    for feature in features:
        sns.lineplot(x='year', y=feature, data=yearly_stats, marker='o')
        plt.title(f'Trend of {feature} Over Years')
        plt.xlabel('Year')
        plt.ylabel(f'Average {feature}')
        plt.show()

# 8. Multicollinearity Check
def multicollinearity_check(teams, numerical_features):
    corr_matrix = teams[numerical_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than 0.8
    highly_correlated = [(column, row, upper.loc[row, column]) 
                         for column in upper.columns 
                         for row in upper.index 
                         if upper.loc[row, column] > 0.8]
    
    if highly_correlated:
        print("Highly Correlated Feature Pairs (Correlation > 0.8):")
        for pair in highly_correlated:
            print(f"{pair[0]} and {pair[1]}: Correlation = {pair[2]:.2f}")
    else:
        print("No highly correlated feature pairs found.\n")
    
    # Visualize the correlation matrix
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_matrix, annot=False, cmap='viridis')
    plt.title('Feature Correlation Matrix for Multicollinearity Check')
    plt.show()

# 9. Feature Importance (Preliminary)
def preliminary_feature_importance(teams, numerical_features):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Define features and target
    X = teams[numerical_features]
    y = teams['next_season_playoff']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3, 
                                                        random_state=42, 
                                                        stratify=y)
    
    # Initialize the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = pd.Series(rf.feature_importances_, index=numerical_features)
    importances = importances.sort_values(ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12,8))
    sns.barplot(x=importances.values, y=importances.index, palette='magma')
    plt.title('Preliminary Feature Importances from Random Forest')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()
    
    print("Top 10 Features by Importance:")
    print(importances.head(10), "\n")

# Main Function to Execute All Steps
def main():
    # Filepath to the dataset
    filepath = 'teams.csv'  # Ensure the CSV file is in the working directory
    
    # Step 1: Load and inspect data
    teams = load_and_inspect_data(filepath)
    
    # Step 2: Descriptive statistics
    descriptive_statistics(teams)
    
    # Step 3: Target variable analysis
    target_variable_analysis(teams)
    
    # Identify numerical features (excluding identifiers and categorical variables)
    exclude_cols = ['year', 'tmID', 'rank', 'playoff', 'next_season_playoff']
    numerical_features = [col for col in teams.columns if col not in exclude_cols]
    
    # Step 4: Feature distributions
    feature_distributions(teams, numerical_features)
    
    # Step 5: Correlation analysis
    correlation_analysis(teams, numerical_features)
    
    # Step 6: Comparative analysis
    comparative_analysis(teams, numerical_features)
    
    # Step 7: Time-series analysis for selected features
    time_series_features = ['won', 'lost', 'o_pts', 'd_pts', 'tmTRB', 'opptmTRB']
    time_series_analysis(teams, time_series_features)
    
    # Step 8: Multicollinearity check
    multicollinearity_check(teams, numerical_features)
    
    # Step 9: Preliminary feature importance
    preliminary_feature_importance(teams, numerical_features)
    
    print("EDA Completed Successfully.")

if __name__ == "__main__":
    main()
