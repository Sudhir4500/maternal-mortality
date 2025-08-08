import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to non-interactive
plt.switch_backend('Agg')

class MaternalMortalityTrendAnalysis:
    """Comprehensive Maternal Mortality Trend Analysis using ML"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.processed_df = None
        self.models = {}
        self.scalers = {}
        
    def load_and_preprocess_data(self, chunk_size=100000):
        """Load data in chunks and preprocess"""
        print("Loading data in chunks...")
        
        # Define key columns for maternal mortality analysis
        key_columns = [
            "Year of interview",
            "Province", 
            "Type of place of residence",
            "Respondent's current age",
            "Highest educational level",
            "Wealth index combined",
            "Total children ever born",
            "Sons who have died",
            "Daughters who have died", 
            "Births in last five years",
            "Births in past year",
            "Ever had a terminated pregnancy",
            "Pregnancy losses",
            "Currently pregnant",
            "Number of antenatal visits during pregnancy",
            "Place of delivery",
            "Delivery by caesarean section",
            "Self reported health status",
            "Age of respondent at 1st birth",
            "Source of drinking water",
            "Type of toilet facility",
            "Household has: electricity",
            "Household has: radio",
            "Household has: television",
            "Religion",
            "Ethnicity"
        ]
        
        try:
            # Read data in chunks
            chunks = []
            for chunk in pd.read_csv(self.file_path, chunksize=chunk_size):
                # Select available columns
                available_cols = [col for col in key_columns if col in chunk.columns]
                chunk_subset = chunk[available_cols].copy()
                chunks.append(chunk_subset)
                
            self.df = pd.concat(chunks, ignore_index=True)
            print(f"Loaded data with shape: {self.df.shape}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Fallback to loading with basic columns
            basic_columns = [
                "Year of interview", "Province", "Type of place of residence",
                "Total children ever born", "Sons who have died", "Daughters who have died",
                "Births in last five years", "Births in past year", 
                "Ever had a terminated pregnancy", "Pregnancy losses"
            ]
            
            chunks = []
            for chunk in pd.read_csv(self.file_path, chunksize=chunk_size):
                available_cols = [col for col in basic_columns if col in chunk.columns]
                chunk_subset = chunk[available_cols].copy()
                chunks.append(chunk_subset)
                
            self.df = pd.concat(chunks, ignore_index=True)
            print(f"Loaded basic data with shape: {self.df.shape}")
            
    def create_mortality_risk_indicators(self):
        """Create maternal mortality risk indicators"""
        print("Creating maternal mortality risk indicators...")
        
        # Create mortality risk proxy variables
        self.df['child_deaths'] = (self.df.get('Sons who have died', 0).fillna(0) + 
                                  self.df.get('Daughters who have died', 0).fillna(0))
        
        self.df['pregnancy_losses'] = self.df.get('Pregnancy losses', 0).fillna(0)
        
        # Maternal mortality risk score (composite indicator)
        self.df['mortality_risk_score'] = (
            self.df['child_deaths'] * 0.3 + 
            self.df['pregnancy_losses'] * 0.7
        )
        
        # High risk pregnancy indicator
        self.df['high_risk_pregnancy'] = (
            (self.df['pregnancy_losses'] > 0) | 
            (self.df['child_deaths'] > 2)
        ).astype(int)
        
        # Birth complications proxy
        if 'Ever had a terminated pregnancy' in self.df.columns:
            self.df['birth_complications'] = self.df['Ever had a terminated pregnancy'].fillna(0)
        else:
            self.df['birth_complications'] = 0
            
        print("Created mortality risk indicators")
        
    def temporal_trend_analysis(self):
        """Analyze temporal trends in maternal mortality indicators"""
        print("Performing temporal trend analysis...")
        
        # Group by year for trend analysis
        if 'Year of interview' in self.df.columns:
            agg_dict = {
                'mortality_risk_score': ['mean', 'sum', 'count'],
                'high_risk_pregnancy': ['mean', 'sum'],
                'pregnancy_losses': ['mean', 'sum'],
                'child_deaths': ['mean', 'sum']
            }
            
            if 'birth_complications' in self.df.columns:
                agg_dict['birth_complications'] = 'mean'
                
            yearly_trends = self.df.groupby('Year of interview').agg(agg_dict).round(4)
            
            yearly_trends.columns = ['_'.join(col).strip() for col in yearly_trends.columns]
            
            # Plot temporal trends
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Maternal Mortality Risk Trends Over Time', fontsize=16)
            
            # Mortality risk score trend
            axes[0,0].plot(yearly_trends.index, yearly_trends['mortality_risk_score_mean'], 'b-', linewidth=2)
            axes[0,0].set_title('Average Mortality Risk Score by Year')
            axes[0,0].set_xlabel('Year')
            axes[0,0].set_ylabel('Risk Score')
            axes[0,0].grid(True, alpha=0.3)
            
            # High risk pregnancy percentage
            axes[0,1].plot(yearly_trends.index, yearly_trends['high_risk_pregnancy_mean'] * 100, 'r-', linewidth=2)
            axes[0,1].set_title('High Risk Pregnancy Percentage by Year')
            axes[0,1].set_xlabel('Year')
            axes[0,1].set_ylabel('Percentage (%)')
            axes[0,1].grid(True, alpha=0.3)
            
            # Pregnancy losses trend
            axes[1,0].plot(yearly_trends.index, yearly_trends['pregnancy_losses_mean'], 'g-', linewidth=2)
            axes[1,0].set_title('Average Pregnancy Losses by Year')
            axes[1,0].set_xlabel('Year')
            axes[1,0].set_ylabel('Pregnancy Losses')
            axes[1,0].grid(True, alpha=0.3)
            
            # Birth complications trend (if available)
            if 'birth_complications' in yearly_trends.columns:
                axes[1,1].plot(yearly_trends.index, yearly_trends['birth_complications'], 'orange', linewidth=2)
                axes[1,1].set_title('Birth Complications Rate by Year')
                axes[1,1].set_xlabel('Year')
                axes[1,1].set_ylabel('Complication Rate')
                axes[1,1].grid(True, alpha=0.3)
            else:
                # Plot alternative metric
                axes[1,1].plot(yearly_trends.index, yearly_trends['mortality_risk_score_count'], 'purple', linewidth=2)
                axes[1,1].set_title('Number of Records by Year')
                axes[1,1].set_xlabel('Year')
                axes[1,1].set_ylabel('Count')
                axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('maternal_mortality_temporal_trends.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close instead of show
            print("Temporal trends plot saved as 'maternal_mortality_temporal_trends.png'")
            
            return yearly_trends
        else:
            print("Year of interview column not found")
            return None
            
    def regional_analysis(self):
        """Analyze regional patterns in maternal mortality"""
        print("Performing regional analysis...")
        
        if 'Province' in self.df.columns:
            regional_stats = self.df.groupby('Province').agg({
                'mortality_risk_score': ['mean', 'std', 'count'],
                'high_risk_pregnancy': 'mean',
                'pregnancy_losses': 'mean',
                'child_deaths': 'mean'
            }).round(4)
            
            regional_stats.columns = ['_'.join(col).strip() for col in regional_stats.columns]
            
            # Plot regional comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Regional Maternal Mortality Risk Analysis', fontsize=16)
            
            # Regional mortality risk scores
            provinces = regional_stats.index
            risk_scores = regional_stats['mortality_risk_score_mean']
            
            axes[0,0].bar(range(len(provinces)), risk_scores, color='skyblue', alpha=0.7)
            axes[0,0].set_title('Average Mortality Risk Score by Province')
            axes[0,0].set_xlabel('Province')
            axes[0,0].set_ylabel('Risk Score')
            axes[0,0].set_xticks(range(len(provinces)))
            axes[0,0].set_xticklabels(provinces, rotation=45)
            
            # High risk pregnancy rates
            if 'high_risk_pregnancy_mean' in regional_stats.columns:
                high_risk_rates = regional_stats['high_risk_pregnancy_mean'] * 100
            else:
                high_risk_rates = regional_stats['high_risk_pregnancy'] * 100
            axes[0,1].bar(range(len(provinces)), high_risk_rates, color='lightcoral', alpha=0.7)
            axes[0,1].set_title('High Risk Pregnancy Rate by Province (%)')
            axes[0,1].set_xlabel('Province')
            axes[0,1].set_ylabel('Percentage (%)')
            axes[0,1].set_xticks(range(len(provinces)))
            axes[0,1].set_xticklabels(provinces, rotation=45)
            
            # Pregnancy losses
            if 'pregnancy_losses_mean' in regional_stats.columns:
                pregnancy_losses = regional_stats['pregnancy_losses_mean']
            else:
                pregnancy_losses = regional_stats['pregnancy_losses']
            axes[1,0].bar(range(len(provinces)), pregnancy_losses, color='lightgreen', alpha=0.7)
            axes[1,0].set_title('Average Pregnancy Losses by Province')
            axes[1,0].set_xlabel('Province')
            axes[1,0].set_ylabel('Pregnancy Losses')
            axes[1,0].set_xticks(range(len(provinces)))
            axes[1,0].set_xticklabels(provinces, rotation=45)
            
            # Child deaths
            if 'child_deaths_mean' in regional_stats.columns:
                child_deaths = regional_stats['child_deaths_mean']
            else:
                child_deaths = regional_stats['child_deaths']
            axes[1,1].bar(range(len(provinces)), child_deaths, color='gold', alpha=0.7)
            axes[1,1].set_title('Average Child Deaths by Province')
            axes[1,1].set_xlabel('Province')
            axes[1,1].set_ylabel('Child Deaths')
            axes[1,1].set_xticks(range(len(provinces)))
            axes[1,1].set_xticklabels(provinces, rotation=45)
            
            plt.tight_layout()
            plt.savefig('regional_maternal_mortality_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close instead of show
            print("Regional analysis plot saved as 'regional_maternal_mortality_analysis.png'")
            
            return regional_stats
        else:
            print("Province column not found")
            return None
            
    def predictive_modeling(self):
        """Build ML models to predict maternal mortality trends"""
        print("Building predictive models...")
        
        # Prepare features for modeling
        feature_columns = []
        
        # Add available numeric features
        numeric_features = ['Total children ever born', 'Births in last five years', 
                           'Births in past year', 'child_deaths']
        
        for feat in numeric_features:
            if feat in self.df.columns:
                feature_columns.append(feat)
                
        # Add year if available
        if 'Year of interview' in self.df.columns:
            feature_columns.append('Year of interview')
            
        # Encode categorical features
        categorical_features = ['Province', 'Type of place of residence']
        label_encoders = {}
        
        for feat in categorical_features:
            if feat in self.df.columns:
                le = LabelEncoder()
                self.df[f'{feat}_encoded'] = le.fit_transform(self.df[feat].fillna('Unknown'))
                feature_columns.append(f'{feat}_encoded')
                label_encoders[feat] = le
                
        # Prepare target variables
        targets = ['mortality_risk_score', 'pregnancy_losses', 'high_risk_pregnancy']
        
        results = {}
        
        for target in targets:
            if target in self.df.columns:
                print(f"\nBuilding model for {target}...")
                
                # Prepare data
                X = self.df[feature_columns].fillna(0)
                y = self.df[target].fillna(0)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train multiple models
                models = {
                    'Linear Regression': LinearRegression(),
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
                }
                
                target_results = {}
                
                for model_name, model in models.items():
                    if model_name == 'Linear Regression':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    target_results[model_name] = {
                        'MSE': mse,
                        'RMSE': rmse,
                        'R2': r2,
                        'MAE': mae,
                        'predictions': y_pred[:100]  # Store first 100 predictions
                    }
                    
                    print(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
                
                results[target] = target_results
                
        return results
        
    def clustering_analysis(self):
        """Perform clustering to identify risk groups"""
        print("Performing clustering analysis...")
        
        # Select features for clustering
        cluster_features = ['mortality_risk_score', 'pregnancy_losses', 'child_deaths']
        
        # Check available features
        available_features = [feat for feat in cluster_features if feat in self.df.columns]
        
        if len(available_features) >= 2:
            X_cluster = self.df[available_features].fillna(0)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_cluster)
            
            self.df['risk_cluster'] = clusters
            
            # Analyze clusters
            cluster_analysis = self.df.groupby('risk_cluster')[available_features].mean()
            
            print("\nCluster Analysis:")
            print(cluster_analysis)
            
            # Plot clusters
            if len(available_features) >= 2:
                plt.figure(figsize=(10, 6))
                scatter = plt.scatter(X_cluster.iloc[:, 0], X_cluster.iloc[:, 1], 
                                    c=clusters, cmap='viridis', alpha=0.6)
                plt.xlabel(available_features[0])
                plt.ylabel(available_features[1])
                plt.title('Maternal Mortality Risk Clusters')
                plt.colorbar(scatter)
                plt.savefig('risk_clusters.png', dpi=300, bbox_inches='tight')
                plt.close()  # Close instead of show
                print("Risk clusters plot saved as 'risk_clusters.png'")
                
            return cluster_analysis
        else:
            print("Insufficient features for clustering")
            return None
            
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n" + "="*60)
        print("MATERNAL MORTALITY TREND ANALYSIS REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"\nDataset Overview:")
        print(f"Total records: {len(self.df):,}")
        print(f"Date range: {self.df.get('Year of interview', pd.Series()).min()} - {self.df.get('Year of interview', pd.Series()).max()}")
        
        if 'Province' in self.df.columns:
            print(f"Number of provinces: {self.df['Province'].nunique()}")
            
        # Risk indicators summary
        print(f"\nMaternal Mortality Risk Indicators:")
        print(f"Average mortality risk score: {self.df['mortality_risk_score'].mean():.4f}")
        print(f"High-risk pregnancies: {self.df['high_risk_pregnancy'].mean()*100:.2f}%")
        print(f"Average pregnancy losses: {self.df['pregnancy_losses'].mean():.4f}")
        print(f"Average child deaths: {self.df['child_deaths'].mean():.4f}")
        
        # Trends summary
        if 'Year of interview' in self.df.columns:
            yearly_stats = self.df.groupby('Year of interview')['mortality_risk_score'].mean()
            trend_direction = "increasing" if yearly_stats.iloc[-1] > yearly_stats.iloc[0] else "decreasing"
            print(f"\nTrend Direction: Mortality risk is {trend_direction} over time")
            
        print("\n" + "="*60)
        
    def run_complete_analysis(self):
        """Run the complete maternal mortality trend analysis"""
        print("Starting Comprehensive Maternal Mortality Trend Analysis...")
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Create mortality indicators
        self.create_mortality_risk_indicators()
        
        # Perform temporal analysis
        temporal_results = self.temporal_trend_analysis()
        
        # Perform regional analysis
        regional_results = self.regional_analysis()
        
        # Build predictive models
        model_results = self.predictive_modeling()
        
        # Perform clustering
        cluster_results = self.clustering_analysis()
        
        # Generate insights report
        self.generate_insights_report()
        
        print("\nAnalysis complete! Check the generated plots and results.")
        
        return {
            'temporal_trends': temporal_results,
            'regional_analysis': regional_results,
            'model_results': model_results,
            'cluster_analysis': cluster_results
        }

# Usage
if __name__ == "__main__":
    # Initialize analysis
    file_path = r"C:\Users\acer\Desktop\maternal mortality\NPGR82FL_output.csv"
    analyzer = MaternalMortalityTrendAnalysis(file_path)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
