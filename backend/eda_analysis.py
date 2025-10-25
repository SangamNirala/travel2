"""
Comprehensive Exploratory Data Analysis (EDA) for Railway Dataset
This script performs complete data preprocessing, cleaning, feature engineering,
and generates visualizations and insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class RailwayEDA:
    def __init__(self, csv_path, output_dir='eda_results'):
        """
        Initialize EDA with dataset path and output directory
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.df = None
        self.df_cleaned = None
        self.insights = {}
        
    def load_data(self):
        """
        Load and inspect the dataset
        """
        print("=" * 80)
        print("STEP 1: DATA LOADING & INSPECTION")
        print("=" * 80)
        
        # Load dataset
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        
        print(f"\n✓ Dataset loaded successfully from: {self.csv_path}")
        print(f"✓ Total records: {len(self.df):,}")
        print(f"✓ Total columns: {len(self.df.columns)}")
        
        # Display first few rows
        print("\n📊 First 5 rows of the dataset:")
        print(self.df.head())
        
        # Data types
        print("\n📋 Column Data Types:")
        print(self.df.dtypes)
        
        # Basic info
        print("\n📈 Dataset Information:")
        print(f"Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values check
        print("\n🔍 Missing Values Check:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
        else:
            print("✓ No missing values found!")
        
        return self.df
    
    def parse_time(self, time_str):
        """
        Parse time string to timedelta
        """
        try:
            if pd.isna(time_str) or time_str == '00:00:00':
                return timedelta(0)
            parts = str(time_str).split(':')
            return timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2]))
        except:
            return timedelta(0)
    
    def clean_data(self):
        """
        Perform data cleaning operations
        """
        print("\n" + "=" * 80)
        print("STEP 2: DATA CLEANING")
        print("=" * 80)
        
        # Create a copy for cleaning
        self.df_cleaned = self.df.copy()
        
        # Convert Distance to numeric
        print("\n🔧 Converting Distance column to numeric...")
        self.df_cleaned['Distance'] = pd.to_numeric(self.df_cleaned['Distance'], errors='coerce')
        
        # Parse time columns
        print("🔧 Parsing time columns...")
        self.df_cleaned['Arrival_Time_Delta'] = self.df_cleaned['Arrival time'].apply(self.parse_time)
        self.df_cleaned['Departure_Time_Delta'] = self.df_cleaned['Departure Time'].apply(self.parse_time)
        
        # Clean station names (remove extra spaces, standardize)
        print("🔧 Standardizing station names and codes...")
        self.df_cleaned['Station Code'] = self.df_cleaned['Station Code'].str.strip().str.upper()
        self.df_cleaned['Station Name'] = self.df_cleaned['Station Name'].str.strip().str.title()
        self.df_cleaned['Source Station'] = self.df_cleaned['Source Station'].str.strip().str.upper()
        self.df_cleaned['Destination Station'] = self.df_cleaned['Destination Station'].str.strip().str.upper()
        
        # Check for invalid data
        invalid_distance = self.df_cleaned['Distance'].isna().sum()
        print(f"\n✓ Invalid distance entries: {invalid_distance}")
        
        # Remove rows with critical missing data
        initial_rows = len(self.df_cleaned)
        self.df_cleaned = self.df_cleaned.dropna(subset=['Distance', 'Station Code', 'Station Name'])
        final_rows = len(self.df_cleaned)
        print(f"✓ Rows removed due to critical missing data: {initial_rows - final_rows}")
        print(f"✓ Final cleaned dataset: {final_rows:,} rows")
        
        return self.df_cleaned
    
    def engineer_features(self):
        """
        Feature engineering: create new useful features
        """
        print("\n" + "=" * 80)
        print("STEP 3: FEATURE ENGINEERING")
        print("=" * 80)
        
        # Calculate travel duration at each station (stop time)
        print("\n🔨 Creating Travel_Duration feature...")
        self.df_cleaned['Travel_Duration'] = self.df_cleaned['Departure_Time_Delta'] - self.df_cleaned['Arrival_Time_Delta']
        self.df_cleaned['Travel_Duration_Minutes'] = self.df_cleaned['Travel_Duration'].dt.total_seconds() / 60
        
        # Handle negative durations (next day departure)
        negative_mask = self.df_cleaned['Travel_Duration_Minutes'] < 0
        self.df_cleaned.loc[negative_mask, 'Travel_Duration_Minutes'] += 24 * 60
        print(f"✓ Adjusted {negative_mask.sum()} negative durations (midnight crossing)")
        
        # Sort by train and sequence
        self.df_cleaned = self.df_cleaned.sort_values(['Train No', 'SEQ'])
        
        # Calculate journey metrics between consecutive stations
        print("🔨 Creating journey metrics between stations...")
        self.df_cleaned['Next_Station_Code'] = self.df_cleaned.groupby('Train No')['Station Code'].shift(-1)
        self.df_cleaned['Next_Station_Name'] = self.df_cleaned.groupby('Train No')['Station Name'].shift(-1)
        self.df_cleaned['Next_Distance'] = self.df_cleaned.groupby('Train No')['Distance'].shift(-1)
        self.df_cleaned['Next_Arrival_Time'] = self.df_cleaned.groupby('Train No')['Arrival_Time_Delta'].shift(-1)
        
        # Journey duration to next station
        self.df_cleaned['Journey_Duration'] = (self.df_cleaned['Next_Arrival_Time'] - self.df_cleaned['Departure_Time_Delta']).dt.total_seconds() / 60
        journey_negative_mask = self.df_cleaned['Journey_Duration'] < 0
        self.df_cleaned.loc[journey_negative_mask, 'Journey_Duration'] += 24 * 60
        
        # Distance between consecutive stations
        self.df_cleaned['Distance_To_Next'] = self.df_cleaned['Next_Distance'] - self.df_cleaned['Distance']
        
        # Calculate average speed (km/h) between stations
        self.df_cleaned['Avg_Speed_KmH'] = (self.df_cleaned['Distance_To_Next'] / (self.df_cleaned['Journey_Duration'] / 60))
        self.df_cleaned['Avg_Speed_KmH'] = self.df_cleaned['Avg_Speed_KmH'].replace([np.inf, -np.inf], np.nan)
        
        print(f"✓ Created {len(['Travel_Duration_Minutes', 'Journey_Duration', 'Distance_To_Next', 'Avg_Speed_KmH'])} new features")
        
        # Store key statistics
        self.insights['total_trains'] = self.df_cleaned['Train No'].nunique()
        self.insights['total_stations'] = self.df_cleaned['Station Code'].nunique()
        self.insights['total_routes'] = len(self.df_cleaned[self.df_cleaned['Next_Station_Code'].notna()])
        
        print(f"\n📊 Dataset Summary:")
        print(f"   - Unique Trains: {self.insights['total_trains']:,}")
        print(f"   - Unique Stations: {self.insights['total_stations']:,}")
        print(f"   - Total Route Segments: {self.insights['total_routes']:,}")
        
        return self.df_cleaned
    
    def perform_eda(self):
        """
        Perform exploratory data analysis with visualizations
        """
        print("\n" + "=" * 80)
        print("STEP 4: EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 80)
        
        # Calculate statistics
        print("\n📊 Statistical Summary:")
        
        # Distance statistics
        distance_stats = self.df_cleaned['Distance'].describe()
        print("\n🚄 Distance Statistics:")
        print(distance_stats)
        self.insights['avg_distance'] = distance_stats['mean']
        self.insights['max_distance'] = distance_stats['max']
        self.insights['min_distance'] = distance_stats['min']
        
        # Travel Duration statistics (stop time at station)
        valid_duration = self.df_cleaned[self.df_cleaned['Travel_Duration_Minutes'] > 0]['Travel_Duration_Minutes']
        if len(valid_duration) > 0:
            duration_stats = valid_duration.describe()
            print("\n⏱️  Stop Duration Statistics (minutes):")
            print(duration_stats)
            self.insights['avg_stop_duration'] = duration_stats['mean']
        
        # Journey Duration statistics (travel time between stations)
        valid_journey = self.df_cleaned[(self.df_cleaned['Journey_Duration'] > 0) & 
                                        (self.df_cleaned['Journey_Duration'] < 1440)]['Journey_Duration']
        if len(valid_journey) > 0:
            journey_stats = valid_journey.describe()
            print("\n🚂 Journey Duration Between Stations Statistics (minutes):")
            print(journey_stats)
            self.insights['avg_journey_duration'] = journey_stats['mean']
            self.insights['max_journey_duration'] = journey_stats['max']
        
        # Speed statistics
        valid_speed = self.df_cleaned[(self.df_cleaned['Avg_Speed_KmH'] > 0) & 
                                      (self.df_cleaned['Avg_Speed_KmH'] < 200)]['Avg_Speed_KmH']
        if len(valid_speed) > 0:
            speed_stats = valid_speed.describe()
            print("\n⚡ Average Speed Statistics (km/h):")
            print(speed_stats)
            self.insights['avg_speed'] = speed_stats['mean']
        
        # Most common routes
        print("\n🔝 Top 10 Source Stations:")
        top_sources = self.df_cleaned['Source Station'].value_counts().head(10)
        print(top_sources)
        self.insights['top_source'] = top_sources.index[0]
        
        print("\n🔝 Top 10 Destination Stations:")
        top_destinations = self.df_cleaned['Destination Station'].value_counts().head(10)
        print(top_destinations)
        self.insights['top_destination'] = top_destinations.index[0]
        
        # Generate all visualizations
        self.create_visualizations()
        
    def create_visualizations(self):
        """
        Create and save all visualizations
        """
        print("\n" + "=" * 80)
        print("STEP 5: GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # 1. Distribution of Distance
        print("\n📊 Creating: Distance Distribution plot...")
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(self.df_cleaned['Distance'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Distance (km)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Distance')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(self.df_cleaned['Distance'])
        plt.ylabel('Distance (km)')
        plt.title('Distance Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_distance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: 01_distance_distribution.png")
        
        # 2. Distribution of Travel Duration (Journey between stations)
        print("📊 Creating: Travel Duration Distribution plot...")
        valid_journey = self.df_cleaned[(self.df_cleaned['Journey_Duration'] > 0) & 
                                        (self.df_cleaned['Journey_Duration'] < 1440)]
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(valid_journey['Journey_Duration'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        plt.xlabel('Journey Duration (minutes)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Journey Duration Between Stations')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(valid_journey['Journey_Duration'])
        plt.ylabel('Journey Duration (minutes)')
        plt.title('Journey Duration Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_journey_duration_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: 02_journey_duration_distribution.png")
        
        # 3. Relationship between Distance and Journey Duration
        print("📊 Creating: Distance vs Journey Duration scatter plot...")
        valid_data = self.df_cleaned[(self.df_cleaned['Distance_To_Next'] > 0) & 
                                     (self.df_cleaned['Journey_Duration'] > 0) &
                                     (self.df_cleaned['Journey_Duration'] < 1440) &
                                     (self.df_cleaned['Distance_To_Next'] < 500)]
        
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(valid_data['Distance_To_Next'], valid_data['Journey_Duration'], 
                   alpha=0.5, color='coral', s=10)
        plt.xlabel('Distance to Next Station (km)')
        plt.ylabel('Journey Duration (minutes)')
        plt.title('Distance vs Journey Duration')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(valid_data['Distance_To_Next'], valid_data['Journey_Duration'], 1)
        p = np.poly1d(z)
        plt.plot(valid_data['Distance_To_Next'].sort_values(), 
                p(valid_data['Distance_To_Next'].sort_values()), 
                "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.kdeplot(data=valid_data, x='Distance_To_Next', y='Journey_Duration', 
                   cmap='YlOrRd', fill=True, levels=10)
        plt.xlabel('Distance to Next Station (km)')
        plt.ylabel('Journey Duration (minutes)')
        plt.title('Density Plot: Distance vs Journey Duration')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_distance_vs_duration.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: 03_distance_vs_duration.png")
        
        # 4. Top Source and Destination Stations
        print("📊 Creating: Top Stations bar charts...")
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Top 20 Source Stations
        top_sources = self.df_cleaned['Source Station'].value_counts().head(20)
        axes[0].barh(range(len(top_sources)), top_sources.values, color='steelblue')
        axes[0].set_yticks(range(len(top_sources)))
        axes[0].set_yticklabels(top_sources.index)
        axes[0].set_xlabel('Number of Trains')
        axes[0].set_title('Top 20 Source Stations', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Top 20 Destination Stations
        top_destinations = self.df_cleaned['Destination Station'].value_counts().head(20)
        axes[1].barh(range(len(top_destinations)), top_destinations.values, color='darkorange')
        axes[1].set_yticks(range(len(top_destinations)))
        axes[1].set_yticklabels(top_destinations.index)
        axes[1].set_xlabel('Number of Trains')
        axes[1].set_title('Top 20 Destination Stations', fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_top_stations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: 04_top_stations.png")
        
        # 5. Average Speed Analysis
        print("📊 Creating: Average Speed analysis plot...")
        valid_speed = self.df_cleaned[(self.df_cleaned['Avg_Speed_KmH'] > 0) & 
                                      (self.df_cleaned['Avg_Speed_KmH'] < 200)]
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(valid_speed['Avg_Speed_KmH'], bins=50, color='purple', edgecolor='black', alpha=0.7)
        plt.xlabel('Average Speed (km/h)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Average Speed Between Stations')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(valid_speed['Avg_Speed_KmH'])
        plt.ylabel('Average Speed (km/h)')
        plt.title('Average Speed Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_speed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: 05_speed_analysis.png")
        
        # 6. Correlation Heatmap
        print("📊 Creating: Correlation heatmap...")
        corr_data = self.df_cleaned[['Distance', 'Travel_Duration_Minutes', 
                                     'Journey_Duration', 'Distance_To_Next', 
                                     'Avg_Speed_KmH']].dropna()
        
        if len(corr_data) > 0:
            plt.figure(figsize=(10, 8))
            correlation_matrix = corr_data.corr()
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1)
            plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/06_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ Saved: 06_correlation_heatmap.png")
        
        # 7. Outlier Detection
        print("📊 Creating: Outlier detection plot...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distance outliers
        axes[0, 0].boxplot(self.df_cleaned['Distance'])
        axes[0, 0].set_ylabel('Distance (km)')
        axes[0, 0].set_title('Distance Outliers')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Journey Duration outliers
        axes[0, 1].boxplot(valid_journey['Journey_Duration'])
        axes[0, 1].set_ylabel('Journey Duration (minutes)')
        axes[0, 1].set_title('Journey Duration Outliers')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Speed outliers
        axes[1, 0].boxplot(valid_speed['Avg_Speed_KmH'])
        axes[1, 0].set_ylabel('Average Speed (km/h)')
        axes[1, 0].set_title('Speed Outliers')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distance to Next outliers
        valid_dist_next = self.df_cleaned[(self.df_cleaned['Distance_To_Next'] > 0) & 
                                          (self.df_cleaned['Distance_To_Next'] < 500)]
        axes[1, 1].boxplot(valid_dist_next['Distance_To_Next'])
        axes[1, 1].set_ylabel('Distance to Next (km)')
        axes[1, 1].set_title('Distance to Next Station Outliers')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_outlier_detection.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: 07_outlier_detection.png")
        
        # 8. Train count analysis
        print("📊 Creating: Train count analysis...")
        train_counts = self.df_cleaned.groupby('Train No').size().sort_values(ascending=False).head(20)
        
        plt.figure(figsize=(14, 6))
        plt.bar(range(len(train_counts)), train_counts.values, color='teal', alpha=0.7)
        plt.xlabel('Train Rank')
        plt.ylabel('Number of Stations')
        plt.title('Top 20 Trains by Number of Stations Covered', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/08_train_station_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: 08_train_station_coverage.png")
        
        print("\n✓ All visualizations created successfully!")
    
    def detect_outliers(self):
        """
        Detect and report outliers
        """
        print("\n" + "=" * 80)
        print("STEP 6: OUTLIER DETECTION")
        print("=" * 80)
        
        outliers_info = []
        
        # Distance outliers using IQR method
        Q1 = self.df_cleaned['Distance'].quantile(0.25)
        Q3 = self.df_cleaned['Distance'].quantile(0.75)
        IQR = Q3 - Q1
        distance_outliers = self.df_cleaned[(self.df_cleaned['Distance'] < (Q1 - 1.5 * IQR)) | 
                                            (self.df_cleaned['Distance'] > (Q3 + 1.5 * IQR))]
        outliers_info.append(f"Distance outliers: {len(distance_outliers)} records")
        
        # Journey Duration outliers
        valid_journey = self.df_cleaned[(self.df_cleaned['Journey_Duration'] > 0) & 
                                        (self.df_cleaned['Journey_Duration'] < 1440)]
        Q1 = valid_journey['Journey_Duration'].quantile(0.25)
        Q3 = valid_journey['Journey_Duration'].quantile(0.75)
        IQR = Q3 - Q1
        duration_outliers = valid_journey[(valid_journey['Journey_Duration'] < (Q1 - 1.5 * IQR)) | 
                                          (valid_journey['Journey_Duration'] > (Q3 + 1.5 * IQR))]
        outliers_info.append(f"Journey Duration outliers: {len(duration_outliers)} records")
        
        # Speed outliers
        valid_speed = self.df_cleaned[(self.df_cleaned['Avg_Speed_KmH'] > 0) & 
                                      (self.df_cleaned['Avg_Speed_KmH'] < 200)]
        Q1 = valid_speed['Avg_Speed_KmH'].quantile(0.25)
        Q3 = valid_speed['Avg_Speed_KmH'].quantile(0.75)
        IQR = Q3 - Q1
        speed_outliers = valid_speed[(valid_speed['Avg_Speed_KmH'] < (Q1 - 1.5 * IQR)) | 
                                     (valid_speed['Avg_Speed_KmH'] > (Q3 + 1.5 * IQR))]
        outliers_info.append(f"Speed outliers: {len(speed_outliers)} records")
        
        for info in outliers_info:
            print(f"   🔍 {info}")
        
        self.insights['outliers'] = outliers_info
        
    def save_outputs(self):
        """
        Save cleaned dataset and summary report
        """
        print("\n" + "=" * 80)
        print("STEP 7: SAVING OUTPUTS")
        print("=" * 80)
        
        # Save cleaned dataset
        cleaned_csv_path = f'{self.output_dir}/cleaned_railway_data.csv'
        self.df_cleaned.to_csv(cleaned_csv_path, index=False)
        print(f"\n✓ Cleaned dataset saved: {cleaned_csv_path}")
        print(f"   - Rows: {len(self.df_cleaned):,}")
        print(f"   - Columns: {len(self.df_cleaned.columns)}")
        
        # Generate summary report
        report_path = f'{self.output_dir}/eda_summary_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RAILWAY DATASET - EXPLORATORY DATA ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Records: {len(self.df_cleaned):,}\n")
            f.write(f"Total Columns: {len(self.df_cleaned.columns)}\n")
            f.write(f"Unique Trains: {self.insights['total_trains']:,}\n")
            f.write(f"Unique Stations: {self.insights['total_stations']:,}\n")
            f.write(f"Total Route Segments: {self.insights['total_routes']:,}\n\n")
            
            f.write("KEY INSIGHTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average Distance: {self.insights.get('avg_distance', 0):.2f} km\n")
            f.write(f"Maximum Distance: {self.insights.get('max_distance', 0):.2f} km\n")
            f.write(f"Minimum Distance: {self.insights.get('min_distance', 0):.2f} km\n\n")
            
            f.write(f"Average Stop Duration: {self.insights.get('avg_stop_duration', 0):.2f} minutes\n")
            f.write(f"Average Journey Duration: {self.insights.get('avg_journey_duration', 0):.2f} minutes\n")
            f.write(f"Maximum Journey Duration: {self.insights.get('max_journey_duration', 0):.2f} minutes\n\n")
            
            f.write(f"Average Speed: {self.insights.get('avg_speed', 0):.2f} km/h\n\n")
            
            f.write(f"Most Common Source Station: {self.insights.get('top_source', 'N/A')}\n")
            f.write(f"Most Common Destination Station: {self.insights.get('top_destination', 'N/A')}\n\n")
            
            f.write("OUTLIERS DETECTED\n")
            f.write("-" * 80 + "\n")
            for outlier in self.insights.get('outliers', []):
                f.write(f"   - {outlier}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("DATA QUALITY NOTES\n")
            f.write("=" * 80 + "\n")
            f.write("1. Time columns successfully parsed and converted to timedelta format\n")
            f.write("2. Midnight crossing scenarios handled correctly\n")
            f.write("3. Station names and codes standardized (uppercase, trimmed)\n")
            f.write("4. Feature engineering completed: Travel_Duration, Journey_Duration, Distance_To_Next, Avg_Speed\n")
            f.write("5. Outliers detected using IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)\n\n")
            
            f.write("GENERATED VISUALIZATIONS\n")
            f.write("-" * 80 + "\n")
            f.write("1. 01_distance_distribution.png - Distribution and box plot of distances\n")
            f.write("2. 02_journey_duration_distribution.png - Distribution of journey durations\n")
            f.write("3. 03_distance_vs_duration.png - Relationship between distance and duration\n")
            f.write("4. 04_top_stations.png - Top 20 source and destination stations\n")
            f.write("5. 05_speed_analysis.png - Average speed distribution and analysis\n")
            f.write("6. 06_correlation_heatmap.png - Feature correlation matrix\n")
            f.write("7. 07_outlier_detection.png - Box plots for outlier detection\n")
            f.write("8. 08_train_station_coverage.png - Trains by station coverage\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("NEXT STEPS\n")
            f.write("=" * 80 + "\n")
            f.write("1. Use cleaned dataset for Neural Network model training\n")
            f.write("2. Implement route optimization algorithms\n")
            f.write("3. Build web interface for railway path optimization\n")
            f.write("4. Integrate prediction models for travel time and distance\n\n")
            
            f.write("Report Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        
        print(f"✓ Summary report saved: {report_path}")
        
    def run_complete_eda(self):
        """
        Run the complete EDA pipeline
        """
        print("\n" + "🚂" * 40)
        print("RAILWAY DATASET - COMPREHENSIVE EDA ANALYSIS")
        print("🚂" * 40 + "\n")
        
        # Execute all steps
        self.load_data()
        self.clean_data()
        self.engineer_features()
        self.perform_eda()
        self.detect_outliers()
        self.save_outputs()
        
        print("\n" + "=" * 80)
        print("✅ EDA PHASE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📁 All outputs saved in: {self.output_dir}/")
        print(f"   - Cleaned dataset: cleaned_railway_data.csv")
        print(f"   - Summary report: eda_summary_report.txt")
        print(f"   - 8 visualization plots (.png files)")
        print("\n" + "🚂" * 40 + "\n")
        
        return self.insights


if __name__ == '__main__':
    # Initialize and run EDA
    eda = RailwayEDA(
        csv_path='Train_details_22122017.csv',
        output_dir='eda_results'
    )
    
    insights = eda.run_complete_eda()
    
    # Print key insights summary
    print("\n📊 KEY INSIGHTS SUMMARY:")
    print("-" * 80)
    print(f"✓ Total Trains: {insights['total_trains']:,}")
    print(f"✓ Total Stations: {insights['total_stations']:,}")
    print(f"✓ Average Journey Duration: {insights.get('avg_journey_duration', 0):.2f} minutes")
    print(f"✓ Average Speed: {insights.get('avg_speed', 0):.2f} km/h")
    print(f"✓ Most Common Source: {insights.get('top_source', 'N/A')}")
    print(f"✓ Most Common Destination: {insights.get('top_destination', 'N/A')}")
    print("-" * 80)
