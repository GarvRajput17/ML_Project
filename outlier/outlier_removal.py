import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class OutlierRemoval:
    """IQR-based outlier removal for Patient Recovery Dataset"""
    
    def __init__(self, iqr_multiplier=1.5):
        """
        Initialize outlier removal with IQR method
        
        Args:
            iqr_multiplier (float): Multiplier for IQR to determine outlier bounds
                                  Default: 1.5 (standard), 3.0 (extreme outliers)
        """
        self.iqr_multiplier = iqr_multiplier
        self.outlier_info = {}
        self.original_shape = None
        self.cleaned_shape = None
        
    def load_data(self, train_path, test_path):
        """Load training and test datasets"""
        print("Loading datasets for outlier analysis...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        self.original_shape = self.train_df.shape
        print(f"Original training data shape: {self.original_shape}")
        print(f"Test data shape: {self.test_df.shape}")
        
        return self.train_df, self.test_df
    
    def detect_outliers_iqr(self, df, columns=None):
        """
        Detect outliers using IQR method
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): Columns to check for outliers. If None, uses all numeric columns
            
        Returns:
            dict: Outlier information for each column
        """
        if columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove 'Id' column if present
            if 'Id' in numeric_columns:
                numeric_columns.remove('Id')
        else:
            numeric_columns = columns
            
        outlier_info = {}
        
        for col in numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - self.iqr_multiplier * IQR
                upper_bound = Q3 + self.iqr_multiplier * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                outlier_info[col] = {
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(df)) * 100,
                    'outlier_indices': outliers.index.tolist()
                }
        
        return outlier_info
    
    def remove_outliers_iqr(self, df, columns=None, inplace=False):
        """
        Remove outliers using IQR method
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): Columns to check for outliers
            inplace (bool): Whether to modify dataframe in place
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        if not inplace:
            df_clean = df.copy()
        else:
            df_clean = df
            
        if columns is None:
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            # Remove 'Id' column if present
            if 'Id' in numeric_columns:
                numeric_columns.remove('Id')
        else:
            numeric_columns = columns
        
        outlier_info = self.detect_outliers_iqr(df_clean, numeric_columns)
        self.outlier_info = outlier_info
        
        # Get all outlier indices
        all_outlier_indices = set()
        for col_info in outlier_info.values():
            all_outlier_indices.update(col_info['outlier_indices'])
        
        # Remove rows with outliers
        df_clean = df_clean.drop(index=list(all_outlier_indices))
        
        print(f"Outlier removal summary:")
        print(f"  Original samples: {len(df)}")
        print(f"  Outliers detected: {len(all_outlier_indices)}")
        print(f"  Cleaned samples: {len(df_clean)}")
        print(f"  Samples removed: {len(df) - len(df_clean)} ({((len(df) - len(df_clean)) / len(df)) * 100:.2f}%)")
        
        return df_clean
    
    def analyze_outliers(self, df, columns=None):
        """Analyze and visualize outliers"""
        print("\n" + "="*60)
        print("OUTLIER ANALYSIS")
        print("="*60)
        
        if columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Id' in numeric_columns:
                numeric_columns.remove('Id')
        else:
            numeric_columns = columns
        
        outlier_info = self.detect_outliers_iqr(df, numeric_columns)
        
        print("Outlier Detection Results:")
        print("-" * 40)
        for col, info in outlier_info.items():
            print(f"\n{col}:")
            print(f"  Q1: {info['Q1']:.4f}, Q3: {info['Q3']:.4f}, IQR: {info['IQR']:.4f}")
            print(f"  Bounds: [{info['lower_bound']:.4f}, {info['upper_bound']:.4f}]")
            print(f"  Outliers: {info['outlier_count']} ({info['outlier_percentage']:.2f}%)")
        
        return outlier_info
    
    def create_outlier_visualizations(self, df, columns=None, save_plots=True):
        """Create visualizations for outlier analysis"""
        print("\nCreating outlier visualizations...")
        
        if columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Id' in numeric_columns:
                numeric_columns.remove('Id')
        else:
            numeric_columns = columns
        
        # Create subplots
        n_cols = min(3, len(numeric_columns))
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_columns):
            if i < len(axes):
                # Box plot
                axes[i].boxplot(df[col].dropna(), patch_artist=True)
                axes[i].set_title(f'{col} - Outlier Detection')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(numeric_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('/Users/garvrajput/StudioProjects/ML PROJ/outlier_analysis.png', 
                       dpi=300, bbox_inches='tight')
            print("Outlier analysis plot saved as 'outlier_analysis.png'")
        
        plt.show()
        
        # Create before/after comparison if outliers were removed
        if hasattr(self, 'outlier_info') and self.outlier_info:
            self._create_before_after_plot(df, save_plots)
    
    def _create_before_after_plot(self, original_df, save_plots=True):
        """Create before/after comparison plot"""
        print("Creating before/after comparison...")
        
        # This would be called after outlier removal
        # For now, just create a summary plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Before removal
        axes[0].hist(original_df['Recovery Index'], bins=30, alpha=0.7, color='red', label='Original')
        axes[0].set_title('Before Outlier Removal')
        axes[0].set_xlabel('Recovery Index')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Summary statistics
        stats_text = f"""
        Original Data:
        Mean: {original_df['Recovery Index'].mean():.2f}
        Std: {original_df['Recovery Index'].std():.2f}
        Min: {original_df['Recovery Index'].min():.2f}
        Max: {original_df['Recovery Index'].max():.2f}
        """
        axes[1].text(0.1, 0.5, stats_text, transform=axes[1].transAxes, 
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1].set_title('Data Statistics')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('/Users/garvrajput/StudioProjects/ML PROJ/outlier_before_after.png', 
                       dpi=300, bbox_inches='tight')
            print("Before/after comparison saved as 'outlier_before_after.png'")
        
        plt.show()
    
    def save_cleaned_data(self, train_clean, test_df, output_dir='.'):
        """Save cleaned datasets"""
        print(f"\nSaving cleaned data to {output_dir}...")
        
        # Save cleaned training data
        train_clean.to_csv(f'{output_dir}/train_cleaned.csv', index=False)
        print("Cleaned training data saved as 'train_cleaned.csv'")
        
        # Test data remains unchanged (no outliers removed from test set)
        test_df.to_csv(f'{output_dir}/test_cleaned.csv', index=False)
        print("Test data saved as 'test_cleaned.csv' (unchanged)")
        
        # Save outlier removal summary
        summary = {
            'original_shape': self.original_shape,
            'cleaned_shape': train_clean.shape,
            'samples_removed': self.original_shape[0] - train_clean.shape[0],
            'removal_percentage': ((self.original_shape[0] - train_clean.shape[0]) / self.original_shape[0]) * 100,
            'iqr_multiplier': self.iqr_multiplier,
            'outlier_info': self.outlier_info
        }
        
        import json
        with open(f'{output_dir}/outlier_removal_summary.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            summary_clean = {k: convert_numpy(v) for k, v in summary.items()}
            json.dump(summary_clean, f, indent=2)
        
        print("Outlier removal summary saved as 'outlier_removal_summary.json'")
        
        return train_clean, test_df
    
    def get_summary(self):
        """Get comprehensive outlier removal summary"""
        print("\n" + "="*60)
        print("OUTLIER REMOVAL SUMMARY")
        print("="*60)
        
        if self.original_shape is None:
            print("No data processed yet!")
            return
        
        print(f"Original data shape: {self.original_shape}")
        print(f"Cleaned data shape: {self.cleaned_shape}")
        print(f"Samples removed: {self.original_shape[0] - self.cleaned_shape[0]}")
        print(f"Removal percentage: {((self.original_shape[0] - self.cleaned_shape[0]) / self.original_shape[0]) * 100:.2f}%")
        print(f"IQR multiplier used: {self.iqr_multiplier}")
        
        if self.outlier_info:
            print(f"\nOutlier details by column:")
            for col, info in self.outlier_info.items():
                print(f"  {col}: {info['outlier_count']} outliers ({info['outlier_percentage']:.2f}%)")
        
        return {
            'original_shape': self.original_shape,
            'cleaned_shape': self.cleaned_shape,
            'samples_removed': self.original_shape[0] - self.cleaned_shape[0],
            'removal_percentage': ((self.original_shape[0] - self.cleaned_shape[0]) / self.original_shape[0]) * 100,
            'iqr_multiplier': self.iqr_multiplier,
            'outlier_info': self.outlier_info
        }

def main():
    """Main function to run outlier removal"""
    print("Starting IQR-based Outlier Removal...")
    print("="*60)
    
    # Initialize outlier removal
    outlier_remover = OutlierRemoval(iqr_multiplier=1.5)  # Standard IQR multiplier
    
    # Load data
    train_df, test_df = outlier_remover.load_data(
        '/Users/garvrajput/StudioProjects/ML PROJ/train.csv',
        '/Users/garvrajput/StudioProjects/ML PROJ/test.csv'
    )
    
    # Analyze outliers before removal
    outlier_info = outlier_remover.analyze_outliers(train_df)
    
    # Create visualizations
    outlier_remover.create_outlier_visualizations(train_df)
    
    # Remove outliers
    print("\n" + "="*50)
    print("REMOVING OUTLIERS")
    print("="*50)
    train_clean = outlier_remover.remove_outliers_iqr(train_df)
    outlier_remover.cleaned_shape = train_clean.shape
    
    # Save cleaned data
    train_clean, test_clean = outlier_remover.save_cleaned_data(train_clean, test_df)
    
    # Get summary
    summary = outlier_remover.get_summary()
    
    print("\nOutlier removal completed!")
    print("Files created:")
    print("- train_cleaned.csv (cleaned training data)")
    print("- test_cleaned.csv (unchanged test data)")
    print("- outlier_removal_summary.json (removal details)")
    print("- outlier_analysis.png (outlier visualizations)")
    print("- outlier_before_after.png (before/after comparison)")
    
    return outlier_remover, summary

if __name__ == "__main__":
    remover, summary = main()
