# Health Tracker Dataset Analysis Script
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings("ignore")

# Safe imports
def try_import_module(name):
    try:
        module = __import__(name)
        return module
    except Exception:
        return None

scipy = try_import_module('scipy')
stats = scipy.stats if scipy is not None else None

sklearn = try_import_module('sklearn')
if sklearn is not None:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error

statsmodels = try_import_module('statsmodels')
if statsmodels is not None:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tsa.stattools import adfuller

# Load CSV - using local dataset path
dataset_path = './Dataset/raw.csv'
if not os.path.exists(dataset_path):
    print(f"Dataset not found at {dataset_path}. Checking for processed.csv...")
    dataset_path = './Dataset/processed.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No dataset found in ./Dataset/ folder.")

# Load data with proper NA handling
df = pd.read_csv(dataset_path, na_values=['NA', 'N/A', '', 'nan', 'NaN'])
nrows, ncols = df.shape
print(f"Loaded '{dataset_path}' -> rows: {nrows}, cols: {ncols}")

print("\nFirst 20 rows of the dataset:")
print(df.head(20))

print("\nColumns and dtypes:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isna().sum())

# Try to detect datetime-like columns and convert
datetime_cols = []
for col in df.columns:
    if col.lower() == 'timestamp':  # Explicitly handle Timestamp column
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            datetime_cols.append(col)
        except Exception:
            pass
    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        datetime_cols.append(col)

if datetime_cols:
    print("\nDetected datetime-like columns:", datetime_cols)
    dt_col = datetime_cols[0]
    df = df.sort_values(dt_col)
    df_indexed = df.set_index(dt_col)
else:
    df_indexed = None

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric columns detected:", numeric_cols)

if numeric_cols:
    print("\nSummary statistics (numeric):")
    print(df[numeric_cols].describe().T)
else:
    print("\nNo numeric columns found (besides Age). This appears to be a categorical dataset.")

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'Timestamp' in categorical_cols:
    categorical_cols.remove('Timestamp')  # Remove timestamp from categorical analysis
print(f"\nCategorical columns detected ({len(categorical_cols)}): {categorical_cols[:10]}{'...' if len(categorical_cols) > 10 else ''}")

# Add Age as numeric if it exists
if 'Age' in df.columns:
    try:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        numeric_cols.append('Age')
        print(f"\nConverted Age to numeric. Age range: {df['Age'].min()}-{df['Age'].max()}")
    except Exception as e:
        print(f"Could not convert Age to numeric: {e}")

# Create output directory in static folder
out_dir = "./static"
os.makedirs(out_dir, exist_ok=True)

# Set matplotlib backend for better compatibility
plt.style.use('default')

# Categorical Data Analysis
if categorical_cols:
    # Value counts for key categorical variables
    key_categorical = categorical_cols[:8]  # Analyze first 8 categorical columns
    for col in key_categorical:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            value_counts = df[col].value_counts().head(15)  # Top 15 values
            value_counts.plot(kind='bar')
            plt.title(f'Distribution of {col}', fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            fname = os.path.join(out_dir, f"bar_{col.replace(' ', '_').replace('/', '_')}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved bar chart for {col}")
    
    # Treatment analysis (if exists)
    if 'treatment' in df.columns:
        plt.figure(figsize=(8, 6))
        treatment_counts = df['treatment'].value_counts()
        colors = ['#ff9999', '#66b3ff']
        plt.pie(treatment_counts.values, labels=treatment_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Treatment Distribution', fontsize=14)
        fname = os.path.join(out_dir, "treatment_pie_chart.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved treatment pie chart")
    
    # Gender analysis
    if 'Gender' in df.columns:
        plt.figure(figsize=(8, 6))
        gender_counts = df['Gender'].value_counts()
        colors = ['#ffcc99', '#ff99cc', '#99ff99', '#99ccff']
        plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
                colors=colors[:len(gender_counts)], startangle=90)
        plt.title('Gender Distribution', fontsize=14)
        fname = os.path.join(out_dir, "gender_pie_chart.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved gender pie chart")
    
    # Country analysis (top countries)
    if 'Country' in df.columns:
        plt.figure(figsize=(12, 6))
        country_counts = df['Country'].value_counts().head(10)
        country_counts.plot(kind='bar')
        plt.title('Top 10 Countries', fontsize=14)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        fname = os.path.join(out_dir, "countries_bar_chart.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved countries bar chart")

# Treatment vs other variables analysis
if 'treatment' in df.columns and categorical_cols:
    key_vars = ['Gender', 'family_history', 'work_interfere', 'benefits', 'seek_help']
    for var in key_vars:
        if var in df.columns:
            plt.figure(figsize=(10, 6))
            cross_tab = pd.crosstab(df[var], df['treatment'])
            cross_tab.plot(kind='bar', stacked=True)
            plt.title(f'Treatment by {var}', fontsize=14)
            plt.xlabel(var, fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Treatment')
            plt.grid(True, alpha=0.3)
            fname = os.path.join(out_dir, f"treatment_by_{var.replace(' ', '_')}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved treatment by {var} chart")

# 1) Histograms
if numeric_cols:
    for col in numeric_cols:
        plt.figure(figsize=(8,6))
        df[col].hist(bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'Histogram: {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        fname = os.path.join(out_dir, f"hist_{col.replace(' ', '_').replace('/', '_')}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved histogram for {col}")

# 2) Boxplots
if numeric_cols:
    plt.figure(figsize=(max(10, len(numeric_cols)*1.2), 8))
    df[numeric_cols].boxplot()
    plt.title("Boxplots (numeric columns)", fontsize=14)
    plt.xticks(rotation=45)
    fname = os.path.join(out_dir, "boxplots_numeric.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved boxplots")

# 3) Scatter matrix for up to 6 columns (reduced for better visualization)
cols_for_scatter = numeric_cols[:6]
if len(cols_for_scatter) >= 2:
    plt.figure(figsize=(12,12))
    scatter_matrix(df[cols_for_scatter], alpha=0.6, diagonal='hist', figsize=(12,12))
    plt.suptitle("Scatter Matrix", fontsize=16)
    fname = os.path.join(out_dir, "scatter_matrix.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved scatter matrix")

# 4) Correlation heatmap
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10,8))
    im = plt.imshow(corr.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.xticks(range(len(corr)), corr.columns, rotation=45)
    plt.yticks(range(len(corr)), corr.index)
    plt.colorbar(im, label='Correlation Coefficient')
    plt.title("Correlation Matrix (numeric columns)", fontsize=14)
    
    # Add correlation values as text
    for i in range(len(corr)):
        for j in range(len(corr)):
            text = plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    fname = os.path.join(out_dir, "correlation_matrix.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved correlation matrix")

# 5) QQ-plots (if scipy available)
if stats is not None and numeric_cols:
    for col in numeric_cols:
        plt.figure(figsize=(8,6))
        try:
            stats.probplot(df[col].dropna(), dist="norm", plot=plt)
            plt.title(f'QQ-plot: {col}', fontsize=14)
            plt.grid(True, alpha=0.3)
            fname = os.path.join(out_dir, f"qq_{col.replace(' ', '_').replace('/', '_')}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved QQ-plot for {col}")
        except Exception as e:
            print(f"QQ-plot failed for {col}: {e}")

# 6) Normality tests
normality_results = []
if stats is not None and numeric_cols:
    for col in numeric_cols:
        series = df[col].dropna()
        n = len(series)
        shapiro_p = None
        jarque_p = None
        try:
            if n <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(series)
            else:
                shapiro_p = np.nan
        except Exception:
            shapiro_p = np.nan
        try:
            jb_stat, jb_p = stats.jarque_bera(series)
            jarque_p = jb_p
        except Exception:
            jarque_p = np.nan
        normality_results.append((col, n, shapiro_p, jarque_p))
    
    normality_df = pd.DataFrame(normality_results, columns=['column','n','shapiro_p', 'jarque_bera_p'])
    print("\nNormality Tests Results:")
    print(normality_df)
    normality_df.to_csv(os.path.join(out_dir, "normality_tests.csv"), index=False)
else:
    print("scipy not available: skipping normality tests.")

# 7) VIF (if statsmodels available)
vif_df = None
if statsmodels is not None and len(numeric_cols) >= 2:
    try:
        X = df[numeric_cols].dropna()
        X = X.loc[:, (X.std()!=0)]
        if len(X.columns) >= 2:
            X_const = sm.add_constant(X)
            vif_data = []
            for i, col in enumerate(X.columns):
                vif = variance_inflation_factor(X_const.values, i+1)
                vif_data.append((col, vif))
            vif_df = pd.DataFrame(vif_data, columns=['feature','VIF']).sort_values('VIF', ascending=False)
            print("\nVariance Inflation Factors:")
            print(vif_df)
            vif_df.to_csv(os.path.join(out_dir, "vif_table.csv"), index=False)
    except Exception as e:
        print("VIF calculation failed:", e)
else:
    print("statsmodels not available or insufficient numeric cols: skipping VIF.")

# 8) PCA (if sklearn available)
pca_results = None
if sklearn is not None and len(numeric_cols) >= 2:
    X = df[numeric_cols].dropna()
    if len(X) > 1:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        pca = PCA(n_components=min(2, len(numeric_cols)))
        pcs = pca.fit_transform(Xs)
        
        plt.figure(figsize=(8,6))
        plt.scatter(pcs[:,0], pcs[:,1] if pcs.shape[1] > 1 else np.zeros(len(pcs)), alpha=0.6)
        plt.title("PCA: Component 1 vs Component 2", fontsize=14)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        if pcs.shape[1] > 1:
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        else:
            plt.ylabel("PC2")
        plt.grid(True, alpha=0.3)
        fname = os.path.join(out_dir, "pca_2d.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved PCA plot")
        pca_results = pca

# 9) KMeans clustering (if sklearn available)
if sklearn is not None and pca_results is not None and len(numeric_cols) >= 2:
    X = df[numeric_cols].dropna()
    if len(X) > 3:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        
        # Elbow plot
        inertias = []
        max_k = min(7, len(X))
        ks = range(1, max_k)
        for k in ks:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            kmeans.fit(Xs)
            inertias.append(kmeans.inertia_)
        
        plt.figure(figsize=(8,6))
        plt.plot(list(ks), inertias, marker='o')
        plt.title("KMeans Elbow Plot", fontsize=14)
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.grid(True, alpha=0.3)
        fname = os.path.join(out_dir, "kmeans_elbow.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved KMeans elbow plot")

        # Cluster visualization
        k_opt = min(3, len(X))
        kmeans = KMeans(n_clusters=k_opt, random_state=0, n_init=10).fit(Xs)
        pcs = PCA(n_components=2).fit_transform(Xs)
        
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(pcs[:,0], pcs[:,1], c=kmeans.labels_, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'KMeans Clusters (k={k_opt}) on PCA Space', fontsize=14)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, alpha=0.3)
        fname = os.path.join(out_dir, "kmeans_pca_clusters.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved KMeans cluster plot")

# 10) Top correlated pairs
corr_pairs = []
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr().abs()
    corr_vals = corr.where(~np.eye(corr.shape[0], dtype=bool))
    stacked = corr_vals.stack().reset_index()
    stacked.columns = ['var1','var2','abs_corr']
    stacked = stacked.sort_values('abs_corr', ascending=False)
    top_pairs = stacked.drop_duplicates(subset=['abs_corr']).head(6)
    print("\nTop Correlated Pairs:")
    print(top_pairs)
    top_pairs.to_csv(os.path.join(out_dir, "top_corr_pairs.csv"), index=False)
    corr_pairs = top_pairs.values.tolist()

# 11) Linear regression for strongest correlated pair (if available)
if corr_pairs and sklearn is not None:
    var1, var2, _ = corr_pairs[0]
    joined = df[[var1,var2]].dropna()
    if len(joined) >= 2:
        X = joined[[var1]].values.reshape(-1,1)
        y = joined[[var2]].values.reshape(-1,1)
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        print(f"\nLinear regression: Predicting {var2} from {var1}")
        print(f"Coef: {model.coef_[0][0]:.6f}, Intercept: {model.intercept_[0]:.6f}, R2: {r2:.4f}, MSE: {mse:.6f}")
        
        residuals = (y - y_pred).flatten()
        plt.figure(figsize=(8,6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, linestyle='--', color='red')
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title(f"Residuals Plot: {var2} ~ {var1}", fontsize=14)
        plt.grid(True, alpha=0.3)
        fname = os.path.join(out_dir, f"residuals_{var1.replace(' ', '_')}_{var2.replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved residuals plot")
    else:
        print("Not enough paired observations for regression.")

# 12) ADF test (if datetime index & statsmodels available)
adf_results = []
if df_indexed is not None and statsmodels is not None:
    for col in numeric_cols:
        series = df_indexed[col].dropna()
        if len(series) >= 20:
            try:
                stat, pval, usedlag, nobs, crit, icbest = adfuller(series)
                adf_results.append((col, stat, pval))
            except Exception:
                adf_results.append((col, np.nan, np.nan))
    if adf_results:
        adf_df = pd.DataFrame(adf_results, columns=['column','adf_stat','p_value']).sort_values('p_value')
        print("\nAugmented Dickey-Fuller Test Results:")
        print(adf_df)
        adf_df.to_csv(os.path.join(out_dir, "adf_results.csv"), index=False)
else:
    print("ADF test skipped (no datetime index or statsmodels not available).")

# Save numeric summary CSV
if numeric_cols:
    desc = df[numeric_cols].describe().T
    desc_path = os.path.join(out_dir, "numeric_summary.csv")
    desc.to_csv(desc_path)
    print(f"\nSaved numeric summary to {desc_path}")

print(f"\nAnalysis complete! All plots and results saved in the '{out_dir}' folder.")
print(f"Generated files:")
for file in os.listdir(out_dir):
    if file.endswith(('.png', '.csv')):
        print(f"  - {file}")
