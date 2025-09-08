# Full analysis pipeline for OSMI / Mental Health in Tech datasets
# Adapted for local environment - outputs to ./static/osmi_analysis/

import os, re, json, zipfile, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Stats & ML
import scipy.stats as stats
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- Helper utilities & paths ---
OUT_DIR = Path("./static/osmi_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
PDF_SUMMARY = OUT_DIR / "analysis_summary.pdf"

def save_fig(name):
    path = PLOTS_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print("Saved:", path)
    plt.close()
    return str(path)

def write_csv(df, name):
    path = OUT_DIR / name
    df.to_csv(path, index=True)
    print("Wrote:", path)
    return str(path)

# --- Load datasets (CSV + JSON) from newDataset folder ---
newdata_dir = "./newDataset/OSMI Mental Health in Tech Survery 2016/"
csv_candidates = [p for p in os.listdir(newdata_dir) if p.lower().endswith(".csv")]
json_candidates = [p for p in os.listdir(newdata_dir) if p.lower().endswith(".json")]

csv_path = None
json_path = None

# pick obvious names if available
for c in csv_candidates:
    if "mental" in c.lower() or "osmi" in c.lower():
        csv_path = newdata_dir + c
        break
if not csv_path and csv_candidates:
    csv_path = newdata_dir + csv_candidates[0]

for j in json_candidates:
    if "osmi" in j.lower() or "survey" in j.lower():
        json_path = newdata_dir + j
        break
if not json_path and json_candidates:
    json_path = newdata_dir + json_candidates[0]

print("Detected CSV:", csv_path)
print("Detected JSON:", json_path)

df_csv = pd.read_csv(csv_path, low_memory=False) if csv_path else pd.DataFrame()
df_json = pd.DataFrame()
if json_path:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Flatten if structure is {responses: [...]}
    responses = raw.get("responses") if isinstance(raw, dict) else raw
    rows = []
    if isinstance(responses, list):
        for r in responses:
            row = {}
            # metadata
            meta = r.get("metadata", {}) if isinstance(r, dict) else {}
            row["date_submit"] = meta.get("date_submit") or meta.get("date_land")
            row["token"] = r.get("token")
            row["completed"] = r.get("completed")
            # flatten answers dict -> columns
            answers = r.get("answers", {}) or {}
            for k, v in answers.items():
                # for choice lists (dict choices) extract label if nested
                row[k] = v
            rows.append(row)
        df_json = pd.DataFrame(rows)
    else:
        # If JSON directly a table
        try:
            df_json = pd.DataFrame(raw)
        except Exception:
            df_json = pd.DataFrame()

print("CSV shape:", df_csv.shape, "JSON shape:", df_json.shape)

# --- Basic cleaning helpers ---
def coerce_numeric_like(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            conv = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")
            if conv.notna().sum() / max(1, len(conv)) > 0.5:
                df[col] = conv
    return df

if not df_csv.empty:
    df_csv = coerce_numeric_like(df_csv)
if not df_json.empty:
    df_json = coerce_numeric_like(df_json)

# Merge CSV+JSON when possible by appending columns (if same number of rows)
# But safest is to primarily analyze CSV (Kaggle) and use JSON flatten when helpful
df_main = df_csv.copy() if not df_csv.empty else df_json.copy()

print("Main dataset columns:", list(df_main.columns))
print("Main dataset shape:", df_main.shape)

# --- Normalize Gender field (robust mapping) ---
def clean_gender_col(df, col_name_guess_list=None):
    """Find a gender-like column, normalize common text variants to canonical categories."""
    df = df.copy()
    # find candidate column
    col = None
    if col_name_guess_list:
        for g in col_name_guess_list:
            if g in df.columns:
                col = g
                break
    if col is None:
        for c in df.columns:
            if "gender" in c.lower() or "sex" == c.lower() or c.lower().startswith("what gender"):
                col = c
                break
    if col is None:
        return df, None
    s = df[col].astype(str).fillna("").str.strip()
    s_norm = s.str.lower().str.replace(r"[^\w\s]", "", regex=True)
    def map_gender(x):
        if x is None or x == "" or x == "nan":
            return np.nan
        x = x.strip().lower()
        # patterns for male
        if re.search(r"\b(male|m|man|cis male|cis-man|male \(trans\)|transmale)\b", x):
            return "Male"
        if re.search(r"\b(female|f|woman|cis female|cis-female|cis-woman|female \(trans\)|transfemale)\b", x):
            return "Female"
        if re.search(r"\b(non ?binary|nonbinary|nb|genderqueer|gender fluid|genderqueer)\b", x):
            return "Non-binary"
        if re.search(r"\b(trans|transgender)\b", x) and re.search(r"\b(male|female)\b", x):
            # "trans female" etc
            if "male" in x: return "Trans Male"
            if "female" in x: return "Trans Female"
        if re.search(r"\b(other|prefer not to say|prefer not|unknown|none)\b", x):
            return "Other/Prefer not to say"
        # fallback heuristics
        if len(x) <= 2 and x in ["m","f"]:
            return "Male" if x=="m" else "Female"
        # last resort return 'Other' to avoid free text explosion
        return "Other"
    df["Gender_cleaned"] = s.apply(map_gender)
    return df, col

df_main, gender_col = clean_gender_col(df_main, col_name_guess_list=["Gender", "gender", "What is your gender?"])
print("Normalized gender column:", gender_col)

# Collapse obvious family_history, treatment columns to canonical values if present
def canonicalize_binary(df, col, true_values=None):
    if col not in df.columns:
        return df
    s = df[col].astype(str).str.strip().str.lower()
    df[col + "_clean"] = np.nan
    if true_values is None:
        true_values = ["yes","y","true","1"]
    df.loc[s.isin(true_values), col + "_clean"] = 1
    df.loc[s.isin(["no","n","false","0"]), col + "_clean"] = 0
    return df

# Find treatment-related columns
treatment_cols = [c for c in df_main.columns if any(word in c.lower() for word in ["treatment", "seek help", "sought"])]
family_history_cols = [c for c in df_main.columns if "family" in c.lower() and "history" in c.lower()]

print("Treatment-related columns found:", treatment_cols)
print("Family history columns found:", family_history_cols)

if treatment_cols:
    df_main = canonicalize_binary(df_main, treatment_cols[0], ["yes"])
    df_main["treatment_clean"] = df_main[treatment_cols[0] + "_clean"]

if family_history_cols:
    df_main = canonicalize_binary(df_main, family_history_cols[0], ["yes"])
    df_main["family_history_clean"] = df_main[family_history_cols[0] + "_clean"]

# If CSV has 'Age' as numeric string field, ensure numeric and clip to plausible range
age_cols = [c for c in df_main.columns if "age" in c.lower()]
if age_cols:
    age_col = age_cols[0]
    df_main["Age"] = pd.to_numeric(df_main[age_col], errors="coerce")
    # remove implausible ages
    df_main.loc[(df_main["Age"] < 10) | (df_main["Age"] > 100), "Age"] = np.nan
    print(f"Age statistics: {df_main['Age'].describe()}")

# --- STATISTICAL TESTS ---
stat_results = {}
pdf = PdfPages(PDF_SUMMARY)

# Create a summary plot first
plt.figure(figsize=(12, 8))
plt.suptitle("OSMI Mental Health Survey 2016 - Dataset Overview", fontsize=16)

# 1) Age by treatment: choose rows where treatment == Yes/No canonical
if "treatment_clean" in df_main.columns and "Age" in df_main.columns:
    age_df = df_main[["Age", "treatment_clean"]].dropna()
    if len(age_df) > 0:
        age_df["treatment_clean"] = age_df["treatment_clean"].astype(int)
        grp_yes = age_df[age_df["treatment_clean"] == 1]["Age"].dropna()
        grp_no  = age_df[age_df["treatment_clean"] == 0]["Age"].dropna()
        
        if len(grp_yes) > 0 and len(grp_no) > 0:
            # Normality
            sh_yes_p = stats.shapiro(grp_yes).pvalue if len(grp_yes)>=3 and len(grp_yes)<=5000 else np.nan
            sh_no_p  = stats.shapiro(grp_no).pvalue if len(grp_no)>=3 and len(grp_no)<=5000 else np.nan
            # Variance equality
            levene_p = stats.levene(grp_yes, grp_no).pvalue if len(grp_yes)>=2 and len(grp_no)>=2 else np.nan
            # Choose t-test or Mann-Whitney
            if (not np.isnan(sh_yes_p) and not np.isnan(sh_no_p) and sh_yes_p>0.05 and sh_no_p>0.05):
                # normal -> t-test
                tstat, tp = stats.ttest_ind(grp_yes, grp_no, equal_var=(levene_p>0.05))
                stat_results["Age_ttest"] = {"t_stat": float(tstat), "p_value": float(tp), "method":"t-test"}
                test_method = "t-test"
            else:
                # nonparametric
                try:
                    mw_stat, mw_p = stats.mannwhitneyu(grp_yes, grp_no, alternative="two-sided")
                    stat_results["Age_mannwhitney"] = {"stat": float(mw_stat), "p_value": float(mw_p), "method":"Mann-Whitney U"}
                    test_method = "Mann-Whitney U"
                except Exception:
                    stat_results["Age_nonparametric_failed"] = {}
                    test_method = "no-test"
            # plot Age distributions
            plt.figure(figsize=(8,5))
            plt.boxplot([grp_no.dropna().values, grp_yes.dropna().values], labels=["No Treatment", "Sought Treatment"])
            plt.title(f"Age by Treatment Status (test used: {test_method})")
            plt.ylabel("Age")
            save_fig("age_by_treatment_box.png")
            pdf.savefig(); plt.close()
        else:
            print("Insufficient data for age vs treatment analysis.")
else:
    print("Skipping Age vs treatment: required columns not present.")

# 2) Chi-square tests between treatment and selected categorical features
def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r,k = confusion_matrix.shape
    # correct bias
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1),(rcorr-1))
    return (np.sqrt(phi2corr/denom) if denom>0 else np.nan)

categorical_candidates = []
# choose several likely categorical fields if present
all_cols = df_main.columns.tolist()
cands = ["Gender_cleaned","Gender","family_history_clean","family_history","remote_work","work_interfere","benefits","tech_company","no_employees","Country","state"]

# Add columns that contain relevant keywords
for col in all_cols:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in ["gender", "work", "company", "benefit", "remote", "tech", "country", "state", "interfere"]):
        if col not in categorical_candidates:
            categorical_candidates.append(col)

# Add our cleaned columns
for c in cands:
    if c in df_main.columns and c not in categorical_candidates:
        categorical_candidates.append(c)

print("Categorical candidates for chi-square:", categorical_candidates[:10])

chi_results = []
if "treatment_clean" in df_main.columns:
    for col in categorical_candidates[:8]:  # Limit to prevent too many tests
        try:
            ct = pd.crosstab(df_main[col].fillna("NA"), df_main["treatment_clean"].fillna(-1))
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2, p, dof, ex = stats.chi2_contingency(ct)
            cramer = cramers_v(ct)
            chi_results.append({"feature": col, "chi2": float(chi2), "p_value": float(p), "cramers_v": float(cramer)})
            # plot stacked bar
            plt.figure(figsize=(10,6))
            ax = ct.plot(kind="bar", stacked=True)
            plt.title(f"{col} vs Treatment Status\n(chi2 p={p:.3g}, Cramer's V={cramer:.3f})")
            plt.xlabel(col.replace('_', ' ').title())
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title="Treatment", labels=["No", "Yes"])
            save_fig(f"chi_{col.replace(' ', '_').replace('/', '_')}_vs_treatment.png")
            pdf.savefig()
        except Exception as e:
            print("Chi test failed for", col, str(e))

    if chi_results:
        chi_df = pd.DataFrame(chi_results).sort_values("p_value")
        write_csv(chi_df, "chi2_results.csv")
        print("\nTop significant associations with treatment:")
        print(chi_df.head())
else:
    print("Skipping chi-square: 'treatment_clean' not present.")

# 3) Nonparametric group comparison for Age across 3+ groups e.g., gender
if "Gender_cleaned" in df_main.columns and "Age" in df_main.columns:
    age_gender = df_main.dropna(subset=["Gender_cleaned","Age"])
    if len(age_gender) > 0:
        group_age = age_gender.groupby("Gender_cleaned")["Age"].apply(list)
        if group_age.shape[0] >= 2:
            # Kruskal-Wallis
            try:
                groups = [np.array(v) for v in group_age if len(v) > 0]
                if len(groups) >= 2:
                    kr_stat, kr_p = stats.kruskal(*groups)
                    stat_results["age_kruskal_gender"] = {"stat": float(kr_stat), "p_value": float(kr_p)}
                    # plot violin-like via boxplot
                    plt.figure(figsize=(10,6))
                    labels = group_age.index.astype(str)
                    data = [group_age.loc[l] for l in labels if len(group_age.loc[l]) > 0]
                    plt.boxplot(data, labels=labels[:len(data)])
                    plt.title(f"Age by Gender (Kruskal p={kr_p:.3g})")
                    plt.ylabel("Age")
                    plt.xticks(rotation=45)
                    save_fig("age_by_gender_box.png")
                    pdf.savefig()
            except Exception as e:
                print("Kruskal failed:", e)

# --- MODELING: logistic regression to predict treatment (Yes/No) ---
model_reports = {}
candidate_features = []

if "treatment_clean" in df_main.columns:
    # Build feature matrix
    df_model = df_main.copy()
    # target
    df_model = df_model.dropna(subset=["treatment_clean"])
    df_model["treatment_target"] = df_model["treatment_clean"].astype(int)
    y = df_model["treatment_target"]
    
    print(f"Treatment distribution: {y.value_counts()}")
    
    # choose candidate features (Age + a few categorical)
    if "Age" in df_model.columns:
        candidate_features.append("Age")
    # add cleaned gender if available
    if "Gender_cleaned" in df_model.columns:
        candidate_features.append("Gender_cleaned")
    # add family_history_clean if available
    if "family_history_clean" in df_model.columns:
        candidate_features.append("family_history_clean")
    
    # Add top significant categorical features from chi-square results
    if chi_results:
        top_features = [r["feature"] for r in chi_results[:5] if r["p_value"] < 0.05]
        for f in top_features:
            if f not in candidate_features and f in df_model.columns:
                candidate_features.append(f)
    
    # Limit features to prevent overfitting
    candidate_features = candidate_features[:8]
    
    print("Model features:", candidate_features)
    
    if candidate_features:
        X = df_model[candidate_features].copy()
        # Remove rows where all features are NaN
        X = X.dropna(how='all')
        y = y.loc[X.index]
        
        if len(X) > 10 and len(y.unique()) > 1:
            # Preprocessing pipeline: numeric impute+scale, categorical impute+onehot
            numeric_feats = [c for c in candidate_features if np.issubdtype(X[c].dtype, np.number)]
            cat_feats = [c for c in candidate_features if c not in numeric_feats]
            
            print(f"Numeric features: {numeric_feats}")
            print(f"Categorical features: {cat_feats}")
            
            transformers = []
            if numeric_feats:
                numeric_transform = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])
                transformers.append(("num", numeric_transform, numeric_feats))
            
            if cat_feats:
                cat_transform = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ])
                transformers.append(("cat", cat_transform, cat_feats))
            
            preproc = ColumnTransformer(transformers=transformers, remainder="drop")
            
            # logistic regression pipeline
            pipe = Pipeline([
                ("preproc", preproc),
                ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))
            ])
            
            # train-test split for final evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
            
            # cross-validated score
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
            print("Logistic CV ROC AUC:", cv_scores.mean(), "±", cv_scores.std())
            
            # fit final
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:,1]
            
            acc = accuracy_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test))>1 else np.nan
            
            print("Logistic test acc:", acc, "roc:", roc)
            
            model_reports["logistic_cv_roc_auc_mean"] = float(cv_scores.mean())
            model_reports["logistic_test_acc"] = float(acc)
            model_reports["logistic_test_roc"] = float(roc)
            
            # classification report & confusion matrix
            creport = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            write_csv(pd.DataFrame(creport).transpose(), "logistic_classification_report.csv")
            write_csv(pd.DataFrame(cm), "logistic_confusion_matrix.csv")
            
            # ROC curve plot
            fpr, tpr, _ = roc_curve(y_test, y_proba) if len(np.unique(y_test))>1 else (None,None,None)
            if fpr is not None:
                plt.figure(figsize=(8,6))
                plt.plot(fpr, tpr, label=f"AUC={roc:.3f}", linewidth=2)
                plt.plot([0,1],[0,1],"--", color="gray", alpha=0.7)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Logistic Regression ROC Curve")
                plt.legend()
                plt.grid(True, alpha=0.3)
                save_fig("logistic_roc.png")
                pdf.savefig()
            
            # Feature importance (coefficients)
            try:
                # Get feature names after preprocessing
                feature_names = []
                if numeric_feats:
                    feature_names.extend(numeric_feats)
                if cat_feats:
                    cat_names = pipe.named_steps["preproc"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_feats)
                    feature_names.extend(cat_names)
                
                coefs = pipe.named_steps["clf"].coef_[0]
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coefs,
                    'abs_coefficient': np.abs(coefs)
                }).sort_values('abs_coefficient', ascending=False)
                
                write_csv(feature_importance, "logistic_feature_importance.csv")
                
                # Plot top features
                top_features = feature_importance.head(10)
                plt.figure(figsize=(10,6))
                plt.barh(range(len(top_features)), top_features['coefficient'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Coefficient Value')
                plt.title('Top 10 Feature Coefficients (Logistic Regression)')
                plt.tight_layout()
                save_fig("logistic_feature_importance.png")
                pdf.savefig()
                
            except Exception as e:
                print("Failed to extract feature importance:", e)
        
        else:
            print("Insufficient data for modeling")
    else:
        print("No candidate features for logistic regression found.")
else:
    print("Skipping modeling: 'treatment_clean' not present.")

# 4) Decision Tree classifier (same features)
if "treatment_target" in locals() and candidate_features and len(X) > 10:
    dt_pipe = Pipeline([
        ("preproc", preproc),
        ("clf", DecisionTreeClassifier(random_state=0, max_depth=4))
    ])
    cv_scores_acc = cross_val_score(dt_pipe, X, y, cv=cv, scoring="accuracy")
    dt_pipe.fit(X, y)
    model_reports["decisiontree_cv_acc_mean"] = float(np.mean(cv_scores_acc))
    print("Decision Tree CV acc mean:", np.mean(cv_scores_acc))
    
    write_csv(pd.DataFrame({"decision_tree_cv_acc": cv_scores_acc}), "dt_cv_scores.csv")

# --- DIMENSIONALITY REDUCTION & CLUSTERING ---
# Create an encoded feature matrix for PCA & clustering
encoded_X = None
encoded_feature_names = None

try:
    if "preproc" in locals() and len(candidate_features) > 1:
        X_sample = X.sample(min(1000, len(X)), random_state=42)  # Sample for efficiency
        encoded_X = preproc.fit_transform(X_sample)
        print(f"Encoded data shape: {encoded_X.shape}")
    else:
        # fallback: simple encoding for visualization
        X_viz = df_main.select_dtypes(include=["number"]).fillna(df_main.select_dtypes(include=["number"]).median())
        if not X_viz.empty:
            encoded_X = StandardScaler().fit_transform(X_viz)
            encoded_feature_names = X_viz.columns.tolist()
except Exception as e:
    print("Encoding for PCA failed:", e)

if encoded_X is not None and encoded_X.shape[1] > 1:
    # PCA 2 components
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(encoded_X)
    
    plt.figure(figsize=(10,7))
    # color by treatment if available
    if "treatment_target" in locals() and len(X_sample) == len(pcs):
        y_sample = y.loc[X_sample.index]
        scatter = plt.scatter(pcs[:,0], pcs[:,1], c=y_sample, cmap="RdYlBu", alpha=0.7)
        plt.colorbar(scatter, label="Treatment Status")
        plt.title("PCA (2 components) colored by Treatment Status")
    else:
        plt.scatter(pcs[:,0], pcs[:,1], alpha=0.7)
        plt.title("PCA (2 components)")
    
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    save_fig("pca_2d.png")
    pdf.savefig()
    
    # KMeans elbow and clusters
    if encoded_X.shape[0] > 10:
        inertias = []
        ks = range(1, min(8, encoded_X.shape[0]//2))
        for k in ks:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            inertias.append(kmeans.fit(encoded_X).inertia_)
        
        plt.figure(figsize=(8,5))
        plt.plot(list(ks), inertias, marker='o', linewidth=2, markersize=8)
        plt.xlabel("k (Number of Clusters)")
        plt.ylabel("Inertia")
        plt.title("KMeans Elbow Plot")
        plt.grid(True, alpha=0.3)
        save_fig("kmeans_elbow.png")
        pdf.savefig()
        
        # choose k=3 if no other guidance
        k_opt = 3 if encoded_X.shape[0] >= 3 else 2
        km = KMeans(n_clusters=k_opt, random_state=0, n_init=10).fit(encoded_X)
        
        plt.figure(figsize=(10,7))
        scatter = plt.scatter(pcs[:,0], pcs[:,1], c=km.labels_, cmap="tab10", alpha=0.8)
        plt.colorbar(scatter, label="Cluster")
        plt.title(f"KMeans Clusters (k={k_opt}) on PCA Space")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        save_fig("kmeans_pca_clusters.png")
        pdf.savefig()
else:
    print("Skipping PCA/KMeans due to insufficient data or encoding failure.")

# --- Dataset Summary Statistics ---
summary_stats = {
    "total_rows": len(df_main),
    "total_columns": len(df_main.columns),
    "missing_data_percentage": (df_main.isnull().sum().sum() / (len(df_main) * len(df_main.columns))) * 100
}

if "Age" in df_main.columns:
    summary_stats["age_mean"] = df_main["Age"].mean()
    summary_stats["age_std"] = df_main["Age"].std()

if "treatment_clean" in df_main.columns:
    treatment_pct = df_main["treatment_clean"].value_counts(normalize=True) * 100
    summary_stats["treatment_yes_pct"] = treatment_pct.get(1, 0)
    summary_stats["treatment_no_pct"] = treatment_pct.get(0, 0)

# Save summary statistics
write_csv(pd.DataFrame([summary_stats]).T, "summary_statistics.csv")

# Create final summary visualization
plt.figure(figsize=(15, 10))
plt.suptitle("OSMI Mental Health Survey 2016 - Analysis Summary", fontsize=16)

# Age distribution
if "Age" in df_main.columns:
    plt.subplot(2, 3, 1)
    df_main["Age"].dropna().hist(bins=30, alpha=0.7)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")

# Treatment distribution
if "treatment_clean" in df_main.columns:
    plt.subplot(2, 3, 2)
    treatment_counts = df_main["treatment_clean"].value_counts()
    plt.pie(treatment_counts.values, labels=["No Treatment", "Sought Treatment"], autopct='%1.1f%%')
    plt.title("Treatment Status Distribution")

# Gender distribution
if "Gender_cleaned" in df_main.columns:
    plt.subplot(2, 3, 3)
    gender_counts = df_main["Gender_cleaned"].value_counts().head(5)
    plt.bar(range(len(gender_counts)), gender_counts.values)
    plt.xticks(range(len(gender_counts)), gender_counts.index, rotation=45)
    plt.title("Top 5 Gender Categories")
    plt.ylabel("Count")

# Missing data heatmap (top 10 columns with most missing data)
plt.subplot(2, 3, 4)
missing_data = df_main.isnull().sum().sort_values(ascending=False).head(10)
if len(missing_data) > 0:
    plt.barh(range(len(missing_data)), missing_data.values)
    plt.yticks(range(len(missing_data)), missing_data.index)
    plt.title("Top 10 Columns with Missing Data")
    plt.xlabel("Missing Count")

# Model performance (if available)
if model_reports:
    plt.subplot(2, 3, 5)
    metrics = ["logistic_test_acc", "logistic_test_roc"]
    values = [model_reports.get(m, 0) for m in metrics]
    plt.bar(["Accuracy", "ROC AUC"], values)
    plt.title("Model Performance")
    plt.ylabel("Score")
    plt.ylim([0, 1])

# Statistical test results
if stat_results:
    plt.subplot(2, 3, 6)
    test_names = []
    p_values = []
    for test, result in stat_results.items():
        if isinstance(result, dict) and "p_value" in result:
            test_names.append(test.replace("_", " ").title())
            p_values.append(result["p_value"])
    
    if test_names:
        colors = ['red' if p < 0.05 else 'blue' for p in p_values]
        plt.bar(range(len(p_values)), p_values, color=colors)
        plt.xticks(range(len(test_names)), test_names, rotation=45)
        plt.title("Statistical Test P-values")
        plt.ylabel("P-value")
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
save_fig("analysis_summary.png")
pdf.savefig()

# --- Save cleaned dataset ---
write_csv(df_main, "df_main_cleaned.csv")

# Close PDF
pdf.close()
print("Saved PDF summary to", PDF_SUMMARY)

# Create results summary report
report_content = f"""# OSMI Mental Health Survey 2016 Analysis Report

## Dataset Overview
- **Total Records**: {summary_stats['total_rows']:,}
- **Total Variables**: {summary_stats['total_columns']}
- **Missing Data**: {summary_stats['missing_data_percentage']:.1f}%

## Key Findings

### Demographics
"""

if "Age" in df_main.columns:
    report_content += f"- **Age**: Mean = {summary_stats.get('age_mean', 0):.1f} years (SD = {summary_stats.get('age_std', 0):.1f})\n"

if "treatment_clean" in df_main.columns:
    report_content += f"- **Treatment Seeking**: {summary_stats.get('treatment_yes_pct', 0):.1f}% sought treatment\n"

report_content += f"""
### Statistical Tests
"""

for test, result in stat_results.items():
    if isinstance(result, dict) and "p_value" in result:
        significance = "significant" if result["p_value"] < 0.05 else "not significant"
        report_content += f"- **{test.replace('_', ' ').title()}**: p = {result['p_value']:.3f} ({significance})\n"

if model_reports:
    report_content += f"""
### Machine Learning Performance
- **Logistic Regression CV ROC AUC**: {model_reports.get('logistic_cv_roc_auc_mean', 0):.3f}
- **Test Accuracy**: {model_reports.get('logistic_test_acc', 0):.3f}
- **Test ROC AUC**: {model_reports.get('logistic_test_roc', 0):.3f}
"""

report_content += f"""
## Files Generated
- **Plots**: {len(list(PLOTS_DIR.glob('*.png')))} visualization files
- **Data**: Cleaned dataset and statistical results
- **Models**: Trained machine learning pipelines
"""

with open(OUT_DIR / "analysis_report.md", "w") as f:
    f.write(report_content)

print("\nAnalysis Summary:")
print(f"- Dataset: {summary_stats['total_rows']:,} records, {summary_stats['total_columns']} variables")
print(f"- Generated {len(list(PLOTS_DIR.glob('*.png')))} plots")
print(f"- Created {len(list(OUT_DIR.glob('*.csv')))} CSV files")
print(f"- Results saved in: {OUT_DIR}")

print("\nTop files generated:")
for i, f in enumerate(sorted(PLOTS_DIR.glob("*.png"))):
    if i < 10:
        print(f"  - {f.name}")
    elif i == 10:
        print(f"  - ... and {len(list(PLOTS_DIR.glob('*.png')))-10} more")
        break

print(f"\n✅ Analysis complete! Check {OUT_DIR} for all results.")
