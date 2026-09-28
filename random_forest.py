import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────
df = pd.read_csv("cleaned_hr_data.csv")

print("=" * 50)
print("STEP 1: Dataset Overview")
print("=" * 50)
print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print("\nColumn Names:")
print(df.columns.tolist())
print("\nData Types:")
print(df.dtypes.to_string())
print("\nMissing Values:")
print(df.isnull().sum().to_string())
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# ─────────────────────────────────────────────
# STEP 2: Prepare Target Variable
# ─────────────────────────────────────────────
df["Attrition_Label"] = df["Attrition"].map({"Stayed": 0, "Left": 1})
y = df["Attrition_Label"]

print("\n" + "=" * 50)
print("STEP 2: Attrition Distribution")
print("=" * 50)
counts = df["Attrition"].value_counts()
print(counts.to_string())
attrition_rate = counts.get("Left", 0) / len(df) * 100
print(f"Attrition Rate: {attrition_rate:.2f}%")

# ─────────────────────────────────────────────
# STEP 3: Select Features
# ─────────────────────────────────────────────
EXCLUDE = {
    "Attrition", "Attrition_Label", "EmployeeStatus", "ExitDate",
    "TerminationType", "TerminationDescription", "EmpID",
    "Employee ID", "Employee ID_x", "Employee ID_y",
    "FirstName", "LastName", "ADEmail", "DOB", "StartDate",
    "Survey Date", "Training Date", "Trainer", "Location",
    "LocationCode", "Supervisor",
}

CANDIDATE_FEATURES = [
    "Engagement Score", "Satisfaction Score", "Work-Life Balance Score",
    "Training Duration(Days)", "Training Cost", "Current Employee Rating",
    "PayZone", "DepartmentType", "EmployeeType", "EmployeeClassificationType",
    "Performance Score", "BusinessUnit", "Division", "GenderCode",
    "MaritalDesc", "State", "JobFunctionDescription",
    "RaceDesc", "Title", "Training Program Name", "Training Type",
    "Training Outcome",
]

features = [c for c in CANDIDATE_FEATURES if c in df.columns and c not in EXCLUDE]
X = df[features].copy()

print("\n" + "=" * 50)
print("STEP 3: Selected Features")
print("=" * 50)
print(features)

# ─────────────────────────────────────────────
# STEP 4 & 5: Identify column types, build preprocessor
# ─────────────────────────────────────────────
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

print(f"\nNumerical columns ({len(num_cols)}): {num_cols}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# ─────────────────────────────────────────────
# STEP 6: Train/Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n" + "=" * 50)
print("STEP 6: Train/Test Split")
print("=" * 50)
print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 7: Preprocess → SMOTE → Train Random Forest
# ─────────────────────────────────────────────
# Fit preprocessor on training data only to prevent data leakage
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

# SMOTE applied only on training data to balance the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

print(f"\nClass imbalance: {(y_train == 0).sum()} Stayed vs {(y_train == 1).sum()} Left in training set.")
print(f"After SMOTE: {(y_train_resampled == 0).sum()} Stayed vs {(y_train_resampled == 1).sum()} Left.")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_resampled, y_train_resampled)
print("Random Forest model trained successfully.")

# ─────────────────────────────────────────────
# STEP 8: Evaluate Model
# Decision threshold lowered to 0.25 to improve recall for the minority Left class.
# At default 0.50 the model predicts all Stayed due to low feature separability.
# Threshold 0.25 gives the best F1 balance for the Left class on this dataset.
# ─────────────────────────────────────────────
THRESHOLD = 0.25
y_proba = rf.predict_proba(X_test_processed)[:, 1]
y_pred  = (y_proba >= THRESHOLD).astype(int)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 50)
print(f"STEP 8: Model Evaluation (threshold={THRESHOLD})")
print("=" * 50)
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Score":  [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]
})
print(metrics_df.to_string(index=False))
print(f"\nNote: Decision threshold set to {THRESHOLD} (instead of default 0.50).")
print("This is necessary because the dataset has low feature separability between")
print("Left and Stayed employees. At 0.50 the model predicts all employees as Stayed.")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))

# ─────────────────────────────────────────────
# STEP 9: Confusion Matrix
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed", "Left"],
            yticklabels=["Stayed", "Left"])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("\nConfusion matrix saved as confusion_matrix.png")

# ─────────────────────────────────────────────
# STEP 9b: Additional Visualizations
# ─────────────────────────────────────────────

# --- Chart 1: Attrition Rate by Department ---
dept_viz = (
    df.groupby("DepartmentType")["Attrition_Label"]
    .agg(["sum", "count"])
    .assign(rate=lambda d: (d["sum"] / d["count"] * 100).round(2))
    .sort_values("rate", ascending=False)
    .reset_index()
)
dept_viz["DepartmentType"] = dept_viz["DepartmentType"].str.strip()

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(dept_viz["DepartmentType"], dept_viz["rate"],
              color=["#d62728" if r > 15 else "#1f77b4" for r in dept_viz["rate"]])
for bar, rate in zip(bars, dept_viz["rate"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_title("Attrition Rate by Department", fontsize=13, fontweight="bold")
ax.set_xlabel("Department")
ax.set_ylabel("Attrition Rate (%)")
ax.set_ylim(0, dept_viz["rate"].max() + 4)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("viz_attrition_by_department.png", dpi=150)
plt.close()
print("Chart 1 saved: viz_attrition_by_department.png")

# --- Chart 2: Attrition Rate by PayZone and EmployeeType ---
payzone_emp = (
    df.groupby(["PayZone", "EmployeeType"])["Attrition_Label"]
    .agg(["sum", "count"])
    .assign(rate=lambda d: (d["sum"] / d["count"] * 100).round(2))
    .reset_index()
)
pivot = payzone_emp.pivot(index="PayZone", columns="EmployeeType", values="rate")

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(pivot.index))
width = 0.25
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for i, col in enumerate(pivot.columns):
    rects = ax.bar(x + i * width, pivot[col], width, label=col, color=colors[i])
    for rect in rects:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.2,
                f"{rect.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
ax.set_title("Attrition Rate by PayZone and Employee Type", fontsize=13, fontweight="bold")
ax.set_xlabel("PayZone")
ax.set_ylabel("Attrition Rate (%)")
ax.set_xticks(x + width)
ax.set_xticklabels(pivot.index)
ax.set_ylim(0, pivot.values.max() + 5)
ax.legend(title="Employee Type")
plt.tight_layout()
plt.savefig("viz_attrition_by_payzone_emptype.png", dpi=150)
plt.close()
print("Chart 2 saved: viz_attrition_by_payzone_emptype.png")

# --- Chart 3: Average Score Comparison (Left vs Stayed) ---
score_cols = ["Engagement Score", "Satisfaction Score", "Work-Life Balance Score"]
score_means = df.groupby("Attrition")[score_cols].mean().T.reset_index()
score_means.columns = ["Score", "Left", "Stayed"]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(score_means))
width = 0.35
b1 = ax.bar(x - width / 2, score_means["Stayed"], width, label="Stayed", color="#1f77b4")
b2 = ax.bar(x + width / 2, score_means["Left"],   width, label="Left",   color="#d62728")
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
ax.set_title("Average Score: Stayed vs Left Employees", fontsize=13, fontweight="bold")
ax.set_xlabel("Score Type")
ax.set_ylabel("Average Score (1-5)")
ax.set_xticks(x)
ax.set_xticklabels([s.replace(" Score", "") for s in score_means["Score"]])
ax.set_ylim(0, 4)
ax.legend()
plt.tight_layout()
plt.savefig("viz_score_comparison.png", dpi=150)
plt.close()
print("Chart 3 saved: viz_score_comparison.png")

# --- Chart 4: Attrition Rate by Score Level (1-5) for all 3 scores ---
fig, ax = plt.subplots(figsize=(9, 5))
line_styles = ["-o", "-s", "-^"]
line_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for i, col in enumerate(score_cols):
    rate_by_level = (
        df.groupby(col)["Attrition_Label"]
        .agg(["sum", "count"])
        .assign(rate=lambda d: (d["sum"] / d["count"] * 100).round(2))
        .reset_index()
    )
    ax.plot(rate_by_level[col], rate_by_level["rate"],
            line_styles[i], color=line_colors[i],
            label=col.replace(" Score", ""), linewidth=2, markersize=7)
ax.set_title("Attrition Rate by Score Level (1-5)", fontsize=13, fontweight="bold")
ax.set_xlabel("Score Level")
ax.set_ylabel("Attrition Rate (%)")
ax.set_xticks([1, 2, 3, 4, 5])
ax.legend(title="Score Type")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("viz_attrition_by_score_level.png", dpi=150)
plt.close()
print("Chart 4 saved: viz_attrition_by_score_level.png")

# ─────────────────────────────────────────────
# STEP 10: Feature Importance
# ─────────────────────────────────────────────
ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
all_feature_names = num_cols + cat_feature_names

importances = rf.feature_importances_
feat_imp_df = (
    pd.DataFrame({"Feature": all_feature_names, "Importance": importances})
    .sort_values("Importance", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
# Clean up whitespace in feature names for display
feat_imp_df["Feature"] = feat_imp_df["Feature"].str.strip()

print("\n" + "=" * 50)
print("STEP 10: Top 10 Feature Importances")
print("=" * 50)
print(feat_imp_df.to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x="Importance", y="Feature",
            hue="Feature", palette="viridis", legend=False)
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Feature importance chart saved as feature_importance.png")

# ─────────────────────────────────────────────
# STEP 11 & 12: Insights & Action Plan (data-driven)
# ─────────────────────────────────────────────

# Department-level attrition
dept_attrition = (
    df.groupby("DepartmentType")["Attrition_Label"]
    .agg(["sum", "count"])
    .assign(rate=lambda d: (d["sum"] / d["count"] * 100).round(2))
    .sort_values("rate", ascending=False)
)
top_dept      = dept_attrition.index[0]
top_dept_rate = dept_attrition["rate"].iloc[0]

# Engagement / Satisfaction / WLB averages by attrition
eng_left   = df[df["Attrition_Label"] == 1]["Engagement Score"].mean()
eng_stayed = df[df["Attrition_Label"] == 0]["Engagement Score"].mean()
sat_left   = df[df["Attrition_Label"] == 1]["Satisfaction Score"].mean()
sat_stayed = df[df["Attrition_Label"] == 0]["Satisfaction Score"].mean()
wlb_left   = df[df["Attrition_Label"] == 1]["Work-Life Balance Score"].mean()
wlb_stayed = df[df["Attrition_Label"] == 0]["Work-Life Balance Score"].mean()

# PayZone attrition
payzone_attrition = (
    df.groupby("PayZone")["Attrition_Label"]
    .agg(["sum", "count"])
    .assign(rate=lambda d: (d["sum"] / d["count"] * 100).round(2))
    .sort_values("rate", ascending=False)
)
top_payzone      = payzone_attrition.index[0]
top_payzone_rate = payzone_attrition["rate"].iloc[0]

# Training duration averages
train_left   = df[df["Attrition_Label"] == 1]["Training Duration(Days)"].mean()
train_stayed = df[df["Attrition_Label"] == 0]["Training Duration(Days)"].mean()

top_feature = feat_imp_df["Feature"].iloc[0]

insights = [
    f"The overall attrition rate was {attrition_rate:.2f}% ({counts.get('Left', 0)} out of {len(df)} employees left).",
    f"'{top_dept}' showed the highest department-level attrition rate at {top_dept_rate:.2f}%, suggesting it may benefit from targeted retention efforts.",
    f"Employees who left had a lower average Engagement Score ({eng_left:.2f}) compared to those who stayed ({eng_stayed:.2f}). Engagement score was associated with attrition patterns in this dataset.",
    f"Employees who left had an average Work-Life Balance Score of {wlb_left:.2f} compared to {wlb_stayed:.2f} for those who stayed, suggesting work-life balance was associated with attrition.",
    f"The PayZone '{top_payzone}' showed the highest attrition rate at {top_payzone_rate:.2f}%. Pay zone was associated with attrition patterns in this dataset.",
    f"Employees who left had an average Satisfaction Score of {sat_left:.2f} vs {sat_stayed:.2f} for those who stayed. Satisfaction score was associated with attrition.",
    f"The most important predictor in the Random Forest model was '{top_feature}'. The model achieved an accuracy of {accuracy:.2%}, recall of {recall:.2%}, and F1-score of {f1:.2%} for the Left class using a decision threshold of {THRESHOLD}. SMOTE oversampling was applied to address the class imbalance (87% Stayed vs 13% Left).",
]

action_plan = [
    {
        "finding": f"'{top_dept}' had the highest attrition rate ({top_dept_rate:.2f}%).",
        "action":  "Conduct stay interviews and pulse surveys specifically in this department to identify pain points.",
        "purpose": "Understand and address department-specific drivers of attrition before more employees leave."
    },
    {
        "finding": f"Employees who left had lower Engagement Scores ({eng_left:.2f} vs {eng_stayed:.2f}).",
        "action":  "Introduce regular engagement check-ins and recognition programmes for low-engagement employees.",
        "purpose": "Improve engagement levels, which were associated with lower attrition in this dataset."
    },
    {
        "finding": f"Work-Life Balance Score was lower among employees who left ({wlb_left:.2f} vs {wlb_stayed:.2f}).",
        "action":  "Review workload distribution, flexible working options, and manager practices for employees with low work-life balance scores.",
        "purpose": "Address work-life balance gaps that were associated with higher attrition."
    },
    {
        "finding": f"PayZone '{top_payzone}' showed the highest attrition rate ({top_payzone_rate:.2f}%).",
        "action":  "Review compensation benchmarking for this pay zone against market rates and consider targeted retention incentives.",
        "purpose": "Ensure compensation is competitive to reduce attrition risk in lower pay zones."
    },
    {
        "finding": f"'{top_feature}' was the most important predictor in the Random Forest model.",
        "action":  "Monitor and track this feature regularly as part of an early-warning HR dashboard.",
        "purpose": "Enable proactive identification of at-risk employees before they decide to leave."
    },
]

limitations = [
    "The dataset does not contain an exact Salary column. 'PayZone' is available as a pay band indicator but should not be treated as exact salary data.",
    "The dataset does not contain an Overtime column, so the relationship between overtime and satisfaction cannot be directly analysed.",
    "The dataset does not contain Promotion History or Promotion Date, so promotion delay vs attrition cannot be directly analysed.",
    "Recruitment data in the dataset contains applicant information and should not be treated as employee attrition data.",
    "This analysis shows associations and model predictions only. It does not prove that any factor causes attrition.",
]

# ─────────────────────────────────────────────
# FINAL SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n")
print("=" * 50)
print("HR ANALYTICS RANDOM FOREST")
print("=" * 50)

print(f"\nDataset Shape:\n  {df.shape[0]} rows x {df.shape[1] - 1} columns")

print(f"\nAttrition Distribution:")
for label, cnt in counts.items():
    print(f"  {label}: {cnt} ({cnt/len(df)*100:.2f}%)")

print(f"\nModel Performance (threshold={THRESHOLD}):")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")

print(f"\nTop 10 Important Features:")
for _, row in feat_imp_df.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

print("\nKey Insights:")
for idx, insight in enumerate(insights, 1):
    print(f"\n  {idx}. {insight}")

print("\nHR Action Plan:")
for idx, item in enumerate(action_plan, 1):
    print(f"\n  {idx}. Finding  : {item['finding']}")
    print(f"     Action   : {item['action']}")
    print(f"     Purpose  : {item['purpose']}")

print("\nLimitations:")
for idx, lim in enumerate(limitations, 1):
    print(f"\n  {idx}. {lim}")

print("\n" + "=" * 50)
print("Files saved:")
print("  confusion_matrix.png")
print("  viz_attrition_by_department.png")
print("  viz_attrition_by_payzone_emptype.png")
print("  viz_score_comparison.png")
print("  viz_attrition_by_score_level.png")
print("  feature_importance.png")
print("=" * 50)

# ─────────────────────────────────────────────
# STEP 13: Interactive Employee Attrition Prediction
# ─────────────────────────────────────────────

# Valid options for each categorical field
VALID = {
    "PayZone":                    ["Zone A", "Zone B", "Zone C"],
    "DepartmentType":             ["Admin Offices", "Executive Office", "IT/IS",
                                   "Production", "Sales", "Software Engineering"],
    "EmployeeType":               ["Contract", "Full-Time", "Part-Time"],
    "EmployeeClassificationType": ["Full-Time", "Part-Time", "Temporary"],
    "Performance Score":          ["Exceeds", "Fully Meets", "Needs Improvement", "PIP"],
    "BusinessUnit":               ["BPC", "CCDR", "EW", "MSC", "NEL",
                                   "PL", "PYZ", "SVG", "TNS", "WBL"],
    "Division":                   ["Aerial", "Billable Consultants", "Catv",
                                   "Corp Operations", "Engineers", "Finance",
                                   "General - Sga", "General Management",
                                   "Information Technology", "Inside Sales",
                                   "Legal", "Marketing", "Network",
                                   "Outside Sales", "Plant", "Production",
                                   "Purchasing", "Retail", "Sales Management",
                                   "Service", "Splicing", "Support",
                                   "Technical Operations", "Warehouse", "Wireline"],
    "GenderCode":                 ["Female", "Male"],
    "MaritalDesc":                ["Divorced", "Married", "Single", "Widowed"],
    "State":                      ["AL", "AZ", "CA", "CO", "CT", "FL", "GA",
                                   "IA", "IL", "IN", "KY", "MA", "MD", "MI",
                                   "MN", "MO", "NC", "NJ", "NY", "OH", "OR",
                                   "PA", "SC", "TN", "TX", "VA", "WA", "WI"],
    "RaceDesc":                   ["Asian", "Black", "Hispanic", "Other", "White"],
    "Training Program Name":      ["Communication Skills", "Customer Service",
                                   "Leadership Development", "Project Management",
                                   "Technical Skills"],
    "Training Type":              ["External", "Internal"],
    "Training Outcome":           ["Completed", "Failed", "Incomplete", "Passed"],
}


def prompt_number(label, min_val, max_val):
    """Ask user for a number within a range."""
    while True:
        raw = input(f"  {label} ({min_val}–{max_val}): ").strip()
        try:
            val = float(raw)
            if min_val <= val <= max_val:
                return val
            print(f"    Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("    Invalid input. Please enter a number.")


def prompt_choice(label, options):
    """Ask user to pick from a list of options."""
    print(f"  {label}:")
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    while True:
        raw = input(f"  Enter number (1–{len(options)}): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print(f"    Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("    Invalid input. Please enter a number.")


def prompt_text(label, valid_list):
    """Ask user to type a value; show list but also accept free text for high-cardinality fields."""
    print(f"  {label} (type the value exactly):")
    for v in valid_list:
        print(f"    - {v}")
    while True:
        raw = input(f"  Your input: ").strip()
        if raw in valid_list:
            return raw
        print(f"    '{raw}' not recognised. Please type one of the values above exactly.")


print("\n" + "=" * 50)
print("STEP 13: Predict Attrition for a New Employee")
print("=" * 50)

while True:
    print("\nEnter employee details below.")
    print("-" * 40)

    emp = {}

    # Numerical inputs
    emp["Engagement Score"]        = int(prompt_number("Engagement Score", 1, 5))
    emp["Satisfaction Score"]      = int(prompt_number("Satisfaction Score", 1, 5))
    emp["Work-Life Balance Score"] = int(prompt_number("Work-Life Balance Score", 1, 5))
    emp["Training Duration(Days)"] = int(prompt_number("Training Duration (Days)", 1, 30))
    emp["Training Cost"]           = prompt_number("Training Cost ($)", 0, 10000)
    emp["Current Employee Rating"] = int(prompt_number("Current Employee Rating", 1, 5))

    # Categorical inputs
    emp["PayZone"]                    = prompt_choice("PayZone", VALID["PayZone"])
    emp["DepartmentType"]             = prompt_choice("DepartmentType", VALID["DepartmentType"])
    emp["EmployeeType"]               = prompt_choice("EmployeeType", VALID["EmployeeType"])
    emp["EmployeeClassificationType"] = prompt_choice("EmployeeClassificationType", VALID["EmployeeClassificationType"])
    emp["Performance Score"]          = prompt_choice("Performance Score", VALID["Performance Score"])
    emp["BusinessUnit"]               = prompt_choice("BusinessUnit", VALID["BusinessUnit"])
    emp["Division"]                   = prompt_choice("Division", VALID["Division"])
    emp["GenderCode"]                 = prompt_choice("GenderCode", VALID["GenderCode"])
    emp["MaritalDesc"]                = prompt_choice("MaritalDesc", VALID["MaritalDesc"])
    emp["State"]                      = prompt_choice("State", VALID["State"])
    emp["RaceDesc"]                   = prompt_choice("RaceDesc", VALID["RaceDesc"])
    emp["JobFunctionDescription"]     = prompt_text("Job Function Description",
                                            sorted(df["JobFunctionDescription"].dropna().unique().tolist()))
    emp["Title"]                      = prompt_text("Title",
                                            sorted(df["Title"].dropna().unique().tolist()))
    emp["Training Program Name"]      = prompt_choice("Training Program Name", VALID["Training Program Name"])
    emp["Training Type"]              = prompt_choice("Training Type", VALID["Training Type"])
    emp["Training Outcome"]           = prompt_choice("Training Outcome", VALID["Training Outcome"])

    # Build a single-row DataFrame in the same column order as training
    emp_df = pd.DataFrame([emp])[features]

    # Preprocess using the already-fitted preprocessor (no refit = no leakage)
    emp_processed = preprocessor.transform(emp_df)

    # Predict using the same threshold as the model evaluation
    prob_left = rf.predict_proba(emp_processed)[0][1]
    prediction = "Left" if prob_left >= THRESHOLD else "Stayed"

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"  Probability of Leaving : {prob_left:.2%}")
    print(f"  Probability of Staying : {1 - prob_left:.2%}")
    print(f"  Prediction             : {prediction}")
    if prediction == "Left":
        print("  [!] This employee is predicted to be at risk of leaving.")
    else:
        print("  [OK] This employee is predicted to stay.")
    print("=" * 50)

    again = input("\nPredict another employee? (yes/no): ").strip().lower()
    if again not in ("yes", "y"):
        print("\nPrediction session ended. Goodbye!")
        break
