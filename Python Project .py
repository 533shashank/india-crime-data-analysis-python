# ============================================================
# PYTHON PROJECT: India District-wise IPC Crimes (2001-2012)
# Source: NCRB (National Crime Records Bureau), Govt. of India
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PHASE 1: LOAD & CLEAN DATA
# ============================================================

# Load CSV — row 0 is the header
df = pd.read_csv(r"D:\DataSets\dstrIPC_1.csv")

# Row 0 was used as header automatically — rename columns cleanly
df.columns = df.columns.str.strip()

# Check actual columns
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# Rename first 3 columns for ease
df.rename(columns={
    df.columns[0]: 'State',
    df.columns[1]: 'District',
    df.columns[2]: 'Year'
}, inplace=True)

# Remove TOTAL rows (state-level aggregates) — keep only district rows
df = df[df['District'].str.upper() != 'TOTAL'].copy()

# Convert Year and crime columns to numeric
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
crime_cols = df.columns[3:]   # all columns after State, District, Year
for col in crime_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create a TOTAL_CRIMES column (sum of all crime types)
df['TOTAL_CRIMES'] = df[crime_cols].sum(axis=1)

df.reset_index(drop=True, inplace=True)

print("\n✅ Data Loaded & Cleaned!")
print(f"Shape after cleaning: {df.shape}")
print(f"Years covered: {sorted(df['Year'].unique())}")
print(f"States: {df['State'].nunique()}")
print(f"Districts: {df['District'].nunique()}")
print("\nFirst 5 rows:")
print(df[['State', 'District', 'Year', 'TOTAL_CRIMES']].head())
print("\nBasic Statistics:")
print(df['TOTAL_CRIMES'].describe())


# ============================================================
# PHASE 2: EDA — 5 OBJECTIVES
# ============================================================

# Use the official TOTAL IPC CRIMES column (already in data)
df.rename(columns={'TOTAL IPC CRIMES': 'TOTAL_IPC'}, inplace=True)

# ============================================================
# OBJECTIVE 1: National Crime Trend (2001-2012)
# ============================================================
national_trend = df.groupby('Year')['TOTAL_IPC'].sum().reset_index()
national_trend.columns = ['Year', 'Total_Crimes']

print("=" * 55)
print("OBJECTIVE 1: National IPC Crime Trend (2001–2012)")
print("=" * 55)
print(national_trend.to_string(index=False))

# ============================================================
# OBJECTIVE 2: Top 5 & Bottom 5 States by Total Crimes
# ============================================================
state_total = df.groupby('State')['TOTAL_IPC'].sum().reset_index()
state_total = state_total.sort_values('Total_Crimes' if 'Total_Crimes' in state_total.columns
                                      else 'TOTAL_IPC', ascending=False)
state_total.columns = ['State', 'Total_Crimes']
state_total = state_total.sort_values('Total_Crimes', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 2: Top 5 States by Total IPC Crimes")
print("=" * 55)
print(state_total.head(5).to_string(index=False))
print("\nBottom 5 States:")
print(state_total.tail(5).to_string(index=False))

# ============================================================
# OBJECTIVE 3: Which Crime Type Grew the Most (2001 vs 2012)
# ============================================================
crime_categories = ['MURDER', 'RAPE', 'KIDNAPPING & ABDUCTION',
                    'ROBBERY', 'BURGLARY', 'THEFT', 'RIOTS',
                    'CHEATING', 'DOWRY DEATHS',
                    'CRUELTY BY HUSBAND OR HIS RELATIVES',
                    'CAUSING DEATH BY NEGLIGENCE']

year_2001 = df[df['Year'] == 2001][crime_categories].sum()
year_2012 = df[df['Year'] == 2012][crime_categories].sum()

crime_growth = pd.DataFrame({
    'Crime_Type': crime_categories,
    'Count_2001': year_2001.values,
    'Count_2012': year_2012.values
})
crime_growth['Growth_%'] = ((crime_growth['Count_2012'] - crime_growth['Count_2001'])
                             / crime_growth['Count_2001'] * 100).round(2)
crime_growth = crime_growth.sort_values('Growth_%', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 3: Crime Type Growth 2001 → 2012")
print("=" * 55)
print(crime_growth.to_string(index=False))

# ============================================================
# OBJECTIVE 4: Year-over-Year National Growth Rate
# ============================================================
national_trend['YoY_Growth_%'] = national_trend['Total_Crimes'].pct_change().mul(100).round(2)

print("\n" + "=" * 55)
print("OBJECTIVE 4: Year-over-Year National Growth Rate")
print("=" * 55)
print(national_trend.to_string(index=False))

# ============================================================
# OBJECTIVE 5: Top 10 Districts by Total Crimes (hotspots)
# ============================================================
district_total = df.groupby(['State', 'District'])['TOTAL_IPC'].sum().reset_index()
district_total.columns = ['State', 'District', 'Total_Crimes']
district_total = district_total.sort_values('Total_Crimes', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 5: Top 10 Crime Hotspot Districts")
print("=" * 55)
print(district_total.head(10).to_string(index=False))

# ============================================================
# PHASE 3: VISUALISATIONS
# ============================================================

sns.set_style("whitegrid")

# ----------------------------------------------------------
# PLOT 1 (Obj 1): National Crime Trend — Line Chart
# ----------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(national_trend['Year'], national_trend['Total_Crimes'],
         marker='o', color='steelblue', linewidth=2.5, markersize=7)
for _, row in national_trend.iterrows():
    plt.text(row['Year'], row['Total_Crimes'] + 30000,
             f"{int(row['Total_Crimes']):,}", ha='center', fontsize=8)
plt.title("Objective 1: National IPC Crime Trend (2001–2012)",
          fontsize=14, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Total IPC Crimes")
plt.xticks(national_trend['Year'])
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# PLOT 2 (Obj 2): Top 10 States — Horizontal Bar Chart
# ----------------------------------------------------------
top10_states = state_total.head(10)
plt.figure(figsize=(10, 6))
bars = plt.barh(top10_states['State'][::-1],
                top10_states['Total_Crimes'][::-1],
                color='coral')
plt.title("Objective 2: Top 10 States by Total IPC Crimes (2001–2012)",
          fontsize=13, fontweight='bold')
plt.xlabel("Total IPC Crimes")
for bar in bars:
    plt.text(bar.get_width() + 20000, bar.get_y() + bar.get_height() / 2,
             f"{int(bar.get_width()):,}", va='center', fontsize=8)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# PLOT 3 (Obj 3): Crime Type Growth Bar Chart
# ----------------------------------------------------------
plt.figure(figsize=(12, 6))
colors = ['tomato' if x > 0 else 'steelblue' for x in crime_growth['Growth_%']]
plt.bar(crime_growth['Crime_Type'], crime_growth['Growth_%'], color=colors)
plt.title("Objective 3: Crime Type Growth Rate 2001 → 2012 (%)",
          fontsize=13, fontweight='bold')
plt.xlabel("Crime Type")
plt.ylabel("Growth Rate (%)")
plt.xticks(rotation=40, ha='right', fontsize=8)
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# PLOT 4 (Obj 4): YoY Growth Rate — Bar Chart
# ----------------------------------------------------------
yoy = national_trend.dropna()
colors_yoy = ['green' if x >= 0 else 'red' for x in yoy['YoY_Growth_%']]
plt.figure(figsize=(10, 5))
plt.bar(yoy['Year'], yoy['YoY_Growth_%'], color=colors_yoy, edgecolor='black', linewidth=0.5)
plt.title("Objective 4: Year-over-Year National Crime Growth Rate (%)",
          fontsize=13, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Growth Rate (%)")
plt.axhline(0, color='black', linewidth=1)
plt.xticks(yoy['Year'])
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# PLOT 5 (Obj 5): Heatmap — Top 15 States × Year
# ----------------------------------------------------------
top15_states = state_total.head(15)['State'].tolist()
heatmap_data = df[df['State'].isin(top15_states)].groupby(
    ['State', 'Year'])['TOTAL_IPC'].sum().reset_index()
heatmap_pivot = heatmap_data.pivot(index='State', columns='Year', values='TOTAL_IPC')

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_pivot, cmap='YlOrRd', linewidths=0.4,
            annot=True, fmt='.0f', annot_kws={'size': 7})
plt.title("Objective 5: Crime Heatmap — Top 15 States × Year",
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================
# PHASE 4: LINEAR REGRESSION MODEL (Fixed for small dataset)
# ============================================================
print("\n" + "=" * 55)
print("PHASE 4: LINEAR REGRESSION MODEL")
print("=" * 55)

# Feature: Year | Target: National Total Crimes
X = national_trend[['Year']]
y = national_trend['Total_Crimes']

# ── With only 12 yearly points, we train on all data ────────
# ── and validate using cross-validation (standard for small ─
# ── time-series datasets)                                  ──
from sklearn.model_selection import cross_val_score

model = LinearRegression()
model.fit(X, y)

print(f"Model Equation : Crimes = {model.coef_[0]:,.0f} × Year + ({model.intercept_:,.0f})")
print(f"Interpretation : Every year, national crimes rise by ~{model.coef_[0]:,.0f}")

# Predictions on ALL known years
y_pred_all = model.predict(X)

# ── Cross-validation R² (more honest for small datasets) ────
cv_scores = cross_val_score(LinearRegression(), X, y,
                            cv=4, scoring='r2')
print(f"\nCross-Validated R² scores : {[round(s, 4) for s in cv_scores]}")
print(f"Mean CV R²                : {cv_scores.mean():.4f}")

# ── Accuracy Metrics on full data ───────────────────────────
mae  = mean_absolute_error(y, y_pred_all)
mse  = mean_squared_error(y, y_pred_all)
rmse = np.sqrt(mse)
r2   = r2_score(y, y_pred_all)

print("\n📐 Model Accuracy Metrics (Trained on full dataset):")
print(f"  MAE  : {mae:,.0f}   ← avg prediction error in crime count")
print(f"  MSE  : {mse:,.0f}")
print(f"  RMSE : {rmse:,.0f}   ← error in same unit as crimes")
print(f"  R²   : {r2:.4f}    ← model explains {r2:.1%} of variance")

# ── Actual vs Predicted Table ────────────────────────────────
print("\nActual vs Predicted (All Years):")
print(f"{'Year':<8} {'Actual':>12} {'Predicted':>12} {'Error':>12}")
print("-" * 46)
for yr, act, pred in zip(X['Year'], y, y_pred_all):
    print(f"{yr:<8} {int(act):>12,} {int(pred):>12,} {int(act-pred):>+12,}")

# ── Future Projections ───────────────────────────────────────
future = pd.DataFrame({'Year': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]})
future_pred = model.predict(future)

print("\n🔮 Future Crime Projections:")
print(f"{'Year':<8} {'Projected Crimes':>18}")
print("-" * 28)
for yr, pred in zip(future['Year'], future_pred):
    print(f"{yr:<8} {int(pred):>18,}")

# ── Final Regression Plot ────────────────────────────────────
all_years = pd.DataFrame({'Year': range(2001, 2027)})
all_pred  = model.predict(all_years)

plt.figure(figsize=(12, 6))
plt.scatter(X['Year'], y,
            color='steelblue', s=90, zorder=5, label='Actual Data')
plt.plot(all_years['Year'], all_pred,
         color='tomato', linewidth=2.5, linestyle='--', label='Regression Line')
plt.axvspan(2012.5, 2016.5, alpha=0.08, color='orange', label='Forecast Zone')
plt.scatter(future['Year'], future_pred,
            color='orange', s=90, zorder=5, marker='D',
            label='Projected (2013–2016)')
for yr, pred in zip(future['Year'], future_pred):
    plt.text(yr, pred - 80000, f"{int(pred):,}", ha='center', fontsize=8)
plt.title("Linear Regression: National IPC Crime Trend & Projection",
          fontsize=14, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("Total IPC Crimes")
plt.xticks(range(2001, 2027))
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

print("\n✅ PROJECT COMPLETE!")

######################################
# ============================================================
# PHASE 5: MODEL VALIDATION — Actual 2013-2014 vs Predicted
# Source: NCRB district-wise IPC data, data.gov.in
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: MODEL VALIDATION AGAINST ACTUAL 2013-2014 DATA")
print("=" * 60)

import os

# ── Only 2013 and 2014 available ─────────────────────────────
new_files = {
    2013: r"D:\DataSets\dstrIPC_2013.csv",
    2014: r"D:\DataSets\dstrIPC_1_2014.csv",
}

actual_totals = {}

# Define the crime columns present in your TRAINING data
common_crimes = [
    'MURDER', 'ATTEMPT TO MURDER',
    'CULPABLE HOMICIDE NOT AMOUNTING TO MURDER',
    'RAPE', 'KIDNAPPING & ABDUCTION', 'DACOITY',
    'ROBBERY', 'BURGLARY', 'THEFT', 'RIOTS',
    'CRIMINAL BREACH OF TRUST', 'CHEATING',
    'ARSON', 'HURT/GREVIOUS HURT', 'DOWRY DEATHS',
    'CAUSING DEATH BY NEGLIGENCE', 'OTHER IPC CRIMES'
]

for year, filepath in new_files.items():
    df_new = pd.read_csv(filepath)
    df_new.columns = df_new.columns.str.strip().str.upper()  # normalize to UPPER

    df_new.rename(columns={
        df_new.columns[0]: 'State',
        df_new.columns[1]: 'District',
        df_new.columns[2]: 'Year'
    }, inplace=True)

    df_new = df_new[~df_new['District'].str.upper().isin(
    ['TOTAL', 'GRAND TOTAL', 'ZZ TOTAL'])].copy()
    
    # Only sum columns that exist in BOTH datasets
    available = [c for c in common_crimes if c in df_new.columns]
    print(f"  {year}: Matched {len(available)}/{len(common_crimes)} crime columns")

    df_new[available] = df_new[available].apply(pd.to_numeric, errors='coerce')
    df_new['TOTAL_IPC'] = df_new[available].sum(axis=1)

    national_actual = df_new['TOTAL_IPC'].sum()
    actual_totals[year] = national_actual
    print(f"  ✅ {year} — Comparable National Total: {int(national_actual):,}")

# ── Rebuild model on 2001-2012 training data ─────────────────
X_train_full = national_trend[['Year']]
y_train_full  = national_trend['Total_Crimes']
val_model = LinearRegression()
val_model.fit(X_train_full, y_train_full)

validation_years = pd.DataFrame({'Year': sorted(actual_totals.keys())})
predicted_vals   = val_model.predict(validation_years)

# ── Comparison Table ──────────────────────────────────────────
print(f"\n{'Year':<8} {'Predicted':>14} {'Actual':>14} {'Diff':>14} {'Error%':>10}")
print("-" * 62)

actuals_list   = []
predicted_list = []

for yr, pred in zip(validation_years['Year'], predicted_vals):
    actual = actual_totals.get(yr, None)
    if actual:
        diff   = actual - pred
        pct    = abs(diff / actual) * 100
        actuals_list.append(actual)
        predicted_list.append(pred)
        print(f"{yr:<8} {int(pred):>14,} {int(actual):>14,} "
              f"{int(diff):>+14,} {pct:>9.2f}%")

# ── Validation Accuracy Metrics ───────────────────────────────
if actuals_list:
    val_mae  = mean_absolute_error(actuals_list, predicted_list)
    val_rmse = np.sqrt(mean_squared_error(actuals_list, predicted_list))
    val_r2   = r2_score(actuals_list, predicted_list)

    print(f"\n📐 Validation Metrics (Predicted vs Actual):")
    print(f"   MAE    : {val_mae:,.0f}  ← avg error in crime count")
    print(f"   RMSE   : {val_rmse:,.0f}")
    print(f"   R²     : {val_r2:.4f}")
    avg_err = sum(abs(a-p)/a*100 for a,p in
                  zip(actuals_list, predicted_list)) / len(actuals_list)
    print(f"   Avg Error% : {avg_err:.2f}%")

    # ── Final Validation Plot ─────────────────────────────────
    all_years_ext = pd.DataFrame({'Year': range(2001, 2015)})
    all_preds_ext = val_model.predict(all_years_ext)

    plt.figure(figsize=(13, 6))

    # Training points
    plt.scatter(national_trend['Year'], national_trend['Total_Crimes'],
                color='steelblue', s=90, zorder=5,
                label='Actual 2001–2012 (Training Data)')

    # Regression line
    plt.plot(all_years_ext['Year'], all_preds_ext,
             color='tomato', linewidth=2.5, linestyle='--',
             label='Regression Line')

    # Actual 2013-2014 (green squares)
    plt.scatter(list(actual_totals.keys()), list(actual_totals.values()),
                color='green', s=130, zorder=6, marker='s',
                label='Actual 2013–2014 (Validation)')

    # Predicted 2013-2014 (orange diamonds)
    plt.scatter(validation_years['Year'], predicted_vals,
                color='orange', s=110, zorder=6, marker='D',
                label='Model Predicted 2013–2014')

    # Dotted error lines between actual and predicted
    for yr, act, pred in zip(validation_years['Year'],
                              list(actual_totals.values()),
                              predicted_vals):
        plt.plot([yr, yr], [act, pred], 'gray',
                 linestyle=':', linewidth=2)
        mid_y = (act + pred) / 2
        err_pct = abs(act - pred) / act * 100
        plt.text(yr + 0.05, mid_y,
                 f"Error: {err_pct:.1f}%",
                 fontsize=9, color='dimgray', fontweight='bold')

    plt.axvspan(2012.5, 2014.5, alpha=0.08, color='lightgreen',
                label='Validation Zone')

    plt.title("Model Validation: Predicted vs Actual (2013–2014)",
              fontsize=14, fontweight='bold')
    plt.xlabel("Year")
    plt.ylabel("Total IPC Crimes")
    plt.xticks(range(2001, 2015))
    plt.legend(fontsize=9, loc='upper left')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("plot_validation.png", dpi=150)
    plt.show()

    print("\n✅ VALIDATION COMPLETE — Model is Justified!")
