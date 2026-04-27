# ============================================================
# PYTHON PROJECT: India District-wise IPC Crimes (2001-2012)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PHASE 1: LOAD DATA
# ============================================================

file_path = "clean_data.csv"   # 👈 your cleaned file

if not os.path.exists(file_path):
    print("❌ File not found. Put clean_data.csv in same folder.")
    exit()

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Rename columns
df.rename(columns={
    df.columns[0]: 'State',
    df.columns[1]: 'District',
    df.columns[2]: 'Year'
}, inplace=True)

# Remove TOTAL rows
df = df[df['District'].astype(str).str.upper() != 'TOTAL'].copy()

# Convert numeric
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

crime_cols = df.columns[3:]
for col in crime_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Total crimes
df['TOTAL_CRIMES'] = df[crime_cols].sum(axis=1)

# Handle TOTAL IPC column
if 'TOTAL IPC CRIMES' in df.columns:
    df.rename(columns={'TOTAL IPC CRIMES': 'TOTAL_IPC'}, inplace=True)
else:
    df['TOTAL_IPC'] = df['TOTAL_CRIMES']

df.reset_index(drop=True, inplace=True)

print("\n✅ Data Loaded & Cleaned Successfully!")

# ============================================================
# DATA SUMMARY
# ============================================================

print("\n📊 Dataset Summary:")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")
print(f"Years Covered: {df['Year'].min()} to {df['Year'].max()}")
print(f"Total States: {df['State'].nunique()}")
print(f"Total Districts: {df['District'].nunique()}")

# ============================================================
# PHASE 2: EDA
# ============================================================

national_trend = df.groupby('Year')['TOTAL_IPC'].sum().reset_index()
national_trend.columns = ['Year', 'Total_Crimes']

state_total = df.groupby('State')['TOTAL_IPC'].sum().reset_index()
state_total.columns = ['State', 'Total_Crimes']
state_total = state_total.sort_values('Total_Crimes', ascending=False)

district_total = df.groupby(['State','District'])['TOTAL_IPC'].sum().reset_index()
district_total.columns = ['State','District','Total_Crimes']
district_total = district_total.sort_values('Total_Crimes', ascending=False)

crime_categories = [
    'MURDER','RAPE','KIDNAPPING & ABDUCTION','ROBBERY',
    'BURGLARY','THEFT','RIOTS','CHEATING','DOWRY DEATHS',
    'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'CAUSING DEATH BY NEGLIGENCE'
]

available = [c for c in crime_categories if c in df.columns]

year_2001 = df[df['Year']==2001][available].sum()
year_2012 = df[df['Year']==2012][available].sum()

crime_growth = pd.DataFrame({
    'Crime_Type': available,
    '2001': year_2001.values,
    '2012': year_2012.values
})

crime_growth['Growth_%'] = ((crime_growth['2012'] - crime_growth['2001']) /
                            crime_growth['2001'] * 100).round(2)

national_trend['YoY_Growth_%'] = national_trend['Total_Crimes'].pct_change()*100

# ============================================================
# FORMATTED OUTPUT (IMPORTANT)
# ============================================================

print("\n" + "="*60)
print("OBJECTIVE 1: National Crime Trend (2001–2012)")
print("="*60)
print(national_trend.to_string(index=False))

print("\n" + "="*60)
print("OBJECTIVE 2: Top 5 States")
print("="*60)
print(state_total.head(5).to_string(index=False))

print("\nBottom 5 States:")
print(state_total.tail(5).to_string(index=False))

print("\n" + "="*60)
print("OBJECTIVE 3: Crime Growth (2001 → 2012)")
print("="*60)
print(crime_growth.to_string(index=False))

print("\n" + "="*60)
print("OBJECTIVE 4: Year-over-Year Growth")
print("="*60)
print(national_trend.to_string(index=False))

print("\n" + "="*60)
print("OBJECTIVE 5: Top 10 Districts")
print("="*60)
print(district_total.head(10).to_string(index=False))

# ============================================================
# PHASE 3: VISUALIZATION
# ============================================================

sns.set_style("whitegrid")

# National trend
plt.figure(figsize=(10,5))
plt.plot(national_trend['Year'], national_trend['Total_Crimes'], marker='o')

for x,y in zip(national_trend['Year'], national_trend['Total_Crimes']):
    plt.text(x,y+30000,f"{int(y):,}",ha='center',fontsize=8)

plt.title("National IPC Crime Trend (2001–2012)")
plt.xlabel("Year")
plt.ylabel("Crimes")
plt.show()

# YoY growth
yoy = national_trend.dropna()
colors = ['green' if x>=0 else 'red' for x in yoy['YoY_Growth_%']]

plt.figure(figsize=(10,5))
plt.bar(yoy['Year'], yoy['YoY_Growth_%'], color=colors)
plt.axhline(0,color='black')
plt.title("Year-over-Year Growth")
plt.show()

# Heatmap
top_states = state_total.head(15)['State']
heatmap_data = df[df['State'].isin(top_states)].groupby(
    ['State','Year'])['TOTAL_IPC'].sum().reset_index()

pivot = heatmap_data.pivot(index='State',columns='Year',values='TOTAL_IPC')

plt.figure(figsize=(12,6))
sns.heatmap(pivot,cmap='YlOrRd',annot=True,fmt=".0f")
plt.title("Crime Heatmap")
plt.show()

# Stacked area
yearly_crime = df.groupby('Year')[available].sum()

plt.figure(figsize=(10,6))
yearly_crime.plot.area()
plt.title("Crime Composition by Category")
plt.show()

# ============================================================
# PHASE 4: MACHINE LEARNING
# ============================================================

print("\n🔷 LINEAR REGRESSION MODEL")

X = national_trend[['Year']]
y = national_trend['Total_Crimes']

model = LinearRegression()
model.fit(X,y)

y_pred = model.predict(X)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

print("\nMAE:",mean_absolute_error(y,y_pred))
print("RMSE:",np.sqrt(mean_squared_error(y,y_pred)))
print("R2:",r2_score(y,y_pred))

# Future predictions
future = pd.DataFrame({'Year': range(2013,2027)})
future_pred = model.predict(future)

print("\nFuture Predictions:")
for yr, pred in zip(future['Year'], future_pred):
    print(yr, ":", int(pred))

# Regression plot
all_years = pd.DataFrame({'Year': range(2001,2027)})
all_pred = model.predict(all_years)

plt.figure(figsize=(12,6))
plt.scatter(X['Year'],y,label='Actual')
plt.plot(all_years['Year'],all_pred,'--',label='Trend')

plt.axvspan(2012.5,2026,alpha=0.1)

plt.scatter(future['Year'],future_pred,color='orange',label='Prediction')

plt.title("Linear Regression Projection")
plt.xlabel("Year")
plt.ylabel("Crimes")
plt.legend()
plt.show()
