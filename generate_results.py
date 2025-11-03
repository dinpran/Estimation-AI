import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
import ssl
import time
import warnings
warnings.filterwarnings('ignore')

ssl._create_default_https_context = ssl._create_unverified_context

print("=" * 80)
print("GENERATING RESULTS FOR RESEARCH PAPER")
print("=" * 80)

# CO2 EMISSIONS
print("\n\nTABLE 1: CO2 EMISSION ESTIMATION")
print("-" * 80)

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
data = pd.read_csv(url)
X = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']].values
y = data['CO2EMISSIONS'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

algorithms = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR()
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"{'Algorithm':<20} {'R2 Mean':<12} {'RMSE Mean':<12} {'Time (s)':<10}")
print("-" * 80)

for name, model in algorithms.items():
    start = time.time()
    r2 = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
    rmse = np.sqrt(-cross_val_score(model, X_scaled, y, cv=kfold, scoring='neg_mean_squared_error'))
    train_time = time.time() - start
    
    print(f"{name:<20} {r2.mean():>6.3f}±{r2.std():.3f}  {rmse.mean():>6.2f}±{rmse.std():.2f}  {train_time:>8.3f}")

# CHURN PREDICTION
print("\n\nTABLE 2: CUSTOMER CHURN PREDICTION")
print("-" * 80)

url2 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_data = pd.read_csv(url2)
churn_data = churn_data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_data['churn'] = churn_data['churn'].astype('int')

X_churn = churn_data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']].values
y_churn = churn_data['churn'].values

scaler2 = StandardScaler()
X_churn_scaled = scaler2.fit_transform(X_churn)

classifiers = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42)
}

print(f"{'Algorithm':<20} {'Accuracy':<12} {'F1-Score':<12} {'Time (s)':<10}")
print("-" * 80)

for name, model in classifiers.items():
    start = time.time()
    acc = cross_val_score(model, X_churn_scaled, y_churn, cv=kfold, scoring='accuracy')
    f1 = cross_val_score(model, X_churn_scaled, y_churn, cv=kfold, scoring='f1')
    train_time = time.time() - start
    
    print(f"{name:<20} {acc.mean():>6.3f}±{acc.std():.3f}  {f1.mean():>6.3f}±{f1.std():.3f}  {train_time:>8.3f}")

print("\n" + "=" * 80)
print("DONE! Copy these results to your research paper!")
print("=" * 80)
