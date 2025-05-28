import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. Load data
df = pd.read_csv('C:/Users/HP PAVILION 15 CS/OneDrive/loan_default_model_Ren/data/processed/cleaned_data.csv')

X = df.drop('fpd_15', axis=1)
y = df['fpd_15']

# 2. Preprocessor
num_feats = X.select_dtypes(include=['float','int']).columns.tolist()
cat_feats = X.select_dtypes(include=['object','category']).columns.tolist()

num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
])
cat_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('ohe', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_feats),
    ('cat', cat_pipe, cat_feats),
])

# 3. Full pipeline
model = Pipeline([
    ('preproc', preprocessor),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
])

# 4. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"Mean ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")
# 5. Fit the model
model.fit(X, y)
# 6. Save the model
import os
os.makedirs('C:/Users/HP PAVILION 15 CS/OneDrive/loan_default_model_Ren/models', exist_ok=True)
import joblib
joblib.dump(model, 'C:/Users/HP PAVILION 15 CS/OneDrive/loan_default_model_Ren/models/xgb_model.pkl')
# 7. Save the preprocessor


