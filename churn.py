import pandas as pd

df = pd.read_csv('teleco_churn.csv')

print(df.head())
print(df.shape)
print(df.info())

print(df['Churn'].value_counts(normalize=True))


print(df.groupby('Churn')['tenure'].describe())

print(df.groupby('Contract')['Churn'].value_counts(normalize=True))

for col in df.columns:
    unique_count = df[col].unique()
    print(f'{col}: {unique_count}')
  
#bining for tenure

df['tenure_bin'] = pd.cut(
    df['tenure'],
    bins = [0,12,24,48,72],
    labels= ['0-1y', '1-2y', '2-4y', '4y+']    
)    
#contract flag
df['is_monthly'] = (df['Contract'] == 'Month-to-month').astype(int)
df['is_long_term'] = df['Contract'].isin(['One year','Two year']).astype(int)

#cost pressure flag

df['high_monthly_charge'] = (
    df['MonthlyCharges'] > df['MonthlyCharges'].median()
).astype(int)

#new customer flag
df['is_new_customer'] = (df['tenure'] <= 6).astype(int)

#service complexity

services = [
    'PhoneService',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies'    
]

df['num_services'] = df[services].apply (
    lambda X : (X=='Yes').sum(),
    axis = 1
)

#TotalCharges = object

df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'], errors='coerce'
)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

print(df.columns)

X = df.drop(columns=['customerID', 'Churn'])
y = (df['Churn'] == 'Yes').astype(int)

print(df.groupby('tenure_bin')['Churn'].value_counts(normalize=True))
print(df.groupby('is_monthly')['Churn'].value_counts(normalize=True))
print(df.groupby('is_new_customer')['Churn'].value_counts(normalize=True))

print(df.groupby('num_services')['Churn'].value_counts(normalize=True))

num_cols = X.select_dtypes(include=['int64','float64']).columns.to_list()

cat_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns.to_list()

#pipeline without leakage

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression

def safe_log1p(x):
    return np.log1p(x)

preprocess = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('log', FunctionTransformer (safe_log1p, feature_names_out= 'one-to-one')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', OneHotEncoder (drop= 'first', handle_unknown= 'ignore'), cat_cols)
        
    ]
)

#baseline model

model = LogisticRegression(
    max_iter= 2000,
    class_weight= 'balanced',
    solver= 'liblinear'
)

# pipeline نهایی

pipe = Pipeline (steps=[
    ('prep', preprocess),
    ('model', model)
])

#test /split train
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state= 42,
    stratify= y 
)


# model training
pipe.fit(X_train, y_train)

# #ارزیابی 
from sklearn.metrics import classification_report, average_precision_score 

y_proba = pipe.predict_proba(X_test)[:,1]
y_pred = (y_proba >= 0.5).astype(int)

print(classification_report (y_test, y_pred, digits = 3))
print('PR-AUC:', average_precision_score(y_test, y_proba))

#features names

feature_names = pipe.named_steps['prep'].get_feature_names_out()
coef = pipe.named_steps['model'].coef_[0]

coef_df = pd.DataFrame({
    'feature': feature_names,
    'coef': coef
}).sort_values (by= 'coef', ascending= False)

coef_df ['odds_ratio'] = np.exp(coef_df['coef'])
print (coef_df)
print(coef_df.head(20))

#feature selection

drop_cols = [
    'tenure',
    'is_new_customer'
]

X_reduced = X.drop(columns= drop_cols, errors= 'raise')
print('tenure' in X_reduced.columns)

num_cols_red = X_reduced.select_dtypes(include=['int64', 'float64']).columns.to_list()
cat_cols_red = X_reduced.select_dtypes(include= ['object','bool', 'category']).columns.to_list()

def safe_log1p(x):
    return np.log1p(x)

preprocess_red = ColumnTransformer(
    transformers= [
        ('num', Pipeline([
            ('log', FunctionTransformer (safe_log1p, feature_names_out= 'one-to-one')),
            ('scaler', StandardScaler())
        ]), num_cols_red),
        ('cat', OneHotEncoder (drop='first', handle_unknown= 'ignore'), cat_cols_red)
    ]
)

model = LogisticRegression (max_iter= 2000, class_weight= 'balanced', solver = 'liblinear')

pipe_red = Pipeline (steps=[
    ('prep', preprocess_red),
    ('model', model)
])

X_train_red, X_test_red, y_train, y_test = train_test_split (
    X_reduced, y,
    test_size= 0.2,
    random_state= 42,
    stratify=y
)

pipe_red.fit(X_train_red,y_train)

y_proba_2 = pipe_red.predict_proba(X_test_red)[:,1]
y_pred_2 = (y_proba_2 >= 0.5).astype(int)

print(classification_report (y_test, y_pred_2 ,digits=3))
print('PR-AUC',average_precision_score (y_test, y_proba_2))

#model comparision with cross Validation 

from sklearn.model_selection import StratifiedKFold , cross_val_score

cv = StratifiedKFold (
    n_splits = 5,
    shuffle= True,
    random_state= 42
)

log_model = LogisticRegression(
    max_iter= 2000,
    class_weight= 'balanced',
    solver= 'liblinear'
)

pipe_log = Pipeline (steps= [
    ('prep', preprocess_red),
    ('model', log_model)
])

log_scores = cross_val_score(
    pipe_log,
    X_reduced,
    y,
    cv= cv,
    scoring= 'average_precision'
)

print('Logistic')

print('PR-AUC mean:' , log_scores.mean())
print('PR-AUC std:', log_scores.std())

# مدل دوم Gradiant Boosting

from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier (
    n_estimators= 200,
    learning_rate= 0.05,
    max_depth= 3,
    random_state=42
)

pipe_gb = Pipeline(steps=[
    ('prep', preprocess_red),
    ('model', gb_model)
])

gb_scores = cross_val_score(
    pipe_gb,
    X_reduced,
    y,
    cv = cv,
    scoring= 'average_precision'
)

print('Gradient Boosting')
print('PR-AUC mean:', gb_scores.mean())
print('PR-AUC std:', gb_scores.std())

#مدل نهایی همان Logistic Regression 
#threshold tuning

from sklearn.metrics import precision_score, recall_score

thresholds = np.arange(0.05,0.95,0.05)

results = []



for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    p = precision_score(y_test, y_pred_t)
    r = recall_score (y_test, y_pred_t)
    results.append((t,p,r))
    
for t,p,r in results:
    print(f'Threshold = {t:.2f}| Precision = {p:.3f} | Recall = {r:.3f}')    
    
    
    