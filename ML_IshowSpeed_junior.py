import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from catboost import CatBoostClassifier, cv
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
from catboost import Pool
import shap


# считываем исходный датасет
df = pd.read_csv('dataset_main.csv')

# Преобразование столбцов с датами в формат datetime
df['client_start_date'] = pd.to_datetime(df['client_start_date'])
df['partnerrolestart_date'] = pd.to_datetime(df['partnerrolestart_date'])

# # Добавление столбца с месяцем, когда клиент был привлечен
df['client_start_month'] = df['client_start_date'].apply(lambda x: x.replace(day=1))
df['partnerrolestart_month'] = df['partnerrolestart_date'].apply(lambda x: x.replace(day=1))

# тут мы решили перейти от формата "одна строка - один партнер" к "одна строка - один партнер на момент конкретного месяца"

date_index = pd.DataFrame(list(product(df['clientbankpartner_pin'].unique(),pd.date_range('2019-03-01', '2020-12-01', freq='MS'))))

res = df.groupby(['clientbankpartner_pin', 'client_start_month'])['client_pin'].count().reset_index() \
  .merge(date_index, how='right', left_on=['clientbankpartner_pin', 'client_start_month'], right_on=[0, 1])


res.drop(columns = ['clientbankpartner_pin', 'client_start_month'], inplace=True)
res.columns = ['count_this_month', 'partner_id', 'month']
res['count_this_month'] = res['count_this_month'].fillna(0)


t = df.groupby(['clientbankpartner_pin','partnerrolestart_month'])['client_pin'].count().reset_index() \
  .groupby('clientbankpartner_pin')['partnerrolestart_month'].count().reset_index()
t[t['partnerrolestart_month'] > 1]


t = df.groupby(['clientbankpartner_pin','partner_src_type_ccode'])['client_pin'].count().reset_index() \
  .groupby('clientbankpartner_pin')['partner_src_type_ccode'].count().reset_index()
t[t['partner_src_type_ccode'] > 1]

# добавление partnerrolestart_month и partner_src_type_ccode
types_dict = df.groupby('clientbankpartner_pin')['partner_src_type_ccode'].min().reset_index()
start_date_dict = df.groupby('clientbankpartner_pin')['partnerrolestart_month'].min().reset_index()

res = res.merge(types_dict, how='left', left_on='partner_id', right_on='clientbankpartner_pin')
res = res.merge(start_date_dict, how='left', left_on='partner_id', right_on='clientbankpartner_pin')
res.drop(columns=['clientbankpartner_pin_y', 'clientbankpartner_pin_x'], inplace=True)

# удаляем месяца до начала сотрудничества с партнёром
res = res[res['month'] >= res['partnerrolestart_month']].copy()

# целевую переменную мы определили как "будет ли партнёр привлекать клиентов в следующие 3 месяца начиная с данного res_x "
res2 = res[['partner_id', 'month', 'count_this_month']]
res2_x = res2[res2['month'] < '2020-09-01']
filter = res2[(res2['month'] >= '2020-06-01') & (res2['month'] < '2020-09-01')].groupby('partner_id')['count_this_month'].sum() > 0
res2_x = res2_x.merge(filter, how='inner', on='partner_id')
res2_x = res2_x[res2_x['count_this_month_y']].drop(columns='count_this_month_y')
res2_x = res2_x.sort_values(['partner_id', 'month'], ascending=[True, True])

# создание целевой переменной 
res2_y = (res2[(res2['month'] >= '2020-09-01') & (res2['month'] < '2020-12-01')].groupby('partner_id')['count_this_month'].sum() > 0) \
        .astype(int).reset_index().rename(columns={'count_this_month':'target'})
res2_y = res2_y.merge(filter, how='inner', on='partner_id')
res2_y = res2_y[res2_y['count_this_month']].drop(columns='count_this_month')
res2_y = res2_y.set_index('partner_id')['target']


res2_submit = res2[res2['month'] < '2020-12-01']
filter_submit = res2[(res2['month'] >= '2020-09-01')].groupby('partner_id')['count_this_month'].sum() > 0
res2_submit = res2_submit.merge(filter_submit, how='inner', on='partner_id')
res2_submit = res2_submit[res2_submit['count_this_month_y']].drop(columns='count_this_month_y')
res2_submit = res2_submit.sort_values(['partner_id', 'month'], ascending=[True, True])

# генерация фичей


res['count_month-1'] = res.groupby('partner_id')['count_this_month'].shift(1)
res['count_month-2'] = res.groupby('partner_id')['count_this_month'].shift(2).fillna(0)
res['count_month-3'] = res.groupby('partner_id')['count_this_month'].shift(3).fillna(0)

# среднее кол-во привлечений за прошлые 3,6,9,12 месяцев
res['avg_3_months'] = res.groupby('partner_id')['count_month-1'].rolling(3, min_periods=1).mean().reset_index(0,drop=True)
res['avg_6_months'] = res.groupby('partner_id')['count_month-1'].rolling(6, min_periods=1).mean().reset_index(0,drop=True)
res['avg_9_months'] = res.groupby('partner_id')['count_month-1'].rolling(9, min_periods=1).mean().reset_index(0,drop=True)
res['avg_12_months'] = res.groupby('partner_id')['count_month-1'].rolling(12, min_periods=1).mean().reset_index(0,drop=True)

#максимальное кол-во привлечений за прошлые 3,6,9,12 месяцев
res['max_3_months'] = res.groupby('partner_id')['count_month-1'].rolling(3, min_periods=1).max().reset_index(0,drop=True)
res['max_6_months'] = res.groupby('partner_id')['count_month-1'].rolling(6, min_periods=1).max().reset_index(0,drop=True)
res['max_9_months'] = res.groupby('partner_id')['count_month-1'].rolling(9, min_periods=1).max().reset_index(0,drop=True)
res['max_12_months'] = res.groupby('partner_id')['count_month-1'].rolling(12, min_periods=1).max().reset_index(0,drop=True)

#минимальное кол-во привлечений за прошлые 3,6,9,12 месяцев
res['min_3_months'] = res.groupby('partner_id')['count_month-1'].rolling(3, min_periods=1).min().reset_index(0,drop=True)
res['min_6_months'] = res.groupby('partner_id')['count_month-1'].rolling(6, min_periods=1).min().reset_index(0,drop=True)
res['min_9_months'] = res.groupby('partner_id')['count_month-1'].rolling(9, min_periods=1).min().reset_index(0,drop=True)
res['min_12_months'] = res.groupby('partner_id')['count_month-1'].rolling(12, min_periods=1).min().reset_index(0,drop=True)

# стандартное отклонение за прошлые 3,6,9,12 месяцев
res['std_3_months'] = res.groupby('partner_id')['count_month-1'].rolling(3, min_periods=1).std().reset_index(0,drop=True).fillna(0)
res['std_6_months'] = res.groupby('partner_id')['count_month-1'].rolling(6, min_periods=1).std().reset_index(0,drop=True).fillna(0)
res['std_9_months'] = res.groupby('partner_id')['count_month-1'].rolling(9, min_periods=1).std().reset_index(0,drop=True).fillna(0)
res['std_12_months'] = res.groupby('partner_id')['count_month-1'].rolling(12, min_periods=1).std().reset_index(0,drop=True).fillna(0)

#относительное стандартное отклонение за прошлые 3,6,9,12 месяцев
res['rel_std_3_months'] = res['std_3_months'] / np.maximum(res['avg_3_months'], 1)
res['rel_std_6_months'] = res['std_6_months'] / np.maximum(res['avg_6_months'], 1)
res['rel_std_9_months'] = res['std_9_months'] / np.maximum(res['avg_9_months'], 1)
res['rel_std_12_months'] = res['std_12_months'] / np.maximum(res['avg_12_months'], 1)

#скользящие средние за последние 3, 6, 9 и 12 месяцев для разницы между значениями 'count_month-1' и 'count_month-2'.
res['diff'] = res['count_month-1'] - res['count_month-2'].fillna(0)
res['avg_diff_3_months'] = res.groupby('partner_id')['diff'].rolling(3, min_periods=1).mean().reset_index(0,drop=True)
res['avg_diff_6_months'] = res.groupby('partner_id')['diff'].rolling(6, min_periods=1).mean().reset_index(0,drop=True)
res['avg_diff_9_months'] = res.groupby('partner_id')['diff'].rolling(9, min_periods=1).mean().reset_index(0,drop=True)
res['avg_diff_12_months'] = res.groupby('partner_id')['diff'].rolling(12, min_periods=1).mean().reset_index(0,drop=True)

# то же самое для относительных значений
res['rel_diff'] = (res['count_month-1'] - res['count_month-2'].fillna(0))/max(res['count_month-2'].fillna(0))
res['avg_rel_diff_3_months'] = res.groupby('partner_id')['diff'].rolling(3, min_periods=1).mean().reset_index(0,drop=True)
res['avg_rel_diff_6_months'] = res.groupby('partner_id')['diff'].rolling(6, min_periods=1).mean().reset_index(0,drop=True)
res['avg_rel_diff_9_months'] = res.groupby('partner_id')['diff'].rolling(9, min_periods=1).mean().reset_index(0,drop=True)
res['avg_rel_diff_12_months'] = res.groupby('partner_id')['diff'].rolling(12, min_periods=1).mean().reset_index(0,drop=True)

# время сотрудничества с партнёром
res['days_since_start'] = np.maximum(0, (res['month'] - res['partnerrolestart_month']).dt.days)

# активность партнёра
res['is_zero_last'] = ((res['count_month-1']==0)).astype(int)
res['two_zeros_last'] = ((res['count_month-1']==0) & (res['count_month-2']==0)).astype(int)
res['count_zeros_lifetime'] = res.groupby('partner_id')['is_zero_last'].cumsum()

res['month_num'] = res['month'].dt.month

# определение таргета для всего датасета
res['target'] = (res['count_this_month'] + res.groupby('partner_id')['count_this_month'].shift(-1).fillna(0) + \
                 res.groupby('partner_id')['count_this_month'].shift(-2).fillna(0) > 0).astype(int)
res.loc[(res['target']==0) & (res['month'] > '2020-09-01'), 'target'] = np.nan

final = res[
    ~res['count_month-1'].isna() &
    ~res['target'].isna() &
    (res['avg_3_months'] > 0)
].drop(columns=['count_this_month', 'partnerrolestart_month',])

X_submit = res[(res['month']=='2020-12-01') & (res['avg_3_months'] > 0)].drop(columns=['count_this_month', 'partnerrolestart_month'])


X = final.drop(columns=['partner_id', 'month','target'])
y = final['target']

# X = final_tsfresh
# y = res2_y

# Разделяем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = {
    'iterations':400,
    'learning_rate':0.01,
    'depth':4
  }
fit_params = {
    'early_stopping_rounds': 10,
    'verbose': 0
}
# model = CatBoostClassifier(**params)
# model.fit(X_train, y_train, **fit_params)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Делаем предсказания на тестовом наборе
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Выводим отчет классификации
print("Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC score:", roc_auc)

model = CatBoostClassifier()
cv_scores = cross_val_score(model.set_params(**params), X, y, scoring='roc_auc', verbose=3, fit_params=fit_params)
avg_cv_score = cv_scores.mean()
print(f'Average ROC-AUC across 5 folds: {avg_cv_score}')



# Create an instance of the XGBClassifier
model2 = XGBClassifier(objective='binary:logistic')

# Fit the model to the training data
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)
y_pred_proba2 = model2.predict_proba(X_test)[:, 1]

# Выводим отчет классификации
print("Classification Report:")
print(classification_report(y_test, y_pred2))

roc_auc = roc_auc_score(y_test, y_pred_proba2)
print("ROC-AUC score:", roc_auc)


params_grid = {
    'iterations': [400, 700, 1000],
    'learning_rate': [0.1, 0.05, 0.01],
    'depth': [4, 6],
}
fit_params = {
    'early_stopping_rounds': 10,
    'verbose': 0
}

# кросс-валидация для модели Catboost
for iter in params_grid['iterations']:
    for lr in params_grid['learning_rate']:
        for d in params_grid['depth']:
          params = {
              'iterations': iter,
              'learning_rate': lr,
              'depth': d
          }
          print(params)
          model = CatBoostClassifier()
          cv_scores = cross_val_score(model.set_params(**params), X, y, scoring='roc_auc', verbose=3, fit_params=fit_params)
          avg_cv_score = cv_scores.mean()
          print(f'Average ROC-AUC across 5 folds: {avg_cv_score}')

X_subm = X_submit.drop(columns=['partner_id', 'month', 'target'])

model = CatBoostClassifier(**params)
model.fit(X, y, **fit_params)
y_subm = model.predict_proba(X_subm)

# получение и сохранение предсказаний после кросс валидации

X_submit['target'] = 1 - y_subm
X_submit[['partner_id', 'target']].rename(columns={'partner_id':'clientbankpartner_pin', 'target':'score'}).to_csv('submission_probs.csv', index=False)
