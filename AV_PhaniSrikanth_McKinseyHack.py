import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

print "Reading Datasets."
train = pd.read_csv("Train_psolI3n.csv")
test = pd.read_csv("Test_09JmpYa.csv")
submit = pd.read_csv("Sample_Submission_NeKh8xT.csv")

train_id, test_id = train['Email_ID'], test['Email_ID']
target = train['Email_Status']
train = train.drop(['Email_ID', 'Email_Status'], axis=1)
test = test.drop(['Email_ID'], axis=1)

alldata = pd.concat([train, test], axis=0)
alldata = alldata.drop(['Customer_Location'], axis=1) # H not present in train

train1 = alldata[0:len(train)]
test1 = alldata[len(train):]

print "Building XGBoost."
dtrain = xgb.DMatrix(train1, label=target, missing=np.nan)
dtest = xgb.DMatrix(test1, missing=np.nan)

params = {'booster':'gbtree', 'objective':'multi:softprob', 'max_depth':4, 'num_class': 3, 'seed': 0,
          'eta':0.1, 'nthread':4, 'subsample':0.9}

num_rounds = 350
clf_xgb = xgb.train(params, dtrain, num_rounds) # 4 fold cv-test-merror:0.18318425+0.000942040172976 
xgb_preds = clf_xgb.predict(dtest)
xgb_preds = pd.DataFrame(xgb_preds)
xgb_preds['Email_ID'] = test_id

print "Building Random Forest."
train2 = train1.fillna(-1)
test2 = test1.fillna(-1)

clf_rf = RandomForestClassifier(n_estimators=500, max_depth=10, n_jobs=-1, random_state=42) # 4-Fold CV: 0.8135270 (+/- 0.002781)
clf_rf.fit(train2, target)
rf_preds = clf_rf.predict_proba(test2)
rf_preds = pd.DataFrame(rf_preds)
rf_preds['Email_ID'] = test_id


print "Ensembling."
ens = xgb_preds.copy()
ens.iloc[:,0] = xgb_preds.iloc[:,0] * 0.85 + rf_preds.iloc[:,0] * 0.15
ens.iloc[:,1] = xgb_preds.iloc[:,1] * 0.85 + rf_preds.iloc[:,1] * 0.15
ens.iloc[:,2] = xgb_preds.iloc[:,2] * 0.85 + rf_preds.iloc[:,2] * 0.15

ens['Email_Status'] = np.argmax(np.array(ens.iloc[:,0:3]), axis=1)
ens[['Email_ID', 'Email_Status']].to_csv("AV_McKinseyHack_Solution.csv", index=False)