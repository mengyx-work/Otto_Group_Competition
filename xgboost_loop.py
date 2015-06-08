import xgboost_classifier as xgboost

# setup parameters for xgboost
param = {}
	
## use softmax multi-class classification
param['objective'] = 'multi:softprob'
## number of classes to classify
param['num_class'] = 9


## scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 15
param['silent'] = 1
param['eval_metric'] = 'mlogloss'
param['nthread'] = 2
param['min_child_weight'] = 3
param['min_child_width'] = 3

param['num_round'] = 20

logloss_score = xgboost.xgboos_classifer(param)
print logloss_score
