
## define the parameter dict for xgboost classifier
def xgboost_para():

	## setup parameters for xgboost
	xgb_param = {}
	## use softmax multi-class classification
	xgb_param['objective'] = 'multi:softprob'
	xgb_param['silent'] = 1
	xgb_param['eval_metric'] = 'mlogloss'
	xgb_param['nthread'] = 3
	xgb_param['num_class'] = 9

	## parameters can be optimized ## 
	xgb_param['eta'] = 0.01
	xgb_param['max_depth'] = 20
	xgb_param['min_child_weight'] = 5
	xgb_param['min_child_width'] = 3  #default 1
	xgb_param['num_round'] = 1000
	#xgb_param['num_round'] = 1

	xgb_param['subsample'] = 0.8  #default 1.
	xgb_param['colsample_bytree'] = 1.  #default 1.
	xgb_param['gamma'] = 0  #default 0
	
	
	return xgb_param


def lasagne_singleLayer_para():

	lasagne_oneLayer_param = {}

	lasagne_oneLayer_param['max_epochs'] = 500
	#lasagne_oneLayer_param['max_epochs'] = 1
	lasagne_oneLayer_param['dense0_num_units'] = 600
	lasagne_oneLayer_param['dropout_p'] = 0.6
	lasagne_oneLayer_param['update_learning_rate'] = 0.01
	lasagne_oneLayer_param['update_momentum'] = 0.01
	
	return lasagne_oneLayer_param



def lasagne_doubleLayer_para():

	lasagne_twoLayer_param = {}

	lasagne_twoLayer_param['max_epochs'] = 100
	#lasagne_twoLayer_param['max_epochs'] = 1
	lasagne_twoLayer_param['dense0_num_units'] = 500
	lasagne_twoLayer_param['dense1_num_units'] = 500
	lasagne_twoLayer_param['dropout_p'] = 0.5
	lasagne_twoLayer_param['update_learning_rate'] = 0.01
	lasagne_twoLayer_param['update_momentum'] = 0.5
	
	return lasagne_twoLayer_param



def lasagne_comp_doubleLayer_para():

	lasagne_twoLayer_param = {}

	lasagne_twoLayer_param['max_epochs'] = 100
	#lasagne_twoLayer_param['max_epochs'] = 1
	lasagne_twoLayer_param['dense0_num_units'] = 500
	lasagne_twoLayer_param['dense1_num_units'] = 2000
	lasagne_twoLayer_param['dropout_p'] = 0.5
	lasagne_twoLayer_param['update_learning_rate'] = 0.01
	lasagne_twoLayer_param['update_momentum'] = 0.5
	
	return lasagne_twoLayer_param

