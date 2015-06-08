import lasagne_classifier as lasagne

param = {}

## optimized and has little impact on result
param['max_epochs'] = 20

## optimized parameters
param['dense0_num_units'] = 300
param['dense1_num_units'] = 300
param['dropout_p'] = 0.5

## unoptimized parameters
param['update_learning_rate'] = 0.01
param['update_momentum'] = 0.9


update_learning_rate_list = [i*0.003+0.003 for i in range(0, 6)]
update_momentum_list = [i*0.2+0.2 for i in range(0, 5)]

output = open('Lasagne_CV_results.txt', 'w')
output.write('update_learning_rate		update_momentum		logloss\n')
for update_learning_rate_value in update_learning_rate_list:
	for update_momentum_value in update_momentum_list:
		param['update_momentum'] = update_momentum_value
		param['update_learning_rate'] = update_learning_rate_value

		logloss = lasagne.lasagne_classifier(param)
		output.write('{0}, {1}, {2}\n'.format(update_learning_rate_value, update_momentum_value, logloss))

#print max_epochs_list 
#result = lasagne.lasagne_classifier(param)
