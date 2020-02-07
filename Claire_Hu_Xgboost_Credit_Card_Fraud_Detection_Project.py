# The RAW Python logics for credit card fraud detection dataset.
# For xgboost modeling
# Claire Hu
#
#Version 2.0 ==> Oct. 29th
#
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score
from catboost import CatBoostRegressor
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import datetime, time
import os
import random
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

random.seed(888)


class XGBOOST_TRAINER(object):

    def __init__(self):
    
        self.layout = self._gen_generator(feature_name_list)

        #get number of records (row) in dataset
        self.num_records = self._record_counter()
        
    def _gen_generator(self, layout_list):
        layout_dict = {}
        for index in range(len(layout_list)):
            feature_name = layout_list[index]
            layout_dict[feature_name] = index
        return layout_dict
        
    def _record_counter(self):
        counter = 0
        with open(data_file, 'r') as f:
            for line in f:
                if "Time" not in line:
                    counter += 1

        return counter
        
        
    def _gen_whole_data_time_series(self):
        feature_train = []
        target_train = []

        feature_test = []
        target_test = []

        start = time.time()

        index = 0
        with open(data_file, 'r') as f:
            for line in f:
                line_list = line.strip('\n').split(',')

                if "Time" not in line:
                    target = int(line_list[self.layout["Class"]].strip('"'))
                    feature_list_str = line_list[0:-1]

                    if index % verbose == 0:
                        print('Have processed number of transactions:', index, 'time cost:', time.time() - start)
                        start = time.time()

                    feature_list_float = []
                    for item in feature_list_str:
                        feature_list_float.append(float(item))

                    if index <= int(float(self.num_records) * float(CUT_RATIO)):
                        feature_train.append(feature_list_float)
                        target_train.append(target)

                    else:
                        feature_test.append(feature_list_float)
                        target_test.append(target)

                    index += 1

        return feature_train, target_train, feature_test, target_test
        
        
    def _gen_whole_data_shuffle_fast(self):

        selected_testing_data_index_list = random.sample(range(0, self.num_records), int(float(self.num_records) * float(CUT_RATIO)))

        #下面三行的作用就是建立一个hashset，使得每次检索random index的时间复杂度是O(1), 会极大的加快程序运行速度。
        selected_index_set = set()
        for item in selected_testing_data_index_list:
            selected_index_set.add(item)

        feature_train = []
        target_train = []

        feature_test = []
        target_test = []

        start = time.time()

        index = 0
        with open(data_file, 'r') as f:
            for line in f:
                line_list = line.strip('\n').split(',')

                if "Time" not in line:
                    target = int(line_list[self.layout["Class"]].strip('"'))
                    feature_list_str = line_list[0:-1]

                    if index % verbose == 0:
                        print('Have processed number of transactions:', index, 'time cost:', time.time() - start)
                        start = time.time()

                    feature_list_float = []
                    for item in feature_list_str:
                        feature_list_float.append(float(item))

                    if index in selected_index_set:
                        feature_train.append(feature_list_float)
                        target_train.append(target)

                    else:
                        feature_test.append(feature_list_float)
                        target_test.append(target)

                    index += 1

        return feature_train, target_train, feature_test, target_test


    #这个也是data sample, but without hashset, the speed is very slow, time complexity is O(n)
    def _gen_whole_data_shuffle(self):

        selected_testing_data_index_list = random.sample(range(0, self.num_records), int(float(self.num_records) * float(CUT_RATIO)))

        feature_train = []
        target_train = []

        feature_test = []
        target_test = []

        start = time.time()

        index = 0
        with open(data_file, 'r') as f:
            for line in f:
                line_list = line.strip('\n').split(',')

                if "Time" not in line:
                    target = int(line_list[self.layout["Class"]].strip('"'))
                    feature_list_str = line_list[0:-1]

                    if index % verbose == 0:
                        print('Have processed number of transactions:', index, 'time cost:', time.time() - start)
                        start = time.time()

                    feature_list_float = []
                    for item in feature_list_str:
                        feature_list_float.append(float(item))

                    if index in selected_testing_data_index_list:
                        feature_train.append(feature_list_float)
                        target_train.append(target)

                    else:
                        feature_test.append(feature_list_float)
                        target_test.append(target)

                    index += 1

        return feature_train, target_train, feature_test, target_test

    #这个就是over sample 的function，里面用到了python的一个package，是imblearn，需要提前安装一下
    def _over_sample_generator_time_series(self):
        ros = RandomOverSampler(random_state = random_state)
        feature_train, target_train, feature_test, target_test = self._gen_whole_data_time_series()
        feature_train_resampled, target_train_resampled = ros.fit_resample(feature_train, target_train)
        return feature_train_resampled, target_train_resampled, feature_test, target_test

    #这个就是under sample 的function，里面用到了python的一个package，是imblearn，需要提前安装一下
    def _under_sample_generator_time_series(self):
        rus = RandomUnderSampler(random_state = random_state)
        feature_train, target_train, feature_test, target_test = self._gen_whole_data_time_series()
        feature_train_resampled, target_train_resampled = rus.fit_resample(feature_train, target_train)
        return feature_train_resampled, target_train_resampled, feature_test, target_test

    # 这个就是SMOTE sample 的function，里面用到了python的一个package，是imblearn，需要提前安装一下
    def _SMOTE_sample_generator_time_series(self):
        sm = SMOTE(random_state=42)
        feature_train, target_train, feature_test, target_test = self._gen_whole_data_time_series()
        feature_train_resampled, target_train_resampled = sm.fit_resample(feature_train, target_train)
        return feature_train_resampled, target_train_resampled, feature_test, target_test
        
    def _train_model_xgboost(self, feature_train, target_train):

        dtrain = xgb.DMatrix(feature_train, label=target_train, feature_names=training_layout)
        bst = xgb.train(param, dtrain, num_round)
        bst.dump_model(model_file)
        return bst

    def _inference_xgboost(self, bst, feature_test, target_test):
        dtest = xgb.DMatrix(feature_test, label=target_test, feature_names=training_layout)
        
        preds = bst.predict(dtest)
        return preds


    def _print_inference_detail(self, target_test, prediction_result, time_test):
        for item_a, item_b, time in zip(target_test, prediction_result, time_test):
            print(time, item_a, item_b)
            
    def _print_feature_importance(self, importance_dict):
        sorted_importance_dict = sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)
        print('Credit Card Fraud Detection Feature Importance List:')
        for item in sorted_importance_dict:
            print('Feature Name:', item[0], '\t\t', 'Feature Abs_importance:', item[1])
            
            
    def _gen_ratio(self, target_train):
        gd_trxn_counter = 0
        fd_trxn_counter = 0

        for item in target_train:
            if item == 1.0:
                fd_trxn_counter += 1
            else:
                gd_trxn_counter += 1

        ratio = float(gd_trxn_counter) / float(fd_trxn_counter)
        print('Number of good transactions in training dataset:', gd_trxn_counter)
        print('Number of fraud transactions in training dataset::', fd_trxn_counter)
        print('GOOD/FRAUD Ratio:', ratio, '\n')
        return ratio
        
        
    def _gen_AUC_CURVE(self, fpr, tpr, roc_auc):
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        #plt.show()
        plt.savefig(AUC_ROC_CURVE_SAVING_PATH + 'xgboost_AUC_ROC_' + model_version + '.jpg')
        
        
    def process(self):

        data_processing_start = time.time()
        if sample_selection_mode == 'time_series':
            print('Is using time_series sampling mode...')
            feature_train, target_train, feature_test, target_test = self._gen_whole_data_time_series()

        elif sample_selection_mode == 'shuffle_slow':
            print('Is using shuffle sampling mode (Normal Speed)...')
            feature_train, target_train, feature_test, target_test = self._gen_whole_data_shuffle()

        elif sample_selection_mode == 'shuffle_fast':
            print('Is using shuffle sampling mode (Fast data processing speed)...')
            feature_train, target_train, feature_test, target_test = self._gen_whole_data_shuffle_fast()

        elif sample_selection_mode == 'time_series_over':
            print('Is using time_series_over mode...')
            feature_train, target_train, feature_test, target_test = self._over_sample_generator_time_series()

        elif sample_selection_mode == 'time_series_under':
            print('Is using time_series_under mode...')
            feature_train, target_train, feature_test, target_test = self._under_sample_generator_time_series()

        else:
            print('Is using time_series_smote mode...')
            feature_train, target_train, feature_test, target_test = self._SMOTE_sample_generator_time_series()


        print('Total time to process all data is:', time.time() - data_processing_start)

        print('Length of training dataset:', len(feature_train), len(target_train))
        print('Length of validation dataset:', len(feature_test), len(target_test))

        ratio = self._gen_ratio(target_train)
        param['scale_pos_weight'] = ratio

        print('Using following parameters:')
        for key in param:
            print(key, param[key])
        print('\n')

        print("Is training the xgboost model...")
        start = time.time()
        model_bst = self._train_model_xgboost(feature_train, target_train)
        print("Training process finished! Time cost:", time.time() - start)

        print('Is inference the predicted results by using the validation data...')
        prediction_result = self._inference_xgboost(model_bst, feature_test, target_test)

        print("Generating the feature importance list...")
        importance_dict = model_bst.get_score(importance_type='gain')
        self._print_feature_importance(importance_dict)

        fpr, tpr, threshold = metrics.roc_curve(np.array(target_test), np.array(prediction_result))
        roc_auc = metrics.auc(fpr, tpr)

        self._gen_AUC_CURVE(fpr, tpr, roc_auc)
        print('The xgboost model performance of AUC-ROC value:', roc_auc)


if __name__ == "__main__":

    # Available parameters:
    # time_series
    # shuffle_slow
    # shuffle_fast
    # time_series_over
    # time_series_under
    # time_series_smote

    sample_selection_mode = 'time_series_under'

    verbose = 1000

    random_state = 888

    DT_format = "%Y%m%d%H%M%S"

    work_dir = '/Users/yj/Documents/claire_project/credit_fraud_detection/'

    currentDT = datetime.datetime.now()
    model_version = currentDT.strftime(DT_format)

    data_file = work_dir + 'creditcard.csv'
    print('Is loading the training file from:', data_file)

    text_model_store_location = work_dir + 'models/'
    if not os.path.exists(text_model_store_location):
        os.makedirs(text_model_store_location)
    print('Xgboost model will store at:', text_model_store_location)
    model_file = text_model_store_location + 'creditcard_xgboost_model_' + model_version + '.md'

    AUC_ROC_CURVE_SAVING_PATH = work_dir + 'PERFORMANCE_CHARTS/'

    if not os.path.exists(AUC_ROC_CURVE_SAVING_PATH):
        os.makedirs(AUC_ROC_CURVE_SAVING_PATH)

    CUT_RATIO = 0.7



    param = {
        'max_depth': 3000,  # the maximum depth of each tree
        'eta': 0.1,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        'objective': 'binary:logistic',  # error evaluation for multiclass training # binary:logistic #multi:softprob
        'scale_pos_weight': 1.0,
        'feature_selector': 'greedy',
        #'tree_method': 'hist',
        #'grow_policy': 'lossguide', # depthwise #Controls a way new nodes are added to the tree.
    }  # the number of classes that exist in this datset #'num_class': 2

    num_round = 20  # the number of training iterations

    feature_name_list = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]

    training_layout = feature_name_list[0:-1]

    print(training_layout)
    
    print('Creating the XGBOOST Training Object...')
    XGBOOST_TRAINER = XGBOOST_TRAINER()
    print('XGBOOST Training Object Create Successfully!')
    XGBOOST_TRAINER.process()