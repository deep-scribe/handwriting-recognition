import multiprocessing
import time
import math
import logging
import datetime
import csv
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# tune params for svm worker
TEST_RATE = 0.5
BINARY_SEARCH_COUNTER_HOLD = 10
GAMMA_COEFFICIENT = 0
CPU_COUNT = multiprocessing.cpu_count() - 1
RANDOM_STATE = 37
FOREST_RANDOM_STATE = 42
FIVE_FOLD = 5
TEN_FOLD = 10
FOLD_COUNTER = TEN_FOLD
ERROR_UPPER_BOUND = 2

N_MIN_RANGE = 100
N_MAX_RANGE = 2000
N_STEP = 100

# search range for hyper-parameter
TREE_ESTIMATOR_RANGE = np.arange(N_MIN_RANGE, N_MAX_RANGE, N_STEP)
ALGORITHM_LIST = ['SAMME', 'SAMME.R']
TREE_DEPTH_LIST = {1, 2, 3, 4}

# raw data file names
DATA_SET_A = 'run_letter_a_format.csv'
# DATA_SET_A = 'run_letter_a_format_mini.csv'
DATA_SET_B = 'run_letter_b_format.csv'
# DATA_SET_B = 'run_letter_b_format_mini.csv'
DATA_SET_C = 'run_letter_c_format.csv'
# DATA_SET_C = 'run_letter_c_format_mini.csv'

CURRENT_TIMESTAMP = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
RESULT_OUTPUT = 'run_boosting_result_' + CURRENT_TIMESTAMP + '.log'
LOGGER_FORMAT_HEADER = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
CUTOFF_LINE = '--------------------------------------------------------------------------------------------------'

# true label index
A_TRUE_LABEL = 0
B_TRUE_LABEL = 1
C_TRUE_LABEL = 2


def read_format_input(read_file_name):
    with open(read_file_name, 'rb') as f:
        reader = csv.reader(f)
        raw_data_list = list(reader)
    return raw_data_list


def cluster_selection_label(letter_a_set, letter_b_set, letter_c_set):
    return np.array(letter_a_set + letter_b_set)
    # return np.array(letter_b_set + letter_c_set)
    # return np.array(letter_a_set + letter_c_set)
    # return np.array(letter_a_set + letter_b_set + letter_c_set)


def cluster_selection_data(letter_a_set, letter_b_set, letter_c_set):
    return np.concatenate((letter_a_set, letter_b_set), axis=0)
    # return np.concatenate((letter_b_set, letter_c_set), axis=0)
    # return np.concatenate((letter_a_set, letter_c_set), axis=0)
    # return np.concatenate((letter_a_set, letter_b_set, letter_c_set), axis=0)


def boosting_worker(current_n_index, x_train_data, y_train_data, cross_fold_count, algorithm_index, tree_depth):
    print '{}{}'.format('Boosting worker with n index', current_n_index)
    boosting_worker_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=tree_depth), n_estimators=TREE_ESTIMATOR_RANGE[current_n_index],
                                             algorithm=ALGORITHM_LIST[algorithm_index])
    boosting_worker_cross_score = cross_val_score(boosting_worker_model, x_train_data, y_train_data, cv=cross_fold_count)
    return 1 - boosting_worker_cross_score.mean()


def boosting_worker_pool_builder(x_train_data, y_train_data, cross_fold_count, algorithm_index, tree_depth):
    boosting_pool_results = []
    boosting_pool = multiprocessing.Pool(processes=CPU_COUNT)
    boosting_pool_results_temp = []

    for n_search_index in range(len(TREE_ESTIMATOR_RANGE)):
        sample_result = boosting_pool.apply_async(boosting_worker, (n_search_index, x_train_data, y_train_data,
                                                        cross_fold_count, algorithm_index, tree_depth, ))
        boosting_pool_results_temp.append(sample_result)

    boosting_pool.close()
    boosting_pool.join()

    for sample_result in boosting_pool_results_temp:
        boosting_pool_results.append(sample_result)

    return boosting_pool_results


def transform_apply_result(input_list):
    result_list = []

    for i in range(len(input_list)):
        result_list.append(input_list[i].get())

    return result_list


def search_n_candidate_linear(error_list, range_list):
    error_list = transform_apply_result(error_list)
    error_list_index_list = np.argsort(error_list)

    high_c = range_list[error_list_index_list[1]]
    low_c = range_list[error_list_index_list[0]]
    return high_c, low_c


def search_n_candidate_binary(high_range, low_range, x_train_data, y_train_data, fold, algorithm_index, tree_depth):
    counter = 0
    print ("high_range: "+str(high_range));
    print ("low range: " + str(low_range));

    while True:
        counter += 1
        ada_model_high = AdaBoostClassifier(DecisionTreeClassifier(max_depth=tree_depth), n_estimators=high_range,
                                             algorithm=ALGORITHM_LIST[algorithm_index])
        cross_score_high = 1 - cross_val_score(ada_model_high, x_train_data, y_train_data, cv=fold).mean()

        boosting_model_low = AdaBoostClassifier(DecisionTreeClassifier(max_depth=tree_depth), n_estimators=low_range,
                                              algorithm=ALGORITHM_LIST[algorithm_index])
        cross_score_low = 1 - cross_val_score(boosting_model_low, x_train_data, y_train_data, cv=fold).mean()

        if counter > BINARY_SEARCH_COUNTER_HOLD:
            if cross_score_high < cross_score_low:
                return high_range, cross_score_high
            else:
                return low_range, cross_score_low
        else:
            if cross_score_high < cross_score_low:
                low_range = int(math.floor((high_range + low_range) / 2.0))
            else:
                high_range = int(math.floor((high_range + low_range) / 2.0))


def get_train_test_error(x_train, y_train, x_test, y_test, best_n, algorithm_index, tree_depth):
    final_boosting_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=tree_depth), n_estimators=best_n,
                                             algorithm=ALGORITHM_LIST[algorithm_index])
    final_boosting_model.fit(x_train, y_train)

    final_boosting_model_train_error = 1 - final_boosting_model.score(x_train, y_train)
    final_boosting_model_test_error = 1 - final_boosting_model.score(x_test, y_test)

    return final_boosting_model_test_error, final_boosting_model_train_error


def main():
    # config file and console logger
    logger = logging.getLogger('cogs118a_runtime')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(RESULT_OUTPUT)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOGGER_FORMAT_HEADER)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.debug('Reading raw input on letter a')
    raw_list_a = read_format_input(DATA_SET_A)
    format_list_a = np.array(raw_list_a).astype(None)
    y_data_set_a = [A_TRUE_LABEL] * format_list_a.shape[0]
    logging.debug('{}{}'.format('Read letter a with feature size', format_list_a.shape))

    raw_list_b = read_format_input(DATA_SET_B)
    format_list_b = np.array(raw_list_b).astype(None)
    y_data_set_b = [B_TRUE_LABEL] * format_list_b.shape[0]
    logging.debug('{}{}'.format('Read letter b with feature size', format_list_b.shape))

    raw_list_c = read_format_input(DATA_SET_C)
    format_list_c = np.array(raw_list_c).astype(None)
    y_data_set_c = [C_TRUE_LABEL] * format_list_c.shape[0]
    logging.debug('{}{}'.format('Read letter c with feature size', format_list_c.shape))

    x_data_set = cluster_selection_data(format_list_a, format_list_b, format_list_c)
    y_data_set = cluster_selection_label(y_data_set_a, y_data_set_b, y_data_set_c)
    logging.debug('{}{}'.format('X data set with feature size', x_data_set.shape))
    logging.debug('{}{}'.format('Y data set with feature size', y_data_set.shape))

    x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set,
                                                        test_size=TEST_RATE, random_state=RANDOM_STATE)

    logging.debug('{}{}'.format('Finished train test split, with test rate ', TEST_RATE))

    # fine tune linear kernel c
    logging.debug('---- Start fine tune Boosting ----')

    for index in range (len(ALGORITHM_LIST)):
        for tree_depth in TREE_DEPTH_LIST:
            boosting_pool_result = boosting_worker_pool_builder(x_train, y_train, FOLD_COUNTER, algorithm_index=index, tree_depth=tree_depth)
            boosting_pool_high_n, boosting_pool_low_n = search_n_candidate_linear(boosting_pool_result, TREE_ESTIMATOR_RANGE)
            boosting_pool_best_n, boosting_pool_valid_error = \
                search_n_candidate_binary(boosting_pool_high_n, boosting_pool_low_n, x_train, y_train, FOLD_COUNTER,
                                          algorithm_index=index, tree_depth=tree_depth)

            boosting_pool_test_error, boosting_pool_best_n_train_error = \
                get_train_test_error(x_train, y_train, x_test, y_test, boosting_pool_best_n,
                                     algorithm_index=index, tree_depth=tree_depth)

            boosting_pool_result_n_transformed = transform_apply_result(boosting_pool_result)
            logger.info('Running Tuning N ' + '\n' + 'The minimum Range is ' + str(
                N_MIN_RANGE) + '\n' + "The maximum Range is " + str(
                N_MAX_RANGE) + '\n' + "The Step size is " + str(
                N_STEP))
            logger.info('The error list is ' + str(boosting_pool_result_n_transformed))
            logger.info('The estimator list is ' + str(TREE_ESTIMATOR_RANGE.tolist()))

            current_boosting_info = 'AdaBoost - current algorithm: ' + ALGORITHM_LIST[index] + ' && current tree depth: '+ str(tree_depth)
            logger.info('{}{}{}'.format(current_boosting_info, ' - Best n: ', boosting_pool_best_n))
            logger.info('{}{}{}'.format(current_boosting_info, ' - Validation Error: ', boosting_pool_valid_error))
            logger.info('{}{}{}'.format(current_boosting_info, ' - Training Error: ', boosting_pool_best_n_train_error))
            logger.info('{}{}{}'.format(current_boosting_info, ' - Testing Error: ', boosting_pool_test_error)
                        + "\n" + CUTOFF_LINE)


if __name__ == "__main__":
    main()
