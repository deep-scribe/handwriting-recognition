import multiprocessing
import time
import logging
import datetime
import csv
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# tune params for svm worker
TEST_RATE = 0.33
BINARY_SEARCH_HOLD = 0.000001
GAMMA_COEFFICIENT = 0
CPU_COUNT = multiprocessing.cpu_count() - 1
RANDOM_STATE = 37
FIVE_FOLD = 5
TEN_FOLD = 10
FOLD_COUNTER = TEN_FOLD
ERROR_UPPER_BOUND = 2

# search range for hyper-parameter
N_MIN_RANGE = 1
N_MAX_RANGE = 100
N_STEP = 1
N_NEIGHBORS_RANGE = np.arange(N_MIN_RANGE, N_MAX_RANGE, N_STEP)
KNN_ALGORITHM_LIST = ['ball_tree', 'kd_tree']

KNN_METRIC_LIST = ['minkowski', 'euclidean', 'l1', 'l2', 'manhattan']

# raw data file names
DATA_SET_A = 'run_letter_a_format.csv'
# DATA_SET_A = 'run_letter_a_format_mini.csv'
DATA_SET_B = 'run_letter_b_format.csv'
# DATA_SET_B = 'run_letter_b_format_mini.csv'
DATA_SET_C = 'run_letter_c_format.csv'
# DATA_SET_C = 'run_letter_c_format_mini.csv'

CURRENT_TIMESTAMP = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
RESULT_OUTPUT = 'run_knn_result_' + CURRENT_TIMESTAMP + '.log'
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
    # return np.array(letter_a_set + letter_b_set + letter_c_set)


def cluster_selection_data(letter_a_set, letter_b_set, letter_c_set):
    # return np.concatenate((letter_a_set, letter_b_set, letter_c_set), axis=0)
    return np.concatenate((letter_a_set, letter_b_set), axis=0)


def knn_worker(current_n_index, x_train_data, y_train_data, cross_fold_count, algorithm_type, metric_type):
    knn_worker_model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS_RANGE[current_n_index],
                                            algorithm=algorithm_type, metric=metric_type)
    knn_worker_cross_score = cross_val_score(knn_worker_model, x_train_data, y_train_data, cv=cross_fold_count)
    return 1 - knn_worker_cross_score.mean()


def knn_worker_pool_builder(x_train_data, y_train_data, cross_fold_count, algorithm_type, metric_type):
    knn_pool_results = []
    knn_pool = multiprocessing.Pool(processes=CPU_COUNT)
    knn_pool_results_temp = []

    for n_search_index in range(len(N_NEIGHBORS_RANGE)):
        sample_result = knn_pool.apply_async(knn_worker, (n_search_index, x_train_data, y_train_data,
                                                          cross_fold_count, algorithm_type, metric_type))
        knn_pool_results_temp.append(sample_result)

    knn_pool.close()
    knn_pool.join()

    for sample_result in knn_pool_results_temp:
        knn_pool_results.append(sample_result)

    return knn_pool_results


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


def search_n_candidate_binary(high_range, low_range, x_train_data, y_train_data, fold, algorithm_type, metric_type):

    while True:
        knn_model_high = KNeighborsClassifier(n_neighbors=high_range, algorithm=algorithm_type, metric=metric_type)
        cross_score_high = 1 - cross_val_score(knn_model_high, x_train_data, y_train_data, cv=fold).mean()

        knn_model_low = KNeighborsClassifier(n_neighbors=low_range, algorithm=algorithm_type, metric=metric_type)
        cross_score_low = 1 - cross_val_score(knn_model_low, x_train_data, y_train_data, cv=fold).mean()

        current_diff = float(cross_score_high - cross_score_low)

        if current_diff < BINARY_SEARCH_HOLD:
            if cross_score_high < cross_score_low:
                return high_range, cross_score_high
            else:
                return low_range, cross_score_low
        else:
            if cross_score_high < cross_score_low:
                low_range = int(math.floor((high_range + low_range) / 2.0))
            else:
                high_range = int(math.floor((high_range + low_range) / 2.0))


def get_train_test_error(x_train, y_train, x_test, y_test, best_n, algorithm_type, metric_type):
    final_knn_model = KNeighborsClassifier(n_neighbors=best_n, algorithm=algorithm_type, metric=metric_type)
    final_knn_model.fit(x_train, y_train)

    final_knn_model_train_error = 1 - final_knn_model.score(x_train, y_train)
    final_knn_model_test_error = 1 - final_knn_model.score(x_test, y_test)

    return final_knn_model_test_error, final_knn_model_train_error


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
    logging.debug('---- Start fine tune KNN ----')
    for knn_algorithm_item in KNN_ALGORITHM_LIST:
        for knn_metric_item in KNN_METRIC_LIST:
            knn_pool_result = knn_worker_pool_builder(x_train, y_train, FOLD_COUNTER,
                                                      knn_algorithm_item, knn_metric_item)
            knn_pool_high_n, knn_pool_low_n = search_n_candidate_linear(knn_pool_result, N_NEIGHBORS_RANGE)
            knn_pool_best_n, knn_pool_valid_error = \
                search_n_candidate_binary(knn_pool_high_n, knn_pool_low_n, x_train, y_train, FOLD_COUNTER,
                                          algorithm_type=knn_algorithm_item, metric_type=knn_metric_item)

            knn_pool_test_error, knn_pool_best_n_train_error = \
                get_train_test_error(x_train, y_train, x_test, y_test, knn_pool_best_n,
                                     algorithm_type=knn_algorithm_item, metric_type=knn_metric_item)

            knn_pool_result_n_transformed = transform_apply_result(knn_pool_result)
            logger.info('Running KNN Tuning N ' + '\n' + 'The minimum Range is ' + str(
                N_MIN_RANGE) + '\n' + "The maximum Range is " + str(
                N_MAX_RANGE) + '\n' + "The Step size is " + str(
                N_STEP))
            logger.info('The error list is ' + str(knn_pool_result_n_transformed))
            logger.info('The c list is ' + str(N_NEIGHBORS_RANGE.tolist()))

            current_knn_info = 'KNN - algorithm: ' + knn_algorithm_item + ', metric: ' + knn_metric_item
            logger.info('{}{}{}'.format(current_knn_info, ' - Best n: ', knn_pool_best_n))
            logger.info('{}{}{}'.format(current_knn_info, ' - Validation Error: ', knn_pool_valid_error))
            logger.info('{}{}{}'.format(current_knn_info, ' - Training Error: ', knn_pool_best_n_train_error))
            logger.info('{}{}{}'.format(current_knn_info, ' - Testing Error: ', knn_pool_test_error)
                        + "\n" + CUTOFF_LINE)


if __name__ == "__main__":
    main()
