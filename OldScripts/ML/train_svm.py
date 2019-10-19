import multiprocessing
import time
import logging
import datetime
import csv
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# tune params for svm worker
TEST_RATE = 0.33
BINARY_SEARCH_HOLD = 0.000001
CPU_COUNT = multiprocessing.cpu_count() - 1
RANDOM_STATE = 37
ERROR_UPPER_BOUND = 2
FIVE_FOLD = 5
TEN_FOLD = 10
FOLD_COUNTER = TEN_FOLD

LINEAR_MIN_RANGE = 0.1
LINEAR_MAX_RANGE = 100
LINEAR_STEP = 0.2

C_RBF_MIN_RANGE = 0.1
C_RBF_MAX_RANGE = 100
C_RBF_STEP = 0.2

G_RBF_MIN_RANGE = 0.001
G_RBF_MAX_RANGE = 1
G_RBF_STEP = 0.001


# search range for hyper-parameter
C_SEARCH_RANGE_LINEAR = np.arange(LINEAR_MIN_RANGE, LINEAR_MAX_RANGE, LINEAR_STEP)
C_SEARCH_RANGE_RBF = np.arange(C_RBF_MIN_RANGE, C_RBF_MAX_RANGE, C_RBF_STEP)
GAMMA_SEARCH_RANGE_RBF = np.arange(G_RBF_MIN_RANGE, G_RBF_MAX_RANGE, G_RBF_STEP)

# raw data file names
# DATA_SET_A = 'run_letter_a_format.csv'
DATA_SET_A = 'run_letter_a_format_mini.csv'
# DATA_SET_B = 'run_letter_b_format.csv'
DATA_SET_B = 'run_letter_b_format_mini.csv'
# DATA_SET_C = 'run_letter_c_format.csv'
DATA_SET_C = 'run_letter_c_format_mini.csv'

CURRENT_TIMESTAMP = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H_%M_%S')
RESULT_OUTPUT = 'run_svm_result_' + CURRENT_TIMESTAMP + '.log'
LOGGER_FORMAT_HEADER = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
CUTOFF_LINE = '--------------------------------------------------------------------------------------------------'

# true label index
A_TRUE_LABEL = 0
B_TRUE_LABEL = 1
C_TRUE_LABEL = 2

BASE_URL = 'C:\\Users\\RUSSE\\PycharmProjects\\COGS118A-FINAL-SP17\\pythonScripts\\'


def read_format_input(read_file_name):
    with open(BASE_URL + read_file_name, 'rb') as f:
        reader = csv.reader(f)
        raw_data_list = list(reader)
    return raw_data_list


def cluster_selection_label(letter_a_set, letter_b_set, letter_c_set):
    return np.array(letter_a_set + letter_b_set)
    # return np.array(letter_a_set + letter_b_set + letter_c_set)


def cluster_selection_data(letter_a_set, letter_b_set, letter_c_set):
    # return np.concatenate((letter_a_set, letter_b_set, letter_c_set), axis=0)
    return np.concatenate((letter_a_set, letter_b_set), axis=0)


def svm_worker_linear(current_c_index, x_train_data, y_train_data, cross_fold_count):
    print '{}{}'.format('Running Linear SVM worker with c index', current_c_index)
    linear_svm_worker_svc = svm.SVC(kernel='linear', C=C_SEARCH_RANGE_LINEAR[current_c_index])
    linear_svm_worker_svc_valid_score = cross_val_score(linear_svm_worker_svc, x_train_data,
                                                        y_train_data, cv=cross_fold_count)
    return 1 - linear_svm_worker_svc_valid_score.mean()


def svm_worker_rbf(current_c_index, current_g_index, x_train_data, y_train_data,
                   cross_fold_count, is_searching_g, best_c_selection):
    if is_searching_g:
        print '{}{}'.format('Running RBF SVM worker (G) with g index', current_g_index)
        rbf_svm_worker_svc = svm.SVC(kernel='rbf', C=best_c_selection,
                                     gamma=GAMMA_SEARCH_RANGE_RBF[current_g_index])
    else:
        print '{}{}'.format('Running RBF SVM worker (C) with c index', current_c_index)
        rbf_svm_worker_svc = svm.SVC(kernel='rbf', C=C_SEARCH_RANGE_RBF[current_c_index])

    rbf_svm_worker_svc_valid_score = cross_val_score(rbf_svm_worker_svc, x_train_data,
                                                     y_train_data, cv=cross_fold_count)
    return 1 - rbf_svm_worker_svc_valid_score.mean()


def linear_svm_pool_builder(x_train_data, y_train_data, cross_fold_count):
    linear_svm_pool_results = []
    linear_svm_pool = multiprocessing.Pool(processes=CPU_COUNT)
    linear_svm_pool_results_temp = []

    for c_search_index in range(len(C_SEARCH_RANGE_LINEAR)):
        sample_result = linear_svm_pool.apply_async(svm_worker_linear,
                                                    (c_search_index, x_train_data,
                                                     y_train_data, cross_fold_count, ))
        linear_svm_pool_results_temp.append(sample_result)

    linear_svm_pool.close()
    linear_svm_pool.join()

    for sample_result in linear_svm_pool_results_temp:
        linear_svm_pool_results.append(sample_result)

    return linear_svm_pool_results


def rbf_svm_pool_builder(x_train_data, y_train_data, cross_fold_count, is_search_g, best_c_selection):
    rbf_svm_pool_results = []

    if is_search_g:
        rbf_svm_pool = multiprocessing.Pool(processes=CPU_COUNT)
        rbf_svm_pool_results_temp = []

        for g_search_index in range(len(GAMMA_SEARCH_RANGE_RBF)):
            sample_result = rbf_svm_pool.apply_async(svm_worker_rbf,
                                                        (-1, g_search_index, x_train_data,
                                                         y_train_data, cross_fold_count,
                                                         True, best_c_selection, ))
            rbf_svm_pool_results_temp.append(sample_result)

        rbf_svm_pool.close()
        rbf_svm_pool.join()
    else:
        rbf_svm_pool = multiprocessing.Pool(processes=CPU_COUNT)
        rbf_svm_pool_results_temp = []

        for c_search_index in range(len(C_SEARCH_RANGE_RBF)):
            sample_result = rbf_svm_pool.apply_async(svm_worker_rbf,
                                                        (c_search_index, -1, x_train_data,
                                                         y_train_data, cross_fold_count,
                                                         False, -1, ))
            rbf_svm_pool_results_temp.append(sample_result)

        rbf_svm_pool.close()
        rbf_svm_pool.join()

    for sample_result in rbf_svm_pool_results_temp:
        rbf_svm_pool_results.append(sample_result)

    return rbf_svm_pool_results


def search_c_candidate_linear(error_list, range_list):

    error_list = transform_apply_result(error_list)
    error_list_index_list = np.argsort(error_list)

    if len(error_list_index_list) > 2:
        high_c = range_list[error_list_index_list[1]]
        low_c = range_list[error_list_index_list[0]]
    else:
        return range_list[0], range_list[1]
    return high_c, low_c


def search_c_candidate_binary(high_range, low_range, x_train_data, y_train_data, fold, kernel_type,
                              is_search_g, best_c):
    while True:
        if is_search_g:
            svc_high = svm.SVC(kernel=kernel_type, C=best_c, gamma=high_range)
        else:
            svc_high = svm.SVC(kernel=kernel_type, C=high_range)

        cross_score_high = 1 - cross_val_score(svc_high, x_train_data, y_train_data, cv=fold).mean()

        if is_search_g:
            svc_low = svm.SVC(kernel=kernel_type, C=best_c, gamma=low_range)
        else:
            svc_low = svm.SVC(kernel=kernel_type, C=low_range)

        cross_score_low = 1 - cross_val_score(svc_low, x_train_data, y_train_data, cv=fold).mean()

        current_diff = abs(float(high_range - low_range))

        if current_diff < BINARY_SEARCH_HOLD:
            if cross_score_high < cross_score_low:
                return high_range, cross_score_high
            else:
                return low_range, cross_score_low
        else:
            if cross_score_high < cross_score_low:
                low_range = (high_range + low_range) / 2.0
            else:
                high_range = (high_range + low_range) / 2.0


def get_train_test_error(x_train, y_train, x_test, y_test, kernel_type, best_c, best_g, has_g):
    if has_g:
        final_svc = svm.SVC(kernel=kernel_type, C=best_c, gamma=best_g).fit(x_train, y_train)

        final_svc_train_error = 1 - final_svc.score(x_train, y_train)
        final_svc_test_error = 1 - final_svc.score(x_test, y_test)
        return final_svc_test_error, final_svc_train_error
    else:
        final_svc = svm.SVC(kernel=kernel_type, C=best_c).fit(x_train, y_train)

        final_svc_train_error = 1 - final_svc.score(x_train, y_train)
        final_svc_test_error = 1 - final_svc.score(x_test, y_test)
        return final_svc_test_error, final_svc_train_error


def transform_apply_result(input_list):
    result_list = []

    for i in range (len(input_list)):
        result_list.append(input_list[i].get())

    return result_list


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

    logger.info("Started Processing SVM")
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
    logging.info('{}{}'.format('X data set with feature size', x_data_set.shape))
    logging.info('{}{}'.format('Y data set with feature size', y_data_set.shape))

    x_train, x_test, y_train, y_test = train_test_split(x_data_set, y_data_set,
                                                        test_size=TEST_RATE, random_state=RANDOM_STATE)

    logging.info('{}{}'.format('Train test split, with test rate ', TEST_RATE))

    # fine tune linear kernel c
    logging.debug('---- Start fine tune linear SVM ----')
    linear_svm_pool_result = linear_svm_pool_builder(x_train, y_train, FOLD_COUNTER)

    linear_svm_pool_high_c, linear_svm_pool_low_c = search_c_candidate_linear(linear_svm_pool_result,
                                                                              C_SEARCH_RANGE_LINEAR)
    linear_svm_pool_best_c, linear_svm_pool_best_c_valid_error = \
        search_c_candidate_binary(linear_svm_pool_high_c, linear_svm_pool_low_c,
                                  x_train, y_train, FOLD_COUNTER, kernel_type='linear',
                                  is_search_g=False, best_c=-1)

    linear_svm_pool_test_error, linear_svm_pool_best_c_train_error =\
        get_train_test_error(x_train, y_train, x_test, y_test, kernel_type='linear',
                             best_c=linear_svm_pool_best_c, best_g=-1, has_g=False)

    linear_svm_pool_result_transformed = transform_apply_result(linear_svm_pool_result)

    logger.info('Running Linear SVM' +'\n' +'The minimum Range is '+ str(LINEAR_MIN_RANGE) + '\n'
                + "The maximum Range is " + str(LINEAR_MAX_RANGE) + '\n' + "The Step size is " + str(LINEAR_STEP))
    logger.info('The error list is ' + str(linear_svm_pool_result_transformed))
    logger.info('The c list is ' + str(C_SEARCH_RANGE_LINEAR.tolist()))

    logger.info('{}{}'.format('linear SVM - Best C: ', linear_svm_pool_best_c))
    logger.info('{}{}'.format('linear SVM - Validation Error: ', linear_svm_pool_best_c_valid_error))
    logger.info('{}{}'.format('linear SVM - Training Error: ', linear_svm_pool_best_c_train_error))
    logger.info('{}{}'.format('linear SVM - Testing Error: ', linear_svm_pool_test_error)+ "\n" + CUTOFF_LINE)

    # fine tune rbf kernel c
    logging.debug('---- Start fine tune rbf SVM finding c ----')
    rbf_svm_pool_result_c = rbf_svm_pool_builder(x_train, y_train, FOLD_COUNTER,
                                                 is_search_g=False, best_c_selection=-1)
    rbf_svm_pool_high_c, rbf_svm_pool_low_c = search_c_candidate_linear(rbf_svm_pool_result_c,
                                                                        C_SEARCH_RANGE_LINEAR)
    rbf_svm_pool_best_c, rbf_svm_pool_best_c_valid_error = \
        search_c_candidate_binary(rbf_svm_pool_high_c, rbf_svm_pool_low_c, x_train, y_train, FOLD_COUNTER,
                                  kernel_type='rbf', is_search_g=False, best_c=-1)

    rbf_svm_pool_test_error_c, rbf_svm_pool_best_c_train_error =\
        get_train_test_error(x_train, y_train, x_test, y_test, kernel_type='rbf', best_c=linear_svm_pool_best_c,
                             best_g=-1, has_g=False)

    rbf_svm_pool_result_c_transformed = transform_apply_result(rbf_svm_pool_result_c)
    logger.info('Running Linear RBF Tuning C ' + '\n' + 'The minimum Range is ' + str(
        C_RBF_MIN_RANGE) + '\n' + "The maximum Range is " + str(C_RBF_MAX_RANGE) + '\n' + "The Step size is " + str(
        C_RBF_STEP))
    logger.info('The error list is ' + str(rbf_svm_pool_result_c_transformed))
    logger.info('The c list is ' + str(C_SEARCH_RANGE_LINEAR.tolist()))

    logger.info('{}{}'.format('RBF SVM (C) - Best C: ', rbf_svm_pool_best_c))
    logger.info('{}{}'.format('RBF SVM (C) - Validation Error: ', rbf_svm_pool_best_c_valid_error))
    logger.info('{}{}'.format('RBF SVM (C) - Training Error: ', rbf_svm_pool_best_c_train_error))
    logger.info('{}{}'.format('RBF SVM (C) - Testing Error: ', rbf_svm_pool_test_error_c) + "\n" + CUTOFF_LINE)

    # fine tune rbf kernel g
    logging.debug('---- Start fine tune rbf SVM finding g ----')
    rbf_svm_pool_result_g = rbf_svm_pool_builder(x_train, y_train, FOLD_COUNTER,
                                                 is_search_g=True, best_c_selection=rbf_svm_pool_best_c)
    rbf_svm_pool_high_g, rbf_svm_pool_low_g = search_c_candidate_linear(rbf_svm_pool_result_g,
                                                                        GAMMA_SEARCH_RANGE_RBF)
    rbf_svm_pool_best_g, rbf_svm_pool_best_g_valid_error = \
        search_c_candidate_binary(rbf_svm_pool_high_g, rbf_svm_pool_low_g, x_train, y_train, FOLD_COUNTER,
                                  kernel_type='rbf', is_search_g=True, best_c=rbf_svm_pool_best_c)

    rbf_svm_pool_test_error_g, rbf_svm_pool_best_g_train_error = \
        get_train_test_error(x_train, y_train, x_test, y_test, kernel_type='rbf', best_c=rbf_svm_pool_best_c,
                             best_g=rbf_svm_pool_best_g, has_g=True)

    rbf_svm_pool_result_g_transformed = transform_apply_result(rbf_svm_pool_result_g)
    logger.info('Running Linear RBF Tuning C ' + '\n' + 'The minimum Range is ' + str(
        G_RBF_MIN_RANGE) + '\n' + "The maximum Range is " + str(G_RBF_MAX_RANGE) + '\n' + "The Step size is " + str(
        G_RBF_STEP))
    logger.info('The error list is ' + str(rbf_svm_pool_result_g_transformed))
    logger.info('The g list is ' + str(GAMMA_SEARCH_RANGE_RBF.tolist()))

    logger.info('{}{}'.format('RBF SVM (G) - Best G: ', rbf_svm_pool_best_g))
    logger.info('{}{}'.format('RBF SVM (G) - Validation Error: ', rbf_svm_pool_best_g_valid_error))
    logger.info('{}{}'.format('RBF SVM (G) - Training Error: ', rbf_svm_pool_best_g_train_error))
    logger.info('{}{}'.format('RBF SVM (G) - Testing Error: ', rbf_svm_pool_test_error_g) + "\n" + CUTOFF_LINE)


if __name__ == "__main__":
    main()
