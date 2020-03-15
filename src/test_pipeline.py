from pipeline import Pipeline, Autocorrect_kernel
import csv
import collections
import data_utils

test_files = [('kelly', '../data_words/kelly'),
            ('kelly_new', '../data_words/kelly_new'),
            ('kevin_quick_brown_fox', '../data_words/kevin_quick_brown_fox'),
            ('kevin_tip', '../data_words/kevin_tip'),
            ('russell_new', '../data_words/russell_new'),
            ('russell_new_2', '../data_words/russell_new_2')]

realtime_file = '../output/realtime_test'


def realtime_experiment():
    pipl = Pipeline()

    word_df = data_utils.load_subject(realtime_file)
    predicted_word = pipl.predict_realtime(word_df, G = 7, K = 10)

    print("Predicted word is", predicted_word)

def kernel_experiment():

    table = collections.defaultdict(dict)
    fieldnames = ['Test_files', 'identity', 'confidence_only', 'hard_freq_dist', 'soft_freq_dist']
    
    pipl = Pipeline()

    for word_filepath in test_files:
        for ac_kernel_name in Autocorrect_kernel.kernels:
            pipl.change_wordfile(word_filepath)
            pipl.change_ac_kernel(ac_kernel_name)
            accuracy = pipl.predict_testfiles()

            table[word_filepath[0]][ac_kernel_name] = accuracy
            table[word_filepath[0]]['Test_files'] = word_filepath[0]


    with open('../ac_experiment.csv', 'w', newline='') as f:

        thewriter = csv.DictWriter(f, fieldnames=fieldnames)

        thewriter.writeheader()

        for key in table:
            thewriter.writerow(table[key])

        f.close()

if __name__ == "__main__":
    
    # kernel_experiment()

    realtime_experiment()