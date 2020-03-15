import sym_spell
import numpy as np
import collections

import timeit


def identity(trajectory_score, frequency, edit_distance, beta = 0.75):
    return 1

def confidence_only(trajectory_score, frequency, edit_distance, beta = 0.75):
    return trajectory_score

def hard_freq_dist(trajectory_score, frequency, edit_distance, beta = 0.75):

    ratio = np.log(frequency) * 1.0 / (edit_distance * 100.0 + 1)

    return ratio * trajectory_score

def soft_freq_dist(trajectory_score, frequency, edit_distance, beta = 0.75):

    ratio = np.power(np.log(frequency), beta / (edit_distance+1))

    return ratio * trajectory_score

def summerize_score(predictor, kernel_func, confidence_map):
    new_confidence_map = collections.defaultdict(float)
    for confidence, word in confidence_map:
        ac_word, ac_dist, ac_freq = predictor.auto_correct(word)
        alpha_score = kernel_func(confidence, ac_freq, ac_dist, beta = 1)
        new_confidence_map[ac_word] += alpha_score

        # print (ac_word, alpha_score)
    return max(new_confidence_map.values()), max(new_confidence_map, key=new_confidence_map.get)



def test_trajectory(predictor, kernel_func):
    import temp_trajectory

    predictions = temp_trajectory.trajs

    total = 0
    correct = 0

    for pred in predictions:
        ground = pred[0]
        confidence_map = pred[1]
        prob, final_word = summerize_score(predictor, kernel_func, confidence_map)
        if ground.lower() == final_word.lower():
            correct += 1
        total += 1


    accuracy = correct * 1.0 / total

    print("correct:", correct, "total:", total)
    return accuracy


def __load_dict_test():
    predictor = sym_spell.initialize()

predictor = sym_spell.initialize()
def __predict_words_test():
    test_trajectory(predictor, hard_freq_dist)
    # predictor.auto_correct("BSSZACH")

def performance_test():
    # print("Time for initializing autocorrect:" , timeit.timeit(__load_dict_test, number=1) / 1.0)
    
    print("Time for correcting one word:" , timeit.timeit(__predict_words_test, number=1) / 65.0)

if __name__ == "__main__":

    performance_test()
    quit()

    predictor = sym_spell.initialize()

    # print("Accuracy with identity:", test_trajectory(predictor, identity))
    # print("Accuracy with confidence only:", test_trajectory(predictor, confidence_only))
    print("Accuracy with hard freq edit_distance:", test_trajectory(predictor, hard_freq_dist))
    # print("Accuracy with soft freq edit_distance:", test_trajectory(predictor, soft_freq_dist))