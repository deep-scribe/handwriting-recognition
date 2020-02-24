from beam import beam_search_top_k_trajectory
from rnn_bilstm import get_net, get_logit, Net
from data_utils import load_subject, get_calibrated_yprs_samples

MODEL_WEIGHT_PATH = '../saved_model/rnn_bilstm_random_resampled_0.pth'

if __name__ == "__main__":
    df = load_subject(
        subject_path='../data_words/words_mini_easy'
    )
    print(df)
    xs, ys = get_calibrated_yprs_samples(
        df=df,
        resampled=True,
        flatten=False,
        is_word_samples=True
    )
    print(xs)
    print(ys)

    print(len(xs))
    print(len(ys))
