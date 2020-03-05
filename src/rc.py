
"""

The code to generate the risk-coverage curve given the model and the test data.
It is assumed that the model is implemented using pytorch

Usage:

$ python
>>> import rc
>>> rc.draw_curve(model, input_data)
# png file saved to the current directory

"""


import numpy as np
import matplotlib.pyplot as plt


def draw_curve(confidence, prediction, label, output_dir='.'):
    '''
    generate the risk coverage curve and save png file to output_dir

    @param: confidence (Tensor): confidence.shape = (n,)
    @param: prediction (Tensor): prediction.shape = (n,)
    @param: label (Tensor):      label.shape = (n,)

    @return: None
    '''

    confidence, prediction, label = confidence.numpy(), prediction.numpy(), label.numpy()
    correctness = (prediction - label) == 0

    zipped = zip(confidence, prediction, label, correctness)

    sorted_res = sorted(zipped, key=lambda x:x[0], reverse=True)

    risk = []
    coverage = []
    mistake_count = 0
    curr_count = 0
    total_count = len(label)

    for conf, pred, lab, correct in sorted_res[:1000]:
        
        curr_count += 1

        if correct == False:
            mistake_count += 1

        risk.append(mistake_count * 1.0 / curr_count)
        coverage.append(curr_count * 1.0 / total_count)

    fig, ax = plt.subplots()

    line1, = ax.plot(coverage, risk, label = "risk-coverage curve")

    ax.fill_between(coverage, risk, alpha = 0.5)
    ax.legend()
    plt.show()







def example():
    '''
    In terminal, run {python rc.py path_to_model path_to_testdata}
    '''
    # if len(sys.argv) != 2:
    #     print('Usage: python data_flatten.py <subject_path>')
    #     quit()

    # subject_path = sys.argv[1]
    # import rnn_bilstm
    import rnn_bilstm
    import torch
    import data_loader
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    import string

    trainx, devx, testx, trainy, devy, testy = data_loader.load_all_classic_random_split(resampled = True, flatten=False)
    print(testy.shape)

    BATCH_SIZE = 1050

    testloader = rnn_bilstm.get_dataloader(testx, testy, BATCH_SIZE)
    net = rnn_bilstm.get_net("../saved_model/rnn_bilstm/rnn_bilstm_random_resampled_0.pth")
    print(rnn_bilstm.acc(net, testloader))

    input, label = next(iter(testloader))
    logit = rnn_bilstm.get_prob(net, input)
    confidence, prediction = torch.max(logit.data, 1)

    draw_curve(confidence + 1, prediction, label)

    # print(logit)
    # _, predicted = torch.max(logit.data, 1)
    # print(torch.sum((predicted - label) == 0).item()/len(label))
    # print(predicted)
    # cm = confusion_matrix(label, predicted)
    # index = list(string.ascii_uppercase)
    # df = pd.DataFrame(data=cm, index=index, columns=index)
    # print(df)


if __name__ == "__main__":
    example()
