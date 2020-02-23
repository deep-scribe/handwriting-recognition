#!/bin/bash
python rnn_bilstm.py random resampled 1
python rnn_bilstm.py random noresampled 1
python rnn_bilstm.py subject resampled 1
python rnn_bilstm.py subject noresampled 1
