without specifying: 
	batch: 50
	pooling window shape = [2]
	pooling stride = [2]


2 layer architecture

		     (input outpit size after pooling)
                                   |
	conv1 (1->32) => pool1(300->150) => conv2 (32->64) => pool2 (150->75) => fc1 (activation function) => dropout => fc2 => loss&accuracy
                 |
    (input output feature/channel)  


3 layer architecture
	conv1 (1->32) => pool1(300->150) => conv2 (32->64) => pool2 (150->75) => conv3 (64->128) => pool3 (75->25) (shape[3]&stride[3]) => 
	fc1 (activation function) => dropout => fc2 => loss&accuracy


5 layer architecture
	conv1 (1->32) => pool1(300->150) => conv2 (32->64) => pool2 (150->75) => conv3 (64->64) => pool3 (75->75) (shape[6]&stride[1]) 
	=> conv4 (64->64) => pool4 (75->75) => conv5 (64->128) => pool5 (75->15) (shape[5]&stride[5]) => 
	fc1 (activation function) => dropout => fc2 => loss&accuracy


*** For the csv file, first row: iteration, second row: training loss, third row: test error
***** Strongly suggest pick the best result (test error) of the last three column (iteration 19700 19800 19900) as the final result of this training 

*** Some conclusion:
1. looks like the large the batch size, the more stable the curve of the training loss
2. generally the larger the batch size the better the result
3. relu looks like the best activation function for this set
4. 2 layers is good enough, 3 layer makes the result slightly better, 5 layers do not help at all.
5. best result: 3 layer max relu with batch 200
