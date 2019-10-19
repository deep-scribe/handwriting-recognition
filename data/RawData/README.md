## The Structure of the Raw Data

|      |      |      |      |      |      |      |      |      |      |      |      |      |      |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |
| ID    |      | yaw    | pitch   | raw    | ax     | ay      | az      | mx    | my    | mz   | ---  | ---  | ---  |
| ----- | :--: | ------ | ------- | ------ | ------ | ------- | ------- | ----- | ----- | ---- | ---- | ---- | ---- |
| **1** |  23  | 25.306 | -22.912 | -7.991 | 436.95 | -173.28 | 1092.65 | -5.17 | -10.1 | 1.24 | -243 | 386  | 93   |

 

* ID: the same ID index means that this belongs to the same handwritting motion.
* ax : acceleration on x-axis
* ay : acceleration on y-axis
* az : acceleration on z-axis
* Yaw: rotation along z-axis (pen stick as axis, vertical)
* Pitch: rotation along y-axis (left-right axis)
* Roll: rotation along x-axis (front-back axis)
* mx: magnetometer x-axis
* my: magnetometer y-axis
* mz: magnetometer z-axis

* I forgot the rest three parameters....
