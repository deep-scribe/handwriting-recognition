## The Structure of the Raw Data

|      |      |      |      |      |      |      |      |      |      |      |      |      |      |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |
| ID    |  TD  | yaw    | pitch   | raw    | ax     | ay      | az      | gx    | gy    | gz   | mx  | my  | mz  |
| **1** |  23  | 25.306 | -22.912 | -7.991 | 436.95 | -173.28 | 1092.65 | -5.17 | -10.1 | 1.24 | -243 | 386  | 93   |

\[index, time_diff, yaw, pitch, roll, 1000\*ax, 1000\*ay, 1000\*az, gx, gy, gz, mx, my, mz\]


* ID: the same ID index means that this belongs to the same handwritting motion.
* TD: Time Difference/Delta of this Frame
* ax : acceleration on x-axis
* ay : acceleration on y-axis
* az : acceleration on z-axis
* Yaw: rotation along z-axis (pen stick as axis, vertical)
* Pitch: rotation along y-axis (left-right axis)
* Roll: rotation along x-axis (front-back axis)
* mx: magnetometer x-axis
* my: magnetometer y-axis
* mz: magnetometer z-axis
* gx: gyroscope on x 
* gy: gyroscope on y 
* gz: gyroscope on z 


More about Yaw Pitch Roll:
* Define output variables from updated quaternion---these are Tait-Bryan angles, commonly used in aircraft orientation.
* In this coordinate system, the positive z-axis is down toward Earth.
* **Yaw** is the angle between Sensor x-axis and Earth magnetic North (or true North if corrected for local declination, looking down on the sensor positive yaw is counterclockwise.
* **Pitch** is angle between sensor x-axis and Earth ground plane, toward the Earth is positive, up toward the sky is negative.
* **Roll** is angle between sensor y-axis and Earth ground plane, y-axis up is positive roll.
* These arise from the definition of the homogeneous rotation matrix constructed from quaternions.
* Tait-Bryan angles as well as Euler angles are non-commutative; that is, the get the correct orientation the rotations must be applied in the correct order which for this configuration is yaw, pitch, and then roll.
* For more see http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles which has additional links.


关于Collect Motion Data时的注意事项:
1. 先让subject熟悉，如何先按按钮，再写对应的小写字母，写完停笔后再release button
2. keyboard control:
    1. “N”: save the motion data of the current letter to a file, and create a new file
    2. “D”: append a “#\n” string to the output buffer.
    3. “P”: purify the output buffer, meaning to recollect the motion data for this current letter.
    4. “R”: initiate a new series of file, and restart collecting from the letter ‘a’.

关于Data 的注意事项：
1. Index仅仅表示如果一行的index相同，那么他们属于同一个handwriting event，index并不是cumulative的，因为每个file的start index可以完全不一样，也有可能中间有一段index由于写错了被删除。
2. '#'表示本行的上面一行及其所属的整个handwriting event是invalid的。
