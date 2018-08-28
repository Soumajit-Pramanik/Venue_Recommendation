DEEPCONN.py implements the code for the algorithm described in [1], adapted for venue recommendation task in Meetup

####################################################################################################################
1. Instructions before running the code:

Please install the "CNTK" package. The detailed instructions for "CNTK" installation is provided in "https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine". Our code is compatible to CNTK version 2.2. Please install the "numpy" package if it is not there already. In general, all other imported packages should be present in the system by default. Please use Python version 2.7 for running the code. 


####################################################################################################################
2. Parameters:

The code has 6 parameters.

a) DROP = Droupout
b) BATCH_SIZE = Minibatch Size
c) LEARNING_RATE
d) TRAINING_ITERATION = Number of Epochs used for training
e) CATEGORY -- The category of Meetup groups for which recommendation is needed; Use -1 for any category, 0 for "Activity" category, 1 for "Hobby" category, 2 for "Social" category, 3 for "Entertainment" category and 4 for "Tech." category
f) NEW_FLAG -- Flag indicating whether we wish to evaluate only for new events; Set this to 1 if the evaluation has to be done only for new events

####################################################################################################################
3. How to run?

a) Run with default parameter values
>>python DEEPCONN.py

This assumes parameter values DROP=0.1, BATCH_SIZE=128, LEARNING_RATE=0.003, TRAINING_ITERATION=2000, CATEGORY=-1, NEW_FLAG=0

b) Run with specific parameter values provided as command-line arguments
>>python DEEPCONN.py <DROP> <BATCH_SIZE> <LEARNING_RATE> <TRAINING_ITERATION> <CATEGORY> <NEW_FLAG>

Example:
>>python DEEPCONN.py 0.1 128 0.005 1000 1 0
This assumes parameter values DROP=0.1, BATCH_SIZE=128, LEARNING_RATE=0.005, TRAINING_ITERATION=1000, CATEGORY=1, NEW_FLAG=0


####################################################################################################################
4. Output:

The code will show the rank of the target venue among all the venues for each target event. Finally, it summarizes the recall and MIR metrics obtained for the algorithm. Additionally, it provides the average execution time (in seconds) taken by the code for generating recommendation for each event.

####################################################################################################################

5. References:
[1] L. Zheng, V. Noroozi, and P. S. Yu, “Joint deep modeling of users and items using reviews for recommendation,” in Proceedings of
WSDM ’17. New York, NY, USA: ACM, 2017, pp. 425–434.
