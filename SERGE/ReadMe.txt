SERGE.py implements the code for the algorithm described in [1], adapted for venue recommendation task in Meetup

####################################################################################################################
1. Instructions before running the code:

Please install the "numpy" package if it is not there already. In general, all other imported packages should be present in the system by default. Please use Python version 2.7 for running the code. 


####################################################################################################################
2. Parameters:

The code has 5 parameters.

a) HISTORY -- The history length (number of past events) to be used for each event
b) MAXIT -- The maximum number of iterations for the random walk
c) EPS -- Epsilon used as stopping criterion for random walk
d) CATEGORY -- The category of Meetup groups for which recommendation is needed; Use -1 for any category, 0 for "Activity" category, 1 for "Hobby" category, 2 for "Social" category, 3 for "Entertainment" category and 4 for "Tech." category
e) NEW_FLAG -- Flag indicating whether we wish to evaluate only for new events; Set this to 1 if the evaluation has to be done only for new events


####################################################################################################################
3. How to run?

a) Run with default parameter values
>>python SERGE.py

This assumes parameter values HISTORY=100, MAXIT=1000, EPS=0.0001, CATEGORY=-1, NEW_FLAG=0 

b) Run with specific parameter values provided as command-line arguments
>>python SERGE.py <HISTORY> <MAXIT> <EPS> <CATEGORY> <NEW_FLAG>

Example:
>>python SERGE.py 100 1000 0.00001 2 0
This assumes parameter values HISTORY=100, MAXIT=1000, EPS=0.00001, CATEGORY=2, NEW_FLAG=0 


####################################################################################################################
4. Output:

The code will show the rank of the target venue among all the events for each target event. Finally it summarizes the recall and MIR metrics obtained for the algorithm. Additionally, it provides average the execution time (in seconds) taken by the code for generating recommendation for each event.


####################################################################################################################

5. References:
[1] S. Liu, B. Wang, and M. Xu, “Serge: Successive event recommendation based on graph entropy for event-based social networks”, IEEE Access, vol. 6, pp. 3020–3030, 2018.
