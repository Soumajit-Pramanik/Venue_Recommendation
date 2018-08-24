import cntk as C
import numpy as np
from cntk.logging import ProgressPrinter
from cntk.layers import *
from cntk.ops import functions
import random
import math
import time
import pickle
from collections import defaultdict
import sys

#Sequence length
SEQ_LEN=100
#Length of the venue representation including distance
VVEC_LEN=104
#Length of the event representation
EVEC_LEN=100
#Length of the group representation
GVEC_LEN=500


#Hidden dimension of LSTMs used in the venue and event modules
H_DIMS = 50
#Hidden dimension of LSTM used in the group module
H_DIMS2 = 200
#Droupout
DROP=0.1
#Minibatch Size
BATCH_SIZE=128
#Learning Rate
LEARNING_RATE=0.005
#Number of Epochs used for training
TRAINING_ITERATION=20
#The category of Meetup groups for which recommendation is needed; Use -1 for any category, 0 for "Activity" category, 1 for "Hobby" category, 2 for "Social" category, 3 for "Entertainment" category and 4 for "Technical" category
CATEGORY=-1 
#Flag indicating whether we wish to evaluate only for new events; Set this to 1 if the evaluation has to be done only for new events 
NEW_FLAG=0


if len(sys.argv)==1:
	print "No arguments provided; Using default values"
	
else:
	if len(sys.argv)==9:
		H_DIMS =int(sys.argv[1])
		H_DIMS2 = int(sys.argv[2])
		DROP=float(sys.argv[3])
		BATCH_SIZE=int(sys.argv[4])
		LEARNING_RATE=float(sys.argv[5])
		TRAINING_ITERATION=int(sys.argv[6])
		CATEGORY=int(sys.argv[7])
		NEW_FLAG=int(sys.argv[8])
	else:
		print "Please provide all 8 arguments"
		exit(0)

print "Parameters - Hidden dimension for venue/event module, Hidden dimension for group module, Dropout, Minibatch Size, Learning Rate, Number of training Epochs, Group category, New Events Flag- ", H_DIMS, H_DIMS2, DROP, BATCH_SIZE, LEARNING_RATE, TRAINING_ITERATION, CATEGORY, NEW_FLAG


#Sequence model for Venue Module
def model1(x):
	with C.layers.default_options(initial_state = 0.1):
		m = C.layers.Recurrence(C.layers.LSTM(H_DIMS,enable_self_stabilization=default_override_or))(x)
		m = C.sequence.last(m)
		m = C.layers.Dropout(DROP)(m)
		return m

#Embedding model for Venue Module
def model2(x):
	with default_options(init = glorot_uniform()):
		h1 = Dense(H_DIMS, activation= C.relu)(x)
		h2 = Dropout(DROP)(h1)		
		return h1
		
#Sequence model for Event Module		
def model3(x):
	with C.layers.default_options(initial_state = 0.1):
		m = C.layers.Recurrence(C.layers.LSTM(H_DIMS,enable_self_stabilization=default_override_or))(x)
		m = C.sequence.last(m)
		m = C.layers.Dropout(DROP)(m)
		return m

#Embedding model for Event Module
def model4(x):
	with default_options(init = glorot_uniform()):
		h1 = Dense(H_DIMS, activation= C.relu)(x)
		h2 = Dropout(DROP)(h1)		
		return h1

#Sequence model for Group Module		
def model5(x):
	with C.layers.default_options(initial_state = 0.1):
		m = C.layers.Recurrence(C.layers.LSTM(H_DIMS2,enable_self_stabilization=default_override_or))(x)
		m = C.sequence.last(m)
		m = C.layers.Dropout(DROP)(m)
		m = C.layers.Dense(H_DIMS, activation= C.relu)(m)
		return m

#Embedding model for Group Module
def model6(x):
	with default_options(init = glorot_uniform()):
		h1 = Dense(H_DIMS2, activation= C.relu)(x)
		h2 = Dropout(DROP)(h1)		
		h3 = Dense(H_DIMS, activation= C.relu)(x)
		return h3

'''
#Final dense layer for combining scores from all three modules 
def final_model(x):
	with default_options(init = glorot_uniform()):
		h1 = Dense(1, activation= C.sigmoid)(x)
		#h2 = Dense(1, activation= C.sigmoid)(h1)
		#h2 = Dropout(0.2)(h1)		
		return h1	
'''

#Returns 0 if the target venue is never used earlier by the group hosting the target event
def check_new_ven(grp_event, event, grp,edata,etime):
	flag=0
	for e1 in grp_event[grp]:
		if e1 in etime and etime[e1]<etime[event]:	#Checks all the events hosted earlier	
			if e1 in edata and edata[e1]==edata[event]: 
				flag=1
				break
	return flag

#Returns the sequence of successful events hosted at a particular venue "ven" successfully before target event "eT" and the sequence of groups hosting them		
def get_venue_vec(vdata,ven,etime, eT,esucc, thres, edesc,gvec,egroup):
	elist=[]
	if ven not in vdata:
		return np.array([]),np.array([])
	
	#Set of events hosted at venue "ven" successfully before target event
	for ev in vdata[ven]:
		if ev in etime and ev in esucc and etime[ev]<etime[eT] and esucc[ev]>thres:
			elist.append((ev,etime[ev]))
	elist=sorted(elist,key=lambda x:x[1],reverse=True)
	elist1=[]
	glist1=[]
	
	#Return maximum "SEQ_LEN" number of	events and groups
	if len(elist)>=SEQ_LEN:
		len1=SEQ_LEN
	else:
		len1=len(elist)
	
	for j in range(0,len1):
		ev=[]
		gv=[]
		event=elist[len1-j-1][0]
		edess=[]
		if event in edesc:
			edess=edesc[event]
		
		#Append event vectors in "elist1" and group vectors in "glist1" 
		if len(edess)>0:		
			for edd1 in edess:
				ev.append(float(edd1))		
			elist1.append(np.array(ev))
			
			ggvec=gvec[egroup[event]]
			for gg1 in ggvec:
				gv.append(float(gg1))
			glist1.append(np.array(gv))	
			
	return np.array(elist1),np.array(glist1)	


def read_input(gvec, etime, egroup, edata, vdata, vset, e_succ, edesc, ven_grp, gcat):
	#Read the input events; In the file, each line consists of a sequence of maximum 101 events, the first 100 events are similar to the last one which is the target event. For each event the event-id, corresponding group-id, venue-id, time and succss label is provided.
	
	f=open('./data/selected_events1.txt','r')
	for line1 in f:
		line=(line1.strip()).split()
		idd=int((len(line)-4)/5) #Number of events in each sequence
		index=0
		ee=[]
		evv=[]
		for j in range(0,idd):
			ev=[]
			event=line[index]         #Event id
			group=int(line[index+1])  #Group id
			venue=int(line[index+2])  #Venue id
			time1=int(line[index+3])  #Time
			label=int(line[index+4])  #Success label 1/0
			index+=5
			
			
			if j<idd-1: 
				#Append representation of the "venue" in ev
				for vv1 in vset[venue]:
					ev.append(float(vv1))
					
				#Add distance of "venue" from host "group" to ev
				ev.append(ven_grp[venue][egroup[event]][0])
				ev.append(ven_grp[venue][egroup[event]][1])
				ev.append(ven_grp[venue][egroup[event]][2])
				ev.append(ven_grp[venue][egroup[event]][3])
				
				#Append entire venue representation in ee
				ee.append(np.array(ev))
			
		
			if j==idd-1: #For the last i.e. target event
				evv.append(float(label)) #Append target event's label to evv
				evx=[]
				vvx=[]
				ggx=[]
			
				#Read the representation of target event
				if event in edesc:
					for edd1 in edesc[event]:
						evx.append(float(edd1))
				
				#Read the representation of target venue		
				for vv1 in vset[venue]:
					vvx.append(float(vv1))
				
				#Add distance of "venue" from host "group" to vvx
				vvx.append(ven_grp[venue][egroup[event]][0])
				vvx.append(ven_grp[venue][egroup[event]][1])
				vvx.append(ven_grp[venue][egroup[event]][2])
				vvx.append(ven_grp[venue][egroup[event]][3])	
			
				#Read the representation of target event's host group
				for gg1 in gvec[egroup[event]]:
					ggx.append(float(gg1))	
		
			
		ee=np.array(ee)
		
		#Read the sequence of successful events hosted at a particular venue "ven" successfully before target event "eT" and the sequence of groups hosting them
		exx,gvx=get_venue_vec(vdata, int(line[index+2]),etime, event,e_succ, thres, edesc, gvec, egroup)
		
		
		#Choose events from only the specified category
		if CATEGORY>-1 and gcat[group]!=CATEGORY:
			continue
		
		#Continue if there are no past successful event similar to target event or no successful event hosted by the target venue in past
		if len(ee)==0 or len(exx)==0: 
			continue
		
		
		X1.append(ee)
		Y1.append(np.array(evv))
		E1.append(exx)
		ED.append(np.array(evx))
		VD.append(np.array(vvx))
		V1.append(int(line[index+2]))
		S1.append(idd)
		EID.append(event)
		G1.append(gvx)
		GD.append(np.array(ggx))
		    
	
	all_list=[]
	all_list0=[]
	all_list1=[]
	
	for i in range(0,len(X1)):
		'''
		#all_list.append((X1[i],Y1[i],V1[i],S1[i],E1[i],VD[i],ED[i],EID[i]))	
		if edata[EID[i]]!=V1[i]:
			print "Goray Gondogol 1"
		'''	
		#insert unsuccessful events to all_list0
		if Y1[i][0]==0:
			all_list0.append((X1[i],Y1[i],V1[i],S1[i],E1[i],VD[i],ED[i],EID[i],G1[i],GD[i]))
		else:
			#insert successful events to all_list
			if Y1[i][0]==1:
				all_list1.append((X1[i],Y1[i],V1[i],S1[i],E1[i],VD[i],ED[i],EID[i],G1[i],GD[i]))	

	print "Reading input files done"
	print "Number of Unsuccessful and Successful events in the dataset - ",len(all_list0),len(all_list1)

	#Find minimum of both
	mm=min(len(all_list0),len(all_list1))

	#Consider same number of Unsuccessful and Successful events; Choose random sample of mm events from all_list0 if it has higher length than all_list1 or vice versa 
	if len(all_list0) <len(all_list1):
		all_list1=random.sample(all_list1,mm)
	else:	
		all_list0=random.sample(all_list0,mm)

	all_list=all_list0+all_list1
	print "Overall number of events for training and testing- ",len(all_list)
	
	#Randomly shuffle the events 
	random.shuffle(all_list)
	return all_list


#Returns "length" number of instances starting from index "start"	
def get_sample(start,length):
	p1=[]
	p2=[]
	p3=[]
	p4=[]
	p5=[]
	p6=[]
	p7=[]
	p8=[]
	p9=[]
	p10=[]
	
	for i in range(start,start+length):
		p1.append(X1[i])
		p2.append(Y1[i])
		p3.append(V1[i])
		p4.append(S1[i])
		p5.append(E1[i])
		p6.append(ED[i])
		p7.append(VD[i])
		p8.append(EID[i])
		p9.append(G1[i])
		p10.append(GD[i])
		
		
	return p1,p2,p3,p4,p5,p6,p7,p8,p9,p10				

#Returns the cosine similarities between two vectors of same length			
def get_cosine_sim(list1, list2):
	sumxx = 0.0
	sumyy = 0.0
	sumxy = 0.0
	for i in range(len(list1)):
		x = list1[i]
		y = list2[i]
		sumxx += x*x
		sumyy += y*y
		sumxy += x*y
	if sumxx == 0.0 or sumyy == 0.0:
		return 0.0
	res = sumxy/math.sqrt(sumxx*sumyy)
	if math.isnan(res):
		print "found nan======================>"
		return 0.0
	return res	


X1=[]  #Stores representations of the sequence of venues successfully hosting events similar to the target event in past 
E1=[]  #Stores representations of the sequence of events successfully hosted at the target venue in past 
G1=[]  #Stores representations of the sequence of groups hosting events successfully at the target venue in past
	
ED=[]  #Stores the representation of the target event
VD=[]  #Stores the representation of the target venue
GD=[]  #Stores the representation of the group hosting the target event
	
V1=[]  #Stores the target venue-id
EID=[] #Stores the target event-id
	
S1=[]  #Stores the sequence length
Y1=[]  #Stores the success label of the target event

print "Reading input files initiated ..."
	
#Read the representations of each group
infile=open('./../input/gvec_mtags_500.txt','r')
gvec={}
for line in infile:
	line=(line.strip()).split()
	gvec[int(line[0])]=[]
	for ll in range(1,len(line)):
		gvec[int(line[0])].append(float(line[ll]))
	
#Read the hosting time of each event		
etime = pickle.load(open('./../input/etime','r'))

#Read the event to group mapping for each event
egroup = pickle.load(open('./../input/egroup','r'))
	
#Read the group category for each group
gcat = pickle.load(open('./../input/grp_cat','r'))
	
#Read the distance between venues and groups; each venue-group pair contains four values- fraction of members staying within 5 miles, 10 miles, 20 miles and 50 miles from the venue 
ven_grp =pickle.load(open('./../ven_grp','r'))
			
#Read the representation of the events	
edesc={}
for line in open('./../input/edescription.txt','r'):
	line=(line.strip()).split()
	edesc[line[0]]=[]
	for ll in range(1,len(line)):
		edesc[line[0]].append(float(line[ll]))
			
infile = open('./../input/edes_new_wnan','r')
edes = pickle.load(infile)
infile.close()
	
for ev in edes:
	if ev not in edesc:
		edesc[ev]=edes[ev]
	
#Read the events hosted by each group
infile = open('./../input/grp_event.txt','r')
grp_event={}
for line in infile:
	line=(line.strip()).split()
	grp_event[int(line[0])]=set()
	for e in range(1,len(line)):
		grp_event[int(line[0])].add(line[e])
infile.close()            
	
#Read the representations of venues
vset={}
for line in open('./../input/raw_venue_vec.txt','r'):
	line=(line.strip()).split()
	vset[int(line[0])]=[]
	for ll in range(1,len(line)):
		vset[int(line[0])].append(float(line[ll]))

#Read the event to venue mapping
edata=pickle.load(open('./../input/edata','r'))

#Create venue to event mapping
vdata={}
for event in edata:	
	if edata[event] == None or edata[event] not in vset:
		continue
	if edata[event] not in vdata:
		vdata[edata[event]]=set()
	vdata[int(edata[event])].add(event)

		
#Read success values of events; Estimate the 66.67th percentile of the success values.
infile = open('./../input/sou_eventatt_grpsize_ratio.pickle','r')
e_succ = pickle.load(infile)
infile.close()
evals=[]
for ev in e_succ:
	evals.append(e_succ[ev])
evals.sort()
thres=evals[int((2.0*float(len(evals)))/3.0)]

#Read input events
all_list=read_input(gvec, etime, egroup, edata, vdata, vset, e_succ, edesc, ven_grp, gcat)
X1=[]
Y1=[]
V1=[]
S1=[]
E1=[]
ED=[]
VD=[]
EID=[]
G1=[]
GD=[]

for i in range(0,len(all_list)):
	X1.append(all_list[i][0])
	Y1.append(all_list[i][1])
	V1.append(all_list[i][2])
	S1.append(all_list[i][3])
	E1.append(all_list[i][4])
	VD.append(all_list[i][5])
	ED.append(all_list[i][6])
	EID.append(all_list[i][7])
	G1.append(all_list[i][8])
	GD.append(all_list[i][9])
			
			
# # Training
# Before we can start training we need to bind our input variables for the model and define what optimizer we want to use. For this example we choose the `adam` optimizer. We choose `squared_error` as our loss function.

#Define axes
i1_axis = C.Axis.new_unique_dynamic_axis('1')
i2_axis = C.Axis.new_unique_dynamic_axis('2')
i3_axis = C.Axis.new_unique_dynamic_axis('3')

# input sequences
x1 = C.sequence.input_variable(VVEC_LEN,sequence_axis=i1_axis, name='i1') #Read sequence of venues for venue module from X1
x3 = C.sequence.input_variable(EVEC_LEN,sequence_axis=i2_axis, name='i2') #Read sequence of events for event module from E1
x7 = C.sequence.input_variable(GVEC_LEN,sequence_axis=i3_axis, name='i3') #Read sequence of groups for group module from G1

x5 = C.input_variable(VVEC_LEN) #Read representation of target venue VD
x4 = C.input_variable(EVEC_LEN) #Read representation of target event ED
x8 = C.input_variable(GVEC_LEN) #Read representation of target venue GD


x3m = model1(x3) #Output of sequence model of event module
x4m = model2(x4) #Target event's embedding

x1m = model3(x1) #Output of sequence model of venue module
x5m = model4(x5) #Target venue's embedding

x7m = model5(x7) #Output of sequence model of group module
x8m = model6(x8) #Target event's group's embedding


#Calculate similarities between embedding and output of sequence model for each module
e2 = C.losses.cosine_distance(x3m, x4m, name = "simi1")
v2 = C.losses.cosine_distance(x1m, x5m, name = "simi2")
g2 = C.losses.cosine_distance(x7m, x8m, name = "simi3")


z = v2 * e2 * g2
#Target event's success label
x2 = C.input_variable(1) #Y1


C.logging.log_number_of_parameters(x1m)
C.logging.log_number_of_parameters(x3m)
C.logging.log_number_of_parameters(x4m)
C.logging.log_number_of_parameters(x5m)
C.logging.log_number_of_parameters(x7m)
C.logging.log_number_of_parameters(x8m)


# Set the learning rate
learning_rate = LEARNING_RATE
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)

# Squared loss function
loss = C.squared_error(x2, z)
# Squared error 
error = C.squared_error(x2, z)


if len(X1)<200:
	BATCH_SIZE=5
print "Minibatch size- ",BATCH_SIZE 

# use adam optimizer
momentum_time_constant = C.momentum_as_time_constant_schedule(BATCH_SIZE / -math.log(0.9)) 
learner = C.fsadagrad(z.parameters, 
					  lr = lr_schedule, 
					  momentum = momentum_time_constant)
trainer = C.Trainer(z, (loss, error), [learner])

loss_summary = []

#TRAINING
EPOCHS = TRAINING_ITERATION

#Using 80% data for training
frac=0.8
start = 0
for epoch in range(0, EPOCHS):
	while start < int(float(len(X1))*frac)-BATCH_SIZE:
		x_batch, l_batch, v_batch, s_batch, e1_batch, ed_batch, vd_batch, EID_batch, g1_batch, gd_batch = get_sample(start,BATCH_SIZE)
		trainer.train_minibatch({x1: x_batch, x2: l_batch, x3: e1_batch, x4: ed_batch, x5: vd_batch, x7: g1_batch, x8: gd_batch})
		start+=BATCH_SIZE
	start=0	
	
	if epoch % 5 ==0:
		training_loss = trainer.previous_minibatch_loss_average
		loss_summary.append(training_loss)
		print("epoch: {}, loss: {:.4f}".format(epoch, training_loss))
	
print("Training took {:.1f} sec".format(time.time() - start))

#Calculate Test Error
start=int(float(len(X1))*frac)
result=[]
err=0.0
ct=0.0
while start<len(X1)-BATCH_SIZE:
	x_batch, l_batch, v_batch, s_batch, e1_batch, ed_batch, vd_batch, EID_batch, g1_batch, gd_batch = get_sample(start,BATCH_SIZE)
	pred_lab=z.eval({x1: x_batch, x3: e1_batch, x4: ed_batch, x5: vd_batch, x7: g1_batch, x8: gd_batch})
	err+=trainer.test_minibatch({x1: x_batch, x2: l_batch, x3: e1_batch, x4: ed_batch, x5: vd_batch, x7: g1_batch, x8: gd_batch})
	ct+=1
	for index in range(len(pred_lab)):
		result.append((pred_lab[index],l_batch[index]))
	start+=BATCH_SIZE	

print 'Test error ',err/ct

#Prediction accuracy calculation
maxacc = 0.0
tempthreshold = 0.0
for threshold1 in range(0,1000):
	threshold = threshold1/1000.0 #Vary the threshold from 0 to 1 to find the best accuracy
	ac = 0
	dc = 0
	for values in result:
		if values[0] >=threshold and values[1] ==1:
			ac += 1
		else:
			if values[0] < threshold and values[1]==0:
				ac += 1
			else:
				dc += 1
	print (ac, dc, float(ac)/float(ac+dc))
	acc = float(ac)/ float(ac+dc)
	if maxacc <= acc:
		maxacc = acc
		tempthreshold = threshold
		
print "Test Accuracy ",maxacc
		
#Calculate Recalls and MIR for testset

'''
#Read venue vectors
fp=open('./../raw_venue_vec.txt','r')
venue_vec={}
for line in fp:
	line=(line.strip()).split()		
	id1=int(line[0])
	venue_vec[id1]=[]
	for id2 in range(1,len(line)):
		venue_vec[id1].append(float(line[id2]))
print (len(venue_vec))
vset={}
vset=venue_vec
'''

start=int(float(len(X1))*frac)

#Initializing metrics 
r1=0	#Recall@1  
r5=0    #Recall@5
r10=0   #Recall@10 
r15=0   #Recall@15
r20=0   #Recall@20
r50=0   #Recall@50
r100=0  #Recall@100

cc=0
avg=0.0
MIR=0.0
all_count=0.0

start_time=time.time()
while start<len(X1):
	x_batch, l_batch, v_batch, s_batch, e1_batch, ed_batch, vd_batch, EID_batch, g1_batch, gd_batch = get_sample(start,1)
	map_id=int(v_batch[0])
	ev_id= EID_batch[0]
	seq_len=int(s_batch[0])
	
	if NEW_FLAG==1:
		#Check if the target venue is a new venue for the host group
		new_ven=check_new_ven(grp_event, ev_id, egroup[ev_id],edata,etime)
		if new_ven==1:
			start+=1	
			continue
			
	if l_batch[0]==1: #if the test event is successful
		result=[]
		all_count+=1
		#For each venue find out the predicted score
		print "\nProcessing Event ",cc
		for id1 in vset:
			e1_batch=[]
			g1_batch=[]
			
			exx,gvx=get_venue_vec(vdata, id1, etime, ev_id, e_succ, thres, edesc, gvec, egroup)		
			e1_batch.append(exx)
			g1_batch.append(gvx)
			
			#Read the representation of the venue	
			for index in range(0,100):
				vd_batch[0][index]=np.float(vset[id1][index])
				
			#Add distance of the venue to the group in the representation
			vd_batch[0][100] = np.float(ven_grp[id1][egroup[ev_id]][0])
			vd_batch[0][101] = np.float(ven_grp[id1][egroup[ev_id]][1])
			vd_batch[0][102] = np.float(ven_grp[id1][egroup[ev_id]][2])
			vd_batch[0][103] = np.float(ven_grp[id1][egroup[ev_id]][3])
			
			#Predict the score		
			pred_lab=z.eval({x1: x_batch, x3: e1_batch, x4: ed_batch, x5: vd_batch, x7: g1_batch, x8: gd_batch})
			result.append((pred_lab[0],id1))
			
		#Sort and rank venues based on the predicted score
		result=sorted(result,key=lambda x:x[0],reverse=True)
		flag0=0
		rank=-2
		for rank1 in range(0,len(result)):
			if result[rank1][1]==map_id:
				flag0=1
				rank=rank1
				
		rank+=1
		
		if rank==-1:
			print "Error: Target venue not found ",rank1,flag0		
		else:	
			print "Rank of the target venue ", rank
			print "=================================================================================\n"
			avg+=float(rank)
			MIR+=1.0/float(rank)
			if rank==1:
				r1+=1				
			if rank<=5:
				r5+=1
			if rank<=10:
				r10+=1
			if rank<=15:
				r15+=1	
			if rank<=20:
				r20+=1
			if rank<=50:
				r50+=1
			if rank<=100:
				r100+=1	
			cc+=1	
							 
	start+=1
	
end_time=time.time()

print "Event Count, Recall@1, Recall@5, Recall@10, Recall@15, Recall@20, Recall@50, Recall@100, MIR ", cc, float(r1)/float(cc),float(r5)/float(cc),float(r10)/float(cc),float(r15)/float(cc),float(r20)/float(cc),float(r50)/float(cc),float(r100)/float(cc), MIR/float(cc)	

print "\n\nTime taken per event in seconds ",float(end_time-start_time)/all_count	
