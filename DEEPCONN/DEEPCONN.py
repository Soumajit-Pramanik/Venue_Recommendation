import cntk as C
import numpy as np
from cntk.logging import ProgressPrinter
from cntk.layers import *
from cntk.ops import functions
import random
import pickle
import math
from operator import itemgetter
import sys
import os
from cntk import Trainer, Axis
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs,\
        INFINITELY_REPEAT
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk import input_variable, cross_entropy_with_softmax, \
        classification_error, sequence
import time

#Word vector lengths for event and venues
VEC_DIM=300
HDIM=50


#Droupout
DROP=0.1
#Minibatch Size
BATCH_SIZE=128
#Learning Rate
LEARNING_RATE=0.003
#Number of Epochs used for training
TRAINING_ITERATION=2000
#The category of Meetup groups for which recommendation is needed; Use -1 for any category, 0 for "Activity" category, 1 for "Hobby" category, 2 for "Social" category, 3 for "Entertainment" category and 4 for "Technical" category
CATEGORY=-1 
#Flag indicating whether we wish to evaluate only for new events; Set this to 1 if the evaluation has to be done only for new events 
NEW_FLAG=0

if len(sys.argv)==1:
	print "No arguments provided; Using default values"
	
else:
	if len(sys.argv)==7:
		DROP=float(sys.argv[1])
		BATCH_SIZE=int(sys.argv[2])
		LEARNING_RATE=float(sys.argv[3])
		TRAINING_ITERATION=int(sys.argv[4])
		CATEGORY=int(sys.argv[5])
		NEW_FLAG=int(sys.argv[6])
	else:
		print "Please provide all 6 arguments"
		exit(0)

print "Parameters - Dropout, Minibatch Size, Learning Rate, Number of training Epochs, Group category, New Events Flag- ", DROP, BATCH_SIZE, LEARNING_RATE, TRAINING_ITERATION, CATEGORY, NEW_FLAG


#Passing the word matrix through Convolution, Max pooling and dense layers
def conv_model(para):
    for i in range(0,HDIM):
        f = Convolution((3,VEC_DIM),activation=C.relu)
        if i==0:
            pp=C.reduce_max(f(para))
        else:
            pp=splice(pp,C.reduce_max(f(para)))    
    
    h3=dense_model(pp)
    return h3

#Dense layer
def dense_model(para):
	with default_options(init = glorot_uniform()):
		h1 = Dense(HDIM, activation= C.relu)(para)
		h2 = Dropout(DROP)(h1)		
		return h1
 	

#Returns 0 if the target venue is never used earlier by the group hosting the target event
def check_new_ven(grp_event, event, grp,edata,etime):
	flag=0
	for e1 in grp_event[grp]:
		if e1 in etime and etime[e1]<etime[event]:	#Checks all the events hosted earlier	
			if e1 in edata and edata[e1]==edata[event]: 
				flag=1
				break
	return flag


def read_input(gcat):
	#Read events fromt the input file
	f=open('./../data/sampled_events.txt','r')
	all_event=set()  

	for lin in f:
		line=(lin.strip()).split()
		idd=int((len(line)-4)/5)
		index=0
	
		for j in range(0,idd):
			event=line[index]
			group=int(line[index+1])
			venue=int(line[index+2])
			time1=int(line[index+3])
			label=int(line[index+4])
			index+=5
			
			if j==idd-1:
				if CATEGORY==-1 or (group in gcat and gcat[group]==CATEGORY):
					all_event.add(event)
					
	f.close()
	return all_event



print "Reading input files initiated ..."

#Reading venue representation 
venue_vec1=pickle.load(open('./../data/venue_vec2','rb'))

venue_vec={}
for v in venue_vec1:
	venue_vec[v]=[]
	for v1 in venue_vec1[v]:
		if v1!=0.0:
			venue_vec[v].append(v1)
		else:
			venue_vec[v].append([0.0]*VEC_DIM)


#Reading event representation			
event_vec=pickle.load(open('./../data/event_vec2','rb'))

lvenue_vec=len(venue_vec)
levent_vec=len(event_vec)

#Read the hosting time of each event		
etime = pickle.load(open('./../data/etime','r'))

#Read the event to group mapping for each event
egroup = pickle.load(open('./../data/egroup','r'))

#Read the events hosted by each group
grp_event = pickle.load(open('./../data/grp_event','r'))	

#Read the event to venue mapping
edata=pickle.load(open('./../data/edata','r'))

#Read success values of events; Estimate the 66.67th percentile of the success values.
e_succ = pickle.load(open('./../data/e_succ','r'))

#Read the group category for each group
gcat = pickle.load(open('./../data/grp_cat','r'))


#Read all events as input from "selected_events_new.txt". The "all_event" list has details of 1866 events, format of each element (event_id, group_id, venue_id, event_time, event_success_label, event_host_id)	
all_events=read_input(gcat)	

evals=[]
for ev in e_succ:
	evals.append(e_succ[ev])
evals.sort()
thres=evals[int((2.0*float(len(evals)))/3.0)]

print "Reading input files done"

lab=[]
ctt1=0
ctt0=0
for e in all_events:
	if e in event_vec and e in edata and edata[e] != None and int(edata[e]) in venue_vec and e in e_succ:
		if e_succ[e]<thres:
		    ctt0+=1
		else:
		    ctt1+=1
print "Number of Unsuccessful and Successful events in the dataset - ", ctt0,ctt1
ct=min(ctt0,ctt1)

ctt1=0
ctt0=0
for e in all_events:
    # if (e1 in edesc or str(e1) in edesc) and (e2 in edesc or str(e2) in edesc):
        # print (e1,e2)
    if e in e_succ and e in event_vec and e in edata and edata[e] != None and int(edata[e]) in venue_vec:    
        if e_succ[e]<thres:
            ctt0+=1
            if (ct>=200 and ctt0<=ct) or ct<200:
		        lab.append((e,edata[e],0))
		       
        if e_succ[e]>=thres:
            ctt1+=1
            if (ct>=200 and ctt1<=ct) or ct<200:
		        lab.append((e,edata[e],1))

lab2=[]
for e,v,lb in lab:
	if e in event_vec and v in venue_vec:
		lab2.append((e,v,lb))
lab=lab2
random.shuffle(lab)

print "Overall number of events for training and testing- ",len(lab)
print "Finished Ground Truth Generation"



#Returns "length" number of instances starting from index "start"	
def get_sample(length,start):
    d1=[]
    v1=[]
    tar=[]
    for i in range(start,start+length):
        e=lab[i][0]
        v=lab[i][1]
        l=lab[i][2]

        try:
            if e in event_vec and v in venue_vec:
            	a=np.array(venue_vec[v],dtype=np.float32)
            	a1=[]
            	a1.append(a)
            	a1=np.array(a1,dtype=np.float32)
                v1.append(a1)
                a=[]
                for e1 in event_vec[e]:
                    if e1!=0.0:
		        		a.append(e1)
                    else:
                        a.append([0.0]*VEC_DIM)
                
                a=np.array(a,dtype=np.float32)
                a1=[]
            	a1.append(a)
            	a1=np.array(a1,dtype=np.float32)
                d1.append(a1)
                tar.append(np.array([l]))
            else:
            	print 'Error'
            	exit()
        except KeyError:
            print 'Error'
            exit()
            continue
            
    return d1,v1,tar


# # Training
# Before we can start training we need to bind our input variables for the model and define what optimizer we want to use. For this example we choose the `adam` optimizer. We choose `squared_error` as our loss function.

i1_axis = C.Axis.new_unique_dynamic_axis('1')
i2_axis = C.Axis.new_unique_dynamic_axis('2')

#Venue part
xv = C.sequence.input_variable((1,2316,VEC_DIM))
hv_conv = conv_model(xv)

#Event part
xe = C.sequence.input_variable((1,2826,VEC_DIM))
he_conv = conv_model(xe)

#Ground Truth Success label
target = C.sequence.input_variable(1,np.float32)

#Predicted success label of target event
venue_model = C.cosine_distance(hv_conv, he_conv, name = "simi")

#Squared loss
venue_loss = C.squared_error(target,venue_model)
#Squared error
venue_error = C.squared_error(target,venue_model)

lr_per_sample = [LEARNING_RATE]
lr_schedule = C.learners.learning_rate_schedule(lr_per_sample, C.learners.UnitType.sample, epoch_size = 10)

momentum_as_time_constant = C.learners.momentum_as_time_constant_schedule(700)
# use adam optimizer
venue_learner = C.learners.adam(venue_model.parameters,
                           lr=lr_schedule, momentum=momentum_as_time_constant)

trainer = C.train.Trainer(venue_model, (venue_loss, venue_error), [venue_learner])



if len(lab)<200:
	BATCH_SIZE=5
print "Minibatch size- ",BATCH_SIZE 


#TRAINING
EPOCHS = TRAINING_ITERATION
start=0

pp = C.logging.ProgressPrinter(128)

#Using 80% data for training
for i in range(EPOCHS):
    e1,v1,tar=get_sample(BATCH_SIZE,start)
    start=start+int(BATCH_SIZE)
    if start>=int(0.8*float(len(lab)))-BATCH_SIZE:
        start=0
    
    trainer.train_minibatch({xv: v1,xe: e1,target: tar})
    pp.update_with_trainer(trainer)

    pp.epoch_summary()
print ('Training Done')

#Calculate Test Error
start=int(0.8*float(len(lab)))
result=[]
err=0.0
ct=0.0

while start<int(len(lab))-int(BATCH_SIZE):
	e1,v1,tar=get_sample(BATCH_SIZE,start)
	start=start+int(BATCH_SIZE)
	err+=trainer.test_minibatch({xv: v1,xe: e1,target: tar})
	pred_lab =venue_model.eval({xv: v1,xe: e1})
	ct+=1
	for index in range(len(pred_lab)):
		result.append((pred_lab[index],tar[index]))

print 'test error ',err/ct

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
	acc = float(ac)/ float(ac+dc)
	if maxacc <= acc:
		maxacc = acc
		tempthreshold = threshold
		
print "Test Accuracy ",maxacc


#Calculate Recalls and MIR for testset

#Initializing metrics 
MIR=0.0
r1=0	#Recall@1  
r5=0    #Recall@5
r10=0   #Recall@10 
r15=0   #Recall@15
r20=0   #Recall@20
r50=0   #Recall@50
r100=0  #Recall@100
count=0

start=int(0.8*float(len(lab)))


start_time=time.time()

while start<int(len(lab)):
    e=lab[start][0]
    map_id=lab[start][1]
    tar=lab[start][2]
    
    if NEW_FLAG==1:
		#Check if the target venue is a new venue for the host group
		new_ven=check_new_ven(grp_event, e, egroup[e], edata, etime)
		if new_ven==1:
			start+=1	
			continue
    
    #Evaluate only if the test event is successful
    if tar==0:
        start+=1
        continue
    if edata[e]!=map_id:
        continue
    
    count+=1
    print "\nProcessing Event ",count
    
    if count==20:
        break
        
    rank_list=[]
    a=[]
    for e1 in event_vec[e]:
        if e1!=0.0:
            a.append(e1)
        else:
            a.append([0.0]*VEC_DIM)
    evec=np.array(a,dtype=np.float32)
    t=[]
    t.append(evec)
    t=np.array(t,dtype=np.float32)
    evec=[]
    evec.append(t)
    
    #For each venue find out the predicted score
    for vt in venue_vec:
        vvec=np.array(venue_vec[vt],dtype=np.float32)
        t=[]
        t.append(vvec)
        t=np.array(t,dtype=np.float32)
        vvec=[]
        vvec.append(t)
        
        #Predict the score	
        predicted_label_prob =venue_model.eval({xv: vvec,xe: evec})
        rank_list.append((vt,predicted_label_prob[0]))
    
    #Sort and rank venues based on the predicted score
    rank_list=sorted(rank_list,key=lambda x:x[1],reverse=True)
    
    r=0.0
    for vt,p in rank_list:
        r+=1.0
        if vt==map_id:
            if r==1:
                r1+=1
            if r<=5:
                r5+=1
            if r<=10:
                r10+=1
            if r<=15:
                r15+=1    
            if r<=20:
                r20+=1
            if r<=50:
                r50+=1
            if r<=100:
                r100+=1
            MIR+=1.0/r
            print "Rank of the target venue ", r
            print "=================================================================================\n"
            break
    start+=1
    
end_time=time.time()  

cc=count
if cc==0 and NEW_FLAG==1:
	print "No new venues in the test set. Please try again. "
else:	
	print "Event Count, Recall@1, Recall@5, Recall@10, Recall@15, Recall@20, Recall@50, Recall@100, MIR ", cc, float(r1)/float(cc),float(r5)/float(cc),float(r10)/float(cc),float(r15)/float(cc),float(r20)/float(cc),float(r50)/float(cc),float(r100)/float(cc), MIR/float(cc)	

	print "\n\nTime taken per event in seconds ",float(end_time-start_time)/float(count)	  
