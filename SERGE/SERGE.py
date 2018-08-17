import random
import math
import time
import pickle
from collections import defaultdict
import numpy as np
import sys

#Default Parameter Setting
#Set the history length (number of past events) to be used for each event
HISTORY=100
#Maximum number of iterations for the random walk 
MAXIT=1000 
#Epsilon used as stopping criterion for random walk
EPS=0.0001 
#The category of Meetup groups for which recommendation is needed; Use -1 for any category, 0 for "Activity" category, 1 for "Hobby" category, 2 for "Social" category, 3 for "Entertainment" category and 4 for "Technical" category
CATEGORY=-1
#Flag indicating whether we wish to evaluate only for new events; Set this to 1 if the evaluation has to be done only for new events 
NEW_FLAG=0

if len(sys.argv)==1:
	print "No arguments provided; Using default values"
	
else:
	if len(sys.argv)==6:
		HISTORY=int(sys.argv[1])
		MAXIT=int(sys.argv[2])
		EPS=float(sys.argv[3])
		CATEGORY=int(sys.argv[4])
		NEW_FLAG=int(sys.argv[5])
	else:
		print "Please provide all 5 arguments"
		exit(0)

print "Parameters - History length, Maximum Iterations, Epsilon, Group category, New Events Flag ", HISTORY, MAXIT, EPS, CATEGORY, NEW_FLAG

def read_input():
	#Read events fromt the input file
	f=open('./../data/input_events.txt',"r")
	all_event=[]  

	count_cat={}
	for i in range(0,5):
		count_cat[i]=0
	
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
			
			if j==idd-1 and event in ehost and event in rsvp:
				if CATEGORY==-1 or (group in gcat and gcat[group]==CATEGORY):
					all_event.append((event,group,venue,time1,label,ehost[event]))
					
	f.close()
	return all_event

			
def create_graph(target,tgroup,time1,index):
	events=set()
	groups=set()
	venues=set()
	tags=set()
	users=set()
	hosts=set()
	
	ev={}
	eg={}
	eh={}
	gh={}
	
	ut={}
	gt={}
	
	eu={}
	ug={}
	
	used_ven=set()
	
	#Add different types of entities like users, events, groups, venues, hosts, tags in the graph
	
	for j in range(index-HISTORY,index+1):
		eve=all_event[j]
		if eve[3]<time1:
			events.add(eve[0])
			groups.add(eve[1])
			eg[eve[0]]=eve[1]
			venues.add(eve[2])
			ev[eve[0]]=eve[2]
			
			
			eh[eve[0]]=set()
			if eve[1] not in gh:
				gh[eve[1]]=set()
			for hh in eve[5]:
				hosts.add(hh)
				eh[eve[0]].add(hh)
				gh[eve[1]].add(hh)
			
			
			#if the group hosting this historical event is also hosting the target event, the venue of the historical event is cosidered as an already used venue
			if eve[1]==tgroup:
				used_ven.add(eve[2])
					
					
			#Find out all users who have sent "Yes" rsvp for this historical event	
			
			eu[eve[0]]=set()
			
			for elem in rsvp[eve[0]]:
				user=elem[0]
				answer=elem[1]
				if str(answer)=='yes':
					eu[eve[0]].add(user)
					users.add(user)
		else:
			if eve[0]==target:
				events.add(eve[0])
				groups.add(eve[1])
				eg[eve[0]]=eve[1]			
				target_ven=eve[2]
				
	if NEW_FLAG==1 and target_ven in used_ven:
		return -1
	
					
	ev_list=list(events)
	gr_list=list(groups)
	ven_list=list(venues)
	user_list=list(users)
	host_list=list(hosts)
	tlist=[]
	for i in range(0,100):
		tlist.append(i)
			
		
	print "Number of events, users, groups, venues, tags and hosts in the graph ", len(ev_list), len(user_list), len(gr_list), len(ven_list), len(tlist), len(host_list)
	
	
	for u in user_list:
		if u in memjoin:
			for grps in memjoin[u]:
				if grps[1]<=time1 and grps[0] in gr_list:
					if u not in ug:
						ug[u]=set()
					ug[u].add(grps[0])
						
	
	#Create Host-Tag links and updating corresponding HT and TH adjacency matrices
	HT=[]
	for i in range(0,len(host_list)):
		l=[]
		for j in range(0,len(tlist)):
			flag=0
			if host_list[i] in mtags:
				for tag in mtags[host_list[i]]:
					if tag in tag_clus and tag_clus[tag]==tlist[j]:
						flag=1
			if flag==1:			
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		HT.append(l)
	
	TH=[]
	for i in range(0,len(tlist)):
		l=[]
		for j in range(0,len(host_list)):
			flag=0
			if host_list[j] in mtags:
				for tag in mtags[host_list[j]]:
					if tag in tag_clus and tag_clus[tag]==tlist[i]:
						flag=1
			if flag==1:			
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		TH.append(l)
	
	#Create User-Tag links and updating corresponding UT and TU adjacency matrices		
	UT=[]
	for i in range(0,len(user_list)):
		l=[]
		for j in range(0,len(tlist)):
			flag=0
			if user_list[i] in mtags:
				for tag in mtags[user_list[i]]:
					if tag in tag_clus and tag_clus[tag]==tlist[j]:
						flag=1
			if flag==1:			
				l.append(1.0)
			else:	
				l.append(0.0)	
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		UT.append(l)
	
	TU=[]
	for i in range(0,len(tlist)):
		l=[]
		for j in range(0,len(user_list)):
			flag=0
			if user_list[j] in mtags:
				for tag in mtags[user_list[j]]:
					if tag in tag_clus and tag_clus[tag]==tlist[i]:
						flag=1
			if flag==1:			
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		TU.append(l)
	
	#Create Group-Tag links and updating corresponding GT and TG adjacency matrices
	GT=[]
	for i in range(0,len(gr_list)):
		l=[]
		for j in range(0,len(tlist)):
			flag=0
			if gr_list[i] in gtags:
				for tag in gtags[gr_list[i]]:
					if tag in tag_clus and tag_clus[tag]==tlist[j]:
						flag=1
			if flag==1:			
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		GT.append(l)
	
	TG=[]
	for i in range(0,len(tlist)):
		l=[]
		for j in range(0,len(gr_list)):
			flag=0
			if gr_list[j] in gtags:
				for tag in gtags[gr_list[j]]:
					if tag in tag_clus and tag_clus[tag]==tlist[i]:
						flag=1
			if flag==1:			
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		TG.append(l)
	
	
	#Create Event-Host links and updating corresponding EH and HE adjacency matrices		
	EH=[]
	for i in range(0,len(ev_list)):
		l=[]
		for j in range(0,len(host_list)):
			if ev_list[i] in eh and host_list[j] in eh[ev_list[i]]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		EH.append(l)
	
	HE=[]
	for i in range(0,len(host_list)):
		l=[]
		for j in range(0,len(ev_list)):
			if ev_list[j] in eh and host_list[i] in eh[ev_list[j]]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		HE.append(l)

	
	#Create Group-Host links and updating corresponding GH and HG adjacency matrices
	GH=[]
	for i in range(0,len(gr_list)):
		l=[]
		for j in range(0,len(host_list)):
			if gr_list[i] in gh and host_list[j] in gh[gr_list[i]]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		GH.append(l)
	
	HG=[]
	for i in range(0,len(host_list)):
		l=[]
		for j in range(0,len(gr_list)):
			if gr_list[j] in gh and host_list[i] in gh[gr_list[j]]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		HG.append(l)
	
	#Create Event-Event similarity links and updating corresponding EE adjacency matrix
	EE=[]
	for i in range(0,len(ev_list)):
		l=[]
		for j in range(0,len(ev_list)):
			if ev_list[j]!=ev_list[i] and ev_list[j] in simi_event[ev_list[i]]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)		
		EE.append(l)
		
	#Create Venue-Venue similarity links and updating corresponding VV adjacency matrix
	VV=[]
	for i in range(0,len(ven_list)):
		l=[]
		for j in range(0,len(ven_list)):
			if ven_list[j]!=ven_list[i] and ven_list[j] in simi_ven[ven_list[i]]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)		
		VV.append(l)
	
		
	#Create User-Event links and updating corresponding UE and EU adjacency matrices			
	UE=[]
	for i in range(0,len(user_list)):
		l=[]
		for j in range(0,len(ev_list)):
			if ev_list[j] in eu and user_list[i] in eu[ev_list[j]]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		UE.append(l)
		
	EU=[]
	for i in range(0,len(ev_list)):
		l=[]
		for j in range(0,len(user_list)):
			if ev_list[i] in eu and user_list[j] in eu[ev_list[i]]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		EU.append(l)	
	
	#Create User-Group links and updating corresponding UG and GU adjacency matrices
	UG=[]
	for i in range(0,len(user_list)):
		l=[]
		for j in range(0,len(gr_list)):
			if user_list[i] in ug and gr_list[j] in ug[user_list[i]]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		UG.append(l)
	
	GU=[]
	for i in range(0,len(gr_list)):
		l=[]
		for j in range(0,len(user_list)):
			if user_list[j] in ug and gr_list[i] in ug[user_list[j]]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		GU.append(l)	
	
	#Create Event-Group links and updating corresponding EG and GE adjacency matrices
	EG=[]
	for i in range(0,len(ev_list)):
		l=[]
		for j in range(0,len(gr_list)):
			if gr_list[j]==eg[ev_list[i]]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)				
		EG.append(l)
		
	GE=[]
	for i in range(0,len(gr_list)):
		l=[]
		for j in range(0,len(ev_list)):
			if eg[ev_list[j]]==gr_list[i]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)	
		GE.append(l)
	
	#Create Event-Venue links and updating corresponding EV and VE adjacency matrices	
	EV=[]
	for i in range(0,len(ev_list)):
		l=[]
		for j in range(0,len(ven_list)):
			if ev_list[i]!=target and ven_list[j]==ev[ev_list[i]]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:		
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)		
		EV.append(l)
										
	VE=[]
	for i in range(0,len(ven_list)):
		l=[]
		for j in range(0,len(ev_list)):
			if ev_list[j] in ev and ev[ev_list[j]]==ven_list[i]:
				l.append(1.0)
			else:	
				l.append(0.0)
		sum1=sum(l)		
		if sum1>0:
			for j in range(0,len(l)):
				l[j]=float(l[j])/float(sum1)		
		VE.append(l)
		
	#Initial vectors for each type of entity and the query vector for target event
	#Event vectors
	evec=[]
	for i in range(0,len(ev_list)):
		evec.append(random.random())
	#Group vectors	
	gvec=[]
	for i in range(0,len(gr_list)):
		gvec.append(random.random())	
	#Venue vectors
	vvec=[]
	for i in range(0,len(ven_list)):
		vvec.append(random.random())	
	#User vectors
	uvec=[]
	for i in range(0,len(user_list)):
		uvec.append(random.random())				
	#Host vectors
	hvec=[]
	for i in range(0,len(host_list)):
		hvec.append(random.random())
	#Tag vectors
	tvec=[]
	for i in range(0,len(tlist)):
		tvec.append(random.random())	
	#Query vector	
	query=[]
	for i in range(0,len(ev_list)):
		if ev_list[i]!=target:
			query.append(0)
		else:
			query.append(1)
			
			
	#RANDOM walk with Restart
	
	for it in range(0,MAXIT):
		nevec=0.16*np.matmul(gvec,GE)+0.16*np.matmul(vvec,VE)+0.16*np.matmul(uvec,UE)+0.16*np.matmul(evec,EE)+0.16*np.matmul(hvec,HE)+0.16*np.array(query)
		ngvec=0.25*np.matmul(evec,EG)+0.25*np.matmul(uvec,UG)+0.25*np.matmul(tvec,TG)+0.25*np.matmul(hvec,HG)
		nvvec=0.5*np.matmul(evec,EV)+0.5*np.matmul(vvec,VV)
		nuvec=0.33*np.matmul(evec,EU)+0.33*np.matmul(gvec,GU)+0.33*np.matmul(tvec,TU)
		nhvec=0.33*np.matmul(evec,EH)+0.33*np.matmul(tvec,TH)+0.33*np.matmul(gvec,GH)
		ntvec=0.33*np.matmul(uvec,UT)+0.33*np.matmul(gvec,GT)+0.33*np.matmul(hvec,HT)
		
		#Initialize flag as 0; make flag=1 only if the change from vector obtained in the previous iteration is more than EPS
		flag=0
		maxdiff=-1
		tevec=[]
		for i in range(0,len(nevec)):				
			tevec.append(nevec[i])
		
		sum_tevec=sum(tevec)
		for i in range(0,len(tevec)):
			if evec[i]-tevec[i]>EPS or tevec[i]-evec[i]>EPS:
				flag=1
				if evec[i]-tevec[i]>maxdiff:
					maxdiff=evec[i]-tevec[i]
				else:
					if tevec[i]-evec[i]>maxdiff:
						maxdiff=tevec[i]-evec[i]
			
		tgvec=[]
		for i in range(0,len(ngvec)):
			tgvec.append(ngvec[i])
		
		sum_tgvec=sum(tgvec)
		for i in range(0,len(tgvec)):
			if gvec[i]-tgvec[i]>EPS or tgvec[i]-gvec[i]>EPS:
				flag=1
					
		ttvec=[]
		for i in range(0,len(ntvec)):
			ttvec.append(ntvec[i])	
		
		sum_ttvec=sum(ttvec)
		for i in range(0,len(ttvec)):
			if tvec[i]-ttvec[i]>EPS or ttvec[i]-tvec[i]>EPS:
				flag=1	
			
		tvvec=[]
		for i in range(0,len(nvvec)):
			tvvec.append(nvvec[i])	
		
		sum_tvvec=sum(tvvec)
		for i in range(0,len(tvvec)):
			if vvec[i]-tvvec[i]>EPS or tvvec[i]-vvec[i]>EPS:
				flag=1
		
		tuvec=[]
		for i in range(0,len(nuvec)):
			tuvec.append(nuvec[i])	
		
		sum_tuvec=sum(tuvec)
		for i in range(0,len(tuvec)):
			if uvec[i]-tuvec[i]>EPS or tuvec[i]-uvec[i]>EPS:
				flag=1	
				
		thvec=[]
		for i in range(0,len(nhvec)):
			thvec.append(nhvec[i])	
		
		sum_thvec=sum(thvec)
		for i in range(0,len(thvec)):
			if hvec[i]-thvec[i]>EPS or thvec[i]-hvec[i]>EPS:
				flag=1				
	
		evec=tevec
		gvec=tgvec						
		vvec=tvvec
		uvec=tuvec
		tvec=ttvec	
		hvec=thvec
		if flag==0:
			break
	
	#Find out the score obtained for the actual venue hosting the target event after the Random Walk and its corresponding rank
	val=-99999
	for i in range(0,len(ven_list)):
		if ven_list[i]==target_ven:
			val=vvec[i]
			
	rank=1
	for i in range(0,len(vvec)):
		if vvec[i]>val:
			rank+=1
	return rank
	
#--------------------------------------------------------------------------------------------------------------------						
#Read tags specified for each group
gtags = pickle.load(open('./../data/gtags','r'))

#Read tags specified for each group
mtags = pickle.load(open('./../data/mtags','r'))

#Read groups joined by each user
memjoin = pickle.load(open('./../data/memjoin','r'))

#Read "RSVP"s sent by each user for each event
rsvp = pickle.load(open('./../data/rsvp','r'))

#Read 100 similar venues for each venue 
simi_ven=pickle.load(open('./../data/simi_ven','r'))

#Read 100 similar events for each event
simi_event=pickle.load(open('./../data/simi_event','r'))

#Read subject clusters assigned to each tag
tag_clus=pickle.load(open('./../data/tag_clus','r'))

#Read host-ids for each event
ehost = pickle.load(open('./../data/ehost','r'))

#Read the category assigned to each group
gcat = pickle.load(open('./../data/grp_cat','r'))

#Read all events as input from "input_events.txt". The "all_event" list has details of 1000 events, format of each element (event_id, group_id, venue_id, event_time, event_success_label, event_host_id)
all_event=read_input()
print "Total number of events ",len(all_event)


#Sort all events by time
all_event=sorted(all_event,key=lambda x:x[3])

#Using 30% earlier events only for history 
length=int(0.3*len(all_event))
train_time=all_event[length-1][3]

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
MIR=0.0 #MIR 
all_count=0.0 


start_time=time.time()
for j in range(length,len(all_event)):
	ev=all_event[j]
	event=ev[0]
	time1=ev[3]
	tgroup=ev[1]
	
	if ev[4]==0: #Consider only successful events for evaluation
		continue
	all_count+=1.0	
	print "\nProcessing Event ",cc
	rank=create_graph(event,tgroup,time1, j) #Calling the random walk function
	print "Rank of the target venue ", rank
	if rank>-1: 
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
			
		print "=================================================================================\n"
end_time=time.time()	

print "Event Count, Recall@1, Recall@5, Recall@10, Recall@15, Recall@20, Recall@50, Recall@100, MIR ", cc, float(r1)/float(cc),float(r5)/float(cc),float(r10)/float(cc),float(r15)/float(cc),float(r20)/float(cc),float(r50)/float(cc),float(r100)/float(cc), MIR/float(cc)	

print "\n\nTime taken per event in seconds ",float(end_time-start_time)/all_count	

