
import numpy as np
import random
from tqdm import tqdm
from ujson import load as json_load
import os
import re
import ujson as json
from collections import Counter
from args import get_setup_args
from codecs import open
import pandas as pd
from transformers import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForImageClassification
from string import whitespace
import requests
import time
import copy
import argparse
from datasets import load_dataset
import json
import csv
import pandas as pd
import random
from datasets import load_dataset
from datasets import load_from_disk

import copy





rel_dic={}

rel_dic_SRE={'empty':0,'no_relation':1}

rel_dic_trex={}

rel_dic_lexical={}

with open('essential_files/prop_wiki_all.json') as f:
    properties_wiki = json.load(f)['prop_wiki_all']
properties_wiki_inv = {y: x for x, y in properties_wiki.items()}


trex_choices_analogykb={}


skip_wikidata=['P20','P159','P27','P19','P40','P22','P26',\
           'P9','P157','P495','P108','P69','P106','P735','P57','P40','P22','P26','P570','P9']

lexical_plus_analogykb_rel={}


def pre_process_verbal_analogy():
	dataframe1 = pd.read_excel('unprocessed_data/analogy_data/osfstorage-archive/VerbalAnalogy-AccuraciesRTsbyitem.xlsx')
	data_special=[]
	test=True
	for index, row in dataframe1.iterrows():
	    data=[]
	    a,b=row['ABpair'].split(':')
	    c,d=row['C-term'],row['correct D-term']
	    d_minus=row['Distracter']
	    Relation=row['Relation']
	    Analogy_Stem=row['Analogy Stem']
	    ACC=row['ACC']
	    RT=row['RT']
	    Semantic_Distance=row['Semantic Distance']
	    Distracter_Salience=row['Distracter Salience']
	    a_1,b_1=row['ABpair.1'].split(':')
	    c_1,d_1=row['C-term.1'],row['correct D-term.1']
	    d_minus_1=row['Distracter.1']
	    Relation_1=row['Relation.1']
	    Analogy_Stem_1=row['Analogy Stem.1']
	    ACC_1=row['ACC.1']
	    RT_1=row['RT.1']
	    Semantic_Distance_1=row['Semantic Distance.1']
	    Distracter_Salience_1=row['Distracter Salience.1']
	   
	    prop1={'Distracter_Salience':Distracter_Salience,'Relation':Relation,\
	    'Analogy_Stem':Analogy_Stem,'ACC':ACC,'RT':RT,'Semantic_Distance':Semantic_Distance}
	    prop2={'Distracter_Salience':Distracter_Salience_1,'Relation':Relation_1,\
	    'Analogy_Stem':Analogy_Stem_1,'ACC':ACC_1,'RT':RT_1,'Semantic_Distance':Semantic_Distance_1}
	    dname='special'
	    item1={'test':test,'w1':a  ,'w2':b ,'w3':c ,'w4':d ,'name':dname,'type':'positive','prop':prop1}
	    item1_negative={'test':test,'w1':a  ,'w2':b ,'w3':c ,'w4':d_minus ,'name':dname,'type':'neutral','prop':prop1}
	    ##
	    item2={'test':test,'w1':a_1  ,'w2':b_1 ,'w3':c_1 ,'w4':d_1 ,'name':dname,'type':'positive','prop':prop2}
	    item2_negative={'test':test,'w1':a_1  ,'w2':b_1 ,'w3':c_1 ,'w4':d_minus_1 ,'name':dname,'type':'neutral','prop':prop2}

	    positives=[item1,item2]
	    negatives=[item1_negative,item2_negative]
	    for d1,d2 in zip(positives,negatives):
	    	D=[]
	    	D.append(d1)
	    	D.append(d2)
	    	data_special.append((D,True))
	return data_special

def sort_data(analogy_relation_classification_data,all_relation_dic,data,dname,DATA_ALL,test=False):
	scan_s_m={}
	rel_datasets=['EVALution_easy','EVALution_hard','semeval_2012_easy','semeval_2012_hard',\
	'wikidata_easy','wikidata_hard','analogykb_easy','analogykb_hard']
	hard_data=['EVALution_hard','semeval_2012_hard',\
	'wikidata_hard','analogykb_hard']

		###
	rel_datasets=['google_easy',]
	hard_data=['google_hard',]
	rel_datasets=['semeval_2012_relbert',]
	hard_data=[]

	print('dname',dname)
	for item in data:


		D=[]
		if type(item)!=dict :
		    item= json.loads(item)
		#print('item',item)
		if 'r' in item.keys():
			r=item['r']
			r=str(r)

		else:
			r=str(-1)

		stem=item['stem'] 

		relation_orig=None
		if 'relation_orig' in item.keys():
			relation_orig=item['relation_orig']
			relation_orig=relation_orig.lower()


		src,target=stem[0],stem[1]

		if dname=='scan':#
			print('item',item)

		eval_d=item['eval_d'] if dname=='scan' else None
		choice=item['choice']
		choice_=copy.deepcopy(item['choice'])

		answer=choice[item['answer']]
		Selected_choices_r = item['Selected_choices_r'] if 'Selected_choices_r' in item.keys() else []
		src2,target2=answer[0],answer[1]
		item1={'test':test,'w1':src  ,'w2':target ,'w3':src2 ,'w4':target2 ,'name':dname,'type':'positive','eval_d':eval_d,'r':relation_orig}
		#item1_negative={'test':test,'w1':src  ,'w2':target ,'w3':target2  ,'w4':src2 ,'name':dname,'type':'negative','eval_d':eval_d}

		item1_negative={'test':test,'w1':src  ,'w2':src2 ,'w3':target   ,'w4':target2 ,'name':dname,'type':'positive','eval_d':eval_d}
		item1_negative_={'test':test,'w1':src  ,'w2':target2 ,'w3':target  ,'w4':src2 ,'name':dname,'type':'positive','eval_d':eval_d}

		item1_={'test':test,'w1':src  ,'w2':src2 ,'w3':target  ,'w4':target2 ,'name':dname,'type':'positive','eval_d':eval_d}

	
		item2={'test':test,'w1':target ,'w2':src ,'w3':target2,'w4':src2 ,'name':dname,'type':'positive','eval_d':eval_d}

		#item2_negative={'test':test,'w1':target ,'w2':src ,'w3':src2 ,'w4':target2 ,'name':dname,'type':'negative','eval_d':eval_d}

		item3={'test':test,'w1':src2  ,'w2':target2 ,'w3':src ,'w4':target ,'name':dname,'type':'positive','eval_d':eval_d}
		
		#item3_negative={'test':test,'w1':src2  ,'w2':target2 ,'w3':target  ,'w4':src ,'name':dname,'type':'negative','eval_d':eval_d}

		item4={'test':test,'w1':target2 ,'w2':src2 ,'w3':target,'w4':src ,'name':dname,'type':'positive','eval_d':eval_d}
		
		#item4_negative={'test':test,'w1':target2 ,'w2':src2 ,'w3':src ,'w4':target ,'name':dname,'type':'negative','eval_d':eval_d}
		#item1_={'test':test,'w1':src  ,'w2':src2 ,'w3':target ,'w4':target2 ,'name':dname,'type':'positive','eval_d':eval_d}
		#item1_neg={'test':test,'w1':src  ,'w2':src2 ,'w3':target2 ,'w4':target ,'name':dname,'type':'positive','eval_d':eval_d}
		##
	
		if  dname in rel_datasets:
			print('Selected_choices_r[0]',Selected_choices_r[0])
	
		
			c_r=Selected_choices_r[0].split('*')[1]
			
			s_r=relation_orig.split('*')[1]
			a_r=relation_orig#.split('*')[1]

			c_r_=Selected_choices_r[0].split('*')[0]
			s_r_=relation_orig.split('*')[0]
			a_r_=relation_orig.split('*')[0]
			equ_r=s_r+'p'
			equ_r_=s_r_+'p'



			relation_d={'c_r':c_r,'c_r_':c_r_,'s_r':s_r,'s_r_':s_r_,'equ_r':equ_r,'equ_r_':equ_r_}

			for k in  relation_d.keys():
				if k in ['c_r_','s_r_','equ_r_']:
					t=relation_d[k]
					t='['+t+']'
					if t not in all_relation_dic['category'].keys():
						all_relation_dic['category'][t]=t
				else:
					t=relation_d[k]
					t='['+t+']'
					if t not in all_relation_dic['relation'].keys():
						all_relation_dic['relation'][t]=t

			##
			Selected_choices_=item['Selected_choices_'] if 'Selected_choices_' in item.keys() else []
			random.shuffle(Selected_choices_)
			
			equ_choice=Selected_choices_[0]
			
			

	

			
			# if c_r not in all_relation_dic['relation'].keys():
			# 	all_relation_dic['relation'][c_r]=c_r
			# if c_r_ not in all_relation_dic['category'].keys():
			# 	all_relation_dic['category'][c_r_]=c_r_

			# if s_r not in all_relation_dic['relation'].keys():
			# 	all_relation_dic['relation'][s_r]=s_r
			# if s_r_ not in all_relation_dic['category'].keys():
			# 	all_relation_dic['category'][s_r_]=s_r_
			# ##
			# if equ_r not in all_relation_dic['relation'].keys():
			# 	all_relation_dic['relation'][equ_r]=equ_r
			# if equ_r_ not in all_relation_dic['category'].keys():
			# 	all_relation_dic['category'][equ_r_]=equ_r_


			rel_data_item={'stem_w1':item1['w1'],'stem_w2':item1['w2'],'answer_w1':item1['w3'],'answer_w2':item1['w4'],\
				'negative_w1':choice[0][0],'negative_w2':choice[0][1],'R':relation_d,'equ_w1':equ_choice[0],'equ_w2':equ_choice[1]}
			analogy_relation_classification_data.append(rel_data_item)



		temp_items=[item1,item1_negative,item1_,item2,item3,item4]#+[item1_negative,item2_negative,item3_negative,item4_negative]
		if  test==False:
			

			positive='positive'

			d_=[item1,]#+[item1_negative,]

			#print('choice',len(choice))
			if len(choice) in scan_s_m.keys():
				scan_s_m[len(choice)]=scan_s_m[len(choice)]+1
			else:
				scan_s_m[len(choice)]=1
			for negative in choice:
				if negative==answer:
					continue

				src2_negative,target2_negative=negative[0],negative[1]


				#positive='neutral'
				if dname in hard_data:
					positive='negative'

				item_negative1={'test':test,'w1':src ,'w2':target ,'w3':src2_negative,'w4':target2_negative ,'name':dname,'type':'neutral','eval_d':eval_d}
				#for t in d_neg_:
				d_.append(item_negative1)
				if dname in rel_datasets:
					#break
					if '*' not in Selected_choices_r[0]:
						print('Selected_choices_r',Selected_choices_r)
			
					break

			for t in d_:
				D.append(t)
			item={'d':D,'name':'name'}
			DATA_ALL.append((D,test))

		
		else:


			d_=[item1,]

			####
			if len(choice) in scan_s_m.keys():
				scan_s_m[len(choice)]=scan_s_m[len(choice)]+1
			else:
				scan_s_m[len(choice)]=1
			for ci,negative in enumerate(choice):
			    if negative==answer:
			        continue
			    if len(Selected_choices_r)>0:
			     	r=Selected_choices_r[ci]
			    else:
			     	r=0

			    src2_negative,target2_negative=negative[0],negative[1]
			    item_negative1={'test':test,'w1':src ,'w2':target ,'w3':src2_negative,'w4':target2_negative ,'name':dname,'type':'neutral','eval_d':eval_d,'r':r}
			    d_.append(item_negative1)
			    #if dname in hard_data:


			    	
			#####
			for t in d_:
			    D.append(t)
			item={'d':D,'name':dname}
			#if dname=='semeval_2012':
				#print('item',item)
			DATA_ALL.append((D,test))

	return DATA_ALL



def google_easy_hard():
	colors_dic = {
	'country':"royalblue",
	'capital':'peru',
	'currency':"red", 
	'city':"darkgreen",
	'state':"cyan",
	'male':"orange",
	'female':"gray",
	'verb':"blue",
	'preterite':"magenta",
	'verb':'darkred',
	'plural':'black',
	'adj':'olive',
	'participle':'pink',
	'nationality':'purple',
	'superlative':'teal',
	'antonym':'violet',
	'adverb':'brown',
	'comparative':'darkcyan',
	'noun':'bisque'
	}
	def get_choices(related_dic_concept_based,abstract_spaces_dic,related_dic,stem,easy_hard='easy'):
		choices=[]
		choices_r=[]
		answer=[]
		C1=stem[0][1]
		C2=stem[1][1]
		##
		population=related_dic_concept_based[C1+C2]


		
		L=[i for i in range(len(population))]
		selected=random.sample(L,1)[0]  
		answer=population[selected]
		answer=(answer['w1'],answer['w2'])
		if easy_hard=='hard':
			population_a=abstract_spaces_dic[stem[0][1]]
			population_b=abstract_spaces_dic[stem[1][1]]

			N=5

			L_a=[i for i in range(len(population_a))]
			L_b=[i for i in range(len(population_b))]
			selected_a=random.sample(L_a,N)
			selected_b=random.sample(L_b,N)  
			for i,j in zip(selected_a,selected_b):
				a,b=population_a[i],population_b[j]
				if (a+b) not in related_dic.keys():
					choices.append((a,b))
					r_='p_'+C1+C2
					choices_r.append(r_)
		else:
			n=0
			while n<4:
				population=related_dic
				population=[population[k] for k in population.keys()]

				N=5
				L=[i for i in range(len(population))]
				selected=random.sample(L,N)  
				for s in selected:
					w1,w2,c1,c2=population[s]['w1'],population[s]['w2'],population[s]['c1'],population[s]['c2']
					if c1!=C1 or C2!=c2:
						choices.append((w1,w2))
						r_='p'+c1+c2
						choices_r.append(r_)
						n=n+1
					if n==4:
						break
		choices.append(answer)
		r_='p'+C1+C2
		choices_r.append(r_)
		return choices,answer,choices_r
	file='unprocessed_data/simple_analogy.json'
	data= json.load(open(file))['data']

	Labels=[]
	All_Wv=[]
	cats_all=[]
	abstract_spaces_dic={}
	SimpleAnalogies_Easy=[]
	SimpleAnalogies_Hard=[]

	related_dic={}
	related_dic_concept_based={}

	analogy_data_easy=[]
	analogy_data_hrad=[]


	for k in data:
		d=data[k]
		for t in d:
			print('t',t)
			w1=t['a']
			w2=t['b']
			c1=t['c1']
			c2=t['c2']
			Labels.append(w1)
			Labels.append(w2)
			cats_all.append(c1)
			cats_all.append(c2)
			related_dic[w1+w2]={'w1':w1,'w2':w2,'c1':c1,'c2':c2}
			temp=c1+c2
			temp2={'w1':w1,'w2':w2,'c1':c1,'c2':c2}
			if temp in related_dic_concept_based.keys():

				related_dic_concept_based[temp].append(temp2)
			else:
				related_dic_concept_based[temp]=[]
				related_dic_concept_based[temp].append(temp2)


			head_a=c1
			tail_a=c2
			if head_a in abstract_spaces_dic.keys():
			    if w1!=head_a:
			        abstract_spaces_dic[head_a].append(w1)
			else:
			    abstract_spaces_dic[head_a]=[]
			    if w1!=head_a:
			        abstract_spaces_dic[head_a].append(w1)
			if tail_a in abstract_spaces_dic.keys():
			    if w2!=tail_a:
			        abstract_spaces_dic[tail_a].append(w2)
			else:
			    abstract_spaces_dic[tail_a]=[]
			    if w2!=tail_a:
			        abstract_spaces_dic[tail_a].append(w2)
			##############################################
	for k in data:
		d=data[k]
		for t in d:
			w1=t['a']
			w2=t['b']
			c1=t['c1']
			c2=t['c2']

			choices_easy=[]
			choices_hard=[]
			####
			stem=[(w1,c1),(w2,c2)]
			related_dic

			choices_easy,answer1,choices_r_easy=get_choices(related_dic_concept_based,abstract_spaces_dic,related_dic,stem,easy_hard='easy')
			choices_hard,answer2,choices_r_hard=get_choices(related_dic_concept_based,abstract_spaces_dic,related_dic,stem,easy_hard='hard')



			stem=(stem[0][0],stem[1][0])
			if answer1==stem or answer2==stem:
				continue
			r_s='p'+c1+c2
			r=r_s
			Selected_choices_r_easy=[choices_r_easy[t] for t in range(len(choices_easy))]
			Selected_choices_r_hard=[choices_r_hard[t] for t in range(len(choices_hard))]

			item_easy={'stem':stem,'choice':choices_easy,'answer':-1,'r':r_s,'similarity_target':1,'Selected_choices_r':Selected_choices_r_easy,'relation_orig':r}
			item_hard={'stem':stem,'choice':choices_hard,'answer':-1,'r':r_s,'similarity_target':1,'Selected_choices_r':Selected_choices_r_hard,'relation_orig':r}

			analogy_data_easy.append(item_easy)
			analogy_data_hrad.append(item_hard)
			# print('item_easy',item_easy)
			# print('item_hard',item_hard)
			# print('###################')

	return analogy_data_easy,analogy_data_hrad




def make_analogy_question_analogykb(rel_dic_SRE_copy,reldic_temp,DATA,data_t='concept_net',optoins_equivariance='mix'):
	total=0
	data_dic={}
	analogy_data=[]
	population=[i for i in range(len(DATA))]
	for i,k in enumerate(DATA):
		data_dic[i]=DATA[i]
		#print('DATA[i]',DATA[i])
		#exit()


	def get_choice_r(optoins_equivariance,k_original):
		

		if 'inv'==optoins_equivariance:
			population=list(data_dic.keys())
			selected=random.sample(population,1)[0]
			rel_t_name=selected
		elif 'mix'==optoins_equivariance:
			toss_p=[1,0]
			toss_selected=random.sample(toss_p,1)[0]
			if toss_selected==1:
				rel_t_name=k_original
			else:
				population=list(data_dic.keys())
				selected=random.sample(population,1)[0]
				rel_t_name=selected
		else:
			rel_t_name=k_original


		return rel_t_name

	
	def get_choice(data_dic,i):
		selected=i
		#print('selected',selected)
		#print('data_dic',len(data_dic.keys()))
		selected= json.loads(data_dic[selected])
		#print('selected',selected.keys())
		tuple_=selected['tuple'] if 'tuple' in  list(selected.keys() )else selected['tuple_r1']
		relation=selected['relation'] if 'relation'  in selected.keys() else selected['relation1'][1] 
		relation=relation if 'concept_net' in data_t else relation if data_t=='wiki_analogous' else relation[1]
		if relation.lower() in properties_wiki_inv.keys() :
			relation=properties_wiki_inv[relation.lower()]
		if relation in reldic_temp.keys():
			relation_l_s=reldic_temp[relation]
		else:
			reldic_temp[relation]=len(list(reldic_temp.keys()))
			relation_l_s=reldic_temp[relation]
		population=tuple_
		Selected_choices=random.sample(population,1)[0]
		return Selected_choices,relation_l_s
	def make(k,total,j,rel_flag=True):
		data=[]
		relation=k['relation']
		relation=relation if 'concept_net' in data_t else relation if data_t=='wiki_analogous' else relation[1]
		tuple_=k['tuple']
		if relation.lower() in properties_wiki_inv.keys() :
			relation=properties_wiki_inv[relation.lower()]
		total=total+len(tuple_)
		population=[ii for ii in range(len(tuple_))]

		if data_t=='wiki' :
			sn =5 if len(tuple_)>5 else len(tuple_)
		else:
			sn =5 if len(tuple_)>5 else len(tuple_)
		selected_original=random.sample(population,sn)
		selected_positive=random.sample(population,sn)
		selected_negative=random.sample(population,sn)
		for s_original,s_positive,s_negative in zip(selected_original,selected_positive,selected_negative):

			stem_original=tuple_[s_original]
			stem_positive=tuple_[s_positive]
			stem_negative=tuple_[s_negative]
			# if data_t=='wiki' :
			# 	print('stem_original',stem_original)
			# 	exit()




			k_original=j
			rel_t_name=get_choice_r(optoins_equivariance,k_original)
			choice1,choice1_r=get_choice(data_dic,rel_t_name)

			k_original=j
			rel_t_name=get_choice_r(optoins_equivariance,k_original)
			choice2,choice2_r=get_choice(data_dic,rel_t_name)

			k_original=j
			rel_t_name=get_choice_r(optoins_equivariance,k_original)
			choice3,choice3_r=get_choice(data_dic,rel_t_name)

			k_original=j
			rel_t_name=get_choice_r(optoins_equivariance,k_original)
			choice4,choice4_r=get_choice(data_dic,rel_t_name)

			rel_t_name=j
			choice5,choice5_r=get_choice(data_dic,rel_t_name)
			choice1=choice1[0] if 'concept_net' in data_t else choice1[:2] if data_t=='wiki' else choice1[0]
			choice2=choice2[0] if 'concept_net'  in data_t else choice2[:2] if data_t=='wiki' else choice2[0]
			choice3=choice3[0] if 'concept_net'  in data_t else choice3[:2] if data_t=='wiki' else choice3[0]
			choice4=choice4[0] if 'concept_net'  in data_t else choice4[:2] if data_t=='wiki' else choice4[0]
			choice5=choice5[0] if 'concept_net'  in data_t else choice5[:2] if data_t=='wiki' else choice5[0]

			stem_original=stem_original[0] if 'concept_net' in data_t else stem_original[:2] if data_t=='wiki' else stem_original[0]
			stem_positive=stem_positive[0] if 'concept_net' in data_t else stem_positive[:2] if data_t=='wiki' else stem_positive[0]
			stem_negative=stem_negative[0] if 'concept_net' in data_t else stem_negative[:2] if data_t=='wiki' else stem_negative[0]
			if data_t=='concept_net_analogous':
				choice1=[choice1[0],choice1[-1]]
				choice2=[choice2[0],choice2[-1]]
				choice3=[choice3[0],choice3[-1]]
				choice4=[choice4[0],choice4[-1]]
				choice5=[choice5[0],choice5[-1]]
				stem_original=[stem_original[0],stem_original[-1]]
				stem_positive=[stem_positive[0],stem_positive[-1]]
				stem_negative=[stem_negative[0],stem_negative[-1]]
			if relation in reldic_temp.keys():
				relation_l=reldic_temp[relation]
			else:
				reldic_temp[relation]=len(list(reldic_temp.keys()))
				relation_l=reldic_temp[relation]
			stem_negative=[stem_negative[1],stem_negative[0]]

			Selected_choices=[choice1,choice2,choice3,choice4,choice5,stem_positive]
			Selected_choices_r=[choice1_r,choice2_r,choice3_r,choice4_r,choice5_r,relation_l]

			# Selected_choices=[choice1,choice2,choice3]
			# Selected_choices_r=[choice1_r,choice2_r,choice3_r]
			# if relation[0].lower()!='p':
	    	# 		print(relation,'kb')
	    			#exit()

			item={'stem':stem_original,'choice':Selected_choices,'answer':-1,'r':relation_l,'similarity_target':1,'Selected_choices_r':Selected_choices_r,'relation_orig':relation}
			analogy_data.append(item)

		return True

	Population=[j for j in range(len(DATA))]
	for j,k in enumerate(DATA):
		k = json.loads(k)


		if 'relation' not in k.keys():
			relation1=k['relation1'][1]
			tuple_r1=k['tuple_r1']
			len1=k['len1']

			relation2=k['relation2'][1]
			tuple_r2=k['tuple_r2']
			len2=k['len2']
			k1={'len':len1,'tuple':tuple_r1,'relation':relation1}
			k2={'len':len2,'tuple':tuple_r2,'relation':relation2}
			temp=make(k1,total,j,rel_flag=False)
			temp=make(k2,total,j,rel_flag=False)
		else:
			temp=make(k,total,j)
	file='wordanalogy'+'rel_dic.json'
	h_data={'rel_dic':rel_dic}
	with open(file, 'w') as fp:
		json.dump(h_data, fp)
	return analogy_data



    
def sre_data(all_psotives_wikidata,reldic_temp,train_eval,benchmarks=['conll','semeval','retacred'],optoins_equivariance='mix',easy_plus=False):
	file='unprocessed_data/SRE_Analogy.json'
	SRE_Analogy= json.load(open(file))['SRE_Analogy']
	no_relation_all=['18','no_relation','P0']
	no_relation_examles=[]
	relation_data_all={}
	analogy_data=[]
	skip_wikidata=[]
	population_set_all=[]




	def get_choices_(r,n,population_set):

		# print('relation_data_all',relation_data_all.keys())
		# exit()



		selected=[]
		selected_r=[]


	

		for i in range(n*3):
			if 'equ' == optoins_equivariance:
				temp_r=r
			elif 'mix' == optoins_equivariance:
				toss_p=[1,0]
				toss_selected=random.sample(toss_p,1)[0]
				if toss_selected==1:
					temp_r=r
				else:
					population=list(relation_data_all.keys())
					slctd=random.sample(population,1)[0]
					temp_r=slctd
			elif easy_plus==False:
				population=list(relation_data_all.keys())
				slctd=random.sample(population,1)[0]
				temp_r=slctd

			if easy_plus==True:
				temp_r='p0'


			population=relation_data_all[temp_r]
			population_set=[]


			population_set_all.extend(population_set)
			L=[ij for ij in range(len(population))  ]
			number_of_samples=1
			s=random.sample(L,number_of_samples)
			t=[population[j] for j in s]
			selected.append(t[-1])
			selected_r.append(temp_r)

		########
		heads=[]
		tails=[]
		Rs=[]
		choice_n=0
		for s,r in zip(selected,selected_r):
			#print('s',s)
			s=s['data']
			head=s['a']
			tail=s['b']
			head=s['a'] if s['a'][:3].lower()!='the'  else s['a'][4:]
			tail=s['b'] if s['b'][:3].lower()!='the'  else s['b'][4:]
			head=head.lower()
			tail=tail.lower()
			head_a=s['e1_abstract']
			tail_a=s['e2_abstract']
			head_a=head_a.split(' ')
			tail_a=tail_a.split(' ')
			X=['X','Y','A','B','C','G','H','K','L','M','N','P','Q',]
			X=[t.lower() for t in X]
			head_a=[t if t.lower() not in X   else '' for t in head_a ]
			head_a=' '.join(head_a)
			head_a=head_a.lower()
			tail_a=[t if t.lower() not in X   else '' for t in tail_a ]
			tail_a=' '.join(tail_a) 
			tail_a=tail_a.lower()
	

			if head_a in head or tail_a in tail:
				continue

			heads.append(head)
			tails.append(tail)
			Rs.append(r)
		heads = heads[::-1]
		choices=set()
		for h,t,r in zip(heads,tails,Rs):
			
			if choice_n==n:
				break
			if (h,t,r) not in choices and (h+t) not in population_set:
				choice_n=choice_n+1
				choices.add((h,t,r))
		return list(choices)
	#################

	for data_type in SRE_Analogy.keys():
		rel_keys={}
		#continue
		if data_type not in benchmarks:
			continue
		print('data_type',data_type)
		for d in SRE_Analogy[data_type][train_eval]:
			#print('d',d)
			rel=d['kbID']
			type_=d['type']
			# print('d',d)
			# exit()
			kbID=d['kbID'] #if (d['kbID']!='semeval' and d['kbID']!='retacred') else None
	
			item={'data_name':data_type,'data':d}
			if kbID!=None:
				rel=kbID
			print('rel',rel)
			if rel==None:
				continue
			if rel[0].lower()!='p':
				continue
			#continue
			if rel=='empty' or rel=='none':
				print('rel',rel)
				exit()
			if type_!=train_eval:
				continue

			if rel not in rel_keys.keys():
				rel_keys[rel]=1
				if rel in no_relation_all:
					no_relation_examles.append(item)
				else:
					if rel not in relation_data_all.keys():
						relation_data_all[rel]=[]
						relation_data_all[rel].append(item)
					else:
						relation_data_all[rel].append(item)
			else:
				rel_keys[rel]=rel_keys[rel]+1
				if rel in no_relation_all:
					no_relation_examles.append(item)
				else:
					if rel not in relation_data_all.keys():
						relation_data_all[rel]=[]
						relation_data_all[rel].append(item)
					else:
						relation_data_all[rel].append(item)

	for r in relation_data_all.keys():
		print('r',r)

		population=relation_data_all[r]

		data_name=population[0]['data_name']
		population=[p['data'] for p in population]
		population_set=set()
		for t in population:
			tmp=t['a']+t['b']
			population_set.add(tmp )
			
				

		L=[ij for ij in range(len(population))  ]
		number_of_samples=10 if len(L)>10 else len(L)

		s=random.sample(L,number_of_samples)
		selected=[population[j] for j in s]

		positive_example=random.sample(L,number_of_samples)
		positive_example_T=[population[j] for j in positive_example]
		positive_example=[[t['a'],t['b']] for t in positive_example_T]

		positive_example_abtract=[[t['e1_abstract'],t['e2_abstract']] for t in positive_example_T]
		X=['X','Y','A','B','C','G','H','K','L','M','N','P','Q',]
		X=[t.lower() for t in X]
		positive_example_abtract_=[]
		for a in positive_example_abtract:
			e1,e2=a[0],a[1]

			e1=e1.split(' ')
			e1=[t if t.lower() not in X   else '' for t in e1 ]
			e1=' '.join(e1)
			#
			e2=e2.split(' ')
			e2=[t if t.lower() not in X   else '' for t in e2 ]
			e2=' '.join(e2)

			positive_example_abtract_.append([e1,e2])





		#print('r R',r)
		for si,s in enumerate(selected):
			r=s['r']

			r_kbID=s['kbID'] #if (s['kbID']!='semeval' and s['kbID']!='retacred') else None
			if r_kbID!=None:
				r=r_kbID
		

			if r not in reldic_temp.keys():
				reldic_temp[r]=len(reldic_temp.keys())
				r_s=reldic_temp[r]
			else:
				r_s=reldic_temp[r]

			stem_original=[s['a'],s['b']]
			#############################


			head=s['a'] if s['a'][:3].lower()!='the'  else s['a'][4:]
			tail=s['b'] if s['b'][:3].lower()!='the'  else s['b'][4:]
			head=head.lower()
			tail=tail.lower()
			head_a=s['e1_abstract']
			tail_a=s['e2_abstract']
			head_a=head_a.split(' ')
			tail_a=tail_a.split(' ')
			X=['X','Y','A','B','C','G','H','K','L','M','N','P','Q',]
			X=[t.lower() for t in X]
			head_a=[t if t.lower() not in X   else '' for t in head_a ]
			head_a=' '.join(head_a)
			head_a=head_a.lower()
			tail_a=[t if t.lower() not in X   else '' for t in tail_a ]
			tail_a=' '.join(tail_a) 
			tail_a=tail_a.lower()
			if head_a in head or tail_a in tail:
				# print('head_a',head_a)
				# print('head',head)
				# print('tail_a',tail_a)
				# print('tail',tail)
				# print('##########')
				continue
			stem_original=[s['a'].lower(),s['b'].lower()]

			item_t={'a':stem_original[0].lower(),'b':stem_original[1],'c1':head_a,'c2':tail_a,'rel':r.lower()}
			if r in all_psotives_wikidata.keys():
				all_psotives_wikidata[r].append(item_t)
		
			else:
				all_psotives_wikidata[r]=[]
				all_psotives_wikidata[r].append(item_t)






			###########

			n=4
			choices=get_choices_(r,n,population_set)
			choices=list(set(choices))

			if len(choices)<3:
				# print('choices',choices)
				# print('r',r)
				#exit()
				continue
			Selected_choices=[]
			Selected_choices_r=[]#[r_s for i in range(len(Selected_choices))]
			for c in choices:
				a,b,rc=c[0],c[1],c[2]
				if a ==stem_original[0] and b==stem_original[1]:
					continue
				Selected_choices.append([a,b])
				Selected_choices_r.append(rc)
			#print('Selected_choices',Selected_choices)
			# if len(Selected_choices)>3 : 
			# 	if positive_example_abtract_[0]!=positive_example_abtract_[1]:
			# 		Selected_choices[-1]=positive_example_abtract_[si] 
			# 	else:
			# 		Selected_choices[-1]=[Selected_choices[-1][0],positive_example_abtract_[si][1]] 
			Selected_choices.append([positive_example[si][0].lower(),positive_example[si][1].lower()])
			# print('positive_example[si]',positive_example[si])
			# exit()

			

			item_t={'a':positive_example[si][0].lower(),'b':positive_example[si][1].lower(),\
			'c1':positive_example_abtract_[si][0].lower(),'c2':positive_example_abtract_[si][1].lower(),'rel':r.lower()}
			if r in all_psotives_wikidata.keys():
				all_psotives_wikidata[r].append(item_t)
		
			else:
				all_psotives_wikidata[r]=[]
				all_psotives_wikidata[r].append(item_t)


			# print('Selected_choices',Selected_choices)
			# print('stem_original',stem_original)
			# print('###')
			Selected_choices_r.append(r)
			if r[0].lower()!='p':
				print(r,'sre',r,'line 938 preprocess_util')
				#exit()
			item={'stem':stem_original,'choice':Selected_choices,'answer':-1,'r':r,'similarity_target':1,'Selected_choices_r':Selected_choices_r,'relation_orig':r}
			
			# if stem_original!=Selected_choices[item['answer']]:
			# 	
			analogy_data.append(item)


			
			#print('item',item)

	return analogy_data,all_psotives_wikidata
						



def semeval(all_psotives_semeval,optoins_equivariance,reldic_temp,train_f,gold,hard='hard',flag=True):


	analogy_data=[]
	rel_dic_={}


	def get_semeval_rels(gold,train_f):

		def get_rel_dic():
			f='unprocessed_data/analogy_data/semeval_2012/SemEval-2012-Gold-Ratings/subcategories-list.txt'
			file1 = open(f, 'r')
			rel_dic_semeval={}
			for line in file1:
				line=line.split(',')
				k=line[:2]
				k=''.join(k)
				k=k.replace(' ','')
				v=line[2:]
				v=[t[1:] if t[0]==' ' else t for t in v ]
				v=['-'.join(t.split(' ')) if  len(t.split(' '))>1 else t for t in v ]
				v='#'.join(v)[:-1]
				rel_dic_semeval[k]=v
			return rel_dic_semeval
		if gold:
			if train_f=='test':
				file_name='unprocessed_data/analogy_data/semeval_2012/SemEval-2012-Gold-Ratings/Testing/'
			else:
				file_name='unprocessed_data/analogy_data/semeval_2012/SemEval-2012-Gold-Ratings/Training/'

		else:
			file_name='unprocessed_data/analogy_data/semeval_2012/SemEval-2012-Platinum-Ratings/Phase2AnswersScaled/'#Phase2Answers/'
		import glob

		rel_dic_semeval=get_rel_dic()
		rel_names_dic={}
		mylist = [f for f in glob.glob(file_name+"*.txt")]
		semeval_2012=[]
		all_positive=[]
		repeated=set()
		rel=[]
		j=0
		for fi, f in enumerate(mylist):
		    file1 = open(f, 'r')
		    count = 0
		    f=f.split('/')[-1]
		    f=f.split('-')[-1]
		    f=f.split('.')[0] 
		    L=[]
		    for line in file1:
		        count += 1
		        temp=line.strip()
		        if temp[0]=='#':
		            continue
		        t=temp.split('\t')
		        j=j+1
		        rel_names_dic[f]=rel_dic_semeval[f]
		        break
		        temp=temp.split(':')
		        L.append(temp)

		    file1.close()
		return rel_names_dic
	def get_all_sorted(gold,train_f,n_sample):
		data_triple_type1={}
		data_triple_type2={}
		data_triple_type3={}
		def check_for_qouta(examples):
			exampels_nw=[]

			for e in examples:
				print('e',e)

				w1,w2=e[0],e[1]
				w1=w1.replace('\"','')
				w1=w1.replace('\'','')

				w2=w2.replace('\"','')
				w2=w2.replace('\'','')

				exampels_nw.append([w1,w2])
			return exampels_nw
	


		if gold:
			if train_f=='test':
				file_name='unprocessed_data/analogy_data/semeval_2012/SemEval-2012-Gold-Ratings/Testing/'
			else:
				file_name='unprocessed_data/analogy_data/semeval_2012/SemEval-2012-Gold-Ratings/Training/'

		else:
			file_name='unprocessed_data/analogy_data/semeval_2012/SemEval-2012-Platinum-Ratings/Phase2AnswersScaled/'#Phase2Answers/'
		import glob

		mylist = [f for f in glob.glob(file_name+"*.txt")]

		all_positive=[]
		repeated=set()
		rel=[]
		j=0
		for fi, f in enumerate(mylist):
			# Opening file
			
			file1 = open(f, 'r')
			count = 0

			# Using for loop
			f_name=f.split('/')[-1]
			f_name=f_name.split('-')[-1]
			f_name=f_name.split('.')[0]


			L=[]
			#print("Using for loop")
			for line in file1:
			    count += 1
			    temp=line.strip()

			    if temp[0]=='#':
			    	continue
			    _,temp=temp.split(' ')

			    temp=temp.split(':')
			    #print(temp)
			    L.append(temp)
			    #print("Line{}: {}".format(count, line.strip()))

			# Closing files
			file1.close()
			key=f_name[:-1]+'#'+f_name[-1]
			general_cat=semeval_subcat_dictionary[key][0]
			general_cat_new=general_cat#+'#'+f_name

			positive_examples=L[:n_sample]# if train_f=='train' else L[5:11] 
			negative_examples=L[-n_sample:]# if train_f=='train' else L[-5:] 
			positive_examples=check_for_qouta(positive_examples)
			negative_examples=check_for_qouta(negative_examples)
			
			rel=general_cat+'*'+f_name

			positive_examples=[[p[0],p[1],rel] for p in positive_examples]

			negative_examples=[[p[0],p[1],'random*random'] for p in negative_examples]

			

			data_triple_type1[f_name]={'positive_examples':positive_examples,'negative_examples':negative_examples}

			data_triple_type2[f_name]=positive_examples



			if general_cat_new in data_triple_type3.keys():


				data_triple_type3[general_cat_new]['positive_examples']=data_triple_type3[general_cat_new]['positive_examples']+positive_examples
				#data_triple_type2[general_cat]['negative_examples']=data_triple_type2[general_cat]['negative_examples']+negative_examples


			else:
				data_triple_type3[general_cat_new]={'positive_examples':[],}
				data_triple_type3[general_cat_new]['positive_examples']=data_triple_type3[general_cat_new]['positive_examples']+positive_examples
				#data_triple_type2[general_cat]['negative_examples']=data_triple_type2[general_cat]['negative_examples']+negative_examples


		data_triple_type2_new={}
		for k in data_triple_type2.keys():
				positive_examples=data_triple_type2[k]
				###
				f=True
				while f:
					all_keys=list(data_triple_type2.keys())
		
					L=[ij for ij in range(len(all_keys))  ]
					number_of_samples=1

					selected_K=random.sample(L,number_of_samples)[0]
					sk=all_keys[selected_K]
					key_k=k[:-1]+'#'+k[-1]
					key_sk=sk[:-1]+'#'+sk[-1]
					if sk==k:
						print('sk,k',sk,k)
						#exit()
						#continue
						f=False
					else:
						f=False

					negative_examples=data_triple_type2[sk]
					negative_examples=negative_examples


				data_triple_type2_new[k]={'positive_examples':positive_examples,'negative_examples':negative_examples}
			






		return data_triple_type1,data_triple_type2_new,data_triple_type3


	def get_postivie_negatives(ti,positive_examples,negative_examples,optoins_equivariance,number_samples):


			random.shuffle(positive_examples)
			random.shuffle(negative_examples)

			positive_examples=positive_examples[:number_samples]
			negative_examples=negative_examples[:number_samples]

			negative_examples_copy=copy.deepcopy(negative_examples)

			##
			positive_heads=[p[0] for p in positive_examples]
			positive_tails=[p[1] for p in positive_examples]
			positive_headTails=positive_heads+positive_tails
			if 'equ'==optoins_equivariance:
				As, Bs,Fnames=[],[],[]
				negative_examples=[]
				for t in positive_examples:
					a,b,fnmaes=t[0],t[1],t[2]
					if len(fnmaes)==0:
						print(fnmaes)
						exit()
					As.append(a)
					Bs.append(b)
					Fnames.append(fnmaes)
				As.reverse()
				#random.shuffle(Bs)
				random.shuffle(Bs)
				for t1,t2,fn in zip(As,Bs,Fnames):
					if len(fn)==0:
						print(fn)
						exit()
		
					negative_examples.append([t1,t2,fn])
			elif 'inv'==optoins_equivariance:
				pass


			return positive_examples,negative_examples


	def create_questions(ti,sample_size,fi,analogy_data,positive_examples,negative_examples,optoins_equivariance,flag=True,negative_examples_=None):
			from itertools import combinations

			PERM = combinations(positive_examples, 2)
			PERM =[p for p in PERM]
	
			random.shuffle(PERM)

			if len(PERM)>sample_size:
				PERM=PERM[:sample_size]

			#for i in range(len(positive_examples[:])) :
			for perm in PERM:
				stem=perm[0]#positive_examples[i]

				rel=stem[-1].split('*')[1]
				cat=stem[-1].split('*')[0]

		
				
				Selected_choices=[]
		
				c1='headc_'+str(rel) #if ti==0 else 'headc_'+str(rel)
				c2='tailc_'+str(rel) #if ti==0 else 'tailc_'+str(rel)
				c=c1+c2
				la=stem[0]
				lb=stem[1]
				item={'a':la,'b':lb,'c1':c1,'c2':c2,'rel':stem[-1]}
				if c in all_psotives_semeval.keys():
					all_psotives_semeval[c].append(item)
			
				else:
					all_psotives_semeval[c]=[]
					all_psotives_semeval[c].append(item)

	
				answer=perm[1]
				s1,s2,_=perm[1]
				Selected_choices=[]
				Selected_choices_=[]
				Selected_choices_r=[]
				pre=None
				lower_n=8 if hard=='hard_plus_8' else  5 if 'equ'!=optoins_equivariance  else 5
				if len(negative_examples)<lower_n :
					lower_n=len(negative_examples)
				negative_examples=random.sample(negative_examples,lower_n)

				for t in negative_examples[:]:

					if t not in positive_examples:
						if t[0] not in stem and t[1] not in stem:
							Selected_choices.append(t)
							print(t)
							Selected_choices_r.append(t[2])
					else:
						if pre!=None:
							Selected_choices.append([t[0],pre[1]])
						
				for t in negative_examples_:
					if t not in positive_examples:
						if t[0] not in stem and t[1] not in stem:
							Selected_choices_.append(t)
						
					else:
						if pre!=None:
							Selected_choices_.append([t[0],pre[1]])
							



					pre=t
				print('Selected_choices',len(Selected_choices))

				if len(Selected_choices)<1:
					continue
				if optoins_equivariance=='equ':
					Selected_choices_r=[t+'p' for t in Selected_choices_r]

				if 'equ'==optoins_equivariance and (hard=='hard_plus' or hard=='hard_plus_8'):
					if [t[-1],s2] not in positive_examples:
						Selected_choices[-1]=[t[-1],s2]
					if [s1,t[-1]] not in positive_examples:
						Selected_choices[-1]=[s1,t[-1]]
				Selected_choices.append(perm[1])
				print('Selected_choices',Selected_choices)
				Selected_choices_r.append(stem[-1])
				Selected_choices=[[t[0],t[1]] for t in Selected_choices]
				item={'stem':stem[:-1],'choice':Selected_choices,'answer':-1,'r':stem[-1],'Selected_choices_':Selected_choices_,\
				'similarity_target':1,'relation_orig':stem[-1],'Selected_choices_r':Selected_choices_r}
				analogy_data.append(item)
				# if '*' not in stem[-1]:
				# 	print('stem[-1]',stem[-1])
				# 	exit()
				# if '*' not in Selected_choices_r[0]:
				# 	print('Selected_choices_r',Selected_choices_r)
				# 	exit()
			


	rel_dic_semeval=get_semeval_rels(gold,train_f)
	### semeval 2012 task 2
	#https://drive.google.com/file/d/0BzcZKTSeYL8VYWtHVmxUR3FyUmc/view?resourcekey=0-qrHajiidXTLFlb5kms70tQ
	
	data_triple_type1,data_triple_type2,data_triple_type3=get_all_sorted(gold,train_f,10)
	fi=0
	#for ti,triples_sorted in enumerate([data_triple_type1,data_triple_type2,data_triple_type3]):
	l_data=[data_triple_type1,data_triple_type3] if flag==False else [data_triple_type1,]
	for ti,triples_sorted in enumerate(l_data):
		for p in triples_sorted.keys():
			f_name=p
			fi=fi+1
			v=triples_sorted[p]
			if f_name in rel_dic_semeval.keys():
				number_samples=-1
			else:
				number_samples=-1
				
			positive_examples=v['positive_examples']
			if ti==0 :
				sample_size=450
				if flag:
					sample_size=20
				negative_examples=v['negative_examples'] 


			elif ti==1:
				sample_size=5040
				if flag:
					sample_size=50



				negative_examples=[]
				all_keys=list(data_triple_type2.keys())
	
				L=[ij for ij in range(len(all_keys))  ]
				number_of_samples=4

				selected_Ks=random.sample(L,number_of_samples)
				selected_Ks=[all_keys[j] for j in selected_Ks]
				for k in selected_Ks:
					if f_name!=k:
						temp=data_triple_type2[k]
						positive_examples_k=temp['positive_examples']
						negative_examples.extend(positive_examples_k)





			positive_examples,negative_examples=get_postivie_negatives(ti,positive_examples,negative_examples,optoins_equivariance,number_samples)
			_,negative_examples_=get_postivie_negatives(ti,positive_examples,negative_examples,'equ',6)
			# positive_examples,negative_examples=get_postivie_negatives(ti,positive_examples,negative_examples,optoins_equivariance,number_samples)
			# positive_examples,negative_examples=get_postivie_negatives(ti,positive_examples,negative_examples,optoins_equivariance,number_samples)



			if len(negative_examples_[0])!=3:
				print('positive_examples',positive_examples)
				print('negative_examples_',negative_examples_)
				exit()


			relation='p'+f_name#rel_dic_semeval[f_name]#'semeval_rel_'+str(j)
			relation_other=rel_dic_semeval[f_name]+'no' if f_name in rel_dic_semeval.keys() else 'p'+f_name
			# for p_e,n_e in zip(positive_examples,negative_examples):
			# 	semeval_2012_item={'head':p_e[0],'tail':p_e[1],'relation':relation}
			# 	semeval_2012_item_={'head':n_e[0],'tail':n_e[1],'relation':relation_other}

			# 	semeval_2012.append(semeval_2012_item)

			if relation in reldic_temp.keys():
				relation_l=reldic_temp[relation]
			else:
				reldic_temp[relation]=len(list(reldic_temp.keys()))
				relation_l=reldic_temp[relation]


			create_questions(ti,sample_size,fi,analogy_data,positive_examples,negative_examples,optoins_equivariance,\
				flag=flag,negative_examples_=negative_examples_)







	# if train_f=='test':
	# 	print('end')
	# 	exit()

	return analogy_data,all_psotives_semeval


def multichoice_questions(DATA_words_fast_text,DATA_multi_choice,data,dname,test):
	type_=test
	data=copy.deepcopy(data)
	for d in data:

		print('d',d)
		if type(d)!=dict :
			d= json.loads(d)
		stem=d['stem']
		choice=d['choice']
		answer=d['answer']

		#print('1.choice[answer]',choice[answer])
		t=answer if answer!=-1 else len(choice)-1
		choice_=[choice[i] for i in range(len(choice)) if i!=t]
		
		
		if 'relation_orig' not in d.keys():
			#print('r',d['r'])
			#print('d',d)
			#continue
			r=0
		else:
			r=d['relation_orig']#'relation_orig':relation
		#print('2.choice[answer]',choice[answer])
		if len(choice_)>=4 and dname not in ['scan','semeval_2012_hard','semeval_2012_hard_plus','semeval_2012_hard_plus_8']:
			choice=choice_[:4]+[choice[answer]]
			answer=len(choice)-1
		else :
			choice=choice_[:]+[choice[answer]]
			answer=len(choice)-1

		if len(choice)<5:
			m=5-len(choice)
			for i in range(m):
				choice.append(['empty1','empty2'])
			#print('choice--',len(choice))



		answer_value=choice[answer]
		answer_value_=choice[answer]
		#print('answer_value',answer_value)
		answer_value=answer_value[0] +' is to '+ answer_value[1]

		###
		fst_txt_c=[{'w1':t[0],'w2':t[1]}  for t in choice]
		if dname=='scan':
			eval_d=d['eval_d']
			item_fast_text={'stem':{'w1':stem[0],'w2':stem[1]},'choice':fst_txt_c,'answer':answer,'r':r,'eval_d':eval_d}


		else:

			item_fast_text={'stem':{'w1':stem[0],'w2':stem[1]},'choice':fst_txt_c,'answer':answer,'r':r}



		question=stem[0] +' is to '+ stem[1]

		random.shuffle(choice)
		choice=[t[0] +' is to '+ t[1] for t in choice]
		label=0
		for ci,c in enumerate(choice):
			if c==answer_value:
				label=ci




		item={'question':question,'label':label}

		for ci,c in enumerate(choice):
			key='c'+str(ci)
			item[key]=c
		# print('item',item)
		#print('dname',dname)
		#print('choice',len(choice))
		#print('label',label)


		# print('++++++')
		if 'c4' not in item.keys():
			#print('no c4')
			exit()
		# if len(choice)!=5:
		# 	print('label',label)
		# 	print('choice',choice)
			#exit()
		# if dname=='semeval_2012':
		# 	print('item',item)
		# 	print('answer',choice[label])
		# 	print('#####')
			#exit()
		if test not in DATA_multi_choice.keys():
			DATA_multi_choice[test]={}
			if dname not in DATA_multi_choice[test].keys():
				DATA_multi_choice[test][dname]=[]
				DATA_multi_choice[test][dname].append(item)
			else:
				DATA_multi_choice[test][dname].append(item)
		else:
			if dname not in DATA_multi_choice[test].keys():
				DATA_multi_choice[test][dname]=[]
				DATA_multi_choice[test][dname].append(item)
			else:
				DATA_multi_choice[test][dname].append(item)




		if test not in DATA_words_fast_text.keys():
			DATA_words_fast_text[test]={}
			if dname not in DATA_words_fast_text[test].keys():
				DATA_words_fast_text[test][dname]=[]
			else:
				DATA_words_fast_text[test][dname].append(item_fast_text)
		else:
			if dname not in DATA_words_fast_text[test].keys():
				DATA_words_fast_text[test][dname]=[]
			else:
				DATA_words_fast_text[test][dname].append(item_fast_text)


	if dname=='google':
		pass
		#exit()
	# if dname=='semeval_2012':
	# 		print('item')
	# 		exit()

def porcess_lexical(data_list,names_list,rel_dic_temp,optoins_equivariance):

	data_sorted={}

	Data_list=[]

	temp={}

	covert_rel={
	'synonym':'P5973',\
	'antonym':'P5974',\
	'partof':'P361',\
	'madeof ':'P186',\
	'instacne of':'P31',\
	'hasproperty':'P1533',\
	'isa':'P31',\
	'partof':'p361',\
	'madeof':'p186',\

	}

	
	# temp_dic={'SYN':'synonym','ANT':'has antonym','PART_OF':'part of','attri':'has quality',\
	# 'Antonym':'has antonym','MadeOf':'material used','HasProperty':'has property','hyper':'instance of','HYPER':'instance of'}

	def get_choices(stem_rel,choice_rel):

		pop=temp[choice_rel]
		L=[i for i in range(len(pop))]
		n=4 if len(L)>4 else len(n)

		selected=random.sample(L,n)
		choices=[pop[i] for i in selected]

		if stem_rel==choice_rel:
			As=[]
			Bs=[]
			for t  in choices:
				a=t['head']
				b=t['tail']
				As.append(a)
				Bs.append(b)
			random.shuffle(As)
			random.shuffle(Bs)
			choices=[(h,t) for h,t in zip(As,Bs)]
		else:
			choices=[(t['head'],t['tail']) for t in choices]

		return choices



	for data,dname in zip(data_list,names_list):
		temp={}
		analogy_data=[]

		for d in data :
			#print('d',d.keys())
			head=d['head']
			tail=d['tail']
			relation=d['relation']
			relation=relation.lower()
			if relation in covert_rel.keys():
				relation=covert_rel[relation]
			relation=relation.lower()
			#print('d',d)
			item={'head':head,'tail':tail,'relation':relation,'data_type':'lexical'}
			
			if relation in temp.keys():
				temp[relation].append(item)
			else:
				temp[relation]=[]
				temp[relation].append(item)
		#print('temp',dname,temp.keys())
		for k in temp.keys():
			relation=k

			#print('1.relation',relation)


			if relation in rel_dic_temp.keys():
					relation=rel_dic_temp[relation]
			else:
				rel_dic_temp[relation]=len(rel_dic_temp.keys())
				relation=rel_dic_temp[relation]
			#print('r.relation',relation)

			if k=='random':
				continue
			population=temp[k]
			L=[i for i in range(len(population))]
			n=200 if len(L)>200 else len(L)
			selected=random.sample(L,n)
			for s in selected:
				stem=population[s]
				stem=[stem['head'],stem['tail']]

				s_positive=random.sample(L,1)[0]
				positive_example=population[s_positive]
				#print('positive_example',positive_example)

				positive_example=[positive_example['head'],positive_example['tail']]

				if positive_example==stem:
					s=random.sample(L,1)
					positive_example=population[s[0]]
					positive_example=[positive_example['head'],positive_example['tail']]

				s1,s2=positive_example[0],positive_example[1]
				Selected_choices=[]

				stem_rel=k
				##
				if 'mix' in optoins_equivariance:

					toss_p=[1,0]
					toss_selected=random.sample(toss_p,1)[0]
					if toss_selected==1:
						choice_rel=k

					else:
						#print('temp',temp.keys())
						k_t=random.sample([t for t in temp.keys()],1)[0]

						while k_t==k:
							#print('k_t',k_t)
							#print('k',k)
							k_t=random.sample([t for t in temp.keys()],1)[0]
			

						choice_rel=k_t




				elif 'equ' in optoins_equivariance:
					choice_rel=k

				else:
					#print('temp',temp.keys())
					k_t=random.sample([t for t in temp.keys()],1)[0]
	
					while k_t==k:
						#print('k_t',k_t)
						#print('k',k)
						k_t=random.sample([t for t in temp.keys()],1)[0]


					choice_rel=k_t


				Selected_choices=get_choices(stem_rel,choice_rel)


				Selected_choices.append([s1,s2])
				#Selected_choices.append(postive_ex)

				relation_l=relation

				item={'data_name':dname,'stem':stem,'choice':Selected_choices,'answer':-1,'r':relation_l,'similarity_target':1,'relation_orig':k,'r':k}
				analogy_data.append(item)

				#print('item',item)
		Data_list.append(analogy_data)
	#print('rel_dic_temp',rel_dic_temp.keys())


	return Data_list
		




def pre_process_Ekar(e_kar):
	e_kar_train=e_kar['train']
	e_kar_test=e_kar['test']
	dic_A={'A':0,'B':1,'C':2,'D':3}
	e_kar_train_data=[]
	e_kar_test_data=[]
	for t in e_kar_train:
		#print(t.keys())
		question=t['question']
		stem=question.split(':')
		choices=t['choices']
		answerKey=t['answerKey']
		#########
		answer=dic_A[answerKey]
	
		choices_label=choices['label']
		choices_text=choices['text']
		choices_text=[t.split(':') for t in choices_text]
		choices_text=[[t[0],t[-1]] for t in choices_text]

		item={'stem':stem,'choice':choices_text,'answer':answer}

		e_kar_train_data.append(item)
	###

	for t in e_kar_test:
		question=t['question']
		stem=question.split(':')
		choices=t['choices']
		answerKey=t['answerKey']
		#########
		answer=dic_A[answerKey]

		choices_label=choices['label']
		choices_text=choices['text']
		choices_text=[t.split(':') for t in choices_text]
		choices_text=[[t[0],t[-1]] for t in choices_text]



		item={'stem':stem,'choice':choices_text,'answer':answer}
		e_kar_test_data.append(item)
	return e_kar_train_data,e_kar_test_data




####
semeval_subcat_dictionary = {'1#a': ('CLASS-INCLUSION', 'Taxonomic'),
 '1#b': ('CLASS-INCLUSION', 'Functional'),
 '1#c': ('CLASS-INCLUSION', 'Singular Collective'),
 '1#d': ('CLASS-INCLUSION', 'Plural Collective'),
 '1#e': ('CLASS-INCLUSION', 'ClassIndividual'),
 '2#a': ('PART-WHOLE', 'Object:Component'),
 '2#b': ('PART-WHOLE', 'Collection:Member'),
 '2#c': ('PART-WHOLE', 'Mass:Potion'),
 '2#d': ('PART-WHOLE', 'Event:Feature'),
 '2#e': ('PART-WHOLE', 'Activity:Stage'),
 '2#f': ('PART-WHOLE', 'Item:Topological Part'),
 '2#g': ('PART-WHOLE', 'Object:Stuff'),
 '2#h': ('PART-WHOLE', 'Creature:Possession'),
 '2#i': ('PART-WHOLE', 'Item:Distinctive Nonpart'),
 '2#j': ('PART-WHOLE', 'Item:Ex-part/Ex-possession'),
 '3#a': ('SIMILAR', 'Synonymity'),
 '3#b': ('SIMILAR', 'Dimensional Similarity'),
 '3#c': ('SIMILAR', 'Dimensional Excessive'),
 '3#d': ('SIMILAR', 'Dimensional Naughty'),
 '3#e': ('SIMILAR', 'Conversion'),
 '3#f': ('SIMILAR', 'Attribute Similarity'),
 '3#g': ('SIMILAR', 'Coordinates'),
 '3#h': ('SIMILAR', 'Change'),
 '4#a': ('CONTRAST', 'Contradictory'),
 '4#b': ('CONTRAST', 'Contrary'),
 '4#c': ('CONTRAST', 'Reverse'),
 '4#d': ('CONTRAST', 'Directional'),
 '4#e': ('CONTRAST', 'Incompatible'),
 '4#f': ('CONTRAST', 'Asymmetric Contrary'),
 '4#g': ('CONTRAST', 'Pseudoantonym'),
 '4#h': ('CONTRAST', 'Defective'),
 '5#a': ('ATTRIBUTE', 'ItemAttribute(noun:adjective)'),
 '5#b': ('ATTRIBUTE', 'Object Attribute:Condition'),
 '5#c': ('ATTRIBUTE', 'ObjectState(noun:noun)'),
 '5#d': ('ATTRIBUTE', 'Agent Attribute:State'),
 '5#e': ('ATTRIBUTE', 'Object:Typical Action (noun.verb)'),
 '5#f': ('ATTRIBUTE', 'Agent/ObjectAttribute:Typical Action'),
 '5#g': ('ATTRIBUTE', 'Action:Action Attribute'),
 '5#h': ('ATTRIBUTE', 'Action:Object Attribute'),
 '5#i': ('ATTRIBUTE', 'Action:Resultant Attribute (verb:noun/adjective)'),
 '6#a': ('NON-ATTRIBUTE', 'Item:Nonattribute (noun:adjective)'),
 '6#b': ('NON-ATTRIBUTE',
  'ObjectAttribute:Noncondition (adjective:adjective)'),
 '6#c': ('NON-ATTRIBUTE', 'Object:Nonstate (noun:noun)'),
 '6#d': ('NON-ATTRIBUTE', 'Attribute:Nonstate (adjective:noun)'),
 '6#e': ('NON-ATTRIBUTE', 'Objects:Atypical Action (noun:verb)'),
 '6#f': ('NON-ATTRIBUTE',
  'Agent/Object Attribute: Atypical Action (adjective:verb)'),
 '6#g': ('NON-ATTRIBUTE', 'Action:Action Nonattribute'),
 '6#h': ('NON-ATTRIBUTE', 'Action:Object Nonattribute'),
 '7#a': ('CASE RELATIONS', 'Agent:Object'),
 '7#b': ('CASE RELATIONS', 'Agent:Recipient'),
 '7#c': ('CASE RELATIONS', 'Agent:Instrument'),
 '7#d': ('CASE RELATIONS', 'Action:Object'),
 '7#e': ('CASE RELATIONS', 'Action:Recipient'),
 '7#f': ('CASE RELATIONS', 'Object:Recipient'),
 '7#g': ('CASE RELATIONS', 'Object:Instrument'),
 '7#h': ('CASE RELATIONS', 'Recipient:Instrument'),
 '8#a': ('CAUSE-PURPOSE', 'Cause:Effect'),
 '8#b': ('CAUSE-PURPOSE', 'Cause:Compensatory Action'),
 '8#c': ('CAUSE-PURPOSE', 'EnablingAgent:Object'),
 '8#d': ('CAUSE-PURPOSE', 'Action/Activity:Goal'),
 '8#e': ('CAUSE-PURPOSE', 'Agent:Goal'),
 '8#f': ('CAUSE-PURPOSE', 'Instrument:Goal'),
 '8#g': ('CAUSE-PURPOSE', 'Instrument:Intended Action'),
 '8#h': ('CAUSE-PURPOSE', 'Prevention'),
 '9#a': ('SPACE-TIME', 'Item:Location'),
 '9#b': ('SPACE-TIME', 'Location:Process/Product'),
 '9#c': ('SPACE-TIME', 'Location:Action/Activity'),
 '9#d': ('SPACE-TIME', 'Location:Instrument/Associated Item'),
 '9#e': ('SPACE-TIME', 'Contiguity'),
 '9#f': ('SPACE-TIME', 'Time Action/Activity'),
 '9#g': ('SPACE-TIME', 'Time Associated Item'),
 '9#h': ('SPACE-TIME', 'Sequence'),
 '9#i': ('SPACE-TIME', 'Attachment'),
 '10#a': ('REFERENCE', 'Sign:Significant'),
 '10#b': ('REFERENCE', 'Expression'),
 '10#c': ('REFERENCE', 'Representation'),
 '10#d': ('REFERENCE', 'Plan'),
 '10#e': ('REFERENCE', 'Knowledge'),
 '10#f': ('REFERENCE', 'Concealment')}

not_found_p={
  "alumnus": "p69",
  "aspect ratio": "p2049",
  "attributed to": "p3342",
  "basionym": "p1430",
  "bbfc rating": "p1657",
  "business division": "p3976",
  "captain": "p4125",
  "catholic rite": "p12564",
  "cero rating": "p4120",
  "co-author": "p209",
  "commander of": "p5856",
  "compatriot": "p1542",
  "contains administrative territorial entity": "p150",
  "contributor(s) to the creative work or subject": "p767",
  "cpu": "p3334",
  "design organization": "p1996",
  "director/manager": "p1037",
  "distributor": "p750",
  "domain": "p361",
  "drug used for treatment": "p2176",
  "ec enzyme classification": "p591",
  "ecoregion (wwf)": "p1599",
  "eight banner register": "p5062",
  "excavation director": "p3632",
  "fach": "p5011",
  "father-in-law": "p858",
  "fictional analog of": "p1540",
  "from fictional universe": "p1080",
  "genre of author": "p2524",
  "genre of composer": "p136",
  "gram staining": "p2865",
  "gpu": "p3335",
  "gui toolkit or framework": "p3095",
  "has antonym": "p2977",
  "has edition": "p747",
  "has part": "p527",
  "has parts of the class": "p2670",
  "has property": "p1533",
  "has quality": "p1552",
  "highway system": "p1243",
  "home port": "p555",
  "iucn conservation status": "p141",
  "iucn protected areas category": "p1358",
  "lake inflows": "p2033",
  "lake outflow": "p2034",
  "lakes on river": "p1764",
  "league": "p1340",
  "license": "p275",
  "lithography": "p2069",
  "located in present-day administrative territorial entity": "p1460",
  "located on astronomical location": "p1995",
  "located on terrain feature": "p706",
  "location of final assembly": "p176",
  "material used": "p186",
  "measured by": "p1156",
  "medical condition": "p1050",
  "medical examinations": "p3333",
  "military rank": "p410",
  "mother-in-law": "p859",
  "mouthpiece": "p3206",
  "mtdna haplogroup": "p1908",
  "music collaborator": "p1778",
  "negative therapeutic predictor": "p4082",
  "objective of project or action": "p1110",
  "office held by head of the organization": "p2388",
  "officeholder": "p1313",
  "opponent": "p1269",
  "organization directed from the office or person": "p488",
  "original language of film or tv show": "p364",
  "original network": "p1792",
  "recorded at": "p780",
  "religion": "p140",
  "roman agnomen": "p1443",
  "roman cognomen": "p1442",
  "roman nomen gentilicium": "p1441",
  "roman praenomen": "p1440",
  "satellite bus": "p4056",
  "secretary general": "p3031",
  "series spin-off": "p974",
  "sitter": "p1810",
  "software development organization": "p1994",
  "solves": "p1535",
  "studies": "p2579",
  "subordinate": "p3075",
  "subsidiary": "p355",
  "synonym": "p1889",
  "symptoms": "p780",
  "top-level internet domain": "p880",
  "tourist office": "p2963",
  "use": "p366",
  "victory": "p2035",
  "vox-atypi classification": "p4223",
  "wikidata property example": "p1855",
  "y-dna haplogroup": "p1907",
  "lyrics by": "p676",
  "köppen climate classification": "p1304"
}