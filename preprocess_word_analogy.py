
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



from preprocess_util import *


def pre_process():
	load_from_local_disk=True
	####

	if load_from_local_disk==False:
		e_kar = load_dataset("jiangjiechen/ekar_english")
		#e_kar.save_to_disk('localdatasets/e_kar')
		e_kar_train_data,e_kar_test_data=pre_process_Ekar(e_kar)

		BLESS = load_dataset("relbert/lexical_relation_classification", "BLESS")
		BLESS.save_to_disk('unprocessed_data/BLESS')
		#
		CogALexV = load_dataset("relbert/lexical_relation_classification", "CogALexV")
		CogALexV.save_to_disk('unprocessed_data/CogALexV')

		EVALution = load_dataset("relbert/lexical_relation_classification", "EVALution")
		EVALution.save_to_disk('unprocessed_data/EVALution')
		##
		ROOT09 = load_dataset("relbert/lexical_relation_classification", "ROOT09")
		ROOT09.save_to_disk('unprocessed_data/ROOT09')

		KandH_plus_N = load_dataset("relbert/lexical_relation_classification", "K&H+N")
		KandH_plus_N.save_to_disk('unprocessed_data/KandH_plus_N')
	else:

		e_kar = load_from_disk('unprocessed_data/e_kar')
		e_kar_train_data,e_kar_test_data=pre_process_Ekar(e_kar)
		BLESS = load_from_disk('unprocessed_data/BLESS')
		BLESS_train=BLESS['train']
		BLESS_test=BLESS['test']
		######################################
		#from datasets import load_dataset

		# BLESS = load_dataset("relbert/lexical_relation_classification", "BLESS")
		# BLESS.save_to_disk('localdatasets/BLESS')
		
	        

		BLESS = load_from_disk('unprocessed_data/BLESS')
		BLESS_train=BLESS['train']
		BLESS_test=BLESS['test']

		CogALexV = load_from_disk('unprocessed_data/CogALexV')
		CogALexV_train=CogALexV['train']
		CogALexV_test=CogALexV['test']

		EVALution = load_from_disk('unprocessed_data/EVALution')

		EVALution_train=EVALution['train']
		EVALution_test=EVALution['test']


		ROOT09 = load_from_disk('unprocessed_data/ROOT09')

		ROOT09_train=ROOT09['train']
		ROOT09_test=ROOT09['test']





	################################################################


	# file_name='unprocessed_data/some_extra/nell_relational_similarity/test.jsonl'
	# with open(file_name, 'r') as json_file:
	#     nell_relational_similarity_test = list(json_file)

	# file_name='unprocessed_data/some_extra/nell_relational_similarity/valid.jsonl'
	# with open(file_name, 'r') as json_file:
	#     nell_relational_similarity_train = list(json_file)

	##
	# file_name='unprocessed_data/some_extra/nell_relational_similarity/test.jsonl'
	# with open(file_name, 'r') as json_file:
	#     nell_relational_similarity_test = list(json_file)

	# file_name='unprocessed_data/some_extra/nell_relational_similarity/valid.jsonl'
	# with open(file_name, 'r') as json_file:
	#     nell_relational_similarity_train = list(json_file)
	##

	# file_name='unprocessed_data/some_extra/t_rex_relational_similarity/test.jsonl'
	# with open(file_name, 'r') as json_file:
	#     t_rex_relational_similarity_test = list(json_file)

	# file_name='unprocessed_data/some_extra/t_rex_relational_similarity/valid.jsonl'
	# with open(file_name, 'r') as json_file:
	#     t_rex_relational_similarity_train = list(json_file)



	 #
	file_name='unprocessed_data/some_extra/scan/test.jsonl'
	with open(file_name, 'r') as json_file:
	    scan_test = list(json_file)

	file_name='unprocessed_data/some_extra/scan/valid.jsonl'
	with open(file_name, 'r') as json_file:
	    scan_valid = list(json_file)

	ds_scan_new=[]
	for t in scan_test:
		#print(t)
		t = json.loads(t)
		stem=t['stem']
		choices=t['choice']
		type=t['prefix']
		answer=t['answer']
		eval_d=t['prefix']
		item={'stem':stem,'choice':choices,'answer':answer,'eval_d':eval_d,'r':0}
		ds_scan_new.append(item)
	scan_test=ds_scan_new
	ds_scan_new=[]
	for t in scan_valid:
		t = json.loads(t)
		stem=t['stem']
		choices=t['choice']
		type=t['prefix']
		answer=t['answer']
		eval_d=t['prefix']
		item={'stem':stem,'choice':choices,'answer':answer,'eval_d':eval_d,'r':0}
		ds_scan_new.append(item)
	scan_valid=ds_scan_new










	file_name='unprocessed_data/some_extra/conceptnet_relational_similarity/test.jsonl'
	with open(file_name, 'r') as json_file:
	    ds_conceptnet_relational_similarity_test = list(json_file)

	file_name='unprocessed_data/some_extra/conceptnet_relational_similarity/valid.jsonl'
	with open(file_name, 'r') as json_file:
	    ds_conceptnet_relational_similarity_train = list(json_file)



	file_name='unprocessed_data/analogy_data/analogy_test_dataset/sat/valid.jsonl'
	#file_name='unprocessed_data/some_extra/sat/valid.jsonl'
	with open(file_name, 'r') as json_file:
	    sat_valid = list(json_file)



	file_name='unprocessed_data/analogy_data/analogy_test_dataset/sat/test.jsonl'
	#file_name='unprocessed_data/some_extra/sat/test.jsonl'
	with open(file_name, 'r') as json_file:
	    sat_test = list(json_file)

	file_name='unprocessed_data/analogy_data/analogy_test_dataset/u2/valid.jsonl'
	file_name='unprocessed_data/some_extra/u2/valid.jsonl'
	with open(file_name, 'r') as json_file:
	    u2_valid = list(json_file)


	file_name='unprocessed_data/analogy_data/analogy_test_dataset/u2/test.jsonl'
	file_name='unprocessed_data/some_extra/u2/test.jsonl'
	with open(file_name, 'r') as json_file:
	    u2_test = list(json_file)


	file_name='unprocessed_data/analogy_data/analogy_test_dataset/u4/valid.jsonl'
	file_name='unprocessed_data/some_extra/u4/valid.jsonl'
	with open(file_name, 'r') as json_file:
	    u4_valid = list(json_file)


	file_name='unprocessed_data/analogy_data/analogy_test_dataset/u4/test.jsonl'
	file_name='unprocessed_data/some_extra/u4/test.jsonl'
	with open(file_name, 'r') as json_file:
	    u4_test = list(json_file)

	file_name='unprocessed_data/analogy_data/analogy_test_dataset/bats/valid.jsonl'
	file_name='unprocessed_data/some_extra/bats/valid.jsonl'
	with open(file_name, 'r') as json_file:
	    bats_valid = list(json_file)


	file_name='unprocessed_data/analogy_data/analogy_test_dataset/bats/test.jsonl'
	file_name='unprocessed_data/some_extra/bats/test.jsonl'
	with open(file_name, 'r') as json_file:
	    bats_test = list(json_file)


	file_name='unprocessed_data/analogy_data/analogy_test_dataset/google/valid.jsonl'
	file_name='unprocessed_data/some_extra/google/valid.jsonl'
	with open(file_name, 'r') as json_file:
	    google_valid = list(json_file)


	file_name='unprocessed_data/analogy_data/analogy_test_dataset/google/test.jsonl'
	file_name='unprocessed_data/some_extra/google/test.jsonl'
	with open(file_name, 'r') as json_file:
	    google_test = list(json_file)

	print('len(google_test),len(google_valid)',len(google_test),len(google_valid))

	google_easy,google_hard=google_easy_hard()


	#optoins_equivariance
	wikidata_analogy_train_easy,wikidata_analogy_test_easy,semeval_data_train_easy,semeval_data_test_easy,analogykb_train_easy,analogykb_test_easy,\
	=None,None,None,None,None,None
	wikidata_analogy_train_hard,wikidata_analogy_test_hard,semeval_data_train_hard,semeval_data_train_hard_plus,semeval_data_train_hard_plus_8,\
	semeval_data_test_hard,semeval_data_test_hard_plus,semeval_data_test_hard_plus_8,\
	analogykb_train_hard,analogykb_test_hard = None,None,None,None,None,None,None,None,None,None
	
	for optoins_equivariance in ['inv','equ']:
		if optoins_equivariance in ['inv','mix']:
			item= pre_process_task(optoins_equivariance,load_from_local_disk)

			wikidata_train_easy=item['wikidata_analogy_train']
			wikidata_test_easy=item['wikidata_analogy_test']
			semeval_data_test_easy=item['semeval_data_test']
			semeval_data_train_easy=item['semeval_data_train']

			analogykb_train_easy=item['analogykb_train']
			analogykb_test_easy=item['analogykb_test']
			all_psotives_semeval=item['all_psotives_semeval']
			all_psotives_wikidata=item['all_psotives_wikidata']
			##
			semeval_2012_test_relbert=item['semeval_2012_test_relbert']
			semeval_2012_train_relbert=item['semeval_2012_train_relbert']

			#wikidata_analogy_test_easy_plus

			# wikidata_analogy_test_easy_plus=item['wikidata_analogy_test_easy_plus']
			# wikidata_analogy_train_easy_plus=item['wikidata_analogy_train_easy_plus']


			data_list=[EVALution_train,]
			names_list=['EVALution',]
			rel_dic_temp={}
			EVALution_train_easy=porcess_lexical(data_list,names_list,rel_dic_temp,optoins_equivariance)[0]
			EVALution_test_easy=porcess_lexical(data_list,names_list,rel_dic_temp,optoins_equivariance)[0]
		

				

			
		else:
			item=pre_process_task(optoins_equivariance,load_from_local_disk)

			wikidata_train_hard=item['wikidata_analogy_train']
			wikidata_test_hard=item['wikidata_analogy_test']
			analogykb_train_hard=item['analogykb_train']
			analogykb_test_hard=item['analogykb_test']

			semeval_data_test_hard=item['semeval_data_test']
			semeval_data_train_hard=item['semeval_data_train']

			semeval_data_train_hard_plus=item['semeval_data_train_hard_plus']
			semeval_data_test_hard_plus=item['semeval_data_test_hard_plus']


			semeval_data_train_hard_plus_8=item['semeval_data_train_hard_plus_8']
			semeval_data_test_hard_plus_8=item['semeval_data_test_hard_plus_8']

			all_psotives_wikidata2=item['all_psotives_wikidata']

			# semeval_2012_test_hard=item['semeval_2012_test']
			# semeval_2012_train_hard=item['semeval_2012_train']


			data_list=[EVALution_train,]
			names_list=['EVALution',]
			rel_dic_temp={}
			EVALution_train_hard=porcess_lexical(data_list,names_list,rel_dic_temp,optoins_equivariance)[0]
			EVALution_test_hard=porcess_lexical(data_list,names_list,rel_dic_temp,optoins_equivariance)[0]
	







	####


	simple_analogy_semeval={'data':all_psotives_semeval}
	file='essential_files/all_psotives_semeval.json'

	all_psotives_semeval={'data':all_psotives_semeval}
	with open(file, 'w') as fp:
		json.dump(all_psotives_semeval, fp)



	file='essential_files/all_psotives_wikidata.json'
	

	file='essential_files/all_psotives_wikidata.json'
	for k in all_psotives_wikidata2:
		data=all_psotives_wikidata2[k]

		r=k
		if r in all_psotives_wikidata.keys():
			for t in data:
				if t not in all_psotives_wikidata[r]:
					all_psotives_wikidata[r].append(t)
	
		else:
			all_psotives_wikidata[r]=[]
			for t in data:
				if t not in all_psotives_wikidata[r]:
					all_psotives_wikidata[r].append(t)


	all_psotives_wikidata={'data':all_psotives_wikidata}
	with open(file, 'w') as fp:
		json.dump(all_psotives_wikidata, fp)


	SCAN_dataset_test=scan_test#scan_data#b = random.sample(scan_data, 50)
	SCAN_dataset_train=scan_valid#scan_data
	# for d in scan_data:
	#     if d not in SCAN_dataset_test:
	#         SCAN_dataset_train.append(d)
	
	data_special=pre_process_verbal_analogy()


	NAMES={\
	'semeval_2012_relbert':(semeval_2012_train_relbert,semeval_2012_test_relbert),\
	'semeval_2012_easy':(semeval_data_train_easy,semeval_data_test_easy),\
	'semeval_2012_hard':(semeval_data_train_hard,semeval_data_test_hard),\
	'semeval_2012_hard_plus':(semeval_data_train_hard_plus,semeval_data_test_hard_plus),\
	'semeval_2012_hard_plus_8':(semeval_data_train_hard_plus_8,semeval_data_test_hard_plus_8),\
	# 'wikidata_easy_plus':(wikidata_analogy_train_easy_plus,wikidata_analogy_test_easy_plus),\

	'wikidata_easy':(wikidata_train_easy,wikidata_test_easy),\
	'wikidata_hard':(wikidata_train_hard,wikidata_test_hard),\
	'analogykb_easy':(analogykb_train_easy,analogykb_test_easy),
	'analogykb_hard':(analogykb_train_hard,analogykb_train_hard),\
	'ekar':(e_kar_train_data,e_kar_test_data),\
	 'RS':(ds_conceptnet_relational_similarity_train,ds_conceptnet_relational_similarity_test),\
	'scan':(SCAN_dataset_train,SCAN_dataset_test),
	'google':(google_valid,google_test),'bats':(bats_valid,bats_test),\
	'u4':(u4_valid,u4_test),'u2':(u2_valid,u2_test),'sat':(sat_valid,sat_test),\

	'google_easy':(google_easy,google_easy),\
	'google_hard':(google_hard,google_hard),\
	'sat_test':(sat_valid,sat_test),
	# 't_rex_relational_similarity':(t_rex_relational_similarity_train,t_rex_relational_similarity_test),\
	# 'nell_relational_similarity':(nell_relational_similarity_train,nell_relational_similarity_test),
	'EVALution_easy':(EVALution_train_easy,EVALution_test_easy),\
	'EVALution_hard':(EVALution_train_hard,EVALution_test_hard),\




	}


	





	total=0
	DATA_ALL=[]
	DATA_ALL.extend(data_special)
	DATA_multi_choice={}
	DATA_words_fast_text={}
	all_relation_dic={'relation':{},'category':{}}
	all_relation_dic_={'none':True}
	analogy_relation_classification_data={'test':[],'train':[]}
	for (dname,data) in NAMES.items():

	    if dname=='RS':
	    	test=True
	    else:
	    	test=True
	    data=data[1]
	    total=total+len(data)
	    print('dname',dname)
	    #print('data',data[0])
	    #continue
	    if dname!='speciale':
	    	multichoice_questions(DATA_words_fast_text,DATA_multi_choice,data,dname,'test')
	    DATA_ALL=sort_data(analogy_relation_classification_data['test'],all_relation_dic,data,dname,DATA_ALL,test=test)
	#exit()
	for (dname,data) in  NAMES.items():

	    #print('len(data),dname',len(data),dname)
	    data=data[0]


	    total=total+len(data)
	    if dname!='speciale':
	    	multichoice_questions(DATA_words_fast_text,DATA_multi_choice,data,dname,'train')
	    DATA_ALL=sort_data(analogy_relation_classification_data['train'],all_relation_dic,data,dname,DATA_ALL,test=False)
	print(total)
	
	#######
	train_semeval_re=analogy_relation_classification_data['train']
	test_semeval_re=analogy_relation_classification_data['test']
	file='unprocessed_data/semeval_2012.json'


	semeval_2012_re={'train':train_semeval_re,'test':test_semeval_re}
	semeval_2012_re={'semeval_2012':semeval_2012_re}
	with open(file, 'w') as fp:
		json.dump(semeval_2012_re, fp)



	file='essential_files/wordanalogy_'+'DATA_multi_choice.json'
	h_data={'DATA_multi_choice':DATA_multi_choice}
	with open(file, 'w') as fp:
		json.dump(h_data, fp)
	#DATA_words_fast_text

	file='essential_files/wordanalogy_'+'DATA_words_fast_text.json'
	h_data={'DATA_multi_choice':DATA_words_fast_text}
	with open(file, 'w') as fp:
		json.dump(h_data, fp)

	DATA_ALL=DATA_ALL+data_special
	file='unprocessed_data/word_analogy_dataset.json'
	word_analogy_dataset={'data':DATA_ALL,'name':'test'}
	with open(file, "w") as outfile: 
	    json.dump(word_analogy_dataset, outfile)

	##################################
	file='unprocessed_data/word_analogy_dataset.json'
	word_analogy_dataset= json.load(open(file))['data']
	special=[]
	data_rest=[]
	data_dic={True:{},False:{}}
	for D,_ in word_analogy_dataset:
		t=False
		for d in D:
			print(d)
			name=d['name']
			test=d['test']
			#print('name',name)
			if name=='scan':
				eval_d=d['eval_d']
				name=name+eval_d

			if name in data_dic[test].keys():
				data_dic[test][name]=data_dic[test][name]+1
			else:
				data_dic[test][name]=1
	print('data_dic',data_dic)
	total_test=[data_dic[True][k] for k in data_dic[True].keys()]
	total_train=[data_dic[False][k] for k in data_dic[False].keys()]
	total_test=sum(total_test)
	total_train=sum(total_train)
	print('total_test,total_train',total_test,total_train)
	for t in data_dic.keys():
		print('********* Test: ',t)
		for p in data_dic[t].keys():
		
			c=data_dic[t][p]
			total=total_train if t==False else total_test
			per=c/total
			print('p,count,%',p, c,per)
			print('++')
	print(len(train_semeval_re),len(test_semeval_re))


	file='essential_files/all_relation_dic.json'
	all_relation_dic_all_=all_relation_dic
	all_relation_dic={'data':all_relation_dic}
	with open(file, 'w') as fp:
		json.dump(all_relation_dic, fp)
	print('all_relation_dic',all_relation_dic)
	print('all_relation_dic',len(all_relation_dic_all_))





def pre_process_task(optoins_equivariance,load_from_local_disk):


	##########################################
	#optoins_equivariance



	#######################################################################
	train_eval='train'
	benchmarks=['conll','semeval','retacred','wikidata']#wikidata
	#benchmarks=['semeval','retacred','wikidata']
	benchmarks=['wikidata',]#wikidata

		#choice_from_no_rel=True
	# all_psotives_wikidata={}
	# reldic_temp=rel_dic_SRE
	# wikidata_analogy_train_easy_plus,_=sre_data(all_psotives_wikidata,reldic_temp,train_eval,benchmarks=benchmarks,optoins_equivariance=optoins_equivariance,easy_plus=True)
	# #benchmarks=['conll','semeval','retacred'],choice_from_no_rel=True)

	# train_eval='dev'
	
	# wikidata_analogy_test_easy_plus,_=sre_data(all_psotives_wikidata,reldic_temp,train_eval,benchmarks=benchmarks,optoins_equivariance=optoins_equivariance,easy_plus=True)
	# random.shuffle(wikidata_analogy_test_easy_plus)
	# wikidata_analogy_test_easy_plus=wikidata_analogy_test_easy_plus[:4000]

	####

	#choice_from_no_rel=True
	all_psotives_wikidata={}
	reldic_temp=rel_dic_SRE
	wikidata_analogy_train,all_psotives_wikidata=sre_data(all_psotives_wikidata,reldic_temp,train_eval,benchmarks=benchmarks,optoins_equivariance=optoins_equivariance)
	#benchmarks=['conll','semeval','retacred'],choice_from_no_rel=True)

	train_eval='dev'
	
	wikidata_analogy_test,all_psotives_wikidata=sre_data(all_psotives_wikidata,reldic_temp,train_eval,benchmarks=benchmarks,optoins_equivariance=optoins_equivariance)
	random.shuffle(wikidata_analogy_test)
	wikidata_analogy_test=wikidata_analogy_test[:4000]


	
	##################################

	####
	file_name_concept_net='unprocessed_data/analogy_data/AnalogyKB/Same_Relation_ConceptNet.json'
	file_name_wikidata='unprocessed_data/analogy_data/AnalogyKB/Same_Relation_Wikidata.json'
	with open(file_name_concept_net, 'r') as json_file:
	    AnalogyKB_concpet_net = list(json_file)

	with open(file_name_wikidata, 'r') as json_file:
	    AnalogyKB_wikidata = list(json_file)


	file_name_concept_net='unprocessed_data/analogy_data/AnalogyKB/Analogous_Relation_ConceptNet.json'
	file_name_wikidata='unprocessed_data/analogy_data/AnalogyKB/Analogous_Relation_Wikidata.json'
	with open(file_name_concept_net, 'r') as json_file:
	    AnalogyKB_concpet_net_analogous = list(json_file)

	with open(file_name_wikidata, 'r') as json_file:
	    AnalogyKB_wikidata_analogous = list(json_file)




	###############



	rel_dic_SRE_copy={}#copy.deepcopy(rel_dic_SRE)


	DATA_wikidata=[]
	DATA_concept_net=[]
	reldic_temp=rel_dic#rel_dic
	#AnalogyKB_concpet_net=make_analogy_question_analogykb(rel_dic_SRE_copy,reldic_temp,AnalogyKB_concpet_net,data_t='concept_net',optoins_equivariance=optoins_equivariance)
	AnalogyKB_wikidata=make_analogy_question_analogykb(rel_dic_SRE_copy,reldic_temp,AnalogyKB_wikidata,data_t='wiki',optoins_equivariance=optoins_equivariance)


	#AnalogyKB_concpet_net_analogous=make_analogy_question_analogykb(rel_dic_SRE_copy,reldic_temp,AnalogyKB_concpet_net_analogous,data_t='concept_net_analogous',optoins_equivariance=optoins_equivariance)
	AnalogyKB_wikidata_analogous=make_analogy_question_analogykb(rel_dic_SRE_copy,reldic_temp,AnalogyKB_wikidata_analogous,data_t='wiki_analogous',optoins_equivariance=optoins_equivariance)

	random.shuffle(AnalogyKB_concpet_net_analogous)
	random.shuffle(AnalogyKB_wikidata_analogous)
	reldic_temp_semeval={}
	gold=False
	semeval_2012=[]
	all_psotives_semeval={}


	if optoins_equivariance!='equ':

		semeval_data_p_train,all_psotives_semeval=semeval(all_psotives_semeval,optoins_equivariance,reldic_temp_semeval,'train',gold,flag=False)
		semeval_data_p_test,all_psotives_semeval=semeval(all_psotives_semeval,optoins_equivariance,reldic_temp_semeval,'test',gold,flag=False)

		gold=True
		semeval_data_g_test,all_psotives_semeval=semeval(all_psotives_semeval,optoins_equivariance,reldic_temp_semeval,'test',gold,flag=False)
		#
		semeval_data_g_train,all_psotives_semeval=semeval(all_psotives_semeval,optoins_equivariance,reldic_temp_semeval,'train',gold,flag=False)
		#############

		semeval_data_train_relbert= semeval_data_p_train 

		semeval_data_test_relbert=semeval_data_g_test
#######
	
	gold=False
	semeval_2012=[]
	all_psotives_semeval_={}
	semeval_data_p_train,all_psotives_semeval=semeval(all_psotives_semeval_,optoins_equivariance,reldic_temp_semeval,'train',gold)
	semeval_data_p_test,all_psotives_semeval=semeval(all_psotives_semeval_,optoins_equivariance,reldic_temp_semeval,'test',gold)

	gold=True
	semeval_data_g_test,all_psotives_semeval=semeval(all_psotives_semeval_,optoins_equivariance,reldic_temp_semeval,'test',gold)
	#
	semeval_data_g_train,all_psotives_semeval=semeval(all_psotives_semeval_,optoins_equivariance,reldic_temp_semeval,'train',gold)
	#############

	semeval_data_train_= semeval_data_p_train 

	semeval_data_test_=semeval_data_g_test



	###################

	if optoins_equivariance=='equ':
		gold=False
		semeval_2012={}
		semeval_data_p_train_hard_plus,_=\
		semeval(semeval_2012,optoins_equivariance,reldic_temp_semeval,'train',gold,hard='hard_plus')

		semeval_data_p_test_hard_plus,_=\
		semeval(semeval_2012,optoins_equivariance,reldic_temp_semeval,'train',gold,hard='hard_plus')

		gold=True
		semeval_data_g_test_hard_plus,_=\
		semeval(semeval_2012,optoins_equivariance,reldic_temp_semeval,'test',gold,hard='hard_plus')
		#
		semeval_data_g_train_hard_plus,_=\
		semeval(semeval_2012,optoins_equivariance,reldic_temp_semeval,'train',gold,hard='hard_plus')


		#########################

		semeval_data_train_hard_plus= semeval_data_p_train_hard_plus+semeval_data_p_test_hard_plus #+semeval_data_p_train

		semeval_data_test_hard_plus=semeval_data_g_test_hard_plus


		####################
		gold=False
		semeval_data_p_train_hard_plus_8,_=\
		semeval(semeval_2012,optoins_equivariance,reldic_temp_semeval,'train',gold,hard='hard_plus_8')
		semeval_data_p_test_hard_plus_8,_=\
		semeval(semeval_2012,optoins_equivariance,reldic_temp_semeval,'test',gold,hard='hard_plus_8')

		gold=True
		semeval_data_g_test_hard_plus_8,_=\
		semeval(semeval_2012,optoins_equivariance,reldic_temp_semeval,'test',gold,hard='hard_plus_8')
		#
		semeval_data_g_train_hard_plus_8,_=\
		semeval(semeval_2012,optoins_equivariance,reldic_temp_semeval,'train',gold,hard='hard_plus_8')
		#########################

		semeval_data_train_hard_plus_8= semeval_data_p_test_hard_plus_8 #+semeval_data_p_train

		semeval_data_test_hard_plus_8=semeval_data_g_test_hard_plus_8




	print('wikidata_analogy_train',len(wikidata_analogy_train))
	print('wikidata_analogy',len(wikidata_analogy_test))

	file='essential_files/'+'wordanalogy'+'rel_dic.json'
	h_data={'rel_dic':rel_dic}
	with open(file, 'w') as fp:
		json.dump(h_data, fp)






	analogykb_concept_net=AnalogyKB_concpet_net[:]#+AnalogyKB_concpet_net_analogous[:-1000]
	analogykb_wikidata=AnalogyKB_wikidata[:]#+AnalogyKB_wikidata_analogous[:-1000]
	analogykb_test=AnalogyKB_concpet_net[:4000]

	analogykb_train=analogykb_wikidata#+analogykb_concept_net

	analogykb_test=AnalogyKB_wikidata_analogous[:1000]#+AnalogyKB_concpet_net_analogous[:1000]





	if optoins_equivariance!='equ':
		item={\
		# 'wikidata_analogy_test_easy_plus':wikidata_analogy_test_easy_plus,\
		# 'wikidata_analogy_train_easy_plus':wikidata_analogy_train_easy_plus,\


		'wikidata_analogy_train':wikidata_analogy_train,\
		'wikidata_analogy_test':wikidata_analogy_test,\
		'semeval_data_train':semeval_data_train_,\
		'semeval_data_test':semeval_data_test_,\
		'analogykb_train':analogykb_train,\
		'analogykb_test':analogykb_test,\
		'all_psotives_wikidata':all_psotives_wikidata,\
		'all_psotives_semeval':all_psotives_semeval,\
		'semeval_2012_train_relbert':semeval_data_train_relbert,\
		'semeval_2012_test_relbert':semeval_data_test_relbert,\

		}
		return item
	else:

		item={\
		'wikidata_analogy_train':wikidata_analogy_train,\
		'wikidata_analogy_test':wikidata_analogy_test,\
		'semeval_data_train':semeval_data_train_,\
		'semeval_data_test':semeval_data_test_,\
		'semeval_data_train_hard_plus':semeval_data_train_hard_plus,\
		'semeval_data_test_hard_plus':semeval_data_test_hard_plus,\
		'semeval_data_train_hard_plus_8':semeval_data_train_hard_plus_8,\
		'semeval_data_test_hard_plus_8':semeval_data_test_hard_plus_8,\
		'analogykb_train':analogykb_train,\
		'analogykb_test':analogykb_test,\
		'all_psotives_wikidata':all_psotives_wikidata,\
		'all_psotives_semeval':all_psotives_semeval,\
		# 'semeval_2012_train':semeval_data_train,\
		# 'semeval_2012_test':semeval_data_test,\

		}
		return item



if __name__ == "__main__":
	pre_process()

