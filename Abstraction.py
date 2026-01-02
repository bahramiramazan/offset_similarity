import torch 

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from tqdm import tqdm
from ujson import load as json_load
import os
import re
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import ujson as json
from collections import Counter
import numpy as np
import os
# import spacy
import ujson as json

from args import get_setup_args
from codecs import open
from collections import Counter
import pandas as pd
import random 

#nlp = spacy.blank("en")
#######################

import torch
from transformers import BertTokenizer, BertModel
import logging
import matplotlib.pyplot as plt


#####################
# import spacy
# nlp2 = spacy.load("en_core_web_lg")


import time 
import requests


def get_ent_instances_or_subclass(e,type='subclass'):
	
	
    number_try=0
    flag=True
    if type=='subclass':
    	p='P279'
    else:
    	p='P31'
    n=1
    while flag:
        if number_try==n:
            #print('rel:',e)
            #print('rel was not found')
            return False

        query1 =(
        ' SELECT ?subclass ?subclassLabel ?description WHERE { ', 
                '?subclass wdt:'+p+' wd:'+e+'.' ,
            '  ?subclass schema:description ?description .',
            ' FILTER (langMatches( lang(?description), "en" ) ) ',
        '\n SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } } LIMIT  100')
        query1="".join(query1)
        # print(query1)
        # print(query1)
        # exit()

        url = 'https://query.wikidata.org/sparql'
        try:

            data = requests.get(url, params={'query': query1, 'format': 'json'},timeout=30).json()


        except requests.exceptions.Timeout:
            #print("Timed out")
            continue
        except :
            #print('failed number_try',number_try)
            number_try=number_try+1

            continue
        Ps=[]
        if len(data['results']['bindings'])>0:
            #print(data['results']['bindings'][0]['rel']['value'])
            for d in data['results']['bindings']:
                #print(d)
                p=d['subclass']['value']
                p=str(p).split('/')[-1]

                l=d['subclassLabel']['value']
                description=d['description']['value']
                l=str(l).split('/')[-1]
                item={'label':l,'q':p,'description':description}
       
                Ps.append(item)
            
       
            return Ps


def get_ent_instances_or_subclass2(e,type='subclass'):
	
	
    number_try=0
    flag=True
    if type=='subclass':
    	p='P279'
    else:
    	p='P31'
    n=1
    while flag:
        if number_try==n:
            #print('rel:',e)
            #print('rel was not found')
            return False

        query1 =(
        ' SELECT ?subclass ?subclassLabel ?description WHERE { ', 
                'wd:'+e+' wdt:'+p+' ?subclass.' ,
            '  ?subclass schema:description ?description .',
            ' FILTER (langMatches( lang(?description), "en" ) ) ',
        '\n SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } } LIMIT  100')
        query1="".join(query1)
        # print(query1)
        # print(query1)
        # exit()

        url = 'https://query.wikidata.org/sparql'
        try:

            data = requests.get(url, params={'query': query1, 'format': 'json'},timeout=30).json()


        except requests.exceptions.Timeout:
            #print("Timed out")
            continue
        except :
            #print('failed number_try',number_try)
            number_try=number_try+1

            continue
        Ps=[]
        if len(data['results']['bindings'])>0:
            #print(data['results']['bindings'][0]['rel']['value'])
            for d in data['results']['bindings']:
                #print(d)
                p=d['subclass']['value']
                p=str(p).split('/')[-1]

                l=d['subclassLabel']['value']
                description=d['description']['value']
                l=str(l).split('/')[-1]
                item={'label':l,'q':p,'description':description}
       
                Ps.append(item)
            
       
            return Ps
def get_instance_label_description(e):

    
    flag=True
    number_try=0
    n=1
    while flag:
        if number_try==n:
            #print('rel:',e)
            #print('rel was not found')
            return False

        query1 =(
        ' SELECT * WHERE { ', 
            '  wd:'+e+' rdfs:label ?label. ',
            '  wd:'+e+' schema:description ?description .',
    
            ' FILTER (langMatches( lang(?description), "en" ) ) ',
        '} LIMIT  1')
        query1="".join(query1)
        #print(query1)
        #print(query1)

        url = 'https://query.wikidata.org/sparql'
        try:

            data = requests.get(url, params={'query': query1, 'format': 'json'},timeout=30).json()


        except requests.exceptions.Timeout:
            #print("Timed out")
            continue
        except :
            #print('failed number_try',number_try)
            number_try=number_try+1

            continue
        Ps=[]
        if len(data['results']['bindings'])>0:
            #print(data['results']['bindings'][0]['rel']['value'])
            for d in data['results']['bindings']:
                print(d)
         
                l=d['label']['value']
                description=d['description']['value']
                l=str(l).split('/')[-1]
                item={'label':l,'description':description}
                Ps.append(item)
            
       
            return Ps

def get_ent_subclass(e):

    
    flag=True
    number_try=0
    n=1
    while flag:
        if number_try==n:
            #print('rel:',e)
            #print('rel was not found')
            return False

        query1 =(
        ' SELECT * WHERE { ', 
            
                'wd:'+e+' wdt:P279 ?instance.' ,
        '} LIMIT  1')
        query1="".join(query1)
        #print(query1)
        #print(query1)

        url = 'https://query.wikidata.org/sparql'
        try:

            data = requests.get(url, params={'query': query1, 'format': 'json'},timeout=30).json()


        except requests.exceptions.Timeout:
            #print("Timed out")
            continue
        except :
            #print('failed number_try',number_try)
            number_try=number_try+1

            continue
        Ps=[]
        if len(data['results']['bindings'])>0:
            #print(data['results']['bindings'][0]['rel']['value'])
            for d in data['results']['bindings']:
                #print(d)
                p=d['instance']['value']
                p=str(p).split('/')[-1]
       
                Ps.append(p)
            
       
            return Ps

def get_instance_label(e):

    
    flag=True
    number_try=0
    n=1
    while flag:
        if number_try==n:
            #print('rel:',e)
            #print('rel was not found')
            return False

        query1 =(
        ' SELECT * WHERE { ', 
            '  wd:'+e+' rdfs:label ?label. ',
            ' FILTER (langMatches( lang(?label), "EN" ) ) ',
        '} LIMIT  1')
        query1="".join(query1)
        #print(query1)
        #print(query1)

        url = 'https://query.wikidata.org/sparql'
        try:

            data = requests.get(url, params={'query': query1, 'format': 'json'},timeout=30).json()


        except requests.exceptions.Timeout:
            #print("Timed out")
            continue
        except :
            #print('failed number_try',number_try)
            number_try=number_try+1

            continue
        Ps=[]
        if len(data['results']['bindings'])>0:
            #print(data['results']['bindings'][0]['rel']['value'])
            for d in data['results']['bindings']:
                #print(d)
         
                l=d['label']['value']
                l=str(l).split('/')[-1]
                Ps.append(l)
            
       
            return Ps

def recusrive_call(e_p,label_p,parent,class_sub_class_dic,level_n_width):
	print('recusrive_call')





	time.sleep(2)
	#L=get_ent_instances(e_p)
	L=get_ent_instances_or_subclass(e_p)

	if L:
		parent=parent.copy()
		parent.append((e_p,label_p))
		for l in L:
			e,label,description=l['q'],l['label'],l['description']
			if e_p not in class_sub_class_dic['has_child'].keys():
					class_sub_class_dic['has_child'][e_p]=[]
			else:
				class_sub_class_dic['has_child'][e_p].append(e)


			if e not in class_sub_class_dic['q_to_labels'].keys():
				class_sub_class_dic['q_to_labels'][e]={'description':description,'parent':parent.copy(),'label':label,'leaf':False,type:'class'}

			len_parent=len(parent)
			if  len_parent<=19:

				if len(class_sub_class_dic['has_child'][e_p])<=level_n_width[len_parent]:
					class_sub_class_dic['has_child'][e_p].append(e)
					t=class_sub_class_dic['has_child'][e_p]
					t=list(set(t))
					class_sub_class_dic['has_child'][e_p]=t
					parent=parent.copy()

					recusrive_call(e,label,parent,class_sub_class_dic,level_n_width)

	else:
		print('parent',parent)
		print('leave',e_p,label_p)
	
		class_sub_class_dic['q_to_labels'][e_p]['leaf']=True

		h_data={'class_sub_class_dic':class_sub_class_dic}
		with open('class_sub_class_dic_save.json', 'w') as fp:
			json.dump(h_data, fp)



def root_entities():
	file='query_wikidata_top_ontology.json'
	wikidata_top_ontology = json.load(open(file))
	#{'item': 'http://www.wikidata.org/entity/Q103940464', 'superclass': 'http://www.wikidata.org/entity/Q35120'}

	#for d in wikidata_top_ontology:
		# item=d['item']
		# item=item.split('/')[-1]
		# superclass=d['superclass']
		# superclass=superclass.split('/')[-1]
		# print(item,superclass)
	e='Q35120'#Q7048977 #Q246672  Q108163 Q11563

	level_n_width={}
	for i in range(20):
		level_n_width[i]=(24-i*3) if (24-i+3)>0 else 2


	parent=[]
	ent_name='entity'

	class_sub_class_dic={'q_to_labels':{},'has_child':{}}
	class_sub_class_dic['q_to_labels'][e]={'parent':[('Q35120','entity')],'label':'bstract-entity','leaf':False}#'abstract entity'

	with open('class_sub_class_dic_save.json') as f:
		class_sub_class_dic = json.load(f)
		class_sub_class_dic=class_sub_class_dic['class_sub_class_dic']

    

	#class_sub_class_dic['has_instances']={}
	option=3
	if option==1:
		class_sub_class_dic['q_to_labels'][e]={'description':'root entity','parent':[],'label':'entity'}#{'parent':[('Q35120','entity')],'label':'bstract-entity'}
		recusrive_call(e,ent_name,parent,class_sub_class_dic,level_n_width)


	###################################################
	else:
		if option==2:

			has_child=class_sub_class_dic['has_child']
			q_to_labels=class_sub_class_dic['q_to_labels']

			keys=list(q_to_labels.keys()).copy()
			for qi,q in enumerate(keys):
				# childs=has_child[q]
				# for child in childs: 
					# q=child
			
				
				item=q_to_labels[q]
				leaf=item['leaf'] if 'leaf' in item.keys() else 0
				if leaf==True: 
					continue
			
				print('item',item)
				print('leaf',leaf)
				print('qi,q,label: ',qi,q,item['label'])

				time.sleep(1)
				instances=get_ent_instances_or_subclass(q,type='instance')
				if instances:
					for l in instances:
						qe,le,description=l['q'],l['label'],l['description']
						if q in class_sub_class_dic['has_instances'].keys():
							class_sub_class_dic['has_instances'][q].append(qe)
							class_sub_class_dic['has_instances'][q]=list(set(class_sub_class_dic['has_instances'][q]))
						else:
							class_sub_class_dic['has_instances'][q]=[]
							class_sub_class_dic['has_instances'][q].append(qe)
							class_sub_class_dic['has_instances'][q]=list(set(class_sub_class_dic['has_instances'][q]))
						if qe not in class_sub_class_dic['q_to_labels'].keys():
							q_p_label=class_sub_class_dic['q_to_labels'][q]['label']
							qe_parent=class_sub_class_dic['q_to_labels'][q]['parent']+[q,q_p_label]
							#print('qe_parent',qe_parent)
						
							class_sub_class_dic['q_to_labels'][qe]={'description':description,'parent':qe_parent,'label':le,'leaf':True,type:'instance_of'}

				

				print('*******************************************************************')
				#class_sub_class_dic
				if instances:
					print('instances',instances)
					h_data={'class_sub_class_dic':class_sub_class_dic}
					with open('class_sub_class_dic_save.json', 'w') as fp:
						json.dump(h_data, fp)
		else:

			has_child=class_sub_class_dic['has_child']
			q_to_labels=class_sub_class_dic['q_to_labels']

			for qc in q_to_labels.keys():
				print('qc',qc)
				if qc in class_sub_class_dic['has_instances'].keys():
						k_instances= class_sub_class_dic['has_instances'][qc]
						k_labels=class_sub_class_dic['q_to_labels'][qc]
						print('*******************************************************************')
						print('k',k_labels)
						print('+++++')
						print('k_instances',k_instances)
						for q in  k_instances:
							if q in class_sub_class_dic['q_to_labels']:
								q_labels=class_sub_class_dic['q_to_labels'][q]['label']
								q_des=class_sub_class_dic['q_to_labels'][q]['description'] if 'description' in class_sub_class_dic['q_to_labels'][q].keys() else 'empty'

								print('q_labels,description',q_labels,'   :   ',q_des)
							else:
								pass
								# time.sleep(1)
								# q_l=get_instance_label(q)
								# if q_l :
								# 	print('q_l',q_l)







def get_ent_super_instances_superclass(e,type='superclass'):
	
	
    number_try=0
    flag=True
    if type=='superclass':
    	p='P279'
    else:
    	p='P31'
    n=1
    while flag:
        if number_try==n:
            #print('rel:',e)
            #print('rel was not found')
            return False

        query1 =(
        ' SELECT ?subclass ?subclassLabel ?description WHERE { ', 
                'wd:'+e+' wdt:'+p+' ?subclass.' ,
            '  ?subclass schema:description ?description .',
            ' FILTER (langMatches( lang(?description), "en" ) ) ',
        '\n SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } } LIMIT  100')
        query1="".join(query1)
        # print(query1)
        # print(query1)
        # exit()

        url = 'https://query.wikidata.org/sparql'
        try:

            data = requests.get(url, params={'query': query1, 'format': 'json'},timeout=30).json()


        except requests.exceptions.Timeout:
            #print("Timed out")
            continue
        except :
            #print('failed number_try',number_try)
            number_try=number_try+1

            continue
        Ps=[]
        if len(data['results']['bindings'])>0:
            #print(data['results']['bindings'][0]['rel']['value'])
            for d in data['results']['bindings']:
                #print(d)
                p=d['subclass']['value']
                p=str(p).split('/')[-1]

                l=d['subclassLabel']['value']
                description=d['description']['value']
                l=str(l).split('/')[-1]
                item={'label':l,'q':p,'description':description}
       
                Ps.append(item)
            
       
            return Ps
		
def root_entites2():

	Books=[]
	file_name='wikidata_entity_instances_subclass_dic.json'
	instances_subclass_dic= json.load(open(file_name))#['instances_dic']
	labels=instances_subclass_dic['labels']

	class_sub_class_dic={'q_to_labels':{},'has_subclass':{},'has_instances':{},'no_instance':[]}
	j=0
	for k in labels.keys():
		q1_l=labels[k]
		q1=k
		time.sleep(1)
		print('****************************************************************')
		print('q1,q1l',q1,q1_l)
		if q1=='Q5':
			continue
		super_class=get_ent_super_instances_superclass(q1)
		super_class_=[q['label'] for q in super_class] if super_class else 0
		print('super_class',super_class_)
		print('++++++++++++++++')

		j=j+1
		visited=[]
		for spi,sp in enumerate(super_class):
			
	
			q,label,description=sp['q'],sp['label'],sp['description']

			if q in visited:
				continue
			visited.append(q)



			########
			if q not in class_sub_class_dic['q_to_labels'].keys():
				item={'q':q,'label':label,'description':description,'parent':list(set([s['q'] for s in super_class])),'type':'class'}
				class_sub_class_dic['q_to_labels'][q]=item

			######
			if q in class_sub_class_dic['has_subclass'].keys():
				class_sub_class_dic['has_subclass'][q].append(q1)
				class_sub_class_dic['has_subclass'][q]=list(set(class_sub_class_dic['has_subclass'][q]))
			else:
				class_sub_class_dic['has_subclass'][q]=[]
				class_sub_class_dic['has_subclass'][q].append(q1)
				class_sub_class_dic['has_subclass'][q]=list(set(class_sub_class_dic['has_subclass'][q]))

			time.sleep(1)
			sub_classes=get_ent_instances_or_subclass(q)
			sub_classes_=[q['label'] for q in sub_classes] if sub_classes else 0
			print('q,label',q,label)
			print('sub_classes',sub_classes_)
			print('++++++++++++++++')
			
			#continue
			
			

			
			if sub_classes:
				for si,sb in enumerate(sub_classes):
					q_sb,label_sb,description_sb=sb['q'],sb['label'],sb['description']
					#############################
					time.sleep(1)
					sub_classes2=get_ent_instances_or_subclass(q_sb)
					if sub_classes2:
						print('sub_classes2',sub_classes2)
					

						print('si,sb',si ,label_sb)
						for si2,sb2 in enumerate(sub_classes2):
							q_sb2,label_sb2,description_sb2=sb2['q'],sb2['label'],sb2['description']
							if q_sb2 not in class_sub_class_dic['q_to_labels'].keys():
								item={'q':q_sb2,'label':label_sb2,'description':description_sb2,'parent':[q_sb],'type':'class'}
								class_sub_class_dic['q_to_labels'][q_sb2]=item

							######
							if q_sb in class_sub_class_dic['has_subclass'].keys():
								class_sub_class_dic['has_subclass'][q_sb].append(q_sb2)
								class_sub_class_dic['has_subclass'][q_sb]=list(set(class_sub_class_dic['has_subclass'][q_sb]))
							else:
								class_sub_class_dic['has_subclass'][q_sb]=[]
								class_sub_class_dic['has_subclass'][q_sb].append(q_sb2)
								class_sub_class_dic['has_subclass'][q_sb]=list(set(class_sub_class_dic['has_subclass'][q_sb]))



							if si2==20:
								break

							##########################
							if q_sb2 not in class_sub_class_dic['has_instances'].keys() and q_sb2 not in class_sub_class_dic['no_instance']:
								print('get instances',q_sb)
								if q_sb2=='Q5':
									continue
								
								time.sleep(1)
								instances=get_ent_instances_or_subclass(q_sb2,type='instance')
								instances_=[q['label'] for q in instances] if instances else 0
								print('q,label',q_sb2,label_sb2)
								print('instances_level2',instances_)
								print('########################')

								if instances==False:

									 class_sub_class_dic['no_instance'].append(q_sb2)
									 class_sub_class_dic['no_instance']=list(set(class_sub_class_dic['no_instance']))
						
								if instances:
									#print()
									for ins in instances:

										q_ins,label_ins,description_ins=ins['q'],ins['label'],ins['description']
										if q_ins not in class_sub_class_dic['q_to_labels'].keys():
											item={'q':q_ins,'label':label_ins,'description':description_ins,'parent':[q_sb2],'type':'instance'}
											class_sub_class_dic['q_to_labels'][q_ins]=item
										if q_sb2 in class_sub_class_dic['has_instances'].keys():
											class_sub_class_dic['has_instances'][q_sb2].append(q_ins)
											class_sub_class_dic['has_subclass'][q_sb2]=list(set(class_sub_class_dic['has_instances'][q_sb2]))
										else:
											class_sub_class_dic['has_instances'][q_sb2]=[]
											class_sub_class_dic['has_instances'][q_sb2].append(q_ins)
											class_sub_class_dic['has_instances'][q_sb2]=list(set(class_sub_class_dic['has_instances'][q_sb2]))

					elif True:

						if q_sb not in class_sub_class_dic['q_to_labels'].keys():
							item={'q':q_sb,'label':label_sb,'description':description_sb,'parent':[q],'type':'class'}
							class_sub_class_dic['q_to_labels'][q_sb]=item

						######
						if q in class_sub_class_dic['has_subclass'].keys():
							class_sub_class_dic['has_subclass'][q].append(q_sb)
							class_sub_class_dic['has_subclass'][q]=list(set(class_sub_class_dic['has_subclass'][q]))
						else:
							class_sub_class_dic['has_subclass'][q]=[]
							class_sub_class_dic['has_subclass'][q].append(q_sb)
							class_sub_class_dic['has_subclass'][q]=list(set(class_sub_class_dic['has_subclass'][q]))




						##########################
						if q_sb not in class_sub_class_dic['has_instances'].keys() and q_sb not in class_sub_class_dic['no_instance']:
							print('get instances',q_sb)
							if q_sb=='Q5':
								continue
							
							time.sleep(1)
							instances=get_ent_instances_or_subclass(q_sb,type='instance')
							instances_=[q['label'] for q in instances] if instances else 0
							print('q,label',q_sb,label_sb)
							print('instances',instances_)
							print('-----------------')
						
							if instances==False:

								 class_sub_class_dic['no_instance'].append(q_sb)
								 class_sub_class_dic['no_instance']=list(set(class_sub_class_dic['no_instance']))
					
							if instances:
								print()
								for ins in instances:

									q_ins,label_ins,description_ins=ins['q'],ins['label'],ins['description']
									if q_ins not in class_sub_class_dic['q_to_labels'].keys():
										item={'q':q_ins,'label':label_ins,'description':description_ins,'parent':[q_sb],'type':'instance'}
										class_sub_class_dic['q_to_labels'][q_ins]=item
									if q_sb in class_sub_class_dic['has_instances'].keys():
										class_sub_class_dic['has_instances'][q_sb].append(q_ins)
										class_sub_class_dic['has_subclass'][q_sb]=list(set(class_sub_class_dic['has_instances'][q_sb]))
									else:
										class_sub_class_dic['has_instances'][q_sb]=[]
										class_sub_class_dic['has_instances'][q_sb].append(q_ins)
										class_sub_class_dic['has_instances'][q_sb]=list(set(class_sub_class_dic['has_instances'][q_sb]))







def get_instances_of_Qs(class_sub_class_dic,Qs):
	dic={}
	for sp in Qs:
		q,label,description=sp['q'],sp['label'],sp['description']
		instances=get_ent_instances_or_subclass(q,type='instance')
		if instances==False:
			return dic,class_sub_class_dic

		if instances:

			Q=instances
			q=q
			type='instance'
			class_sub_class_dic=update_file(class_sub_class_dic,type,Q,q, flag='instances')
		if instances:
			item={'sub_classes':instances,'self':sp}
			dic[q]=item
		else:
			item={'sub_classes':None,'self':sp}
			#dic[q]=item
	return dic,class_sub_class_dic

def get_sub_classes_of_Qs(class_sub_class_dic,Qs):
	dic={}
	for sp in Qs:
		q,label,description=sp['q'],sp['label'],sp['description']


		sub_classes=get_ent_instances_or_subclass(q)
		if sub_classes==False:
			return dic,class_sub_class_dic
		if sub_classes:

			Q=sub_classes
			q=q
			type='class'
			class_sub_class_dic=update_file(class_sub_class_dic,type,Q,q, flag='subclass')
		if sub_classes:
			item={'sub_classes':sub_classes,'self':sp}
			dic[q]=item
		else:
			item={'sub_classes':None,'self':sp}
			#dic[q]=item
	return dic,class_sub_class_dic

def update_file(class_sub_class_dic,type,Q,q, flag='subclass'):
	if len(Q)>40:
		Q=Q[:40]



	temp='has_subclass' if flag=='subclass' else 'has_instances'

	for spi,sp in enumerate(Q):

		instance_q,instance_label,instance_description=sp['q'],sp['label'],sp['description']



		########
		if instance_q not in class_sub_class_dic['q_to_labels'].keys():
			item={'q':instance_q,'label':instance_label,'description':instance_description,'parent':q,'type':type}
			#print('item ',item)
			class_sub_class_dic['q_to_labels'][instance_q]=item

		######
		if q in class_sub_class_dic[temp].keys():
			class_sub_class_dic[temp][q].append(instance_q)
			class_sub_class_dic[temp][q]=list(set(class_sub_class_dic[temp][q]))
		else:
			class_sub_class_dic[temp][q]=[]
			class_sub_class_dic[temp][q].append(instance_q)
			class_sub_class_dic[temp][q]=list(set(class_sub_class_dic[temp][q]))
	return class_sub_class_dic


def Publish_Books(TOPICS_Category,q_to_labels,type='type1'):
    import itertools
    Books=[]
    TOPICS_Category_new={}
    for q in TOPICS_Category.keys():
        sub_topics=TOPICS_Category[q]
        if len(sub_topics)<1:
            continue
        TOPICS_Category_new[q]=sub_topics

    print('TOPICS_Category_new',len(TOPICS_Category_new.keys()))

    topic_population=list(TOPICS_Category_new.keys())

    #combinations = list(itertools.combinations(topic_population, 4))
    for i in range(len(TOPICS_Category_new.keys())):
        sampled = random.sample(topic_population, 5)
        books_biblo={}
        for k in range(4):
            books_biblo[k]={}

        sample_n=[]
        for i in range(len(sampled)):  # A loop to repeat the generation of colour
            sample_n.append(random.sample(sampled,1)[0])
        print('sample_n',sample_n)
        temp=[i for i in range(len(sample_n))]
        m=random.sample(temp,1)[0]
        analogy_topic=sample_n[m]
        flag=False
        q_temp_=None
        print('sample_n',sample_n)

        for j in range(4):

            topics={}
            f=False
            for ti,Q in enumerate(sample_n):
                Q_item=q_to_labels[Q] if Q in q_to_labels.keys() else 'empty'
                Q_label=Q_item['label'] if  Q_item!='empty' else 'empty'
                sub_topics=TOPICS_Category[Q]
                #print('sub_topics',len(sub_topics))
                print('analogy_topic,Q',analogy_topic,Q)

                if str(analogy_topic).lower()==str(Q).lower():
                    if flag==False:
                        f=True
                        
                        sub_topics_population=list(sub_topics.keys())
                        q_temp=random.sample(sub_topics_population,1)[0]
                        q_t_upper=q_temp
                        q_temp_=q_t_upper

               
                        
                        q_temp=sub_topics[q_temp]
        
                        flag= q_temp
                       #print('q_temp_',q_temp_)
                    
                    else:
                        f=True
                        q_temp=flag
                        q_t_upper=q_temp_#['q']

                else:
                    sub_topics_population=list(sub_topics.keys())
                    q_temp=random.sample(sub_topics_population,1)[0] 
                    q_t_upper=q_temp
                    print('sub_topics_population',len(sub_topics_population))
          
                    q_temp=sub_topics[q_temp]
   
                ####
                print('analogy_topic,q_t_upper',analogy_topic,q_t_upper)

                if q_t_upper==None:
                    print('q_temp_',q_temp_,len(flag))
                    exit()

                q_temp_population=[i for i in range(len(q_temp))]
                n=random.sample(q_temp_population,1)[0]
                topic_selected=q_temp[n]
                #print('topic_selected',topic_selected)
               

                q_t=topic_selected['q']
                q_label=topic_selected['label']
                #print('q_t',q_t)
                item={'super_class':Q,'super_class_label':Q_label,'topic_selected':topic_selected,\
                'q_t_upper':q_t_upper,'analogy_topic':analogy_topic,\
                'q':q_t,'label':q_label}
                #print('item',item)
                print('+++')
                topics[q_t]=item
            if f==False:
                print('exit')
                exit()
            books_biblo[j]=topics
        Books.append(books_biblo)

    #exit()
    for b in Books:
        print('*********************####################******************************')
        for t in b.keys():
            
            key=b[t].keys()
            print('###################')
            q_f={}
            q_f_upper={}
            q_f_p={}
            for k in key:
                print('--------')
                
                print('label',b[t][k]['label'])
                print('q',b[t][k]['q'])
                print('super_class',b[t][k]['super_class'])
                print('super_class_label',b[t][k]['super_class_label'])
                print('des',b[t][k]['topic_selected']['description'])
                q=b[t][k]['q']
                q_t_upper=b[t][k]['q_t_upper']
                analogy_topic=b[t][k]['analogy_topic']
                print('q_t_upper',q_t_upper)
                print('analogy_topic',analogy_topic)
                ###
                if q_t_upper not in q_f_upper.keys():
                    q_f_upper[q_t_upper]=1
                else:
                     q_f_upper[q_t_upper]= q_f_upper[q_t_upper]+1




                #####
                super_class=b[t][k]['super_class']
                if q not in q_f.keys():
                    q_f[k]=1
                else:
                     q_f[k]= q_f[k]+1

                if super_class not in q_f_p.keys():
                    q_f_p[super_class]=1
                else:
                     q_f_p[super_class]= q_f_p[super_class]+1
            print('q_f',q_f)
            print('q_f_upper',q_f_upper)
            print('q_f_p',q_f_p)



                # len_f={}
                # #if len(sub_topics)<1:
                #     #continue
                # print('**************',len(sub_topics))
                # for t in sub_topics:

                #     ln=len(t)
                #     if ln in len_f.keys():
                #         len_f[ln]=len_f[ln]+1
                #     else:
                #         len_f[ln]=1
                # if len(len_f.keys())!=0:
                #     print('len_f',len_f)



def Arrange_Topics():

    print('Arrange_Topics')
    file_name='abstraction.json'
    abstraction= json.load(open(file_name))['abstraction']
    q_to_labels=abstraction['q_to_labels']
    has_subclass=abstraction['has_subclass']
    has_instances=abstraction['has_instances']
    print('has_subclass',len(has_subclass))
    print('has_instances',len(has_instances))
    print('q_to_labels',len(q_to_labels))
    #exit()
    TOPICS_Category={}
    for Q in has_subclass.keys():
        subs=has_subclass[Q]
        if Q in q_to_labels.keys():
            Q_label=q_to_labels[Q]['label']
        else:
            Q_label='empty'
        #print('************************************************************')
        TOPICS={}
        for Qs in subs:
            if Qs not in has_instances.keys():
                continue
            sub_instances=has_instances[Qs]

            Qs_item=q_to_labels[Qs] if Qs in q_to_labels.keys() else 'empty'
            Qs_label=Qs_item['label'] if Qs_item!='empty' else 'none'
            #print('##############################')
            # print('Qs',Qs)
            # print('Q_label',Q_label)
            # print('Qs_label',Qs_label)
            # print('sub_instances',sub_instances)
            topics=[]
            for qs in sub_instances:
                if qs in q_to_labels.keys():
                    temp=q_to_labels[qs] if qs in q_to_labels.keys() else 'empty'
                    #print('temp',temp)
                    topics.append(temp)
            if Qs not in TOPICS.keys():
                TOPICS[Qs]=topics
        if Q not in TOPICS_Category.keys():
            TOPICS_Category[Q]=TOPICS


    Publish_Books(TOPICS_Category,q_to_labels)





def root_entites3():
    Arrange_Topics()
    print('test')
    exit()
    Books={}
    file_name='wikidata_entity_instances_subclass_dic.json'
    instances_subclass_dic= json.load(open(file_name))#['instances_dic']
    labels=instances_subclass_dic['labels']

    entities_context = json.load(open('unprocessed_data/wikipedia_distant//entities_context.json'))


    class_sub_class_dic={'q_to_labels':{},'has_subclass':{},'has_instances':{},'no_instance':[]}
    file_name='abstraction.json'
    class_sub_class_dic= json.load(open(file_name))['abstraction']
    Q_TO_L=class_sub_class_dic['q_to_labels']
    j=0
    for k in tqdm(labels.keys()):
        j=j+1

        q1_l=labels[k]
        q1=k
        if q1 in Q_TO_L.keys():
            continue
        time.sleep(1)
        # print('****************************************************************')
        # print('q1,q1l',q1,q1_l)
        if q1=='Q5':
        	continue
        super_class=get_ent_super_instances_superclass(q1)
        super_class_=[q['label'] for q in super_class] if super_class else 0
        # print('super_class',super_class_)
        # print('++++++++++++++++')
        if super_class==False:
        	continue
        if len(super_class)>10:
        	super_class=super_class[:10]
        if q1 not in class_sub_class_dic['q_to_labels'].keys():
        	item={'q':q1,'label':q1_l,'description':'','parent':[],'type':'class'}
        	#print('item ',item)



        
        visited=[]
        for spi,sp in enumerate(super_class):
            q,label,description=sp['q'],sp['label'],sp['description']

            if q in visited:
            	continue
            visited.append(q)
            ########
            if q not in class_sub_class_dic['q_to_labels'].keys():
            	item={'q':q,'label':label,'description':description,'parent':[],'type':'class'}
            	class_sub_class_dic['q_to_labels'][q]=item

            ######
            if q in class_sub_class_dic['has_subclass'].keys():
            	class_sub_class_dic['has_subclass'][q].append(q1)
            	class_sub_class_dic['has_subclass'][q]=list(set(class_sub_class_dic['has_subclass'][q]))
            else:
            	class_sub_class_dic['has_subclass'][q]=[]
            	class_sub_class_dic['has_subclass'][q].append(q1)
            	class_sub_class_dic['has_subclass'][q]=list(set(class_sub_class_dic['has_subclass'][q]))
            time.sleep(1)
            sub_classes=get_ent_instances_or_subclass(q)
            if sub_classes==False:
                continue
            sub_classes_=[q['label'] for q in sub_classes] if sub_classes else 0
            if sub_classes:
            	Q=sub_classes
            	q=q
            	type='class'
            	update_file(class_sub_class_dic,type,Q,q, flag='subclass')

            instances,class_sub_class_dic=get_instances_of_Qs(class_sub_class_dic,sub_classes)
            sub_classes,class_sub_class_dic=get_sub_classes_of_Qs(class_sub_class_dic,sub_classes)

            h_data={'abstraction':class_sub_class_dic}
            with open('abstraction.json', 'w') as fp:
            	json.dump(h_data, fp)
            # print('********************************')
            # print('instances',instances)
            # print('####')
            for k in instances:
                item=instances[k]
                self_prop=item['self']
                ins_data=item['sub_classes']


                #print('self_prop',self_prop)

                for ins in ins_data:
                    label,description=ins['label'],ins['description']
                    # print('ins label',label)
                    # print('ins description',description)
                    # print('*************')


                # print('********************************')
                # print('instances',instances)
                # print('####')
            for k in sub_classes:
                item=sub_classes[k]
                self_prop=item['self']
                ins_data=item['sub_classes']

                #print('self_prop',self_prop)
                for ins in ins_data:
                    label,description=ins['label'],ins['description']
                #     print('subcls label',label)
                #     print('subcls description',description)
                # print('*************')







##############################################


def leave_instances():
	import json
	entities_context = json.load(open('unprocessed_data/wikipedia_distant//entities_context.json'))

	kbID_instancces={}
	for key in entities_context.keys():
	    data=entities_context[key]
	    if len(data['instances'])==0:
	        continue
	    #print(data['instances'])
	    for inst in data['instances']:
	        kb_instance=inst['kbID']
	        kbID_instancces[kb_instance]=inst
	###

	subclass_instances={}
	for key in kbID_instancces.keys():
	    print('*******************************')
	    d=kbID_instancces[key]
	    lab=kbID_instancces[key]['label']
	    print(d)
	    time.sleep(1)
	    subclass_of=get_ent_subclass(key)
	    print('subclass_of',subclass_of)
	    if subclass_of==False :
	        continue
	    
	    for sb in subclass_of:

	        
	        if sb in subclass_instances:
	            item={'key':key,'label':lab}
	            subclass_instances[sb]['inst'].append(item)
	        else:
	            time.sleep(1)
	            l=get_instance_label(sb)
	            if l:
	                l=l[0]
	                subclass_instances[sb]={'label':l,'inst':[]}
	                item={'key':key,'label':lab}
	                subclass_instances[sb]['inst'].append(item)


def get_abstract_data():
    # # adapted from "https://github.com/minggg/squad/blob/master/setup.py"

    #args_ = get_setup_args()
    #print("args",args_)
    print('abstract')
    #root_entities()
    root_entites3()
