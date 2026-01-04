import pandas as pd 
import torch 
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
from collections import OrderedDict
from json import dumps
from tqdm import tqdm
from ujson import load as json_load
import logging
import os
import re
import string
import torch.utils.data as data
import tqdm
import numpy as np
import ujson as json
from collections import Counter
#from analogy_data import *
import time
from transformers import BertTokenizer, BertModel
import logging
import transformers
from transformers.models.bert.configuration_bert import BertConfig 
from transformers.models.bert.modeling_bert import BertEmbeddings 
import torch
###
from heinsen_routing import EfficientVectorRouting as Routing
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForImageClassification
import torch.nn as nn
import torch
import math
from torch.nn.functional import normalize
##
from transformers import XLNetTokenizer
from transformers import AutoTokenizer, LongT5Model
import copy
####


item={'sat':[],'ekar':None,'u2':None,'u4':None,'google':None,'bats':None,'sat':None}
Not_correctly_predicted={'data':item}

# file='essential_files/Not_correctly_predicted.json'
# Not_correctly_predicted= json.load(open(file))['data']

file_name='essential_files/id2prop.json'
id2kbid= json.load(open(file_name))['id2prop']

# with open('essential_files/wordanalogyrel_dic_sre.json') as f:
#     wordanalogyrel_dic = json.load(f)['rel_dic']


# wordanalogyrel_dic_rev = {y: x for x, y in wordanalogyrel_dic.items()}
# ##
# with open('essential_files/wordanalogyrel_dic.json') as f:
#     wordanalogyrel_dic_kb = json.load(f)['rel_dic']


# wordanalogyrel_dic_kb_rev = {y: x for x, y in wordanalogyrel_dic_kb.items()}

# SRE_Analogy={'wikidata':[],'retacred':[],'conll':[],'semeval':[]}


# file='essential_files/SRE_Analogy.json'
# SRE_Analogy= json.load(open(file))['SRE_Analogy']
###
# with open('essential_files/kbid_as_special_tokens.json') as f:
#     kbid_as_special_tokens = json.load(f)


import itertools

def update2(r,r_,data_dic,temp):
    if r_ in data_dic.keys():
        if r in data_dic[r_].keys():
            data_dic[r_][r].append(temp)
        else:
            data_dic[r_][r]=[]
            data_dic[r_][r].append(temp)
    else:
        data_dic[r_]={}
        data_dic[r_][r]=[]
        data_dic[r_][r].append(temp)
def batch_n(P,R_,batch_n_a,batch_n_p,batch_n_n,i,Rs,fl=False):
    old_p=None
    if len(R_.keys())<=2:
        return batch_n_a ,batch_n_p,batch_n_n,Rs
    if i==None:
        i=0
    for ri,r_ in enumerate(R_.keys()):
        if ri!=i and fl:
            continue
        else:
            p=P[r_]
            p_temp=[p[tr] for tr in p.keys()]

            
            p_temp = list(itertools.chain.from_iterable(p_temp))
            p_temp_new=[]

            random.shuffle(p_temp)
            a=p_temp[0]
            p=p_temp[1] if len(p_temp)>1 else p_temp[0]
            n=p_temp[0]

            p_temp=[p_temp[0],]
            

            j=0
            for p in p_temp:
          
                if len(Rs)!=0:
                    if r_ ==Rs[-1]:
                        break
                if fl:
                    Rs.append(r_)
                    batch_n_n.append(n)

                    return batch_n_a,batch_n_p,batch_n_n,Rs
                else:
                    
                    p=p+0.0000001
                    batch_n_a.append(a)
                    batch_n_p.append(p)
                    Rs.append(r_)
    
                ri=ri+1
                ri=ri%(len(R_.keys()))
                if fl:
                    return batch_n_a ,batch_n_p,batch_n_n,Rs
                batch_n_a,batch_n_p,batch_n_n,Rs=batch_n(P,R_,batch_n_a,batch_n_p,batch_n_n,ri,Rs,fl=True)
                old_p=p
                j=j+1

    min_l=min([len(batch_n_a),len(batch_n_p),len(batch_n_n)])

    return batch_n_a[:min_l],batch_n_p[:min_l],batch_n_n[:min_l],Rs
    



def update2_rc(r,r_,data_dic,temp):
    if r_ in data_dic.keys():
        if r in data_dic[r_].keys():
            data_dic[r_][r].append(temp)
        else:
            data_dic[r_][r]=[]
            data_dic[r_][r].append(temp)
    else:
        data_dic[r_]={}
        data_dic[r_][r]=[]
        data_dic[r_][r].append(temp)
def batch_n_rc(A,P,N,R_,batch_n_a,batch_n_p,batch_n_n,i,Rs,fl=False):
    if i==None:
        i=0
    for ri,r_ in enumerate(R_.keys()):
        if ri!=i and fl:
            continue
        else:
            a_=A[r_]
            p_=P[r_]
            n_=N[r_]
            anchor_temp=[a_[tr] for tr in a_.keys()]
            positive_temp=[p_[tr] for tr in p_.keys()]
            negative_temp=[n_[tr] for tr in n_.keys()]
            anchor_temp = list(itertools.chain.from_iterable(anchor_temp))
            positive_temp = list(itertools.chain.from_iterable(positive_temp))
            negative_temp = list(itertools.chain.from_iterable(negative_temp))
            random.shuffle(anchor_temp)
            random.shuffle(positive_temp)
            random.shuffle(negative_temp)
  
            for a,p,n in zip(anchor_temp,positive_temp,negative_temp):
                if len(Rs)!=0:
                    if r_ ==Rs[-1]:
                        break
                if fl:
                    ##

                    batch_n_n.append(p)
                    return batch_n_a,batch_n_p,batch_n_n,Rs
                else:
                    batch_n_a.append(a)
                    batch_n_p.append(p)
                    
                Rs.append(r_)
                ri=ri+1
                ri=ri%(len(R_.keys()))

                if fl:
                    return batch_n_a,batch_n_p,batch_n_n,Rs
                batch_n_a,batch_n_p,batch_n_n,Rs=batch_n_rc(A,P,N,R_,batch_n_a,batch_n_p,batch_n_n,ri,Rs,fl=True)
                break
    return batch_n_a,batch_n_p,batch_n_n,Rs

def get_permuation_positive(src,target,src2,target2):

        item1={'w1':src  ,'w2':target ,'w3':src2 ,'w4':target2 }
        #item1_negative={'test':test,'w1':src  ,'w2':target ,'w3':target2  ,'w4':src2 ,'name':dname,'type':'negative','eval_d':eval_d}

        item1_negative={'w1':src  ,'w2':src2 ,'w3':target   ,'w4':target2 }
        item1_negative_={'w1':src  ,'w2':target2 ,'w3':target  ,'w4':src2 }

        item1_={'w1':src  ,'w2':src2 ,'w3':target  ,'w4':target2 }

    
        item2={'w1':target ,'w2':src ,'w3':target2,'w4':src2 }

        #item2_negative={'test':test,'w1':target ,'w2':src ,'w3':src2 ,'w4':target2 ,'name':dname,'type':'negative','eval_d':eval_d}

        item3={'w1':src2  ,'w2':target2 ,'w3':src ,'w4':target }
        
        #item3_negative={'test':test,'w1':src2  ,'w2':target2 ,'w3':target  ,'w4':src ,'name':dname,'type':'negative','eval_d':eval_d}

        item4={'w1':target2 ,'w2':src2 ,'w3':target,'w4':src }

        data={'positive':[item1,item1_negative,item1_,item2,item3,item4],'negative':[]}
        #[item1,item1_negative,item1_,item2,item3,item4]#
        #[item1_negative,item2_negative,item3_negative,item4_negative]
        return data

def set_tokenizer(data_selected,tokenizer,tokenizer_special_dic='default'):

    special_tokens_dict = {'additional_special_tokens': ['[mask]','[p0]','[e11]','[e12]','[e21]','[e22]','[entsep]','[es]','[en]', '[CLS]','[SEP]']}
    if tokenizer_special_dic=='re' and data_selected!='wordanalogy':
        data_selected='wikidata'
        file='essential_files/'+str(data_selected)+'rel_dic.json'
        kbid_as_special_tokens=[]
        with open(file) as f:
            data = json.load(f)['rel_dic']
            for t in data.keys():
                #print('t',t)
                temp=data[t]['kbID'] 
                temp='['+str(temp)+']'
                kbid_as_special_tokens.append(temp)
        for key in kbid_as_special_tokens:
            key=key.lower()
            special_tokens_dict['additional_special_tokens'].append(str(key).lower())
        
        file='essential_files/all_relation_dic.json'
        file='essential_files/all_relation_dic.json'

        with open(file) as f:
            data=json.load(f)['data']
            kbid_as_special_tokens1 = data['relation']
            kbid_as_special_tokens2 = data['category']

        for key in kbid_as_special_tokens1.keys():
            t=kbid_as_special_tokens1[key]

            special_tokens_dict['additional_special_tokens'].append(str(t).lower())
        for key in kbid_as_special_tokens2.keys():
            t=kbid_as_special_tokens2[key]
            special_tokens_dict['additional_special_tokens'].append(str(t).lower())



        tokenizer.add_special_tokens(special_tokens_dict)

    elif tokenizer_special_dic=='semeval_2012_re':
        file='essential_files/all_relation_dic.json'
        file='essential_files/all_relation_dic.json'

        with open(file) as f:
            data=json.load(f)['data']
            kbid_as_special_tokens1 = data['relation']
            kbid_as_special_tokens2 = data['category']

        for key in kbid_as_special_tokens1.keys():
            t=kbid_as_special_tokens1[key]

            special_tokens_dict['additional_special_tokens'].append(str(t).lower())
        for key in kbid_as_special_tokens2.keys():
            t=kbid_as_special_tokens2[key]
            special_tokens_dict['additional_special_tokens'].append(str(t).lower())
        ###
        file='essential_files/'+'wikidata'+'rel_dic.json'
        kbid_as_special_tokens=[]
        with open(file) as f:
            data = json.load(f)['rel_dic']
            for t in data.keys():
                #print('t',t)
                temp=data[t]['kbID'] 
                temp='['+str(temp)+']'
                kbid_as_special_tokens.append(temp)
        for key in kbid_as_special_tokens:
            key=key.lower()
            special_tokens_dict['additional_special_tokens'].append(str(key).lower())


        tokenizer.add_special_tokens(special_tokens_dict)
    else: 
        file='essential_files/all_relation_dic.json'
        file='essential_files/all_relation_dic.json'

        with open(file) as f:
            data=json.load(f)['data']
            kbid_as_special_tokens1 = data['relation']
            kbid_as_special_tokens2 = data['category']

        for key in kbid_as_special_tokens1.keys():
            t=kbid_as_special_tokens1[key]

            special_tokens_dict['additional_special_tokens'].append(str(t).lower())
        for key in kbid_as_special_tokens2.keys():
            t=kbid_as_special_tokens2[key]
            special_tokens_dict['additional_special_tokens'].append(str(t).lower())
        ###
        file='essential_files/'+'wikidata'+'rel_dic.json'
        kbid_as_special_tokens=[]
        with open(file) as f:
            data = json.load(f)['rel_dic']
            for t in data.keys():
                #print('t',t)
                temp=data[t]['kbID'] 
                temp='['+str(temp)+']'
                kbid_as_special_tokens.append(temp)
        for key in kbid_as_special_tokens:
            key=key.lower()
            special_tokens_dict['additional_special_tokens'].append(str(key).lower())


        tokenizer.add_special_tokens(special_tokens_dict)



    return tokenizer


def _get_pretrained_transformer3(data_selected,modality,tokenizer_special_dic='default'):

    if modality=='bert-large-uncased':
        config = { 'name': 'bert-large-uncased','d_depth': 25, 'chunk_len': 512,  'emb':1024}
        tokenizer = AutoTokenizer.from_pretrained(config['name'])
        transformer = BertModel.from_pretrained(config['name'], output_hidden_states=True)
        print('modality',modality)

    elif modality=='bert-base-uncased':
        config = { 'name': 'bert-base-uncased','d_depth': 13, 'chunk_len': 768,  'emb':768}
        tokenizer = AutoTokenizer.from_pretrained(config['name'])
        transformer = BertModel.from_pretrained(config['name'], output_hidden_states=True)
    elif modality=='roberta-large':
        config = { 'name': 'roberta-large', 'revision': '5069d8a', 'd_depth': 25, 'chunk_len': 512, 'emb':1024 }
        #tokenizer = AutoTokenizer.from_pretrained(config['name'], revision=config['revision'])
        #transformer = AutoModelForMaskedLM.from_pretrained(config['name'], output_hidden_states=True, revision=config['revision'])
        config = { 'name': 'roberta-large', 'revision': '5069d8a', 'd_depth': 25, 'chunk_len': 512, }
        #tokenizer = AutoTokenizer.from_pretrained(config['name'], revision=config['revision'])
        #transformer = AutoModelForMaskedLM.from_pretrained(config['name'], output_hidden_states=True, revision=config['revision'])

        #transformer.save_pretrained("pretrained/roberta-large", from_pt=True) 
        try:
            transformer=AutoModelForMaskedLM.from_pretrained("pretrained/roberta-large")
            tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")
        except:
            tokenizer = AutoTokenizer.from_pretrained(config['name'], revision=config['revision'])
            transformer = AutoModelForMaskedLM.from_pretrained(config['name'], output_hidden_states=True, revision=config['revision'])






    elif modality=='roberta-base':
        config = { 'name': 'roberta-base', 'd_depth': 13, 'chunk_len': 512,  'emb':768}
        tokenizer = AutoTokenizer.from_pretrained(config['name'])
        transformer = AutoModelForMaskedLM.from_pretrained(config['name'], output_hidden_states=True)


    #######
    # t_1024=['t5-large','bert-large-uncased','roberta-large','opt','prophetnet']
    # t_768=['gpt2','bert_base_uncased','roberta-base','flaxopt']
    # t_512=['t5-small',]

    elif modality=='t5-small':
        from transformers import T5Model
        config = { 'name': 't5-small',  'd_depth': 1, 'chunk_len': 512, 'emb':512 }
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        transformer = T5Model.from_pretrained("google-t5/t5-small")


    elif modality=='t5-large':
        from transformers import T5Model
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")
        config = { 'name': 't5-large',  'd_depth': 1, 'chunk_len': 512,  'emb':1024}
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")
        transformer = T5Model.from_pretrained("google-t5/t5-large")

    elif modality=='gpt2':
        from transformers import GPT2Tokenizer
        from transformers import GPT2Tokenizer, GPT2Model
        config = { 'name': 'gpt2',  'd_depth': 13, 'chunk_len': 512,  'emb':768}
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        transformer = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)


    elif modality=='opt':
        from transformers import OPTModel
        config = { 'name': 'opt',  'd_depth': 24, 'chunk_len': 512,  'emb':1024}
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        transformer = OPTModel.from_pretrained("facebook/opt-350m", output_hidden_states=True)

    elif modality=='gpt':
        from transformers import  FlaxOPTModel

        config = { 'name': 'gpt',  'd_depth': 13, 'chunk_len': 512, 'emb':768}

        from transformers import OpenAIGPTLMHeadModel

        tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
        transformer = OpenAIGPTLMHeadModel.from_pretrained("openai-community/openai-gpt")

    elif modality=='prophetnet':
        from transformers import ProphetNetEncoder
        config = { 'name': 'prophetnet',  'd_depth': 13, 'chunk_len': 512, 'emb':1024}
        tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
        transformer = ProphetNetEncoder.from_pretrained("patrickvonplaten/prophetnet-large-uncased-standalone",output_hidden_states=True)


    tokenizer=set_tokenizer(data_selected,tokenizer,tokenizer_special_dic=tokenizer_special_dic)
    return config, transformer,tokenizer








#####

def torch_from_json(path, dtype=torch.float32):
    """
    # 
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor

def rc_data(args):

    torch.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    print('data_type',args.data_type)

    # Get embeddings
    data_type=args.data_type
    if data_type=='tacred':
        args.train_record_file='./data/data_tacred/'+args.train_record_file
        args.dev_record_file='./data/data_tacred/'+args.dev_record_file
        args.test_record_file='./data/data_tacred/'+args.test_record_file

        args.dev_rev_record_file='./data/data_tacred/'+args.dev_rev_record_file
        args.test_rev_record_file='./data/data_tacred/'+args.test_rev_record_file

    if data_type=='retacred':
        args.train_record_file='./data/data_retacred/'+args.train_record_file
        args.dev_record_file='./data/data_retacred/'+args.dev_record_file
        args.test_record_file='./data/data_retacred/'+args.test_record_file
    elif data_type=='nyt':
        args.train_record_file='./data/data_nyt/'+args.train_record_file
        args.dev_record_file='./data/data_nyt/'+args.dev_record_file
    elif data_type=='wikidata':
        args.train_record_file='./data/data_wikidata/'+args.train_record_file
        args.dev_record_file='./data/data_wikidata/'+args.dev_record_file

    elif data_type=='conll':
        args.train_record_file='./data/data_conll/'+args.train_record_file
        args.dev_record_file='./data/data_conll/'+args.dev_record_file
    elif data_type=='semeval':
        args.train_record_file='./data/data_semeval/'+args.train_record_file
        args.dev_record_file='./data/data_semeval/'+args.dev_record_file

    elif data_type=='wordanalogy':
        args.train_record_file='./data/data_wordanalogy/'+args.train_record_file
        args.dev_record_file='./data/data_wordanalogy/'+args.dev_record_file

        #args.dev_record_file=args.train_record_file

    elif data_type=='BLESS':
        args.train_record_file='./data/data_BLESS/'+args.train_record_file
        args.dev_record_file='./data/data_BLESS/'+args.dev_record_file

    elif data_type=='EVALution':
        args.train_record_file='./data/data_EVALution/'+args.train_record_file
        args.dev_record_file='./data/data_EVALution/'+args.dev_record_file


    elif data_type=='CogALexV':
        args.train_record_file='./data/data_CogALexV/'+args.train_record_file
        args.dev_record_file='./data/data_CogALexV/'+args.dev_record_file


    elif data_type=='ROOT09':
        args.train_record_file='./data/data_ROOT09/'+args.train_record_file
        args.dev_record_file='./data/data_ROOT09/'+args.dev_record_file


    elif data_type=='KandH_plus_N':
        args.train_record_file='./data/data_KandH_plus_N/'+args.train_record_file
        args.dev_record_file='./data/data_KandH_plus_N/'+args.dev_record_file

    elif data_type=='semeval_2012':

        args.train_record_file='./data/data_semeval_2012/'+args.train_record_file
        args.dev_record_file='./data/data_semeval_2012/'+args.dev_record_file


    elif data_type=='conceptqa':
        args.train_record_file='./data/data_conceptqa/'+args.train_record_file
        args.dev_record_file='./data/data_conceptqa/'+args.dev_record_file

    #not in ['semeval','wordanalogy','BLESS','EVALution','CogALexV','ROOT09','KandH_plus_N']


    train_dataset = RC_Data(args,args.train_record_file,'train')
    print('len(train_dataset)',len(train_dataset))

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    print('devloader')
    dev_dataset = RC_Data(args,args.dev_record_file,'eval')
    print('len(dev_dataset)',len(dev_dataset))


    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                     collate_fn=collate_fn)
    if data_type=='tacred' or data_type=='retacred':

        test_dataset = RC_Data(args,args.test_record_file,'eval')
        test_loader = data.DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                     collate_fn=collate_fn)
        if data_type=='tacred':
            dev_rev_dataset = RC_Data(args,args.dev_rev_record_file,'eval')
            dev_rev_loader = data.DataLoader(dev_rev_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                         collate_fn=collate_fn)

            test_rev_dataset = RC_Data(args,args.test_rev_record_file,'eval')
            test_rev_loader = data.DataLoader(test_rev_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                         collate_fn=collate_fn)

            return train_loader,(dev_loader,test_loader,dev_rev_loader,test_rev_loader)
        else:
            return train_loader,(dev_loader,test_loader)




    return train_loader,(dev_loader,)






class Args_dic(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'


def first_samples_only(data,filter_type):
    data_keys=list(data.keys())
    data_L=[data[k] for k in data_keys]
    Data_new={}
    for key in data_keys:
        Data_new[key]=[]
    random_l=[]
    random_flag=False
    for k in data_keys:
        old_instance_id=-1
        old_d=None
        j_n=0
        temp_L=[]
        # negative=2 , neutral =1, positive=0
        for (not_positive_not_negaative,d) in tqdm(zip(data['not_positive_not_negaative'],data[k])):
            if not_positive_not_negaative== filter_type:
                #print('test')
        
                continue
                #pass

            #print('d',d)
            Data_new[k].append(d)
    for k in data_keys:
       
            Data_new[k]=np.array(Data_new[k])


    return Data_new



def subset_samples_only(data, train_or_eval,args):

    types_to_keep=args.wordanalogy_train_data if train_or_eval=='train' else args.wordanalogy_test_data

    file_name='essential_files/word_analogy_types_dic.json'
    word_analogy_types_dic= json.load(open(file_name))['word_analogy_types_dic']
    word_analogy_types_dic_rev = {y: x for x, y in word_analogy_types_dic.items()}


    data_keys=list(data.keys())
    data_L=[data[k] for k in data_keys]
    Data_new={}
    for key in data_keys:
        Data_new[key]=[]
    random_l=[]
    random_flag=False
    for k in data_keys:
        old_instance_id=-1
        old_d=None
        j_n=0
        temp_L=[]
        for (y,d) in zip(data['y'],data[k]):

            if word_analogy_types_dic_rev[y] not in types_to_keep:
                continue
                #pass
            else:
                Data_new[k].append(d)
    for k in data_keys:
            Data_new[k]=np.array(Data_new[k])
    return Data_new 


def exp_three(data,train_or_eval,args):


    if args.person_person=='all':
        types_to_keep=[0,1]
    elif args.person_person=='person-person':
        types_to_keep=[1,]
    elif  args.person_person=='person-person*':
        if  train_or_eval=='train' :
            types_to_keep=[0,1]
        else:
             types_to_keep=[1,]


    data_keys=list(data.keys())
    data_L=[data[k] for k in data_keys]
    Data_new={}
    for key in data_keys:
        Data_new[key]=[]
    random_l=[]
    random_flag=False
    for k in data_keys:
        old_instance_id=-1
        old_d=None
        j_n=0
        temp_L=[]
        for (y,d) in tqdm(zip(data['subset'],data[k])):


            if y not in types_to_keep:
                continue
                #pass
            else:
                Data_new[k].append(d)
    for k in data_keys:
            Data_new[k]=np.array(Data_new[k])
    return Data_new 

class RC_Data(data.Dataset):

    def __init__(self,args, data_path,train_or_eval):
        print(data_path)
        super(RC_Data, self).__init__()

        dataset = np.load(data_path)
        print('train_or_eval',train_or_eval)

        if train_or_eval=='train' and args.filter_type!=None and args.data_type=='wordanalogy':
            print('1')

            dataset=first_samples_only(dataset,args.filter_type)
        if  args.data_type=='wordanalogy' :
            print('2')

            dataset=subset_samples_only(dataset, train_or_eval,args)
        elif args.experiment_no=='three' and args.data_type=='retacred':
            dataset=exp_three(dataset, train_or_eval,args)
            print('3')


        self.ids = dataset['id']
        self.args=args




        if args.data_type not in ['wordanalogy','semeval_2012','conceptqa'] :
            self.sentence_ents_flagged_tokens=torch.from_numpy(dataset['sentence_ents_flagged_tokens']).long()
            self.sentence_tokens=torch.from_numpy(dataset['sentence_tokens']).long()
            self.sentence_masked_flagged_tokens=torch.from_numpy(dataset['sentence_masked_flagged_tokens']).long()

            self.sentence_ents_flagged_tokens_masks=torch.from_numpy(dataset['sentence_ents_flagged_tokens_masks']).long()
            self.sentence_tokens_masks=torch.from_numpy(dataset['sentence_tokens_masks']).long()
            self.sentence_masked_flagged_tokens_masks=torch.from_numpy(dataset['sentence_masked_flagged_tokens_masks']).long()
            if args.data_type not in ['semeval','wordanalogy','BLESS','EVALution','CogALexV','ROOT09','KandH_plus_N','semeval_2012']:
                self.sentecne_ents_abstracted_flagged_tokens=torch.from_numpy(dataset['sentecne_ents_abstracted_flagged_tokens']).long()
                self.sentecne_ents_abstracted_flagged_tokens_masks=torch.from_numpy(dataset['sentecne_ents_abstracted_flagged_tokens_masks']).long()

                self.EntsAbst_flagged_tokens=torch.from_numpy(dataset['EntsAbst_flagged_tokens']).long()
                self.EntsAbst_flagged_plus_rel_tokens=torch.from_numpy(dataset['EntsAbst_flagged_plus_rel_tokens']).long()
                self.abstracted_ents_flagged_tokens=torch.from_numpy(dataset['abstracted_ents_flagged_tokens']).long()
                self.abstracted_ents_flagged_plus_rel_tokens=torch.from_numpy(dataset['abstracted_ents_flagged_plus_rel_tokens']).long()

                self.sentecne_entabs_flagged_tokens=torch.from_numpy(dataset['sentecne_entabs_flagged_tokens']).long()
                
                self.sentecne_entabs_flagged_tokens_masks=torch.from_numpy(dataset['sentecne_entabs_flagged_tokens_masks']).long()


                self.EntsAbst_flagged_tokens_masks=torch.from_numpy(dataset['EntsAbst_flagged_tokens_masks']).long()
                self.EntsAbst_flagged_plus_rel_tokens_masks=torch.from_numpy(dataset['EntsAbst_flagged_plus_rel_tokens_masks']).long()
                self.abstracted_ents_flagged_tokens_masks=torch.from_numpy(dataset['abstracted_ents_flagged_tokens_masks']).long()
                self.abstracted_ents_flagged_plus_rel_tokens_masks=torch.from_numpy(dataset['abstracted_ents_flagged_plus_rel_tokens_masks']).long()
                self.ent_abstract_Len_Target=torch.from_numpy(dataset['ent_abstract_Len_Target']).long()


            self.ents_flagged_tokens=torch.from_numpy(dataset['ents_flagged_tokens']).long()
            self.ents_flagged_plus_rel_tokens=torch.from_numpy(dataset['ents_flagged_plus_rel_tokens']).long()

            self.ents_flagged_tokens_masks=torch.from_numpy(dataset['ents_flagged_tokens_masks']).long()
            self.ents_flagged_plus_rel_tokens_masks=torch.from_numpy(dataset['ents_flagged_plus_rel_tokens_masks']).long()

            self.y=torch.from_numpy(dataset['y']).long()
            self.y0=torch.from_numpy(dataset['y0']).long()
            self.person_person=torch.from_numpy(dataset['subset']).long()

            self.Len_Target=torch.from_numpy(dataset['Len_Target']).long()
            self.abstract_Len_Target=torch.from_numpy(dataset['abstract_Len_Target']).long()
            print('test 3')
            
            
 

        elif args.data_type=='wordanalogy':

            self.a=torch.from_numpy(dataset['a']).long()
            self.b=torch.from_numpy(dataset['b']).long()
            self.c=torch.from_numpy(dataset['c']).long()
            self.d=torch.from_numpy(dataset['d']).long()
            #r_label
            self.r_label=torch.from_numpy(dataset['r_label']).long()

            self.ab=torch.from_numpy(dataset['ab']).long()
            self.cd=torch.from_numpy(dataset['cd']).long()
            self.abcd=torch.from_numpy(dataset['abcd']).long()

            self.a_masks=torch.from_numpy(dataset['a_masks']).long()
            self.b_masks=torch.from_numpy(dataset['b_masks']).long()
            self.c_masks=torch.from_numpy(dataset['c_masks']).long()
            self.d_masks=torch.from_numpy(dataset['d_masks']).long()
            self.r_label_masks=torch.from_numpy(dataset['r_label_masks']).long()

            self.ab_masks=torch.from_numpy(dataset['ab_masks']).long()
            self.cd_masks=torch.from_numpy(dataset['cd_masks']).long()
            self.abcd_masks=torch.from_numpy(dataset['abcd_masks']).long()

            self.y=torch.from_numpy(dataset['y']).long()
            self.similarity=torch.from_numpy(dataset['similarity']).long()
            self.r=torch.from_numpy(dataset['r']).long()
            self.r1=torch.from_numpy(dataset['r1']).long()
            self.y0=torch.from_numpy(dataset['y0']).long()
            self.r=torch.from_numpy(dataset['r']).long()
        
            self.not_positive_not_negaative=torch.from_numpy(dataset['not_positive_not_negaative']).long()

        elif args.data_type=='semeval_2012':
            self.s=torch.from_numpy(dataset['s_tokens']).long()
            self.s_masks=torch.from_numpy(dataset['s_masks']).long()
            self.s_plus_rel=torch.from_numpy(dataset['s_plus_rel_tokens']).long()

            self.a=torch.from_numpy(dataset['a_tokens']).long()
            self.a_masks=torch.from_numpy(dataset['a_masks']).long()
            self.a_plus_rel=torch.from_numpy(dataset['a_plus_rel_tokens']).long()

            self.n=torch.from_numpy(dataset['n_tokens']).long()
            self.n_masks=torch.from_numpy(dataset['n_masks']).long()
            self.n_plus_rel=torch.from_numpy(dataset['n_plus_rel_tokens']).long()

            self.equ=torch.from_numpy(dataset['equ_tokens']).long()
            self.equ_masks=torch.from_numpy(dataset['equ_masks']).long()
            self.equ_plus_rel=torch.from_numpy(dataset['equ_plus_rel_tokens']).long()

            self.s_r=torch.from_numpy(dataset['s_r']).long()
            self.s_r_=torch.from_numpy(dataset['s_r_']).long()


            self.a_r=torch.from_numpy(dataset['a_r']).long()
            self.a_r_=torch.from_numpy(dataset['a_r_']).long()

            self.n_r=torch.from_numpy(dataset['n_r']).long()
            self.n_r_=torch.from_numpy(dataset['n_r_']).long()

            self.equ_r=torch.from_numpy(dataset['equ_r']).long()
            self.equ_r_=torch.from_numpy(dataset['equ_r_']).long()


            self.s_len=torch.from_numpy(dataset['s_len']).long()
            self.a_len=torch.from_numpy(dataset['a_len']).long()
            self.n_len=torch.from_numpy(dataset['n_len']).long()
            self.equ_len=torch.from_numpy(dataset['equ_len']).long()

        elif args.data_type=='conceptqa':
            self.q=torch.from_numpy(dataset['q']).long()
            self.answer=torch.from_numpy(dataset['answer']).long()
            self.a=torch.from_numpy(dataset['a']).long()
            self.b=torch.from_numpy(dataset['b']).long()
            self.c=torch.from_numpy(dataset['c']).long()
            self.d=torch.from_numpy(dataset['d']).long()
            self.e=torch.from_numpy(dataset['e']).long()
            self.y=torch.from_numpy(dataset['y']).long()


        self.valid_idxs=[i for i in range(len(self.ids ))]



      


    def __getitem__(self, idx):
        #print('idx',idx)
        idx = self.valid_idxs[idx]
        #print('idx',idx)

        if self.args.data_type not in ['semeval','wordanalogy','BLESS','EVALution','CogALexV','ROOT09','KandH_plus_N','semeval_2012','conceptqa']:
            example={
            'sentence_ents_flagged_tokens':self.sentence_ents_flagged_tokens[idx],\
            'sentence_tokens':self.sentence_tokens[idx],\
            'sentence_masked_flagged_tokens':self.sentence_masked_flagged_tokens[idx],\
            'sentecne_ents_abstracted_flagged_tokens':self.sentecne_ents_abstracted_flagged_tokens[idx],\
            
            'sentecne_ents_abstracted_flagged_tokens_masks':self.sentecne_ents_abstracted_flagged_tokens_masks[idx],\
            'sentence_ents_flagged_tokens_masks':self.sentence_ents_flagged_tokens_masks[idx],\
            'sentence_tokens_masks':self.sentence_tokens_masks[idx],\
            'sentence_masked_flagged_tokens_masks':self.sentence_masked_flagged_tokens_masks[idx],\
            'sentecne_entabs_flagged_tokens':self.sentecne_entabs_flagged_tokens[idx],\
            'sentecne_entabs_flagged_tokens_masks':self.sentecne_entabs_flagged_tokens_masks[idx],\
            
      
            'EntsAbst_flagged_tokens':self.EntsAbst_flagged_tokens[idx],\
            'EntsAbst_flagged_plus_rel_tokens':self.EntsAbst_flagged_plus_rel_tokens[idx],\
            'abstracted_ents_flagged_tokens':self.abstracted_ents_flagged_tokens[idx],\
            'abstracted_ents_flagged_plus_rel_tokens':self.abstracted_ents_flagged_plus_rel_tokens[idx],\
           
            'ents_flagged_tokens':self.ents_flagged_tokens[idx],\
            'ents_flagged_plus_rel_tokens':self.ents_flagged_plus_rel_tokens[idx],\

            'EntsAbst_flagged_tokens_masks':self.EntsAbst_flagged_tokens_masks[idx],\
            'EntsAbst_flagged_plus_rel_tokens_masks':self.EntsAbst_flagged_plus_rel_tokens_masks[idx],\
            'abstracted_ents_flagged_tokens_masks':self.abstracted_ents_flagged_tokens_masks[idx],\
            'abstracted_ents_flagged_plus_rel_tokens_masks':self.abstracted_ents_flagged_plus_rel_tokens_masks[idx],\
           
            'ents_flagged_tokens_masks':self.ents_flagged_tokens_masks[idx],\
            'ents_flagged_plus_rel_tokens_masks':self.ents_flagged_plus_rel_tokens_masks[idx],\



            'y':self.y[idx],\
            'y0':self.y0[idx],\
            'ids':self.ids[idx],\
            'Len_Target':self.Len_Target[idx],\
            'abstract_Len_Target':self.abstract_Len_Target[idx],\
            'ent_abstract_Len_Target':self.ent_abstract_Len_Target[idx],\
            }
 

        elif self.args.data_type=='wordanalogy':
            example={
            'a':self.a[idx],\
            'b':self.b[idx],\
            'c':self.c[idx],\
            'd':self.d[idx],\
            'r':self.r[idx],\
            'r1':self.r1[idx],\
            'r_label':self.r_label[idx],\

            'ab':self.ab[idx],\
            'cd':self.cd[idx],\
            'abcd':self.abcd[idx],\

            'a_masks':self.a_masks[idx],\
            'b_masks':self.b_masks[idx],\
            'c_masks':self.c_masks[idx],\
            'd_masks':self.d_masks[idx],\
            'r_label_masks':self.r_label_masks[idx],\


            'ab_masks':self.ab_masks[idx],\
            'cd_masks':self.cd_masks[idx],\
            'abcd_masks':self.abcd_masks[idx],\

            'y':self.y[idx],\
            'similarity':self.similarity[idx],\
            'y0':self.y0[idx],\
            'ids':self.ids[idx],\
            }
        elif self.args.data_type in ['BLESS','EVALution','CogALexV','semeval','ROOT09','KandH_plus_N']:
            example={
            'sentence_ents_flagged_tokens':self.sentence_ents_flagged_tokens[idx],\
            'sentence_ents_flagged_tokens_masks':self.sentence_ents_flagged_tokens_masks[idx],\
            'sentence_tokens':self.sentence_tokens[idx],\
            'sentence_masked_flagged_tokens':self.sentence_masked_flagged_tokens[idx],\

  
            'ents_flagged_tokens':self.ents_flagged_tokens[idx],\
            'ents_flagged_plus_rel_tokens':self.ents_flagged_plus_rel_tokens[idx],\

            'ents_flagged_tokens_masks':self.ents_flagged_tokens_masks[idx],\
            'ents_flagged_plus_rel_tokens_masks':self.ents_flagged_plus_rel_tokens_masks[idx],\



            'y':self.y[idx],\
            'y0':self.y0[idx],\
            'ids':self.ids[idx],\
            'Len_Target':self.Len_Target[idx],\

            }
        elif self.args.data_type=='semeval_2012':

            example={
            's':self.s[idx],\
            's_masks':self.s_masks[idx],\
            's_plus_rel':self.s_plus_rel[idx],\


            'a':self.a[idx],\
            'a_masks':self.a_masks[idx],\
            'a_plus_rel':self.a_plus_rel[idx],\

            'n':self.n[idx],\
            'n_masks':self.n_masks[idx],\
            'n_plus_rel':self.n_plus_rel[idx],\

            'equ':self.n[idx],\
            'equ_masks':self.n_masks[idx],\
            'equ_plus_rel':self.n_plus_rel[idx],\


            's_r':self.s_r[idx],\
            's_r_':self.s_r_[idx],\


            'a_r':self.a_r[idx],\
            'a_r_':self.a_r_[idx],\

            'n_r':self.n_r[idx],\
            'n_r_':self.n_r_[idx],\

            'equ_r':self.equ_r[idx],\
            'equ_r_':self.equ_r_[idx],\


            'ids':self.ids[idx],\
   
   




            's_len':self.s_len[idx],\
            'a_len':self.a_len[idx],\
            'n_len':self.n_len[idx],\
            'equ_len':self.equ_len[idx],\

            }

        elif self.args.data_type=='conceptqa':
            example={
            'q':self.q[idx],\
            'answer':self.answer[idx],\
            'a':self.a[idx],\
            'b':self.b[idx],\
            'c':self.c[idx],\
            'd':self.d[idx],\
            'e':self.e[idx],\
            'y':self.y[idx],\
            'ids':self.ids[idx],\




            }







        return example

    def __len__(self):
        return len(self.valid_idxs)


def collate_fn(Examples):
    """
    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """
    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype)

    def merge_1d(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded

    def merge_2d(matrices, dtype=torch.int64, pad_value=0):
        heights = [(m.sum(1) != pad_value).sum() for m in matrices]
        widths = [(m.sum(0) != pad_value).sum() for m in matrices]
        padded = torch.zeros(len(matrices), max(heights), max(widths), dtype=dtype)
        for i, seq in enumerate(matrices):
            height, width = heights[i], widths[i]
            padded[i, :height, :width] = seq[:height, :width]
        return padded

    DATA={}
    d1_temp=['r','r1','y0','y','abstract_Len_Target','Len_Target','ent_abstract_Len_Target','similarity',\
    's_r','s_r_',\
    'a_r','a_r_',\
    'n_r','n_r_',\
    'equ_r','equ_r_',\
    's_len','a_len','n_len','equ_len'
    ]
    for key in Examples[0].keys():
        data=[]
        for examples in  Examples:
            temp=examples[key]
            data.append(temp)
        if key in d1_temp:
            #print('key',key)
            data=torch.stack(data)
            #print(data.shape)
            data = merge_0d(data)
        elif key =='ids':
            data = data

        else:


            data=torch.stack(data)
            #print(data.shape)
            data = merge_1d(data)
        DATA[key]=data
    return DATA




def print_similarity_and_dist(args,Similarity_in_Hidden_Layers):
    import os

    # Specify the file path
    file_path = 'essential_files/head_tail_sim_dic.json'

    # Check if the file exists
    if os.path.exists(file_path):
        head_tail_sim_dic={}
        head_tail_sim_dic= json.load(open(file_path))['data']
    else:
        head_tail_sim_dic={}
        file= 'essential_files/head_tail_sim_dic.json'
        h_data={'data':head_tail_sim_dic}
        with open(file, 'w') as fp:
            json.dump(h_data, fp)

    if args.wordanalogy_model not in head_tail_sim_dic.keys():
        head_tail_sim_dic[args.wordanalogy_model]={}
    else:

        head_tail_sim_dic[args.wordanalogy_model]={}

       

    temp_sim=[]
    temp_dist=[]
    for dname in Similarity_in_Hidden_Layers.keys():
        temp_sim=[]
        temp_dist=[]
        for h in Similarity_in_Hidden_Layers[dname]['positive']['head']['sim']:
            t=Similarity_in_Hidden_Layers[dname]['positive']['head']['sim'][h]
      
            avg=sum(t)/len(t)
            temp_sim.append(avg.cpu().item())

            #print('h,avg sim',h,avg)

        for h in Similarity_in_Hidden_Layers[dname]['positive']['head']['dist']:
            t=Similarity_in_Hidden_Layers[dname]['positive']['head']['dist'][h]
            avg=sum(t)/len(t)
            temp_dist.append(avg.cpu().item())
            #print('h,avg dist',h,avg)
        # print('head')
        # print('temp_sim',temp_sim)
        # print('temp_dist',temp_dist)
        head_tail_sim_dic[args.wordanalogy_model][dname]={}
        head_tail_sim_dic[args.wordanalogy_model][dname]['head_sim']=temp_sim
        head_tail_sim_dic[args.wordanalogy_model][dname]['head_dist']=temp_dist
        temp_sim=[]
        temp_dist=[]

        temp_sim=[]
        temp_dist=[]
        for h in Similarity_in_Hidden_Layers[dname]['positive']['tail']['sim']:
            t=Similarity_in_Hidden_Layers[dname]['positive']['tail']['sim'][h]
            avg=sum(t)/len(t)
            temp_sim.append(avg.cpu().item())

            #print('h,avg sim',h,avg)
        #dname


        for h in Similarity_in_Hidden_Layers[dname]['positive']['tail']['dist']:
            t=Similarity_in_Hidden_Layers[dname]['positive']['tail']['dist'][h]
            avg=sum(t)/len(t)
            temp_dist.append(avg.cpu().item())
            #print('h,avg dist',h,avg)

        # print('tail')
        # print('temp_sim',temp_sim)
        # print('temp_dist',temp_dist)
        if len(temp_sim)>26 or len(temp_dist)>26:
            print('tail_sim')
            exit()

        head_tail_sim_dic[args.wordanalogy_model][dname]['tail_sim']=temp_sim
        head_tail_sim_dic[args.wordanalogy_model][dname]['tail_dist']=temp_dist
 
        print('negative')
        ##

        temp_sim=[]
        temp_dist=[]
        for h in Similarity_in_Hidden_Layers[dname]['negative']['head']['sim']:
            t=Similarity_in_Hidden_Layers[dname]['negative']['head']['sim'][h]
            avg=sum(t)/len(t)
            temp_sim.append(avg.cpu().item())

            #print('h,avg sim',h,avg)

        for h in Similarity_in_Hidden_Layers[dname]['negative']['head']['dist']:
            t=Similarity_in_Hidden_Layers[dname]['negative']['head']['dist'][h]
            avg=sum(t)/len(t)
            temp_dist.append(avg.cpu().item())
            #print('h,avg dist',h,avg)
        # print('head')
        # print('temp_sim',temp_sim)
        # print('temp_dist',temp_dist)
        if len(temp_sim)>26 or len(temp_dist)>26:
            print('head_sim_')
            exit()
        head_tail_sim_dic[args.wordanalogy_model][dname]['head_sim_']=temp_sim
        head_tail_sim_dic[args.wordanalogy_model][dname]['head_dist_']=temp_dist
        temp_sim=[]
        temp_dist=[]

        for h in Similarity_in_Hidden_Layers[dname]['negative']['tail']['sim']:
            t=Similarity_in_Hidden_Layers[dname]['negative']['tail']['sim'][h]
            avg=sum(t)/len(t)
            temp_sim.append(avg.cpu().item())

            #print('h,avg sim',h,avg)

        for h in Similarity_in_Hidden_Layers[dname]['negative']['tail']['dist']:
            t=Similarity_in_Hidden_Layers[dname]['negative']['tail']['dist'][h]
            avg=sum(t)/len(t)
            temp_dist.append(avg.cpu().item())
            #print('h,avg dist',h,avg)

        # print('tail')
        # print('temp_sim',temp_sim)
        # print('temp_dist',temp_dist)
        if len(temp_sim)>26 or len(temp_dist)>26:
            print('tail_sim_')
            exit()
        head_tail_sim_dic[args.wordanalogy_model][dname]['tail_sim_']=temp_sim
        head_tail_sim_dic[args.wordanalogy_model][dname]['tail_dist_']=temp_dist
        print('args.wordanalogy_model',args.wordanalogy_model)
    # file= 'essential_files/head_tail_sim_dic.json'
    # h_data={'data':head_tail_sim_dic}
    # with open(file, 'w') as fp:
    #     json.dump(h_data, fp)

    for modelname in head_tail_sim_dic.keys():
        if modelname!=args.wordanalogy_model:
            continue
        print('#########################'+str(modelname)+'###############')

        for dataname in head_tail_sim_dic[modelname].keys():
            print('dataname',dataname)
            for sim in head_tail_sim_dic[modelname][dataname].keys():
                name=modelname+'_'+dataname+'_'+sim
                value=head_tail_sim_dic[modelname][dataname][sim]
                print(name+'=',value)

         




def update_matrix(not_correct_co_occenrence_matrix,not_correct_count_dic,eval_d_p,negative_q_predicted):
    ws_source=[eval_d_p['w1'],eval_d_p['w2']]
    ws_target=[eval_d_p['w3'],eval_d_p['w4']]

    ws_all=ws_source +ws_target
    #ws_all.sort(reverse=True)
    negative_q_predicted
    ws_target_prdicted=[negative_q_predicted['w3'],negative_q_predicted['w4']]
    
    #print('ws_all',ws_all)
    ws_all_string='_'.join(ws_all)

    perm={'x':eval_d_p['w1'],'x*':eval_d_p['w2'],'y':eval_d_p['w3'],'y*':eval_d_p['w4']}
    perm_inv={eval_d_p['w1']:'x',eval_d_p['w2']:'x*',eval_d_p['w3']:'y',eval_d_p['w4']:'y*'}

    for w in ws_all:
        w_position=perm_inv[w]
        if w in not_correct_count_dic.keys():
            item={'analogous terms':ws_all_string,'position':w_position,'predicted_ws':ws_target_prdicted}
            not_correct_count_dic[w].append(item)
        else:
            not_correct_count_dic[w]=[]
            item={'analogous terms':ws_all_string,'position':w_position,'predicted_ws':ws_target_prdicted}
            not_correct_count_dic[w].append(item)



    if ws_all_string in not_correct_co_occenrence_matrix.keys():
        for w in ws_all:
            w_position=perm_inv[w]

            item=not_correct_co_occenrence_matrix[ws_all_string][w]
            w_position=perm_inv[w]
            item[w_position]=item[w_position]+1
            not_correct_co_occenrence_matrix[ws_all_string][w]=item

    else:
        not_correct_co_occenrence_matrix[ws_all_string]={}

        for w in ws_all:

            item={'x':0,'x*':0,'y':0,'y*':0}
            w_position=perm_inv[w]
            item[w_position]=item[w_position]+1

            
            not_correct_co_occenrence_matrix[ws_all_string][w]=item



def check_in(not_found,Ws,not_corrct_terms):
    Ws=[w.lower() for w in Ws]
    T=False
    w_cat_dic={'w1':'True','w2':'True','w3':'True','w4':'True'}
    for w,wn in zip(Ws,list(w_cat_dic.keys())):
        if w in not_corrct_terms['correct']:
            if w not in not_corrct_terms['not_correct']:
                continue
            else:
                correct_rate=not_corrct_terms['correct'][w]
                Notcorrect_rate=not_corrct_terms['not_correct'][w]
                if correct_rate>Notcorrect_rate:
                    #print(w,correct_rate,Notcorrect_rate)

                    continue
                else:
                    #print('fals',w,correct_rate,Notcorrect_rate)
                    w_cat_dic[wn]='False'
                    continue
              

        elif w in not_corrct_terms['not_correct']:
            if w not in not_corrct_terms['correct']:
                w_cat_dic[wn]='False'
                continue
            else:
                correct_rate=not_corrct_terms['correct'][w]
                Notcorrect_rate=not_corrct_terms['not_correct'][w]
                if correct_rate<Notcorrect_rate:
                    #print('fals',w,correct_rate,Notcorrect_rate)
                    w_cat_dic[wn]='False'
                    continue
                else:
                    
                    continue
        else:
            not_found=not_found+1
            #print('not_found',not_found)
            w_cat_dic[wn]='0'
            # print('w',w)
            # print('not_corrct_terms',not_corrct_terms)
            # print('w',w)
            # exit()
            continue
            #return None,not_found,w_cat_dic
    tot=[t for t in w_cat_dic.values()]
    tot_=[]
    for t in tot:
        if t!='0':
            tot_.append(t)

    tot=[1 if t=='True' else 0 for t in w_cat_dic.values()]

    N=len(tot)
    tot=sum(tot)

    if tot>=(N/2):
        return True, not_found,w_cat_dic
    else:
        return False,not_found,w_cat_dic






#####

def basian_analysis_of_conceptqa_vs_analogyqa(predictions_all,word_analogy_types_dic_rev):
    

    file='essential_files/conceptqa_TF_table.json'
    conceptqa_TF_table= json.load(open(file))['conceptqa_TF_table']

    probabilities={'relTrue_SemanticTrue':[],'relTrue_SemanticFalse':[] ,'relFalse_SemanticFalse':[],'relFalse_SemanticTrue':[],'notfound':[]}
    probabilities_s_given_r={'True':[],'False':[] }
    probabilities_r_given_s=[]


    print('basian_analysis')
    for k in predictions_all.keys():
        tp=word_analogy_types_dic_rev[k]
        print('tp',tp)
        correct=0
        not_found=0
        not_correct=0
        for question in predictions_all[k]:
            eval_d=predictions_all[k][question]['positive'][0]['eval']['supports']

            positive_analogies_similarity=predictions_all[k][question]['positive'][0]['similarity']
            negative_analogies=predictions_all[k][question]['negative']
            choices_sim=[]
            for q in negative_analogies:
                sim =q['similarity']
                choices_sim.append(sim)
            if len(choices_sim)<3:
                continue
            max_sim=max(choices_sim)
            if positive_analogies_similarity>max_sim:
                correct=correct+1
                L=[eval_d['w1'],eval_d['w2'],eval_d['w3'],eval_d['w4']]
                semantic_t,not_found,w_cat_dic=check_in(not_found,L,conceptqa_TF_table)
                probabilities_s_given_r['True'].append(w_cat_dic)

                for t in w_cat_dic.values():
                    if t=='True':
                        probabilities_r_given_s.append(['t','t'])

                    elif t=='False':
                        probabilities_r_given_s.append(['t','f'])
                    elif t=='0':
                        probabilities_r_given_s.append(['t','0'])
                

            else:
                not_correct=not_correct+1
                L=[eval_d['w1'],eval_d['w2'],eval_d['w3'],eval_d['w4']]
                semantic_t,not_found,w_cat_dic=check_in(not_found,L,conceptqa_TF_table)
                # if t==None:
                #     continue
                #     continue
                probabilities_s_given_r['False'].append(w_cat_dic)
                for t in w_cat_dic.values():
                    if t=='True':
                        probabilities_r_given_s.append(['f','t'])

                    elif t=='False':
                        probabilities_r_given_s.append(['f','f'])
                    elif t=='0':
                        probabilities_r_given_s.append(['f','0'])
    total=correct+not_correct
    print('correct',correct)
    print('not_correct',not_correct)
    print('total',total)
    acc=correct/(correct+not_correct)
    print('acc',acc)

    #print('model_name',model_name)

    
   
    
    #print('correct_rel_based_acc',correct_rel_based_acc)
    print('###############')
    print('probabilities_s_given_r',probabilities_s_given_r)
    for ki,k in enumerate(probabilities_s_given_r):
        pattern={}
        ti=0
        true=0
        false=0
        for t in probabilities_s_given_r[k]:
            ti=ti+1
            temp=[i for i in t.values()]
            for t in temp:
                #print('t',t)
                if t=='True':
                    true=true+1
                elif t=='False':
                    false=false+1

            temp='-'.join(temp)
            if temp in pattern:
                pattern[temp]=pattern[temp]+1
            else:
                pattern[temp]=1
        print('k',k,ki,ti)

        # for p in pattern:
        #     print('pattern',p,pattern[p])
        print('t',true)
        print('f',false)
        print('s=t|r=',k,'=',true/(true+false+1))
        print('s=f|r=',k,'=',false/(true+false+1))
        ##############
    R={'Rt_given_s_f':0,'Rt_given_s_t':0,'Rf_given_s_f':0,'Rf_given_s_t':0}
    for t in probabilities_r_given_s:
        r,s=t[0],t[1]
        if r=='t' and s=='t':
            R['Rt_given_s_t']+=1
        elif  r=='t' and s=='f':
            R['Rt_given_s_f']+=1


        if r=='f' and s=='t':
            R['Rf_given_s_t']+=1
        elif  r=='f' and s=='f':
            R['Rf_given_s_f']+=1

    print('R',R)
    print('p(r=t|s=f)=',R['Rt_given_s_f']/(R['Rt_given_s_f']+R['Rf_given_s_f']))
    print('p(r=f|s=f)=',R['Rf_given_s_f']/(R['Rf_given_s_f']+R['Rt_given_s_f']))
    print('p(r=t|s=t)=',R['Rt_given_s_t']/(R['Rt_given_s_t']+R['Rf_given_s_t']))
    print('p(r=f|s=t)=',R['Rf_given_s_t']/(R['Rt_given_s_t']+R['Rf_given_s_t']))






    print('#####')
    print('not_found',not_found)
    return acc


      

       
def print_predictions(predictions_all,word_analogy_types_dic_rev):

    lexical_data=['sre','BLESS','EVALution','CogALexV','ROOT09','KandH_plus_N','semeval_data']

    relation_based_acc={}

    not_correct_co_occenrence_matrix={}

    not_correct_count_dic={}

    all_w={}
    acc_for_evaluation_ekar=0
    acc_for_evaluation_sat=0


    # file='essential_files/wordanalogyrel_dic_sre.json'
    # with open(file) as f:
    #     wordanalogyrel_dic_sre = json.load(f)['rel_dic']
    #     wordanalogyrel_dic_sre_rev = {y: x for x, y in wordanalogyrel_dic_sre.items()}
    ##
    # file='essential_files/wordanalogyrel_dic_semeval.json'
    # with open(file) as f:
    #     wordanalogyrel_dic_semeval_2012 = json.load(f)['rel_dic']
    #     wordanalogyrel_dic_semeval_2012_rev = {y: x for x, y in wordanalogyrel_dic_semeval_2012.items()}


 
    polar_acc={'positive':{'correct':0,'not_correct':0},'negative':{'correct':0,'not_correct':0}}
    overall_cm={'correct':0,'not_correct':0,'equal':0,'polar_acc':copy.deepcopy(polar_acc)}
    for k in predictions_all.keys():
        tp=word_analogy_types_dic_rev[k]
        print('tp',tp)
        polar_acc={'positive':{'correct':0,'not_correct':0},'negative':{'correct':0,'not_correct':0}}


        flag=False
        p={'correct':0,'not_correct':0,'equal':0,'polar_acc':copy.deepcopy(polar_acc)}
        n={'correct':0,'not_correct':0,}
        dic={'positive':p,'negative':n}
        if tp=='special':
            flag=True

            t1={'correct':0,'not_correct':0,'equal':0,'polar_acc':copy.deepcopy(polar_acc)}
            t2={'High':copy.deepcopy(t1),'Low':copy.deepcopy(t1)}
            t={'Categorical':copy.deepcopy(t2),'Comp':copy.deepcopy(t2),'Causal':copy.deepcopy(t2)}
            p={'Near':copy.deepcopy(t),'Far':copy.deepcopy(t)}
        elif tp=='scan':
            flag=True
            t1={'correct':0,'not_correct':0,'equal':0,'polar_acc':copy.deepcopy(polar_acc)}
            p={'science':copy.deepcopy(t1),'metaphor':copy.deepcopy(t1)}
        correct=0
        not_correct=0

        

        #print('predictions_all[k]',predictions_all[k])

   
        for question in predictions_all[k]:



            polar_acc_q={'positive':{'correct':0,'not_correct':0},'negative':{'correct':0,'not_correct':0}}
            qi_no=0
            for q in predictions_all[k][question]['positive']:

                polar_acc=q['polar_acc']['value']
                qi_no=qi_no+1
                eval_d_=q['eval']['supports']
                ws=[eval_d_['w1'],eval_d_['w2'],eval_d_['w3'],eval_d_['w4']]
                ws_t='-'.join(ws)
                for w in ws:
                    if w in all_w.keys():
                        all_w[w].append(ws_t)
                    else:
                        all_w[w]=[]

                        all_w[w].append(ws_t)


                if qi_no==1:
                    eval_d_p=q['eval']['supports']
                sim_d=q['similarity']
                if polar_acc==1:
                    polar_acc_q['positive']['correct']=polar_acc_q['positive']['correct']+1
                else:
                    polar_acc_q['positive']['not_correct']=polar_acc_q['positive']['not_correct']+1

            
            positive_analogies_similarity=predictions_all[k][question]['positive'][0]['similarity']
            negative_analogies=predictions_all[k][question]['negative']
            negative_similarities=[]
            if flag:
                if tp=='special':
                    eval_d=predictions_all[k][question]['positive'][0]['eval']
                    #['Distracter_Salience', 'Relation', 'Analogy_Stem', 'ACC', 'RT', 'Semantic_Distance']
                    Distracter_Salience=eval_d['supports']['prop']['Distracter_Salience']
                    Relation=eval_d['supports']['prop']['Relation']
                    Semantic_Distance=eval_d['supports']['prop']['Semantic_Distance']
                elif tp=='scan':
                    eval_d=predictions_all[k][question]['positive'][0]['eval']['supports']['eval_d']
                    #print('eval_d',eval_d)
                    #exit()


            temp=True
            #print('###############')
            #print('lennegative_analogies',len(negative_analogies))
            for q in negative_analogies:

                sim =q['similarity']
                #print('sim',sim)
                polar_acc=q['polar_acc']['value']
                if polar_acc==1:
                    polar_acc_q['negative']['correct']=polar_acc_q['negative']['correct']+1
                else:
                    polar_acc_q['negative']['not_correct']=polar_acc_q['negative']['not_correct']+1
                #if args.wordanalogy_model!='classification_train_head':
                if sim>=positive_analogies_similarity:
                    negative_q_predicted=q['eval']['supports']
             
                    if sim==positive_analogies_similarity:
                        temp='equal'
                    else:
                        temp=False
                    # print('not correct')
                    # print('positive_analogies_similarity',positive_analogies_similarity)
                    # print('sim',sim) 
                else:


                    pass
            if tp in lexical_data :
                #print('tp',tp)
                #print('temp',lexical_data)
                y=predictions_all[k][question]['positive'][0]['y']
                r=predictions_all[k][question]['positive'][0]['r']
                r=r.item()

                # print('y',y)
                # print('r',r)
                # if tp in ['BLESS','EVALution','CogALexV','ROOT09','KandH_plus_N']:
                #     r_label=wordanalogyrel_dic_lexical_rev[r]
                # elif tp in ['wikidata','conll']:
                #     r_label=wordanalogyrel_dic_sre_rev[r]
                # elif tp =='semeval_data':
                #     r_label=wordanalogyrel_dic_semeval_2012_rev[r]
                # else:
                #     continue
                ##############################
                r=r#r_label
                # print('r_label',r_label)
                # print('****')

                if tp in relation_based_acc.keys():
                    if r in relation_based_acc[tp].keys():
                        if temp!=True:
                            relation_based_acc[tp][r]['not_correct']=copy.deepcopy(relation_based_acc[tp][r]['not_correct'])+1

                        else:
                            relation_based_acc[tp][r]['correct']=copy.deepcopy(relation_based_acc[tp][r]['correct'])+1

                    else:
                        relation_based_acc[tp][r]={'correct':0,'not_correct':0}
                        if temp!=True:
                            relation_based_acc[tp][r]['not_correct']=copy.deepcopy(relation_based_acc[tp][r]['not_correct'])+1

                        else:
                            relation_based_acc[tp][r]['correct']=copy.deepcopy(relation_based_acc[tp][r]['correct'])+1


                else:

                    relation_based_acc[tp]={r:{'correct':0,'not_correct':0}}
                
                    if r in relation_based_acc[tp].keys():
                        if temp!=True:
                            relation_based_acc[tp][r]['not_correct']=copy.deepcopy(relation_based_acc[tp][r]['not_correct'])+1

                        else:
                            relation_based_acc[tp][r]['correct']=copy.deepcopy(relation_based_acc[tp][r]['correct'])+1
                    else:
                        relation_based_acc[tp][r]={'correct':0,'not_correct':0}
                        if temp!=True:
                            relation_based_acc[tp]['not_correct']=copy.deepcopy(relation_based_acc[tp][r]['not_correct'])+1

                        else:
                            relation_based_acc[tp][r]['correct']=copy.deepcopy(relation_based_acc[tp][r]['correct'])+1




            if flag:
                if tp=='special':
                    #print('polar_acc_q',polar_acc_q)
                    if temp==True:
                        p[Semantic_Distance][Relation][Distracter_Salience]['correct']=p[Semantic_Distance][Relation][Distracter_Salience]['correct']+1
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['positive']['correct']=polar_acc_q['positive']['correct']+\
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['positive']['correct']
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['negative']['correct']=polar_acc_q['negative']['correct']+\
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['negative']['correct']

                        ##
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['positive']['not_correct']=polar_acc_q['positive']['not_correct']+\
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['positive']['not_correct']
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['negative']['not_correct']=polar_acc_q['negative']['not_correct']+\
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['negative']['not_correct']

                    elif temp==False:
                        p[Semantic_Distance][Relation][Distracter_Salience]['not_correct']=p[Semantic_Distance][Relation][Distracter_Salience]['not_correct']+1
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['positive']['correct']=polar_acc_q['positive']['correct']+\
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['positive']['correct']
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['negative']['correct']=polar_acc_q['negative']['correct']+\
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['negative']['correct']

                        ##
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['positive']['not_correct']=polar_acc_q['positive']['not_correct']+\
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['positive']['not_correct']
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['negative']['not_correct']=polar_acc_q['negative']['not_correct']+\
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['negative']['not_correct']
                    elif temp=='equal':
                        p[Semantic_Distance][Relation][Distracter_Salience]['equal']=p[Semantic_Distance][Relation][Distracter_Salience]['equal']+1
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['positive']['correct']=polar_acc_q['positive']['correct']+\
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['positive']['correct']
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['negative']['correct']=polar_acc_q['negative']['correct']+\
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['negative']['correct']

                        ##
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['positive']['not_correct']=polar_acc_q['positive']['not_correct']+\
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['positive']['not_correct']
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['negative']['not_correct']=polar_acc_q['negative']['not_correct']+\
                        p[Semantic_Distance][Relation][Distracter_Salience]['polar_acc']['negative']['not_correct']
                if tp=='scan':
                    if temp==True:
                        p[eval_d]['correct']=p[eval_d]['correct']+1
                        p[eval_d]['polar_acc']['positive']['correct']=p[eval_d]['polar_acc']['positive']['correct']+polar_acc_q['positive']['correct']
                        p[eval_d]['polar_acc']['negative']['correct']=p[eval_d]['polar_acc']['negative']['correct']+polar_acc_q['negative']['correct']
                        
                        p[eval_d]['polar_acc']['positive']['not_correct']=p[eval_d]['polar_acc']['positive']['not_correct']+polar_acc_q['positive']['not_correct']
                        p[eval_d]['polar_acc']['negative']['not_correct']=p[eval_d]['polar_acc']['negative']['not_correct']+polar_acc_q['negative']['not_correct']
                        ##
                        overall_cm['correct']=overall_cm['correct']+1
                        overall_cm['polar_acc']['positive']['correct']=overall_cm['polar_acc']['positive']['correct']+polar_acc_q['positive']['correct']
                        overall_cm['polar_acc']['negative']['correct']=overall_cm['polar_acc']['negative']['correct']+polar_acc_q['negative']['correct']
                        
                        overall_cm['polar_acc']['positive']['not_correct']=overall_cm['polar_acc']['positive']['not_correct']+polar_acc_q['positive']['not_correct']
                        overall_cm['polar_acc']['negative']['not_correct']=overall_cm['polar_acc']['negative']['not_correct']+polar_acc_q['negative']['not_correct']
                        

                    elif temp==False:
    
                        p[eval_d]['not_correct']=p[eval_d]['not_correct']+1
                        p[eval_d]['polar_acc']['positive']['correct']=p[eval_d]['polar_acc']['positive']['correct']+polar_acc_q['positive']['correct']
                        p[eval_d]['polar_acc']['negative']['correct']=p[eval_d]['polar_acc']['negative']['correct']+polar_acc_q['negative']['correct']
                        
                        p[eval_d]['polar_acc']['positive']['not_correct']=p[eval_d]['polar_acc']['positive']['not_correct']+polar_acc_q['positive']['not_correct']
                        p[eval_d]['polar_acc']['negative']['not_correct']=p[eval_d]['polar_acc']['negative']['not_correct']+polar_acc_q['negative']['not_correct']
                        ##
                        overall_cm['not_correct']=overall_cm['not_correct']+1
                        overall_cm['polar_acc']['positive']['correct']=overall_cm['polar_acc']['positive']['correct']+polar_acc_q['positive']['correct']
                        overall_cm['polar_acc']['negative']['correct']=overall_cm['polar_acc']['negative']['correct']+polar_acc_q['negative']['correct']
                        
                        overall_cm['polar_acc']['positive']['not_correct']=overall_cm['polar_acc']['positive']['not_correct']+polar_acc_q['positive']['not_correct']
                        overall_cm['polar_acc']['negative']['not_correct']=overall_cm['polar_acc']['negative']['not_correct']+polar_acc_q['negative']['not_correct']
                        

                    elif temp=='equal':

                        p[eval_d]['equal']=p[eval_d]['equal']+1
                        p[eval_d]['polar_acc']['positive']['correct']=p[eval_d]['polar_acc']['positive']['correct']+polar_acc_q['positive']['correct']
                        p[eval_d]['polar_acc']['negative']['correct']=p[eval_d]['polar_acc']['negative']['correct']+polar_acc_q['negative']['correct']
                        
                        p[eval_d]['polar_acc']['positive']['not_correct']=p[eval_d]['polar_acc']['positive']['not_correct']+polar_acc_q['positive']['not_correct']
                        p[eval_d]['polar_acc']['negative']['not_correct']=p[eval_d]['polar_acc']['negative']['not_correct']+polar_acc_q['negative']['not_correct']
                        ###
                        overall_cm['equal']=overall_cm['equal']+1
                        overall_cm['polar_acc']['positive']['correct']=overall_cm['polar_acc']['positive']['correct']+polar_acc_q['positive']['correct']
                        overall_cm['polar_acc']['negative']['correct']=overall_cm['polar_acc']['negative']['correct']+polar_acc_q['negative']['correct']
                        
                        overall_cm['polar_acc']['positive']['not_correct']=overall_cm['polar_acc']['positive']['not_correct']+polar_acc_q['positive']['not_correct']
                        overall_cm['polar_acc']['negative']['not_correct']=overall_cm['polar_acc']['negative']['not_correct']+polar_acc_q['negative']['not_correct']
                        



                #print('p',p)
                #continue
            else:
                if temp==True:
                    #print('1*********')
                    p['correct']=p['correct']+1
                    p['polar_acc']['positive']['correct']=p['polar_acc']['positive']['correct']+polar_acc_q['positive']['correct']
                    p['polar_acc']['negative']['correct']=p['polar_acc']['negative']['correct']+polar_acc_q['negative']['correct']
                    ###
                    p['polar_acc']['positive']['not_correct']=p['polar_acc']['positive']['not_correct']+polar_acc_q['positive']['not_correct']
                    p['polar_acc']['negative']['not_correct']=p['polar_acc']['negative']['not_correct']+polar_acc_q['negative']['not_correct']
                    
                    #####
                    overall_cm['correct']=overall_cm['correct']+1
                    overall_cm['polar_acc']['positive']['correct']=overall_cm['polar_acc']['positive']['correct']+polar_acc_q['positive']['correct']
                    overall_cm['polar_acc']['negative']['correct']=overall_cm['polar_acc']['negative']['correct']+polar_acc_q['negative']['correct']
                    
                    overall_cm['polar_acc']['positive']['not_correct']=overall_cm['polar_acc']['positive']['not_correct']+polar_acc_q['positive']['not_correct']
                    overall_cm['polar_acc']['negative']['not_correct']=overall_cm['polar_acc']['negative']['not_correct']+polar_acc_q['negative']['not_correct']
                    


                elif temp==False:
                    #print('2*********')
                    update_matrix(not_correct_co_occenrence_matrix,not_correct_count_dic,eval_d_p,negative_q_predicted)
            
                    p['not_correct']=p['not_correct']+1
                    p['polar_acc']['positive']['correct']=p['polar_acc']['positive']['correct']+polar_acc_q['positive']['correct']
                    p['polar_acc']['negative']['correct']=p['polar_acc']['negative']['correct']+polar_acc_q['negative']['correct']
                    ###
                    p['polar_acc']['positive']['not_correct']=p['polar_acc']['positive']['not_correct']+polar_acc_q['positive']['not_correct']
                    p['polar_acc']['negative']['not_correct']=p['polar_acc']['negative']['not_correct']+polar_acc_q['negative']['not_correct']
                    ###

                    overall_cm['not_correct']=overall_cm['not_correct']+1
                    overall_cm['polar_acc']['positive']['correct']=overall_cm['polar_acc']['positive']['correct']+polar_acc_q['positive']['correct']
                    overall_cm['polar_acc']['negative']['correct']=overall_cm['polar_acc']['negative']['correct']+polar_acc_q['negative']['correct']
                    
                    overall_cm['polar_acc']['positive']['not_correct']=overall_cm['polar_acc']['positive']['not_correct']+polar_acc_q['positive']['not_correct']
                    overall_cm['polar_acc']['negative']['not_correct']=overall_cm['polar_acc']['negative']['not_correct']+polar_acc_q['negative']['not_correct']
                    
                elif temp=='equal':
                    #print('3*********')
                    update_matrix(not_correct_co_occenrence_matrix,not_correct_count_dic,eval_d_p,negative_q_predicted)
                    p['equal']=p['equal']+1
                    p['polar_acc']['positive']['correct']=p['polar_acc']['positive']['correct']+polar_acc_q['positive']['correct']
                    p['polar_acc']['negative']['correct']=p['polar_acc']['negative']['correct']+polar_acc_q['negative']['correct']
                    ###
                    p['polar_acc']['positive']['not_correct']=p['polar_acc']['positive']['not_correct']+polar_acc_q['positive']['not_correct']
                    p['polar_acc']['negative']['not_correct']=p['polar_acc']['negative']['not_correct']+polar_acc_q['negative']['not_correct']
                    ###
                    overall_cm['equal']=overall_cm['equal']+1
                    overall_cm['polar_acc']['positive']['correct']=overall_cm['polar_acc']['positive']['correct']+polar_acc_q['positive']['correct']
                    overall_cm['polar_acc']['negative']['correct']=overall_cm['polar_acc']['negative']['correct']+polar_acc_q['negative']['correct']
                    
                    overall_cm['polar_acc']['positive']['not_correct']=overall_cm['polar_acc']['positive']['not_correct']+polar_acc_q['positive']['not_correct']
                    overall_cm['polar_acc']['negative']['not_correct']=overall_cm['polar_acc']['negative']['not_correct']+polar_acc_q['negative']['not_correct']
                    
        if tp=='special' :
            tc=0
            tnc=0
            teq=0
            ##
            tppc=0
            tppnc=0

            tpnc=0
            tpnnc=0

            for K in p:
                L_p=[]
                for k in p[K]:
                    k_p={'correct':0,'not_correct':0,'p':{'correct':0,'not_correct':0},'n':{'correct':0,'not_correct':0},\
                    'total':{'correct':0,'not_correct':0}}
                    for k1 in p[K][k]:
                        c=p[K][k][k1]['correct']
                        nc=p[K][k][k1]['not_correct']
                        eq=p[K][k][k1]['equal']

                        ppc=p[K][k][k1]['polar_acc']['positive']['correct']
                        ppnc=p[K][k][k1]['polar_acc']['positive']['not_correct']
                        ##
                        pnc=p[K][k][k1]['polar_acc']['negative']['correct']
                        pnnc=p[K][k][k1]['polar_acc']['negative']['not_correct']
                        ###########
                        k_p['correct']=k_p['correct']+c
                        k_p['not_correct']=k_p['not_correct']+nc+eq
                        #
                        k_p['p']['correct']=k_p['p']['correct']+ppc
                        k_p['p']['not_correct']=k_p['p']['not_correct']+ppnc
                        ##
                        k_p['n']['correct']=k_p['n']['correct']+pnc
                        k_p['n']['not_correct']=k_p['n']['not_correct']+pnnc
                        ####################


                        tc=tc+c
                        tnc=tnc+nc
                        teq=teq+eq
                        ##

                        tppc=tppc+ppc
                        tppnc=tppnc+ppnc

                        tpnc=tpnc+pnc
                        tpnnc=tpnnc+pnnc
                        print(K,k,k1,' correct:',c)
                        print(K,k,k1,' not_correct:',nc)
                        print(K,k,k1,' eq:',eq)
                        print('q_acc',c/(c+nc+eq))
                        ##
                        print(K,k,k1,' p/n positive correct:',ppc)
                        print(K,k,k1,' p/n positive not-correct:',ppnc)

                        print(K,k,k1,' p/n negative correct:',pnc)
                        print(K,k,k1,' p/n negative not-correct:',pnnc)
                        print('******')
                    ###########
                    print('+++')
                    L_p.append(k_p)
                    k_p_correct=k_p['correct']
                    k_p_not_correct=k_p['not_correct']
                    #
                    k_p_p_correct=k_p['p']['correct']
                    k_p_P_not_correct=k_p['p']['not_correct']
                    ##
                    k_p_n_correct=k_p['n']['correct']
                    k_p_n_not_correct=k_p['n']['not_correct']
                    ######

                    q_acc=k_p_correct/(k_p_correct+k_p_not_correct)

                    pp_acc=k_p_p_correct/(k_p_p_correct+k_p_P_not_correct)

                    pn_acc=k_p_n_correct/(k_p_n_correct+k_p_n_not_correct)

                    over_all_p=(k_p_p_correct+k_p_n_correct)/(k_p_p_correct+k_p_n_correct+k_p_P_not_correct+k_p_n_not_correct)
                    

                    print('q_acc',q_acc)
                    print('pp_acc',pp_acc)
                    print('pn_acc',pn_acc)
                    print('over_all_p',over_all_p)
                    print('+++++')
            #####


            print('total correct',tc)
            print('total not correct',tnc)
            print('total equal',teq)
      
            print('question-acc',(tc)/(tc+tnc+teq))
      
            print('total p/n correct',tppc)
            print('total p/n not-correct',tppnc)

            print('total p/n negative correct',tpnc)
            print('total p/n negative not-correct',tpnnc)


        elif tp=='scan' :
            for K in p:

                tc=p[K]['correct']
                tnc=p[K]['not_correct']
                teq=p[K]['equal']

                ###
                polar_acc=p[K]['polar_acc']
                pc=polar_acc['positive']['correct']
                pnc=polar_acc['positive']['not_correct']
                ##
                nc=polar_acc['negative']['correct']
                nnc=polar_acc['negative']['not_correct']
                print('p',K,p[K])
                print('positive_correct',pc/(pc+pnc))
                print('negative_correct',nc/(nc+nnc))
                print('overall',(pc+nc)/(nc+pc+nnc+pnc))
                print('question-acc',(tc)/(tc+tnc+teq))
        else:
            tc=p['correct']
            tnc=p['not_correct']
            teq=p['equal']
  
            #
            polar_acc=p['polar_acc']
            pc=polar_acc['positive']['correct']
            pnc=polar_acc['positive']['not_correct']
            ##
            nc=polar_acc['negative']['correct']
            nnc=polar_acc['negative']['not_correct']
            print('p',p)
            print('tc',tc)
            print('tnc',tnc)
            print('teq',teq)
            # print('positive_correct',pc/(pc+pnc))
            # print('negative_correct',nc/(nc+nnc))
            print('overall',(pc+nc)/(nc+pc+nnc+pnc))
            print('quetion-acc',(tc)/(tc+tnc+teq))
            if tp=='ekar':
                acc_for_evaluation_ekar=(tc)/(tc+tnc+teq)
            if tp=='sat':
                acc_for_evaluation_sat=(tc)/(tc+tnc+teq)

        print('+++++++++++++++++++++++++++++++')
        print('overall_cm',overall_cm)
        print('+++++++++++++++++++++++++++++++#######')

        for d in relation_based_acc.keys():
            print('data',data)
            dic_r_acc={}

            acc_data=relation_based_acc[d]
            for r in acc_data.keys():
                print('r',r)
                correct=acc_data[r]['correct']
                not_correct=acc_data[r]['not_correct']
                total=correct+not_correct
                acc=correct/(correct+not_correct+1)

                dic_r_acc[r]=acc
            dic_r_acc_sorted=dict(sorted(dic_r_acc.items(), key=lambda item: item[1]))
            for r in dic_r_acc_sorted.keys():
                acc=dic_r_acc_sorted[r]
                print('r',r)
                print('acc',acc)
                print('####')


        print('not correct word freq')
        count=0
        not_correct_count_dic
        not_correct_co_occenrence_matrix
        ws_dic={}

    return acc_for_evaluation_ekar,acc_for_evaluation_sat

###

def cal_similarity(eval_epxeriment_data,negative=False):
    import random
    from itertools import combinations

    def rSubset(arr, r):
        return list(combinations(arr, r))

    L_head=[]
    L_tail=[]

    L_head_all_h=[]
    L_tail_all_h=[]

    L_head_all_h_dist=[]
    L_tail_all_h_dist=[]
    for k in eval_epxeriment_data.keys():
        n=len(eval_epxeriment_data[k])
        m=4 if n>4 else n
        population=list(eval_epxeriment_data[k].keys())
        selected=random.sample(population, m)

        item_head={}
        item_tail={}

        pairs=rSubset(selected, 2)
        for p in pairs:
            Hiddens_head_source=None
            Hiddens_tail_source=None

            Hiddens_head_target=None
            Hiddens_tail_target=None
            for h in range(len(eval_epxeriment_data[k][p[0]])):


                t_k=k
                if negative and t_k==k: 
                    t_k=random.choice( list(eval_epxeriment_data.keys())) if negative else k
                    if t_k==k:
                        t_k=random.choice( list(eval_epxeriment_data.keys())) if negative else k
                    t_population=list(eval_epxeriment_data[t_k].keys())
                    #if len(t_population)<3:
                        #continue
                    t_selected_1=random.sample(t_population, 2)[0]
                else:
                    t_k=k
                    t_selected_1=p[1]

      
                head_source,tail_source=eval_epxeriment_data[k][p[0]][h]['head'],eval_epxeriment_data[k][p[0]][h]['tail']
                head_target,tail_target=eval_epxeriment_data[t_k][t_selected_1][h]['head'],eval_epxeriment_data[t_k][t_selected_1][h]['tail']
                

                head_source=torch.tensor(head_source).sum(0)
                tail_source=torch.tensor(tail_source).sum(0)

                head_target=torch.tensor(head_target).sum(0)
                tail_target=torch.tensor(tail_target).sum(0)

                Hiddens_head_source=head_source if Hiddens_head_source==None else torch.add(head_source,Hiddens_head_source) 
                Hiddens_tail_source=tail_source if Hiddens_tail_source==None else torch.add(tail_source,Hiddens_tail_source)

                Hiddens_head_target=head_target if Hiddens_head_target==None else torch.add(head_target,Hiddens_head_target)
                Hiddens_tail_target=tail_target if Hiddens_tail_target==None else torch.add(tail_target,Hiddens_head_target)


                cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                head_sim = cos(head_source, head_target)
                tail_sim = cos(tail_source, tail_target)

                head_dist = torch.nn.functional.pairwise_distance(head_source, head_target)#(w1_h - w3_h).pow(2).sum(1).sqrt()
                tail_dist = torch.nn.functional.pairwise_distance(tail_source, tail_target)#(w2_h - w4_h).pow(2).sum(1).sqrt()


                item_head[h]={'sim':head_sim,'dist':head_dist}
                item_tail[h]={'sim':tail_sim,'dist':tail_dist} 
            L_head.append(item_head)
            L_tail.append(item_tail)
            ##
            cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
            head_sim_all = cos(Hiddens_head_source, Hiddens_head_target)
            tail_sim_all = cos(Hiddens_tail_source, Hiddens_tail_target)
            head_dist_all = torch.nn.functional.pairwise_distance(Hiddens_head_source, Hiddens_head_target)#(w1_h - w3_h).pow(2).sum(1).sqrt()
            tail_dist_all = torch.nn.functional.pairwise_distance(Hiddens_tail_source, Hiddens_tail_target)#(w2_h - w4_h).pow(2).sum(1).sqrt()




            ##
            L_head_all_h.append(head_sim_all)
            L_tail_all_h.append(tail_sim_all)

            L_head_all_h_dist.append(head_dist_all)
            L_tail_all_h_dist.append(tail_dist_all)






    evg_all_h={'head':{'sim':{},'dist':{}},'tail':{'sim':{},'dist':{}}}
    k=list(eval_epxeriment_data.keys())[0]
    print('k',k)
    p=list(eval_epxeriment_data[k].keys())[0]
    for h in range(len(eval_epxeriment_data[k][p])):
        print('h',h)
        head_h_sim=[t[h]['sim'] for t in L_head]
        tail_h_sim=[t[h]['sim'] for t in L_tail]


        head_h_dist=[t[h]['dist'] for t in L_head]
        tail_h_dist=[t[h]['dist'] for t in L_tail]



        head_sim_avg_h = sum(head_h_sim) / len(head_h_sim)
        tail_sim_avg_h = sum(tail_h_sim) / len(tail_h_sim)


        head_dist_avg_h = sum(head_h_dist) / len(head_h_dist)
        tail_dist_avg_h = sum(tail_h_dist) / len(tail_h_dist)

        print('head_sim_avg_h',head_sim_avg_h)
        print('tail_sim_avg_h',tail_sim_avg_h)
        print('len(tail_h_dist),',len(tail_h_dist))

        print('head_dist_avg_h',head_dist_avg_h)
        print('tail_dist_avg_h',tail_dist_avg_h)
        print('len(tail_h_sim),',len(tail_h_sim))
        print('+++')
        if h in evg_all_h['head']['sim'].keys():
            evg_all_h['head']['sim'][h].append(head_sim_avg_h.item())
        else:
            evg_all_h['head']['sim'][h]=[]
            evg_all_h['head']['sim'][h].append(head_sim_avg_h.item())
        ###
        if h in evg_all_h['head']['dist'].keys():
            evg_all_h['head']['dist'][h].append(head_dist_avg_h.item())
        else:
            evg_all_h['head']['dist'][h]=[]
            evg_all_h['head']['dist'][h].append(head_dist_avg_h.item())


        if h in evg_all_h['tail']['sim'].keys():
            evg_all_h['tail']['sim'][h].append(head_sim_avg_h.item())
        else:
            evg_all_h['tail']['sim'][h]=[]
            evg_all_h['tail']['sim'][h].append(head_sim_avg_h.item())

                ###
        if h in evg_all_h['tail']['dist'].keys():
            evg_all_h['tail']['dist'][h].append(head_dist_avg_h.item())
        else:
            evg_all_h['tail']['dist'][h]=[]
            evg_all_h['tail']['dist'][h].append(head_dist_avg_h.item())





    head_sim_all_avg_h = sum(L_head_all_h) / len(L_head_all_h)
    tail_sim_all_avg_h = sum(L_tail_all_h) / len(L_tail_all_h)


    head_dist_all_avg_h = sum(L_head_all_h_dist) / len(L_head_all_h_dist)
    tail_dist_all_avg_h = sum(L_tail_all_h_dist) / len(L_tail_all_h_dist)


    print('head_sim_all_avg_h',head_sim_all_avg_h)
    print('tail_sim_all_avg_h',tail_sim_all_avg_h)
    print('len(L_tail_all_h),',len(L_tail_all_h))
    ####
    print('head_dist_all_avg_h',head_dist_all_avg_h)
    print('tail_dist_all_avg_h',tail_dist_all_avg_h)
    print('len(L_tail_all_h_dist),',len(L_tail_all_h_dist))

    sim_head=[evg_all_h['head']['sim'][h][0] for h in evg_all_h['head']['sim'].keys()]
    dist_head=[evg_all_h['head']['dist'][h][0] for h in evg_all_h['head']['dist'].keys()]

    sim_tail=[evg_all_h['tail']['sim'][h][0] for h in evg_all_h['tail']['sim'].keys()]
    dist_tail=[evg_all_h['tail']['dist'][h][0]for h in evg_all_h['tail']['dist'].keys()]

    print('sim_head',sim_head)
    print('dist_head',dist_head)

    #
    print('sim_tail',sim_tail)
    print('dist_tail',dist_tail)
    




    print('+++')




def get_settings(mode,model_name,data_name):

    data_selected=data_name
    tokenizer_special_dic='semeval_2012_re' if data_name=='semeval_2012' else 're' 
    if 'wordanalogy' in data_name :
        tokenizer_special_dic='default'

    _, _,tokenizer =  _get_pretrained_transformer3(data_selected,model_name,tokenizer_special_dic=tokenizer_special_dic) 

    Train_Args= {
    "batch_size": 128,
    "data_type": 'wikidata',
    "dev_record_file": "dev.npz",
    "test_record_file": "test.npz",
    "dev_rev_record_file": "dev_rev.npz",
    "test_rev_record_file": "test_rev.npz",
    "lr": 1e-5,
    "name": "baseline",
    "save_dir": "./models_saved",
    "seed": 224,
    "train_record_file": "train.npz",
    "local_rank":-1,
    "data_parallel":False,
    'abstract':'flagged_ents',
    'mode':'eval',
    'model_to_train':'rc',#unsupervise_re ,,rc
    'load_from_checkpoint':False,
    'vocab_size':len(tokenizer),
    'data_parallel':False,
    'checkpoint_path':"models_saved",
    'epoch':1,
    'scheduler_step':1200,
    'f1_score':'micro',
    'print_eval':False,
    'num_workers':0,
    'embed_size':1024,
    'n_class':352,
    'save':True,
    'both_ab_not_ab':'',
    'heads':['head_3',],
    'model_name':model_name,
    'experiment':False,
    'h1_embed_size':2,
    'h2_embed_size':512,
    'h3_embed_size':512,
    'train':True,
    'wordanalogy_model':'baseline',
    'train_on_all':False,
    'filter_type':None,
    'abcd_attention':False,
    'wordanalogy_train_data':['analogykb',],
    'wordanalogy_test_data':['ekar','RS','scan','google','bats','u4','u2','sat'],
    'epcoh_n':5,
    'fin_tune':False,
    'experiment_no':'four',
    'person_person':'all',
    'only_train_classifier':0,
    'similarity_measure':'offset',
    'wordanalogy_pretrain':False,
    'offset_classification':True,
    'epoch':0
    }

    torch.manual_seed(0)
    
    args=Args_dic(Train_Args)
    t_1024=['t5-large','bert-large-uncased','roberta-large','opt','prophetnet']
    t_768=['gpt2','bert_base_uncased','roberta-base','flaxopt']
    t_512=['t5-small',]

    if model_name in t_1024:
        args.embed_size=1024 
    elif model_name in t_768:
        args.embed_size=768
    elif model_name in t_512:
        args.embed_size=512

    args.mode=mode
    if mode=='eval':
        args.load_from_checkpoint=True
    args.data_type=data_name
    if data_name=='tacred':
        args.f1_score='micro'

    if data_name=='semeval':
        args.abstract='flagged_ents'
    elif args.data_type=='BLESS':
        args.n_class=6
        args.h3_embed_size=58
    elif args.data_type=='CogALexV':
        args.n_class=6
        args.h3_embed_size=58

    elif args.data_type=='EVALution':
        args.n_class=7
        args.h3_embed_size=58
    elif args.data_type=='ROOT09':
        args.n_class=5
        args.h3_embed_size=58

    elif args.data_type=='lexicalPlusAnalogykb':
        args.n_class=875

    elif args.data_type=='tacred':
        args.n_class=42
        args.h3_embed_size=42
    elif args.data_type=='retacred':
        args.n_class=41
        args.h3_embed_size=41

    elif args.data_type=='wikidata':
        args.n_class=352
        args.h3_embed_size=352

    elif args.data_type=='conll':
        args.n_class=5
        args.h3_embed_size=5
    elif data_name=='semeval':
        args.n_class=19
        args.f1_score='macro'
    elif data_name=='semeval_2012':
        file='essential_files/all_relation_dic.json'
        
        with open(file) as f:
            no_rel = json.load(f)['data']['relation']
            args.n_class=len(no_rel)


    if args.abstract!='learn_abstract':
        args.save_dir=args.checkpoint_path+'/'+args.abstract+'/model_'+str(args.abstract)+'_'+str(args.data_type)+'.t7'

    else:
        args.save_dir=args.checkpoint_path+'/model_'+str(args.abstract)+'_'+str(args.data_type)+'_learn_abstract.t7'

    # set n_gpu
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        print('device',device)
        if args.data_parallel:
            args.n_gpu = torch.cuda.device_count()
        else:
            args.n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    if args.device==torch.device('cpu'):
        args.batch_size=2

    return args


 