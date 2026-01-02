import os
from os.path import exists
from torch.nn.functional import log_softmax, pad
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import warnings
import argparse
from analogy_util import *
from RC_Model import *
from Analogy_Model import *

from Multichoice_Model import run_mc
# from conceptqa_mtcqa import run_mc_conceptqa
#from Additional_Experiments import plots_and_custom_experiments
from Additional_Experiments import *

import json
import pandas as pd
import csv
#######
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt##
###########
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from transformers import BertTokenizer, BertModel
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForImageClassification
import logging
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import os
import numpy as np
from sklearn.metrics import f1_score
import copy
logger = logging.getLogger(__name__)
from Abstraction import *

from Train_Eval import *
from analogy_util import _get_pretrained_transformer3





def get_settings(mode,model_name,data_name):

    #tokenizer = AutoTokenizer.from_pretrained("./tokenizer/") if model_name=='roberta-large' else BertTokenizer.from_pretrained("./tokenizer/")
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
    #temp=['BLESS','EVALution','CogALexV']
    if data_name=='tacred':
        args.f1_score='micro'

    if data_name=='semeval':
        args.abstract='flagged_ents'



    elif args.data_type=='BLESS':
        args.n_class=6
        args.h3_embed_size=58
        #args.combin_two=True

    elif args.data_type=='CogALexV':
        args.n_class=6
        args.h3_embed_size=58
        #args.combin_two=True

    elif args.data_type=='EVALution':
        args.n_class=7
        args.h3_embed_size=58
        #args.combin_two=True

    elif args.data_type=='ROOT09':
        args.n_class=5
        args.h3_embed_size=58


    elif args.data_type=='lexicalPlusAnalogykb':
        args.n_class=875


    elif args.data_type=='tacred':
        args.n_class=42
        args.h3_embed_size=42
        #args.combin_two=True
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
        file='essential_files/localdatasets/all_relation_dic.json'
        
        with open(file) as f:
            no_rel = json.load(f)['data']['relation']
            args.n_class=len(no_rel)


    if args.abstract!='learn_abstract':
        args.save_dir=args.checkpoint_path+'/'+args.abstract+'/model_'+str(args.abstract)+'_'+str(args.data_type)+'.t7'

    #model_to_train='unsupervise_re'
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


 




def experiment_run(data_name,experiment,mode='train',model_to_train='rc',backend_model_name='bert-base-uncased'):
    

        args=get_settings(mode,backend_model_name,data_name)
        args.model_name=backend_model_name
        model_name=backend_model_name


        if str(experiment)=='additional_exp':

            additinal_experiments=[\
            'cat2_solve_conceptqa',\
            'cat2_basian_analysis',\
            'cat2_plot_semantic_space_questions',\
            'cat2_plot_space_based_similarity',\
            'cat2_check_r_head_tail_concepts_types',\

            'cat1_print_sample',\
            'cat1_plot_sim_vs_abstract'
            'cat1_semantic_space_experiment',\
            'cat1_solve_analogies',\
            'cat1_plot_permutations_dist',\
            'cat1_interagreement_between_models_on_data',\
            'cat1_interagreement_between_models',\
            'do_table_6'
            ]

            experiment_name='cat2_basian_analysis'

            plots_and_custom_experiments(experiment_name,args=args)

        elif str(experiment)=='sentential_re_paper': #re

            expert_head=('head_1','head_3','head_2','head_conditional')
            h3=('head_conditional',)

            h=h3
            print('h',h)

            args=get_settings(mode,model_name,data_name)
            args.heads=list(h)
            args.h3_embed_size=512
            args.abstract='flagged_ents'
            args.experiment_no='sentential_re_paper'
            args.epcoh_n=2
            args.model_to_train=model_to_train#
            print('model_to_train',model_to_train)


            if 'head_1' in args.heads and 'head_conditional' in args.heads:
                args.batch_size=12
                args.load_from_checkpoint=True if mode=='train' else True

            else:
                args.batch_size=64
                args.load_from_checkpoint=False if mode=='train' else False
            args.save=True
            args.experiment=True
            args.load_from_checkpoint=True if mode!='train' else False
            if args.device==torch.device('mps'):
                args.batch_size=1
            print('batch_size',args.batch_size)
                #model_name='roberta-large'

         
            
            train_loader,dev_loader_b=rc_data(args)
            print('len',len(train_loader))
            print('len',len(dev_loader_b))
            DATA={'train':train_loader,'dev':dev_loader_b}

            georoc_train_eval(data_name,mode=mode,exp_args=args,DATA=DATA)


        elif str(experiment)=='lexical_offset':# wodanalogy_re_model

                expert_head=('head_1','head_3','head_2','head_conditional')
                h3=('head_3',)

                h=h3
                print('h',h)

                args=get_settings(mode,model_name,data_name)
                args.heads=list(h)
                args.h3_embed_size=512
                args.abstract='flagged_ents'#'flagged_ents'
                args.experiment_no='six'
                args.epcoh_n=2
                args.model_to_train=model_to_train#
                print('model_to_train',model_to_train)
                args.h3_embed_size=512
     
                args.n_class
                print('args.n_class',args.n_class)


                args.batch_size=24
                args.load_from_checkpoint=False if mode=='train' else False
                args.save=True
                args.experiment=True
                args.load_from_checkpoint=True if mode!='train' else False
                if args.device==torch.device('mps') or args.device==torch.device('cpu'):
                    args.batch_size=4
                print('batch_size',args.batch_size)

                train_loader,dev_loader_b=rc_data(args)
                print('len',len(train_loader))
                print('len',len(dev_loader_b))
                DATA={'train':train_loader,'dev':dev_loader_b}

                georoc_train_eval(data_name,mode=mode,exp_args=args,DATA=DATA)


     


        elif str(experiment)=='semeval_2012':

            expert_head=('head_1','head_3','head_2','head_conditional')
            h3=('head_conditional',)

            h=h3
            print('h',h)

            args=get_settings(mode,model_name,data_name)
            args.heads=list(h)
            args.h3_embed_size=512
            args.abstract='flagged_ents'#'flagged_ents'
            args.experiment_no='semeval_2012'
            args.epcoh_n=3
            args.model_to_train=model_to_train#
            print('model_to_train',model_to_train)


            if 'head_1' in args.heads and 'head_conditional' in args.heads:
                args.batch_size=12
                args.load_from_checkpoint=True if mode=='train' else True

            else:
                args.batch_size=24
                args.load_from_checkpoint=True if mode=='train' else True
            args.save=True
            args.experiment=True
            args.load_from_checkpoint=True if mode!='train' else True
            if args.device==torch.device('mps') or args.device==torch.device('cpu'):
                args.batch_size=4
            print('batch_size',args.batch_size)
                #model_name='roberta-large'
            
            train_loader,dev_loader_b=rc_data(args)
            print('len',len(train_loader))
            print('len',len(dev_loader_b))
            DATA={'train':train_loader,'dev':dev_loader_b}

            georoc_train_eval(data_name,mode=mode,exp_args=args,DATA=DATA)


        #['paper2_WikidataPretraining','paper2_EVALutionPretraining','paper2_lexicalZeroshotTraining'+'paper2_RE_Trained_lexicalTraining']
        elif experiment=='mtcqa':
            mtqa_experiment_names=['conceptqa','pretraining_with_easy_hard','overfit']
            name='pretraining_with_easy_hard'
            run_mc(name)

        elif str(experiment)=='overfit':
            # pre-trian with wikidata




            args=get_settings(mode,model_name,data_name)
            args.model_name=backend_model_name




            abstracs_flag=['flagged_ents',]
            args.heads=list(('head_3',),)
            args.batch_size=24
            args.model_to_train=model_to_train#
            args.h3_embed_size=512
            args.epcoh_n=3
            if args.device==torch.device('mps'):
                    args.batch_size=4
            print('args.heads',args.heads)
            print('args.batch_size',args.batch_size)
            args.experiment_no='conceptqa'
            train_loader,dev_loader_b=rc_data(args)
            print('args.experiment_no',args.experiment_no)
            DATA={'train':train_loader,'dev':dev_loader_b}

            for a in abstracs_flag:
                args.abstract=a
                args.save=True
                args.experiment=True
                args.save=True
                args.experiment=True
                args.load_from_checkpoint=True if mode=='eval' else False
                print('a',a)
                print('mode',mode)#conceptqa_easy # conceptqa_hard
                args['hard']='conceptqa_easy'
                args['route_or_baseline']='route'
                print('args.hard',args.hard)
                print('args.route_or_baseline',args.route_or_baseline)
                georoc_train_eval(data_name,mode=mode,exp_args=args,DATA=DATA)


        elif str(experiment)=='wordanalogy':


            import os
            #os.system("python main.py  --task preprocess  --data wordanalogy")
            args.data_type=data_name
            #args.model_name='roberta-large'
            #args.model_name='bert-large-uncased'
            args.heads=list(('head_3',))
            args.save=False
            args.experiment=True
            #args.h3_embed_size=768
            if args.device==torch.device('mps'):
                args.batch_size=2
            print('test')
            L=['classification_head_train','baseline','train_route','baseline_train','route_train_head']

            L=['sentence_route_train',]#sentence_route
            L=['sentence_route',]#sentence_route
            wordanalogy_train_data=['EVALution',]
            #wordanalogy_train_data=['wikidata','semeval_2012','analogykb',]#['sre',]#['semeval_data',]
            #wordanalogy_train_data=['ekar','RS','google','bats','u4','u2','sat']
            wordanalogy_test_data=['ekar','RS','scan','google','bats','u4','u2','sat','special']
        
            # wordanalogy_train_data=['u4','u2','sat','ekar','google']
            # wordanalogy_test_data=['u4','u2','sat','ekar','google']
            #t_rex_relational_similarity
            #nell_relational_similarity
        

            wordanalogy_train_data=['semeval_2012','analogykb','wikidata','RS']
            wordanalogy_train_data=['RS','semeval_2012_easy','analogykb_easy','wikidata_easy','t_rex_relational_similarity','nell_relational_similarity']


            wordanalogy_test_data=['wikidata_hard','wikidata_easy','ekar','google','bats','u4','u2','sat','special']
            wordanalogy_test_data=['analogykb_easy','analogykb_hard',\
            'wikidata_easy','wikidata_hard','semeval_2012_easy','semeval_2012_hard','ekar','google','bats','u4','u2','sat']
            wordanalogy_train_data=['wikidata_easy','RS',]
            wordanalogy_test_data=['semeval_2012_easy','semeval_2012_hard','analogykb_easy','analogykb_hard','wikidata_easy','wikidata_hard']
            wordanalogy_test_data=['wikidata_easy','wikidata_hard','ekar','google','bats','u4','u2','sat','special','scan','RS','google_hard']
            #wordanalogy_test_data=['wikidata_easy','wikidata_hard','analogykb_easy','analogykb_hard','semeval_2012_easy','semeval_2012_hard']
            wordanalogy_train_data=['wikidata_easy','semeval_2012_relbert','RS','EVALution_easy',]

            wordanalogy_test_data=['EVALution_easy','EVALution_hard']

            wordanalogy_train_data=['scan',]
            #names=['BLESS','CogALexV','EVALution','ROOT09','KandH_plus_N'], wikidata , conll

            
            
            exp=0
            if exp==0:
                    for m in L:
                        model_name=args.model_name
                        model_name=backend_model_name#'roberta-large'
               
                        args=get_settings(mode,model_name,data_name)
                        args.similarity_measure='offset'#pairwise_cosine_sim#offset
                        args.heads=['head_3',]
                        args.experiment_no='wordanalogy'
                        if model_name=='bert-base-uncased':
                            args.embed_size=768
                            args.lr=1e-5
                        else:
                            args.lr=0.5e-5

                        args.wordanalogy_model=m
                        args.model_to_train=model_to_train#'wordanalogy_re_model'#
                        if args.device==torch.device('mps'):
                            args.batch_size=2
                        else:
                            args.batch_size=24
             
                        args.wordanalogy_train_data=wordanalogy_train_data
                        args.wordanalogy_test_data=wordanalogy_test_data
                        args.load_from_checkpoint=True
             
                        args.epcoh_n=10
                                          
                        args.save=False
                        args.fin_tune=False 
                        if 'route' in args.wordanalogy_model:
                            test=False
                            if test:
                                args.heads=['head_3',]
                                args.h3_embed_size=512
                                #args.h3_embed_size=352#512
                                file_name='essential_files/wordanalogyrel_dic.json'
                                word_analogyrel_dic= json.load(open(file_name))['rel_dic']
                                args.n_class=352#len(word_analogyrel_dic)
                                args.load_from_checkpoint=True
                            else:
                                args.h3_embed_size=512
                                file_name='essential_files/wordanalogyrel_dic_sre.json'
                                word_analogyrel_dic= json.load(open(file_name))['rel_dic']
                                args.n_class=len(word_analogyrel_dic)
                          
                        
                        if m in 'classification_train_head':
                            args.h3_embed_size=2

                        
                        ######################
                        print(',data_,m',m)
                        print('fin_tune',args.fin_tune)
                        print(' args.wordanalogy_model', args.wordanalogy_model)
                        print('batch_size',args.batch_size)
                        print('args.wordanalogy_test_data',args.wordanalogy_test_data)
                        print('args.wordanalogy_train_data',args.wordanalogy_train_data)
                        print('args.n_class',args.n_class)

                        #######################


                        train_loader,dev_loader_b=rc_data(args)
                        DATA={'train':train_loader,'dev':dev_loader_b}
                


                        if 'train' in m:
                            args.train=='train'
                            model=georoc_train_eval(data_name,mode='train',exp_args=args,DATA=DATA)
                        else:
                            args.train=='eval'

                            model=georoc_train_eval(data_name,mode='eval',exp_args=args,DATA=DATA)

                        
                        #################################################
                        print('fine tunning ')
                        print('***#######')
                        #exit()
                        #evluate_on_conceptqa(args,model,m)



                        #continue
                 
                            #exit()
                        #do_table_6(model,args)
                        #exit()



# def get_conceptqa_questions(abstract_spaces_dic,n_size=50):
#     from itertools import permutations

#     heat_map=[]
#     min_n=1
#     max_n=0
#     label_names=[]
#     Questions=[]
#     keys=list(abstract_spaces_dic.keys())
#     random.shuffle(keys)
#     for ki,k in enumerate(keys):
#         population=abstract_spaces_dic[k]
#         L=[si for si in range(len(population))]
#         N=n_size if len(L)>n_size else len(L)
#         # if N<n_size:
#         #     continue
#         s=random.sample(L,N)
        
#         perm = permutations(s, 2)
#         selected=[(population[t[0]],population[t[1]]  ) for t in perm]
#         pairs=selected#[(selected[i:i+2]) for i in range(0,len(selected),2)]
#         for p in pairs:
#             #print('p',p)
#             # exit()

#             if len(p)<2:
#                 continue
#             q,answer=p[0],p[1]
#             q,q_related=q[0],q[1]
#             answer=answer[0]
#             if q==answer:
#                 continue
#             ###
#             choice_n=5
#             choices=[q_related,]
#             choices=[]
#             for i in range(4):
#                 population_=list(abstract_spaces_dic.keys())
#                 L=[si for si in range(len(population_))]
#                 N=1 if len(L)>1 else len(L)

#                 s=random.sample(L,N)[0]
#                 selected_key=population_[s]
#                 if k==selected_key:
#                     continue

#                 population2=abstract_spaces_dic[selected_key]
#                 L2=[si for si in range(len(population2))]
#                 N2=1 if len(L2)>1 else len(L2)
#                 if len(L2)<1:
#                     continue
#                 s2=random.sample(L2,N2)[0]
#                 selected2=population2[s2] 
#                 selected2=selected2[0]
#                 #for w2 in selected2:
#                 #print('selected',selected)
#                 # print('selected2',selected2)
#                 # exit()
#                 choices.append(selected2)
#             #choices.append(answer)
#             if len(choices)<4:
#                 continue
#             random.shuffle(choices)
#             item={'q':q,'answer':answer,'choices':choices,'key':k}
#             Questions.append(item)
#             #print('item',item)
#     #exit()
#     return Questions
# @torch.no_grad
# def get_embedding(model,word):
#     word=word
#     tokenized_text=model.tokenizer.tokenize(word)
#     w_idx = model.tokenizer.convert_tokens_to_ids(tokenized_text)
#     w_mask = [1] * len(w_idx)
#     w_idx=torch.tensor(w_idx).long().unsqueeze(0).to(model.device)
#     w_mask=torch.ones_like(w_idx)
#     #print('w_idx',w_idx)
#     #print('w_mask',w_mask)
#     #.transformer(input_ids=ab,attention_mask=ab_mask).hidden_states
#     w_e=model.transformer(input_ids=w_idx,attention_mask=w_mask).hidden_states
#     #print('w_e',w_e)
#     w_e=w_e[-1]
#     w_e=torch.sum(w_e,1)
#     return w_e
# @torch.no_grad
# def get_sim_(w1,w2,model,sim_f='cosine_similarity'):
#     cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
#     e_w1=get_embedding(model,w1)#model.get_embedding(model,w1,model_name)
#     e_w2=get_embedding(model,w2)#model.get_embedding(model,w2,model_name)
  
#     if sim_f=='cosine_similarity':
#         s=cos(e_w1 ,e_w2 )
#     elif sim_f=='euclidean_distance':
#         s=torch.nn.functional.pairwise_distance(e_w1, e_w2)
#     else:
#         v1,v2=e_w1,e_w2
#         n1=np.linalg.norm(v1)
#         n2=np.linalg.norm(v2)
#         v=v1.dot(v2.T)
#         s=v/(n1*n2)
#         s=s.cpu().detach().item()


#     return s,e_w1,e_w2
@torch.no_grad
def evluate_on_conceptqa(args,MODEL):

    # abstract_spaces_dic_wikidata=get_abstract_spaces_dic()
    
    # abstract_spaces_dic_google=get_google_abstract_space_dic()
    # abstract_spaces_dic_semeval=get_google_abstract_space_dic(semeval_or_google='semeval_2012')



    model=MODEL.transformer
    tokenizer=MODEL.tokenizer
    modelname=args.model_name
    model=LLMs(modelname,(model,tokenizer))




    sim_f='cosine_similarity'
    do_table_6(modelname,model=model)

    # print('wikidata')
    # plot_semantic_space_questions(model,model_name,abstract_spaces_dic_wikidata,sim_f=sim_f,n_size=20)
    # print('semeval')
    # plot_semantic_space_questions(model,model_name,abstract_spaces_dic_semeval,sim_f=sim_f,n_size=5)
    # print('google')
    # plot_semantic_space_questions(model,model_name,abstract_spaces_dic_google,sim_f=sim_f,n_size=5)

    #plot_semantic_space_vs_other(modelname,model,abstract_spaces_dic,every_n=15,simple=False)







    # def get_embedding(model,word):
    #     word=word
    #     tokenized_text=model.tokenizer.tokenize(word)
    #     w_idx = model.tokenizer.convert_tokens_to_ids(tokenized_text)
    #     w_mask = [1] * len(w_idx)
    #     w_idx=torch.tensor(w_idx).long().unsqueeze(0).to(model.device)
    #     w_mask=torch.ones_like(w_idx)
    #     #print('w_idx',w_idx)
    #     #print('w_mask',w_mask)
    #     #.transformer(input_ids=ab,attention_mask=ab_mask).hidden_states
    #     w_e=model.transformer(input_ids=w_idx,attention_mask=w_mask).hidden_states
    #     #print('w_e',w_e)
    #     w_e=w_e[-1]
    #     w_e=torch.sum(w_e,1)
    #     return w_e
    # def get_sim_(w1,w2,model,sim_f='cosine_similarity'):
    #     cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    #     e_w1=get_embedding(model,w1)#model.get_embedding(model,w1,model_name)
    #     e_w2=get_embedding(model,w2)#model.get_embedding(model,w2,model_name)
    #     s=cos(e_w1 ,e_w2 )
    #     s=s.cpu().detach().item()


    #     return s,e_w1,e_w2

  

        
    # colors_dic = {
    # 'country':"royalblue",
    # 'capital':'peru',
    # 'currency':"red", 
    # 'city':"darkgreen",
    # 'state':"cyan",
    # 'male':"orange",
    # 'female':"gray",
    # 'verb':"blue",
    # 'preterite':"magenta",
    # 'verb':'darkred',
    # 'plural':'black',
    # 'adj':'olive',
    # 'participle':'pink',
    # 'nationality':'purple',
    # 'superlative':'teal',
    # 'antonym':'violet',
    # 'adverb':'brown',
    # 'comparative':'darkcyan',
    # 'noun':'bisque'
    # }
    # file='essential_files/simple_analogy.json'
    # data= json.load(open(file))['data']
    # ###
    # # file='essential_files/localdatasets/simple_analogy_semeval.json'
    # # data= json.load(open(file))['data']['test']



    # Labels=[]
    # All_Wv=[]
    # cats_all=[]
    # model_name='baseline'
    # abstract_spaces_dic={}
    # for k in data:

    #     d=data[k]
    #     for t in d:
    #         # print(t)
    #         # print('k',k)
    #         # print('+++')
    #         # exit()

    #         w1=t['a']
    #         w2=t['b']
    #         c1=t['c1']
    #         c2=t['c2']

    #         e_w1=get_embedding(model,w1)#model.get_embedding(model,w1,model_name)
    #         e_w2=get_embedding(model,w2)#model.get_embedding(model,w2,model_name)


            
            



 
    #         All_Wv.append(e_w1.cpu().detach().numpy())
    #         All_Wv.append(e_w2.cpu().detach().numpy())
    #         Labels.append(w1)
    #         Labels.append(w2)

    #         cats_all.append(c1)
    #         cats_all.append(c2)
    #         ##
    #         head_a=c1
    #         tail_a=c2
    #                     ###
    #         if head_a in abstract_spaces_dic.keys():
    #             if w1!=head_a:
    #                 abstract_spaces_dic[head_a].append((w1,w2))
    #             #abstract_spaces_dic[head_a]=list(set(abstract_spaces_dic[head_a]))
    #         else:
    #             abstract_spaces_dic[head_a]=[]
    #             if w1!=head_a:
    #                 abstract_spaces_dic[head_a].append((w1,w2))
    #             #abstract_spaces_dic[head_a]=list(set(abstract_spaces_dic[head_a]))


    #         if tail_a in abstract_spaces_dic.keys():
    #             if w2!=tail_a:
    #                 abstract_spaces_dic[tail_a].append((w2,w1))
    #             #abstract_spaces_dic[tail_a]=list(set(abstract_spaces_dic[tail_a]))
    #         else:
    #             abstract_spaces_dic[tail_a]=[]
    #             if w2!=tail_a:
    #                 abstract_spaces_dic[tail_a].append((w2,w1))
    #             #abstract_spaces_dic[tail_a]=list(set(abstract_spaces_dic[tail_a]))
    # ###
    # # plot_semantic_space_vs_other(m,model,abstract_spaces_dic,every_n=1,simple=True)
    # # exit()
    # Questions=get_conceptqa_questions(abstract_spaces_dic,n_size=10)
    # correct=0
    # not_correct=0
    # space_wise_acc={}
    # space_average_of_similarity={}
    # sim_f='cosine_similarity'
    # sim_f='euclidean_distance'
    # sim_f='dot_product'
    # for t in Questions:
    #     #print('t',t)

    #     q=t['q']
    #     choices=t['choices']
    #     answer=t['answer']
    #     key=t['key']
    #     if key not in space_wise_acc.keys():
    #         space_wise_acc[key]={'correct':0,'not_correct':0}
    #         space_average_of_similarity[key]=[]

    #     #sim_answer=get_sim_(model,q,answer,model_name,sim_f=sim_f)
    #     sim_answer,e_w1,e_w2=get_sim_(q,answer,model)

    #     space_average_of_similarity[key].append(sim_answer)



    #     choice_sim=[]
    #     for c in choices:
    #         #sim=get_sim_(q,c,model,sim_f=sim_f)
    #         sim,_,c_e=get_sim_(q,c,model)
    #         choice_sim.append(sim)
    #     if sim_f in ['cosine_similarity','dot_product']:
    #         max_sim_choice=max(choice_sim)
    #         if sim_answer>max_sim_choice:
    #             correct=correct+1
    #             space_wise_acc[key]['correct']=space_wise_acc[key]['correct']+1
    #         else:
    #             not_correct=not_correct+1
    #             space_wise_acc[key]['not_correct']=space_wise_acc[key]['not_correct']+1
    #     else:
    #         min_sim_choice=min(choice_sim)
    #         if sim_answer>min_sim_choice:
    #             not_correct=not_correct+1
    #             space_wise_acc[key]['not_correct']=space_wise_acc[key]['not_correct']+1

    #         else:

    #             correct=correct+1
    #             space_wise_acc[key]['correct']=space_wise_acc[key]['correct']+1
    # print('correct',correct)
    # print('not_correct',not_correct)
    # print('acc',correct/(correct+not_correct))
    # overall_acc=correct/(correct+not_correct)
    # overall_correct=correct
    # overall_not_correct=not_correct
    # print('overall_acc',overall_acc)
    # print('overall_correct',overall_correct)
    # print('overall_not_correct',overall_not_correct)
    # #print('space_wise_acc',space_wise_acc)
    # all_acc={}
    # for k in space_wise_acc.keys():
    #     t=space_wise_acc[k]
    #     correct=t['correct']
    #     not_correct=t['not_correct']
    #     acc=correct/(correct+not_correct)
    #     all_acc[k]=(acc,correct,not_correct)

    # ##
    # sorted_x = sorted(all_acc.items(), key=lambda kv: kv[1])
    # print('sorted_x',sorted_x)

    ###

    # import numpy as np
    # from sklearn.decomposition import PCA
    # import pandas as pd
    # import matplotlib.pyplot as plt

    # pca = PCA(n_components=100)

    # arr = np.concatenate(All_Wv, axis=0)
    # x_2d = pca.fit_transform(arr)
    
    # Ws_label=Labels
    # arrows=zip(Ws_label,x_2d)
    # j=0
    # colors = ["blue", "red", "green", "orange", "blue",'yellow']#+["blue", "gray", "gray", "gray", "gray",'green']
    # plt.figure(figsize=(34, 34))

    # arrows=zip(Ws_label,x_2d[:,:])
    # j=0
    # x_max=-1
    # x_min=1
    # y_max=-1
    # y_min=1
    # for (label, vec)  in arrows:
    #     # plt.arrow(origin[0], origin[1], vec[0], vec[1],
    #     #           head_width=0.02, length_includes_head=True,
    #     #           color=color, linewidth=2)
    #     vec=vec#*4
    #     j=j+1
    #     if (j+1)%2!=0:
    #         pass
    #         #continue



    #     cat=cats_all[j-1]
    #     clr=colors_dic[cat]
    #     #print('cat',cat,label)
    #     # if cat not in selected_cat:
    #     #     continue

    #     plt.text(vec[0], vec[1]+0.0002, label, fontsize=12)

    #     if vec[0]< x_min:
    #         x_min=vec[0]
    #     if vec[0]> x_max:
    #         x_max=vec[0]
    #     ##
    #     if vec[1]< y_min:
    #         y_min=vec[1]
    #     if vec[1]> y_max:
    #         y_max=vec[1]


    #     plt.scatter(vec[0], vec[1],color=clr, zorder=5,s=50)



    # # Axis formatting
    # step=100
    # widthx=(x_max-x_min)/step
    # widthy=(y_max-y_min)/step


    # plt.axhline(0, color="black", linewidth=widthx)
    # plt.axvline(0, color="black", linewidth=widthy)
    # plt.xlim(x_min-0.00005, x_max+0.00005)
    # plt.ylim(y_min-0.00005, y_max+0.00005)
    # plt.gca().set_aspect("equal", adjustable="box")
    # plt.title("offset Visualized with PCA")

    # plt.savefig('essential_files/pca/pca'+str(m)+'orig'+'.png')
    # plt.close()




# def plot_semantic_space_vs_other(model_name,model,abstract_spaces_dic,every_n=15,simple=False):


#     heat_map=[]
#     min_n=1
#     max_n=0
#     label_names=[]
#     spaces_list=list(abstract_spaces_dic.keys())
#     #random.shuffle(spaces_list)
#     total=1000
#     count=0
#     for ki,k in enumerate(spaces_list):

#         # if len(k)>20: 
#         #     continue
#         if ki%every_n!=0:
#             continue
#         population=abstract_spaces_dic[k]
#         # if len(population)<20:
#         #     continue

#         L=[si for si in range(len(population))]
#         N=20 if len(L)>20 else len(L)
#         s=random.sample(L,N)
#         selected=[population[j_] for j_ in s]
#         map_={}
#         label_names.append(k)
#         for w1 in selected:
#             w1=w1[0]
#             for ki2,k2 in enumerate(spaces_list):

#                 # if len(k2)>20: 
#                 #     continue
#                 if ki2%every_n!=0:
#                     continue
      
#                 population2=abstract_spaces_dic[k2]
#                 # if len(population2)<20:
#                 #     continue

#                 L2=[si for si in range(len(population2))]
#                 N2=20 if len(L2)>20 else len(L2)
#                 s2=random.sample(L2,N2)
#                 selected2=[population2[j_] for j_ in s2]
#                 for w2 in selected2:
#                     w2=w2[0]
     
#                     #sim=get_offset_sim(model,w1,w2,w1_,w2_,model_name)

#                     #sim=get_sim(model,w1,w2,model_name)
#                     sim,e_w1,e_w2=get_sim_(w1,w2,model)

#                     sim=sim+(1-sim)/2#1/math.exp(200*(-sim))
#                     count=count+1
#                     sim=sim.item()
#                     # print('sim',sim)
#                     # exit()

#                     key=str(k)+'#'+str(k2)
#                     if key in map_.keys():
#                         map_[key].append(sim)
#                     else:
#                         map_[key]=[]
#                         map_[key].append(sim)
       
#         all_sim=[sum(map_[t])/len(map_[t]) for t in map_.keys()]
#         min_n_=min(all_sim)
#         max_n_=max(all_sim)
#         if min_n> min_n_:
#             min_n=min_n_

#         if max_n< max_n_:
#             max_n=max_n_
        
#         heat_map.append(all_sim)

#     print('label_names',len(label_names))
#     print('heat_map',len(heat_map))

#     ###
#     import pandas as pd
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     import numpy as np



#     df= pd.DataFrame(heat_map, index=label_names, columns=label_names)

#     # --- 2. Create Custom Annotation Dataframe ---

#     # Initialize a DataFrame of the same shape, filled with empty strings
#     annot_df = pd.DataFrame(
#         np.full_like(df.values, '', dtype=object), 
#         index=df.index, 
#         columns=df.columns
#     )

#     # # Place your custom labels in the desired cells (e.g., at row 'prophetnet', col 'bertB')
#     # annot_df.loc['prophetnet', 'bertB'] = 'MAX!'
#     # annot_df.loc['t5L', 'robertaB'] = 'Min'

#     # Optional: Keep the original numerical values for all other cells
#     # for i in range(len(label_names)):
#     #     for j in range(len(label_names)):
#     #         # If the cell is not one of the custom ones, use the formatted numerical value
#     #         #if annot_df.iloc[i, j] == '':
#     #         annot_df.iloc[i, j] = f"{df.iloc[i, j]:.2f}"

#     # --- 3. Generate the Plot with Custom Annotations ---
#     if simple:
#         plt.figure(figsize=(4, 4))
#     else:
#         plt.figure(figsize=(12, 12))
#     cmap = sns.color_palette("crest", as_cmap=True)

#     sns.heatmap(
#         df,
#         annot=annot_df,          # Pass the custom annotation DataFrame here
#         fmt='s',                 # Use 's' to indicate we are passing strings, not float format
#         cmap=cmap,               
#         linewidths=0.5,          
#         cbar=True,               
#         vmin=min_n,               
#         vmax=max_n,
#         # Optional: Customize the appearance of the labels
#         annot_kws={"fontsize": 15, "fontweight": "bold"} 
#     )
#     if simple==False:
#         plt.title('Similarity between Different Category(Semantic Spaces) of Words', fontsize=15, pad=20)
#     plt.xticks(rotation=90)
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     #plt.show()
#     plt.savefig('images/sim_dis/semantic_spaces_simple'+str(model_name)+'hard.png')
#     plt.close()


