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
from Additional_Experiments import *
import json
import pandas as pd
import csv
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt##
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

from Train_Eval import *
from analogy_util import _get_pretrained_transformer3







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
            'evaluate_gpt'
            ]

            experiment_name='evaluate_gpt'

            plots_and_custom_experiments(experiment_name,args=args)

        elif str(experiment)=='sentential_re_paper': #re

            expert_head=('head_1','head_3','head_2','head_conditional')
            h3=('head_3',)

            h=h3
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

            train_loader,dev_loader_b=rc_data(args)
            print('len',len(train_loader))
            print('len',len(dev_loader_b))
            DATA={'train':train_loader,'dev':dev_loader_b}

            georoc_train_eval(data_name,mode=mode,exp_args=args,DATA=DATA)


        elif str(experiment)=='lexical_offset':# wodanalogy_re_model


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

            h3=('head_conditional',)

            h=h3
            print('h',h)

            args=get_settings(mode,model_name,data_name)
            args.heads=list(h)
            args.h3_embed_size=512
            args.abstract='flagged_ents'
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

            train_loader,dev_loader_b=rc_data(args)

            DATA={'train':train_loader,'dev':dev_loader_b}

            georoc_train_eval(data_name,mode=mode,exp_args=args,DATA=DATA)

        elif experiment=='conceptqa':
            h3=('head_3',)

            h=h3
            print('h',h)

            args=get_settings(mode,model_name,data_name)
            args.heads=list(h)
            args.h3_embed_size=512
            args.abstract='flagged_ents'#'flagged_ents'
            args.experiment_no='conceptqa'
            args.epcoh_n=2
            args.model_to_train=model_to_train#
            print('model_to_train',model_to_train)
            args.h3_embed_size=512
 
            args.n_class
            print('args.n_class',args.n_class)
            args['route_or_baseline']='route'
            args['hard']='conceptqa_easy'


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



        elif experiment=='mtcqa':
            mtqa_experiment_names=['conceptqa','pretraining_with_easy_hard']
            name='pretraining_with_easy_hard'
            run_mc(name)

        elif str(experiment)=='wordanalogy':

            L=['classification_head_train','baseline','train_route','baseline_train','route_train_head']

            L=['sentence_route_train',]
            L=['sentence_route',]




            wordanalogy_test_data=['wikidata_hard','wikidata_easy','ekar','google','bats','u4','u2','sat','special']
            wordanalogy_test_data=['analogykb_easy','analogykb_hard',\
            'wikidata_easy','wikidata_hard','semeval_2012_easy','semeval_2012_hard','ekar','google','bats','u4','u2','sat']
            wordanalogy_train_data=['wikidata_easy','RS',]
            wordanalogy_test_data=['semeval_2012_easy','semeval_2012_hard','analogykb_easy','analogykb_hard','wikidata_easy','wikidata_hard']
            
            wordanalogy_test_data=['wikidata_easy','wikidata_hard','ekar','google','bats','u4','u2','sat','special','scan','RS','google_hard']
            wordanalogy_train_data=['wikidata_easy','semeval_2012_relbert','RS','EVALution_easy',]

            wordanalogy_test_data=['special','sat']

            wordanalogy_train_data=['scan',]
 
            
            exp=0
            if exp==0:
                    for m in L:
                        model_name=args.model_name
                        model_name=backend_model_name
               
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
                        args.model_to_train=model_to_train
                        if args.device==torch.device('mps'):
                            args.batch_size=2
                        else:
                            args.batch_size=24
             
                        args.wordanalogy_train_data=wordanalogy_train_data
                        args.wordanalogy_test_data=wordanalogy_test_data
            
                        args.epcoh_n=4
                                          
                        args.save=False
                        args.fin_tune=False 
                        if 'route' in args.wordanalogy_model:
                            args.heads=['head_3',]
                            args.h3_embed_size=512

                            file_name='essential_files/wordanalogyrel_dic.json'
                            word_analogyrel_dic= json.load(open(file_name))['rel_dic']
                            args.n_class=352
                            args.load_from_checkpoint=True
   
                          
                        
                        if m in 'classification_train_head':
                            args.h3_embed_size=2

                        train_loader,dev_loader_b=rc_data(args)
                        DATA={'train':train_loader,'dev':dev_loader_b}
                        if 'train' in m:
                            args.train=='train'
                            model=georoc_train_eval(data_name,mode='train',exp_args=args,DATA=DATA)
                        else:
                            args.train=='eval'

                            model=georoc_train_eval(data_name,mode='eval',exp_args=args,DATA=DATA)

                     