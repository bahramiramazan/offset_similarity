
PATH = '.'#Add you directory here
import sys
sys.path.append(PATH)
import torch.nn as nn


import torch
import random
import numpy as np
model_name = 'bert-base-uncased'

import scipy
####
import json
from sklearn.metrics.pairwise import cosine_similarity

#PATH = Add you directory here
import sys
sys.path.append(PATH)
#####

import torch.nn as nn
import torch
import math
from torch.nn.functional import normalize
##
from transformers import XLNetTokenizer
from transformers import AutoTokenizer, LongT5Model
import random
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import ujson as json
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForImageClassification
import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel
from analogy_util import _get_pretrained_transformer3

import copy
from Experiment_Util import *

import numpy as np
from gpt_util import *
#########################################################################
all_models_names=['prophetnet','opt','gpt2','t5-large','t5-small','roberta-base','roberta-large','bert-large-uncased','bert-base-uncased']




def semantic_space_experiment(experiment_name,abstract,model_name,args):

    if 'conceptqa' in experiment_name:

        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

        model=get_model(model_name)

        abstract_spaces_dic_wikidata=get_abstract_spaces_dic_wikidata()



        abstract_spaces_dic_semeval_2012=get_google_abstract_space_dic(semeval_or_google='semeval_2012')
        abstract_spaces_dic_wikidata_easy=get_google_abstract_space_dic(semeval_or_google='wikidata')

        sim_f='cosine_similarity'
        #sim_f='distance'

        abstract_spaces_dic=abstract_spaces_dic_semeval_2012
        print('conceptQA Easy',model_name)
        easy=True
        _=plot_semantic_space_questions(model,model_name,abstract_spaces_dic,sim_f=sim_f,n_size=4,easy=easy)


        print('conceptQA Hard',model_name)
        easy=False
        _=plot_semantic_space_questions(model,model_name,abstract_spaces_dic,sim_f=sim_f,n_size=4,easy=easy)


    if 'basian_analysis' in experiment_name:
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        model=get_model(model_name)
        abstract_spaces_dic_semeval_2012=get_google_abstract_space_dic(semeval_or_google='semeval_2012')
        abstract_spaces_dic_wikidata_easy=get_google_abstract_space_dic(semeval_or_google='wikidata')

        sim_f='cosine_similarity'
        #sim_f='distance'

        abstract_spaces_dic=abstract_spaces_dic_wikidata_easy
        easy=False
        not_corrct_terms=plot_semantic_space_questions(model,model_name,abstract_spaces_dic,sim_f=sim_f,n_size=4,easy=easy)
     

        modelname=model_name
        dn='wikidata_easy'
        solve_analogies(modelname,dn,not_corrct_terms,bayesian_analysis=True)
        dn='wikidata_hard'
        solve_analogies(modelname,dn,not_corrct_terms,bayesian_analysis=True)
        print('end#########')

    elif 'plot_semantic_space_questions' in  experiment_name:
        sim_f='cosine_similarity'
        model=get_model(model_name)
        abstract_spaces_dic=get_google_abstract_space_dic(semeval_or_google='semeval_2012')
        plot_semantic_space_questions(model,model_name,abstract_spaces_dic,sim_f=sim_f,n_size=10,plot=True)

    elif 'plot_space_based_similarity' in experiment_name:
        model_name='fasttext'
        model=get_model(model_name)
        #plot_semantic_space_vs_other(model,model_name,abstract_spaces_dic,every_n=5,simple=False)
        #abstract_spaces_dic=get_abstract_spaces_dic_wikidata()
        abstract_spaces_dic=get_google_abstract_space_dic(semeval_or_google='semeval_2012')
        plot_semantic_space_vs_other(model,model_name,abstract_spaces_dic,every_n=2,min_space_len=5)

    elif 'check_r_head_tail_concepts_types' in experiment_name:
        all_related=get_abstract_spaces_dic_wikidata(train_or_dev='dev',f='all_related')
        all_related=get_abstract_spaces_dic_wikidata(f='all_related')
        check_r_head_tail_concepts(all_related)
    

@torch.no_grad()
def plots_and_custom_experiments(experiment_name,args=None):


    if experiment_name=='do_table_6':
        model_name='fasttext'
        do_table_6(model_name,model=None)

    if  'cat2' in experiment_name:
        abstract='original'

        #abstract='random'
        model_name='bert-large-uncased'
        model_name='bert-large-uncased'
        #model_name='fasttext'
        #model_name='prophetnet'
        model_name='opt'
        # model_name='gpt2'
        for model_name in ['fasttext',]:#['bert-large-uncased','roberta-large','fasttext','gpt2','opt','prophetnet']:
            #for abstract in ['orginal','random','random_word']:
            semantic_space_experiment(experiment_name,abstract,model_name,args)


    else:
        if 'evaluate_gpt' in experiment_name:
            evaluate_chatgpt()


        if 'plot_sim' in experiment_name:
                #all_related,all_abstracts,all_words,f='all_related'
                 all_related,all_abstracts,all_words=get_abstract_spaces_dic_wikidata(f='all_related')
                 ##for abstract in ['orginal','random','random_word']:
                 abstract='random'
                 model_name=''

                 plot_sim_with_abs(abstract,model_name,all_abstracts,all_words)

                 all_related=get_google_abstract_space_dic(semeval_or_google='semeval_2012',f='all_related')
                 same_rel_sim(model_name,all_related)

        if  'print_sample' in experiment_name:
            dn='wikidata_easy_plus'
            print_sample(dn)


        elif  'solve_analogies' in experiment_name:

            for model_name in ['fasttext','gpt2','roberta-large','bert-large-uncased','prophetnet','t5-large','opt']:
                modelname=model_name
                for dn in ['scan',]:
                    data_name=dn
                    semantic_sim=False
                    easy_hard='equ'
                    not_corrct_terms={'correct':{},'not_correct':{}}
                    #sim_f#pairwise_sim#offset_cosine_similarity#distance
                    solve_analogies(modelname,dn,not_corrct_terms,sim_f='offset_cosine_similarity',bayesian_analysis=False)


        elif 'interagreement_between_models_on_data' in experiment_name or 'interagreement_between_models' in experiment_name:

            if 'interagreement_between_models_on_data' in experiment_name:
                models_name=('fasttext','t5-large')
                m1=models_name[0]
                m2=models_name[1]
                        
                for dn in ['u4','sat','ekar','u2', 'u4','google','bats','semeval_2012_easy']:
                    l=interagreement_between_models(m1,m2,dn)
                    print(l)
                    exit()
            elif  'interagreement_between_models' in experiment_name:
                interagreement_between_models_l=[]
                #data=args.data
                model_names=['fasttext','prophetnet','opt','gpt2','t5-large','t5-small','roberta-base','roberta-large','bert-large-uncased','bert-base-uncased']

                for m1 in model_names:
                    for m2 in model_names:
                        dn='sat'
                        l=interagreement_between_models(m1,m2,dn)
                        interagreement_between_models_l.append(l)
                ##
                models = [
                    'fasttext', 'prophetnet', 'opt', 'gpt2', 't5L',
                    't5S', 'robertaB', 'robertaL', 'bertL', 'bertB'
                ]
                plot_agreement_heat_map(interagreement_between_models_l,models)
                


        elif 'plot_permutations_dist' in  experiment_name:  
            #['sat','ekar','u2', 'u4','google','wikidata','bats','semeval_2012']
            #['glove','word2vec']
      
            file='essential_files/histogramfile.json'
            histogramfile= json.load(open(file))['histogramfile']
            item={'sat':[],'ekar':[],'u2':[],'u4':[],'google':[],'bats':[],'wikidata':[],'analogykb':[]}
            histogramfile={'perpendicular':copy.deepcopy(item),'valid':copy.deepcopy(item),'crossed':copy.deepcopy(item),'negative':copy.deepcopy(item)}

            #histogramfile_={'perpendicular':item,'valid':item}

            for valid_p in ['perpendicular','valid','crossed','negative']:
                model_name='fasttext'
                for dn in ['sat','ekar','u2', 'u4','google','bats','semeval_2012_easy']:
                    get_perm_similarities(histogramfile,dn,model_name,valid_p)
     
            for valid_p in ['perpendicular','valid','crossed','negative']:
                model_name='fasttext'
                for dn in ['sat','ekar','u2', 'u4','google','bats','semeval_2012_easy']:
                    data_name=dn
                    plot_permutations_dist(data_name)



