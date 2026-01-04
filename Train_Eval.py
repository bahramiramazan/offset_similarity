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
from Additional_Experiments import do_table_6
# from Multichoice_Model import run_mc
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
# from Abstraction import *
###

from Additional_Experiments import LLMs_RC
from Additional_Experiments import solve_analogies


from Additional_Experiments import LLMs

def get_constant_schedule(optimizer, last_epoch=-1):
    """ Create a schedule with a constant learning rate.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)



class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):


        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0




        mask = torch.nonzero(target.data == self.padding_idx)

        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y,norm,retain_graph=True,fl=False):
        x = self.generator(x)

        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        if fl:
            loss.backward()
            
        else:
            
            loss.backward(retain_graph=retain_graph)

        if self.opt is not None and fl:
            self.opt.step()
            self.opt.zero_grad()
        return loss.data * norm


confusion_matrix={}
def get_original_label(y,relations,batch,i):
    sentence_flagged_idxs=batch['sentence_flagged_idxs'][i,:]
    abstracted_ents_flagged_idxs=batch['abstracted_ents_flagged_idxs'][i,:]


    sentence=tokenizer.convert_ids_to_tokens(sentence_flagged_idxs)
    abstract=tokenizer.convert_ids_to_tokens(abstracted_ents_flagged_idxs)
    #############
    temp=y[i].item()
    temp=Relation_counter_tacerd[temp]
    temp=str(temp).upper()


    temp=id2kbid[temp]

    y_t=tacerd_rel[temp]


    temp=relations[i].item()
    temp=Relation_counter_tacerd[temp]
    temp=str(temp).upper()

    temp=id2kbid[temp]
    r_t=tacerd_rel[temp]
    t=str(y_t)+'-'+str(r_t)
    if t in confusion_matrix:
        confusion_matrix[t]=confusion_matrix[t]+1
    else:
        confusion_matrix[t]=1
    print('y',y_t)
    print('r',r_t)
    print('sentence',sentence)
    print('abstract',abstract)
    print('confusion_matrix',confusion_matrix)
    print('****')


@torch.no_grad
def conceptqa_eval_wordanalogy(args,model,abstract,batch_data,device):


   
    word_concept_similarity_category={'correct':{},'not_correct':{}}
    word_concept_similarity_category_={'correct':{},'not_correct':{}}
    file='evaluation_data/'+str(args.data_type)+'dev_eval.json'
    eval_data= json.load(open(file))['data']
    for di,data in enumerate(batch_data):

    
        correct=0
        not_correct=0
        #
        correct_=0
        not_correct_=0
        for i_d_,x in enumerate(tqdm(data)): 
            examples=x
            batch={}
            eval_epxeriment_data=None
          
            for key in examples.keys():
                batch[key]=examples[key].to(device) if key!='ids' else examples[key]

            answer_sim,a_sim,b_sim,c_sim,d_sim,e_sim= model(batch,args,eval_epxeriment_data=eval_epxeriment_data,eval=True)
            ###


            iD=examples['ids']

         
            for i in range(batch['q'].shape[0]):
                answer_sim_=answer_sim[i]
                a_sim_=a_sim[i].item() #if args.hard=='conceptqa_easy' else 0
                b_sim_=b_sim[i].item()
                c_sim_=c_sim[i].item()
                d_sim_=d_sim[i].item() #if args.hard=='conceptqa_hard' else 0
                e_sim_=e_sim[i].item() #
                ##
                #print('iD',iD)
                eval_d=eval_data[iD[i]]['supports']
                choices=eval_d['choices']
                q=eval_d['q']
                answer=eval_d['answer']
                max_sim_easy=max([b_sim_,c_sim_,d_sim_,e_sim_])
                max_sim_hard=max([a_sim_,b_sim_,d_sim_,e_sim_])
   
                if answer_sim_>max_sim_easy:
                    #print('correct')
                    correct=correct+1
                    ##
                    if q in word_concept_similarity_category['correct'].keys():
                        word_concept_similarity_category['correct'][q]=word_concept_similarity_category['correct'][q]+1
                    else:
                        word_concept_similarity_category['correct'][q]=1

                    if answer in word_concept_similarity_category['correct'].keys():
                        word_concept_similarity_category['correct'][answer]=word_concept_similarity_category['correct'][answer]+1
                    else:
                        word_concept_similarity_category['correct'][answer]=1
                else:
                    not_correct=not_correct+1
                    ###
                    if q in word_concept_similarity_category['not_correct'].keys():
                        word_concept_similarity_category['not_correct'][q]=word_concept_similarity_category['not_correct'][q]+1
                    else:
                        word_concept_similarity_category['not_correct'][q]=1

                    if answer in word_concept_similarity_category['not_correct'].keys():
                        word_concept_similarity_category['not_correct'][answer]=word_concept_similarity_category['not_correct'][answer]+1
                    else:
                        word_concept_similarity_category['not_correct'][answer]=1
                #####
                if answer_sim_>max_sim_hard:
                    #print('correct')
                    correct_=correct_+1
                    ##
                    if q in word_concept_similarity_category_['correct'].keys():
                        word_concept_similarity_category_['correct'][q]=word_concept_similarity_category_['correct'][q]+1
                    else:
                        word_concept_similarity_category_['correct'][q]=1

                    if answer in word_concept_similarity_category_['correct'].keys():
                        word_concept_similarity_category_['correct'][answer]=word_concept_similarity_category_['correct'][answer]+1
                    else:
                        word_concept_similarity_category_['correct'][answer]=1
                else:
                    not_correct_=not_correct_+1
                    ###
                    if q in word_concept_similarity_category_['not_correct'].keys():
                        word_concept_similarity_category_['not_correct'][q]=word_concept_similarity_category_['not_correct'][q]+1
                    else:
                        word_concept_similarity_category_['not_correct'][q]=1

                    if answer in word_concept_similarity_category_['not_correct'].keys():
                        word_concept_similarity_category_['not_correct'][answer]=word_concept_similarity_category_['not_correct'][answer]+1
                    else:
                        word_concept_similarity_category_['not_correct'][answer]=1
    #####

        print('correct',correct)
        print('not_correct',not_correct)
        acc=correct/(correct+not_correct)
        print('acc easy',acc)
        acc_hard=correct_/(correct_+not_correct_)
        print('acc hard',acc_hard)
        file='essential_files/conceptqa_TF_table.json'
        h_data={'conceptqa_TF_table':word_concept_similarity_category}
        with open(file, 'w') as fp:
            json.dump(h_data, fp)


        file='essential_files/conceptqa_TF_table_.json'
        h_data={'conceptqa_TF_table':word_concept_similarity_category_}
        with open(file, 'w') as fp:
            json.dump(h_data, fp)


        return acc







@torch.no_grad()    
def rc_eval_wordanalogy(args,model,abstract,batch_data,device,idx2word='none'):


    file_name='essential_files/word_analogy_types_dic.json'
    word_analogy_types_dic= json.load(open(file_name))['word_analogy_types_dic']
    new_w_dic={}
    for k in word_analogy_types_dic.keys():
        if k in args.wordanalogy_test_data:
            new_w_dic[k]=word_analogy_types_dic[k]
    word_analogy_types_dic=new_w_dic



    file='evaluation_data/'+str(args.data_type)+'dev_eval.json'
    eval_data= json.load(open(file))['data']


    predictions_all={}

    word_analogy_types_dic_rev = {y: x for x, y in word_analogy_types_dic.items()}

    types=list(word_analogy_types_dic.values())
    #################################################################

    ##############################################################

    for t in types:
        # if word_analogy_types_dic_rev[t] in args.wordanalogy_test_data:
        print('t',t)
        predictions_all[t]={}
    total_correct=0
    total_not_correct=0
    a_c_sim=[0.1,]
    b_d_sim=[0.1,]

    a_c_dist=[0.1,]
    b_d_dist=[0.1,]

    a_c_sim_n=[0.1,]
    b_d_sim_n=[0.1,]

    a_c_dist_n=[0.1,]
    b_d_dist_n=[0.1,]
    model.eval()
    for di,data in enumerate(batch_data):
        c_1=0
        c_total=0
        eval_epxeriment_data={}
        Similarity_in_Hidden_Layers={}
        temp_dic_layer={'positive':{'head':{'sim':{}, 'dist':{}},'tail':{'sim':{}, 'dist':{}} },'negative':{'head':{'sim':{}, 'dist':{}},'tail':{'sim':{}, 'dist':{}} } }
        for i_d_,x in enumerate(tqdm(data)):  
            examples=x
            y=examples['y']
            y0=examples['y0']
            ids=examples['ids']
            R=examples['r']

            #print('R',R)

            ########
            batch={}
            for key in examples.keys():
                batch[key]=examples[key].to(device) if key!='ids' else examples[key]
            ######################################
            ids=batch['ids']
            if args.wordanalogy_model not in 'classification_train_head' :
                x1,x2,loss,w3_w4_sim,w1_w3_sim,w2_w4_sim,w1_w3_dist,w2_w4_dist,similarity_in_hidden_layers = model(batch,args,eval_epxeriment_data=eval_epxeriment_data,eval=True)
                
                for No,iD in enumerate(ids):
                    eval_d=eval_data[iD]

                    # print('eval_d',eval_d)
                    # exit()


            


                    question_no,analogy_no=iD.split('-')
                    x1_i,x2_i=x1[No,:].unsqueeze(0),x2[No,:].unsqueeze(0)

                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                    similarity = cos(x1_i, x2_i)

                    # print('question_no',question_no)




                    #if (y[No].item()!=8) and  ('sentence' not in  args.wordanalogy_model):
                    if y0[No].item()==1:
                        a_c_sim.append(w1_w3_sim[No].item())
                        b_d_sim.append(w2_w4_sim[No].item())

                        a_c_dist.append(w1_w3_dist[No].item())
                        b_d_dist.append(w2_w4_dist[No].item())
                    else:
                        a_c_sim_n.append(w1_w3_sim[No].item())
                        b_d_sim_n.append(w2_w4_sim[No].item())


                        a_c_dist_n.append(w1_w3_dist[No].item())
                        b_d_dist_n.append(w2_w4_dist[No].item())
                    positive_negative='positive' if y0[No].item()==1 else 'negative'
                    for h_layer in similarity_in_hidden_layers.keys():

                        temp=similarity_in_hidden_layers[h_layer]
                        head_sim=temp['head'][0][No]
                        head_dist=temp['head'][1][No]
                        # tail
                        tail_sim=temp['tail'][0][No]
                        tail_dist=temp['tail'][1][No]
                        dataname=eval_d['supports']['name']
                        # print('eval_d',eval_d)
                        # exit()
                        if dataname not in Similarity_in_Hidden_Layers.keys():
                            Similarity_in_Hidden_Layers[dataname]=copy.deepcopy(temp_dic_layer)


                        if h_layer in Similarity_in_Hidden_Layers[dataname][positive_negative]['head']['sim'].keys():
                    
                            Similarity_in_Hidden_Layers[dataname][positive_negative]['head']['sim'][h_layer].append(head_sim)
                        else:
                            Similarity_in_Hidden_Layers[dataname][positive_negative]['head']['sim'][h_layer]=[]
                   
                            Similarity_in_Hidden_Layers[dataname][positive_negative]['head']['sim'][h_layer].append(head_sim)



                        if h_layer in Similarity_in_Hidden_Layers[dataname][positive_negative]['head']['dist'].keys():
                      
                            Similarity_in_Hidden_Layers[dataname][positive_negative]['head']['dist'][h_layer].append(head_dist)
                        else:
                            Similarity_in_Hidden_Layers[dataname][positive_negative]['head']['dist'][h_layer]=[]
                            t=(head_dist,eval_d)
                            Similarity_in_Hidden_Layers[dataname][positive_negative]['head']['dist'][h_layer].append(head_dist)
                        #####

                        if h_layer in Similarity_in_Hidden_Layers[dataname][positive_negative]['tail']['sim'].keys():
                       
                            Similarity_in_Hidden_Layers[dataname][positive_negative]['tail']['sim'][h_layer].append(tail_sim)
                        else:
                            Similarity_in_Hidden_Layers[dataname][positive_negative]['tail']['sim'][h_layer]=[]

                            Similarity_in_Hidden_Layers[dataname][positive_negative]['tail']['sim'][h_layer].append(tail_sim)



                        if h_layer in Similarity_in_Hidden_Layers[dataname][positive_negative]['tail']['dist'].keys():
                      
                            Similarity_in_Hidden_Layers[dataname][positive_negative]['tail']['dist'][h_layer].append(tail_dist)
                        else:
                            Similarity_in_Hidden_Layers[dataname][positive_negative]['tail']['dist'][h_layer]=[]
                         
                            Similarity_in_Hidden_Layers[dataname][positive_negative]['tail']['dist'][h_layer].append(tail_dist)

                    if positive_negative=='negative':
                        if similarity<0:
                            value=1
                        else:
                            value=0
                    else:
                        if similarity>0:
                            value=1
                        else:
                            value=0

                    polar_acc={type:positive_negative,'value':value}
                    if args.similarity_measure!='offset':
                        #similarity=w3_w4_sim[No]#.item()
                        if args.similarity_measure=='pairwise_cosine_sim':
                            sim_t=(w1_w3_sim+w2_w4_sim)/2
                            similarity=sim_t[No]
                            # print('similarity',similarity)
                            # print('eval_d',eval_d)
                            # print('####')
                            #exit()
                        elif args.similarity_measure=='pairwise_cosine_dist':
                            sim_t=w1_w3_dist+w2_w4_dist
                            sim_t=1/(sim_t+0.00001)
                            similarity=sim_t[No]


                    item={'similarity':similarity.item(),'y0':y0[No].item(),'r':R[No],'y':y[No],'eval':eval_d,'id':iD,'polar_acc':polar_acc}



                    if question_no in predictions_all[y[No].item()]:
                        predictions_all[y[No].item()][question_no][positive_negative].append(item)
                    else:
                        predictions_all[y[No].item()][question_no]={'positive':[],'negative':[]}
                        predictions_all[y[No].item()][question_no][positive_negative].append(item)
            else:
                h=model(batch,args,eval=True)
                #print('h',h.shape)
                for No,iD in enumerate(ids):
                    eval_d=eval_data[iD]

           

                    question_no,analogy_no=iD.split('-')
              
                    similarity = h[No,:].tolist()
                    S=None
                    for s in similarity:
                        if S==None:
                            S=s
                        else:
                            if s>S:
                                S=s
                    similarity=S


                    label =y0[No].item() if y0[No].item() ==1 else 0
                    value=similarity#.argmax(axis=-1)==label
                    value=int(value)
                    positive_negative='positive' if y0[No].item()==1 else 'negative'
                    polar_acc={type:positive_negative,'value':value}

                    item={'similarity':similarity,'y0':y0[No].item(),'r':R[No],'y':y[No],'eval':eval_d,'id':iD,'polar_acc':polar_acc}


                    if question_no in predictions_all[y[No].item()]:
                        predictions_all[y[No].item()][question_no][positive_negative].append(item)
                    else:
                        predictions_all[y[No].item()][question_no]={'positive':[],'negative':[]}
                        predictions_all[y[No].item()][question_no][positive_negative].append(item)
        if False:

            print_similarity_and_dist(args,Similarity_in_Hidden_Layers)


    if args.wordanalogy_model not in 'classification_train_head' :


        print('a_c_sim',sum(a_c_sim)/len(a_c_sim))
        print('b_d_sim',sum(b_d_sim)/len(b_d_sim))
        print('a_c_sim_n',sum(a_c_sim_n)/len(a_c_sim_n))
        print('b_d_sim_n',sum(b_d_sim_n)/len(b_d_sim_n))


        print('a_c_dist',sum(a_c_dist)/len(a_c_dist))
        print('b_d_dist',sum(b_d_dist)/len(b_d_dist))
        print('a_c_dist_n',sum(a_c_dist_n)/len(a_c_dist_n))
        print('b_d_dist_n',sum(b_d_dist_n)/len(b_d_dist_n))



    acc_for_evaluation_ekar,acc_for_evaluation_sat=print_predictions(predictions_all,word_analogy_types_dic_rev)

    #basian_analysis_of_conceptqa_vs_analogyqa(predictions_all,word_analogy_types_dic_rev)
    model.train()
    print('acc_for_evaluation_ekar,acc_for_evaluation_sat',acc_for_evaluation_ekar,acc_for_evaluation_sat)

    return acc_for_evaluation_sat





@torch.no_grad()    
def rc_eval(args,model,abstract,batch_data,device,idx2word='none'):
    temp=['BLESS','EVALution','CogALexV','ROOT09','KandH_plus_N','semeval_2012']
    torch.manual_seed(0)
    if args.data_type!='semeval_2012':
        file='essential_files/'+args.data_type+'rel_dic.json'
        with open(file) as f:
            rel_dic = json.load(f)['rel_dic']
            rel_dic_new={}
            kbid2id={}
            for k in rel_dic.keys():
                rel_dic_new[k]=rel_dic[k]['kbID']

            rel_dic_rev = {y: x for x, y in rel_dic_new.items()}
    ####

    ##
    torch.manual_seed(0)


    subclass_f1={}



    with open('essential_files/properties-with-labels.json') as f:
        properties = json.load(f)

        properties_p_to_id = {y: x for x, y in properties.items()}



    file='essential_files/Relation_counter_'+str(args.data_type)+'.json'
    with open(file) as f:
        Relation_counter_p_int = json.load(f)

    if args.data_type=='semeval_2012':
        Relation_counter_p_int=Relation_counter_p_int['relation']
        properties_p_to_id=Relation_counter_p_int
    Relation_counter_int_p = {y: x for x, y in Relation_counter_p_int.items()}


    #####################
    if args.data_type in ['semeval_2012','wikidata']:
        model_rc_analogy=LLMs_RC(model)
        modelname=args.model_name
        # dn='scan'
        # not_corrct_terms={'correct':{},'not_correct':{}}
        # acc=solve_analogies(modelname,dn,not_corrct_terms,model=model_rc_analogy,model_type='re')

        dn='google_easy'
        not_corrct_terms={'correct':{},'not_correct':{}}
        acc=solve_analogies(modelname,dn,not_corrct_terms,model=model_rc_analogy,model_type='re')

        dn='google_hard'
        not_corrct_terms={'correct':{},'not_correct':{}}
        aa=solve_analogies(modelname,dn,not_corrct_terms,model=model_rc_analogy,model_type='re')

        # dn='u4'
        # not_corrct_terms={'correct':{},'not_correct':{}}
        # acc=solve_analogies(modelname,dn,not_corrct_terms,model=model_rc_analogy,model_type='re')

        # dn='semeval_2012_easy'
        # not_corrct_terms={'correct':{},'not_correct':{}}
        # solve_analogies(modelname,dn,not_corrct_terms,model=model_rc_analogy,model_type='re')

        # dn='semeval_2012_hard'
        # not_corrct_terms={'correct':{},'not_correct':{}}
        # solve_analogies(modelname,dn,not_corrct_terms,model=model_rc_analogy,model_type='re')

        return acc



    ################
    PREDICTIONS=[]
    Y=[]
    PREDICTIONS_h=[]
    Y_0=[]
    n_epochs=1

    not_correct_dic={}

    predictions_not_corect={}
    model.eval()

    F1_l=[]
    F1_h_l=[]
    for di,data in enumerate(batch_data):
        print('di',di)
        # if di==0:
        #     continue
        
        c_1=0
        c_total=0
        eval_epxeriment_data={}

        for i_d_,x in enumerate(tqdm(data)):  

            examples=x




            y=examples['y'] #if 'y' in examples.keys() else 0 #else torch.ones_like(examples['s_r_'] ) #examples['s_r'] 
            y0=examples['y0']# if 'y0' in examples.keys() else 0# else torch.ones_like(examples['s_r_'] ) #examples['s_r'] 

            ########
            batch={}
            for key in examples.keys():
                batch[key]=examples[key].to(device) if key!='ids' else examples[key]
            ######################################
            ids=batch['ids']


            if (di!=1 and args.data_type=='retacred') or  True:
                eval_epxeriment_data=None
                #continue

            h,predictions,relations = model(batch,args,eval_epxeriment_data=eval_epxeriment_data,eval=True)

            relations=relations.argmax(axis=-1)

            h=h.argmax(axis=-1) if h!= None else 0
            ###
            y = y.to(device) 
 
         
            correct = relations == y

            PREDICTIONS.extend(relations.flatten().tolist())
            Y.extend(y.flatten().tolist())

        if di ==1 and args.data_type=='conll' :
            print('eval_epxeriment_data',eval_epxeriment_data.keys())
            cal_similarity(eval_epxeriment_data)
            print('NEGATIVE******************************************')
            cal_similarity(eval_epxeriment_data,negative=True)
        f1_macro=f1_score(Y, PREDICTIONS, average='macro')
        f1_micor=f1_score(Y, PREDICTIONS, average='micro')
        print('f1_macro',f1_macro)
        print('f1_micor',f1_micor)

        if 'head_1' in args.heads:
            f1_macro_h=f1_score(Y_0, PREDICTIONS_h, average='macro')
            f1_micor_h=f1_score(Y_0, PREDICTIONS_h, average='micro')
            print('f1_macro_h',f1_macro_h)
            print('f1_micor_h',f1_micor_h)


        print('subclass F1')
        dic_r_f1_micro={}
        dic_r_f1_macro={}

        for c in subclass_f1.keys():
            #print('class',c)
            f1_macro_h=f1_score(subclass_f1[c]['Y'], subclass_f1[c]['relation'], average='macro')
            f1_micor_h=f1_score(subclass_f1[c]['Y'], subclass_f1[c]['relation'], average='micro')
            # print('f1_macro_h',f1_macro_h)
            # print('f1_micor_h',f1_micor_h)
            dic_r_f1_micro[c]=f1_micor_h
            dic_r_f1_macro[c]=f1_macro_h

        dic_r_f1_micro_sorted=dict(sorted(dic_r_f1_micro.items(), key=lambda item: item[1]))
        dic_r_f1_macro_sorted=dict(sorted(dic_r_f1_macro.items(), key=lambda item: item[1]))

        for r in dic_r_f1_macro_sorted.keys():
            f1_macro_h=dic_r_f1_macro_sorted[r]
            f1_micor_h=dic_r_f1_micro_sorted[r]
            print('r',r)
            print('f1_macro_h',f1_macro_h)
            print('f1_micor_h',f1_micor_h)
            print('####')


    

    return f1_micor#F1_l[-1]




def georoc_train_eval(data_name,mode='train',exp_args=None,DATA=None,model_for_fine_tune=None):
    torch.manual_seed(64)
    if exp_args==None:

        model_name='bert-base-uncased'
        #model_name='roberta-large'
        
        #args.both_ab_not_ab='both'
        args=get_settings(mode,model_name,data_name)
        train_loader,dev_loader_b=rc_data(args)
        print('test',len(train_loader),len(dev_loader_b))
    else:
        args=exp_args
        train_loader,dev_loader_b=DATA['train'],DATA['dev']
        print('test',len(train_loader),len(dev_loader_b))
    
    
    abstract=args.abstract 

   
    model_to_train=args.model_to_train


    if model_to_train=='rc':
        model = Relation_Classifier_Model(
            args 
        ).to(args.device)
    elif model_to_train=='wordanalogy_re_model':
        model = Analogy_RE_Model(
           args
        ).to(args.device)
    ############
    Loss_Trend_pre=[]

    i_skip_pre=0
    start_epoch=1
    # if args.data_type=='wordanalogy':
    #     args.load_from_checkpoint=False
    if args.load_from_checkpoint:
        print('args.save_dir')
        # exit()
        if args.experiment_no in ['lexical_offset','sentential_re_paper',]:
            args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_'+str(args.data_type)+'.t7'
            args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_'+str(args.model_to_train)+str(args.data_type)+'.t7'
            args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_rc'+'wikidata'+'.t7'

            print('args.save_dir',args.save_dir)
        elif args.experiment_no in ['conceptqa',]:

            args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_'+str(args.model_to_train)+str(args.data_type)+'.t7'
            if args.sameconcept==False:
                args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_'+str(args.model_to_train)+'wikidata'+'.t7'

            print('args.save_dir',args.save_dir)
        elif args.experiment_no=='semeval_2012':
            if args.train_from_scratch:
                args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_rc'+'wikidata'+'.t7'
            else:
                args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_'+str(args.model_to_train)+str(args.data_type)+'.t7'




        elif args.experiment_no=='wordanalogy':
            if 'EVALution' in args.backend_trained:
                args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_'+str(args.model_to_train)+'EVALution'+'.t7'

            elif 'wikidata' in args.backend_trained:
                args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_'+str(args.model_to_train)+'wikidata'+'.t7'

            elif args.backend_trained=='similaroffset':
                args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_'+str(args.model_to_train)+str(args.data_type)+'.t7'
            


        if args.device==torch.device('mps'):

            checkpoint = torch.load(args.save_dir,weights_only=False,map_location=torch.device('mps'))
            print('mps')
        else:
            checkpoint = torch.load(args.save_dir,weights_only=False)
        ##########

        tmp=['wordanalogy',]
        print('args.experiment_no ',args.experiment_no )

        if args.experiment_no in ['semeval_2012','sentential_re_paper','conceptqa','five','six']:
            print('t')
            
            if args.experiment_no in ['semeval_2012','sentential_re_paper']:
                new_state_dict = OrderedDict()
                transformer=OrderedDict()
                transformer_decoder=OrderedDict()
  
                state_dict=checkpoint['state_dict']

                for k, v in state_dict.items():
 
                 
                    if 'transformer_decoder' in k:
                        k=k.replace('transformer_decoder.','')
                        transformer_decoder[k]=v
    

                    if 'transformer' in k:
                        k=k.replace('transformer.','')
                        transformer[k]=v
                    name=k
                    new_state_dict[name] = v

                model.transformer_decoder.load_state_dict(transformer_decoder)
                model.transformer.load_state_dict(transformer)

                print('loaded')

            else:
                model.load_state_dict(checkpoint['state_dict'])
                learning_rate=checkpoint['learning_rate']
            
        if args.experiment_no in tmp:
            new_state_dict = OrderedDict()
            transformer=OrderedDict()
            head_3=OrderedDict()
            state_dict=checkpoint['state_dict']
            for k, v in state_dict.items():
                if 'module' in k:
                    name = k[7:] # remove `module.`
                    if 'head_3' in k:
                        head_3[len('head_3'):]=v
                    if 'transformer' in k:
                        transformer[len('transformer'):]=v
                else:
                    if 'head_3' in k:
                        k=k.replace('head_3.','')
                        head_3[k]=v
                    if 'transformer' in k:
                        k=k.replace('transformer.','')
                        transformer[k]=v
                    name=k
                new_state_dict[name] = v

            model.head_3.load_state_dict(head_3)
            model.transformer.load_state_dict(transformer)
            print('loaded')

        Loss_Trend_pre=[]

    print('mode',mode)
    if model_for_fine_tune!=None:
        pass
        #model=model_for_fine_tune

    ###########
    if mode=='train':
        v=args.vocab_size

        

        criterion = nn.CrossEntropyLoss()
        if args.data_type=='semeval_2012':
            criterion = nn.BCELoss()#nn.CrossEntropyLoss()
        lr =args.lr
        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=lr
        )
        if args.load_from_checkpoint and args.mode!='eval':
            pass
           #optimizer.load_state_dict(checkpoint['optimizer'])
        if 'head_conditional' in args.heads:
            criterion2 = LabelSmoothing(size=v, padding_idx=0, smoothing=0.0)
            SimpleLossCompute_=SimpleLossCompute(model.generator, criterion2, optimizer)
        else:
            SimpleLossCompute_=None

        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.05, total_iters=12)
  

    
    ##
    if args.data_parallel:
        model = nn.DataParallel(model)# torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        model=model.to(device)

    Loss_Trend=Loss_Trend_pre+[] if args.load_from_checkpoint else [] 
    
    flag=0
    if mode =='eval':
        if args.data_type=='wordanalogy':
            args.train=False
            f1=rc_eval_wordanalogy(args,model,args.abstract,dev_loader_b,args.device)
            f1=0
        elif args.data_type=='conceptqa':
            print('con')
            f1=conceptqa_eval_wordanalogy(args,model,args.abstract,dev_loader_b,args.device)
        else:
            print()
            f1=rc_eval(args,model,abstract,dev_loader_b,args.device)
            print('end eval')
    F1=0

    if mode=='train':
        data=train_loader
        n_epochs=args.epcoh_n
        #initialize_relation_emb(model,optimizer,scheduler)
        for epoch in range(start_epoch,n_epochs):
                print('epoch',epoch)
                model.train()
                if args.model_to_train!='rc':
                    if epoch <= args.only_train_classifier:
                        model.set_only_classifier_train(True)
                    else:
                        model.set_only_classifier_train(False)
                epoch_loss = 0
                #model.set_only_head_train()
                args.epoch=epoch
                
                train_model(model,optimizer,scheduler,data,dev_loader_b,args,SimpleLossCompute_,criterion,Loss_Trend,epoch,F1)
        
    return model

def train_model(model,optimizer,scheduler,data,dev_loader_b,args,SimpleLossCompute_,criterion,Loss_Trend,epoch,F1):
    device=args.device
    model_to_train=args.model_to_train

    for i_d,x in enumerate(tqdm(data)):
        examples=x
        batch={}
        for key in examples.keys():
            batch[key]=examples[key].to(device) if key!='ids' else examples[key]

        if args.data_type not in ['wordanalogy','semeval_2012','conceptqa']:


            y0=batch['y0']
            y=batch['y']
            if args.abstract=='flagged_ents':
                ents_flagged_idxs=batch['ents_flagged_tokens'] 
                ents_flagged_plus_rel_idxs=batch['ents_flagged_plus_rel_tokens']
                Len_Target =batch['Len_Target']

            elif args.abstract=='abstract':
                ents_flagged_idxs=batch['abstracted_ents_flagged_tokens'] 
                ents_flagged_plus_rel_idxs=batch['abstracted_ents_flagged_plus_rel_tokens'] 
                Len_Target =batch['abstract_Len_Target']
            elif args.abstract=='mix':
                ents_flagged_idxs=batch['EntsAbst_flagged_tokens'] 
                ents_flagged_plus_rel_idxs=batch['EntsAbst_flagged_plus_rel_tokens'] 
                Len_Target =batch['ent_abstract_Len_Target']
            else:
                ents_flagged_idxs=batch['EntsAbst_flagged_tokens'] 
                ents_flagged_plus_rel_idxs=batch['EntsAbst_flagged_plus_rel_tokens'] 
                Len_Target =batch['ent_abstract_Len_Target']
        
        ######################################
        if args.data_type=='wordanalogy':
            y0=batch['y0']

            if 'baseline'  in args.wordanalogy_model:
                label= batch['r']
                h,loss = model(batch,args)
                l=0
                loss=loss
            elif 'sentence' in args.wordanalogy_model:
                label= batch['r']
                label_= batch['r1']

                h,h_,l= model(batch,args)

                loss_=0#criterion(h_, label_) if args.fin_tune==False else 0

                loss=l

            elif 'route' in args.wordanalogy_model:
                label= batch['r']

                h,l= model(batch,args)
                loss=criterion(h, label) if args.fin_tune==False else 0
                loss=loss+l if args.fin_tune==False else l
            else:
                label= torch.where(y0 == -1, 0, y0)
                # print('y0',y0)
                # print('label',label)
                h= model(batch,args)
                loss=criterion(h, label)
                #print('loss',loss) 

            loss.backward()
            optimizer.step()
            model.zero_grad()
            continue
  

        elif args.data_type not in ['wordanalogy','semeval_2012','conceptqa']:
            h,predictions,relations = model(batch,args)
            y = y.to(device) 
            loss_relation = criterion(relations, y) 
            loss_relation_h =criterion(h, y0) if 'head_1' in args.heads  else 0
            loss=loss_relation_h+loss_relation
            h=h.argmax(axis=1) if 'head_1' in args.heads  else h
            correct = relations.argmax(axis=1) == y

            temp=False if args.abstract=='mask' else True

            print('loss',loss)

            loss.backward(retain_graph=True)
            labels =ents_flagged_plus_rel_idxs[:,1:,] if model_to_train=='rc' else ents_flagged_idxs[:,1:,]
            Len_Target=batch['Len_Target']
            if 'head_conditional' in args.heads:
                loss_deocoder = SimpleLossCompute_(predictions, labels,len(torch.unique(labels)))
                #print('loss_deocoder',loss_deocoder)
        elif args.data_type=='semeval_2012':
            out1,out2 ,loss= model(batch,args)
            s_r=batch['s_r']

            if args.epoch>=0:
                loss.backward(retain_graph=True)
     
     


            labels=batch['s_plus_rel'][:,1:,].clone()
            s_len=batch['s_len']
            predictions=out1
            device=predictions.device
            # for bi,indx in enumerate(s_len.tolist()):
            #     predictions[bi,indx-3:]=torch.zeros_like(predictions[bi,indx-3:]).to(device)


            loss_deocoder = SimpleLossCompute_(predictions, labels,len(torch.unique(labels)),retain_graph=True,fl=False)
            print('loss_deocoder',loss_deocoder)

            labels=batch['a_plus_rel'][:,1:,].clone()
            predictions=out2
            a_len=batch['a_len']
            device=predictions.device
            # for bi,indx in enumerate(a_len.tolist()):
            #     predictions[bi,indx-3:]=torch.zeros_like(predictions[bi,indx-3:]).to(device)

            loss_deocoder = SimpleLossCompute_(predictions, labels,len(torch.unique(labels)),retain_graph=False,fl=True)
            print('loss_deocoder',loss_deocoder)
        elif args.data_type=='conceptqa':
            loss= model(batch,args)
            loss.backward(retain_graph=False)
        ##############
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        ##
        optimizer.step()
        ##
        if (i_d+1)%args.scheduler_step==0:
            scheduler.step()
        model.zero_grad()
    scheduler.step()

    args.save_dir=args.checkpoint_path+'/model_'+str(args.model_name)+'_'+str(args.model_to_train)+str(args.data_type)+'.t7'


    temp='None'
    state = {
            'Train_Args':args,
            'i_d':i_d,
            'epoch': epoch,
            'learning_rate':optimizer.param_groups[0]["lr"],
            'state_dict': deepcopy(model.state_dict()),
            'classifier_head':temp,
            'optimizer': optimizer.state_dict(),
    }
    savepath=args.save_dir

    
    if args.data_type!='wordanalogy' and args.data_type!='conceptqa':
        model.eval()
        f1=rc_eval(args,model,args.abstract,dev_loader_b,device)
        if f1!=None:
            if args.save and f1>F1:
                F1=f1
                print('saving.....')
                print('savepath',savepath)
               
                torch.save(state,savepath)
        else:
            savepath='models_saved/backup.t7'
            torch.save(state,savepath)


        model.train()
    elif args.data_type=='conceptqa':
        f1=conceptqa_eval_wordanalogy(args,model,args.abstract,dev_loader_b,device)
        if f1!=None:
            if args.save and f1>F1:
                F1=f1
                print('saving.....')
                print('savepath',savepath)
               
                torch.save(state,savepath)
        else:
            savepath='models_saved/backup.t7'
            torch.save(state,savepath)

    else:
        model.eval()
        f1=rc_eval_wordanalogy(args,model,args.abstract,dev_loader_b,device)
        if f1!=None:
            if args.save and f1>F1:
                F1=f1
                print('saving.....')
                print('savepath',savepath)
               
                torch.save(state,savepath)
        else:
            savepath='models_saved/backup.t7'
            torch.save(state,savepath)
        model.train()
   

    







