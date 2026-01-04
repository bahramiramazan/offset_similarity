
PATH = '.'#Add you directory here
import sys
sys.path.append(PATH)
import torch.nn as nn

from benchmarks.web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999

from benchmarks.web.analogy import *
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

import numpy as np
#########################################################################
all_models_names=['prophetnet','opt','gpt2','t5-large','t5-small','roberta-base','roberta-large','bert-large-uncased','bert-base-uncased']




# Load GloVe embeddings
def load_glove(file_path):
    embeddings = {}
    with open(file_path, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings



def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    c=0
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        c=c+1
        if c==300:
            break

    return data

###
class LLMs_RC:

    def __init__(self,model):

        self.transformer,self.tokenizer=model.transformer,model.tokenizer
        self.model=model

        self.device=model.device
        self.model.to(self.device)

        self.name=model_name
        self.args=model.args

        print('model initialiized')

    def idx_sen(self,tokenized_text):
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(indexed_tokens)
            if len(tokenized_text)!=len(indexed_tokens):
                print('not equal')
                exit()
            return indexed_tokens,segments_ids

    @torch.no_grad
    def get_words_offset(self,w1,w2):

        e1=w1
        e2=w2
        label_head_e=e1.split(' ')
        label_tail_e=e2.split(' ')
        flags=[['[e11]'],['[e12]'],['[e21]'],['[e22]']]
        ents_flagged_tokens=flags[0]+label_head_e+flags[1]+flags[2]+label_tail_e+flags[3]
        temp_= '[mask]'
        ents_flagged_plus_rel_tokens=ents_flagged_tokens.copy()+[temp_,]


        sentence=' '.join(ents_flagged_tokens)
        marked_text = "[CLS] " + sentence + " [SEP]"
        sentence_tokens = self.tokenizer.tokenize(marked_text)
        temp_idx,temp_segment_idx=self.idx_sen(sentence_tokens)

        ents_flagged_tokens=torch.tensor(temp_idx).to(self.model.device) 
        ents_flagged_tokens_masks=torch.tensor(temp_segment_idx).to(self.model.device)
        #####

        sentence=' '.join(ents_flagged_plus_rel_tokens)
        temp = "[CLS]" + sentence + " [SEP]"
        temp_tokens = self.tokenizer.tokenize(temp)
        #print('temp_tokens',temp_tokens)
        temp_idx,temp_segment_idx=self.idx_sen(temp_tokens)
        ents_flagged_plus_rel_tokens=temp_idx

        Len_Target=len(ents_flagged_plus_rel_tokens)
        rel_n=2
        rel_id=ents_flagged_plus_rel_tokens[Len_Target-rel_n]

        tok_rel_id=self.tokenizer.convert_ids_to_tokens([rel_id])
        #print('tok_rel_id',tok_rel_id)
        ents_flagged_plus_rel_tokens=torch.tensor(ents_flagged_plus_rel_tokens).to(self.model.device)
        Len_Target=torch.tensor([Len_Target,]).to(self.model.device)
        ents_flagged_tokens=ents_flagged_tokens.unsqueeze(0)
        ents_flagged_tokens_masks=ents_flagged_tokens_masks.unsqueeze(0)
        ents_flagged_plus_rel_tokens=ents_flagged_plus_rel_tokens.unsqueeze(0)
        Len_Target=Len_Target.unsqueeze(0)
        batch={'s':ents_flagged_tokens,\
        's_masks':ents_flagged_tokens_masks,\
        's_plus_rel':ents_flagged_plus_rel_tokens,\
        's_len':Len_Target
    
        }


        rep=self.model(batch,self.args,return_offset=True)
        return rep




class LLMs:

    def __init__(self,model_name,model=None,route_flag=False):
        if model==None:
            data_selected,modality='wordanalogy',model_name
            tokenizer_special_dic='none'
            _,self.model,self.tokenizer=_get_pretrained_transformer3(data_selected,model_name,tokenizer_special_dic=None)
        else:
            if route_flag==False:
                self.model,self.tokenizer=model[0],model[1]
            else:
                #route_flag
                self.model,self.tokenizer,self.head_3=model[0],model[1],model[2]

        self.device=torch.device('mps')
        self.model.to(self.device)
        self.name=model_name
        self.route_flag=route_flag
    def get_word_embed(self,w,route=False):



        marked_text = " " + w + ""
        tokenized_text = self.tokenizer.tokenize(marked_text)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens)

        indexed_tokens=torch.tensor(indexed_tokens).unsqueeze(0).to(self.device)
        segments_ids=torch.tensor(segments_ids).unsqueeze(0).to(self.device)

        if 't5' not in self.name:
            e=self.model(input_ids=indexed_tokens,attention_mask=segments_ids).hidden_states
            if self.route_flag:
                e=torch.stack(e,0)
                e=torch.swapaxes(e,0,1)
                e=torch.swapaxes(e, 1, 2)
                e=self.head_3(e)
                e=e.unsqueeze(-1)
            if self.name=='opt':
                e=e[:-1]
                e=e[-1]

            else:
                e=e[-1]
            e=e.cpu()
            e=torch.sum(e, 1)
            e=e.numpy()
            return e
        else:

            ab_decoder_input_ids = self.model._shift_right(indexed_tokens)
            e = self.model(input_ids=indexed_tokens, decoder_input_ids=ab_decoder_input_ids)
            e = e.last_hidden_state
            e=e.cpu()
            e=torch.sum(e, 1)
        e=e.numpy()
        return e


###################################


def get_model(modelname):
    model_name=modelname
    if model_name=='fasttext':

        import io
        import fasttext
        model = fasttext.load_model("fasttext/crawl-300d-2M-subword/crawl-300d-2M-subword.bin")
    elif model_name=='word2vec':
        from gensim.models import KeyedVectors
            # Load pretrained Word2Vec (Google News)
        path = "gloves-word2vec/GoogleNews-vectors-negative300.bin"
        model = KeyedVectors.load_word2vec_format(path, binary=True)
    elif model_name=='glove':
        # Output: Word2Vec format file
        glove_path = "gloves-word2vec/dolma_300_2024_1.2M.100_combined.txt"

        model = load_glove(glove_path)
    elif model_name in all_models_names:
        model=LLMs(model_name)

    return model


def get_offset_sim(model,w1,w2,w3,w4,model_name,sim_f='cosine_similarity'):


    stem_w1=w1
    stem_w2=w2
    stem_w3=w3
    stem_w4=w4
    stem_d=None
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    fl=True
    if model_name in ['glove','word2vec']:

        if model_name=='glove':
            if stem_w1 in model.keys() and stem_w2 in model.keys() and tem_w3 in model.keys() and stem_w4 in model.keys():
                stem_d=model[stem_w1] -model[stem_w2]

                stem_d_=model[stem_w3] -model[stem_w4]
            else:

                fl=False

        if model_name=='word2vec':
            if stem_w1 in model.key_to_index.keys() and stem_w2 in model.key_to_index.keys() and stem_w3 in model.key_to_index.keys() and stem_w4 in model.key_to_index.keys():
                stem_d=model[stem_w1] -model[stem_w2]
                stem_d_=model[stem_w3] -model[stem_w4]

            else:
                fl=False
    if fl==False:
        return None
    elif model_name=='fasttext':
        stem_d=model.get_word_vector(stem_w1) -model.get_word_vector(stem_w4)
        stem_d_=model.get_word_vector(stem_w3) -model.get_word_vector(stem_w4)
    elif model_name in all_models_names:
        #e=model.get_word_embed('king') 
        stem_d=model.get_word_embed(stem_w1) -model.get_word_embed(stem_w2)
        stem_d_=model.get_word_embed(stem_w3) -model.get_word_embed(stem_w4)

    stem_d=torch.from_numpy(stem_d).unsqueeze(0)
    stem_d_=torch.from_numpy(stem_d_).unsqueeze(0)

    
    if sim_f=='cosine_similarity':
        s=cos(stem_d ,stem_d_ )
    else:
        s=torch.nn.functional.pairwise_distance(stem_d, stem_d_)
    


    return s.item()




def get_embedding(model,w,model_name):
    stem_d=None
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6) 
    fl=True
    if model_name in ['glove','word2vec']:

        if model_name=='glove':
            if w in model.keys()  :
                e=model[w]
            else:
                fl=False
        if model_name=='word2vec':
            if w in model.key_to_index.keys() :
                e=model[w]
            else:
                fl=False
    if fl==False:
        return None
    elif model_name=='fasttext':
        e=stem_d=model.get_word_vector(w)
        e=torch.from_numpy(e).unsqueeze(0)

    elif model_name in all_models_names:
        e=model.get_word_embed(w) 
        e=torch.from_numpy(e)
    return e

def get_sim(model,w1,w2,model_name,sim_f='cosine_similarity'):


    stem_w1=w1
    stem_w2=w2

    stem_d=None
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6) 
    fl=True
    if model_name in ['glove','word2vec']:

        if model_name=='glove':
            if stem_w1 in model.keys() and stem_w2 in model.keys() :
                stem_d=model[stem_w1] -model[stem_w2]


            else:

                fl=False

        if model_name=='word2vec':
            if stem_w1 in model.key_to_index.keys() and stem_w2 in model.key_to_index.keys() :
                w1_e=model[stem_w1]
                w2_e=model[stem_w2]  
             

            else:
                fl=False
    if fl==False:
        return None
    elif model_name=='fasttext':
        w1_e=stem_d=model.get_word_vector(stem_w1)
        w2_e=model.get_word_vector(stem_w2)

    elif model_name in all_models_names:
        #e=model.get_word_embed('king') 
        w1_e=model.get_word_embed(stem_w1) 
        w2_e=model.get_word_embed(stem_w2)


    w1_e=torch.from_numpy(w1_e).unsqueeze(0)
    w2_e=torch.from_numpy(w2_e).unsqueeze(0)
   
    if sim_f=='cosine_similarity':
        s=cos(w1_e ,w2_e )
    else:
        s=torch.nn.functional.pairwise_distance(w1_e, w2_e)
    

    return s.item()
####





def get_conceptqa_questions(abstract_spaces_dic,n_size=20,easy=False):

    heat_map=[]
    min_n=1
    max_n=0
    label_names=[]
    Questions=[]
    keys=list(abstract_spaces_dic.keys())
    random.shuffle(keys)
    for ki,k in enumerate(keys):
        population=abstract_spaces_dic[k]
        L=[si for si in range(len(population))]
        N=n_size if len(L)>n_size else len(L)
        # if N<n_size:
        #     continue
        #N=len(L)
        s=random.sample(L,N)
        selected=population#[population[j_] for j_ in s]
        PAIRS=[]
        for S in selected:
            L=[si for si in range(len(population))]
  
            N=len(L) if len(L)<4 else 4
            sc=random.sample(L,N)
            for s in sc:

                c=population[s]
                if c==S:
                    continue
                PAIRS.append([S,c])

        pairs=PAIRS#[(selected[i:i+2]) for i in range(0,len(selected),2)]

        for p in pairs:

            if len(p)<2:
                #print('p',p)
                p=[p[0],pairs[0][0]]

            q,answer=p[0],p[1]
            q,q_related=q[0].lower(),q[1].lower()
            answer=answer[0].lower()
            if q==answer:
                continue
            ###
            choice_n=5
            choices=[]#q_related
            choices=[q_related,] if easy==False else []
            j=0
            while j<5:
                population_=list(abstract_spaces_dic.keys())
                L=[si for si in range(len(population_))]
                N=1 if len(L)>1 else len(L)

                s=random.sample(L,N)[0]
                selected_key=population_[s]
                if k==selected_key:
                    continue

                population2=abstract_spaces_dic[selected_key]
                L2=[si for si in range(len(population2))]
                N2=1 if len(L2)>1 else len(L2)
                if len(L2)<1:
                    continue
                s2=random.sample(L2,N2)[0]
                selected2=population2[s2] 
                selected2=selected2[0]
                if q_related==selected2:
                    continue
                choices.append(selected2.lower())
                j=j+1
            if len(choices)<3:
                print('lll',choices)
                exit()
                continue
            else:
                choices=choices[:5]

            #random.shuffle(choices)
            item={'q':q,'answer':answer,'choices':choices,'key':k}
            Questions.append(item)
            #print('item',item)
    return Questions

def plot_semantic_space_questions(model,model_name,abstract_spaces_dic,sim_f='cosine_similarity',n_size=50,plot=False,easy=False):

    word_concept_similarity_category={}
    allword_dic={}
    Questions=get_conceptqa_questions(abstract_spaces_dic,n_size=50,easy=easy)
    print('len',len(Questions))
    correct=0
    not_correct=0
    space_wise_acc={}
    space_average_of_similarity={}
    for t in Questions:
        #print('t',t)

        q=t['q'].lower()
        choices=t['choices']
        answer=t['answer'].lower()
        # print('q',q)
        # print('answer',answer)
        key=t['key']
        if key not in space_wise_acc.keys():
            space_wise_acc[key]={'correct':0,'not_correct':0}
            space_average_of_similarity[key]=[]

        sim_answer=get_sim(model,q,answer,model_name,sim_f=sim_f)
        space_average_of_similarity[key].append(sim_answer)
        choice_sim=[]
        for c in choices:
            sim=get_sim(model,q,c,model_name,sim_f=sim_f)
            choice_sim.append(sim)
        if sim_f=='cosine_similarity':
            max_sim_choice=max(choice_sim)
            if sim_answer>max_sim_choice:
                correct=correct+1
                space_wise_acc[key]['correct']=space_wise_acc[key]['correct']+1
                ###############################
                if q in word_concept_similarity_category.keys():
                    word_concept_similarity_category[q]['correct'].append(sim_answer)
                else:
                    word_concept_similarity_category[q]={'correct':[sim_answer,],'not_correct':[]}

                if answer in word_concept_similarity_category.keys():
                    word_concept_similarity_category[answer]['correct'].append(sim_answer)
                else:
                    word_concept_similarity_category[answer]={'correct':[sim_answer,],'not_correct':[]}
          
            else:
                not_correct=not_correct+1
                space_wise_acc[key]['not_correct']=space_wise_acc[key]['not_correct']+1
                ########################################################################
                if q in word_concept_similarity_category.keys():
                    word_concept_similarity_category[q]['not_correct'].append(sim_answer)
                else:
                    word_concept_similarity_category[q]={'correct':[],'not_correct':[sim_answer,]}

                if answer in word_concept_similarity_category.keys():
                    word_concept_similarity_category[answer]['not_correct'].append(sim_answer)
                else:
                    word_concept_similarity_category[answer]={'correct':[],'not_correct':[sim_answer,]}


        else:
            min_sim_choice=min(choice_sim)
            if sim_answer>min_sim_choice:
                not_correct=not_correct+1
                space_wise_acc[key]['not_correct']=space_wise_acc[key]['not_correct']+1

            else:

                correct=correct+1
                space_wise_acc[key]['correct']=space_wise_acc[key]['correct']+1
        # print('word_concept_similarity_category',word_concept_similarity_category)
        # print('t',t)
        # exit()

 


    print('correct',correct)
    print('not_correct',not_correct)
    print('acc',correct/(correct+not_correct))
    overall_acc=correct/(correct+not_correct)
    overall_correct=correct
    overall_not_correct=not_correct
    print('overall_acc',overall_acc)
    print('overall_correct',overall_correct)
    print('overall_not_correct',overall_not_correct)
    all_acc={}
    for k in space_wise_acc.keys():
        t=space_wise_acc[k]
        correct=t['correct']
        not_correct=t['not_correct']
        acc=correct/(correct+not_correct)
        all_acc[k]=(acc,correct,not_correct)

    ##
    sorted_x = sorted(all_acc.items(), key=lambda kv: kv[1])
    concept_correct_count=[]
    for k in sorted_x:
        acc=k[1]
        # print('k',k)
        # print('acc',acc)
        # print('######################################')
        if acc[0]>.55:
            concept_correct_count.append(k)

    print('overall_acc',overall_acc)
    print('overall_correct',overall_correct)
    print('overall_not_correct',overall_not_correct)
    #######
    print(' all space_wise_acc',len(space_wise_acc.keys()))
    print('concept_correct_count',concept_correct_count)
    print('concept_correct_count',len(concept_correct_count))
    # print('word_concept_similarity_category',word_concept_similarity_category)
    # exit()
    

    if plot==False:
        return word_concept_similarity_category


    else :
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        Category=[]
        values=[]

        for ki,k in enumerate(sorted_x):
            acc=k[1]
            if len(k[0])<20  and ki%5==0:
                Category.append(k[0])
                values.append(acc[0])
                print('k',k[0])
                print('acc',acc[0])
        data = {
            'Category':Category ,
            'Value': values
        }
        df = pd.DataFrame(data)

        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_theme(style="whitegrid")
        f, ax = plt.subplots(figsize=(28, 44))
        sns.set_color_codes("pastel")
        sns.barplot(x="Value", y="Category", data=data,
                     color="b",dodge=False, width=0.3)
        ax.tick_params(axis="y", labelsize=18)
        ax.tick_params(axis="x", labelsize=18)
        ax.set(xlim=(0, 1.1), ylabel="",
               xlabel="")
        sns.despine(left=True, bottom=True)
        plt.savefig('images/sim_dis/semantic_spaces_acc.png')
        plt.close()
        #######
        average_sim_in_space={}
        for k in space_average_of_similarity.keys():
            L=space_average_of_similarity[k]
            average_sim_in_space[k]=sum(L)/len(L)


        Category=[]
        values=[]

        for ki,k in enumerate(sorted_x):
            acc=k[1]
            #if len(k[0])<20  and ki%5==0:
            Category.append(ki)
            
            L=space_average_of_similarity[k[0]]
            #average_sim_in_space[k[0]]=sum(L)/len(L)
            values.append(sum(L)/len(L))


        data = {
            'Semantic Spaces Sorted by Accuracy on ConceptQA': Category,
            'Average Query-Answer sim (100 Q)': values
        }
        df = pd.DataFrame(data)
        print('df',df.head())
        sns.set(font_scale=4)
        f, ax = plt.subplots(figsize=(24, 24))

        sns.lineplot(data=data, x="Semantic Spaces Sorted by Accuracy on ConceptQA", y="Average Query-Answer sim (100 Q)")
        ax.tick_params(axis="y", labelsize=24)
        ax.tick_params(axis="x", labelsize=24)

        #plt.show()
        plt.savefig('images/sim_dis/semantic_spaces_line.png')
        plt.close()
        return word_concept_similarity_category

    








def plot_semantic_space_vs_other(model,model_name,abstract_spaces_dic,every_n=5,simple=False,min_space_len=10):


    heat_map=[]
    min_n=1
    max_n=0
    label_names=[]
    spaces_list=list(abstract_spaces_dic.keys())
    random.shuffle(spaces_list)
    total=1000
    count=0
    for ki,k in enumerate(spaces_list):

        if len(k)>20: 
            continue
        if ki%every_n!=0:
            continue
        population=abstract_spaces_dic[k]
        # if len(population)<20:
        #     continue

        L=[si for si in range(len(population))]
        N=20 if len(L)>20 else len(L)
        if N<min_space_len:
            continue
        s=random.sample(L,N)
        selected=[population[j_] for j_ in s]
        map_={}
        label_names.append(k)
        for w1 in selected:
            w1=w1[0]
            for ki2,k2 in enumerate(spaces_list):

                if len(k2)>20: 
                    continue
                if ki2%every_n!=0:
                    continue
      
                population2=abstract_spaces_dic[k2]
                # if len(population2)<20:
                #     continue

                L2=[si for si in range(len(population2))]
                N2=20 if len(L2)>20 else len(L2)
                if N2<min_space_len:
                    continue
                s2=random.sample(L2,N2)
                selected2=[population2[j_] for j_ in s2]
                for w2 in selected2:
                    w2=w2[0]
     
                    if w1==w2:
                        continue
               

                    sim=get_sim(model,w1,w2,model_name)
                    sim=sim+(1-sim)/2#1/math.exp(200*(-sim))
                    count=count+1

                    key=str(k)+'#'+str(k2)
                    if key in map_.keys():
                        map_[key].append(sim)
                    else:
                        map_[key]=[]
                        map_[key].append(sim)
       
        all_sim=[sum(map_[t])/len(map_[t]) for t in map_.keys()]
        min_n_=min(all_sim)
        max_n_=max(all_sim)
        if min_n> min_n_:
            min_n=min_n_

        if max_n< max_n_:
            max_n=max_n_
        print('all_sim',len(all_sim))
        
        heat_map.append(all_sim)



    print('label_names',len(label_names))
    print('heat_map',len(heat_map))
    simple=True if len(heat_map)<20 else False

    ###
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np





    df= pd.DataFrame(heat_map, index=label_names, columns=label_names)
    annot_df = pd.DataFrame(
        np.full_like(df.values, '', dtype=object), 
        index=df.index, 
        columns=df.columns
    )


    if simple:
        plt.figure(figsize=(6, 6))
        
    else:
        if len(heat_map)<50:
            plt.figure(figsize=(12, 12))
        else:

            plt.figure(figsize=(18, 18))
        
    cmap = sns.color_palette("crest", as_cmap=True)

    sns.heatmap(
        df,
        annot=annot_df,          
        fmt='s',                
        cmap=cmap,               
        linewidths=0.5,          
        cbar=True,               
        vmin=min_n,               
        vmax=max_n,
        # Optional: Customize the appearance of the labels
        annot_kws={"fontsize": 15, "fontweight": "bold"} 
    )
    if simple==False:
        plt.title('Similarity between Different Category(Semantic Spaces) of Words', fontsize=15, pad=20)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    #plt.show()
    plt.savefig('images/sim_dis/semantic_spaces'+str(model_name)+'.png')
    plt.close()






def check_r_head_tail_concepts(all_related):
    r_head_tail_concept={}
    count_=0

    for k in all_related:
        L=all_related[k]
        r_head_tail_concept[k]={'pair':set()}
        for t in L:
            head_a=t['head_a']
            tail_a=t['tail_a']
            #r_head_tail_concept[k]['head_a'].add(head_a)
            #r_head_tail_concept[k]['tail_a'].add(tail_a)
            p=(head_a,tail_a)
            r_head_tail_concept[k]['pair'].add(p)
            count_=count_+1


    count=0
    total=0
    all_f=[]

    for k in r_head_tail_concept.keys():

        t=r_head_tail_concept[k]['pair']
        total=total+1

        all_f.append(len(t))
        if len(t)<6:
            count=count+1
        print('k',k)
        print('t',t)
        print('####################')
    print('count',count)
    print('total',total)
    ##
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np


    X=[i for i in range(max(all_f))]
    category_counts = pd.Series(all_f).value_counts().reset_index()
    category_counts.columns = ['Category', 'Frequency']
    plt.figure(figsize=(9, 9))
    sns.barplot(x='Frequency', y='Category', data=category_counts, palette='mako',errorbar=None)
    plt.xlabel('No of Unique head tail types', fontsize=12)
    plt.ylabel('No of Relations ', fontsize=12)

    plt.tight_layout()
    #plt.show()
    plt.savefig('images/sim_dis/relation_unique_abstract_fre.png')


def get_google_abstract_space_dic(embeddings=False,model=None,model_name=None,semeval_or_google='google',f='semantic_space_dic'):
    if semeval_or_google=='google':
        file='essential_files/simple_analogy.json'
        data= json.load(open(file))['data']
        ###
    elif semeval_or_google=='semeval_2012':
        file='essential_files/simple_analogy_semeval.json'
        file='essential_files/all_psotives_semeval.json'
        data= json.load(open(file))['data']

    elif semeval_or_google=='wikidata':
        file='essential_files/all_psotives_wikidata.json'
        data= json.load(open(file))['data']
        all_psotives_wikidata=data

    Labels=[]
    all_related={}
    All_Wv=[]
    cats_all=[]
    abstract_spaces_dic={}
    for k in data:

        d=data[k]
        for t in d:
            
            w1=t['a']
            w2=t['b']
            c1=t['c1'].lower()
            c2=t['c2'].lower()
            w1=w1.lower()
            w2=w2.lower()
            head_a=c1
            tail_a=c2

            kbID=k
            item={'w1':w1,'w2':w2,'head_a':head_a,'tail_a':tail_a}
            if kbID in all_related.keys():
                all_related[kbID].append(item)

            else:
                all_related[kbID]=[]
                all_related[kbID].append(item)


            #print('w1,w2',w1,w2)
            if head_a in abstract_spaces_dic.keys():
                if w1!=head_a:
                    abstract_spaces_dic[head_a].append((w1,w2))

                else:
                    #print('w1,head_a',w1,head_a)
                    continue
                    abstract_spaces_dic[head_a].append((w1,w2))
            else:
                abstract_spaces_dic[head_a]=[]
                if w1!=head_a:
                    abstract_spaces_dic[head_a].append((w1,w2))
                else:
                    #print('w1,head_a',w1,head_a)
                    continue
                    abstract_spaces_dic[head_a].append((w1,w2))
            ##########################
            if tail_a in abstract_spaces_dic.keys():
                if w2!=tail_a:
                    abstract_spaces_dic[tail_a].append((w2,w1))
                else:
                    #print('w2,tail_a',w2,tail_a)
                    continue
                    abstract_spaces_dic[tail_a].append((w2,w1))
            else:
                abstract_spaces_dic[tail_a]=[]
                if w2!=tail_a:
                    abstract_spaces_dic[tail_a].append((w2,w1))
                else:
                    #print('w2,tail_a',w2,tail_a)
                    continue
                    abstract_spaces_dic[tail_a].append((w2,w1))
    #exit()
    if f=='semantic_space_dic':
        return abstract_spaces_dic
    else:
        return all_related


def get_abstract_spaces_dic_wikidata(train_or_dev='dev',f='semantic_space_dic'):
    benchmarks='wikidata'
    train_eval='train'
    file='essential_files/SRE_Analogy.json'
    SRE_Analogy= json.load(open(file))['SRE_Analogy']['wikidata'][train_or_dev]
    no_relation_all=['18','no_relation','P0']
    no_relation_examles=[]
    relation_data_all={}
    analogy_data=[]
    skip_wikidata=[]
    x_axis=[]
    y_axis=[]
    h_sim=[]
    t_sim=[]
    offset_sim=[]
    all_abstracts=set()
    all_words=set()
    
    abstract_spaces_dic={}
    all_related={}

    j=0
    data_type=train_or_dev
    for d in SRE_Analogy:
        # if j==1000:
        #     break
        j=j+1
        rel=d['r']
        type_=d['type']
        #print('d',d)
        #exit()
        kbID=d['kbID'] if (d['kbID']!='semeval' and d['kbID']!='retacred') else None

        item={'data_name':data_type,'data':d}
        if kbID!=None:
            rel=kbID
  
  

        t_w1=d['a'] if d['a'][:3].lower()!='the'  else d['a'][4:]
        t_w2=d['b'] if d['b'][:3].lower()!='the'  else d['b'][4:]
        t_w1=t_w1.lower()
        t_w2=t_w2.lower()
        all_words.add(t_w1)
        all_words.add(t_w2)



        head_a=d['e1_abstract']
        tail_a=d['e2_abstract']
        ####
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
        all_abstracts.add(head_a)
        all_abstracts.add(tail_a)
        ###
        if head_a in abstract_spaces_dic.keys():
            if t_w1!=head_a:
                abstract_spaces_dic[head_a].append((t_w1,t_w2))
        else:
            abstract_spaces_dic[head_a]=[]
            if t_w1!=head_a:
                abstract_spaces_dic[head_a].append((t_w1,t_w2))

        if tail_a in abstract_spaces_dic.keys():
            if t_w2!=tail_a:
                abstract_spaces_dic[tail_a].append((t_w2,t_w1))
        else:
            abstract_spaces_dic[tail_a]=[]
            if t_w2!=tail_a:
                abstract_spaces_dic[tail_a].append((t_w2,t_w1))

        ############
        kbID=d['r']
        item={'w1':t_w1,'w2':t_w2,'head_a':head_a,'tail_a':tail_a}
        if kbID in all_related.keys():
            all_related[kbID].append(item)

        else:
            all_related[kbID]=[]
            all_related[kbID].append(item)

##
    abstract_spaces_dic_new={}
    for k in abstract_spaces_dic.keys():
        data=abstract_spaces_dic[k]
        data=list(set(data))
        if len(data)>5:
            abstract_spaces_dic_new[k]=data
    abstract_spaces_dic=abstract_spaces_dic_new
    if f=='semantic_space_dic':
        return abstract_spaces_dic
    else:
        return all_related,all_abstracts,all_words




def print_sample(dn):

    import matplotlib.pyplot as plt
    import numpy as np
    # ##
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    dn ='scan'

    file='essential_files/wordanalogy_'+'DATA_multi_choice.json'


    file='essential_files/wordanalogy_'+'DATA_words_fast_text.json'

    data= json.load(open(file))['DATA_multi_choice']
    print('data',data['test'].keys())
    print('data',data['train'].keys())


    ###
    x=[]
    y=[]

    data_test=data['test'][dn]
    data_train=data['train'][dn]

 


    R=[]
    letters=['a','b','c','d','e','f']
    answer_set={}
    for di,d in enumerate(data_test):
        #print('d',d)
        # if di==100:
        #     break

        stem=d['stem']
        choice=d['choice']
        random.shuffle(choice)
        answer=d['answer']
        answer=choice[answer]
        print('chose the correct option for the question.')
        print('QuestionNO: ', di,' : ',stem['w1'], 'is to ', stem['w2'],' as: ')
        random.shuffle(letters)
        
        for ci,c in enumerate(choice):
            label=letters[ci]
            if c==answer:
                answer_set[di]=(answer,label)
            if c['w1']=='empty1':
                continue

            print(label,' : ',c['w1'] ,' is to ', c['w2'])
        r=d['r']
        R.append(r)

        print('###########')
    print('answer_set',)
    print(answer_set)
    


def plot_permutations_dist(data_name):

    import matplotlib.pyplot as plt
    import numpy as np

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    def flatten_extend(matrix):
        flat_list = []
        for row in matrix:
            if type(row)==type([]):
                flat_list.extend(row)
            else:
                flat_list.append(row)
        return flat_list
    file='essential_files/histogramfile.json'
    histogramfile= json.load(open(file))['histogramfile']

    data_name=data_name

    data_name='sat'
    sat_perpendicular=histogramfile['perpendicular'][data_name]
    sat_valid=histogramfile['valid'][data_name]

    sat_crossed=histogramfile['crossed'][data_name]
    sat_negative=histogramfile['negative'][data_name]

    ##
    data_name='ekar'
    ekar_perpendicular=histogramfile['perpendicular'][data_name]
    ekar_valid=histogramfile['valid'][data_name]

    ekar_crossed=histogramfile['crossed'][data_name]
    ekar_negative=histogramfile['negative'][data_name]
    ###

    data_name='google'
    google_perpendicular=histogramfile['perpendicular'][data_name]
    google_valid=histogramfile['valid'][data_name]

    google_crossed=histogramfile['crossed'][data_name]
    google_negative=histogramfile['negative'][data_name]

    ###

    data_name='u2'
    u2_perpendicular=histogramfile['perpendicular'][data_name]
    u2_valid=histogramfile['valid'][data_name]

    u2_crossed=histogramfile['crossed'][data_name]
    u2_negative=histogramfile['negative'][data_name]

    def return_df(data_L,dataname):
        lists=data_L
        names=['perpendicular','valid','crossed','negative']
        Names=[]
        Data=[]
        Data_Name=[]
        for L,name in zip(lists,names):
            for l in L:
                Data.append(l)
                Names.append(name)
                Data_Name.append(dataname)
        import pandas as pd

        data = {
          "Offset Similarity": Data,
          "Permutation": Names,
          'data_name':Data_Name,

        }
        #load data into a DataFrame object:
        df = pd.DataFrame(data)
        return df





    data_L=[sat_perpendicular,sat_valid,sat_crossed,sat_negative]
    dataname='sat'

    sat_df=return_df(data_L,dataname)

    data_L=[ekar_perpendicular,ekar_valid,ekar_crossed,ekar_negative]
    dataname='ekar'
    ekar_df=return_df(data_L,dataname)


    data_L=[google_perpendicular,google_valid,google_crossed,google_negative]
    dataname='google'
    google_df=return_df(data_L,dataname)
    ##
    data_L=[u2_perpendicular,u2_valid,u2_crossed,u2_negative]
    dataname='u2'
    u2_df=return_df(data_L,dataname)


    df = pd.concat([sat_df, ekar_df,google_df,u2_df], ignore_index=True)


    sns.set_context("notebook", font_scale=1.8)

    g = sns.displot(
        data=df, x="Offset Similarity", col="data_name", hue='Permutation', col_wrap=2,  
        height=4, aspect=1.2, kind="kde",
    )

    #g.set_titles("Data: {col_name}")
    g.set_titles("")
    #sns.set_position([0.2, 1.1])
    #g.tight_layout()

    sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False)
    #sns.move_legend(g, "upper left", bbox_to_anchor=(.85, .45), title='Permutations:')

    for ax in g.axes:
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_xlabel("Offset Similarity ", visible=True)
        #ax.set_ylabel("Body mass", visible=True)
    plt.tight_layout()

    #g.move_legend(ax, "top right")

    # Single legend for the whole figure
    #fig.legend(handles, labels, loc='upper center', ncol=4)
    #plt.tight_layout(rect=[1, 0, 0, 0.99])  # leave space for legend
    plt.savefig('images/sim_dis/combined.png')
    plt.close()





def get_perm_similarities(histogramfile,data_name,model_name,valid_p):


    nearest_neighbour=False

    import numpy as np
    from sklearn.decomposition import PCA
    import pandas as pd


    from sklearn.decomposition import PCA as sklearnPCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE

    import matplotlib.pyplot as plt
    import numpy as np



    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)


    file='essential_files/wordanalogy_'+'DATA_multi_choice.json'


    file='essential_files/wordanalogy_'+'DATA_words_fast_text.json'
    choice_from_no_rel='inv'

    file='essential_files/wordanalogy_'+'DATA_words_fast_text.json'

    data= json.load(open(file))['DATA_multi_choice']

    print('data',data['test'].keys())
    print('data',data['train'].keys())

    selected_data=data_name


    DATA_Test=data['test'][selected_data]
    DATA_Train=data['train'][selected_data]
    model=get_model(model_name)

    wod_lenth=1
    correct=0
    total=0
    not_correct=0
    total=0
    fl=True

    import numpy as np



    similarity=[]
    for word in DATA_Test:
        #print('__')
        answer=word['answer']
        stem_w1=word['stem']['w1']
        stem_w2=word['stem']['w2']

        answer=word['answer']

        choice=word['choice']
        answer=choice[answer]
        w3=answer['w1']
        w3=w3[1:] if w3[0]=='"' else w3
        w3=w3[:-1] if w3[-1]=='"' else w3

        w4=answer['w2']

        w4=w4[1:] if w4[0]=='"' else w4
        w4=w4[:-1] if w4[-1]=='"' else w4



        stem_w3=w3#word['stem']['w1']
        stem_w4=w4#word['stem']['w1']
        ##
        stem_w1=stem_w1[1:] if stem_w1[0]=='"' else stem_w1
        stem_w1=stem_w1[:-1] if stem_w1[-1]=='"' else stem_w1

        stem_w2=stem_w2[1:] if stem_w2[0]=='"' else stem_w2
        stem_w2=stem_w2[:-1] if stem_w2[-1]=='"' else stem_w2

        stem_w3=stem_w3[1:] if stem_w3[0]=='"' else stem_w3
        stem_w3=stem_w3[:-1] if stem_w3[-1]=='"' else stem_w3

        stem_w4=stem_w4[1:] if stem_w4[0]=='"' else stem_w4
        stem_w4=stem_w4[:-1] if stem_w4[-1]=='"' else stem_w4

        s=get_offset_sim(model,stem_w1,stem_w2,stem_w3,stem_w4,model_name,sim_f='cosine_similarity')


        #s_t=s.item()
        similarity.append(s)


    arr = np.array(similarity)

    data=arr
    model_name

    selected_data

    histogramfile[valid_p][selected_data]=similarity

    file='essential_files/histogramfile.json'
    h_data={'histogramfile':histogramfile}
    with open(file, 'w') as fp:
        json.dump(h_data, fp)



def plot_agreement_heat_map(data,modelnames):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np


    models = modelnames


    df = pd.DataFrame(data, index=models, columns=models)


    d=[]

    for d_ in data:
        data_minus1_to_1_list = [(2 * v - 1.) if v!=1 else 1  for v in d_]
        d.append(data_minus1_to_1_list)
        
    data=d

    df2 = pd.DataFrame(d, index=models, columns=models)

    annot_df = pd.DataFrame(
        np.full_like(df.values, '', dtype=object), 
        index=df.index, 
        columns=df.columns
    )


    annot_df.loc['prophetnet', 'bertB'] = 'MAX!'
    annot_df.loc['t5L', 'robertaB'] = 'Min'


    for i in range(len(models)):
        for j in range(len(models)):

            annot_df.iloc[i, j] = f"{df2.iloc[i, j]:.2f}"

    # --- 3. Generate the Plot with Custom Annotations ---
    plt.figure(figsize=(9, 8))
    cmap = sns.color_palette("crest", as_cmap=True)

    sns.heatmap(
        df,
        annot=annot_df,          
        fmt='s',                 
        cmap=cmap,               
        linewidths=0.5,          
        cbar=True,               
        vmin=0.55,               
        vmax=1.00,
        annot_kws={"fontsize": 15, "fontweight": "bold"} 
    )

    plt.title('Offset Similarity Agreement between Models for Semeval 2012', fontsize=15, pad=20)
    plt.xticks(rotation=70)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('images/itermodel_agreement.png')
    plt.close()

def interagreement_between_models(m1,m2,dn,plot=True):


    import matplotlib.pyplot as plt
    import numpy as np
    # ##
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import math


    model1=get_model(m1)
    model2=get_model(m2)

    file='essential_files/wordanalogy_'+'DATA_multi_choice.json'


    file='essential_files/wordanalogy_'+'DATA_words_fast_text.json'
    choice_from_no_rel='equ'
    file='essential_files/wordanalogy_'+'DATA_words_fast_text.json'
    data= json.load(open(file))['DATA_multi_choice']
    print('data',data['test'].keys())
    print('data',data['train'].keys())

    ###
    x=[]
    y=[]
    data_test_EVALution=data['test'][dn]
    data_train_EVALution=data['train'][dn]

    for word in data_test_EVALution:
        #print('__')
        answer=word['answer']
        stem_w1=word['stem']['w1']
        stem_w2=word['stem']['w2']

        answer=word['answer']

        choice=word['choice']
        answer=choice[answer]
        w3=answer['w1']
        w3=w3[1:] if w3[0]=='"' else w3
        w3=w3[:-1] if w3[-1]=='"' else w3

        w4=answer['w2']

        w4=w4[1:] if w4[0]=='"' else w4
        w4=w4[:-1] if w4[-1]=='"' else w4



        stem_w3=w3#word['stem']['w1']
        stem_w4=w4#word['stem']['w1']
        ##
        stem_w1=stem_w1[1:] if stem_w1[0]=='"' else stem_w1
        stem_w1=stem_w1[:-1] if stem_w1[-1]=='"' else stem_w1

        stem_w2=stem_w2[1:] if stem_w2[0]=='"' else stem_w2
        stem_w2=stem_w2[:-1] if stem_w2[-1]=='"' else stem_w2

        stem_w3=stem_w3[1:] if stem_w3[0]=='"' else stem_w3
        stem_w3=stem_w3[:-1] if stem_w3[-1]=='"' else stem_w3

        stem_w4=stem_w4[1:] if stem_w4[0]=='"' else stem_w4
        stem_w4=stem_w4[:-1] if stem_w4[-1]=='"' else stem_w4

        offsetsim_model1=get_offset_sim(model1,stem_w1,stem_w2,stem_w3,stem_w4,m1)

        offsetsim_model2=get_offset_sim(model2,stem_w1,stem_w2,stem_w3,stem_w4,m2)



        if offsetsim_model1!=None and offsetsim_model2!=None:

            x.append(offsetsim_model1)
            y.append(offsetsim_model2)

    if plot:
        import pandas as pd

        data = {
          "x": x,
          "y": y,

        }
        #load data into a DataFrame object:
        df = pd.DataFrame(data)
        fig = plt.figure(figsize=(12, 12))
        sns.set_context("notebook", font_scale=1.5)

        g=sns.displot(df, x="x", y="y",kind="kde")
        
        #plt.title("offset similarity "+str(dn))
        if m2=='bert-large-uncased' :
            m2='BertL'
        g.fig.suptitle('DATA : '+ str(dn))#, fontsize=10)
        plt.xlabel('Offset sim: '+str(m1))#,fontsize=10)
        plt.ylabel('Offset sim: '+str(m2))#,fontsize=10)

        plt.savefig('images/'+str(m1)+'and'+str(m2)+str(dn)+'.png')
        plt.close()

        print('test',dn)
   

        return
    else:

        distance_all=[]
        for xi,yi in zip(x,y):
            sim_distrance=(xi-yi)**2
            #Import math Library
            sim_distrance=math.sqrt(sim_distrance)*1000
            
            v=1/math.exp(sim_distrance)


            distance_all.append(v)
        return distance_all

@torch.no_grad()
def solve_analogies(modelname,dn,not_corrct_terms,model=None,model_type='baseline',sim_f='offset_cosine_similarity',bayesian_analysis=False):


    nearest_neighbour=False



    import numpy as np
    from sklearn.decomposition import PCA
    import pandas as pd


    from sklearn.decomposition import PCA as sklearnPCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE



    file='essential_files/wordanalogy_'+'DATA_multi_choice.json'


    file='essential_files/wordanalogy_'+'DATA_words_fast_text.json'

    data= json.load(open(file))['DATA_multi_choice']

    print('data',data['test'].keys())
    print('data',data['train'].keys())

    data_test_EVALution=data['test'][dn]
    data_train_EVALution=data['train'][dn]

    #model=
    model_name=modelname

    if model==None:
        model=get_model(model_name)

        

    probabilities_s_given_r={'True':[],'False':[] }
    probabilities_r_given_s=[]




    wod_lenth=1
    correct=0
    total=0
    not_correct=0
    total=0
    fl=True
    not_found=0
    correct_rel_based_acc={}
    joint_distribution_r_s=[]
    scan_acc={'science':{'correct':0,'not_correct':0}, 'metaphor':{'correct':0,'not_correct':0}}
    for word in data_test_EVALution:


        eval_d=word['eval_d'] if 'eval_d' in word.keys() else None


        answer=word['answer']
        stem_w1=word['stem']['w1']
        stem_w2=word['stem']['w2']

        stem_w1=stem_w1[1:] if stem_w1[0]=='"' else stem_w1
        stem_w1=stem_w1[:-1] if stem_w1[-1]=='"' else stem_w1

        stem_w2=stem_w2[1:] if stem_w2[0]=='"' else stem_w2
        stem_w2=stem_w2[:-1] if stem_w2[-1]=='"' else stem_w2


        total=total+1

        fl=True
        w1,w2=None,None
        model_name
        model

        if model_type=='baseline':
            w1=get_embedding(model,stem_w1,model_name)
            w2=get_embedding(model,stem_w2,model_name)

            stem_d=w1-w2
        else:

            stem_d=model.get_words_offset(stem_w1,stem_w2)
   
        choice=word['choice']
        r=word['r']
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        similarity=[]
        pca_arr=[stem_d,]
        pca_wordpairs=[stem_w1+'-'+stem_w2,]
        #print('choice',len(choice))
        answer_w1,answer_w2=None,None
        for ci,t in enumerate(choice):
            t_w1=t['w1']
            t_w2=t['w2']

  

            if len(t_w1)==0 or len(t_w2)==0:
                continue
            #print('t_w1',t_w1)
            if 'semeval_2012' in dn:
                t_w1=t_w1[1:] if t_w1[0]=='"' else t_w1
                t_w1=t_w1[:-1] if t_w1[-1]=='"' else t_w1

                

                t_w2=t_w2[1:] if t_w2[0]=='"' else t_w2
                t_w2=t_w2[:-1] if t_w2[-1]=='"' else t_w2
            if ci==answer:
                answer_w1,answer_w2=t_w1,t_w2


            pca_wordpairs.append(t_w1+'-'+t_w2)

            ###
            if model_type=='baseline':
                x1=get_embedding(model,t_w1,model_name)
                x2=get_embedding(model,t_w2,model_name)

                x_d=x1-x2
            else:
                x_d=model.get_words_offset(t_w1,t_w2)


            if sim_f=='offset_cosine_similarity':
                s=cos(stem_d,x_d)
            elif sim_f=='distance':
                s1=torch.nn.functional.pairwise_distance(w1, x1)
                s2=torch.nn.functional.pairwise_distance(w2, x2)
                s=(s1+s2)/2
                s=1/s
            elif sim_f=='pairwise_sim':
                s1=cos(w1, x1)
                s2=cos(w2, x2)
                s=(s1+s2)/2
           


            similarity.append(s.item())



        predicted=similarity.index(max(similarity))
        ##
        if len(similarity)<answer:
            print('choice',choice)
            print('answer',answer)
            print('word[]',word['stem'])
            #exit()
        sim_answer=similarity[answer-1] if len(similarity)<1 else 0

        

        def check_in(not_found,Ws,not_corrct_terms):
            Ws=[w.lower() for w in Ws]
            T=False
            w_cat_dic={'w1':'0','w2':'0','w3':'0','w4':'0'}
            for w,wn in zip(Ws,list(w_cat_dic.keys())):


                if w in not_corrct_terms.keys():
                    c,nc=not_corrct_terms[w]['correct'],not_corrct_terms[w]['not_correct']
                    n_c,n_nc=len(c),len(nc)
                    avg_c=sum(c)/n_c if n_c!=0 else 0
                    avg_nc=sum(nc)/n_nc if n_nc!=0 else 0

                    if n_c>=n_nc:
                        w_cat_dic[wn]=('True',avg_c)
                        
                    else:
                        w_cat_dic[wn]=('False',avg_nc)

                else:
                    not_found=not_found+1
                    w_cat_dic[wn]=('0',0)

                    continue

            tot=[t for t in w_cat_dic.values()]
            tot_=[]
            for t in tot:
                if t!='0':
                    tot_.append(t)

            return not_found,w_cat_dic






        plot_dic={}
        if predicted==answer:

            if eval_d and dn=='scan':
                scan_acc[eval_d]['correct']=scan_acc[eval_d]['correct']+1
            correct=correct+1
            ##
            stem_w1
            stem_w1
            answer_w1
            answer_w2
            answer
            L=[stem_w1,stem_w2,answer_w1,answer_w2]
            not_found,w_cat_dic=check_in(not_found,L,not_corrct_terms)

            for v in w_cat_dic.values():
                if v=='0':
                    continue
                p=(('True',sim_answer),v)

                joint_distribution_r_s.append(p)
        


            if r in correct_rel_based_acc.keys():
                correct_rel_based_acc[r]['correct']=correct_rel_based_acc[r]['correct']+1
            else:
                correct_rel_based_acc[r]={'correct':0,'not_correct':0}
                correct_rel_based_acc[r]['correct']=1
        else:

            if eval_d and dn=='scan':
                scan_acc[eval_d]['not_correct']=scan_acc[eval_d]['not_correct']+1
            not_correct=not_correct+1
            if r in correct_rel_based_acc.keys():
                correct_rel_based_acc[r]['not_correct']=correct_rel_based_acc[r]['not_correct']+1
            else:
                correct_rel_based_acc[r]={'correct':0,'not_correct':0}
                correct_rel_based_acc[r]['not_correct']=1
            ##

            L=[stem_w1,stem_w2,answer_w1,answer_w2]
            not_found,w_cat_dic=check_in(not_found,L,not_corrct_terms)
            for v in w_cat_dic.values():
                if v=='0':
                    continue
                #p=('False',v)
                p=(('False',sim_answer),v)

                joint_distribution_r_s.append(p)

    print('correct',correct)
    print('not_correct',not_correct)
    print('total',total)
    acc=correct/(correct+not_correct)
    print('acc',acc)
    print('data',dn)
    print('model_name',model_name)
    if dn=='scan':
        #print('scan_acc',scan_acc)
        science=scan_acc['science']['correct']/(scan_acc['science']['correct']+scan_acc['science']['not_correct'])
        print('science',science)
        metaphor=scan_acc['metaphor']['correct']/(scan_acc['metaphor']['correct']+scan_acc['metaphor']['not_correct'])
        print('metaphor',metaphor)


    if bayesian_analysis==False:
        return acc

    #return acc
    def conditional(Data_joint,condition_on,conditoin_v):
        data=[]
        if condition_on=='relation':
            for row in Data_joint:
                r,s=row[0][0],row[1][0]
                if r==conditoin_v:
                    data.append(row)
      
        elif condition_on=='semantic':
            for row in Data_joint :
                r,s=row[0][0],row[1][0]
                if s==conditoin_v:
                    data.append(row)
       
        return data
    ###

    condition_on='semantic'
    conditoin_v='True'
    joint_data=joint_distribution_r_s.copy()
    st=conditional(joint_data,condition_on,conditoin_v)

    condition_on='semantic'
    conditoin_v='False'
    joint_data=joint_distribution_r_s.copy()
    sf=conditional(joint_data,condition_on,conditoin_v)

    print('st',len(st)/(len(st)+len(sf)))

    ##
    condition_on='relation'
    conditoin_v='True'
    joint_data=joint_distribution_r_s.copy()
    rt=conditional(joint_data,condition_on,conditoin_v)

    condition_on='relation'
    conditoin_v='False'
    joint_data=joint_distribution_r_s.copy()
    rf=conditional(joint_data,condition_on,conditoin_v)

    print('rt',len(rt)/(len(rt)+len(rf)))





    ###


    
    condition_on='semantic'
    conditoin_v='True'
    joint_data=joint_distribution_r_s.copy()
    r_given_st=conditional(joint_data,condition_on,conditoin_v)



    condition_on='relation'
    conditoin_v='True'
    joint_data=r_given_st.copy()
    rt_given_st=conditional(joint_data,condition_on,conditoin_v)

    condition_on='relation'
    conditoin_v='False'
    joint_data=r_given_st.copy()
    rf_given_st=conditional(joint_data,condition_on,conditoin_v)



    p_rt_given_st=len(rt_given_st)/len(r_given_st)
    p_rf_given_st=len(rf_given_st)/len(r_given_st)
    print('p_rt_given_st+p_rf_given_st',p_rt_given_st+p_rf_given_st)
    ###################################################

    condition_on='semantic'
    conditoin_v='False'
    joint_data=joint_distribution_r_s.copy()
    r_given_sf=conditional(joint_data,condition_on,conditoin_v)

    condition_on='relation'
    conditoin_v='True'
    joint_data=r_given_sf.copy()
    rt_given_sf=conditional(joint_data,condition_on,conditoin_v)

    condition_on='relation'
    conditoin_v='False'
    joint_data=r_given_sf.copy()
    rf_given_sf=conditional(joint_data,condition_on,conditoin_v)

    p_rt_given_sf=len(rt_given_sf)/len(r_given_sf)
    p_rf_given_sf=len(rf_given_sf)/len(r_given_sf)

    print('p_rt_given_sf+p_rf_given_sf',p_rt_given_sf+p_rf_given_sf)
    ##########################################################################



    condition_on='relation'
    conditoin_v='True'
    joint_data=joint_distribution_r_s.copy()
    s_given_rt=conditional(joint_data,condition_on,conditoin_v)

    condition_on='semantic'
    conditoin_v='True'
    joint_data=s_given_rt.copy()
    st_given_rt=conditional(joint_data,condition_on,conditoin_v)

    condition_on='semantic'
    conditoin_v='False'
    joint_data=s_given_rt.copy()
    sf_given_rt=conditional(joint_data,condition_on,conditoin_v)



    p_st_given_rt=len(st_given_rt)/len(s_given_rt)
    p_sf_given_rt=len(sf_given_rt)/len(s_given_rt)

    print('p_st_given_rt+p_sf_given_rt',p_st_given_rt+p_sf_given_rt)

    ###################################################

    condition_on='relation'
    conditoin_v='False'
    joint_data=joint_distribution_r_s.copy()
    s_given_rf=conditional(joint_data,condition_on,conditoin_v)

    condition_on='semantic'
    conditoin_v='True'
    joint_data=s_given_rf.copy()
    st_given_rf=conditional(joint_data,condition_on,conditoin_v)

    condition_on='semantic'
    conditoin_v='False'
    joint_data=s_given_rf.copy()
    sf_given_rf=conditional(joint_data,condition_on,conditoin_v)

    p_st_given_rf=len(st_given_rf)/len(s_given_rf)
    p_sf_given_rf=len(sf_given_rf)/len(s_given_rf)
    print('p_st_given_rf+p_sf_given_rf',p_st_given_rf+p_sf_given_rf)





    print('p_rt_given_sf',p_rt_given_sf)
    print('p_rf_given_sf',p_rf_given_sf)

    print('p_rt_given_st',p_rt_given_st)
    print('p_rf_given_st',p_rf_given_st)
    print('##')

    print('p_st_given_rf',p_st_given_rf)
    print('p_sf_given_rf',p_sf_given_rf)

    print('p_st_given_rt',p_st_given_rt)
    print('p_sf_given_rt',p_sf_given_rt)
    print('###################')

    import numpy as np
    from scipy.stats import chi2_contingency
    import pandas as pd

    joint_distribution_r_s_=[]
    for r in joint_distribution_r_s:
        if r[0][0]=='0' or r[1][0]=='0':
            continue
        joint_distribution_r_s_.append(r)



    data1=[t[0][0] for  t in joint_distribution_r_s_]
    data2=[ t[1][0] for t in joint_distribution_r_s_]
    import pandas as pd
    from scipy.stats import chi2_contingency

    data = {
        'offset_similarity': data1,
        'word_similarity': data2
    }
    df = pd.DataFrame(data)

    contingency_table = pd.crosstab(df['offset_similarity'], df['word_similarity'])

    print("--- Summary Table (Counts) ---")
    print(contingency_table)
    print("-" * 30)


    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-Square Statistic: {chi2:.6f}")
    print(f"P-value: {p:.6f}")


    return acc



def fetch_conceptqa(source='google'):
    import numpy as np
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    #item={'q':q,'answer':answer,'choices':choices,'key':k}
    if source=='google':
        #get_google_abstract_space_dic(embeddings=False,model=None,model_name=None,semeval_or_google='google',f='semantic_space_dic')
        abstract_spaces_dic=get_google_abstract_space_dic()
        Questions=get_conceptqa_questions(abstract_spaces_dic,n_size=5)
    elif source=='wikidata':
       # abstract_spaces_dic=get_abstract_spaces_dic_w()
        abstract_spaces_dic=get_google_abstract_space_dic(semeval_or_google='wikidata')
        Questions=get_conceptqa_questions(abstract_spaces_dic,n_size=50)

    
    else:

        abstract_spaces_dic=get_google_abstract_space_dic(semeval_or_google='semeval_2012')
        Questions=get_conceptqa_questions(abstract_spaces_dic,n_size=5)
    data={'X':[],'y':[]}

    total=0
    for Q in Questions:
        q=Q['q']
        answer=Q['answer']
        choices=Q['choices']
        key=Q['key']
        t=[q,answer]
        if q==answer:
            continue
        y=1.
        data['X'].append(t)
        data['y'].append(y)
        total=total+1

        j=0
        for ci, c in enumerate(choices):
            if j==3:
                break
            if c==answer or q==c:
                y=1.
                #print('c',c,answer,q)
                #exit()
                continue
            else:
                y=-1.
            j=j+1
            t=[q,c]
            data['X'].append(t)
            data['y'].append(y)
            total=total+1

            #break
    data['X']= np.array(data['X'])
    data['y'] = np.array(data['y'])
    data = dotdict(data)
    return data







@torch.no_grad()
def do_table_6(model_name,model=None):

    from benchmarks.web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999

    #from benchmarks.web.analogy import *

    from benchmarks.web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW, fetch_TR9856


    if model==None:
        model=get_model(model_name)
    #fetch_conceptqa()

    seed = 907
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tasks = {
        # "MEN": fetch_MEN(),
        # "WS353": fetch_WS353(),
        # "SIMLEX999": fetch_SimLex999(),
        # 'conceptqa_google':fetch_conceptqa(source='google'),
        # 'conceptqa_semeval':fetch_conceptqa(source='semeval'),
        'conceptqa_wikidata':fetch_conceptqa(source='wikidata')
    }
    print('successs')

    corr_simlex = []

    for d in tasks.keys():
        for epoch in range(3):
            seed = 907

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)


            A = np.vstack([get_embedding(model,word,model_name)[0].cpu() for word in tasks[d].X[:, 0]])
            B = np.vstack([get_embedding(model,word,model_name)[0].cpu() for word in tasks[d].X[:, 1]])
            scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
            corr_simlex.append(scipy.stats.spearmanr(scores, tasks[d].y,nan_policy='omit').correlation)
        print('data',d)
        print("Tuned", sum(corr_simlex)/1)




def plot_sim_with_abs(abstract,model_name,all_abstracts,all_words):

    #################
    model=get_model(model_name)
    file='unprocessed_data/SRE_Analogy.json'
    train_or_dev='train'
    SRE_Analogy= json.load(open(file))['SRE_Analogy']['wikidata'][train_or_dev]

    from torch import linalg as LA
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    x_axis=[]
    y_axis=[]
    h_sim=[]
    t_sim=[]
    offset_sim=[]


    benchmarks='wikidata'
    train_eval='train'
    j=0
    all_abstracts=list(all_abstracts)
    all_words=list(all_words)
    if abstract=='random_word':
        all_abstracts=all_words
    #data_type


    for d in SRE_Analogy:
        if j==2000:
            break
        j=j+1
        rel=d['r']
        type_=d['type']
        #print('d',d)
        #exit()
        kbID=d['kbID'] if (d['kbID']!='semeval' and d['kbID']!='retacred') else None

        #item={'data_name':data_type,'data':d}
        if kbID!=None:
            rel=kbID
        if type_!=train_eval:
            continue

        t_w1=d['a']
        t_w2=d['b']
        e1_abstract=d['e1_abstract']
        e2_abstract=d['e2_abstract']

        ##
        if 'random' not in abstract:
            e1_abstract=e1_abstract.split(' ')
            e2_abstract=e2_abstract.split(' ')
           
            X=['X','Y','A','B','C','G','H','K','L','M','N','P','Q',]
            X=[t.lower() for t in X]
            e1_abstract=[t if t.lower() not in X   else '' for t in e1_abstract ]
            e1_abstract=' '.join(e1_abstract)

            e2_abstract=[t if t.lower() not in X   else '' for t in e2_abstract ]
            e2_abstract=' '.join(e2_abstract) 
        else:
            L=[si for si in range(len(all_abstracts))]
            #print('L',L)
            s=random.sample(L,2)
            population=all_abstracts
            selected=[population[j_] for j_ in s]

            e1_abstract=selected[0]
            e2_abstract=selected[1]


        if model_name in all_models_names:
            tw1_e=model.get_word_embed(t_w1)
            tw2_e=model.get_word_embed(t_w2)
            e1_abstract_e=model.get_word_embed(e1_abstract)
            e2_abstract_e=model.get_word_embed(e2_abstract)

        elif model_name=='fasttext':


            tw1_e=model.get_word_vector(t_w1)
            tw2_e=model.get_word_vector(t_w2)
            e1_abstract_e=model.get_word_vector(e1_abstract)
            e2_abstract_e=model.get_word_vector(e2_abstract)
        ############




        x_d=tw1_e -tw2_e

        x_d_abstract=e1_abstract_e -e2_abstract_e
        ##
        x_d=torch.from_numpy(x_d).unsqueeze(0)
        x_d_abstract=torch.from_numpy(x_d_abstract).unsqueeze(0)

        offsetsim=cos(x_d ,x_d_abstract )
        print('+++++')
        offset_legnth=LA.vector_norm(x_d)
        offset_legnth_abstract=LA.vector_norm(x_d_abstract)
        print('offset_legnth',offset_legnth)
        print('offset_legnth_abstract',offset_legnth_abstract)

        if e2_abstract==e1_abstract:
            continue



        print('t_w1',t_w1)
        print('e1_abstract',e1_abstract)
        print('****')

        print('t_w2',t_w2)
        print('e2_abstract',e2_abstract)
        print('offsetsim',offsetsim)
        
        tw1_e=torch.from_numpy(tw1_e).unsqueeze(0)
        tw2_e=torch.from_numpy(tw2_e).unsqueeze(0)

        e1_abstract_e=torch.from_numpy(e1_abstract_e).unsqueeze(0)
        e2_abstract_e=torch.from_numpy(e2_abstract_e).unsqueeze(0)

        head_sim=cos(tw1_e ,e1_abstract_e )
        tail_sim=cos(tw2_e ,e2_abstract_e )
        print('head_sim',head_sim)
        print('tail_sim',tail_sim)

        head_tail_sim=(head_sim.item()+tail_sim.item())/2
        ###


        print('head_tail_sim',head_tail_sim)
        print('offsetsim',offsetsim)
        print('########################')
        print('head_sim.item()',head_sim.item())
        h=head_sim.item()
        t=tail_sim.item()
        offset=offsetsim.item()
        #sim=sim+(1-sim)/2#1/math.exp(200*(-sim))
        y_axis.append(head_tail_sim)
        x_axis.append(offset)
        
        h_sim.append(h)
        t_sim.append(t)
        offset_sim.append(offset)
        print('h,t,offset',h,t,offset)


    ############################################################################

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    # ##
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    data = {
      "x": x_axis,
      "y": y_axis,

    }
    #load data into a DataFrame object:
    df = pd.DataFrame(data)
    fig = plt.figure(figsize=(12, 12))
    sns.set_context("notebook", font_scale=1.5)

    g=sns.displot(df, x="x", y="y",kind="kde")
    

    #g.fig.suptitle(str(model_name))#, fontsize=10)
    plt.xlabel('Offset sim')#,fontsize=10)
    plt.ylabel('avg head-tail sim ')#,fontsize=10)

    plt.savefig('images/sim_dis/Heat_tail_sim_vs_offset_sim'+str(abstract)+str(model_name)+'.png')
    plt.close()

    return





def same_rel_sim(model_name,all_related):

    model=get_model(model_name)
    head_tail_sim_all=[]
    min_n=1
    max_n=0
    offsetsim_all=[]

    rel_dic={}
    for ki,k in enumerate(all_related.keys()):
        population=all_related[k]
        L=[si for si in range(len(population))]
        N=10 if len(L)>10 else len(L)
        s=random.sample(L,N)
        selected=[population[j_] for j_ in s]
        for s1 in selected:
            w1=s1['w1']
            w2=s1['w2']
            for s2 in selected:

                w3=s2['w1']
                w4=s2['w2']
                #print('w1,w3 , w2,w4',w1,w3 , w2,w4)
                if w1==w3 and w2==w4:
                    continue
                head_sim=get_sim(model,w1,w3,model_name)
                tail_sim=get_sim(model,w2,w4,model_name)

                head_tail_sim=(head_sim+tail_sim)/2
                offsetsim=get_offset_sim(model,w1,w2,w3,w4,model_name)
                offsetsim=offsetsim#.item()
                ##
                print('head_tail_sim',head_tail_sim)
                print('offsetsim',offsetsim)
                head_tail_sim_all.append(head_tail_sim)
                offsetsim_all.append(offsetsim)

                if k in rel_dic.keys():
                    rel_dic[k]['head_tail_sim'].append(head_tail_sim)
                    rel_dic[k]['offsetsim'].append(offsetsim)
                else:
                    rel_dic[k]={'head_tail_sim':[],'offsetsim':[]}
            

                    rel_dic[k]['head_tail_sim'].append(head_tail_sim)
                    rel_dic[k]['offsetsim'].append(offsetsim)

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    # ##
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    data = {
      "x": offsetsim_all,
      "y": head_tail_sim_all,

    }
    #load data into a DataFrame object:
    df = pd.DataFrame(data)
    fig = plt.figure(figsize=(12, 12))
    sns.set_context("notebook", font_scale=1.5)

    g=sns.displot(df, x="x", y="y",kind="kde")
    

    #g.fig.suptitle(str(model_name))#, fontsize=10)
    plt.xlabel('Offset sim')#,fontsize=10)
    plt.ylabel('avg head-tail sim ')#,fontsize=10)
    k=k.replace('\\','#')
    k=k.replace('/','#')
    k=k.split(' ')
    k='-'.join(k)

    plt.savefig('images/sim_dis/'+str(model_name)+'.png')
    plt.close()





def similarity_in_same_rel(model,model_name,all_related):

    dn='wikidata'
    head_tail_sim_all=[]
    min_n=1
    max_n=0
    offsetsim_all=[]

    rel_dic={}

    file='essential_files/wordanalogy_'+'DATA_multi_choice.json'


    file='essential_files/wordanalogy_'+'DATA_words_fast_text.json'
    choice_from_no_rel='inv'
    file='essential_files/wordanalogy_'+'DATA_words_fast_text.json'
    data= json.load(open(file))['DATA_multi_choice']
    print('data',data['test'].keys())
    print('data',data['train'].keys())

    ###
    x=[]
    y=[]
    #names=['BLESS','CogALexV','EVALution','ROOT09','KandH_plus_N']
    #['semeval_2012','sre','analogykb','ekar','RS','scan','google','bats','u4','u2','sat']
    data_test_EVALution=data['test'][dn]
    data_train_EVALution=data['train'][dn]

    similar_relation=False

    for word in data_test_EVALution:
        #print('__')
        answer=word['answer']
        stem_w1=word['stem']['w1']
        stem_w2=word['stem']['w2']

        answer=word['answer']

        choice=word['choice']
        answer=choice[answer]
        w3=answer['w1']
        w3=w3[1:] if w3[0]=='"' else w3
        w3=w3[:-1] if w3[-1]=='"' else w3

        w4=answer['w2']

        w4=w4[1:] if w4[0]=='"' else w4
        w4=w4[:-1] if w4[-1]=='"' else w4

        for c in choice:

            if similar_relation==False:

                w3=c['w1']
                w3=w3[1:] if w3[0]=='"' else w3
                w3=w3[:-1] if w3[-1]=='"' else w3

                w4=c['w2']

                w4=w4[1:] if w4[0]=='"' else w4
                w4=w4[:-1] if w4[-1]=='"' else w4




            if similar_relation==False and stem_w1==w3 and stem_w2==w4:
                continue


            head_sim=get_sim(model,stem_w1,w3,model_name)
            tail_sim=get_sim(model,stem_w2,w4,model_name)

            head_tail_sim=(head_sim+tail_sim)/2






            offsetsim=get_offset_sim(model,stem_w1,stem_w2,w3,w4,model_name)
            offsetsim=offsetsim#.item()
            head_tail_sim_all.append(head_tail_sim)
            offsetsim_all.append(offsetsim)

            if head_tail_sim<0:
                print('head_tail_sim',head_tail_sim)

            if head_sim<0:
                print('head_sim',head_sim)
            if tail_sim<0:
                print('tail_sim',tail_sim)

            if similar_relation:
                break



    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    # ##
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import math



    data = {
      "x": offsetsim_all,
      "y": head_tail_sim_all,

    }
    #load data into a DataFrame object:
    df = pd.DataFrame(data)
    fig = plt.figure(figsize=(12, 12))
    sns.set_context("notebook", font_scale=1.5)

    g=sns.displot(df, x="x", y="y",kind="kde")
    

    #g.fig.suptitle(str(model_name))#, fontsize=10)
    plt.xlabel('Offset sim')#,fontsize=10)
    plt.ylabel('avg head-tail sim ')#,fontsize=10)

    plt.savefig('images/sim_dis/rel2/'+str(choice_from_no_rel)+str(similar_relation)+'-'+str(model_name)+str(dn)+'.png')
    plt.close()

