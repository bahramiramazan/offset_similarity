import os
from os.path import exists
from torch.nn.functional import log_softmax, pad
from transformers import BertTokenizer, BertModel
import logging
import transformers
from transformers.models.bert.configuration_bert import BertConfig 
from transformers.models.bert.modeling_bert import BertEmbeddings 
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

import random
from itertools import combinations
import torch
import torch.nn.functional as F
from analogy_util import _get_pretrained_transformer3, update2_rc,batch_n_rc

import json

import random
import urllib.request

import numpy as np
import torch
from torch import nn
import itertools

def internet_connection():
    try:
        urllib.request.urlopen('http://google.com')
        return True
    except Exception:
        return False


def fix_seed(seed: int = 12, cuda: bool = True):
    """ Fix random seed. """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

## taken from relbert
def loss_triplet(
        tensor_anchor,
        tensor_positive,
        tensor_negative,
        augument=False,
        tensor_positive_parent=None,
        tensor_negative_parent=None,
        margin: float = 1.0,
        linear=None,
        device: str = 'gpu'):
    """ Triplet Loss """
    bce = nn.BCELoss()

    def classification_loss(v_anchor, v_positive, v_negative):
        # the 3-way discriminative loss used in SBERT
        if linear is not None:
            feature_positive = torch.cat([v_anchor, v_positive, torch.abs(v_anchor - v_positive)], dim=1)
            feature_negative = torch.cat([v_anchor, v_negative, torch.abs(v_anchor - v_negative)], dim=1)
            feature = torch.cat([feature_positive, feature_negative])
            pred = torch.sigmoid(linear(feature))
            label = torch.tensor([1] * len(feature_positive) + [0] * len(feature_negative), dtype=torch.float32, device=device)
            return bce(pred, label.unsqueeze(-1))
        return 0

    def main_loss(v_anchor, v_positive, v_negative):
        distance_positive = torch.sum((v_anchor - v_positive) ** 2, -1) ** 0.5
        distance_negative = torch.sum((v_anchor - v_negative) ** 2, -1) ** 0.5
        return torch.sum(torch.clip(distance_positive - distance_negative - margin, min=0))

    loss = main_loss(tensor_anchor, tensor_positive, tensor_negative)
    loss += main_loss(tensor_positive, tensor_anchor, tensor_negative)
    loss += classification_loss(tensor_anchor, tensor_positive, tensor_negative)

    def sample_augmentation(v_anchor, v_positive):
        v_anchor_aug = v_anchor.unsqueeze(-1).permute(2, 0, 1).repeat(len(v_anchor), 1, 1).reshape(len(v_anchor), -1)
        v_positive_aug = v_positive.unsqueeze(-1).permute(2, 0, 1).repeat(len(v_positive), 1, 1).reshape(
            len(v_positive), -1)
        v_negative_aug = v_positive.unsqueeze(-1).permute(0, 2, 1).repeat(1, len(v_positive), 1).reshape(
            len(v_positive), -1)
        return v_anchor_aug, v_positive_aug, v_negative_aug

    # In-batch Negative Sampling
    # No elements in single batch share same relation type, so here we construct negative sample within batch
    # by regarding positive sample from other entries as its negative. The original negative is the hard
    # negatives from same relation type and the in batch negative is easy negative from other relation types.
    if augument:
        a, p, n = sample_augmentation(tensor_anchor, tensor_positive)
        loss += main_loss(a, p, n)
        a, p, n = sample_augmentation(tensor_positive, tensor_anchor)
        loss += main_loss(a, p, n)

        # contrastive loss of the parent class
        if tensor_positive_parent is not None and tensor_negative_parent is not None:
            loss += main_loss(tensor_anchor, tensor_positive_parent, tensor_negative_parent)
            loss += main_loss(tensor_positive_parent, tensor_anchor, tensor_negative_parent)
            loss += classification_loss(tensor_anchor, tensor_positive_parent, tensor_negative_parent)
    return loss

# taken from relbert
def loss_nce(tensor_positive,
             tensor_negative,
             temperature: float = 1.0,
             info_loob: bool = False,
             linear=None,
             device: str = 'gpu'):
    """ NCE loss"""
    ############
    # if tensor_positive.shape[0]>tensor_negative.shape[0]:
    #     tensor_negative=tensor_negative.repeat(tensor_positive.shape[0],1)
    # elif tensor_positive.shape[0]<tensor_negative.shape[0]:
    #     tensor_positive=tensor_positive.repeat(tensor_negative.shape[0],1)

    # print('tensor_positive',tensor_positive.shape)
    # print('tensor_negative',tensor_negative.shape)
    ########################
    bce = nn.BCELoss()
    cos_3d = torch.nn.CosineSimilarity(dim=2)
    eps = 1e-5
    logit_n = torch.exp(
        cos_3d(tensor_positive.unsqueeze(1), tensor_negative.unsqueeze(0)) / temperature
    )
    deno_n = torch.sum(logit_n, dim=-1)  # sum over negative
    logit_p = torch.exp(
        cos_3d(tensor_positive.unsqueeze(1), tensor_positive.unsqueeze(0)) / temperature
    )
    if info_loob:
        loss = torch.sum(- torch.log(logit_p / (deno_n.unsqueeze(-1) + eps)))
    else:
        loss = torch.sum(- torch.log(logit_p / (deno_n.unsqueeze(-1) + logit_p + eps)))
    if linear is not None:
        batch_size_positive = len(tensor_positive)
        for i in range(batch_size_positive):
            features = []
            labels = []
            for j in range(batch_size_positive):
                feature = torch.cat(
                    [tensor_positive[i], tensor_positive[j], torch.abs(tensor_positive[i] - tensor_positive[j])],
                    dim=0)
                features.append(feature)
                labels.append([1])
            for j in range(len(tensor_negative)):
                feature = torch.cat(
                    [tensor_positive[i], tensor_negative[j], torch.abs(tensor_positive[i] - tensor_negative[j])],
                    dim=0)
                features.append(feature)
                labels.append([0])
            pred = torch.sigmoid(linear(torch.stack(features)))
            labels = torch.tensor(labels, dtype=torch.float32, device=device)
            loss += bce(pred, labels)
    return loss


class RoutingHead(nn.Module):
    """Route [n pos, d_depth, d_inp] to [n_out, d_out] ([n_out] if d_out is 1)."""

    def __init__(self, transformer_config, kwds_by_routing):
        super().__init__()
        d_depth, d_inp = (transformer_config['d_depth'], kwds_by_routing[0]['d_inp'])
        self.normalize = nn.LayerNorm(d_inp, elementwise_affine=False)
        self.W = nn.Parameter(torch.ones(d_depth, d_inp))
        self.B = nn.Parameter(torch.zeros(d_depth, d_inp))
        self.route = nn.Sequential(*[Routing(**kwds) for kwds in kwds_by_routing])
        for name, param in self.route.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    nn.init.xavier_normal_(param)


    def forward(self, x):
        x = self.normalize(x)      # [..., n pos, d_depth, d_inp]
        x = x * self.W + self.B    # [..., n pos, d_depth, d_inp]
        x = x.flatten(-3,-2)       # [..., n_inp, d_inp]
        x = self.route(x)          # [..., n_out, d_out]
        return x.squeeze(-1)       # if d_out is 1, remove it


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)





class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob=0.2, bias=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.gelu = nn.GELU()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                x = self.gelu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x




class Relation_Classifier_Model(nn.Module):
    """

    """
    def __init__(
        self,
        args

    ):
        super().__init__()

        vocab_size, d_model = args.vocab_size,args.embed_size

        self.embed_size=args.embed_size
        ####
        args.feed_y0=False
        model_name=args.model_name
        self.pretrained_name=model_name
        #self.tokenizer = get_tokenizer(model_name)
        self.d_model = d_model
        dim=0
        heads={'head_1':512,'head_2':args.h2_embed_size,'head_3':args.h3_embed_size,'head_conditional':args.embed_size}
        for h in heads.keys():
            if h in args.heads:
     
                dim=dim+heads[h]

 
        input_sizes=[dim,dim,dim,args.n_class]
        #input_sizes=[1024*3,1024,768, 256,2]
        self.classifier=MLP(input_sizes)
        for name, param in self.classifier.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    if args.data_type!='wikidata':
                        nn.init.xavier_normal_(param)

        if args.data_type=='semeval_2012':
            file='essential_files/localdatasets/all_relation_dic.json'
        
            with open(file) as f:
                no_rel = json.load(f)['data']['category']
                cls_n=len(no_rel)

        self.linear = torch.nn.Linear(1024 * 3, 1)  # three way feature
        self.linear.weight.data.normal_(std=0.02)
      

        self.head=8
        self.args=args
        self.modality=args.model_name#'roberta-large' if model_name=='roberta-large'  else 'bert-base-uncased'
        tmp=['paper2_WikidataPretraining','paper2_EVALutionPretraining','paper2_lexicalZeroshotTraining','paper2_RE_Trained_lexicalTraining','wordanalogy']

        tokenizer_special_dic='semeval_2012_re' if args.data_type=='semeval_2012' else 're'
        data_selected=args.data_type
        #self.transformer_config, self.transformer = _get_pretrained_transformer2(self.modality) if args.data_type=='wordanalogy' else  _get_pretrained_transformer3(self.modality) 
        self.transformer_config, self.transformer,self.tokenizer =  _get_pretrained_transformer3(data_selected,self.modality,tokenizer_special_dic=tokenizer_special_dic) 


        self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.transformer.resize_token_embeddings(50812)

        ###***head_conditional***
        if 'head_conditional' in args.heads:

            ##########
            decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_size, nhead=self.head,batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
            ##
            for name, param in self.transformer_decoder.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    nn.init.xavier_normal_(param)
                    #nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

            ln_tokenizer = len(self.tokenizer) if model_name=='roberta-large' else len(self.tokenizer)
            self.generator=Generator(self.embed_size,ln_tokenizer)
            #self.PositionalEncoding=PositionalEncoding(self.embed_size)
            
       
            for name, param in self.generator.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    if args.data_type!='wikidata':
                        nn.init.xavier_normal_(param)

                    else:
                        nn.init.xavier_normal_(param)


        self.device=args.device
        
        ###
        config = BertConfig(
                    num_hidden_layers=2, 
                    hidden_act="relu", 
                    num_attention_heads=2,
                    hidden_dropout_prob=0.5, 
                    vocab_size=args.vocab_size ,
                    hidden_size=args.embed_size
                        )

        ###########################################################
        if 'head_1' in args.heads:
            self.nn_linear2= nn.Linear(2,512,bias=False)
            for name, param in self.nn_linear2.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    nn.init.xavier_normal_(param)



        #####
        d_emb = args.embed_size
        n_classes= args.h1_embed_size#args.n_class
        d_hid, n_cls = (d_emb, n_classes)
        n_hid=128
        if 'head_1' in args.heads:
            self.head_1 = RoutingHead(self.transformer_config, kwds_by_routing=[
                { 'n_inp':    -1, 'n_out': n_hid, 'd_inp': d_emb, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_hid, 'd_inp': d_hid, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_cls, 'd_inp': d_hid, 'd_out':     1, },
            ])
            self.m = nn.Softmax(dim=-1)


        

        #####
        d_emb = args.embed_size
        n_classes= args.h2_embed_size
        d_hid, n_cls = (d_emb, n_classes)
        n_hid=256
        if 'head_2' in args.heads:
            self.head_2 = RoutingHead(self.transformer_config, kwds_by_routing=[
                { 'n_inp':    -1, 'n_out': n_hid, 'd_inp': d_emb, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_hid, 'd_inp': d_hid, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_cls, 'd_inp': d_hid, 'd_out':     1, },
            ])
        # for p in self.head_.parameters():
        #     p.requires_grad = False
        # for p in self.head_.route[-1].parameters():
        #     p.requires_grad = True
        #####
        Test=True
        if Test:
            d_emb = args.embed_size
            n_classes= args.h3_embed_size
            d_hid, n_cls = (d_emb, n_classes)
            n_hid=256
        else:
            d_emb = args.embed_size
            n_classes= args.h2_embed_size
            d_hid, n_cls = (d_emb, n_classes)
            n_hid=512




        if 'head_3' in args.heads:
            self.head_3 = RoutingHead(self.transformer_config, kwds_by_routing=[
                { 'n_inp':    -1, 'n_out': n_hid, 'd_inp': d_emb, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_hid, 'd_inp': d_hid, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_cls, 'd_inp': d_hid, 'd_out':     1, },
            ])
     

    def get_hidden_states(self,t_idxs,t_masks):
        x=self.transformer(input_ids=t_idxs,attention_mask=t_masks).hidden_states
        return x
    def get_r(self,args,t_idxs,Len_Target,x1,abstract_test):
        flg_n=-1
        ents_flagged_plus_rel_idxs=t_idxs
        Tgt_Sentence_idxs=ents_flagged_plus_rel_idxs

        mask=torch.ones_like(ents_flagged_plus_rel_idxs).to(ents_flagged_plus_rel_idxs.device)
        ####
        for i in range(Len_Target.shape[0]):
            if Len_Target[i].item()<39:
                row=Len_Target[i].item()
                mask[i,row-3:]=0
        #####
        if args.data_type=='wikidata' :
            x2 = self.transformer(ents_flagged_plus_rel_idxs,attention_mask=mask)['hidden_states'][flg_n]
        else:
            x2 = self.transformer(ents_flagged_plus_rel_idxs,attention_mask=mask)['hidden_states'][flg_n]

        if args.data_type=='wikidata' :

            z1=x1#self.transformer_encoder(x1,mask=temp_mask.bool())

        else:
            #z1=self.transformer_encoder(x1,mask=sentence_flagged_masks_idxs)

            z1=x1#self.transformer_encoder(x1,mask=temp_mask.bool())
        ####
        tgtmask=Tgt_Sentence_idxs[:,:]!=0
        tgtmask=tgtmask.unsqueeze(-1).repeat(1*self.head,1,tgtmask.shape[1]).to(self.device)
        mask=nn.Transformer.generate_square_subsequent_mask(x2.shape[1])
        mask[mask == float("0")] = 1.
        mask[mask == float("-Inf")] = 0
        mask=mask.unsqueeze(0).repeat(tgtmask.shape[0],1,1).to(self.device)
        tgtmask=torch.mul(tgtmask, mask)
        if args.data_type=='wikidata':
            z2=x2#self.transformer_encoder(x2,mask=tgtmask,is_causal=True)

        else:
            z2=x2#self.transformer_encoder(x2,mask=tgtmask,is_causal=True)
        memory = z1
        tgt =z2[:,:-1] 
        tgt_mask=Tgt_Sentence_idxs[:,:-1]!=0
        tgt_mask=tgt_mask.unsqueeze(-1).repeat(1*self.head,1,tgt_mask.shape[1])
        mask=nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        mask[mask == float("0")] = 1.
        mask[mask == float("-Inf")] = 0
        mask=mask.unsqueeze(0).repeat(tgt_mask.shape[0],1,1).to(self.device)
        tgt_mask=torch.mul(tgt_mask, mask)
        out = self.transformer_decoder(tgt, memory,tgt_mask=tgt_mask,tgt_is_causal=True)

        if abstract_test:
            return out
        rel_n=3
        ids=Tgt_Sentence_idxs[0,Len_Target[i].item()-rel_n]
        ###
        # print('#######')
        # print('out[i,Len_Target[i].item()-rel_n,:]',ids)
        # token=self.tokenizer.convert_ids_to_tokens([ids])
        # print('token',token)
        # print('ln',len(Tgt_Sentence_idxs[0]))
        # print('Len_Target',Len_Target)

        # token=self.tokenizer.convert_ids_to_tokens(Tgt_Sentence_idxs[0])
        # print('token2',token)

        ###

        for i in range(Len_Target.shape[0]):
            if Len_Target[i].item()>39:
                if i==0:
                    r=torch.zeros(1,self.d_model).to(out.device)
                else:
                    temp=torch.zeros(1,self.d_model).to(r.device)
                    r=torch.cat((r,temp),0)
            elif i==0:
                r=out[i,Len_Target[i].item()-rel_n,:].unsqueeze(0).to(out.device)
            else:
                temp=out[i,Len_Target[i].item()-rel_n,:].unsqueeze(0).to(r.device)
                r=torch.cat((r,temp),0)
        return r,out

    def data(self,args,batch):
            if args.abstract=='mask':
                sentence_flagged_idxs=batch['sentence_masked_flagged_tokens']
                sentence_flagged_masks_idxs=batch['sentence_masked_flagged_tokens_masks']
                ents_flagged_plus_rel_idxs=batch['ents_flagged_plus_rel_tokens']
                Len_Target=batch['Len_Target']
            elif args.abstract=='abstract':
                sentence_flagged_idxs=batch['abstracted_ents_flagged_tokens']
                sentence_flagged_masks_idxs=batch['abstracted_ents_flagged_tokens_masks']
                ents_flagged_plus_rel_idxs=batch['abstracted_ents_flagged_plus_rel_tokens']
                Len_Target=batch['abstract_Len_Target']
            elif args.abstract=='flagged_ents':
                sentence_flagged_idxs=batch['ents_flagged_tokens'] 
                sentence_flagged_masks_idxs=batch['ents_flagged_tokens_masks'] 
                # sentence_flagged_idxs=batch['sentence_ents_flagged_tokens']
                # sentence_flagged_masks_idxs=batch['sentence_ents_flagged_tokens_masks']
                ents_flagged_plus_rel_idxs=batch['ents_flagged_plus_rel_tokens']
                Len_Target=batch['Len_Target']
            else:
                #sentence_flagged_idxs=batch['sentecne_entabs_flagged_tokens']
                #sentence_flagged_masks_idxs=batch['sentecne_entabs_flagged_tokens_masks']
                sentence_flagged_idxs=batch['EntsAbst_flagged_tokens'] 
                sentence_flagged_masks_idxs=batch['EntsAbst_flagged_tokens_masks'] 
                ents_flagged_plus_rel_idxs=batch['EntsAbst_flagged_plus_rel_tokens']
                Len_Target=batch['ent_abstract_Len_Target']
                n=Len_Target[0]

            return sentence_flagged_idxs,sentence_flagged_masks_idxs,ents_flagged_plus_rel_idxs,Len_Target

    def main_loss(self,v_anchor, v_positive, v_negative=None,loss=0,loss_f='empty',augument=False):
                if loss_f=='nce':
   
                    for i in range(len(v_anchor)):
                        
                        for j in range(len(v_positive)):
                            temp_p=v_anchor[i].unsqueeze(0)
                            temp_n=v_positive[j].unsqueeze(0)
                            loss_temp=loss_nce(temp_p,temp_n,linear=self.linear,device=v_positive[0].device)
                            loss=loss+loss_temp
                elif loss_f=='nce_':
                    a=torch.stack(v_anchor,0)
                    p=torch.stack(v_positive,0)
                    #n=torch.stack(v_negative,0)
                 

                    loss_temp=loss_nce(a,p,linear=self.linear,device=a.device)
                    loss=loss+loss_temp
          

     
                elif loss_f=='triplete':
                    if augument==True:
                        for i in range(len(v_anchor)):

                            for j in range(len(v_positive)):
                                for k in range(len(v_negative)):
                                    a=v_anchor[i].unsqueeze(0)
                                    p=v_positive[j].unsqueeze(0)
                                    n=v_negative[k].unsqueeze(0)
                                    loss_temp=loss_triplet(a, p, n,augument=augument)
                                    loss=loss+loss_temp
                    else:
                        a=torch.stack(v_anchor,0)
                        p=torch.stack(v_positive,0)
                        n=torch.stack(v_negative,0)
                        loss_temp=loss_triplet(a, p, n,augument=augument)
                        loss=loss+loss_temp



                elif loss_f=='triplete_':
                    a=torch.stack(v_anchor,0)
                    p=torch.stack(v_positive,0)
                    n=torch.stack(v_negative,0)
                 

                    loss_temp=loss_triplet(a, p, n,augument=augument)
                    loss=loss+loss_temp
                return loss



    def forward(self, batch,args,eval_epxeriment_data=None,eval=False,abstract_test=False,return_offset=False):
        flg_n=-1 
        if args.data_type!='semeval_2012':

            sentence_flagged_idxs,sentence_flagged_masks_idxs,ents_flagged_plus_rel_idxs,Len_Target=self.data(args,batch)

            if 'head_conditional' in args.heads:
                t_idxs=sentence_flagged_idxs
                t_masks=sentence_flagged_masks_idxs
                x=self.get_hidden_states(t_idxs,t_masks)
                if (args.data_type=='retacred'  or  args.data_type=='conll') and eval_epxeriment_data!=None:
                    L=sentence_flagged_idxs
                    self.get_head_tail_rep(L,batch,x,eval_epxeriment_data)

                  
            x1=x[flg_n]

            if 'head_1' in args.heads or 'head_2' in args.heads or 'head_3' in args.heads:
                if args.feed_y0==False:
                    e,h=None,None
                    h2,h3=None,None
                    if 'head_1' in args.heads:
                        temp=True if eval==True else False
                        e,h = self.forward_head(batch,args,head='head_1',eval=temp,abstract_test=False)
                    if 'head_2' in args.heads:
                        h2=self.forward_head(batch,args,head='head_2',eval=eval,abstract_test=False)
                    if 'head_3' in args.heads:

                        h3=self.forward_head(batch,args,head='head_3',eval_epxeriment_data=eval_epxeriment_data,eval=eval,abstract_test=False)
                    temp=['head_3','head_2','head_1']
                    hs=[t for th,t in zip(temp,(h3,h2,e)) if th in args.heads]
                    e=torch.cat(hs,-1)
                else:
                    y0=batch['y0']
                    e=self.y_emb(y0)

                    h=None
            else:
                h=None
            if 'head_conditional' in args.heads:
                r,out=self.get_r(args,ents_flagged_plus_rel_idxs,Len_Target,x1,abstract_test)
                x=r
                if return_offset:
                    return r


            if 'head_1' in args.heads or 'head_2' in args.heads or 'head_3' in args.heads:
                if 'head_conditional' in args.heads:
                    x=torch.cat((x,e),-1)
                else:
                    out=None
                    x=e
            else:
                x=x
            relations = self.classifier(x)
            return h,out,relations
        else:
            #dict_keys(['s', 's_masks', 's_plus_rel', 'a', 'a_masks', 'a_plus_rel', 'n', 'n_masks', 'n_plus_rel', 's_len', 'a_len', 'n_len'])
            s=batch['s']
            s_masks=batch['s_masks']
            s_len=batch['s_len']
 
            s_plus_rel=batch['s_plus_rel']

            t_idxs=s
            t_masks=s_masks
            s_emb=self.get_hidden_states(t_idxs,t_masks)[-1]
            memory=s_emb
            Len_Target=s_len
            ents_flagged_plus_rel_idxs=s_plus_rel
            s_r_emb,out1=self.get_r(args,ents_flagged_plus_rel_idxs,Len_Target,memory,abstract_test)
            ##

            if return_offset:
                    return s_r_emb
            s_r=batch['s_r']
            s_r_=batch['s_r_']

            a=batch['a']
            a_masks=batch['a_masks']
            a_len=batch['a_len']
            a_r=batch['a_r']
            a_r_=batch['a_r_']
            a_plus_rel=batch['a_plus_rel']

            t_idxs=a
            t_masks=a_masks
            a_emb=self.get_hidden_states(t_idxs,t_masks)[-1]
            memory=a_emb
            Len_Target=a_len
            ents_flagged_plus_rel_idxs=a_plus_rel
            a_r_emb,out2=self.get_r(args,ents_flagged_plus_rel_idxs,Len_Target,memory,abstract_test)
            ##

            n=batch['n']
            n_masks=batch['n_masks']
            n_len=batch['n_len']
            n_r=batch['n_r']
            n_r_=batch['n_r_']
            n_plus_rel=batch['n_plus_rel']



            t_idxs=n
            t_masks=n_masks
            n_emb=self.get_hidden_states(t_idxs,t_masks)[-1]

            memory=n_emb
            Len_Target=n_len
            ents_flagged_plus_rel_idxs=n_plus_rel
            n_r_emb,_=self.get_r(args,ents_flagged_plus_rel_idxs,Len_Target,memory,abstract_test)
            ##############################
            equ=batch['equ']
            equ_masks=batch['equ_masks']
            equ_len=batch['equ_len']
            equ_r=batch['equ_r']
            equ_r_=batch['equ_r_']
            equ_plus_rel=batch['equ_plus_rel']



            t_idxs=equ
            t_masks=equ_masks
            equ_emb=self.get_hidden_states(t_idxs,t_masks)[-1]

            memory=equ_emb
            Len_Target=equ_len
            ents_flagged_plus_rel_idxs=equ_plus_rel
            equ_r_emb,_=self.get_r(args,ents_flagged_plus_rel_idxs,Len_Target,memory,abstract_test)



            p_d_=torch.abs(s_r_emb-a_r_emb)
            n_d_=torch.abs(s_r_emb-n_r_emb)

            p_d=torch.cat((s_r_emb,a_r_emb,p_d_),-1)
            n_d=torch.cat((s_r_emb,n_r_emb,n_d_),-1)
            ###

            #######
            v_anchor=s_r_emb
            v_positive=a_r_emb
            v_negative=n_r_emb


            r_unique_positive=a_r.flatten().tolist()
            r_unique_positive_=a_r_.flatten().tolist()

            r_unique_positive_dic={}
            r_unique_positive_dic_={}
            for r,r_ in zip(r_unique_positive,r_unique_positive_):
                if r not in r_unique_positive_dic.keys():
                    r_unique_positive_dic[r]={'s':[],'a':[],'n':[]}
                    r_unique_positive_dic_[r_]={'s':[],'a':[],'n':[]}



            anchor={}
            positive={}
            negative={}

            anchor_={}
            positive_={}
            negative_={}
            R={}
            R_={}
            for ri,(r,r_) in enumerate(zip(a_r.tolist(),a_r_.tolist())):
                R[r]=1
                R_[r_]=1

                temp_s=s_r_emb[ri]
                temp_a=a_r_emb[ri]
                temp_n=n_r_emb[ri]

                update2_rc(r,r_,anchor_,temp_s)
                update2_rc(r,r_,positive_,temp_a)
                update2_rc(r,r_,negative_,temp_n)
                ###
                update2_rc(r_,r,anchor,temp_s)
                update2_rc(r_,r,positive,temp_a)
                update2_rc(r_,r,negative,temp_n)
                
                
            loss=0
    
            loss=self.main_loss(a_r_emb, equ_r_emb,loss=loss,loss_f='nce')
    

            if args.epoch>=1:
                batch_n_a=[]
                batch_n_p=[]
                batch_n_n=[]
                i=None
                Rs=[]

                A=anchor_
                P=positive_
                N=negative_
                batch_n_a,batch_n_p,batch_n_n,Rs=batch_n_rc(A,P,N,R_,batch_n_a,batch_n_p,batch_n_n,i,Rs)




                loss=self.main_loss(batch_n_a, batch_n_p, v_negative=batch_n_n,loss=loss,loss_f='triplete_',augument=True)

                A=anchor
                P=positive
                N=negative
                Rs=[]
                print(A.keys())
                batch_n_a_,batch_n_p_,batch_n_n_,Rs=batch_n_rc(A,P,N,R,batch_n_a,batch_n_p,batch_n_n,i,Rs)
                loss=self.main_loss(batch_n_a_, batch_n_p_, v_negative=batch_n_n_,loss=loss,loss_f='triplete_',augument=True)
   
            
                for r1_ in R_.keys():
                    a_=anchor_[r1_]
                    p_=positive_[r1_]
                    n_=negative_[r1_]
                    anchor_temp=[a_[tr] for tr in a_.keys()]
                    positive_temp=[p_[tr] for tr in p_.keys()]
                    negative_temp=[n_[tr] for tr in n_.keys()]

                    anchor_temp = list(itertools.chain.from_iterable(anchor_temp))
                    positive_temp = list(itertools.chain.from_iterable(positive_temp))
                    negative_temp = list(itertools.chain.from_iterable(negative_temp))
                    Positive=[]
                    for t in positive_temp:
                        Positive.append(t)
                    Negative=[]
                    for r2_ in R_.keys():
                        if r1_==r2_:
                            continue

                        a_=anchor_[r2_]
                        p_=positive_[r2_]
                        n_=negative_[r2_]
                        anchor_temp=[a_[tr] for tr in a_.keys()]
                        positive_temp=[p_[tr] for tr in p_.keys()]
                        negative_temp=[n_[tr] for tr in n_.keys()]

                        anchor_temp = list(itertools.chain.from_iterable(anchor_temp))
                        positive_temp = list(itertools.chain.from_iterable(positive_temp))
                        negative_temp = list(itertools.chain.from_iterable(negative_temp))
                        for t in positive_temp:
                                 Negative.append(t)
                        for t in anchor_temp:
                                 Negative.append(t)

                    p=torch.stack(Positive,0)
                    n=torch.stack(Negative,0)
                    loss=self.main_loss(p, n,loss=loss,loss_f='nce')
                #####
                for r1_ in R.keys():
                    a_=anchor[r1_]
                    p_=positive[r1_]
                    n_=negative[r1_]
                    anchor_temp=[a_[tr] for tr in a_.keys()]
                    positive_temp=[p_[tr] for tr in p_.keys()]
                    negative_temp=[n_[tr] for tr in n_.keys()]

                    anchor_temp = list(itertools.chain.from_iterable(anchor_temp))
                    positive_temp = list(itertools.chain.from_iterable(positive_temp))
                    negative_temp = list(itertools.chain.from_iterable(negative_temp))
                    Positive=[]
                    for t in positive_temp:
                        Positive.append(t)
                    Negative=[]
                    for r2_ in R.keys():
                        if r1_==r2_:
                            continue
                        a_=anchor[r2_]
                        p_=positive[r2_]
                        n_=negative[r2_]
                        anchor_temp=[a_[tr] for tr in a_.keys()]
                        positive_temp=[p_[tr] for tr in p_.keys()]
                        negative_temp=[n_[tr] for tr in n_.keys()]

                        anchor_temp = list(itertools.chain.from_iterable(anchor_temp))
                        positive_temp = list(itertools.chain.from_iterable(positive_temp))
                        negative_temp = list(itertools.chain.from_iterable(negative_temp))
                        for t in positive_temp:
                                 Negative.append(t)
                        for t in anchor_temp:
                                 Negative.append(t)

                    p=torch.stack(Positive,0)
                    n=torch.stack(Negative,0)
                    loss=self.main_loss(p, n,loss=loss,loss_f='nce')
        


                print('loss',loss)
            if eval:
                return None,out,postive_classified


            else:
                return out1,out2 ,loss#postive_classified,negative_classified,loss
                #return out, s_r_classified,s_r_classified_,a_r_classified,a_r_classified_,n_r_classified,n_r_classified_
