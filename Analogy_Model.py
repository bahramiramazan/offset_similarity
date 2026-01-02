import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import torch.nn.functional as F
import torch
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
import random
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import ujson as json
from analogy_util import _get_pretrained_transformer3, update2,batch_n,get_permuation_positive
import torch.nn.functional as F
import itertools
####
#analogy_util

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

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
    """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
    # inherit from Module
    super().__init__()     

    # initialize dropout                  
    self.dropout = nn.Dropout(p=dropout)      

    # create tensor of 0s
    pe = torch.zeros(max_length, d_model)    

    # create position column   
    k = torch.arange(0, max_length).unsqueeze(1)  

    # calc divisor for positional encoding 
    div_term = torch.exp(                                 
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)    

    # calc cosine on odd indices   
    pe[:, 1::2] = torch.cos(k * div_term)  

    # add dimension     
    pe = pe.unsqueeze(0)          

    # buffers are saved in state_dict but not trained by the optimizer                        
    self.register_buffer("pe", pe)                        

  def forward(self, x):
    """
    Args:
      x:        embeddings (batch_size, seq_length, d_model)
    
    Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
    """
    # add positional encoding to the embeddings
    x = x + self.pe[:, : x.size(1)].requires_grad_(False) 

    # perform dropout
    return self.dropout(x)


def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)



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










class Analogy_RE_Model(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder.
    """
    def __init__(
        self,
        args

    ):
        super().__init__()

        vocab_size, d_model = args.vocab_size,args.embed_size#embeddings.size()

        self.embed_size=args.embed_size
        ####
        args.feed_y0=False
        #bert-base-uncased roberta-large
        model_name=args.model_name#'roberta-large'#'bert-base-uncased'# 'roberta-large'
        self.modality=args.model_name#'roberta-large' if model_name=='roberta-large'  else 'bert-base-uncased'
 
        
        self.pretrained_name=model_name
        #(modality,tokenizer_special_dic='default')
        tokenizer_special_dic='re'

        data_selected=args.data_type
        #self.transformer_config, self.transformer = _get_pretrained_transformer2(self.modality) if args.data_type=='wordanalogy' else  _get_pretrained_transformer3(self.modality) 
        self.transformer_config, self.transformer,self.tokenizer =  _get_pretrained_transformer3(data_selected,self.modality,tokenizer_special_dic=tokenizer_special_dic) 


        if model_name!='flaxopt':
            self.transformer.resize_token_embeddings(50633)
        self.d_model = d_model
        dim=0
        heads={'head_1':512,'head_2':args.h2_embed_size,'head_3':args.h3_embed_size,'head_conditional':args.embed_size}
        for h in heads.keys():
            if h in args.heads:
     
                dim=dim+heads[h]

        self.args=args
        input_sizes=[dim,dim,dim,args.n_class]
        input_sizes=[dim,dim,dim,args.n_class]
        print('input_sizes',input_sizes)
        self.classifier=MLP(input_sizes)
        for name, param in self.classifier.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    if args.data_type!='wikidata':
                        nn.init.xavier_normal_(param)
                        #nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))

        self.linear = torch.nn.Linear(512 * 3, 1)  # three way feature
        self.linear.weight.data.normal_(std=0.02)
        
        # self.head=8
        # ###***head_conditional***
        # if 'head_conditional' in args.heads:

        #     encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=self.head,batch_first=True)
        #     self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        #     for name, param in self.transformer_encoder.named_parameters():
        #         if 'weight' in name and param.data.dim() == 2:
        #             nn.init.xavier_normal_(param)
        #             #nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

        #     ##########
        #     decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_size, nhead=self.head,batch_first=True)
        #     self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        #     ##
        #     for name, param in self.transformer_decoder.named_parameters():
        #         if 'weight' in name and param.data.dim() == 2:
        #             nn.init.xavier_normal_(param)
        #             #nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

        #     ln_tokenizer = len(self.tokenizer) if model_name=='roberta-large' else len(self.tokenizer)
        #     self.generator=Generator(self.embed_size,ln_tokenizer)
        #     self.PositionalEncoding=PositionalEncoding(self.embed_size)
            
       
        #     for name, param in self.generator.named_parameters():
        #         if 'weight' in name and param.data.dim() == 2:
        #             if args.data_type!='wikidata':
        #                 nn.init.xavier_normal_(param)
                        
        #                 #nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
        #             else:
        #                 nn.init.xavier_normal_(param)
        #                 #nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
        #             #nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
        ############

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
        #self.tokenizer = get_tokenizer(model_name)#AutoTokenizer.from_pretrained("./tokenizer/") if model_name=='roberta-large' else BertTokenizer.from_pretrained("./tokenizer/")

        if 'head_1' in args.heads:
            self.nn_linear2= nn.Linear(2,512,bias=False)

            for name, param in self.nn_linear2.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    nn.init.xavier_normal_(param)
                    #nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
                    #nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    #nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
                    #nn.init.orthogonal_(param)
                    #nn.init.sparse_(param, sparsity=0.1)



        print('self.tokenizer',len(self.tokenizer))
        ##

        #####
        #if args.data_type=='wordanalogy':
        d_emb = args.embed_size
        n_classes= args.h3_embed_size 
        d_hid, n_cls = (d_emb, n_classes)
        n_hid=256 if args.data_type=='wordanalogy' else 256
        if args.data_type=='wordanalogy' and args.model_name=='bert-base-uncased':
            n_hid=128





        if 'head_3' in args.heads:
            self.head_3 = RoutingHead(self.transformer_config, kwds_by_routing=[
                { 'n_inp':    -1, 'n_out': n_hid, 'd_inp': d_emb, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_hid, 'd_inp': d_hid, 'd_out': d_hid, },
                { 'n_inp': n_hid, 'n_out': n_cls, 'd_inp': d_hid, 'd_out':     1, },
            ])
        

    def set_only_classifier_train(self,flag):
            ##
            for p in self.head_3.parameters():
                p.requires_grad = not(flag)
            for p in self.transformer.parameters():
                p.requires_grad = not(flag)

            for p in self.classifier.parameters():
                p.requires_grad = True


    def set_only_head_train(self):
        ##
        for p in self.head_3.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = True



    def get_head_tail_rep(self,L,batch,x,eval_epxeriment_data=False,wordanalogy=False):

        e1_se=self.tokenizer.convert_tokens_to_ids(['[e11]','[e12]'])
        e2_se=self.tokenizer.convert_tokens_to_ids(['[e21]','[e22]'])
        y=batch['y']
        # print('e1_se',e1_se)
        # print('e2_se',e2_se)
        #print('x',x.shape)
        sentence_flagged_idxs=L

        Batch_h_t=[]
 

        population=[b for b in range(L.shape[0])]
        selected=random.sample(population, int(len(population)/2))
        for b in range(L.shape[0]):
            # if b not in selected:
            #     continue
    

            if e1_se[0] in L.tolist()[b]:

                e1_start=L.tolist()[b].index(e1_se[0])
            else: 
                continue 
            if e1_se[1] in L.tolist()[b]:

                e1_end=L.tolist()[b].index(e1_se[1])
            else:
                e1_end=-1
            ###

            if e2_se[0] in L.tolist()[b]:
                e2_start=L.tolist()[b].index(e2_se[0])

            else:
                e2_start=-1
         
 

            if e2_se[1] in L.tolist()[b]:
                e2_end=L.tolist()[b].index(e2_se[1])
            else:
                e2_end=-1


            y=batch['y'][b].cpu().detach().item()
            item_id=batch['ids'][b]
     
            temp_eval_d={}


            if wordanalogy==False:
                for h in range(len(x)):
                    temp=x[h][b,e1_start:e1_end+1,:] 
                    head=temp.cpu().detach().numpy().tolist()
                    #print(temp.shape)
                    temp=x[h][b,e2_start:e2_end+1,:] 
                    tail=temp.cpu().detach().numpy().tolist()
                    temp_eval_d[h]={'head':head,'tail':tail}
                if y in eval_epxeriment_data.keys():
                    eval_epxeriment_data[y][item_id]=temp_eval_d
                else:
                    eval_epxeriment_data[y]={}
                    eval_epxeriment_data[y][item_id]=temp_eval_d
            else:
                

                temp=[t[b,e1_start:e1_end+1,:]  for t in x]
                temp=torch.stack(temp,0)
                head=temp.unsqueeze(0)#.cpu().detach()
     


                temp=[t[b,e2_start:e2_end+1,:]  for t in x]
                temp=torch.stack(temp,0)
                tail=temp.unsqueeze(0)#.cpu().detach()
                #print(head.shape,tail.shape)
      
                temp_eval_d={'head':head,'tail':tail}
                
                Batch_h_t.append(temp_eval_d)

        if wordanalogy==True:
            H=[]
            T=[]
            for b in range(len(Batch_h_t)):
                head=Batch_h_t[b]['head']
                tail=Batch_h_t[b]['tail']
         

                H.append(torch.sum(head,-2))
                T.append(torch.sum(tail,-2))
 
            H=torch.concat(H,0)
            T=torch.concat(T,0)
            Batch_h_t={'head':H,'tail':T}
      
        return Batch_h_t


    def encode(self,word,args):
        
        #word = "[CLS] " + word + " [SEP]"
        word='[e11]'+word+'[e12]'+'[e21]'+'MASK'+'[e22]'
        tokenized_text=self.tokenizer.tokenize(word)
        w_idx = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        w_mask = [1] * len(w_idx)
        w_idx=torch.tensor(w_idx).long().unsqueeze(0).to(self.device)

        w_mask=torch.ones_like(w_idx)
        # print(w_idx.shape)
        # print(w_mask.shape)
        temp='not_route' if args.wordanalogy_model=='baseline' or args.wordanalogy_model=='baseline_train'  else 'route'
   
        if args.wordanalogy_model in 'classification_train_head':
                abcd=w_idx
                abcd_mask=w_mask
                abcd=self.transformer(input_ids=abcd,attention_mask=abcd_mask).hidden_states
                abcd_mask=abcd_mask.unsqueeze(-1).repeat(1,1,abcd[0].shape[-1])
                abcd=[abcd[i]*abcd_mask for i in range(len(abcd))]
                abcd = torch.stack(abcd, dim=-2)
                abcd = self.head_3(abcd)
                #print('')

                return abcd
        elif temp=='route':
            w=self.transformer(input_ids=w_idx,attention_mask=w_mask).hidden_states
            w=[v[:,1:-4] for v in w]
            # print('w',w[0].shape)
            # exit()
            w = torch.stack(w, dim=-2)

            
            w_h = self.head_3(w)
            return w_h#.unsqueeze(0)
        else :
        
            w=self.transformer(input_ids=w_idx,attention_mask=w_mask).hidden_states
            sum_embeddings_w = torch.sum(w[args.layer_number], 1)
            w_mask_sum = w_mask[0].sum(0)
            #w_mask_sum = torch.clamp(w_mask_sum, min=1e-9)
            w_h=sum_embeddings_w[0] / w_mask_sum
            return w_h.unsqueeze(0).unsqueeze(0)
     
    def forward(self, batch,args,eval_epxeriment_data=None,eval=False,abstract_test=False):
        # for b in batch.keys():
        #     print('key',b)
        #     print('batch',batch[b])
        #     print('***')
        # exit()

        if args.data_type=='wordanalogy':


            w1=batch['a'] 
            w1_mask=batch['a_masks'] 
            w2=batch['b']
            w2_mask=batch['b_masks']
            w3=batch['c']
            w3_mask=batch['c_masks']
            w4=batch['d']
            w4_mask=batch['d_masks']
            y0=batch['y0']
            abcd=batch['abcd'] 
            abcd_mask=batch['abcd_masks'] 
            #
            ab=batch['ab'] 
            ab_mask=batch['ab_masks'] 
            cd=batch['cd']
            cd_mask=batch['cd_masks']
            y0=batch['y0']
            similarity=batch['similarity']
            r=batch['r']
            r1=batch['r1']
            r_label=batch['r_label']
            r_label_mask=batch['r_label_masks']

      
            ##
            abcd_l=abcd.tolist()
            ab_l=ab.tolist()

            cd_l=cd.tolist()


            #####
            #'abcd':['^']+w1+['*']+w2+['#']+w3+['@']+w4+['&'],\
            #if args.fin_tune==False:
                #'abcd':['^']+w1+['*']+w2+['#']+w3+['@']+w4+['&'],\
            if 'classification'  in args.wordanalogy_model:
                w1_se=self.tokenizer.convert_tokens_to_ids(['^','*'])
                w2_se=self.tokenizer.convert_tokens_to_ids(['*','#'])
                w3_se=self.tokenizer.convert_tokens_to_ids(['#','@'])
                w4_se=self.tokenizer.convert_tokens_to_ids(['@','&'])
                t=['roberta-large','roberta-base','opt','gp2']
                if self.pretrained_name in t :

                    w1_se=self.tokenizer.convert_tokens_to_ids(['Ġ^','Ġ*'])
                    w2_se=self.tokenizer.convert_tokens_to_ids(['Ġ*','Ġ#'])
                    w3_se=self.tokenizer.convert_tokens_to_ids(['Ġ#','Ġ@'])
                    w4_se=self.tokenizer.convert_tokens_to_ids(['Ġ@','Ġ&'])
            else:

                w1_se=self.tokenizer.convert_tokens_to_ids(['[e11]','[e12]'])
                w2_se=self.tokenizer.convert_tokens_to_ids(['[e21]','[e22]'])
                w3_se=self.tokenizer.convert_tokens_to_ids(['[e11]','[e12]'])
                w4_se=self.tokenizer.convert_tokens_to_ids(['[e21]','[e22]'])
                t=['roberta-large','roberta-base','opt','gpt2']
                if self.pretrained_name in t :
                    #Ġ
                    w1_se=self.tokenizer.convert_tokens_to_ids(['[e11]','[e12]'])
                    w2_se=self.tokenizer.convert_tokens_to_ids(['[e21]','[e22]'])
                    w3_se=self.tokenizer.convert_tokens_to_ids(['[e11]','[e12]'])
                    w4_se=self.tokenizer.convert_tokens_to_ids(['[e21]','[e22]'])
                    # print('w1_se',w1_se)
                    # print('ab_l',ab_l)
                    # exit()

            #####
            similarity_in_hidden_layers={}

           #print('self.tokenizer',len(self.tokenizer))
            #exit()



            temp='not_route' if args.wordanalogy_model=='baseline' or args.wordanalogy_model=='baseline_train'  else 'route'
            temp_route='not_route' if ('baseline' in args.wordanalogy_model or 'baseline_train' in args.wordanalogy_model)  else 'route'
            #print('temp_route',temp_route)
            if args.wordanalogy_model in 'classification_train_head':
                abcd=self.transformer(input_ids=abcd,attention_mask=abcd_mask).hidden_states
                abcd_mask=abcd_mask.unsqueeze(-1).repeat(1,1,abcd[0].shape[-1])
                #abcd=[abcd[i]*abcd_mask for i in range(len(abcd))]
                abcd = torch.stack(abcd, dim=-2)
                abcd = self.head_3(abcd)
      


                return abcd


            else:

                if 'classification' not in args.wordanalogy_model:
                    if 't5' not in self.pretrained_name:
                        ab=self.transformer(input_ids=ab,attention_mask=ab_mask).hidden_states
                        cd=self.transformer(input_ids=cd,attention_mask=cd_mask).hidden_states
                        # print('ab',ab)
                        # exit()
                        if self.pretrained_name=='opt' or self.pretrained_name=='flaxopt':
                            ab=ab[:-1]
                            cd=cd[:-1]
                    else:
                        ab_decoder_input_ids = self.transformer._shift_right(ab)
                        cd_decoder_input_ids = self.transformer._shift_right(cd)

                        # forward pass
                        ab = self.transformer(input_ids=ab, decoder_input_ids=ab_decoder_input_ids)
                        cd = self.transformer(input_ids=cd, decoder_input_ids=cd_decoder_input_ids)
         
                        ab = ab.last_hidden_state
                        cd = cd.last_hidden_state
                        ab=[ab,]
                        cd=[cd,]

                        w1_se=[w+1 for w in w1_se ] 
                        w2_se=[w+1 for w in w2_se ] 
                        w3_se=[w+1 for w in w3_se ] 
                        w4_se=[w+1 for w in w4_se ] 





                else:
                    abcd=self.transformer(input_ids=abcd,attention_mask=abcd_mask).hidden_states
                    if self.pretrained_name=='opt':
                        abcd=abcd[:-1]

                w_1,w_2,w_3,w_4=[],[],[],[]
                max_length={'w1':0,'w2':0,'w3':0,'w4':0}
                for i in range(len(abcd_l)):
                    # print('ab_l',ab_l[i])
                    # ###
                    # print('ab_l[i]',ab_l[i])
                    # print('ab_l[i]',self.tokenizer.convert_ids_to_tokens(ab_l[i]))

                    abcd_source=abcd_l if 'classification'  in args.wordanalogy_model else ab_l
                    #print('abcd_source',abcd_source)
                    
                    w1_start=abcd_source[i].index(w1_se[0]) # if w1_se[0] in abcd_source[i] else 0
                    w1_end=abcd_source[i].index(w1_se[1]) if w1_se[1] in abcd_source[i] else len(abcd_source[i])-1

                    #
                    w2_start=abcd_source[i].index(w2_se[0]) if w2_se[0] in abcd_source[i] else len(abcd_source[i])-1
                    w2_end=abcd_source[i].index(w2_se[1])  if w2_se[1] in abcd_source[i] else len(abcd_source[i])-1

                    #############################
                    abcd_source= abcd_l  if 'classification'  in args.wordanalogy_model else cd_l

                    w3_start=abcd_source[i].index(w3_se[0]) if  w3_se[0] in abcd_source[i] else len(abcd_source[i])-1
                    w3_end=abcd_source[i].index(w3_se[1]) if w3_se[1] in abcd_source[i] else len(abcd_source[i])-1
                    #

                    w4_start=abcd_source[i].index(w4_se[0])if w4_se[0] in abcd_source[i] else len(abcd_source[i])-1
                    w4_end=abcd_source[i].index(w4_se[1]) if w4_se[1] in abcd_source[i] else len(abcd_source[i])
        



                    #print('if w4_se[1] in abcd_source[i] else len(abcd_source[i])', w4_se[1] in abcd_source[i] )

                    ##########
                    # select batch
                    abcd_source=abcd  if 'classification'  in args.wordanalogy_model else ab
                    out=[o[i] for o in abcd_source]

                    #print('out',out)
          
                    out=[o[w1_start:w1_end+1,:] for o in out]
                    shappes=[o.shape[0] for o in out]
                    t=shappes[0]
                    max_length['w1']= t if t>max_length['w1'] else max_length['w1']
                    w_1.append(out)
                    ##############
                    ##########
                    # select batch
                    abcd_source=abcd if 'classification'  in args.wordanalogy_model else ab
                    out=[o[i] for o in abcd_source]
          
                    out=[o[w2_start:w2_end+1,:] for o in out]
                    shappes=[o.shape[0] for o in out]
                    t=shappes[0]
                    max_length['w2']= t if t>max_length['w2'] else max_length['w2']
                    w_2.append(out)


                    ###############################################
                    # select batch
                    abcd_source= abcd  if 'classification'  in args.wordanalogy_model else cd
                    out=[o[i] for o in abcd_source]
          
                    out=[o[w3_start:w3_end+1,:] for o in out]
                    shappes=[o.shape[0] for o in out]
                    t=shappes[0]
                    max_length['w3']= t if t>max_length['w3'] else max_length['w3']
                    w_3.append(out)


                    ##########
                    # select batch
                    abcd_source=abcd  if 'classification'  in args.wordanalogy_model else cd
                    out=[o[i] for o in abcd_source]
          
                    out=[o[w4_start:w4_end+1,:] for o in out]
                    shappes=[o.shape[0] for o in out]
                    t=shappes[0]
                    max_length['w4']= t if t>max_length['w4'] else max_length['w4']
                    w_4.append(out)
                    
             

          

                w_1_,w_2_,w_3_,w_4_=[],[],[],[]
                for w in w_1:

                    w_t = torch.stack(w, dim=-2)
                    #print()
                    temp=torch.zeros(max_length['w1']-w_t.shape[0],w_t.shape[1],w_t.shape[2]).to(self.device)
                    w_t=torch.cat((w_t,temp),0)
                    w_1_.append(w_t)
                    #print('w1',w_t.shape)



                for w in w_2:

                    w_t = torch.stack(w, dim=-2)
                    temp=torch.zeros(max_length['w2']-w_t.shape[0],w_t.shape[1],w_t.shape[2]).to(self.device)
                    w_t=torch.cat((w_t,temp),0)
                    w_2_.append(w_t)
                    #print('w2',w_t.shape)

                for w in w_3:

                    w_t = torch.stack(w, dim=-2)
                    temp=torch.zeros(max_length['w3']-w_t.shape[0],w_t.shape[1],w_t.shape[2]).to(self.device)
                    w_t=torch.cat((w_t,temp),0)
                    w_3_.append(w_t)
                    #print('w3',w_t.shape)


                for w in w_4:

                    w_t = torch.stack(w, dim=-2)
                    temp=torch.zeros(max_length['w4']-w_t.shape[0],w_t.shape[1],w_t.shape[2]).to(self.device)
                    w_t=torch.cat((w_t,temp),0)
                    w_4_.append(w_t)
                    #print('w4',w_t.shape)
                w1 = torch.stack(w_1_, dim=0)
                w2 = torch.stack(w_2_, dim=0)
                w3 = torch.stack(w_3_, dim=0)
                w4 = torch.stack(w_4_, dim=0)


   



                #abcd=[self.PositionalEncoding(t) for t in abcd]
                model='route' if temp_route=='route' else 'baseline'


                w1=[w1[:,:,i,:] for i in range(w1.shape[-2])]
                w2=[w2[:,:,i,:] for i in range(w2.shape[-2])]
                w3=[w3[:,:,i,:] for i in range(w3.shape[-2])]
                w4=[w4[:,:,i,:] for i in range(w4.shape[-2])]

                #print('len',len(w1))
                w1_=[]
                w2_=[]
                w3_=[]
                w4_=[]

                cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

                for h_layer in range(len(w1)):
 
                    head_sim=cos(torch.sum(w1[h_layer], 1), torch.sum(w3[h_layer], 1)) 
                    tail_sim= cos(torch.sum(w2[h_layer], 1), torch.sum(w4[h_layer], 1)) 

                    

                    

                    head_dist = torch.nn.functional.pairwise_distance(torch.sum(w1[h_layer], 1), torch.sum(w3[h_layer], 1))#(w1_h - w3_h).pow(2).sum(1).sqrt()
                    tail_dist = torch.nn.functional.pairwise_distance(torch.sum(w4[h_layer], 1), torch.sum(w4[h_layer], 1))#(w2_h - w4_h).pow(2).sum(1).sqrt()
                    item={'head':(head_sim,head_dist),'tail':(tail_sim,tail_dist)}
     
                    similarity_in_hidden_layers[h_layer]=item




         
                if temp_route=='route' :


                    if 'sentence' in args.wordanalogy_model :

                        ########################


                        w1 = torch.stack(w1, dim=0)
                        w1=torch.swapaxes(w1, 0, 1)
                        w1=torch.swapaxes(w1, 1, 2)

                        #
                        w2 = torch.stack(w2, dim=0)
                        w2=torch.swapaxes(w2, 0, 1)
                        w2=torch.swapaxes(w2, 1, 2)
                        #
                        w3 = torch.stack(w3, dim=0)
                        w3=torch.swapaxes(w3, 0, 1)
                        w3=torch.swapaxes(w3, 1, 2)
                        #                  
                        w4 = torch.stack(w4, dim=0)
                        w4=torch.swapaxes(w4, 0, 1)
                        w4=torch.swapaxes(w4, 1, 2)
                        ##
                        #                                                 ##
                        # r_label = torch.stack(r_label, dim=0)
                        # r_label=torch.swapaxes(r_label, 0, 1)
                        # r_label=torch.swapaxes(r_label, 1, 2)
                        # #####
                        # ##
                        w1_h = self.head_3(w1)  # [chunks in batch, n pos, d_depth, d_emb]
                        w2_h = self.head_3(w2)
                        w3_h = self.head_3(w3)
                        #print(torch.max(w4),w4.shape)
                        w4_h = self.head_3(w4)


                        x1=w1_h-w2_h
                        x2=w3_h-w4_h
             
  
                else:
                    w1=w1[-1]
                    w2=w2[-1]
                    w3=w3[-1]
                    w4=w4[-1]

                    sum_embeddings_w1 = torch.sum(w1, 1)
                    sum_embeddings_w2 = torch.sum(w2, 1)
                    sum_embeddings_w3 = torch.sum(w3, 1)
                    sum_embeddings_w4 = torch.sum(w4, 1)

                    for i in range(w1.shape[0]):
                        w1_mask_sum = w1_mask[i].sum(0)
                        w2_mask_sum = w2_mask[i].sum(0)
                        w3_mask_sum = w3_mask[i].sum(0)
                        w4_mask_sum = w4_mask[i].sum(0)
                        w1_h_temp=sum_embeddings_w1[i] / w1_mask_sum
                        w2_h_temp=sum_embeddings_w2[i] / w2_mask_sum
                        w3_h_temp=sum_embeddings_w3[i] / w3_mask_sum
                        w4_h_temp=sum_embeddings_w4 [i]/ w4_mask_sum
                        if i==0:
                            w1_h=w1_h_temp.unsqueeze(0)
                            w2_h=w2_h_temp.unsqueeze(0)
                            w3_h=w3_h_temp.unsqueeze(0)
                            w4_h=w4_h_temp.unsqueeze(0)
                        else:
                            w1_h=torch.cat((w1_h,w1_h_temp.unsqueeze(0)),0)
                            w2_h=torch.cat((w2_h,w2_h_temp.unsqueeze(0)),0)
                            w3_h=torch.cat((w3_h,w3_h_temp.unsqueeze(0)),0)
                            w4_h=torch.cat((w4_h,w4_h_temp.unsqueeze(0)),0)
                    x1=w1_h-w2_h
                    x2=w3_h-w4_h
                    x1_n=w2_h-w1_h

                ###########################
                #relation_emb=self.relation_encoder(relation_emb)
                #relation_emb=r_label
                #########################################################

                if 'sentence' in args.wordanalogy_model:
                    if eval==False:
      
                        CosLoss = nn.CosineEmbeddingLoss()
                             

                        w1_w3_dist = torch.nn.functional.pairwise_distance(w1_h, w3_h)#(w1_h - w3_h).pow(2).sum(1).sqrt()
                        w1_w4_dist = torch.nn.functional.pairwise_distance(w1_h, w4_h)#(w2_h - w4_h).pow(2).sum(1).sqrt()
                        w1_w2_dist = torch.nn.functional.pairwise_distance(w1_h, w2_h)#(w2_h - w4_h).pow(2).sum(1).sqrt()
                        a,b,c,d=w1_h,w2_h,w3_h,w4_h

                        x1_vertical=a-d
                        x2_vertical=b-c

                        r_parent=batch['r']
                        r_sub=batch['r']
                        ###
                        loss=0
                        if args.fin_tune==False:
                            P_rp={}
                            P_rs={}
                     
                            R_p={}
                            R_s={}
                            for ri,(r_p,r_s) in enumerate(zip(r_parent.tolist(),r_sub.tolist())):
                                R_p[r_p]=1
                                R_s[r_s]=1

                                x=x1[ri]#.unsqueeze(0)

                  
                                update2(r_s,r_p,P_rp,x)
                                update2(r_p,r_s,P_rs,x)
                            ##
                            loss_batch_aug_neg=0
                            ##
                            for r1 in R_p:
                                data_temp1=P_rp[r1]
                                data_temp1=[data_temp1[k]for k in data_temp1.keys()]
                                positive=[]
                                for t in data_temp1:
                                    for t_ in t:
                                        positive.append(t_)
                                

                                negative=[]
                                for r2 in R_p:
                                    if r1==r2:
                                        continue
                                    data_temp2=P_rp[r2]
                                    data_temp2=[data_temp2[k]for k in data_temp2.keys()]
                                    for t in data_temp2:
                                        for t_ in t:
                                            negative.append(t_)
                                if len(positive)<1 or len(negative)<1:
                                        continue
                                positive=torch.stack(positive,0)
                                negative=torch.stack(negative,0)
                            
                                loss_batch_aug_neg+=loss_nce(positive,negative,linear=self.linear,device=negative.device)
                                ################
                                for r1 in R_s:
                                    data_temp1=P_rs[r1]
                                    data_temp1=[data_temp1[k]for k in data_temp1.keys()]
                                    positive=[]
                                    for t in data_temp1:
                                        for t_ in t:
                                            positive.append(t_)
                                    negative=[]
                                    for r2 in R_s:
                                        if r1==r2:
                                            continue
                                        data_temp2=P_rs[r2]
                                        data_temp2=[data_temp2[k]for k in data_temp2.keys()]
                                        for t in data_temp2:
                                            for t_ in t:
                                                negative.append(t_)
                                    if len(positive)<1 or len(negative)<1:
                                        continue
                                    positive=torch.stack(positive,0)
                                    negative=torch.stack(negative,0)

                                    loss_batch_aug_neg+=loss_nce(positive,negative,linear=self.linear,device=negative.device)

                            batch_n_p=[]
                            batch_n_n=[]
                            batch_n_a=[]
                            i=None
                            Rs=[]
                            batch_n_a,batch_n_p,batch_n_n,Rs=batch_n(P_rp,R_p,batch_n_a,batch_n_p,batch_n_n,i,Rs)
                            #print('Rs**',Rs)
                            if len(batch_n_p)==len(batch_n_n) and  len(batch_n_n) == len(batch_n_a) and len(batch_n_a) not in [0,1]:

                                #print(batch_n_p[0].shape)
                                batch_n_p=torch.stack(batch_n_p,0)
                                batch_n_n=torch.stack(batch_n_n,0)
                                batch_n_a=torch.stack(batch_n_a,0)
                                a=batch_n_a
                                p=batch_n_p
                                n=batch_n_n
                                loss_batch_aug_neg+=loss_triplet(a, p, n,augument=True)

                            batch_n_p_=[]
                            batch_n_n_=[]
                            batch_n_a_=[]
                            i=None
                            Rs_=[]
                            batch_n_a_,batch_n_p_,batch_n_n_,Rs=batch_n(P_rs,R_s,batch_n_a_,batch_n_p_,batch_n_n_,i,Rs_)
      
                            if len(batch_n_p_)==len(batch_n_n_) and  len(batch_n_n_) == len(batch_n_a_) and len(batch_n_a_) not in [0,1]:

                                
                                batch_n_p_=torch.stack(batch_n_p_,0)
                                batch_n_n_=torch.stack(batch_n_n_,0)
                                batch_n_a_=torch.stack(batch_n_a_,0)
                                loss_batch_aug_neg+=loss_triplet(a, p, n,augument=True)

                            loss=loss_batch_aug_neg
                        if loss==0 or args.fin_tune:
                            loss=CosLoss(x1, x2, y0)
                        ####
                  

                        out=0
                        out_=0

                        return out,out_,loss


                 
                    else:
                        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                        w1_w3_sim = cos(w1_h, w3_h)
                        w2_w4_sim = cos(w2_h, w4_h)

                        w3_w4_sim=cos(w3_h, w4_h)
        

                     
                        CosLoss = nn.CosineEmbeddingLoss()

                        loss=0

                        w1_w3_dist = torch.nn.functional.pairwise_distance(w1_h, w3_h)#(w1_h - w3_h).pow(2).sum(1).sqrt()
                        w2_w4_dist = torch.nn.functional.pairwise_distance(w2_h, w4_h)#(w2_h - w4_h).pow(2).sum(1).sqrt()
                        return  x1,x2,loss,w3_w4_sim,w1_w3_sim,w2_w4_sim,w1_w3_dist,w2_w4_dist,similarity_in_hidden_layers
     



                else:




                    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
                    w1_w3_sim = cos(w1_h, w3_h)
                    w2_w4_sim = cos(w2_h, w4_h)

                    w3_w4_sim=cos(w3_h, w4_h)

                    CosLoss = nn.CosineEmbeddingLoss()
                    loss=CosLoss(x1, x2, y0)
            

                    

                       
                    if eval==False:
                        #loss=CosLoss(x1, x2, y0)
                        print('loss',loss)
                       
                        return out,loss 
                    else:
                        w1_w3_dist = torch.nn.functional.pairwise_distance(w1_h, w3_h)#(w1_h - w3_h).pow(2).sum(1).sqrt()
                        w2_w4_dist = torch.nn.functional.pairwise_distance(w2_h, w4_h)#(w2_h - w4_h).pow(2).sum(1).sqrt()
                        return  x1,x2,loss,w3_w4_sim,w1_w3_sim,w2_w4_sim,w1_w3_dist,w2_w4_dist,similarity_in_hidden_layers

        elif  args.data_type in ['wikidata','EVALution','tacred','retacred','conll']:
        
  

            if args.abstract=='mask':
                sentence_flagged_idxs=batch['sentence_masked_flagged_tokens']
                sentence_flagged_masks_idxs=batch['sentence_masked_flagged_tokens_masks']
                ents_flagged_plus_rel_idxs=batch['ents_flagged_plus_rel_tokens']
                Len_Target=batch['Len_Target']
       

            elif args.abstract=='abstract':

                sentence_flagged_idxs=batch['sentecne_ents_abstracted_flagged_tokens']
                sentence_flagged_masks_idxs=batch['sentecne_ents_abstracted_flagged_tokens_masks']
                ents_flagged_plus_rel_idxs=batch['abstracted_ents_flagged_plus_rel_tokens']
                #ents_flagged_plus_rel_idxs=batch['abstracted_ents_flagged_idxs']
                Len_Target=batch['abstract_Len_Target']


         
            elif args.abstract=='flagged_ents':

                sentence_flagged_idxs=batch['ents_flagged_tokens'] 
                sentence_flagged_masks_idxs=batch['ents_flagged_tokens_masks'] 
                # sentence_flagged_idxs=batch['sentence_ents_flagged_tokens']
                # sentence_flagged_masks_idxs=batch['sentence_ents_flagged_tokens_masks']
                ents_flagged_plus_rel_idxs=batch['ents_flagged_plus_rel_tokens']
                Len_Target=batch['Len_Target']


            else:
                #sentence_masked_idxs sentence_flagged_idxs
                sentence_flagged_idxs=batch['EntsAbst_flagged_tokens'] 
                sentence_flagged_masks_idxs=batch['EntsAbst_flagged_tokens_masks'] 
                ents_flagged_plus_rel_idxs=batch['EntsAbst_flagged_plus_rel_tokens']
                Len_Target=batch['ent_abstract_Len_Target']
                n=Len_Target[0]




                n=Len_Target[0]
                temp=ents_flagged_plus_rel_idxs[0,n-3]


                #print('temp',temp)

            tmp=['paper2_WikidataPretraining','paper2_EVALutionPretraining','paper2_lexicalZeroshotTraining','paper2_RE_Trained_lexicalTraining','six']

            # print('args.experiment_no ',args.experiment_no )
            # exit()
            if args.experiment_no in tmp :
                if args.abstract=='flagged_ents':


                    sentence_flagged_idxs=batch['ents_flagged_tokens']
                    sentence_flagged_masks_idxs=batch['ents_flagged_tokens_masks']
                elif args.abstract=='mix':
                    sentence_flagged_idxs=batch['EntsAbst_flagged_plus_rel_tokens']
                    sentence_flagged_masks_idxs=batch['EntsAbst_flagged_plus_rel_tokens_masks']


            x1=self.transformer(input_ids=sentence_flagged_idxs,attention_mask=sentence_flagged_masks_idxs).hidden_states
            # x1_=[]
            # for xi in range(len(x1)):
            #         if xi%2!=0:
            #             x1_.append(x1[xi])
            # x1=x1_



            
            if eval_epxeriment_data!=None and (args.data_type=='retacred'  or  args.data_type=='conll') :
                        p=[1,2,3,4,5,6,7]
                        s=random.sample(p,1)[0]

                        if s in [1,2]:


                            L=sentence_flagged_idxs
                            self.get_head_tail_rep(L,batch,x1,eval_epxeriment_data)





            ################################

            e1_se=self.tokenizer.convert_tokens_to_ids(['[e11]','[e12]'])
            e2_se=self.tokenizer.convert_tokens_to_ids(['[e21]','[e22]'])

            if self.pretrained_name=='roberta-large':
                #Ġ
                e1_se=self.tokenizer.convert_tokens_to_ids(['[e11]','[e12]'])
                e2_se=self.tokenizer.convert_tokens_to_ids(['[e21]','[e22]'])




            e_1,e_2=[],[],
            max_length={'w1':0,'w2':0,'w3':0,'w4':0}
            abcd_l=sentence_flagged_idxs.tolist()
            abcd= x1
     

            for i in range(len(abcd_l)):

                abcd_source=abcd_l
    
                e1_start=abcd_source[i].index(e1_se[0]) if e1_se[0] in abcd_source[i] else 1
                e1_end=abcd_source[i].index(e1_se[1]) if e1_se[1] in abcd_source[i] else len(abcd_source[i])-1

                #
                e2_start=abcd_source[i].index(e2_se[0]) if e2_se[0] in abcd_source[i] else len(abcd_source[i])-1
                e2_end=abcd_source[i].index(e2_se[1])  if e2_se[1] in abcd_source[i] else len(abcd_source[i])-1

                #############################

                # print('if e1_se[0] in abcd_source[i',e1_se[0] in abcd_source[i])
                # print('e2_se[0] in abcd_source[i]',e2_se[0] in abcd_source[i])

                # print('if e1_se[1] in abcd_source[i',e1_se[1] in abcd_source[i])
                # print('e2_se[1] in abcd_source[i]',e2_se[1] in abcd_source[i])
                # exit()




                #print('if w4_se[1] in abcd_source[i] else len(abcd_source[i])', w4_se[1] in abcd_source[i] )

                ##########
                # select batch
                abcd_source=abcd 
                out=[o[i] for o in abcd_source]
      
                out=[o[e1_start:e1_end+1,:] for o in out]
                shappes=[o.shape[0] for o in out]
                t=shappes[0]
                max_length['w1']= t if t>max_length['w1'] else max_length['w1']
                e_1.append(out)
                ##############
                ##########
                # select batch
                abcd_source=abcd 
                out=[o[i] for o in abcd_source]
      
                out=[o[e2_start:e2_end+1,:] for o in out]
                shappes=[o.shape[0] for o in out]
                t=shappes[0]
                max_length['w2']= t if t>max_length['w2'] else max_length['w2']
                e_2.append(out)


                ###############################################

         

            # print('e_1',e_1[0].shape)
            # print('e_2',e_2[0].shape)

            e_1_,e_2_=[],[]
            for e in e_1:


                e_t = torch.stack(e, dim=-2)
                #print('e1',e[0].shape)
                temp=torch.zeros(max_length['w1']-e_t.shape[0],e_t.shape[1],e_t.shape[2]).to(self.device)
                e_t=torch.cat((e_t,temp),0)
                e_1_.append(e_t)
                #print('e1',e.shape)

            for e in e_2:

                #print('e2',e.shape)

                e_t = torch.stack(e, dim=-2)
                temp=torch.zeros(max_length['w2']-e_t.shape[0],e_t.shape[1],e_t.shape[2]).to(self.device)
                e_t=torch.cat((e_t,temp),0)
                e_2_.append(e_t)
                #print('e2',e[0].shape)


            e1 = torch.stack(e_1_, dim=0)
            e2 = torch.stack(e_2_, dim=0)



            e1=[e1[:,:,i,:] for i in range(e1.shape[-2])]
            e2=[e2[:,:,i,:] for i in range(e2.shape[-2])]


       
            e1 = torch.stack(e1, dim=0)
            e1=torch.swapaxes(e1, 0, 1)
            e1=torch.swapaxes(e1, 1, 2)

            #
            e2 = torch.stack(e2, dim=0)
            e2=torch.swapaxes(e2, 0, 1)
            e2=torch.swapaxes(e2, 1, 2)
            #######
            e1_h = self.head_3(e1)  # [chunks in batch, n pos, d_depth, d_emb]
            e2_h = self.head_3(e2)


  
            x=e1_h-e2_h


      
            relations = self.classifier(x)
            h=None
            out=None

            return h,out,relations#out#tgt,relations
        elif args.data_type=='conceptqa':

            
            q=batch['q']
            answer=batch['answer']#answer
            a=batch['a']
            b=batch['b']
            c=batch['c']
            d=batch['d']
            e=batch['e']
            y=batch['y']
            ###
            q_e = self.transformer(input_ids=q).hidden_states
            answer_e = self.transformer(input_ids=answer).hidden_states
            a_e = self.transformer(input_ids=a).hidden_states
            b_e = self.transformer(input_ids=b).hidden_states
            c_e = self.transformer(input_ids=c).hidden_states

            d_e = self.transformer(input_ids=d).hidden_states
            e_e = self.transformer(input_ids=d).hidden_states
            #print(q_e.shape)
            if args.route_or_baseline!='route':
                q_e=q_e[-1].sum(1)
                answer_e=answer_e[-1].sum(1)

                a_e=a_e[-1].sum(1)
                b_e=b_e[-1].sum(1)
                c_e=c_e[-1].sum(1)
                d_e=d_e[-1].sum(1)
                e_e=e_e[-1].sum(1)
            else:
                q_e=torch.stack(q_e,0)
                q_e=torch.swapaxes(q_e,0,1)
                q_e=torch.swapaxes(q_e, 1, 2)
                #w1 torch.Size([2, 6, 25, 1024])
                #print(q_e.shape)

                answer_e=torch.stack(answer_e,0)
                answer_e=torch.swapaxes(answer_e,0,1)
                answer_e=torch.swapaxes(answer_e, 1, 2)

                # print(answer_e.shape)
                # exit()

                a_e=torch.stack(a_e,0)
                a_e=torch.swapaxes(a_e,0,1)
                a_e=torch.swapaxes(a_e, 1, 2)


                b_e=torch.stack(b_e,0)
                b_e=torch.swapaxes(b_e,0,1)
                b_e=torch.swapaxes(b_e, 1, 2)

                c_e=torch.stack(c_e,0)
                c_e=torch.swapaxes(c_e,0,1)
                c_e=torch.swapaxes(c_e, 1, 2)


                d_e=torch.stack(d_e,0)
                d_e=torch.swapaxes(d_e,0,1)
                d_e=torch.swapaxes(d_e, 1, 2)

                e_e=torch.stack(e_e,0)
                e_e=torch.swapaxes(e_e,0,1)
                e_e=torch.swapaxes(e_e, 1, 2)



                q_e=self.head_3(q_e)
                answer_e=self.head_3(answer_e)

                a_e=self.head_3(a_e)
                b_e=self.head_3(b_e)
                c_e=self.head_3(c_e)
                d_e=self.head_3(d_e)
                e_e=self.head_3(e_e)

                #print('q_e',q_e.shape)


            #print('d',d_e.shape)
            CosLoss = nn.CosineEmbeddingLoss()
            if eval==False:
                positive_label=torch.ones(q.shape[0]).to(q.device)
                negative_label=torch.zeros(q.shape[0]).to(q.device)
                ######
                y_dic={}
                for ti,t in enumerate(y.tolist()):
                    if t in y_dic.keys():
                        temp=a_e[ti]
                        y_dic[t].append(temp)
                    else:
                        temp=a_e[ti]
                        y_dic[t]=[]
                        y_dic[t].append(temp)

                positive=[]
                negative=[]
                #print('y_dic',y_dic.keys())
                for k in y_dic.keys():
                    p=y_dic[k]
                    for k_ in y_dic.keys():
                        if k==k_:
                            continue
                        n=y_dic[k_]
                        positive.extend(p)
                        negative.extend(n)
                        if len(positive)>len(negative):
                            idx=len(positive)-len(negative)
                            positive=positive[:-idx]
                        elif len(positive)<len(negative):
                            idx=len(negative)-len(positive)
                            negative=negative[:-idx]

                positive=torch.stack(positive,0)
                negative=torch.stack(negative,0)
                # print('positive',positive.shape)
                # print('negative',negative.shape)
                negative_label_aug=-1*torch.ones(positive.shape[0]).to(q.device)

                aug_n_loss=CosLoss(positive, negative, negative_label_aug)









                ######conceptqa_easy # conceptqa_hard
                loss_answer = CosLoss(q_e, answer_e, positive_label)
                loss_a = CosLoss(q_e, a_e, negative_label)  if args.hard=='conceptqa_hard' else 0
                loss_b = CosLoss(q_e, b_e, negative_label)
                loss_c = CosLoss(q_e, c_e, negative_label)
                loss_d = CosLoss(q_e, d_e, negative_label) 
                loss_e = CosLoss(q_e, e_e, negative_label) if args.hard=='conceptqa_easy' else 0

                loss=loss_answer+loss_a+loss_b+loss_c+loss_d+aug_n_loss+loss_e
                return loss
            else:

                cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
          
       
                answer_sim=cos(q_e, answer_e)

                a_sim =cos(q_e, a_e)

                b_sim=cos(q_e, b_e)
                c_sim=cos(q_e, c_e)
                d_sim=cos(q_e, d_e)
                e_sim=cos(q_e, e_e)


                return answer_sim,a_sim,b_sim,c_sim,d_sim,e_sim

        

   






