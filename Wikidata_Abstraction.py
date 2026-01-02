

# util code for finding entiy type in the dataset wikidata
# written by Ramazan Bahrami



def find_of_and_add_subscript(label,subscript='x'):
    label_token=label.split(' ')
    new_label=[]
    temp=True
    for ti,t in enumerate(label_token):
        if t=='of':
            temp=False
            new_label.append(subscript)

        new_label.append(t)
    new_label=' '.join(new_label)
    return new_label,temp

def get_ent_instances_or_subclass_up(e,type='sublclass'):

    if type=='sublclass':
        p='P279'
    else:
        p='P31'
    flag=True
    number_try=0
    n=1
    while flag:
        if number_try==n:
            return False
        query1 =(
        ' SELECT ?subclass ?subclassLabel WHERE { ', 
            
                'wd:'+e+' wdt:'+p+' ?subclass.' ,
        '\n SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } } LIMIT  100')
        query1="".join(query1)
        url = 'https://query.wikidata.org/sparql'
        try:
            data = requests.get(url, params={'query': query1, 'format': 'json'},timeout=300).json()
        except requests.exceptions.Timeout:
            continue
        except :
            number_try=number_try+1
            continue
        Ps=[]
        if len(data['results']['bindings'])>0:
            for d in data['results']['bindings']:
                #print(d)
                p=d['subclass']['value']
                p=str(p).split('/')[-1]

                l=d['subclassLabel']['value']
                l=str(l).split('/')[-1]
                item=(p,l)
                Ps.append(item)
            return Ps


def get_ent_instances_or_subclass_down(e,type='sublclass'):
    if type=='sublclass':
        p='P279'
    else:
        p='P31'
    flag=True
    number_try=0
    n=1
    while flag:
        if number_try==n:
            return False
        query1 =(
        ' SELECT ?subclass ?subclassLabel WHERE { ', 
            
                '?subclass wdt:'+p+' wd:'+e+'.' ,
        '\n SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } } LIMIT  100')
        query1="".join(query1)
        url = 'https://query.wikidata.org/sparql'
        try:
            data = requests.get(url, params={'query': query1, 'format': 'json'},timeout=300).json()
        except requests.exceptions.Timeout:
            continue
        except :
            number_try=number_try+1
            continue
        Ps=[]
        if len(data['results']['bindings'])>0:
            for d in data['results']['bindings']:
                #print(d)
                p=d['subclass']['value']
                p=str(p).split('/')[-1]

                l=d['subclassLabel']['value']
                l=str(l).split('/')[-1]
                item=(p,l)
                Ps.append(item)
            return Ps

def get_preferred_instances(instances_heirarchy,instance):
    instance_l=[e['kbID'] for e in instance]
    ########################################
    def update_h(instances_heirarchy,heirarchy,type='instance_of'):
        for key in heirarchy.keys():
            value=heirarchy[key]
            for v in value:
                value_k,value_l=v[0],v[1]
                tupl_item=(value_k,value_l)
                if type=='subclass_of':

                    if key not in instances_heirarchy['subclass']['has_subclass'].keys():
                        item={'keys':[]}
                        instances_heirarchy['subclass']['has_subclass'][key]=item
                        instances_heirarchy['subclass']['has_subclass'][key]['keys'].append(value_k)
                        instances_heirarchy['subclass']['has_subclass'][key]['keys']=\
                        list(set(instances_heirarchy['subclass']['has_subclass'][key]['keys']))
                    else:
                        instances_heirarchy['subclass']['has_subclass'][key]['keys'].append(value_k)
                        instances_heirarchy['subclass']['has_subclass'][key]['keys']=\
                        list(set(instances_heirarchy['subclass']['has_subclass'][key]['keys']))

                    if value_k not in instances_heirarchy['subclass']['subclass_of'].keys():
                        item={'keys':[]}
                        instances_heirarchy['subclass']['subclass_of'][value_k]=item
                        instances_heirarchy['subclass']['subclass_of'][value_k]['keys'].append(key)
                        instances_heirarchy['subclass']['subclass_of'][value_k]['keys']=\
                        list(set(instances_heirarchy['subclass']['has_subclass'][key]['keys']))
                    else:
                        instances_heirarchy['subclass']['subclass_of'][value_k]['keys'].append(key)
                        instances_heirarchy['subclass']['subclass_of'][value_k]['keys']=\
                        list(set(instances_heirarchy['subclass']['has_subclass'][key]['keys']))
                if type=='instance_of':
                    if key not in instances_heirarchy['instance']['has_instance'].keys():
                        item={'keys':[]}
                        instances_heirarchy['instance']['has_instance'][key]=item
                        instances_heirarchy['instance']['has_instance'][key]['keys'].append(value_k)
                        instances_heirarchy['instance']['has_instance'][key]['keys']=\
                        list(set(instances_heirarchy['instance']['has_instance'][key]['keys']))
          
                    else:
                        instances_heirarchy['instance']['has_instance'][key]['keys'].append(value_k)
                        instances_heirarchy['instance']['has_instance'][key]['keys']=\
                        list(set(instances_heirarchy['instance']['has_instance'][key]['keys']))
                    if value_k not in instances_heirarchy['instance']['instance_of'].keys():
                        item={'keys':[]}
                        instances_heirarchy['instance']['instance_of'][value_k]=item
                        instances_heirarchy['instance']['instance_of'][value_k]['keys'].append(key)
                        instances_heirarchy['instance']['instance_of'][value_k]['keys']=\
                        list(set(instances_heirarchy['instance']['instance_of'][value_k]['keys']))
                    else:
                        instances_heirarchy['instance']['instance_of'][value_k]['keys'].append(key)
                        instances_heirarchy['instance']['instance_of'][value_k]['keys']=\
                        list(set(instances_heirarchy['instance']['instance_of'][value_k]['keys']))
        return instances_heirarchy

    def get_hierarchy(q,ql,h,L,temp='up'):
        if L:
            if temp=='up':
                for ins in L:
                    k,l=ins[0],ins[1]
                    if k not in instances_heirarchy['q_to_labels'].keys():
                        instances_heirarchy['q_to_labels'][k]=l
                    if k in h:
                        h[k].append((q,ql))
                    else:
                        h[k]=[]
                        h[k].append((q,ql))
            if temp=='down':
                t=False
                L_qs=[e[0] for e in L]

                for l in instance_l:
                    if l in L_qs:
                        t=True
                if t:
                    for ins in L:
                        k,l=ins[0],ins[1]
                        if k not in instances_heirarchy['q_to_labels'].keys():
                            instances_heirarchy['q_to_labels'][k]=l
                        if q in h:
                            h[q].append((k,l))
                        else:
                            h[q]=[]
                            h[q].append((k,l))
        return h

    heirarchy_subclass={}
    heirarchy_instances={}
    for e in instance:
        q=e['kbID']
        ql=e['label']
        time.sleep(1)
        subclass_of=get_ent_instances_or_subclass_up(q,type='sublclass')
        subclass_of_down=get_ent_instances_or_subclass_down(q,type='sublclass')
        time.sleep(1)
        instances_of=get_ent_instances_or_subclass_up(q,type='instance_of')
        instances_of_down=get_ent_instances_or_subclass_down(q,type='sublclass')
        heirarchy=get_hierarchy(q,ql,heirarchy_instances,instances_of,temp='up')
        heirarchy=get_hierarchy(q,ql,heirarchy_instances,instances_of_down,temp='down')
        heirarchy=get_hierarchy(q,ql,heirarchy_subclass,subclass_of,temp='up')
        heirarchy=get_hierarchy(q,ql,heirarchy_subclass,subclass_of_down,temp='down')
    instances_heirarchy=update_h(instances_heirarchy,heirarchy_instances,type='instance_of')
    instances_heirarchy=update_h(instances_heirarchy,heirarchy_subclass,type='subclass_of')
    return instances_heirarchy

        
def check_for_prefered_instance(instances_heirarchy,instances):
    def get_intersection(dic_l,type='of'):
        L=[]
        for k in dic_l.keys():
            l=dic_l[k][type]
            L.append(l)
        count_dic={}
        for l in L:
            for q in l:
                if q not in count_dic.keys():
                    count_dic[ q]=1
                else:
                    count_dic[ q]=count_dic[ q]+1
        return count_dic
    def get_prefered(type='instance'):
        temp_dic={}
        selected=None
        temp=False
        if type=='instance':
            has='has_instance'
            of='instance_of'
        else:
            has='has_subclass'
            of='subclass_of'

        for e1 in instances:
            eq1,el1=e1['kbID'],e1['label']

            if eq1 not in temp_dic.keys():
                item={'has':[],'of':[]}
                temp_dic[eq1]=item
            e1_has_instance,e1_instance_of=[],[]
            if eq1 in instances_heirarchy[type][has].keys():
                e1_has_instance=instances_heirarchy[type][has][eq1]['keys']
                temp_dic[eq1]['has']=e1_has_instance
            if eq1 in instances_heirarchy[type][of].keys():
                e1_instance_of=instances_heirarchy[type][of][eq1]['keys']
                temp_dic[eq1]['of']=e1_instance_of
            for e2 in instances:
                if temp:
                    break
                eq2,el2=e2['kbID'],e2['label']
                ##############################
                if eq2 not  in temp_dic.keys():
                    item={'has':[],'of':[]}
                    temp_dic[eq2]=item
                ############
                if eq1==eq2:
                    continue
                else:
                    #´check has instance
                    if eq2 in e1_has_instance:

                        selected=[e1]
                        temp=True
                    if temp:
                        continue
                    if eq2 in instances_heirarchy[type][has].keys():
                        e2_has_instance=instances_heirarchy[type][has][eq2]['keys']
                        temp_dic[eq2]['has']=e2_has_instance
                        if eq1 in e2_has_instance:
                            selected=[e2]
                            temp=True
                    if temp:
                        continue

                    if eq2 in instances_heirarchy[type][of].keys():
                        e2_instance_of=instances_heirarchy[type][of][eq2]['keys']
                        temp_dic[eq2]['of']=e2_instance_of
                        if eq1 in e2_instance_of:
                            selected=[e1]
                            temp=True
                    if temp:
                        continue
            if temp:
                break
        return selected,temp_dic
    selected,t_dic1=get_prefered(type='subclass')
    if selected==None:
        selected,t_dic2=get_prefered(type='instance')
    return selected





def anonymous_flag_all_ents(entities_context,instances_heirarchy,token,vertexDic,has_superclass_no_superclass_count):

    token_flagged_all_ents=[]
    ents_start=[min(v['tokenpositions']) for v in vertexDic if len(v['tokenpositions'])>=1]
    ents_end=[max(v['tokenpositions']) for v in vertexDic if len(v['tokenpositions'])>=1]
    ents_qs=[v for v in vertexDic if len(v['tokenpositions'])>=1]
    q_instances_list=[]
    q_instance_dic={}
    update=False
    for v in ents_qs:

        q,l=v['kbID'],v['lexicalInput']
        if q in entities_context:
                    q_instances=entities_context[q]['instances']
                    if len(q_instances)==1:

                        t=q_instances[0]
                        #print('onle selection',t)
                        q_instances_list.append(t)

                    if update:
                        for q in q_instances:
                            kbID=q['kbID']
                            label=q['label']
                            if kbID in instances_heirarchy['subclass']['subclass_of'].keys():
                                temp=instances_heirarchy['subclass']['subclass_of'][kbID]
                                has_superclass_no_superclass_count['has_superclass']=\
                                has_superclass_no_superclass_count['has_superclass']+1
                                ### 
                                continue
                                # for t in temp['keys']:
                                #     if t in instances_heirarchy['subclass']['subclass_of'].keys():
                                #     else:
                                #         if t in instances_heirarchy['q_to_labels'].keys():
                                #             l=instances_heirarchy['q_to_labels'][t]
                                #             item={'kbID':t,'label':l}
                                #             instance=[item]
                                #             instances_heirarchy=get_preferred_instances(instances_heirarchy,instance)
                                #             new=t in instances_heirarchy['subclass']['subclass_of'].keys()
                     
                            else:
                                instances_heirarchy=get_preferred_instances(instances_heirarchy,instance)
                                new=kbID in instances_heirarchy['subclass']['subclass_of'].keys()
    
                    elif len(q_instances)==0 :
                        item={'kbID': q, 'label': l}
                        if q in instances_heirarchy['instance']['instance_of'].keys():

                            item={'kbID': q, 'label': l}
                            q_instances=instances_heirarchy['instance']['instance_of'][q]
                            q_instances=q_instances['keys']  if isinstance(q_instances, dict) else q_instances

                            if len(q_instances)==1:
                                qins=q_instances[0]
                                qins_l=instances_heirarchy['q_to_labels'][qins]
                                item={'kbID': qins, 'label': qins_l}
                                q_instances_list.append(item)
                            elif len(q_instances)!=0:
                                temp_L=[]
                                for qins in q_instances:
                                    qins_l=instances_heirarchy['q_to_labels'][qins]
                                    item={'kbID': qins, 'label': qins_l}
                                    temp_L.append(item)
                                selected_instance=check_for_prefered_instance(instances_heirarchy,temp_L)
                                if selected_instance!=None:
                                    q_instances_list.append(selected_instance[0])
                                else:
                                    t=temp_L[0]

                                    q_instances_list.append(t)
                        # elif q in instances_heirarchy['subclass']['subclass_of'].keys(): 
                        #     q_instances=instances_heirarchy['subclass']['subclass_of'][q]
                        #     q_instances=q_instances['keys']  if isinstance(q_instances, dict) else q_instances
                        #     #print('q_instances',q_instances)
                        #     CNT_var['count']=CNT_var['count']+1
                        #     if len(q_instances)==1:
                        #         qins=q_instances[0]
                        #         qins_l=instances_heirarchy['q_to_labels'][qins]
                             
                        #         item={'kbID': qins, 'label': qins_l}

                       
                        #         q_instances_list.append(item)
                        #         #print('1.replace item',item)

                        #     elif len(q_instances)!=0:
                        #         temp_L=[]
                        #         for qins in q_instances:
                                    
                        #             qins_l=instances_heirarchy['q_to_labels'][qins]
                                 
                        #             item={'kbID': qins, 'label': qins_l}
                        #             temp_L.append(item)

                        #         selected_instance=check_for_prefered_instance(instances_heirarchy,temp_L)
                        #         if selected_instance!=None:
                        #             q_instances_list.append(selected_instance[0])
                        #             #print('2.replace item',item)
                        #         else:
                        #             t=temp_L[0]
                        #             #print('2.replace item',t)
                
                        #             q_instances_list.append(t)

                        else:
                            item={'kbID': q, 'label': l}

                            # instance=[item]
                            # temp1=q in instances_heirarchy['subclass']['subclass_of'].keys()
                            # print('temp1',temp1)
                            # temp2=q in instances_heirarchy['instance']['instance_of'].keys()
                            # print('temp2',temp2)
                            # if q not in CNT_var['subclass_of_not_found'] or q not in CNT_var['instance_of_not_found']:  
                            
                            #     instances_heirarchy=get_preferred_instances(instances_heirarchy,instance)
                            #     temp1=q in instances_heirarchy['subclass']['subclass_of'].keys()
                            #     print('temp1',temp1)
                            #     if temp1==False:
                            #         CNT_var['subclass_of_not_found'].add(q)
                            #     temp2=q in instances_heirarchy['instance']['instance_of'].keys()
                            #     print('temp2',temp2)
                            #     if temp2==False:
                            #         CNT_var['instance_of_not_found'].add(q)

                            q_instances_list.append(item)
                            
                    else:
                        selected_instance=check_for_prefered_instance(instances_heirarchy,q_instances)
                        if selected_instance!=None:
                            q_instances_list.append(selected_instance[0])
                        else:
                            t=q_instances[0]
                            q_instances_list.append(t)
        else:
            label='date'
            item={'kbID': q, 'label': label}
            if len(l)!=len(str('1993-01-01')):
                print('item',item)
            q_instances_list.append(item)

    between_s_e_flag=False
    token_abstracted_flagged_all_ents=[]
    label_freq={}
    additional_l=[str(i) for i in range(20)]
    subscript_l=['X','Y','A','B','C','G','H','K','L','M','N','P','Q',]+additional_l
    abstracted_labels_list=[]
    abstracted_q_to_labels={}
    #########
    for si,q in enumerate(ents_qs):

        kbID=q['kbID']

        temp=q_instances_list[si]['label']
        ###
        if temp not in label_freq.keys():
            label_freq[temp]=0
        else:
            label_freq[temp]=1+label_freq[temp]
        label_of_indexed,temp1=find_of_and_add_subscript(temp)

        subscript1=subscript_l[label_freq[temp]]
        label_of_indexed=label_of_indexed+' '+str(subscript1)  if temp1==True else label_of_indexed
        temp=label_of_indexed
        abstracted_labels_list.append(temp)

        abstracted_q_to_labels[kbID]=temp
    for ti,t in enumerate(token):

        for si,start in enumerate(ents_start):
            if ti ==start:
                between_s_e_flag=True
                temp=abstracted_labels_list[si]
                temp=temp.split(' ')
                token_abstracted_flagged_all_ents.append('[es]')
                token_abstracted_flagged_all_ents=token_abstracted_flagged_all_ents+temp

        for end in ents_end:
            if ti ==end+1:
                between_s_e_flag=False
            if ti ==end:
                token_abstracted_flagged_all_ents.append('[en]')
                
        if between_s_e_flag==False:
            token_abstracted_flagged_all_ents.append(t)

    ####################################################
    for ti,t in enumerate(token):
       
        for start in ents_start:
            if ti ==start:
                token_flagged_all_ents.append('[es]')

        token_flagged_all_ents.append(t)
        for end in ents_end:
            if ti ==end:
                token_flagged_all_ents.append('[en]')
    

    return token_flagged_all_ents,token_abstracted_flagged_all_ents,abstracted_q_to_labels