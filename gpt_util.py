

from Experiment_Util import *


def get_lexical_data(dn):
    from datasets import load_dataset
    from datasets import load_from_disk

    if dn=='BLESS':
        BLESS = load_from_disk('unprocessed_data/BLESS')
        BLESS_train=BLESS['train']
        BLESS_test=BLESS['test']
        file='essential_files/'+str(dn)+'rel_dic.json'
        relations= json.load(open(file))['rel_dic'].keys()
        return BLESS_test,relations

    if dn=='CogALexV':

        CogALexV = load_from_disk('unprocessed_data/CogALexV')
        CogALexV_train=CogALexV['train']
        CogALexV_test=CogALexV['test']
        file='essential_files/'+str(dn)+'rel_dic.json'
        relations= json.load(open(file))['rel_dic'].keys()
        return CogALexV_test,relations
    if dn=='EVALution':

        EVALution = load_from_disk('unprocessed_data/EVALution')

        EVALution_train=EVALution['train']
        EVALution_test=EVALution['test']
        file='essential_files/'+str(dn)+'rel_dic.json'
        relations= json.load(open(file))['rel_dic'].keys()
        return EVALution_test,relations

    if dn=='ROOT09':

        ROOT09 = load_from_disk('unprocessed_data/ROOT09')

        ROOT09_train=ROOT09['train']
        ROOT09_test=ROOT09['test']
        file='essential_files/'+str(dn)+'rel_dic.json'
        relations= json.load(open(file))['rel_dic'].keys()
        return ROOT09_test,relations
    if dn=='KandH_plus_N':



        KandH_plus_N = load_from_disk('unprocessed_data/KandH_plus_N')

        KandH_plus_N_train=KandH_plus_N['train']
        KandH_plus_N_test=KandH_plus_N['test']
        file='essential_files/'+str(dn)+'rel_dic.json'
        relations= json.load(open(file))['rel_dic'].keys()
        return ROOT09_test,relations


def make_conceptqa_questions(easy,test_or_dev='test'):
    benchmarks='wikidata'
    train_eval='train'
    file='unprocessed_data/SRE_Analogy.json'
    SRE_Analogy= json.load(open(file))['SRE_Analogy']['wikidata']

    data=SRE_Analogy[test_or_dev]
    abstract_spaces_dic=get_google_abstract_space_dic(semeval_or_google='wikidata')
    Questions =get_conceptqa_questions(abstract_spaces_dic,n_size=2,easy=easy)
    all_data=[]

    answer_set={}


    for qi,question in enumerate(Questions):
        q=question['q']
        answer=question['answer']
        choices=question['choices']
        key=question['key']
        letters=['a','b','c','d','e','f','g','h','q']
        q='questionNO'+str(qi)+': what is similar to'+ str(q)+' .'
        item={'questions':q,'options':{}}
        random.shuffle(letters)
        for ci,c in enumerate(choices):
            l=letters[ci]
            option= str(c)
            item['options'][l]=option
            if c==answer:
                answer_set[qi]=(answer,l)
        all_data.append(item)
    return all_data,answer_set 


     


def Lexical_relations_as_mtcq(dn):


    data,relations=get_lexical_data(dn)

    relations=list(relations)
    all_data=[]
    print('relations',relations)
    answer_set={}
    letters=['a','b','c','d','e','f','g','h','q']
    for ti,t in enumerate(data):
        {'head': 'lobby', 'tail': 'wall', 'relation': 'HasA'}
        head=t['head']
        tail=t['tail']
        relation=t['relation'].lower()

        q='questionNO'+str(ti)+':chose chose the relatin between '+ str(head)+' and '+ str(tail)+' .'

        item={'questions':q,'options':{}}
        
        random.shuffle(relations)
        random.shuffle(letters)

        for ci,c in enumerate(relations):
            l=letters[ci]
          
  
            option= str(c)
            item['options'][l]=option
            if c==relation:
                answer_set[ti]=(relation,l)
                print('c,l',c,l)
                print('q',q)
                print('#######')
        all_data.append(item)
    return all_data,answer_set 
     

def analogy_questions_as_json(dn='sat'):
    import matplotlib.pyplot as plt
    import numpy as np
    # ##
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import math


    file='essential_files/wordanalogy_'+'DATA_multi_choice.json'


    file='essential_files/wordanalogy_'+'DATA_words_fast_text.json'

    data= json.load(open(file))['DATA_multi_choice']
    print('data',data['test'].keys())
    print('data',data['train'].keys())


    ###
    x=[]
    y=[]
    #names=['BLESS','CogALexV','EVALution','ROOT09','KandH_plus_N']
    #['semeval_2012','sre','analogykb','ekar','RS','scan','google','bats','u4','u2','sat']
    data_test=data['test'][dn]
    data_train=data['train'][dn]

    scan_type='science' #science#metaphor


    R=[]
    letters=['a','b','c','d','e','f']

    if dn=='scan':
        letters=['choice'+str(i) for i in range(200)]

    answer_set={}
    all_data=[]
    random.shuffle(data_test)
    if dn in ['semeval_2012_easy','semeval_2012_hard','wikidata_easy','wikidata_hard']:
    	data_test=data_test[:1000]


    for di,d in enumerate(data_test):
 
        eval_d=d['eval_d'] if 'eval_d' in d.keys() else None
        # if dn=='scan' and scan_type!=eval_d:
        #     continue

        stem=d['stem']
        choice=d['choice']
      
        answer=d['answer']
        answer=choice[answer]
        random.shuffle(choice)

        q='QuestionNO: ', di,' : '+ stem['w1'] + ' is to ' + stem['w2'] + ' as: '
        random.shuffle(letters)
    

        item={'questions':q,'options':{},'eval_d':eval_d}
        for ci,c in enumerate(choice):
            label=letters[ci]
            print('ci',ci)
            
            if c['w1']=='empty1':
                continue

            option= c['w1']  + ' is to ' + c['w2']
            item['options'][label]=option
            if c==answer:
                answer_set[di]=(answer,label)
                print('c,label',c,label)
                print('q',q)
                print('#######')
        all_data.append(item)

    file='essential_files/all_data.json'
    h_data={'data':all_data,'answer_set':answer_set}
    with open(file, 'w') as fp:
        json.dump(h_data, fp)
    return all_data,answer_set

def evaluate_chatgpt():
    from openai import OpenAI
    import json
    key=''
    client = OpenAI(api_key=key)
    modelname="gpt-4.1-mini",
    #modelname="gpt-5-nano-2025-08-07",

    def solve_questions(batch):
        prompt = f"""
    You are solving analogy multiple-choice questions.

    Rules:
    - Choose exactly one correct option per question
    - Output ONLY valid JSON
    - questin id as keys, correct choice label as value
    - No explanations

    Questions:
    {json.dumps(batch, ensure_ascii=False)}
    """

        response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=[
                {"role": "system", "content": "You are a logic and analogy expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=1
        )

        #print('response',response)
        print('response.choices[0].message.content',response.choices[0].message.content)

        temp= response.choices[0].message.content
        temp=temp.replace('```json','')
        temp=temp.replace('```','')
        return temp

        return json.loads(response.choices[0].message.content)




    import time
    def chunks(lst, n=10):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    

    def safe_solve(batch, retries=3):
        return solve_questions(batch)
        for _ in range(retries):
            try:
                return solve_questions(batch)
            except json.JSONDecodeError:
                time.sleep(5)
        raise ValueError("Failed JSON output")

    all_answers = {}
    j=0

    lexical_data=['ROOT09','EVALution','CogALexV','BLESS']
    analaogy_data=['EVALution_easy','EVALution_hard','semeval_2012_easy','semeval_2012_hard','wikidata_easy','wikidata_hard']
    for da in ['semeval_2012_hard',]:
        dataname=da
        easy=True if 'easy' in dataname else False
        all_questions,solution_set =Lexical_relations_as_mtcq(da) if da  in lexical_data  else  \
        analogy_questions_as_json(dn=dataname) if da not in ['conceptqa_hard','conceptqa_easy'] else make_conceptqa_questions(easy=easy)

        file='essential_files/gpt/answers'+dataname+'_gpt-5-nano.json'
        file='essential_files/gpt/answers'+dataname+'_40-mini.json'
        for batch in chunks(all_questions, 10):

            result = safe_solve(batch)
            result= json.loads(result)
            print('result',result)
            print(type(result))
            all_answers.update(result)
            print('j',j)
            j=j+1

            h_data={'all_answers':all_answers,'solution_set':solution_set,'all_questions':all_questions,'batch':batch}
            with open(file, 'w') as fp:
                json.dump(h_data, fp)
