

# OffsetSimilarity: To reprocdue the result for the paper : 

Offset Similarity in Semantically Similar Words







# Dataset
Download Sentential RE DATA, if you may train EqualProbR, otherwise skip:

+ [Wikidata](https://drive.google.com/file/d/1mmKLh6a78GVNizBoCGhs5ZMYJX2g-DIU/view?usp=sharing)


Download only the folowing word analogy question answering datasets(For review we have included these files as tar.gz, only extract it in the home directory, and allow the empty folders to be replaced)

+ [sat, u2,u4,bats,google](https://github.com/asahi417/analogy-language-model) download and place it under the folder : unprocessed_data/analogy_data.
+ [scan](https://huggingface.co/datasets/relbert/analogy_questions) after download/clone  data from [Tamara Czinczoll](https://arxiv.org/abs/2211.15268) and place  test.jsonl, and valid.jsonl  under the foler :unprocessed_data/some_extra/scan/.
+ [distractor verbal analogy](https://osf.io/cd7b9/overview) from [Jones et. al](https://link.springer.com/article/10.3758/s13423-022-02062-8#Sec13) and place it under the foler  unprocessed_data/analogy_data/osfstorage-archive/

+ [google_easy,google_hard](https://huggingface.co/datasets/almogtavor/google-analogy-dataset) was generated from the google analogy data by by [Mikolov et a](https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt)

+ [RS](https://huggingface.co/datasets/relbert/analogy_questions) after download/clone  data from [ushio-etal-2021](https://github.com/asahi417/AnalogyTools) and place  test.jsonl, and valid.jsonl  under the foler :unprocessed_data/some_extra/RS/

+ [ekar](https://huggingface.co/datasets/jiangjiechen/ekar_english) data from [chen-etal-2022](https://aclanthology.org/2022.findings-acl.311/) can be used with hugginface library. 


Download following chekcing points for some of the word embedding models:
+ [Fasttext](https://fasttext.cc/docs/en/english-vectors.html) extract and  place in the folder fasttext
+ [word2vec](https://code.google.com/archive/p/word2vec/) extract and  place it in the loves-word2vec
+ [Glove](https://nlp.stanford.edu/projects/glove/) extract and place in the folder gloves-word2vec
# Setup
```requirements
pip install git+https://github.com/glassroom/heinsen_routing
pip install transformers
pip3 install torch torchvision
pip installl tqdm
```

# Running
### Preprocessing Analogy



> preprocess General: \
- First run the fllowing to make multiple choice analogy questions, easy and hard: 
```
 python preprocess_word_analogy.py

```


> Preprocess Specific: 
- The following data can be proprocess with the commmand below 
- datanames=['conceptqa','wordanalogy','semeval_2012']+ ['BLESS','EVALution','CogALexV','ROOT09']

`Tokenizer_Names=['bert-large-uncased','gpt1','roberta-large','opt','prophetnet','t5-large']`

```
python main.py  --task preprocess  --data *dataName*  --tokenizer_name tokenizerName

```




#### unsupervised vector Offset(without training)


> 1. To  evaluate the word embedding models on AnalogyQA-Easy or Hard as in the Table2, effecient approach is to  run the follwoing commands in order: 


* First preprocess (change roberta-large to word embedding model of your choice) :
```
python main.py  --task preprocess  --data wordanalogy --tokenizer_name roberta-large

```

* Now To evalaute (with roberta-large)run the following command
`If you want to run on specific data only, change test datasets on Experimens.py, Lines`

```
python main.py  --task train  --data wordanalogy  --experiment wordanalogy  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large

```


---


> 2. For fasttext, and also other models (except trained models and gpt4.0) `first set the variable experiment_name='cat1_solve_analogies fasttext and then  run the following: 
Note: change the data and model in lines 169-172 , Additional_Experiments.py , default is fasttext.

```

python main.py  --task train  --data wordanalogy  --experiment additional_exp  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large

```


> 3. To collect response from  ChatGPT : first `set the variable experiment_name=evaluate_gpt, in Experiments.py line 73`  and then  run the following: 

```

python main.py  --task train  --data wordanalogy  --experiment additional_exp  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large

```
- Note1: The chatGPT can evaluated on ConceptQA(easy/hard), analogyQA(Easy/Hard), and common word analogy benchmarks, you just need to select the data you want to evaluate on `The data, and model can be changed in Additional_Experiments.py line 170`

`- Note2: To Evaluate the response, use GPT notebook. Some times minor adjustments may be needed. `




### Trained Models

- Training the models  Mini-RelBERT, EqualProbR, Sameconcept, similarOffset


> 1. Mini-RelBERT

* First preprocess (change roberta-large to word embedding model of your choice) :
```
python main.py  --task preprocess  --data semeval_2012 --tokenizer_name roberta-large

```

* Now to Train 

```
python main.py  --task train  --data semeval_2012  --experiment semeval_2012  --model_to_train  rc  --tokenizer_name roberta-large

```

- To Evaluate ('change evaluation data on Train_Eval.py, line 612, defualt is sat')

```
python main.py  --task eval  --data semeval_2012  --experiment semeval_2012  --model_to_train  rc  --tokenizer_name roberta-large

```



> 2. SimiarOffset

* First preprocess (change roberta-large to word embedding model of your choice) :
```
python main.py  --task preprocess  --data wordanalogy --tokenizer_name roberta-large

```

* Now to Train 

```
python main.py  --task train  --data wordanalogy  --experiment wordanalogy  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large

```

- To Evaluate ('change evaluation data on Train_Eval.py, line 612, defualt is sat')

```
python main.py  --task eval  --data wordanalogy  --experiment wordanalogy  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large

```




> 3. EqualProbR and Lexical Relation Classification ( Table5, Tabel6,)
- For EqualProbR, we train on  lexical relation classification data set(EVALution, and on related Entities from wikidata) 

`data=['BLESS','EVALution','CogALexV','ROOT09', 'wikidata']`

* First preprocess (change roberta-large to word embedding model of your choice) :
```
python main.py  --task preprocess  --data dataname --tokenizer_name roberta-large

```

* Now to Train and evaluate on test set 

```
python main.py  --task train  --data dataname  --experiment lexical_offset  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large

```

- To Evaluate EqualPRob R on word analogy run the follwoing, make sure to set (` Table=EqualProbR`)

```
python main.py  --task eval  --data dataname  --experiment wordanalogy  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large

```

`- Note: all other word anlogy datasets are evauated on EqualProbR model trained on wikdiata, while EVALutionEasy/Hard on model trained on EVALution `

### MTCQA model
`Set the specific experiment related to mtcqa in Multchoice_Model.py line 51 , and then run the following `
```
python main.py  --task train  --data mtcqa  --experiment mtcqa  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large
```


### Plots and Additional Experiments From Appendix: 


> 1. Pretraing with analogyQA-Easy vs Hard 


> 1. Bayesain Analysis of word semantics and word relations 

> 3. Various Permutation of word Analogy 



> 4. Ineragreement between models

