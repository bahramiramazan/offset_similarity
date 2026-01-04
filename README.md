

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
First run the fllowing to make multiple choice analogy questions, easy and hard: 
```
 python preprocess_word_analogy.py

```


> Preprocess Specific: 
The following data can be proprocess with the commmand below 
datanames=['conceptqa','wordanalogy','semeval_2012']+ ['BLESS','EVALution','CogALexV','ROOT09']

Tokenizer_Names=['bert-large-uncased','gpt1','roberta-large','opt','prophetnet','t5-large']

```
python main.py  --task preprocess  --data *dataName*  --tokenizer_name tokenizerName

```



### Table2 

#### unsupervised vector Offset(without training)
1. To  evaluate the word embedding models on AnalogyQA-Easy or Hard as in the Table2, effecient approach is to  run the follwoing commands in order: 


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


>3. To evlauate chagtp : first `set the variable experiment_name=evaluate_gpt, in Experiments.py line 73`  and then  run the following: 

```

python main.py  --task train  --data wordanalogy  --experiment additional_exp  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large

```
`The data, and model can be changes in Additional_Experiments.py line 170`

> 3. To train EquaplProbR run the following : 

`First pre-process the data`
```
python main.py  --task preprocess  --data wikidata  --tokenizer_name roberta-large

```

`For training set the variable Table='table2' (line 245, Experiments.py),  and run the following:`
```
python main.py  --task train  --data wordanalogy  --experiment wordanalogy  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large

```

To evaluate EqualProbR on word analogyqa-easy/hard, `change the variable Table='table2_EquaProbR'`, and then run the follwoing command: 

```
python main.py  --task train  --data wordanalogy  --experiment wordanalogy  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large

```


To evaluate on word analogy,`change the variable Table='table3_EquaProbR'`, and then run the follwoing command: 

```
python main.py  --task train  --data wordanalogy  --experiment wordanalogy  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large

