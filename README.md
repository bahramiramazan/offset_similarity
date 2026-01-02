

# OffsetSimilarity

Offset Similarity in Semantically Similar Words







# Dataset
Download Sentential RE DATA, if you may train EqualProbR

+ [Wikidata](https://drive.google.com/file/d/1mmKLh6a78GVNizBoCGhs5ZMYJX2g-DIU/view?usp=sharing)
+ [sat, u2,u4,bats,google](https://github.com/asahi417/analogy-language-model) download and place it under the folder : unprocessed_data/analogy_data.
+ [scan](https://huggingface.co/datasets/relbert/analogy_questions) download the data from [Tamara Czinczoll](https://arxiv.org/abs/2211.15268) and place under the foler :unprocessed_data/some_extra/scan/.
+ [distractor verbal analogy](https://osf.io/cd7b9/overview) from [Jones et. al](https://link.springer.com/article/10.3758/s13423-022-02062-8#Sec13) and place it under the foler  unprocessed_data/analogy_data/osfstorage-archive/

+ [google_easy,google_hard](hhttps://huggingface.co/datasets/almogtavor/google-analogy-dataset) was generated from the processed google analogy data by Daniel Korchinski.


Tacred Extensions: 
Download following chekcing points for some of the word embedding modesl:
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
### Preprocessing

The parameters are in the config file. 
Download the datasets, embeddings etc. and place in the expected locations.

> preprocess Wordanalogy  data: \
First run the fllowing to make multiple choice analogy questions, easy and hard: 
```
 python preprocess_word_analogy.py

```
To reproduce the result from Table 2, there are two possible options. 
> 1. First change the Experiments.py, line x to cat1_solve_analogies, and run the following code. Keep in mind that the model, and data can be changed in Additional_Experiments.py file, line 181. Deafault is Fasttext, with word analogy dataset SAT. 

```

python main.py  --task train  --data wordanalogy  --experiment additional_exp  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large

```


> 2. The above approach is quick but slow. To run it fastly, first preprocess the data with the tokenizer from concerned word embedding models with the following commmand: 

tokenizer_name  should be from ['oberta-large','gpt2','bert-uncased-large','prophetnet','opt', 't5-large']

```

python main.py  --task preprocess  --data wordanalogy   --tokenizer_name roberta-large

```
and then run with the follwing command 


```

python main.py  --task train  --data wordanalogy  --experiment wordanalogy  --model_to_train  wordanalogy_re_model  --tokenizer_name roberta-large

```
