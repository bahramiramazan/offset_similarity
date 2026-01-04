
from transformers import AutoTokenizer
from transformers import DataCollatorForMultipleChoice
import numpy as np
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
import json 
###
from datasets import Dataset
import evaluate
from transformers import RobertaTokenizer
import json 


import json
from pathlib import Path
##
import subprocess
import sys


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

from Additional_Experiments import *


###

def pip_install(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Example usage:


def run_mc(experiment):

	#	file='wordanalogy_'+'DATA_multi_choice'
	#pip3 install datasets
	#pip_install("datasets")

	mtqa_experiment_names=['overfit','pretraining_with_easy_hard','overfit']

	fine_tune=False
	if fine_tune:

		model_=AutoModelForMultipleChoice.from_pretrained("mc_model/checkpoint-1")
		tokenizer = AutoTokenizer.from_pretrained("./tokenizer_mc/")
	else:
		#model=AutoModelForMultipleChoice.from_pretrained("FacebookAI/roberta-large")
		model_=AutoModelForMultipleChoice.from_pretrained("pretrained/AutoModelForMultipleChoice")
		#model_=AutoModelForMultipleChoice.from_pretrained("FacebookAI/roberta-large")
		#model_.save_pretrained("pretrained/AutoModelForMultipleChoice", from_pt=True) 
		tokenizer = AutoTokenizer.from_pretrained("./tokenizer_mc/")


	for  experiment in mtqa_experiment_names:
		train_MTCQA(experiment,model_,tokenizer)




def train_MTCQA(experiment,model_,tokenizer,):
	accuracy = evaluate.load("accuracy")
	#tokenizer = AutoTokenizer.from_pretrained("./tokenizer_mc/")
	collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
	# model=AutoModelForMultipleChoice.from_pretrained("FacebookAI/roberta-large")
	# model.save_pretrained("pretrained/AutoModelForMultipleChoice", from_pt=True) 
	# tokenizer = AutoTokenizer.from_pretrained("./tokenizer_mc/")
	



	ending_names = ["c0", "c1", "c2", "c3","c4"]




	def preprocess_function(examples):



		first_sentences = [[context] * 5 for context in examples['question']]
		question_headers = [' as']*len(first_sentences)

		second_sentences = [
		    [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
		]

		first_sentences = sum(first_sentences, [])
		second_sentences = sum(second_sentences, [])

		tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
		return {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}


	def make_conceptqa_questions(test_or_dev,easy=True):
	    benchmarks='wikidata'
	    train_eval='train'
	    file='essential_files/SRE_Analogy.json'
	    SRE_Analogy= json.load(open(file))['SRE_Analogy']['wikidata']
	    print('SRE_Analogy',SRE_Analogy.keys())
	    data=SRE_Analogy[test_or_dev]
	    print(len(data))
	    #abstract_spaces_dic=get_abstract_spaces_dic_wikidata()
	    
	    abstract_spaces_dic=get_google_abstract_space_dic(semeval_or_google='wikidata')
	    Questions_train=get_conceptqa_questions(abstract_spaces_dic,n_size=2,easy=easy)
	    return Questions_train

	def make_concepqaqa_multichoice(data,easy=True):
		DATA=[]
		for d in data:
			#print(d.keys())
			q=d['q'] #if 'q' in d.keys() else d['question']
			answer=d['answer']
			choices=d['choices']
			key=d['key']
			choices=choices[:4] if easy else choices[1:5]
			choices.append(answer)
			random.shuffle(choices)
			for ci ,c in enumerate(choices):
			

				if c==answer:
					label=ci

			item={'question':q,'label':label}

			for ci,c in enumerate(choices):
				key='c'+str(ci)
				item[key]=c
			DATA.append(item)
		return DATA
			
			




	if 'conceptqa' in experiment:

		conceptqa_train=make_conceptqa_questions('train')
		conceptqa_test=make_conceptqa_questions('test')
		conceptqa_train_easy=make_concepqaqa_multichoice(conceptqa_train,easy=True)
		conceptqa_test_easy=make_concepqaqa_multichoice(conceptqa_test,easy=True)

		conceptqa_train_easy = Dataset.from_list(conceptqa_train_easy)
		conceptqa_test_easy = Dataset.from_list(conceptqa_test_easy)
		print('conceptqa easy')
		tokenized_test_conceptqa = conceptqa_test_easy.map(preprocess_function, batched=True)
		tokenized_train_conceptqa= conceptqa_train_easy.map(preprocess_function, batched=True)

		tests_data=[]
		test=tokenized_test_conceptqa
		train=tokenized_train_conceptqa

		model=copy.deepcopy(model_)
		train_eval(model,collator,tokenizer,train,test,tests_data)
		##
		print('conceptqa hard')
		conceptqa_train_easy=make_concepqaqa_multichoice(conceptqa_train,easy=False)
		conceptqa_test_easy=make_concepqaqa_multichoice(conceptqa_test,easy=False)
		
		conceptqa_train_easy = Dataset.from_list(conceptqa_train_easy)
		conceptqa_test_easy = Dataset.from_list(conceptqa_test_easy)


		tokenized_test_conceptqa = conceptqa_train_easy.map(preprocess_function, batched=True)
		tokenized_train_conceptqa= conceptqa_train_easy.map(preprocess_function, batched=True)

		tests_data=[]
		test=tokenized_test_conceptqa
		train=tokenized_train_conceptqa

		model=copy.deepcopy(model_)
		train_eval(model,collator,tokenizer,train,test,tests_data)


	
	else :
	



		file='essential_files/wordanalogy_'+'DATA_multi_choice.json'
		file='essential_files/wordanalogy_'+'DATA_multi_choice.json'

		data= json.load(open(file))['DATA_multi_choice']
		sre_temp='analogykb'
		#sre_temp='analogykb'
		#sre_temp='sat'
		semeval_2012='semeval_2012_easy'
		data_test_semeval_2012_easy=data['test'][semeval_2012]
		data_train_semeval_2012_easy=data['train'][semeval_2012] #+data['train']['analogykb']

		semeval_2012='semeval_2012_hard'
		data_test_semeval_2012_hard=data['test'][semeval_2012]
		data_train_semeval_2012_hard=data['train'][semeval_2012] #+data['train']['analogykb']


		analogykb='analogykb_easy'
		data_test_analogykb=data['test'][analogykb]
		data_train_analogykb=data['train'][analogykb] #+data['train']['analogykb']

		wikidata='wikidata_easy'
		data_test_wikidata_easy=data['test'][wikidata]
		data_train_wikidata_easy=data['train'][wikidata] #+data['train']['analogykb']

		wikidata='wikidata_hard'
		data_test_wikidata_hard=data['test'][wikidata]
		data_train_wikidata_hard=data['train'][wikidata] #+data['train']['analogykb']








		data_test_sat=data['test']['sat']
		data_train_sat=data['train']['sat']


		data_test_u2=data['test']['u2']
		data_train_u2=data['train']['u2']

		data_test_u4=data['test']['u4']
		data_train_u4=data['train']['u4']


		data_test_ekar=data['test']['ekar']
		data_train_ekar=data['train']['ekar']

		data_test_google=data['test']['google']
		data_train_google=data['train']['google']

		data_test_bats=data['test']['bats']
		data_train_bats=data['train']['bats']










		

		#print('data',data["train"][10])
		data_test_sat = Dataset.from_list(data_test_sat)
		data_train_sat = Dataset.from_list(data_train_sat)

		data_test_ekar = Dataset.from_list(data_test_ekar)
		data_train_ekar = Dataset.from_list(data_train_ekar)

		data_test_u2 = Dataset.from_list(data_test_u2)
		data_train_u2 = Dataset.from_list(data_train_u2)
		##

		data_test_u4 = Dataset.from_list(data_test_u4)
		data_train_u4 = Dataset.from_list(data_train_u4)

		data_test_google = Dataset.from_list(data_test_google)
		data_train_google = Dataset.from_list(data_train_google)


		data_test_bats = Dataset.from_list(data_test_bats)
		data_train_bats = Dataset.from_list(data_train_bats)

		data_test_wikidata_easy = Dataset.from_list(data_test_wikidata_easy)
		data_train_wikidata_easy = Dataset.from_list(data_train_wikidata_easy)

		data_test_wikidata_hard = Dataset.from_list(data_test_wikidata_hard)
		data_train_wikidata_hard = Dataset.from_list(data_train_wikidata_hard)

		data_test_semeval_2012_hard = Dataset.from_list(data_test_semeval_2012_hard)
		data_train_semeval_2012_hard = Dataset.from_list(data_train_semeval_2012_hard)

		data_test_semeval_2012_easy = Dataset.from_list(data_test_semeval_2012_easy)
		data_train_semeval_2012_easy = Dataset.from_list(data_train_semeval_2012_easy)








		tokenized_test_sat = data_test_sat.map(preprocess_function, batched=True)
		tokenized_train_sat = data_train_sat.map(preprocess_function, batched=True)


		tokenized_test_ekar = data_test_ekar.map(preprocess_function, batched=True)
		tokenized_train_ekar = data_train_ekar.map(preprocess_function, batched=True)


		tokenized_test_u2 = data_test_u2.map(preprocess_function, batched=True)
		tokenized_train_u2 = data_train_u2.map(preprocess_function, batched=True)



		tokenized_test_u4 = data_test_u4.map(preprocess_function, batched=True)
		tokenized_train_u4 = data_train_u4.map(preprocess_function, batched=True)


		tokenized_test_google = data_test_google.map(preprocess_function, batched=True)
		tokenized_train_google = data_train_google.map(preprocess_function, batched=True)



		tokenized_test_bats = data_test_bats.map(preprocess_function, batched=True)
		tokenized_train_bats = data_train_bats.map(preprocess_function, batched=True)

		#####################################
		tokenized_test_wikidata_easy = data_test_wikidata_easy.map(preprocess_function, batched=True)
		tokenized_train_wikidata_easy= data_train_wikidata_easy.map(preprocess_function, batched=True)


		tokenized_test_wikidata_hard = data_test_wikidata_hard.map(preprocess_function, batched=True)
		tokenized_train_wikidata_hard= data_train_wikidata_hard.map(preprocess_function, batched=True)

		####
		tokenized_test_semeval_2012_easy = data_test_semeval_2012_easy.map(preprocess_function, batched=True)
		tokenized_train_semeval_2012_easy= data_train_semeval_2012_easy.map(preprocess_function, batched=True)


		tokenized_test_semeval_2012_hard = data_test_semeval_2012_hard.map(preprocess_function, batched=True)
		tokenized_train_semeval_2012_hard= data_train_semeval_2012_hard.map(preprocess_function, batched=True)



		if 'overfit' in experiment:

	
			names=['sat','ekar','u2','u4','google']

			for i,t in enumerate([(tokenized_train_sat,tokenized_test_sat),(tokenized_train_ekar,tokenized_test_ekar),(tokenized_train_u2,tokenized_test_u2),\
						(tokenized_train_u4,tokenized_test_u4),(tokenized_train_google,tokenized_test_google)]):
						train,test=t[0],t[1]
						print('data',names[i])
						if names[i]!='u4':
							tests_data=[]
							train=test

							model=copy.deepcopy(model_)
							


							train_eval(model,collator,tokenizer,train,test,tests_data,overfit=True)
		elif 'easy_hard' in experiment:
			print('pretraining with wikidata_easy')
			train,test=tokenized_train_wikidata_easy,tokenized_test_wikidata_easy
			tests_data=[tokenized_test_sat,tokenized_test_ekar,tokenized_test_u2,tokenized_test_u4,tokenized_test_google,tokenized_test_bats]

			model=copy.deepcopy(model_)
			train_eval(model,collator,tokenizer,train,test,tests_data)


			print('pretraining with wikidata_hard')

			train,test=tokenized_train_wikidata_hard,tokenized_test_wikidata_hard
			tests_data=[tokenized_test_sat,tokenized_test_ekar,tokenized_test_u2,tokenized_test_u4,tokenized_test_google,tokenized_test_bats]

			model=copy.deepcopy(model_)
			train_eval(model,collator,tokenizer,train,test,tests_data)


			print('pretraining with semeval_2012_easy')
		

			train,test=tokenized_train_semeval_2012_easy,tokenized_test_semeval_2012_easy
			tests_data=[tokenized_test_sat,tokenized_test_ekar,tokenized_test_u2,tokenized_test_u4,tokenized_test_google,tokenized_test_bats]

			model=copy.deepcopy(model_)
			train_eval(model,collator,tokenizer,train,test,tests_data)


			print('pretraining with semeval_2012_hrad')
			train,test=tokenized_train_semeval_2012_hard,tokenized_test_semeval_2012_hard
			tests_data=[tokenized_test_sat,tokenized_test_ekar,tokenized_test_u2,tokenized_test_u4,tokenized_test_google,tokenized_test_bats]

			model=copy.deepcopy(model_)
			train_eval(model,collator,tokenizer,train,test,tests_data)






def train_eval(model,collator,tokenizer,data_train,data_eval,tests_data,overfit=False):
	

	def compute_metrics(eval_pred):
		accuracy = evaluate.load("accuracy")
		predictions, labels = eval_pred
		predictions = np.argmax(predictions, axis=1)
		return accuracy.compute(predictions=predictions, references=labels)







	save_strategy='epoch' #if fine_tune==False else 'no'

	load_best_model_at_end=True #if fine_tune==False else False
	eval_strategy='epoch' #if fine_tune==False else 'no'

	epoch=3 if overfit==False else 10



	training_args = TrainingArguments(
	    output_dir="mc_model",
	    eval_strategy=eval_strategy,
	    save_strategy=save_strategy,
	    save_total_limit=1,
	    load_best_model_at_end=load_best_model_at_end,
	    learning_rate=5e-5,
	    per_device_train_batch_size=4,
	    per_device_eval_batch_size=4,
	    num_train_epochs=epoch,
	    weight_decay=0.01,
	    push_to_hub=False,
	)

	trainer = Trainer(
	    model=model,
	    args=training_args,
	    train_dataset=data_train,
	    eval_dataset=data_eval,
	    processing_class=tokenizer,
	    data_collator=collator,
	    compute_metrics=compute_metrics,
	)

	trainer.train()

	print('##############################################################')
	print('###############   Evaluation ###########################')
	print('##############################################################')
	for t in tests_data:

		print('sat')
		results = trainer.evaluate(eval_dataset=t)
		print('res')
		print(results)



