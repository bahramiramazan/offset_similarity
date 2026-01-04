



from relbert import RelBERT
# from relbert import cosine_similarity
import json
import torch
###
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os


@torch.no_grad()  
def relbert_test():



	file='essential_files/wordanalogy_'+'DATA_multi_choice.json'


	file='essential_files/wordanalogy_'+'DATA_words_fast_text.json'
	choice_from_no_rel='inv'

	file='essential_files/wordanalogy_'+'DATA_words_fast_text.json'

	data= json.load(open(file))['DATA_multi_choice']

	print('data',data['test'].keys())
	print('data',data['train'].keys())

	print('data['']',data['train'].keys())

	dataname='wikidata_hard'


	data_test_EVALution=data['test'][dataname]
	data_train_EVALution=data['train'][dataname]

	model = RelBERT()
	# device=torch.device('mps')
	# model.model.to(device)



	correct=0
	not_correct=0

	total=0

	for d in data_test_EVALution:
		#print('d',d)
		stem=d['stem']
		w1_stem=stem['w1']
		w2_stem=stem['w2']
		choice=d['choice']
		answer=d['answer']
		w1_w2_emb = model.get_embedding([w1_stem, w2_stem])
		print('##############################################')
		print('w1_stem',w1_stem)
		print('w2_stem',w2_stem)
		pca_wordpairs=[w1_stem+'-'+w2_stem,]

		similarity=[]
		t= np.asarray(w1_w2_emb, dtype=np.float32)
		pca_arr=[t.reshape(1,-1)]
		for c in choice:

			w1_c=c['w1']
			w2_c=c['w2']
			print('****')

			pca_wordpairs.append(c['w1']+'-'+c['w2'])


			print('w1_c',w1_c)
			print('w2_c',w2_c)

			w1_c_w2_c_emb = model.get_embedding([w1_c, w2_c])

			t= np.asarray(w1_c_w2_c_emb, dtype=np.float32)

			print('t',t.shape)

			pca_arr.append(t.reshape(1,-1))
		

			c_sim=cosine_similarity(w1_w2_emb, w1_c_w2_c_emb)

			similarity.append(c_sim)

		max_sim=similarity.index(max(similarity))+1


		if answer+1==max_sim:
			correct=correct+1
		else:
			not_correct=not_correct+1


		print('correct',correct)
		print('not_correct',not_correct)
		print('similarity',similarity)
		print('answer',answer)
		print('max_sim',max_sim)
		print('++++')
		continue






if __name__ == "__main__":
	relbert_test()