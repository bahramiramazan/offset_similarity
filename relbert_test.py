



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

	###
	#names=['BLESS','CogALexV','EVALution','ROOT09','KandH_plus_N']
	#['semeval_2012','sre','analogykb','ekar','RS','scan','google','bats','u4','u2','sat']

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

		predicted=similarity.index(max(similarity))
		#exit()
		plot_dic={}
		if predicted==answer:
		    correct=correct+1
		else:
		    not_correct=not_correct+1
		#####

		pca = PCA(n_components=2)

		print('pca_arr',pca_arr[0].shape)

		arr = np.concatenate(pca_arr, axis=0)
		print('arr',arr.shape)
		x_2d = pca.fit_transform(arr)

		#x_2d = (x_2d - x_2d.min()) / (x_2d.max() - x_2d.min())

		x_2d = 2.*(x_2d - np.min(x_2d))/np.ptp(x_2d)-1

		print('x_2d',x_2d)
		for  w,v in zip(pca_wordpairs,x_2d):
		    v=v.tolist()
		    w=w[1:-1]

		    plot_dic[w]=[v[0],v[1]]


		total=total+1
		if total>1000: 
		    break
		print('plot_dic',plot_dic)
		import matplotlib.pyplot as plt

		arrows=plot_dic

		colors = ["blue", "gray", "gray", "gray", "gray",'green']
		colors = ["blue", "gray", "gray", "gray", "gray",'green']
		# Plot arrows
		plt.figure(figsize=(6,6))
		origin = (0, 0)

		similarity_ = [float(i)/sum(similarity) for i in similarity]
		similarity_= [1,]+ similarity_
		similarity_=[s if s>0 else 0.1 for s in similarity_]




		similarity_=[int(s*100) for s in similarity_]
		print('similarity_',similarity_)
		j=0
		for (label, vec), color in zip(arrows.items(), colors):
		    # plt.arrow(origin[0], origin[1], vec[0], vec[1],
		    #           head_width=0.02, length_includes_head=True,
		    #           color=color, linewidth=2)
		    vec=vec*4
		    plt.text(vec[0]+0.02, vec[1]+0.02, label, fontsize=12, color=color)
		    if j==(predicted+1) and j!=len(similarity_):
		        color='orange'


		    plt.scatter(vec[0], vec[1], color=color, s=100, zorder=5)
		    j=j+1


		# Axis formatting
		plt.axhline(0, color="black", linewidth=0.05)
		plt.axvline(0, color="black", linewidth=0.05)
		plt.xlim(-1.2, 1.2)
		plt.ylim(-1.2, 1.2)
		plt.gca().set_aspect("equal", adjustable="box")
		plt.title("offset Visualized with PCA")
		#plt.show()

		plt.savefig('images/'+str(choice_from_no_rel)+'/'+pca_wordpairs[0][1:-1]+'.png')







	exit()











	model = RelBERT()

	v_tokyo_japan = model.get_embedding(['Tokyo', 'Japan'])


	v_paris_france, v_music_pizza, v_london_tokyo = model.get_embedding([['Paris', 'France'], ['music', 'pizza'], ['London', 'Tokyo']])

	v_tokyo_japan_v_paris_france=cosine_similarity(v_tokyo_japan, v_paris_france)

	v_tokyo_japan_v_music_pizza=cosine_similarity(v_tokyo_japan, v_music_pizza)

	print('v_tokyo_japan_v_paris_france',v_tokyo_japan_v_paris_france)
	print('v_tokyo_japan_v_music_pizza',v_tokyo_japan_v_music_pizza)





if __name__ == "__main__":
	relbert_test()