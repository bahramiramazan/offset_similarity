
import argparse
import os
from os.path import exists
# from torch.nn.functional import log_softmax, pad
# from torch.optim.lr_scheduler import LambdaLR
# from torch.utils.data import DataLoader


from Train_Eval import *
from Experiments import *
from analogy_data import pre_process
##

def get_args():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-t", "--task", help=" data_collection,proprocess, train , eval")
    argParser.add_argument("-d", "--data", help=" wikidata, nyt")
    argParser.add_argument("-e", "--experiment", default=None,help=" experiment")
    argParser.add_argument("-model", "--model_to_train", default='rc',help=" experiment")
    argParser.add_argument('--tokenizer_name',
        type=str,
        default='bert-base-uncased')
    args = argParser.parse_args()
    print("args=%s" % args)

    print("args.name=%s" % args.task)
    print("args.name=%s" % args.data)
    print("args.name=%s" % args.experiment)
    return args
def pre_process_(data_name):
    pre_process(data_name)



if __name__ == "__main__":
    # python main.py  --task train/eval/preprocess/collect
    args=get_args()
    task=args.task
    data_name=args.data
    experiment=args.experiment
    model_to_train=args.model_to_train
    backend_model_name=args.tokenizer_name
    print('task',task)
    print('eval eq',task=='eval')
    #paper2_lexicalZeroshotTraining
    print('backend_model_name',backend_model_name)
    print('experiment',experiment)


    exps=['conceptqa','semeval_2012','additional_exp','mtcqa','mtcaq_wordanalogy','lexical_offset','sentential_re_paper']
    
    os.environ['CUDA_LAUNCH_BLOCKING']="1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    print('experiment in exps ',experiment in exps )



    if experiment in exps  or experiment=='wordanalogy':

        experiment_run(data_name,experiment,mode=task,model_to_train=model_to_train,backend_model_name=backend_model_name)





    elif str(task) =="train"  :
        print("train")
        georoc_train_eval(data_name,mode='train')
    elif str(task) =="eval":
        print("eval")
        georoc_train_eval(data_name,mode='eval')

    elif str(task)=='preprocess' :
        pre_process_(data_name)
    # elif str(task)=='collect' :
    #     georoc_collect_data() 
    elif str(task)=='abstraction' :
        get_abstract_data() 

    elif str(task)=='gpt' :
        correct_predictions() 
    else:
        print('task could be one of the  train or eval or preprocess or collect')
    
###
##python main.py  --task train  --data retacred  --experiment one --model_to_train wordanalogy_re_model/rc
###

  
