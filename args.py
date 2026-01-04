""" # adapted from "https://github.com/minggg/squad/blob/master/args.py"

"""

import argparse


def get_setup_args(data_name):
    """Get arguments needed in setup.py."""
    parser = argparse.ArgumentParser('Georoc')
    parser.add_argument('--tokenizer_name',
            type=str,
            default='bert-base-uncased')

    add_common_args(parser,data_name)
    parser.add_argument('--task',
                        type=str,
                        default='preprocess')

    parser.add_argument('--data',
                        type=str,
                        default='wikidata')



 

   

  
    parser.add_argument('--abstract_limit',
                        type=int,
                        default=400,
                        help='Max number of words in a abstract')

    parser.add_argument('--test_abstract_limit',
                        type=int,
                        default=1000,
                        help='Max number of words in a abstract at test time')




  



    args = parser.parse_args()

    return args




def add_common_args(parser,data_name):




    parser.add_argument('--train_record_file',
                        type=str,
                        default='./data/data_'+str(data_name)+'/train.npz')

    parser.add_argument('--dev_record_file',
                        type=str,
                        default='./data/data_'+str(data_name)+'/dev.npz')

    parser.add_argument('--dev_rev_record_file',
                    type=str,
                    default='./data/data_'+str(data_name)+'/dev_rev.npz')

    parser.add_argument('--test_record_file',
                    type=str,
                    default='./data/data_'+str(data_name)+'/test.npz')

    parser.add_argument('--test_rev_record_file',
                type=str,
                default='./data/data_'+str(data_name)+'/test_rev.npz')
  
  
    parser.add_argument('--train_eval_file',
                        type=str,
                        default='./data/train_eval.json')
    parser.add_argument('--dev_eval_file',
                        type=str,
                        default='./data/dev_eval.json')



