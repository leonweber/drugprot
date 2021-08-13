#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(module)s: %(message)s', level = 'INFO')
logger = logging.getLogger(__name__)

import pdb

import copy
import bioc
import argparse
import os
import pickle
import numpy as np
from src.biosyn import (
    DictionaryDataset,
    BioSyn,
    TextPreprocess
)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Create dataset with BioSyn normalization')
    parser.add_argument('--dict-cache', help = 'Path to cached dictionary: create if not existing')
    parser.add_argument('--entity', choices = ('gene', 'chemical'), help='entity type')
    parser.add_argument('--file', required=True, help='File to which apply normalization')
    parser.add_argument('--model-dir', required=True, help='Directory for model')
    parser.add_argument('--dictionary-path', type=str, default=None, help='dictionary path')
    parser.add_argument('--use-cuda',  action="store_true")
    
    args = parser.parse_args()
    return args
    
def cache_or_load_dictionary(biosyn, dictionary_path, dict_cache):
    
    cached_dictionary_path = os.path.join(dict_cache, 'dictionary.npy')
    cached_dict_sparse_embeds_path = os.path.join(dict_cache, 'dict_sparse_embeds.npy')
    cached_dict_dense_embeds_path = os.path.join(dict_cache, 'dict_dense_embeds.npy')
    
    # If exist, load the cached dictionary
    if os.path.exists(dict_cache):
        
        dictionary = np.load(cached_dictionary_path)
        dict_sparse_embeds = np.load(cached_dict_sparse_embeds_path)
        dict_dense_embeds = np.load(cached_dict_dense_embeds_path)
        
        print("Loaded cached dictionary from {}".format(dict_cache))

    else:
        os.makedirs(dict_cache)
        
        dictionary = DictionaryDataset(dictionary_path = dictionary_path).data
        np.save(cached_dictionary_path, dictionary)
        
        dictionary_names = dictionary[:,0]
        
        dict_sparse_embeds = biosyn.embed_sparse(names=dictionary_names, show_progress=True)
        np.save(cached_dict_sparse_embeds_path, dict_sparse_embeds)
        
        dict_dense_embeds = biosyn.embed_dense(names=dictionary_names, show_progress=True)
        np.save(cached_dict_dense_embeds_path, dict_dense_embeds)
        
    
        print("Saved dictionary into cached file {}".format(dict_cache))

    return dictionary, dict_sparse_embeds, dict_dense_embeds

def main(args):
    # load biosyn model
    biosyn = BioSyn().load_model(
                    path=args.model_dir,
                    max_length=25,
                    use_cuda=args.use_cuda)
   
    
    
    # cache or load dictionary
    dictionary, dict_sparse_embeds, dict_dense_embeds = cache_or_load_dictionary(biosyn, args.dictionary_path, args.dict_cache)

    collection = bioc.load(args.file)
    
    tpp = TextPreprocess()
    
    for doc in collection.documents:
        for p in doc.passages:
            for sent in p.sentences:
                annotations = copy.deepcopy(sent.annotations)
                mentions = [a.text if a.infons.get('type').replace('-Y','').replace('-N','').lower() == args.entity else None for a in annotations]
                to_be_linked_idxs = [idx for idx,m in enumerate(mentions) if m is not None]
                mentions = [tpp.run(m) for m in mentions if m is not None]
                
                if len(mentions) == 0:
                    continue
                
                mentions_sparse_embeds = biosyn.embed_sparse(names=np.asarray(mentions))
                mentions_dense_embeds = biosyn.embed_dense(names=np.asarray(mentions))
                
                # calcuate score matrix and get top 5
                sparse_score_matrix = biosyn.get_score_matrix(
                    query_embeds=mentions_sparse_embeds,
                    dict_embeds=dict_sparse_embeds
                )
                dense_score_matrix = biosyn.get_score_matrix(
                    query_embeds=mentions_dense_embeds,
                    dict_embeds=dict_dense_embeds
                )
                sparse_weight = biosyn.get_sparse_weight().item()
                hybrid_score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
                hybrid_candidate_idxs = biosyn.retrieve_candidate(
                    score_matrix = hybrid_score_matrix, 
                    topk = 5
                )
        
                # # get predictions from dictionary
                predictions = dictionary[hybrid_candidate_idxs]
                
                for i,idx in enumerate(to_be_linked_idxs):
                    mention_prediction = predictions[i]
                    top_mention_prediction = mention_prediction[0]
                    top_mention_prediction_id = top_mention_prediction[1]
                    sent.annotations[idx].infons['identifier'] = top_mention_prediction_id
                    
    with open(f'{args.file}.{args.entity}', 'w') as fp:
        collection = bioc.dump(collection, fp)
                

if __name__ == '__main__':
    args = parse_args()
    main(args)
