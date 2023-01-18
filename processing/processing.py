import pandas as pd
import nltk
import numpy as np
import numpy.ma as ma
from nltk.tokenize import sent_tokenize
import re
import argparse
import os
from multiprocessing import Pool

# import sys
# sys.path.append('/Users/shikunova/opt/anaconda3/lib/python3.8/site-packages')

import spacy
# import en_core_web_sm

import logging
logging.basicConfig(filename='info.log', level=logging.INFO)

from tqdm import tqdm 
tqdm.pandas()

class MorphTagger():
    
    """
    a class for morhological tagging of English text
    args:
        model -- callable which tags a sequence (spacy/nltk/stanza)
        model_type -- string in ['spacy', 'nltk', 'stanza']
    """
    
    def __init__(self, 
                 model_type,
                 model=None):
        
        self.model = model
        self.model_type = model_type
        
    def load_model(self):
        
        if not self.model:
            self.model = spacy.load('en_core_web_sm')
#             self.model = en_core_web_sm.load()
        
    def tag_seq(self, seq, text_only=True):
        
        if text_only:
            return [token.text for token in self.model(seq)]
        else:
            return [token for token in self.model(seq)]
        
    def predict(self, seq):
        
        return [token[1] for token in tag_seq(seq, self.model_type, self.model)]
    
class LinearRules():
    
    def __init__(self, 
                 tagger, 
                 neg_markers=['not', 'no'],
                 neg_position=2,
                 pst_position=3):
        
        self.tagger = tagger
        self.tagger.load_model()
        self.neg_markers = neg_markers
        self.neg_position = neg_position
        self.pst_position = pst_position
        
    def shift_negation(self, sent, as_list=False):
        
        # get tagged sentence
        tg_sent = tagger.tag_seq(sent, text_only=False)
        
        # get a mask of all negatively polarized tokens
        is_neg = lambda token: str(token.morph) == 'Polarity=Neg'
        neg_mask = np.array([is_neg(token) for token in tg_sent])
        if neg_mask.sum() == 0:
            return None
        
        # remove negatively polarized items
        lemmatize = lambda token: token.text if token.lemma_ not in self.neg_markers else None
        lm_sent = [lemmatize(token) for token in tg_sent if lemmatize(token)]
        
        # calculate position (move if falls onto punctuation or newline)
        position = min(self.pst_position, len(lm_sent)-1)
        if position != len(lm_sent)-1:
            while (tg_sent[position].pos_ == 'PUNCT') or (tg_sent[position].text == '\n') and (position < len(lm_sent)-1):
                position += 1
            # add negation
            lm_sent = lm_sent[:position] + ['not'] + lm_sent[position:]
        
        if as_list:
            return lm_sent
        else:
            return ' '.join(lm_sent)    
        
    def question_reverse(self, sent):
        if len(sent) >= 1:
            if sent[-1] == '?' and re.match(r'\w', sent):
                tr_sent = tagger.tag_seq(sent, text_only=True)[-2::-1] + ['?']
                return ' '.join(tr_sent)
            else:
                return None
        
    
    def shift_past(self, sent, as_list=False):
        
        # get tagged sentence
        tg_sent = tagger.tag_seq(sent, text_only=False)
        if len(tg_sent) == 0:
            return None
        
        # get a mask of past simple tokens
        is_pst_verb = lambda token: str(token.morph) == 'Tense=Past|VerbForm=Fin'
        pst_mask = np.array([is_pst_verb(token) for token in tg_sent])
        if pst_mask.sum() == 0:
            return None
        
        # calculate position
        try:
            position = min(self.pst_position, len(tg_sent)-1)
            pst_positions = np.arange(len(tg_sent))[pst_mask]
            while ((tg_sent[position].pos_ == 'PUNCT') or (tg_sent[position].text == '\n')) \
                and (position < len(tg_sent)-1):
                position += 1
        except:
            logging.info(f'Problematic sentence for shift_past: {sent}')
                    
        # this function checks whether the token on position is past simple 
        def lemmatize(token):
            if str(token.morph) == 'Tense=Past|VerbForm=Fin':
                if token.i == position:
                    return token.text
                else:
                    return token.lemma_
            else:
                if token.i == position:
                    return token.lemma_ + 'ed'
                else:
                    return token.text
            
        # remove past simple morphology and attach to the token on position
        lm_sent = [lemmatize(token) for token in tg_sent]
        
        if as_list:
            return lm_sent
        else:
            return ' '.join(lm_sent)
        
def process_text(text, f):
    
    res = ''
    # tokenize into sents
    sents = sent_tokenize(text)
    # split headers from body text
    sents = sum([sent.split('\n') for sent in sents], [])
    for sent in sents:
        tr_sent = f(sent)
        if tr_sent:
            res += tr_sent + '\n'
            
    # only return contentful results
    if res != []:
        return res
    else: 
        return None
    
def process_file(fn, output_path, rules):
    
    data = pd.read_parquet(fn)
    processed_text = data['text'].progress_apply(lambda x: process_text(x, rules.shift_past) 
                                                 if process_text(x, rules.shift_past) else x)
    processed_text = processed_text.progress_apply(lambda x: process_text(x, rules.shift_negation) 
                                                 if process_text(x, rules.shift_negation) else x)
    processed_text = processed_text.progress_apply(lambda x: process_text(x, rules.question_reverse) 
                                                 if process_text(x, rules.question_reverse) else x)
    pd.DataFrame({
        'processed_text': processed_text,
        'text': data['text']
    }).to_parquet(f'{output_path}/{func.__name__}_{fn.split('/')[-1]}')
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to the wiki dataset", type=str)
    parser.add_argument("output_path", help="path to save output files", type=str)
    parser.add_argument("num_workers", help="number of workers, type 0 for no multiprocessing", type=int, default=0)
#     parser.add_argument("transform", help="transformation to apply", type=str, choices=[
#         'shift_past', 'shift_negation', 'question_reverse'])
    args = parser.parse_args()
    
    logging.info('args read')
    
    tagger = MorphTagger('spacy')
    rules = LinearRules(tagger)
#     funcs = {'shift_past': rules.shift_past, 
#              'shift_negation': rules.shift_negation, 
#              'question_reverse': rules.question_reverse}
    
    files = list(filter(lambda x: 'parquet' in x, os.listdir(args.data_path)))
    
    if args.num_workers == 0:
        for file in files:
            process_file(file, output_path=args.output_path, rules=rules)
            logging.info('file processed')
    else:
        with Pool(args.num_workers) as p:
            p.map(lambda x: process_file(file, output_path=args.output_path, rules=rules), files)
    
if __name__ == "__main__":
    main()