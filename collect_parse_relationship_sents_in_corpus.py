"""
Test and parse relationship sentences
in reference corpus, then compute frequency
statistics for different gender combinations.
"""
import os
from argparse import ArgumentParser
import bz2
import re

import numpy as np
import pandas as pd
import spacy
from nltk.corpus import wordnet
from tqdm import tqdm
tqdm.pandas()
from data_helpers import load_relationship_occupation_template_data
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
PUNCT_MATCHERS = [
    [re.compile('&apos;'), "'"],
]
def clean_web_text(text):
    clean_text = text.strip()
    for m, w in PUNCT_MATCHERS:
        clean_text = m.sub(w, clean_text)
    # remove HTML junk
    text_soup = BeautifulSoup(clean_text)
    clean_text = text_soup.text
    return clean_text

POSSESSOR_WORD_MATCHER_LOOKUP = {
    'es' : re.compile('|'.join(['del', 'de'])),
    'fr' : re.compile('|'.join(['de', 'du'])),
    'it' : re.compile('|'.join(['de', 'di', 'del', 'da', 'della'])),
}

def is_token_connected_to_possessor(token, possessor_word_matcher):
    token_children = list(token.children)
#     print(f'children = {token_children}')
    possessor_children = list(filter(lambda x: possessor_word_matcher.match(x.text.lower()) is not None, token_children))
    return len(possessor_children)
def find_phrase_possessor(sent, phrase_word, lang):
    ## TODO: make it even more strict => "the X of Y" where "Y" is parent of "of"
    # get possessor of phrase via "nmod"
#     print(f'sent = {[x for x in sent]}; phrase_word = {phrase_word}')
    phrase_token_matches = list(filter(lambda x: x.text.lower() == phrase_word, sent))
    possessor = None
    if(len(phrase_token_matches) > 0):
        phrase_token = phrase_token_matches[0]
        # look for source noun with NMOD dep and possessor child
        possessor_word_matcher = POSSESSOR_WORD_MATCHER_LOOKUP[lang]
        phrase_children = list(filter(lambda x: x.dep_=='nmod' and is_token_connected_to_possessor(x, possessor_word_matcher), phrase_token.children))
        if(len(phrase_children) > 0):
            possessor = phrase_children[0]
    return possessor

PERSON_CATEGORY_MATCHER = re.compile('person.n.01')
LEMMA_NUM_MATCHER = re.compile('(?<=\.n\.)(\d+)(?=\.)')
WORDNET_LANG_LOOKUP = {
    'es' : 'spa',
    'fr' : 'fra',
    'it' : 'ita',
}
def is_word_a_person(word, lang):
    wordnet_lang = WORDNET_LANG_LOOKUP[lang]
    word_is_person = False
    # assume capital letter => name => person
    if(word.istitle()):
        word_is_person = True
    else:
    #     print(f'word type={type(word)}')
        word_lemmas = wordnet.lemmas(word, lang=wordnet_lang)
#         print(f'word={word}; lemmas={word_lemmas}')
        # find main word sense
        # sort lemmas by number: lower number => more "core" meaning
        word_lemma_nums = list(map(lambda x: int(LEMMA_NUM_MATCHER.search(str(x)).group(0)) if LEMMA_NUM_MATCHER.search(str(x)) is not None else np.inf, word_lemmas))
        if(len(word_lemma_nums) > 0):
            max_word_lemma_num = min(word_lemma_nums)
            main_lemmas = [x for x,y in zip(word_lemmas, word_lemma_nums) if y==max_word_lemma_num]
            # best case: match lemma name w/ weird format e.g. "donna.n.8.donna"
            word_lemma_matcher = re.compile(f'Lemma\(\'({word})\.n.+')
            if(len(main_lemmas) > 1):
                word_match_main_lemma = list(filter(lambda x: word_lemma_matcher.match(str(x)), main_lemmas))
                if(len(word_match_main_lemma) > 0):
                    main_lemma = word_match_main_lemma[0]
                else:
                    main_lemma = main_lemmas[0]
                # get hypernyms for main lemma
                main_lemma_hypernym_paths = main_lemma.synset().hypernym_paths()
                main_lemma_main_path = main_lemma_hypernym_paths[0]
                path_contains_person_category = any(map(lambda x: PERSON_CATEGORY_MATCHER.match(x.name()), main_lemma_main_path))
                if(path_contains_person_category):
                    word_is_person = True
    return word_is_person

def main():
    parser = ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('lang')
    args = vars(parser.parse_args())
    data_file = args['data_file']
    lang = args['lang']

    ## get sentences
    occupation_words, relationship_word_data, relationship_sents, langs, lang_art_PRON_lookup, lang_POSS_PRON_lookup = load_relationship_occupation_template_data()
    out_dir = os.path.dirname(data_file)
    relationship_sent_data_file = os.path.join(out_dir, f'lang={lang}_relationship_words_sent_data.gz')
    if(not os.path.exists(relationship_sent_data_file)):
        subject_genders = ['male', 'female']
        relationship_sents = []
        relationship_phrases = []
        for gender_i in subject_genders:
            # possessive pronoun + noun e.g. "sua moglie"
            # #         pron_i = lang_POSS_PRON_lookup[lang][gender_i]
            # #         # remove "il"/"la" for IT pron
            # #         if(lang == 'it'):
            # #             pron_i = pron_i.split(' ')[-1]
            #         relationship_phrases_i = relationship_word_data.loc[:, f'{lang}_{gender_i}'].apply(lambda x: f'{pron_i} {x}')
            # normal relationship words e.g. "la moglie" => looking for possessors "la moglie del generale"
            relationship_phrases_i = relationship_word_data.loc[:, f'{lang}_{gender_i}']
            relationship_phrases.extend(relationship_phrases_i.values.tolist())
        relationship_phrase_matcher = re.compile('|'.join(relationship_phrases))
        matching_sents = []
        for l in tqdm(bz2.open(data_file, 'rt')):
            l = clean_web_text(l)
            l_sents = sent_tokenize(l)
            for sent_i in l_sents:
                relationship_phrase_search_l = relationship_phrase_matcher.search(sent_i.lower())
                if (relationship_phrase_search_l is not None):
                    matching_sents.append([relationship_phrase_search_l.group(0), sent_i])
        matching_sents = pd.DataFrame(matching_sents, columns=['relationship_word', 'sent'])
        matching_sents = matching_sents.assign(**{'lang': lang})
        relationship_sents.append(matching_sents)
        relationship_sents = pd.concat(relationship_sents, axis=0)
        ## save to file
        relationship_sents.to_csv(relationship_sent_data_file, sep='\t', compression='gzip', index=False)  # single word
    else:
        relationship_sents = pd.read_csv(relationship_sent_data_file, sep='\t')
    # tmp debugging
    # relationship_sents = relationship_sents.head(1000)

    ## get possessors
    nlp_pipeline = load_spacy_model(lang)

    relationship_sents = relationship_sents.assign(**{
        'sent_parse' : relationship_sents.loc[:, 'sent'].progress_apply(nlp_pipeline)
    })
    relationship_sents = relationship_sents.assign(**{
        'relationship_word_source' : relationship_sents.progress_apply(lambda x: find_phrase_possessor(x.loc['sent_parse'], x.loc['relationship_word'], lang), axis=1)
    })
    # get word source gender from morphology
    relationship_sents = relationship_sents.assign(**{
        'relationship_word_source_gender': relationship_sents.loc[:, 'relationship_word_source'].apply(
            lambda x: x.morph.get('Gender')[0] if x is not None and len(x.morph.get('Gender')) > 0 else None)
    })
    relationship_sents = relationship_sents[relationship_sents.loc[:, 'relationship_word_source_gender'].apply(lambda x: x is not None)]
    relationship_sents_with_source = relationship_sents[relationship_sents.loc[:, 'relationship_word_source'].apply(lambda x: x is not None)]
    # fix gender labels
    gender_lookup = {
        'Masc': 'male',
        'Fem': 'female',
    }
    relationship_sents_with_source = relationship_sents_with_source.assign(**{'relationship_word_source_gender': relationship_sents_with_source.loc[:,'relationship_word_source_gender'].apply(gender_lookup.get)})
    ## add target gender
    langs = [lang]
    genders = ['male', 'female']
    relationship_word_gender_lookup = {
        l: {g: relationship_word_data.loc[:, f'{l}_{g}'].values for g in genders}
        for l in langs
    }
    relationship_word_gender_lookup = {
        k: {v: k1 for k1, v1 in v.items() for v in v1}
        for k, v in relationship_word_gender_lookup.items()
    }
    ## look up gender, compare w/ head noun gender, etc
    relationship_sents_with_source = relationship_sents_with_source.assign(**{
        'relationship_word_gender': relationship_sents_with_source.apply(lambda x: relationship_word_gender_lookup[x.loc['lang']][x.loc['relationship_word'].split(' ')[-1]], axis=1)
    })
    
    ## filter to "person" nouns via wordnet
    relationship_sents_with_source = relationship_sents_with_source.assign(**{
        'relationship_word_source_is_person': relationship_sents_with_source.apply(lambda x: is_word_a_person(x.loc['relationship_word_source'].text, lang=x.loc['lang']), axis=1)
    })
    relationship_sents_with_person_source = relationship_sents_with_source[relationship_sents_with_source.loc[:, 'relationship_word_source_is_person']]
    ## save to file for analysis!!
    relationship_sents_with_source_file = os.path.join(out_dir, f'lang={lang}_relationship_words_with_source_sent_data.gz')
    relationship_sents_with_person_source.to_csv(relationship_sents_with_source_file, sep='\t', compression='gzip', index=False)

def load_spacy_model(lang):
    lang_model_lookup = {
        'es': 'es_dep_news_trf',
        'fr': 'fr_dep_news_trf',
        'it': 'it_core_news_lg',
    }
    lang_model_name = lang_model_lookup[lang]
    nlp_pipeline = spacy.load(lang_model_name)
    return nlp_pipeline

if __name__ == '__main__':
    main()
