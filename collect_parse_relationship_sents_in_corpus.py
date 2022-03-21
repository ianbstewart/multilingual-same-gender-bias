"""
Test and parse relationship sentences
in reference corpus, then compute frequency
statistics for different gender combinations.
"""
import os
from argparse import ArgumentParser
import bz2
import re
import pandas as pd
from tqdm import tqdm
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

def main():
    parser = ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('lang')
    args = vars(parser.parse_args())
    data_file = args['data_file']
    lang = args['lang']

    occupation_words, relationship_word_data, relationship_sents, langs, lang_art_PRON_lookup, lang_POSS_PRON_lookup = load_relationship_occupation_template_data()
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
    out_dir = os.path.basename(data_file)
    relationship_sents.to_csv(os.path.join(out_dir, f'lang={lang}_relationship_words_sent_data.gz'), sep='\t', compression='gzip', index=False)  # single word

if __name__ == '__main__':
    main()