import pandas as pd
from itertools import combinations, product
from unidecode import unidecode
import re
import numpy as np
from ast import literal_eval

VOWEL_MATCHER = re.compile('^[aeiou].+')
def build_relationship_target_sentence(lang, subject_word, sentence, article_pronoun_lookup, poss_pronoun_lookup, relationship_word_gender, relationship_word, subject_word_gender=None):
    target_poss_pronoun = poss_pronoun_lookup[relationship_word_gender]
    if(lang == 'en'):
        target_poss_pronoun = poss_pronoun_lookup[subject_word_gender]
    subject_word_article = article_pronoun_lookup[subject_word_gender]
    # clean up pronouns: catch vowels
    clean_subject_word = unidecode(subject_word)
    if(VOWEL_MATCHER.match(clean_subject_word)):
        if(lang == 'fr' or lang=='it'):
            subject_word_article = "l'"
    clean_relationship_word = unidecode(relationship_word)
    if(VOWEL_MATCHER.match(clean_relationship_word)):
        if(lang == 'fr'):
            target_poss_pronoun = "son"
    # add extra space for pronoun
    if(subject_word_article != "l'"):
        subject_word_article += ' '
    subject_word_NP = f'{subject_word_article}{subject_word}'
    replacement_pairs = [
        ('X', subject_word_NP), ('PRON', target_poss_pronoun), ('Y', relationship_word)
    ]
    for x, y in replacement_pairs:
        sentence = sentence.replace(x,y)
    return sentence

OCCUPATION_NON_GENDER_LANGS={'en'}
def generate_occupation_relationship_sentence_data(relationship_sents, occupation_words, relationship_words, 
                                                   lang_art_PRON_lookup, lang_POSS_PRON_lookup,
                                                   relationship_type='same_gender', langs=['es']):
    data = []
    for lang_i in langs:
        article_pronoun_lookup_i = lang_art_PRON_lookup[lang_i]
        poss_pronoun_lookup_i = lang_POSS_PRON_lookup[lang_i]
        sents_i = relationship_sents.loc[:, f'{lang_i}_sentence'].values
        sent_topics_i = relationship_sents.loc[:, 'topic'].values
        # limit to occupation words that have different forms for female/male
        if(lang_i not in OCCUPATION_NON_GENDER_LANGS):
            occupation_words_i = occupation_words[occupation_words.loc[:, f'{lang_i}_female']!=occupation_words.loc[:, f'{lang_i}_male']]
        else:
            occupation_words_i = occupation_words.copy()
        female_occupation_words_i = occupation_words_i.loc[:, f'{lang_i}_female'].values
        male_occupation_words_i = occupation_words_i.loc[:, f'{lang_i}_male'].values
        female_relationship_words_i = relationship_words.loc[:, f'{lang_i}_female'].values
        male_relationship_words_i = relationship_words.loc[:, f'{lang_i}_male'].values
        if(relationship_type=='same_gender'):
            male_subject_occupation_relationship_combos_i = [(x[0], x[1], 'male') for x in list(product(male_occupation_words_i, male_relationship_words_i))]
            female_subject_occupation_relationship_combos_i = [(x[0], x[1], 'female') for x in list(product(female_occupation_words_i, female_relationship_words_i))]
        else:
            male_subject_occupation_relationship_combos_i = [(x[0], x[1], 'female') for x in list(product(male_occupation_words_i, female_relationship_words_i))]
            female_subject_occupation_relationship_combos_i = [(x[0], x[1], 'male') for x in list(product(female_occupation_words_i, male_relationship_words_i))]

        for sent_j, sent_topic_j in zip(sents_i, sent_topics_i):
            subject_word_gender_i = 'male'
            for subject_word_k, relationship_word_k, relationship_gender_k in male_subject_occupation_relationship_combos_i:
                final_sent_j = build_relationship_target_sentence(lang_i, subject_word_k, sent_j, article_pronoun_lookup_i, poss_pronoun_lookup_i, relationship_gender_k, relationship_word_k, subject_word_gender=subject_word_gender_i)
                data.append([final_sent_j, lang_i, subject_word_k, relationship_word_k, subject_word_gender_i, sent_topic_j])
            subject_word_gender_i = 'female'
            for subject_word_k, relationship_word_k, relationship_gender_k in female_subject_occupation_relationship_combos_i:
                final_sent_j = build_relationship_target_sentence(lang_i, subject_word_k, sent_j, article_pronoun_lookup_i, poss_pronoun_lookup_i, relationship_gender_k, relationship_word_k, subject_word_gender=subject_word_gender_i)
                data.append([final_sent_j, lang_i, subject_word_k, relationship_word_k, subject_word_gender_i, sent_topic_j])
    data = pd.DataFrame(data, columns=['sent', 'lang', 'subject_word', 'relationship_word', 'subject_gender', 'relationship_topic'])
    other_gender_lookup = {'male' : 'female', 'female' : 'male'}
    if(relationship_type=='same_gender'):
        data = data.assign(**{'relationship_gender' : data.loc[:, 'subject_gender'].values})
    else:
        data = data.assign(**{'relationship_gender' : data.loc[:, 'subject_gender'].apply(other_gender_lookup.get)})
    return data

def load_relationship_occupation_template_data():
    occupation_words = pd.read_csv('data/multilingual_gender_occupations.tsv', sep='\t', index_col=False)
    # copy English occupations => male/female columns
    occupation_words = occupation_words.assign(**{
        'en_male' : occupation_words.loc[:, 'en'].values,
        'en_female': occupation_words.loc[:, 'en'].values,
    })
    relationship_words = pd.read_csv('data/multilingual_relationship_words.tsv', sep='\t', index_col=False)
    relationship_sents = pd.read_csv('data/multilingual_relationship_sentences.tsv', sep='\t', index_col=False)
    langs = ['es', 'fr', 'it', 'en']
    lang_art_PRON_lookup = {
        'es' : {
            'female' : 'la',
            'male' : 'el',
        },
        'fr' : {
            'female' : 'la',
            'male' : 'le',
        },
        'it' : {
            'female' : 'la',
            'male' : 'il',
        },
        'en' : {
            'female' : 'the',
            'male' : 'the',
        }
    }
    lang_POSS_PRON_lookup = {
        'es' : {
            'female' : 'su',
            'male' : 'su',
        },
        'fr' : {
            'female' : 'sa',
            'male' : 'son',
        },
        'it' : {
            'female' : 'la sua',
            'male' : 'il suo',
        },
        'en' : {
            'female' : 'her',
            'male' : 'his'
        }
    }
    return occupation_words, relationship_words, relationship_sents, langs, lang_art_PRON_lookup, lang_POSS_PRON_lookup    

## convert subject and translation words -> en
def extract_subject_relationship_gender(data, relationship_words):
    ## get gender of possessive pronoun for subject + gender of relationship word
    pronoun_matcher = re.compile('his|her')
    pronoun_gender_lookup = {'his' : 'male', 'her' : 'female'}
    en_female_relationship_words = relationship_words.loc[:, 'en_female'].apply(unidecode).values.tolist()
    en_male_relationship_words = relationship_words.loc[:, 'en_male'].apply(unidecode).values.tolist()
    en_relationship_words = en_female_relationship_words + en_male_relationship_words
    relationship_word_matcher = re.compile('|'.join(en_relationship_words))
    relationship_gender_lookup = {
        x : 'male' for x in en_male_relationship_words
    }
    relationship_gender_lookup.update({x : 'female' for x in en_female_relationship_words})
    for x in data.loc[:, 'translation_txt'].values:
        try:
            relationship_gender_lookup[relationship_word_matcher.search(unidecode(x)).group(0)]
        except Exception as e:
            print(f'bad text {x}')
    #         break
    data = data.assign(**{
        'translation_subject_gender' : data.loc[:, 'translation_txt'].apply(lambda x: unidecode(x)).apply(lambda x: pronoun_gender_lookup[pronoun_matcher.search(x).group(0)] if pronoun_matcher.search(x) is not None else None),
        'translation_relationship_gender' : data.loc[:, 'translation_txt'].apply(lambda x: unidecode(x)).apply(lambda x: relationship_gender_lookup[relationship_word_matcher.search(x).group(0)] if relationship_word_matcher.search(x) is not None else None),
    })
    return data
def translate_subject_relationship_words(data, occupation_words, relationship_words, langs=['es', 'fr', 'it']):
    word_other_en_lookup = {}
    en_occupation_words = occupation_words.loc[:, 'en'].values.tolist()
    en_female_relationship_words = relationship_words.loc[:, 'en_female'].apply(unidecode).values.tolist()
    en_male_relationship_words = relationship_words.loc[:, 'en_male'].apply(unidecode).values.tolist()
    en_relationship_words = en_female_relationship_words + en_male_relationship_words
    for lang in langs:
        lang_relationship_words = relationship_words.loc[:, f'{lang}_female'].values.tolist() + relationship_words.loc[:, f'{lang}_male'].values.tolist()
        lang_occupation_words = occupation_words.loc[:, f'{lang}_female'].values.tolist() + occupation_words.loc[:, f'{lang}_male'].values.tolist()
        word_lang_en_lookup = dict(zip(lang_relationship_words, en_relationship_words+en_relationship_words))
        word_lang_en_lookup.update(dict(zip(lang_occupation_words, en_occupation_words+en_occupation_words)))
        word_other_en_lookup[lang] = word_lang_en_lookup
    data = data.assign(**{
        'subject_word_en' : data.apply(lambda x: word_other_en_lookup[x.loc['lang']][x.loc['subject_word']], axis=1),
        'relationship_word_en' : data.apply(lambda x: word_other_en_lookup[x.loc['lang']][x.loc['relationship_word']], axis=1),
    })
    return data

def load_clean_relationship_sent_data(langs=['es', 'fr', 'it', 'en']):
    occupation_words, relationship_words, relationship_sents, langs, lang_art_PRON_lookup, lang_POSS_PRON_lookup = load_relationship_occupation_template_data()
    same_gender_relationship_sent_data = generate_occupation_relationship_sentence_data(relationship_sents, 
                                                                                        occupation_words, 
                                                                                        relationship_words,
                                                                                        lang_art_PRON_lookup, 
                                                                                        lang_POSS_PRON_lookup,
                                                                                        relationship_type='same_gender', 
                                                                                        langs=langs)
    diff_gender_relationship_sent_data = generate_occupation_relationship_sentence_data(relationship_sents, 
                                                                                        occupation_words, 
                                                                                        relationship_words,
                                                                                        lang_art_PRON_lookup, 
                                                                                        lang_POSS_PRON_lookup,
                                                                                        relationship_type='diff_gender', 
                                                                                        langs=langs)
    relationship_sent_data = pd.concat([
        same_gender_relationship_sent_data.assign(**{'relationship_type' : 'same_gender'}),
        diff_gender_relationship_sent_data.assign(**{'relationship_type' : 'diff_gender'}),
    ], axis=0)
    ## add EN translations + relationship-target categories
    relationship_sent_data = translate_subject_relationship_words(relationship_sent_data, occupation_words, relationship_words, langs=langs)
    # add relationship target categories
    relationship_target_categories = {
    'FRIEND' : ['boyfriend', 'girlfriend'],
    'ENGAGE' : ['fiance', 'fiancee'],
    'SPOUSE' : ['husband', 'wife'],
    }
    relationship_target_categories = {
        v1 : k for k, v in relationship_target_categories.items() for v1 in v
    }
    relationship_sent_data = relationship_sent_data.assign(**{
        'relationship_word_category' : relationship_sent_data.loc[:, 'relationship_word_en'].apply(relationship_target_categories.get)
    })
    return relationship_sent_data

def load_clean_translation_data(data_file):
    data = pd.read_csv(data_file, compression='gzip', sep='\t')
    # remove accidental EN translations
    if('en' in data.loc[:, 'lang'].unique()):
        data = data[data.loc[:, 'lang']!='en']
    # reassign gender to subject/relationships
    occupation_words, relationship_words, relationship_sents, langs, lang_art_PRON_lookup, lang_POSS_PRON_lookup = load_relationship_occupation_template_data()
    # remove accidental EN translations
    langs = list(filter(lambda x: x!='en', langs))
    data = extract_subject_relationship_gender(data, relationship_words)
    data = translate_subject_relationship_words(data, occupation_words, relationship_words)
    data = data.assign(**{
        'subject_gender_match': (data.loc[:, 'subject_gender'] == data.loc[:, 'translation_subject_gender']).astype(int),
        'relationship_gender_match': (data.loc[:, 'subject_gender'] == data.loc[:, 'translation_relationship_gender']).astype(int),
    })
    valid_data = data.dropna(subset=['translation_subject_gender', 'translation_relationship_gender'], how='all')
    valid_data = valid_data.assign(**{
        'subject_relationship_gender_match': valid_data.loc[:, ['relationship_gender_match', 'subject_gender_match']].min(axis=1)
    })
    # fix relationship category names
    relationship_target_categories = {
        'FRIEND': ['boyfriend', 'girlfriend'],
        'ENGAGE': ['fiance', 'fiancee'],
        'SPOUSE': ['husband', 'wife'],
    }
    relationship_target_categories = {
        v1: k for k, v in relationship_target_categories.items() for v1 in v
    }
    valid_data = valid_data.assign(**{
        'relationship_word_category': valid_data.loc[:, 'relationship_word_en'].apply(relationship_target_categories.get)
    })
    return valid_data

def str2array(s):
    """
    Convert string to numpy array.

    :param s:
    :return:
    """
    # Remove space after [
    s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s=re.sub('[,\s]+', ', ', s)
    return np.array(literal_eval(s))