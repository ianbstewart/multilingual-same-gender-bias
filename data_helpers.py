import pandas as pd
from itertools import combinations, product
from unidecode import unidecode
import re

VOWEL_MATCHER = re.compile('^[aeiou].+')
def build_relationship_target_sentence(lang, subject_word, sentence, article_pronoun_lookup, poss_pronoun_lookup, relationship_word_gender, relationship_word):
    poss_pronoun = poss_pronoun_lookup[relationship_word_gender]
    subject_word_pronoun = article_pronoun_lookup[relationship_word_gender]
    # clean up pronouns: catch vowels
    clean_subject_word = unidecode(subject_word)
    if(VOWEL_MATCHER.match(clean_subject_word)):
        if(lang == 'fr' or lang=='it'):
            subject_word_pronoun = "l'"
    clean_relationship_word = unidecode(relationship_word)
    if(VOWEL_MATCHER.match(clean_relationship_word)):
        if(lang == 'fr'):
            poss_pronoun = "son"
    # add extra space for pronoun
    if(subject_word_pronoun != "l'"):
        subject_word_pronoun += ' '
    subject_word_NP = f'{subject_word_pronoun}{subject_word}'
    replacement_pairs = [
        ('X', subject_word_NP), ('PRON', poss_pronoun), ('Y', relationship_word)
    ]
    for x, y in replacement_pairs:
        sentence = sentence.replace(x,y)
    return sentence

def generate_occupation_relationship_sentence_data(relationship_sents, occupation_words, relationship_words, 
                                                   lang_art_PRON_lookup, lang_POSS_PRON_lookup, 
                                                   relationship_type='same_gender', langs=['es']):
    data = []
    for lang_i in langs:
        article_pronoun_lookup_i = lang_art_PRON_lookup[lang_i]
        poss_pronoun_lookup_i = lang_POSS_PRON_lookup[lang_i]
        sents_i = relationship_sents.loc[:, f'{lang_i}_sentence'].values
        # limit to occupation words that have different forms for female/male
        occupation_words_i = occupation_words[occupation_words.loc[:, f'{lang_i}_female']!=occupation_words.loc[:, f'{lang_i}_male']]
        female_occupation_words_i = occupation_words_i.loc[:, f'{lang_i}_female'].values
        male_occupation_words_i = occupation_words_i.loc[:, f'{lang_i}_male'].values
        female_relationship_words_i = relationship_words.loc[:, f'{lang_i}_female'].values
        male_relationship_words_i = relationship_words.loc[:, f'{lang_i}_male'].values
        if(relationship_type=='same_gender'):
            male_subject_occupation_relationship_combos_i = list(product(male_occupation_words_i, male_relationship_words_i))
            female_subject_occupation_relationship_combos_i = list(product(female_occupation_words_i, female_relationship_words_i))
        else:
            male_subject_occupation_relationship_combos_i = list(product(male_occupation_words_i, female_relationship_words_i))
            female_subject_occupation_relationship_combos_i = list(product(female_occupation_words_i, male_relationship_words_i))
        for sent_j in sents_i:
            for subject_word_k, relationship_word_k in male_subject_occupation_relationship_combos_i:
                final_sent_j = build_relationship_target_sentence(lang_i, subject_word_k, sent_j, article_pronoun_lookup_i, poss_pronoun_lookup_i, 'male', relationship_word_k)
                data.append([final_sent_j, lang_i, subject_word_k, relationship_word_k, 'male'])
            for subject_word_k, relationship_word_k in female_subject_occupation_relationship_combos_i:
                final_sent_j = build_relationship_target_sentence(lang_i, subject_word_k, sent_j, article_pronoun_lookup_i, poss_pronoun_lookup_i, 'female', relationship_word_k)
                data.append([final_sent_j, lang_i, subject_word_k, relationship_word_k, 'female'])
    data = pd.DataFrame(data, columns=['sent', 'lang', 'subject_word', 'relationship_word', 'subject_gender'])
    other_gender_lookup = {'male' : 'female', 'female' : 'male'}
    if(relationship_type=='same_gender'):
        data = data.assign(**{'relationship_gender' : data.loc[:, 'subject_gender'].values})
    else:
        data = data.assign(**{'relationship_gender' : data.loc[:, 'subject_gender'].apply(other_gender_lookup.get)})
    return data

def load_relationship_occupation_template_data():
    occupation_words = pd.read_csv('data/multilingual_gender_occupations.tsv', sep='\t', index_col=False)
    relationship_words = pd.read_csv('data/multilingual_relationship_words.tsv', sep='\t', index_col=False)
    relationship_sents = pd.read_csv('data/multilingual_relationship_sentences.tsv', sep='\t', index_col=False)
    langs = ['es', 'fr', 'it']
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
def translate_subject_relationship_words(data, occupation_words, relationship_words):
    word_other_en_lookup = {}
    langs = ['es', 'fr', 'it']
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