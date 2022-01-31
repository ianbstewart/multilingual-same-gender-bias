"""
Generate translation data,
e.g. relationships.
"""
import os
from argparse import ArgumentParser
import pandas as pd
from data_helpers import load_clean_relationship_sent_data
from transformers import MBartTokenizer
from datasets.arrow_dataset import Dataset

def main():
    parser = ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('--data_type', default='relationship')
    parser.add_argument('--source_langs', nargs='+', default=['es', 'fr', 'it'])
    parser.add_argument('--model_type', default='mbart')
    args = vars(parser.parse_args())
    out_dir = args['out_dir']
    data_type = args['data_type']
    model_type = args['model_type']
    source_langs = args['source_langs']

    if(data_type == 'relationship'):
        sentence_data = load_clean_relationship_sent_data(langs=source_langs)
        # occupation_words, relationship_words, relationship_sents, langs, lang_art_PRON_lookup, lang_POSS_PRON_lookup = load_relationship_occupation_template_data()
        # langs = ['es', 'fr', 'it']
        # same_gender_sentence_data = generate_occupation_relationship_sentence_data(relationship_sents,
        #                                                                            occupation_words, relationship_words,
        #                                                                            lang_art_PRON_lookup, lang_POSS_PRON_lookup,
        #                                                                            relationship_type='same_gender', langs=langs)
        # diff_gender_sentence_data = generate_occupation_relationship_sentence_data(relationship_sents,
        #                                                                            occupation_words, relationship_words,
        #                                                                            lang_art_PRON_lookup, lang_POSS_PRON_lookup,
        #                                                                            relationship_type='diff_gender', langs=langs)
        # sentence_data = pd.concat([
        #     same_gender_sentence_data.assign(**{'sentence_type' : 'same_gender'}),
        #     diff_gender_sentence_data.assign(**{'sentence_type' : 'diff_gender'}),
        # ], axis=0)
        ## TODO: join English translations
        en_sentence_data = load_clean_relationship_sent_data(langs=['en'])
        en_sentence_data = en_sentence_data.rename(columns={'sent' : 'sent_en'})
        data_id_cols = ['subject_word_en', 'subject_gender', 'relationship_word_en', 'relationship_topic', 'relationship_type']
        sentence_data = pd.merge(sentence_data, en_sentence_data.loc[:, ['sent_en']+data_id_cols], on=data_id_cols)
        # tmp debugging
        sentence_data.to_csv('tmp.gz', sep='\t', compression='gzip', index=False)

    # save separate file for each language
    # for lang_i, data_i in sentence_data.groupby('lang'):
    #     out_file_i = os.path.join(out_dir, f'translation_data_type={data_type}_lang={lang_i}.gz')
    #     data_i.to_csv(out_file_i, sep='\t', compression='gzip', index=False)

    ## convert to dataset format for training/testing in MT models
    max_length = 128
    if (model_type == 'mbart'):
        lang_token_lookup = {
            'en': 'en_XX',
            'es': 'es_XX',
            'fr': 'fr_XX',
            'it': 'it_IT',
        }
        model_name = 'facebook/mbart-large-50'
    for lang_i, data_i in sentence_data.groupby('lang'):
        target_lang_token_i = lang_token_lookup[lang_i]
        if(model_type == 'mbart'):
            tokenizer_i = MBartTokenizer.from_pretrained(model_name, tgt_lang=target_lang_token_i,
                                                         cache_dir=out_dir)
        dataset_i = tokenizer_i(data_i.loc[:, 'sent'].values.tolist(), max_length=max_length)
        dataset_i = Dataset.from_dict(dataset_i)
        output_data_i = tokenizer_i(data_i.loc[:, 'sent_en'].values.tolist(), max_length=max_length)
        dataset_i = dataset_i.add_column('labels', list(map(lambda x: x[:-1], output_data_i['input_ids'])))
        for id_col_j in data_id_cols:
            dataset_i = dataset_i.add_column(id_col_j, data_i.loc[:, id_col_j].values)
        ## TODO: train/val/test split
        # test_pct = 0.1
        # train_test_data = dataset.train_test_split(test_size=test_pct, seed=123)
        # train_dataset = train_test_data['train']
        # N_train = int(len(train_dataset) * 0.8)
        # train_train_data = train_dataset.select(list(range(N_train))).flatten_indices()
        # train_val_data = train_dataset.select(list(range(N_train, len(train_dataset)))).flatten_indices()
        # test_data = train_test_data['test']
        out_dir_i = os.path.join(out_dir, f'translation_data_type={data_type}_lang={lang_i}')
        dataset_i.save_to_disk(out_dir_i)

if __name__ == '__main__':
    main()