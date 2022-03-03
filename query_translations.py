"""
Query external APIs for translations.
"""
import os
from argparse import ArgumentParser
from data_helpers import load_clean_relationship_sent_data, set_up_google_translate_client, get_google_translations


def main():
    parser = ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('--langs', nargs='+', default=['es', 'fr', 'it'])
    args = vars(parser.parse_args())
    out_dir = args['out_dir']
    langs = args['langs']

    ## load data
    relationship_sent_data = load_clean_relationship_sent_data(langs=langs)

    ## load client
    project_parent, client = set_up_google_translate_client()

    ## translate
    target_lang = 'es'
    translation_sents = relationship_sent_data.loc[:, 'sent'].progress_apply(lambda x: get_google_translations(x, target_lang, client, project_parent))
    relationship_sent_data = relationship_sent_data.assign(**{
        'translation_txt' : translation_sents
    })

    ## save
    for relationship_type_i, data_i in relationship_sent_data.groupby('relationship_type'):
        relationship_type_i = relationship_type_i.replace("_", '')
        out_file_i = os.path.join(out_dir, f'multilingual_occupation_relationship={relationship_type_i}_model=googletranslate_translations.gz')
        data_i.to_csv(out_file_i, sep='\t', compression='gzip', index=False)

if __name__ == '__main__':
    main()