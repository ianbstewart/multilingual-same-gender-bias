"""
Compare MT APIs' performance in aggregate and
by sub-group within data.
"""
from argparse import ArgumentParser
import os
import pandas as pd
import spacy
def main():
    parser = ArgumentParser()
    parser.add_argument(
        'output_files',
        nargs='+'
    )
    parser.add_argument(
        '--output_dir',
        default='data/output/'
    )
    args = parser.parse_args()

    ## load data
    data = []
    for output_file in args.output_files:
        output_model = os.path.basename(output_file).replace('.csv', '')
        output_data = pd.read_csv(output_file)
        output_data = output_data.assign(**{
            'model' : output_model
        })
        data.append(output_data)
    data = pd.concat(data)

    ## get target sent pronouns
    nlp_pipeline = spacy.load('en_core_web_md')
    pronouns = {'his', 'her', 'their'}
    pronoun_gender_lookup = {'his': 'male', 'her': 'female', 'their': 'neutral'}
    data = data.assign(**{'target_sent_doc': list(nlp_pipeline.pipe(data.loc[:, 'target_sent']))})
    data = data.assign(**{
        'target_sent_pronouns': data.loc[:, 'target_sent_doc'].apply(
            lambda x: list(filter(lambda y: y.text.lower() in pronouns, x))
        )
    })
    data = data.assign(**{
        'target_sent_pronoun_gender': data.loc[:, 'target_sent_pronouns'].apply(
            lambda x: pronoun_gender_lookup.get(x[0].text)
        )
    })

    ## compute aggregate/per-type accuracy
    aggregate_accuracy = data.groupby(['subject_gender', 'relationship_gender']).apply(
        lambda x: sum(x.loc[:, 'target_sent_pronoun_gender'] == x.loc[:, 'subject_gender']) / x.shape[0]
    )
    per_model_accuracy = data.groupby(['model', 'subject_gender', 'relationship_gender']).apply(
        lambda x: sum(x.loc[:, 'target_sent_pronoun_gender'] == x.loc[:, 'subject_gender']) / x.shape[0]
    )
    print(aggregate_accuracy)
    print(per_model_accuracy)
    ## plot etc

if __name__ == '__main__':
    main()
