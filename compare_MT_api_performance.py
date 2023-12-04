"""
Compare MT APIs' performance in aggregate and
by sub-group within data.
"""
from argparse import ArgumentParser
import os
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
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
    relationship_gender_var = 'relationship_target_gender'
    lang_name_lookup = {
        'es' : 'Spanish',
        'fr' : 'French',
        'it' : 'Italian',
    }
    for output_file in args.output_files:
        output_model = os.path.basename(output_file).replace('.csv', '')
        output_data = pd.read_csv(output_file)
        output_data = output_data.assign(**{
            'model' : output_model
        })
        ## rename col for plotting
        output_data.rename(columns={
            'relationship_gender' : relationship_gender_var,
        }, inplace=True)
        ## fix language names
        output_data = output_data.assign(**{
            'lang' : output_data.loc[:, 'lang'].apply(lambda x: lang_name_lookup[x])
        })
        ## fix model names
        output_data = output_data.assign(**{
            'model' : output_data.loc[:, 'model'].apply(lambda x: x[0].upper() + x[1:])
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

    ## compute aggregate/per-sub-group accuracy
    subject_gender_var = 'subject_gender'
    pronoun_correct_var = 'pronoun_gender_correct'
    data = data.assign(**{
        pronoun_correct_var: data.loc[:, 'target_sent_pronoun_gender'] == data.loc[:, subject_gender_var]
    })
    accuracy_var = 'accuracy'
    aggregate_accuracy = data.groupby([subject_gender_var, relationship_gender_var]).apply(
        lambda x: x.loc[:, pronoun_correct_var].mean()
    ).reset_index().rename(columns={0 : accuracy_var})
    per_model_accuracy = data.groupby(['model', subject_gender_var, relationship_gender_var]).apply(
        lambda x: x.loc[:, pronoun_correct_var].mean()
    ).reset_index().rename(columns={0 : accuracy_var})
    per_lang_accuracy = data.groupby(['lang', subject_gender_var, relationship_gender_var]).apply(
        lambda x: x.loc[:, pronoun_correct_var].mean()
    ).reset_index().rename(columns={0 : accuracy_var})
    aggregate_acc_file = os.path.join(args.output_dir, 'aggregate_accuracy.csv')
    per_model_acc_file = os.path.join(args.output_dir, 'per_model_accuracy.csv')
    per_lang_acc_file = os.path.join(args.output_dir, 'per_lang_accuracy.csv')
    aggregate_accuracy.to_csv(aggregate_acc_file, index=False)
    per_model_accuracy.to_csv(per_model_acc_file, index=False)
    per_lang_accuracy.to_csv(per_lang_acc_file, index=False)

    ## plot!! seaborn
    plot_subject_vs_relationship_gender_accuracy(
        data, pronoun_correct_var,
        relationship_gender_var, subject_gender_var
    )
    aggregate_fig_file = os.path.join(
        args.output_dir, 'aggregate_accuracy.png'
    )
    plt.savefig(aggregate_fig_file)
    ## per-model
    sub_group_var = 'model'
    plot_subject_vs_relationship_gender_accuracy_per_sub_group(
        data, pronoun_correct_var, relationship_gender_var,
        sub_group_var, subject_gender_var
    )
    per_sub_group_file = os.path.join(
        args.output_dir, f'per_{sub_group_var}_accuracy.png'
    )
    plt.savefig(per_sub_group_file)
    ## per-language
    sub_group_var = 'lang'
    plot_subject_vs_relationship_gender_accuracy_per_sub_group(
        data, pronoun_correct_var, relationship_gender_var,
        sub_group_var, subject_gender_var
    )
    per_sub_group_file = os.path.join(
        args.output_dir, f'per_{sub_group_var}_accuracy.png'
    )
    plt.savefig(per_sub_group_file)

def plot_subject_vs_relationship_gender_accuracy_per_sub_group(
        data, pronoun_correct_var, relationship_gender_var,
        sub_group_var, subject_gender_var
    ):
    """
    Plot accuracy for sentences, with subject gender = x axis
    and relationship gender = hue.
    Plot same thing per each unique sub-group (e.g. model)

    :param data:
    :param pronoun_correct_var:
    :param relationship_gender_var:
    :param sub_group_var:
    :param subject_gender_var:
    :return:
    """
    sub_groups = list(sorted(data.loc[:, sub_group_var].unique()))
    N_sub_groups = len(sub_groups)
    f, axs = plt.subplots(1, N_sub_groups, figsize=(5 * N_sub_groups, 3))
    for i, sub_group in enumerate(sub_groups):
        ax = axs[i]
        data_model = data[data.loc[:, sub_group_var] == sub_group]
        plot_subject_vs_relationship_gender_accuracy(
            data_model, pronoun_correct_var, relationship_gender_var,
            subject_gender_var, ax=ax
        )
        ax.set_title(sub_group)
    plt.tight_layout()


def plot_subject_vs_relationship_gender_accuracy(
        data, pronoun_correct_var, relationship_gender_var,
        subject_gender_var, ax=None
    ):
    """
    Plot accuracy for sentences, with subject gender = x axis
    and relationship gender = hue.

    :param data:
    :param pronoun_correct_var:
    :param relationship_gender_var:
    :param subject_gender_var:
    :param ax:
    :return:
    """
    sns.barplot(
        data=data, x=subject_gender_var, y=pronoun_correct_var,
        hue=relationship_gender_var, ax=ax
    )
    ## fix plot labels
    if(ax is None):
        plt.ylabel('Pronoun gender accuracy')
        plt.xlabel('Subject gender')
        plt.legend(
            title='Relationship target gender',
            loc='upper right'
        )
    else:
        ax.set_ylabel('Pronoun gender accuracy')
        ax.set_xlabel('Subject gender')
        ax.legend(
            title='Relationship target gender',
            loc='upper right'
        )

if __name__ == '__main__':
    main()
