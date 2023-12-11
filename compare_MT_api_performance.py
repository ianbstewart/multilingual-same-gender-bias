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
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.api import Logit

def plot_subject_vs_relationship_gender_accuracy_per_sub_group(
        data, pronoun_correct_var, relationship_type_var,
        sub_group_var,
    ):
    """
    Plot accuracy for sentences, with subject gender = x axis
    and relationship gender = hue.

    :param data:
    :param pronoun_correct_var:
    :param relationship_type_var:
    :param sub_group_var:
    :return:
    """
    sub_groups = list(sorted(data.loc[:, sub_group_var].unique()))
    N_sub_groups = len(sub_groups)
    width = 4
    height = 3.5
    f, axs = plt.subplots(
        1, N_sub_groups,
        figsize=(width * N_sub_groups, height),
        sharey='row'
    )
    for i, sub_group in enumerate(sub_groups):
        ax = axs[i]
        data_model = data[data.loc[:, sub_group_var] == sub_group]
        plot_subject_vs_relationship_gender_accuracy(
            data_model, pronoun_correct_var, relationship_type_var,
            ax=ax,
        )
        ax.set_title(sub_group)
        if(i > 0):
            ax.set_ylabel('')
    plt.tight_layout()


def plot_subject_vs_relationship_gender_accuracy(
        data, pronoun_correct_var, relationship_type_var,
        ax=None,
    ):
    """
    Plot accuracy for sentences, with subject gender = x axis
    and relationship gender = hue.

    :param data:
    :param pronoun_correct_var:
    :param relationship_type_var:
    :param ax:
    :return:
    """
    sns.set_style("whitegrid")
    barplot_ax = sns.barplot(
        data=data, x=relationship_type_var, y=pronoun_correct_var,
        # hue=relationship_gender_var,
        ax=ax
    )
    ## add mean values above bars
    barplot_ax.bar_label(barplot_ax.containers[0], fmt='%.3f')
    ## fix plot labels
    if(ax is None):
        plt.ylabel('Pronoun gender accuracy')
        plt.xlabel('Relationship type')
        # plt.legend(
        #     title='Relationship target gender',
        #     loc='lower right'
        # )
    else:
        ax.set_ylabel('Pronoun gender accuracy')
        ax.set_xlabel('Relationship type')
        # ax.get_legend().remove()

def load_clean_data(output_files):
    """
    Load + clean output data.

    :param output_files:
    :return: combined clean output DataFrame
    """
    data = []
    relationship_gender_var = 'relationship_target_gender'
    lang_name_lookup = {
        'es': 'Spanish',
        'fr': 'French',
        'it': 'Italian',
    }
    for output_file in output_files:
        output_model = os.path.basename(output_file).replace('.csv', '')
        output_data = pd.read_csv(output_file)
        output_data = output_data.assign(**{
            'model': output_model
        })
        ## rename col for plotting
        output_data.rename(columns={
            'relationship_gender': relationship_gender_var,
        }, inplace=True)
        ## fix language names
        output_data = output_data.assign(**{
            'lang': output_data.loc[:, 'lang'].apply(lambda x: lang_name_lookup[x])
        })
        ## fix model names
        output_data = output_data.assign(**{
            'model': output_data.loc[:, 'model'].apply(lambda x: x[0].upper() + x[1:])
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
    data = data.assign(**{
        'relationship_type' : data.apply(
            lambda x: 'Same-gender' if x.loc['relationship_target_gender']==x.loc['subject_gender'] else 'Different-gender',
            axis=1
        )
    })
    return data

def get_mean_attribute_per_occupation(
        attr, occupation_categories, occupation_data, pop_var='total_pop',
        occupation_var='Occupation'
    ):
        ## get weighted mean of attribute per occupation
        ## ex. if "artist" has 2 categories, compute X% * attr_X + Y% * attr_Y
        category_match_occupation_data = occupation_data[
            occupation_data.loc[:, occupation_var].isin(occupation_categories)]
        attr_per_pop = np.nan
        if (category_match_occupation_data.shape[0] > 0):
            attr_per_pop = (
                    (
                            category_match_occupation_data.loc[:, pop_var] *
                            category_match_occupation_data.loc[:, attr]
                    ).sum() /
                    category_match_occupation_data.loc[:, pop_var].sum()
            )
        return attr_per_pop

def clean_occupation_metadata():
    # occupation_data = pd.read_csv('data/multilingual_gender_occupations.tsv', sep='\t')
    occupation_category_data = pd.read_csv('data/metadata/occupation_categories.tsv', sep='\t')
    # convert categories => list
    category_vars = list(filter(lambda x: x.startswith('categories_'), occupation_category_data.columns))
    occupation_category_data = occupation_category_data.assign(**{
        var_i: occupation_category_data.loc[:, var_i].apply(lambda x: x.split(';') if type(x) is str else [])
        for var_i in category_vars
    })
    ## get other metadata
    age_occupation_data = pd.read_csv('data/metadata/age_occupation_data.tsv', sep='\t')
    income_gender_occupation_data = pd.read_csv('data/metadata/female_work_representation_data.tsv', sep='\t')
    # fix column names
    age_occupation_data.rename(columns={'Median\n age': 'median_age', 'Total, 16\nyears and\n over': 'total_pop'},
                               inplace=True)
    income_gender_occupation_data.rename(
        columns={'Median earnings': 'median_earnings', 'Percentage of women in occupational group': 'women_pct',
                 'Number of full-time workers': 'total_pop'}, inplace=True)
    # remove occupation w/ insufficient data
    age_occupation_data = age_occupation_data[age_occupation_data.loc[:, 'median_age'] != '–']
    # fix numbers
    number_vars = ['median_earnings', 'total_pop', 'median_age']
    for number_var_i in number_vars:
        if (number_var_i in income_gender_occupation_data.columns):
            income_gender_occupation_data = income_gender_occupation_data.assign(**{
                number_var_i: income_gender_occupation_data.loc[:, number_var_i].apply(
                    lambda x: x.replace(',', '') if type(x) is str else x).astype(float)
            })
        if (number_var_i in age_occupation_data.columns):
            age_occupation_data = age_occupation_data.assign(**{
                number_var_i: age_occupation_data.loc[:, number_var_i].apply(
                    lambda x: x.replace(',', '') if type(x) is str else x).astype(float)
            })

    # income
    income_attr = 'median_earnings'
    occupation_category_data = occupation_category_data.assign(**{
        'income': occupation_category_data.loc[:, 'categories_BOLS'].apply(
            lambda x: get_mean_attribute_per_occupation(income_attr, x, income_gender_occupation_data))
    })
    # gender
    gender_attr = 'women_pct'
    occupation_category_data = occupation_category_data.assign(**{
        'women_pct': occupation_category_data.loc[:, 'categories_BOLS'].apply(
            lambda x: get_mean_attribute_per_occupation(gender_attr, x, income_gender_occupation_data))
    })
    # age
    age_attr = 'median_age'
    occupation_category_data = occupation_category_data.assign(**{
        'age': occupation_category_data.loc[:, 'category_DOL'].apply(
            lambda x: get_mean_attribute_per_occupation(age_attr, [x], age_occupation_data))
    })
    # Z-norm everything for regression!!
    occupation_social_vars = ['income', 'women_pct', 'age']
    for var_i in occupation_social_vars:
        scaler = StandardScaler()
        vals_i = scaler.fit_transform(occupation_category_data.loc[:, var_i].values.reshape(-1, 1))
        occupation_category_data = occupation_category_data.assign(**{
            var_i: vals_i
        })
    return occupation_category_data

def run_regression(output_dir, pronoun_correct_var, data, subject_gender_var, relationship_type):
    """
    Run regression to predict whether the pronoun was translated correctly.

    :param relationship_type:
    :param args:
    :param pronoun_correct_var:
    :param data:
    :param subject_gender_var:
    :return:
    """
    relationship_category_var = 'relationship_word_category'
    lang_var = 'lang'
    model_var = 'model'
    occupation_vars = ['income', 'women_pct', 'age']
    independent_vars = [
                           subject_gender_var, lang_var, model_var, relationship_category_var,
                       ] + occupation_vars
    regression_equation = f'{pronoun_correct_var} ~ {"+".join(independent_vars)}'
    model = Logit.from_formula(regression_equation, data=data)
    reg_results = model.fit()
    ## diagnostics
    print(f'LLR = {reg_results.llr}, LLR p = {reg_results.llr_pvalue}')
    ## write to .csv file
    reg_results_summary = reg_results.summary()
    reg_results_summary_file = os.path.join(output_dir, f'{relationship_type}_acc_regression.tex')
    with open(reg_results_summary_file, 'w') as out_file:
        out_file.write(reg_results_summary.as_latex())

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

    relationship_gender_var = 'relationship_target_gender'
    output_files = args.output_files
    relationship_type_var = 'relationship_type'
    data = load_clean_data(output_files)

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
    width = 3
    height = 2.5
    plt.figure(figsize=(width, height))
    plot_subject_vs_relationship_gender_accuracy(
        data, pronoun_correct_var,
        relationship_type_var
    )
    plt.tight_layout()
    aggregate_fig_file = os.path.join(
        args.output_dir, 'aggregate_accuracy.png'
    )
    plt.savefig(aggregate_fig_file)
    ## per-model
    sub_group_vars = [
        'subject_gender', 'model', 'lang', 'subject_word_en'
    ]
    for sub_group_var in sub_group_vars:
        sub_group_vals = data.loc[:, sub_group_var].unique()
        plt.figure(figsize=(width*len(sub_group_vals), height))
        plot_subject_vs_relationship_gender_accuracy_per_sub_group(
            data, pronoun_correct_var,
            relationship_type_var,
            sub_group_var
        )
        per_sub_group_file = os.path.join(
            args.output_dir, f'per_{sub_group_var}_accuracy.png'
        )
        plt.savefig(per_sub_group_file)
    ## interaction effects??
    ## lang x subject word
    for lang, data_per_lang in data.groupby('lang'):
        sub_group_var = 'subject_word_en'
        plot_subject_vs_relationship_gender_accuracy_per_sub_group(
            data_per_lang, pronoun_correct_var, relationship_type_var,
            sub_group_var
        )
        per_sub_group_file = os.path.join(
            args.output_dir, f'per_{sub_group_var}_accuracy_lang={lang}.png'
        )
        plt.savefig(per_sub_group_file)

    ## regression
    ## dependent var = translation correct (only same-gender data)
    ## independent vars
    ## same-gender/diff-gender relationship
    ## subject gender
    ## lang
    ## relationship target
    ## relationship action
    ## occupation stats: income, female representation, median age
    ## accuracy_var ~ subject_gender + lang_var + relationship_target + relationship_action + occupation_income + occupation_female_representation + occupation_age
    ## get occupation data
    occupation_metadata = clean_occupation_metadata()
    # tmp debug
    print(f'occupation metadata cols = {occupation_metadata.columns}')
    occupation_metadata.rename(columns={'occupation' : 'subject_word_en'}, inplace=True)
    ## merge everything lol
    data = pd.merge(
        data,
        occupation_metadata,
        on='subject_word_en',
        how='left',
    ).fillna(0.)
    ## fix dependent var type
    data = data.assign(**{
        pronoun_correct_var: data.loc[:, pronoun_correct_var].astype(int)
    })
    same_gender_data = data[data.loc[:, 'subject_gender']==data.loc[:, 'relationship_target_gender']]
    diff_gender_data = data[data.loc[:, 'subject_gender']!=data.loc[:, 'relationship_target_gender']]
    relationship_type = 'same_gender'
    run_regression(args.output_dir, pronoun_correct_var, same_gender_data, subject_gender_var, relationship_type)
    ## TODO: singular matrix??
    # relationship_type = 'diff_gender'
    # tmp debug
    # print(f'diff-gender pronoun acc = {diff_gender_data.loc[:, pronoun_correct_var].mean()}')
    # run_regression(args.output_dir, pronoun_correct_var, diff_gender_data, subject_gender_var, relationship_type)


if __name__ == '__main__':
    main()
