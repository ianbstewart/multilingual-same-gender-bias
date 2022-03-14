"""
Train classifier for same/diff gender
relationship sentences.
"""
from argparse import ArgumentParser

import torch
from datasets import load_metric
from transformers import MBartForSequenceClassification, TrainingArguments, Trainer
from datasets.arrow_dataset import Dataset
from data_helpers import load_clean_relationship_sent_data, load_multilingual_tokenizer

def process_train_model(lang, model_name, data, out_dir,
                        output_var='relationship_type',
                        output_var_default='same_gender'):
    """
    Process data and train model.

    :param lang:
    :param model_name:
    :param data:
    :return:
    """
    tokenizer = load_multilingual_tokenizer(tgt_lang_token=lang)
    max_length = 64
    input_data = tokenizer.batch_encode_plus(data.loc[:, 'sent'].values,
                                             max_length=max_length,
                                             truncation=True)
    # add labels
    input_data['labels'] = (data.loc[:, output_var] == output_var_default).astype(int)
    input_data = Dataset.from_dict(input_data)
    input_data.set_format(columns=['input_ids', 'attention_mask', 'labels'], type='torch')
    # split train/test
    split_input_data = input_data.train_test_split(train_size=0.9, seed=123)
    train_data = split_input_data['train']
    test_data = split_input_data['test']
    ## load model
    model = MBartForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(torch.cuda.current_device())
    ## train
    batch_size = 4
    num_train_epochs = 5
    training_args = TrainingArguments(
        out_dir,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=num_train_epochs
    )
    compute_metric = load_metric('f1')
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metric,
    )
    trainer.train()

def main():
    parser = ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('--model_name', default='facebook/mbart-large')
    parser.add_argument('--lang', default='es')
    args = vars(parser.parse_args())
    out_dir = args['out_dir']
    model_name = args['model_name']
    lang = args['lang']

    ## load data
    relationship_sent_data = load_clean_relationship_sent_data(langs=[lang])
    # process
    process_train_model(lang, model_name, relationship_sent_data, out_dir)

if __name__ == '__main__':
    main()