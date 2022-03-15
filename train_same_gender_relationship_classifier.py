"""
Train classifier for same/diff gender
relationship sentences.
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import load_metric
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from transformers import MBartForSequenceClassification, TrainingArguments, Trainer
from datasets.arrow_dataset import Dataset
from data_helpers import load_clean_relationship_sent_data, load_multilingual_tokenizer

def train_model(model_name, out_dir,
                train_data, test_data,
                tokenizer):
    """
    Train model.

    :param lang:
    :param model_name:
    :param data:
    :return:
    """
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
    # tmp debug lol
    def compute_metric_func(eval_pred):
        # print(f'predictions = {eval_pred.predictions}')
        score_dict = {
            'f1' : f1_score(eval_pred.predictions[0].argmax(axis=1),
                            eval_pred.label_ids)
        }
        # return compute_metric.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids)
        return score_dict
    # compute_metric_func = lambda x: compute_metric.compute(predictions=np.argmax(x.predictions, axis=1), references=x.label_ids)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metric_func,
    )
    trainer.train()


def get_train_test_data(data, lang, output_var='relationship_type', output_var_default='same_gender'):
    """
    Get train/test data.

    :param data:
    :param lang:
    :param output_var:
    :param output_var_default:
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
    return train_data, test_data, tokenizer

def test_model(out_dir, test_data):
    """
    Test model.

    :param out_dir:
    :param test_data:
    :return:
    """
    # reload
    latest_checkpoint_dir = list(sorted(os.listdir(out_dir), key=lambda x: int(x.split('-')[-1])))[-1]
    latest_checkpoint_dir = os.path.join(out_dir, latest_checkpoint_dir)
    model = MBartForSequenceClassification.from_pretrained(latest_checkpoint_dir)
    model.eval()
    with torch.no_grad():
        test_data_output = [model(**{k: v.unsqueeze(0).to(torch.cuda.current_device()) for k, v in x.items()}) for x in
                            tqdm(test_data)]
    test_data_output_logits = torch.vstack([x['logits'] for x in test_data_output])
    test_data_output_logit_labels = test_data_output_logits.argmax(axis=1).cpu()
    print(classification_report(test_data['labels'], test_data_output_logit_labels))

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
    output_var = 'relationship_type'
    output_var_default = 'same_gender'
    train_data, test_data, tokenizer = get_train_test_data(relationship_sent_data, lang, output_var, output_var_default)
    # train
    # train_model(model_name, out_dir, train_data, test_data, tokenizer)

    ## test model
    test_model(out_dir, test_data)


if __name__ == '__main__':
    main()