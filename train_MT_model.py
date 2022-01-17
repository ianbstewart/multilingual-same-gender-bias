"""
Train MT model on predefined corpora.
"""
import os
from argparse import ArgumentParser
from datasets import load_dataset, load_metric
from transformers import MBartTokenizer, MBartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import numpy as np

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels
METRIC = load_metric("sacrebleu")
def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = METRIC.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

import numpy as np
from datasets import load_metric

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels
METRIC = load_metric("sacrebleu")
def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = METRIC.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def main():
    parser = ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('--source_lang', default='es') # es, fr, it
    parser.add_argument('--dataset', default='europarl_bilingual')
    parser.add_argument('--model_type', default='mbart') # mbart,
    args = vars(parser.parse_args())
    out_dir = args['out_dir']
    dataset_name = args['dataset']
    source_lang = args['source_lang']
    model_type = args['model_type']

    # load tokenizer
    lang_token_lookup = {
        'en': 'en_XX',
        'es': 'es_XX',
        'fr': 'fr_XX',
        'it': 'it_IT',
    }
    target_lang_token = lang_token_lookup['en']
    if (model_type == 'mbart'):
        model_name = 'facebook/mbart-large-50'
        tokenizer = MBartTokenizer.from_pretrained(model_name, tgt_lang=target_lang_token, cache_dir=out_dir)
        # model = MBartForConditionalGeneration.from_pretrained(model_name)
    ## load data
    data_dir = os.path.join(out_dir, f'data_{source_lang}')
    train_data_file = os.path.join(out_dir, 'train_data')
    if(not os.path.exists(train_data_file)):
        dataset = load_dataset(dataset_name, lang1='en', lang2=source_lang)
        dataset = dataset['train']
        # flip source/target lang in data
        src_data = [x[source_lang] for x in dataset['translation']]
        tgt_data = [x['en'] for x in dataset['translation']]
        dataset.remove_columns('translation')
        # tokenize text etc
        en_token = 'en_XX'
        max_length = 128
        src_txt = [en_token + ' ' + x for x in src_data]
        src_token = [tokenizer(x, max_length=max_length) for x in src_txt]
        tgt_token = [tokenizer(x, max_length=max_length) for x in tgt_data]
        tgt_token = [{'input_ids': x['input_ids'][1:]} for x in tgt_token]
        dataset = dataset.add_column('input_ids', [x['input_ids'] for x in src_token])
        dataset = dataset.add_column('attention_mask', [x['attention_mask'] for x in src_token])
        dataset = dataset.add_column('labels', [x['input_ids'] for x in tgt_token])
        # split into train/val/test etc
        test_pct = 0.1
        train_test_data = dataset.train_test_split(test_size=test_pct, seed=123)
        train_dataset = train_test_data['train']
        N_train = int(len(train_dataset) * 0.8)
        train_train_data, train_val_data = train_dataset.select(list(range(N_train))), train_dataset.select(
            list(range(N_train, len(train_dataset))))
        test_data = train_test_data['test']
        # save => load later
        train_train_data.save_to_disk(train_data_file)
        train_val_data.save_to_disk(os.path.join(data_dir, 'val_data'))
        test_data.save_to_disk(os.path.join(data_dir, 'test_data'))
    else:
        train_train_data = load_dataset(train_data_file)
        train_val_data = load_dataset(os.path.join(data_dir, 'val_data'))
    ## load model
    if(model_type == 'mbart'):
        model = MBartForConditionalGeneration.from_pretrained(model_name, cache_dir=out_dir)
        device_id = 0
        device = f'cuda:{device_id}'
        model.to(device)

    ## set up trainer
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
    batch_size = 16
    max_steps = 3
    ## TODO: retrain w/ more epochs?? performance is basically noise
    training_args = Seq2SeqTrainingArguments(
        f'finetune_translate_mbart_lang={source_lang}',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=3,
        predict_with_generate=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_train_data,
        eval_dataset=train_val_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )
    ## train!!
    trainer.train()
    ## evaluate!
    ## TODO: test

if __name__ == '__main__':
    main()
