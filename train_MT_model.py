"""
Train MT model on predefined corpora.
"""
import os
import sys
from argparse import ArgumentParser
from datasets import load_dataset, load_metric, load_from_disk
from transformers import MBartTokenizer, MBartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from data_helpers import load_multilingual_tokenizer

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
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--model_dir', default=None) # pretrained model dir
    args = vars(parser.parse_args())
    out_dir = args['out_dir']
    dataset_name = args['dataset']
    source_lang = args['source_lang']
    model_type = args['model_type']
    sample_size = args['sample_size']
    model_dir = args['model_dir']

    # load tokenizer
    lang_token_lookup = {
        'en': 'en_XX',
        'es': 'es_XX',
        'fr': 'fr_XX',
        'it': 'it_IT',
    }
    target_lang = 'en'
    target_lang_token = lang_token_lookup[target_lang]
    if(model_type == 'mbart'):
        tokenizer = load_multilingual_tokenizer(target_lang_token)
        # model = MBartForConditionalGeneration.from_pretrained(model_name)
    ## load data
    data_dir = os.path.join(out_dir, f'data_{source_lang}')
    train_data_file = os.path.join(data_dir, 'train_data')
    if(not os.path.exists(train_data_file)):
        # custom data set
        if(os.path.exists(dataset_name)):
            dataset = load_from_disk(dataset_name)
        else:
            dataset = load_dataset(dataset_name, lang1='en', lang2=source_lang)
            dataset = dataset['train']
            # flip source/target lang in data
            src_data = [x[source_lang] for x in dataset['translation']]
            tgt_data = [x['en'] for x in dataset['translation']]
            dataset.remove_columns('translation')
            ## TODO: error w/ val data?
            # if(sample_size is not None):
            #     src_data = src_data[:sample_size]
            #     tgt_data = tgt_data[:sample_size]
            # tokenize text etc
            # en_token = 'en_XX'
            max_length = 128
            # src_txt = [en_token + ' ' + x for x in src_data]
            src_token = [tokenizer(x, max_length=max_length) for x in src_data]
            tgt_token = [tokenizer(x, max_length=max_length) for x in tgt_data]
            # tgt_token = [{'input_ids': x['input_ids'][1:]} for x in tgt_token]
            dataset = dataset.add_column('input_ids', [x['input_ids'] for x in src_token])
            dataset = dataset.add_column('attention_mask', [x['attention_mask'] for x in src_token])
            dataset = dataset.add_column('labels', [x['input_ids'] for x in tgt_token])
        if(sample_size is not None and sample_size < len(dataset)):
            dataset = dataset.shuffle(seed=123).select(list(range(sample_size))).flatten_indices()
        # split into train/val/test etc
        test_pct = 0.1
        train_test_data = dataset.train_test_split(test_size=test_pct, seed=123)
        train_dataset = train_test_data['train']
        N_train = int(len(train_dataset) * 0.8)
        train_train_data = train_dataset.select(list(range(N_train))).flatten_indices()
        train_val_data = train_dataset.select(list(range(N_train, len(train_dataset)))).flatten_indices()
        test_data = train_test_data['test']
        # save => load later
        train_train_data.save_to_disk(train_data_file)
        train_val_data.save_to_disk(os.path.join(data_dir, 'val_data'))
        test_data.save_to_disk(os.path.join(data_dir, 'test_data'))
    else:
        train_train_data = load_from_disk(train_data_file)
        train_val_data = load_from_disk(os.path.join(data_dir, 'val_data'))
    ## load model
    if (model_dir is not None):
        model = MBartForConditionalGeneration.from_pretrained(model_dir, cache_dir=out_dir)
    elif(model_type == 'mbart'):
        model_name = 'facebook/mbart-large-50'
        model = MBartForConditionalGeneration.from_pretrained(model_name, cache_dir=out_dir)
    device_id = 0
    device = f'cuda:{device_id}'
    model.to(device)

    ## set up trainer
    batch_size = 4
    num_train_epochs = 3
    ## TODO: w/ multilingual model => retrain w/ more epochs?? performance is basically noise when using multiple langs
    ## TODO: allow re-training in case of timeout, e.g. load model and trainer from latest checkpoint
    training_out_dir = f'finetune_translate_mbart_lang={source_lang}'
    if(not os.path.exists(training_out_dir)):
        os.mkdir(training_out_dir)
    training_checkpoints = list(filter(lambda x: 'checkpoint' in x, os.listdir(training_out_dir)))
    # train only if we have no checkpoints or model dir is provided
    if(len(training_checkpoints) == 0 or model_dir is not None):
        # output fine-tuned model in separate sub-dir because data is terrible
        if(model_dir is not None):
            training_out_dir = os.path.join(training_out_dir, 'finetune')
        training_args = Seq2SeqTrainingArguments(
            training_out_dir,
            evaluation_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=num_train_epochs,
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
    # most_recent_checkpoint = os.path.join(training_out_dir, training_checkpoints[0])
    # trained_model = MBartForConditionalGeneration.from_pretrained(most_recent_checkpoint)
    ## evaluate!
    # test_data = load_from_disk(os.path.join(data_dir, 'test_data'))
    # test_cols = ['input_ids', 'attention_mask', 'labels']
    # test_data.set_format(columns=test_cols, type='torch')
    # with torch.no_grad():
    #     device_id = 0
    #     device = f'cuda:{device_id}'
    #     model.to(device)
    #     ## TODO: generate w/ constraints e.g. no repetition
    #     test_output = [trained_model.generate(**{c: x[c].to(device).unsqueeze(0) for c in test_cols}) for x in tqdm(test_data)]
    #     # write output
    #     test_output = tokenizer.batch_decode(test_output)
    #     test_output_data = pd.DataFrame(test_output, columns=['pred_output'])
    #     test_output_data = test_output_data.assign(**{
    #         'input' : tokenizer.batch_decode(test_data['input_ids']),
    #         'output' : tokenizer.batch_decode(test_data['labels'])
    #     })
    #     test_output_file = os.path.join(training_out_dir, f'test_data_output.gz')
    #     test_output_data.to_csv(test_output_file, sep='\t', compression='gzip', index=False)
    # ## TODO: BLEU, ROUGE, METEOR

if __name__ == '__main__':
    main()
