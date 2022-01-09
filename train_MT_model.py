"""
Train MT model on predefined corpora.
"""
import os
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import MBartTokenizer, MBartForConditionalGeneration

def main():
    parser = ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('--source_lang', default='es') # es, fr, it
    parser.add_argument('--dataset', default='europarl-bilingual')
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
        tokenizer = MBartTokenizer.from_pretrained(model_name, tgt_lang=target_lang_token)
        # model = MBartForConditionalGeneration.from_pretrained(model_name)
    ## load data
    data_dir = os.path.join(out_dir, f'data_{source_lang}')
    train_data_file = os.path.join(data_dir, 'train_data')
    if(not os.exists(train_data_file)):
        dataset = load_dataset(dataset_name, cache_dir=out_dir, lang1='en', lang2=source_lang)
        # flip source/target lang in data
        src_data = [x[source_lang] for x in dataset['translation']]
        tgt_data = [x['en'] for x in dataset['translation']]
        dataset.remove_column('translation')
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
        train_dataset, train_val_data = train_dataset.select(list(range(N_train))), train_dataset.select(
            list(range(N_train, len(train_dataset))))
        test_data = train_test_data['test']
        # save => load later as streaming
        ## TODO: why does this crash RAM??
        import torch
        train_dataset.save_to_disk(train_data_file)
        train_val_data.save_to_disk(os.path.join(data_dir, 'val_data'))
        test_data.save_to_disk(os.path.join(data_dir, 'test_data'))
    ## load model
    if(model_type == 'mbart'):
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        device_id = 0
        device = f'cuda:{device_id}'
        model.to(device)

if __name__ == '__main__':
    main()