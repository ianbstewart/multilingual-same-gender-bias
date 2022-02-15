"""
Test fully-trained MT model.
"""
import os
import re
from argparse import ArgumentParser

import pandas as pd
import torch
from transformers import MBartTokenizer, MBartForConditionalGeneration
from datasets import load_from_disk, load_metric
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('out_dir')
    parser.add_argument('test_data_dir')
    parser.add_argument('--device_id', type=int, default=0)
    args = vars(parser.parse_args())
    model_dir = args['model_dir']
    out_dir = args['out_dir']
    test_data_dir = args['test_data_dir']
    device_id = args['device_id']

    # load model
    model = MBartForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = MBartTokenizer.from_pretrained(model_dir)
    test_data = load_from_disk(test_data_dir)
    # fix column types
    test_data.set_format('torch', ['input_ids', 'attention_mask', 'labels'])
    # tmp debugging
    # sample_size = 100
    # test_data = test_data.select(list(range(sample_size)))
    model.eval()
    device = f'cuda:{device_id}'
    model.to(device)

    # generate
    test_cols = ['input_ids', 'attention_mask']
    with torch.no_grad():
        test_pred_output = [model.generate(**{c: x[c].to(device).unsqueeze(0) for c in test_cols}, top_p=0.9, no_repeat_ngram_size=3)[0] for x in tqdm(test_data)]
    # convert to str
    test_pred_output_str = [tokenizer.decode(x, skip_special_tokens=True) for x in test_pred_output]
    # write to file
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    test_file = os.path.join(out_dir, 'test_data_output.gz')
    test_input = tokenizer.batch_decode(test_data['input_ids'], skip_special_tokens=True)
    test_output = tokenizer.batch_decode(test_data['labels'], skip_special_tokens=True)
    test_output_data = pd.DataFrame([test_input, test_output, test_pred_output_str],
                                    index=['input', 'output', 'pred']).transpose()
    # test_output_data.to_csv(test_file, sep='\t', compression='gzip', index=False)

    ## compute accuracy scores!!
    # convert back to tokens => BLEU
    underscore_matcher = re.compile('^▁')
    test_output_data = test_output_data.assign(**{
        'output_tokens' : test_output_data.loc[:, 'output'].apply(lambda x: list(map(lambda y: underscore_matcher.sub('', y), tokenizer.tokenize(x)))),
        'pred_tokens' : test_output_data.loc[:, 'pred'].apply(lambda x: list(map(lambda y: underscore_matcher.sub('', y), tokenizer.tokenize(x)))),
    })
    bleu_metric = load_metric('bleu')
    rouge_metric = load_metric('rouge')
    # meteor_metric = load_metric('meteor')
    test_output_data = test_output_data.assign(**{
        'BLEU_score' : test_output_data.apply(lambda x: bleu_metric.compute(predictions=[x.loc['pred_tokens']], references=[[x.loc['output_tokens']]])['bleu'], axis=1),
        'ROUGE_score' : test_output_data.apply(lambda x: rouge_metric.compute(predictions=[x.loc['pred']], references=[x.loc['output']])['rougeL'].mid.fmeasure, axis=1),
    })
    test_output_data.to_csv(test_file, sep='\t', compression='gzip', index=False)

if __name__ == '__main__':
    main()