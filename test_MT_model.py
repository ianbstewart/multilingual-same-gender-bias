"""
Test fully-trained MT model.
"""
import os
from argparse import ArgumentParser

import pandas as pd
import torch
from transformers import MBartTokenizer, MBartForConditionalGeneration
from datasets import load_from_disk
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
    model.eval()
    device = f'cuda:{device_id}'
    model.to(device)

    # generate
    test_cols = ['input_ids', 'attention_mask']
    with torch.no_grad():
        test_pred_output = [model.generate(**{c: x[c].to(device).unsqueeze(0) for c in test_cols}) for x in tqdm(test_data)]
    # write to file
    test_file = os.path.join(out_dir, 'test_data_output.gz')
    test_input = tokenizer.batch_decode(test_data['input_ids'], skip_special_tokens=True)
    test_output = tokenizer.batch_decode(test_data['labels'])
    test_output_data = pd.DataFrame([test_input, test_output, test_pred_output], columns=['input', 'output', 'pred'])
    test_output_data.to_csv(test_file, sep='\t', compression='gzip', index=False)

    # compute accuracy metrics
    # bleu_metric = load_metric('BLEU')
    # rouge_metric = load_metric('rouge')

if __name__ == '__main__':
    main()
