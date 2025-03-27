import json
import argparse
from transformers import AutoTokenizer
from datasets import load_from_disk

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class Tokenizer(object):
    def __init__(self, input_path, tokenizer_path, num_workers, output_path, ignore_index=-100):
        self.dataset = load_from_disk(input_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.num_workers = num_workers
        self.output_path = output_path
        self.ignore_index = ignore_index
        
    def merge(self, record):
        record['chosen_messages'] = record['context'] + [record['chosen']]
        record['rejected_messages'] = record['context'] + [record['rejected']]
        chosen_ids = self.process(record, 'chosen_messages')
        rejected_ids = self.process(record, 'rejected_messages')

        return {
            "input_ids": chosen_ids['input_ids']+rejected_ids['input_ids'],
            "token_labels": chosen_ids['token_labels']+rejected_ids['token_labels'],
            "category_ids": chosen_ids['category_ids']+rejected_ids['category_ids'],
            "seq_lens": [len(chosen_ids['input_ids']), len(rejected_ids['input_ids'])],
            "log_probs": [record['reference_chosen_log_prob'], record['reference_rejected_log_prob']]
        }



    def process(self, record, key):
        input_ids = self.tokenizer.apply_chat_template(record[key],tokenize=True, add_generation_prompt=False)
        token_labels = [-100] * len(input_ids)
        category_ids = [0] * len(input_ids)
        prompt_ids = self.tokenizer.apply_chat_template(record[key][:-1], tokenize=True, add_generation_prompt=True)
        start_idx = len(prompt_ids)
        token_labels[start_idx:] = input_ids[start_idx:]
        category_ids[start_idx:] = [record['meta']['category_id']] * (len(input_ids) - start_idx)
        return {
            "input_ids": input_ids,
            "token_labels": token_labels,
            "category_ids": category_ids,
        }

    def tokenize(self):
        self.dataset = self.dataset.map(
            self.merge,
            num_proc=self.num_workers,
            remove_columns=self.dataset.column_names,
            desc="分词"
        )

    def save(self):
        self.dataset.save_to_disk(self.output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='数据路径')
    parser.add_argument('--output-path', type=str, help='数据保存路径')
    parser.add_argument('--tokenizer-path', type=str, help='tokenizer路径')
    parser.add_argument('--num-workers', type=int, help='并行处理的工作进程数')
    return parser.parse_args()

def main():
    args = parse_args()
    tokenizer = Tokenizer(input_path=args.input_path,
                        tokenizer_path=args.tokenizer_path,
                        num_workers=args.num_workers,
                      output_path=args.output_path)
    tokenizer.tokenize()
    tokenizer.save()

if __name__ == "__main__":
    main()