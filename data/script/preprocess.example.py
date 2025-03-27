import json
from functools import partial
from collections import Counter
from typing import Dict, Any, List
from pathlib import Path
from datasets import load_dataset, Dataset


def process_record(record: Dict[str, Any], n2i: Dict[str, int], i2n: Dict[int, str]) -> Dict[str, Any]:
    """处理单条数据记录

    Args:
        record: 原始数据记录
        n2i: name到id的映射字典
        i2n: id到name的映射字典

    Returns:
        处理后的数据记录
    """

    try:
        category_id = n2i[record['repo_name']]
        category_name = i2n[category_id]

        return {
            "context": [
                {
                    "role": "user",
                    "content": record['query']
                }
            ],
            "chosen": {
                "role": "assistant",
                "content": record['responses'][0] if record['GSB'] == 'G' else record['responses'][1]
            },
            "rejected": {
                "role": "assistant",
                "content": record['responses'][1] if record['GSB'] == 'G' else record['responses'][0]
            },
            "reference_chosen_log_prob": record['log_probs'][0] if record['GSB'] == 'G' else record['log_probs'][1],
            "reference_rejected_log_prob": record['log_probs'][1] if record['GSB'] == 'G' else record['log_probs'][0],
            "meta": {
                "category_id": category_id,
                "category_name": category_name
            }
        }
    except KeyError as e:
        raise KeyError(f"处理记录时发生错误，缺少必要字段: {e}")


def generate_category_mapping(dataset: Dataset) -> tuple[Dict[int, str], Dict[str, int], List[Dict]]:
    """生成类别映射和配置信息

    Args:
        dataset: 原始数据集

    Returns:
        tuple包含:
        - id2name: id到name的映射
        - name2id: name到id的映射
        - ratio_config: 采样率配置
    """
    id2name = {}
    name2id = {}
    ratio_config = []

    name_cnt = Counter(dataset['repo_name'])
    category_id = 1  # category_id 为 0 时不统计损失

    for name in name_cnt:
        id2name[category_id] = name
        name2id[name] = category_id
        ratio_config.append({
            "category_id": category_id,
            "category_name": name,
            "size": name_cnt[name],
            "sample_rate": 1.0
        })
        category_id += 1

    return id2name, name2id, ratio_config


def main():
    """主函数"""
    # 定义路径
    raw_data_path = 'raw/fake.data.jsonl'
    config_path = 'config/fake.data.json'
    preprocessed_data_path = 'dataset/fake.data.preprocessed.jsonl'

    # 确保必要的目录存在
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    Path(preprocessed_data_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        # 加载数据集
        dataset = load_dataset('json', data_files=raw_data_path)['train1']

        # 生成配置
        id2name, name2id, ratio_config = generate_category_mapping(dataset)

        config = {
            "data_name": "fake_data",
            "ratio": ratio_config
        }

        # 保存配置
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(config, ensure_ascii=False, indent=4))

        # 处理数据集
        process_record_partial = partial(process_record, n2i=name2id, i2n=id2name)
        preprocessed_dataset = dataset.map(
            process_record_partial,
            num_proc=10,
            remove_columns=dataset.column_names
        )

        # 保存处理后的数据集
        preprocessed_dataset.save_to_disk(preprocessed_data_path)

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()



