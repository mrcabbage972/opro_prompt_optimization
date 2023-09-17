import json
import os
from typing import List
from data_io import download_file, get_cache_dir
from schema import ProblemExample


def get_dataset(split: str = None) -> List[ProblemExample]:
        assert split in ['test', 'train']

        source_uri = f"https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"

        target_path = os.path.join(get_cache_dir(), f'gsm8k_{split}.jsonl')
        download_file(source_uri=source_uri, target_path=target_path)
        with open(target_path, 'r') as fin:
            raw_data = [json.loads(x) for x in fin]

        dataset = [ProblemExample(question=x['question'], answer=x['answer']) for x in raw_data]

        return dataset