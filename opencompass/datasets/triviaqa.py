import json
import os.path as osp
from copy import deepcopy
from typing import Union

from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import check_is_refuse

from .base import BaseDataset


@LOAD_DATASET.register_module()
class TriviaQADataset(BaseDataset):

    @staticmethod
    def load(path: str, lang: str, n_infer_dict: Union[dict, None] = None):
        dataset = DatasetDict()
        for split in ['dev', 'train']:
            filename = osp.join(path, lang, f'{split}.json')
            with open(filename, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            for i in range(len(data_list)):
                cur_dict = data_list[i]
                data_list[i]['id_and_answers'] = {
                    'qid': cur_dict['qid'],
                    'infer_id': str(0),
                    'lang': cur_dict['lang'],
                    'answers': cur_dict['answers']
                }

            if n_infer_dict is not None:
                n_infer = n_infer_dict[split]
                new_data_list = deepcopy(data_list)
                for i in range(1, n_infer):
                    copy_data_list = deepcopy(data_list)
                    for j in range(len(copy_data_list)):
                        copy_data_list[j]['id_and_answers']['infer_id'] = str(
                            i)
                    new_data_list.extend(copy_data_list)
                data_list = new_data_list

            dataset[split] = Dataset.from_list(data_list)

        return dataset


@ICL_EVALUATORS.register_module()
class TriviaQAEvaluator(BaseEvaluator):
    """
    该Evaluator可以用于评估in-context-learning条件下的base模型。
    args:
        splitters: list of str, default=None
            由于base模型在回答完成问题后，通常不会自动停止输出，而是“编造”新的问题和答案对。
            因此，使用splitter切割模型生成的文本，以便进行评估。
    """

    def __init__(self, splitters=None, **kwargs):
        super().__init__(**kwargs)
        if splitters is None:
            self.splitters = []
        else:
            self.splitters = [_.lower() for _ in splitters]

    def score(self, predictions, references, origin_prompt):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}

        details = {}
        total, correct, incorrect, refuse = 0, 0, 0, 0
        for index, (raw_pred,
                    id_and_ref) in enumerate(zip(predictions, references)):
            qid = id_and_ref['qid']
            lang = id_and_ref['lang']
            answers = id_and_ref['answers']
            infer_id = id_and_ref.get('infer_id', '0')

            split_pred = raw_pred
            split_done = False  # 是否切割成功，初始值为False
            for splitter in self.splitters:
                if splitter in raw_pred.lower():
                    kept = raw_pred.lower().split(splitter)[0]
                    split_pred = raw_pred[:len(kept)]
                    split_done = True  # 切割成功
                    break

            if check_is_refuse(split_pred, lang):
                is_correct = False
                is_incorrect = False
                is_refuse = True
                refuse += 1
                pred = '<REFUSE>'
            else:
                pred = split_pred.lower().strip()
                processed_answers = [ans.lower() for ans in answers]
                # pred = split_pred
                # processed_answers = answers

                is_correct = any([ans in pred for ans in processed_answers])
                is_incorrect = not is_correct
                is_refuse = False

                if is_correct:
                    correct += 1
                else:
                    incorrect += 1

            total += 1
            details[str(index)] = {
                'prompt': origin_prompt[index],
                'raw_pred': raw_pred,
                'split_done': split_done,
                'split_pred': split_pred,
                'pred': pred,
                'refr': answers,
                'is_correct': is_correct,
                'is_incorrect': is_incorrect,
                'is_refuse': is_refuse,
                'qid': qid,
                'lang': lang,
                'infer_id': infer_id,
            }

        assert total == correct + incorrect + refuse
        results = {
            'correct_ratio': correct / total * 100,
            'incorrect_ratio': incorrect / total * 100,
            'refuse_ratio': refuse / total * 100,
            'details': details
        }

        return results
