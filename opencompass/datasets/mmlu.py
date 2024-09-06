import os.path as osp
from copy import deepcopy
from typing import Union

# import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import check_is_refuse, first_option_postprocess

from .base import BaseDataset


@LOAD_DATASET.register_module()
class MMLUDataset(BaseDataset):
    @staticmethod
    def load(path: str,
             name: str,
             n_infer_dict: Union[dict, None] = None) -> DatasetDict:
        dataset = DatasetDict()
        for split in ['val', 'test']:
            filename = osp.join(path, split, f'{name}_{split}.csv')
            df = pd.read_csv(filename)
            df.fillna('None', inplace=True)
            expected_cols = [
                'id', 'lang', 'question', 'A', 'B', 'C', 'D', 'target'
            ]
            assert set(expected_cols).issubset(
                df.columns), f'Columns mismatch: {df.columns}'
            df['id_and_target'] = df.apply(
                lambda x: {
                    'id': x.id,
                    'infer_id': str(0),
                    'lang': x.lang,
                    'target': x.target,
                    'option_candidates': {
                        option: [
                            option,
                            f'{option}.',
                            f'({option})',
                            x[option].rstrip('.'),
                            x[option].rstrip('.') + '.',
                            x[option].strip(),
                            x[option].lower(),
                            f'{option}. {x[option]}',
                            f'({option}) {x[option]}',
                        ]
                        for option in 'ABCD'
                    },
                },
                axis=1,
            )

            if n_infer_dict is not None:
                n_infer = n_infer_dict[split]
                # duplicate rows for n_infer times
                # modify id_and_target['infer_id'] accordingly
                dfs = [df]
                for i in range(1, n_infer):
                    df_copy = deepcopy(df)
                    id_and_target_list = []
                    for _, row in df_copy.iterrows():
                        id_and_target = deepcopy(row['id_and_target'])
                        id_and_target['infer_id'] = str(i)
                        id_and_target_list.append(id_and_target)
                    df_copy['id_and_target'] = id_and_target_list
                    dfs.append(df_copy)
                df = pd.concat(dfs, ignore_index=True)
                df.sort_values('id', inplace=True)
                df.reset_index(drop=True, inplace=True)
                print(f'Inferencing {n_infer} times for {filename}')

            raw_data = df.to_dict(orient='records')
            dataset[split] = Dataset.from_list(raw_data)
        return dataset


@ICL_EVALUATORS.register_module()
class MMLU_Evaluator(BaseEvaluator):
    """
    ref: opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator
    """

    def score(self, predictions, references, origin_prompt) -> dict:

        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length.'}

        details = {}
        correct, total = 0, 0
        for index, (pred, id_and_ref) in enumerate(zip(predictions,
                                                       references)):
            qid = id_and_ref['id']
            ref = id_and_ref['target']
            is_correct = pred == ref
            correct += is_correct
            details[str(index)] = {
                'prompt': origin_prompt[index],
                'pred': pred,
                'refr': ref,
                'is_correct': is_correct,
                'qid': qid,
            }
            total += 1

        results = {'accuracy': correct / total * 100, 'details': details}

        return results


@ICL_EVALUATORS.register_module()
class MMLU_HonestEvaluator(BaseEvaluator):
    """
    ref: opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator
    """

    def score(self, predictions, references, origin_prompt) -> dict:

        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length.'}

        details = {}
        total, correct, incorrect, refuse, fail_parse = 0, 0, 0, 0, 0
        for index, (raw_pred,
                    id_and_ref) in enumerate(zip(predictions, references)):
            qid = id_and_ref['id']
            lang = id_and_ref['lang']
            ref = id_and_ref['target']
            infer_id = id_and_ref.get('infer_id', '0')

            if check_is_refuse(raw_pred, lang):
                is_correct = False
                is_incorrect = False
                is_refuse = True
                is_fail_parse = False
                refuse += 1
                pred = '<REFUSE>'
            else:
                is_refuse = False
                pred = first_option_postprocess(raw_pred, 'ABCD')
                if not pred:
                    is_correct = False
                    is_incorrect = False
                    is_fail_parse = True
                    fail_parse += 1
                else:
                    is_fail_parse = False
                    is_correct = pred == ref
                    is_incorrect = not is_correct
                    if is_correct:
                        correct += 1
                    else:
                        incorrect += 1

            total += 1
            details[str(index)] = {
                'prompt': origin_prompt[index],
                'raw_pred': raw_pred,
                'pred': pred,
                'refr': ref,
                'is_correct': is_correct,
                'is_incorrect': is_incorrect,
                'is_refuse': is_refuse,
                'is_fail_parse': is_fail_parse,
                'qid': qid,
                'lang': lang,
                'infer_id': infer_id,
            }

        assert total == correct + incorrect + refuse + fail_parse
        results = {
            'correct_ratio': correct / total * 100,
            'incorrect_ratio': incorrect / total * 100,
            'refuse_ratio': refuse / total * 100,
            'fail_parse_ratio': fail_parse / total * 100,
            'details': details
        }

        return results


@ICL_EVALUATORS.register_module()
class MMLU_EditDistEvaluator(BaseEvaluator):
    """
    ref:
    opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator
    opencompass.openicl.icl_evaluator.EDAccEvaluator

    该Evaluator用于评估in-context-learning条件下的base模型。
    args:
        splitters: list of str, default ['\n\n']
            由于base模型在回答完成问题后，通常不会自动停止输出，而是“编造”新的问题和答案对。
            因此，使用splitter切割模型生成的文本，以便进行评估。
    """

    def __init__(self, splitters=['\n\n'], **kwargs):
        super().__init__(**kwargs)
        self.splitters = [_.lower() for _ in splitters]

        from rapidfuzz.distance import Levenshtein
        self.dist = Levenshtein.distance

    def score(self, predictions, references, origin_prompt) -> dict:

        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length.'}

        details = {}
        total, correct, incorrect, fail_parse = 0, 0, 0, 0
        for index, (raw_pred,
                    id_and_ref) in enumerate(zip(predictions, references)):
            qid = id_and_ref['id']
            lang = id_and_ref['lang']
            ref = id_and_ref['target']
            # option_candidates = id_and_ref.get('option_candidates', [ref])
            infer_id = id_and_ref.get('infer_id', '0')

            split_pred = raw_pred
            split_done = False  # 是否切割成功，初始值为False
            for splitter in self.splitters:
                if splitter in raw_pred.lower():
                    kept = raw_pred.lower().split(splitter)[0]
                    split_pred = raw_pred[:len(kept)]
                    split_done = True  # 切割成功
                    break

            # # 如果没有切割成功，则严格匹配candidates
            # if not split_done:
            #     split_pred = split_pred.strip()
            #     pred = '<FAIL_PARSE>'
            #     for option, candidates in option_candidates.items():
            #         if split_pred in candidates:
            #             pred = option
            #             break

            #     if pred not in option_candidates:
            #         is_fail_parse = True
            #         is_correct = False
            #         is_incorrect = False
            #         fail_parse += 1
            #     else:
            #         is_fail_parse = False
            #         is_correct = pred == ref
            #         is_incorrect = not is_correct
            #         if is_correct:
            #             correct += 1
            #         else:
            #             incorrect += 1
            # else:  # 如果切割成功，则使用编辑距离进行匹配
            #     edit_dists = dict()
            #     for option, candidates in option_candidates.items():
            #         cur_dist = np.min(
            #             [self.dist(split_pred, cand) for cand in candidates])
            #         edit_dists[option] = cur_dist

            #     pred = min(edit_dists, key=edit_dists.get)

            pred = first_option_postprocess(split_pred, 'ABCD')
            if not pred:
                is_correct = False
                is_incorrect = False
                is_fail_parse = True
                fail_parse += 1
            else:
                is_fail_parse = False
                is_correct = pred == ref
                is_incorrect = not is_correct
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
                'refr': ref,
                'is_correct': is_correct,
                'is_incorrect': is_incorrect,
                'is_fail_parse': is_fail_parse,
                'qid': qid,
                'lang': lang,
                'infer_id': infer_id,
            }

        assert total == correct + incorrect + fail_parse
        results = {
            'correct_ratio': correct / total * 100,
            'incorrect_ratio': incorrect / total * 100,
            'fail_parse_ratio': fail_parse / total * 100,
            'details': details
        }

        return results
