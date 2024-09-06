from mmengine import read_base

with read_base():
    from ..cmmlu.cmmlu_ppl_041cbf import cmmlu_datasets

for d in cmmlu_datasets:
    d['abbr'] = 'demo_' + d['abbr']
    d['reader_cfg']['test_range'] = '[0:4]'
