"""Synthetic tests only; real SAL/ASR results are deliberately not claimed.

Inputs: tiny arrays, SQLite fixtures and transcript mocks. Outputs: pytest checks.
Assumptions: CPU torch installed. Usage: python -m pytest tests/test_interval_correction.py.
Limitations: no audio, pretrained weights or statistically meaningful performance.
"""
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interval_correction.core import Candidate, Config, Record, apply, canonical, conflict, frame_span, generate, rule_utilities, runs, select
from interval_correction.data import load_records, make_splits, validate_partitions
from interval_correction.metrics import diagnostics, evaluate, gain, match, risk


def record(scores=None, original=(), truth=(), uid='r', group='g', split='train', dt=.1):
    scores = np.asarray(scores if scores is not None else [.1]*10)
    return Record(uid, scores, dt, len(scores)*dt, original, group, split, truth)


def test_empty_predictions_labels():
    r = record()
    pool = generate(r, Config())
    assert [c.operation for c in pool] == ['KEEP']
    assert risk((), (), Config()) == 0
    assert evaluate((), (), Config())['event_f1'] == 0
    assert diagnostics(r, pool, Config())['candidate_coverage'] is None
    assert evaluate((), ((0., .1),), Config())['missed_events'] == 1
    assert evaluate(((0., .1),), (), Config())['false_positive_events'] == 1


def test_half_open_adjacent_overlap_bounds():
    assert canonical([(-1., .2), (.1, .3), (.3, .5), (.9, 2.)], 1.) == ((0., .3), (.3, .5), (.9, 1.))
    assert canonical([(2., 3.)], 1.) == ()
    assert match(((0., .1),), ((.1, .2),), .5)[1] == []
    with pytest.raises(ValueError):
        canonical([(1., 0.)], 2.)
    with pytest.raises(ValueError):
        canonical([(0., float('nan'))], 2.)


@pytest.mark.parametrize('span,expected', [((.2, .3),(2,3)), ((.21,.29),(2,3)), ((-.1,.11),(0,2)), ((.9,2.),(9,10))])
def test_timestamp_frame_conversion(span, expected):
    assert frame_span(*span, .1, 10) == expected
    assert runs(np.array([False, True, True]), .1, .25) == ((.1, .25),)


def test_short_events_and_matching_cardinality():
    cfg = Config(short_duration=.1)
    metrics = evaluate(((0., .02),), ((0., .02),), cfg)
    assert metrics['short_event_recall'] == 1.
    assert metrics['matched_event_tiou'] == 1.
    assert evaluate(((0., 1.),), ((0., .4), (.6, 1.)), cfg)['matched_events'] == 0
    # Maximum eligible count first, even when greedy largest-IoU could differ.
    from scipy.optimize import linear_sum_assignment
    assert len(match(((0., .6), (.4, 1.)), ((0., .6), (.4, 1.)), .2)[1]) == 2


def test_insert_below_main_threshold_and_gain_sign():
    r = record([.1, .1, .45, .45, .1], truth=((.2,.4),))
    pool = generate(r, Config())
    insert = next(c for c in pool if c.operation == 'INSERT')
    assert max(r.scores) < .5
    assert gain(r, insert, Config()) > 0
    r.truth = ()
    assert gain(r, insert, Config()) < 0
    assert gain(r, pool[0], Config()) == 0


def test_candidate_generation_never_reads_gt_or_semantics():
    r = record([.1,.4,.4,.1,.8,.8], original=((.4,.6),))
    a = generate(r, Config())
    r.truth = ((0.,.6),)
    r.words = [{'status':'fn', 'truth':'malicious'}]
    assert generate(r, Config()) == a


def test_split_correct_and_wrong_gain():
    cfg = Config()
    r = record([.8,.8,.1,.1,.8,.8], original=((0.,.6),), truth=((0.,.2),(.4,.6)))
    c = next(c for c in generate(r,cfg) if c.operation == 'SPLIT')
    assert gain(r,c,cfg) > 0
    r.truth = ((0.,.6),)
    assert gain(r,c,cfg) < 0


def test_merge_correct_and_separate_events_not_mergeable():
    cfg = Config(merge_gap=.3)
    r = record([.8,.8,.4,.8,.8], original=((0.,.2),(.3,.5)), truth=((0.,.5),))
    pool = generate(r,cfg)
    c = next(c for c in pool if c.operation == 'MERGE')
    assert gain(r,c,cfg) > 0
    assert rule_utilities(r,[c],cfg)[0] > 0
    far = record([.8,.8,.05,.05,.05,.05,.8,.8], original=((0.,.2),(.6,.8)))
    assert not any(c.operation == 'MERGE' for c in generate(far,cfg))
    r.scores[2] = .05
    assert rule_utilities(r,[c],cfg)[0] == 0


def test_rules_delete_insert_keep_and_boundary():
    cfg = Config()
    r = record([.55,.55,.1,.1,.45,.45], original=((0.,.2),), truth=())
    pool = generate(r,cfg)
    after, edits = select(r,pool,rule_utilities(r,pool,cfg))
    assert {c.operation for c in edits} == {'DELETE','INSERT'}
    assert after == ((.4,.6),) or np.allclose(after, ((.4,.6),))
    r = record([.1,.4,.8,.8,.4,.1], original=((.2,.4),))
    assert any(c.operation == 'ADJUST' and np.allclose(c.after,((.1,.5),)) for c in generate(r,cfg))
    pool = generate(r,cfg)
    after, edits = select(r,pool,[0.]*len(pool))
    assert after == r.original and not edits
    assert apply(r,pool[0]) == r.original


def test_conflicting_operations_not_stacked():
    r = record([.8,.8,.1,.1,.8,.8], original=((0.,.6),))
    pool = generate(r,Config())
    delete = next(c for c in pool if c.operation == 'DELETE')
    split = next(c for c in pool if c.operation == 'SPLIT')
    assert conflict(delete,split)
    after, edits = select(r,[delete,split],[2.,1.])
    assert after == () and edits == [delete]
    assert select(r,[delete],[.01],.01)[1] == []
    with pytest.raises(ValueError):
        select(r,[delete],[float('nan')])


def test_oracle_upper_bound_and_duration():
    cfg = Config()
    r = record([.1,.45,.45,.1], truth=((.1,.3),))
    result = diagnostics(r,generate(r,cfg),cfg)
    assert result['candidate_coverage'] == 1.
    assert result['offline_oracle_bound_kind'] == 'exact_compatible_subset'
    assert result['offline_oracle_improvement_upper_bound'] >= result['offline_oracle_single_edit_gain']
    relaxed = diagnostics(r,generate(r,cfg),cfg,exact_limit=0)
    assert relaxed['offline_oracle_improvement_upper_bound'] >= result['offline_oracle_improvement_upper_bound']
    assert evaluate(((0.,1.),), ((.2,.8),), cfg)['false_positive_duration'] == pytest.approx(.4)


def test_sqlite_adapter_class_ties_grid_tail_and_readonly(tmp_path):
    cfg = dict(split='train',resolution_seconds=.16)
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    dbfile = tmp_path/'predictions.sqlite'
    logits = np.array([[0.,0.],[-1.,1.],[2.,0.]],dtype=np.float32)
    with sqlite3.connect(dbfile) as db:
        db.execute('CREATE TABLE predictions(uid TEXT, ordinal INT, logits BLOB, labels BLOB, seconds REAL)')
        db.execute('INSERT INTO predictions VALUES(?,?,?,?,?)',('r',0,logits.tobytes(),np.array([0,1,0],dtype=np.uint8).tobytes(),.43))
    before = dbfile.read_bytes()
    source = tmp_path/'groups.jsonl'
    source.write_text(json.dumps(dict(uid='r',source_group='s',split='train'))+'\n')
    r = load_records(tmp_path,source)[0]
    assert r.scores[0] == .5 and r.original == ((0.,.16),(.32,.43))
    assert dbfile.read_bytes() == before
    cfg['resolution_seconds'] = .02
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='Grid'):
        load_records(tmp_path,source)


def test_nested_group_splits_and_no_leakage():
    records = [record(uid=str(i),group=str(i//2)) for i in range(30)]
    m = make_splits(records,42,validation_groups=2,sizes=(4,10,20))
    validate_partitions(m)
    assert set(m['train_subsets']['4']['ids']) <= set(m['train_subsets']['10']['ids']) <= set(m['train_subsets']['20']['ids'])
    assert m == make_splits(records,42,validation_groups=2,sizes=(4,10,20))
    assert not m['official_test_available']
    records[0].split = 'test'
    with pytest.raises(ValueError,match='leakage'):
        make_splits(records)


def test_semantic_fallback_no_targets_and_input_dimension():
    from interval_correction.model import ACOUSTIC_DIM, SEMANTIC_DIM, features, semantic_features
    r = record([.4,.4,.1])
    c = generate(r,Config())[1]
    assert features(r,c).shape == (ACOUSTIC_DIM,)
    assert features(r,c,True).shape == (ACOUSTIC_DIM+SEMANTIC_DIM,)
    assert not semantic_features(r,c.region).any()
    r.words = [dict(text='not',start=0.,end=.2,primary_type='negation',confidence=.8)]
    assert semantic_features(r,c.region)[0] == 1
    r.words[0]['end'] = 100.
    assert not semantic_features(r,c.region).any()
    r.words[0]['status'] = 'fn'
    with pytest.raises(ValueError,match='forbidden'):
        features(r,c,True)


def test_gain_model_smoke_reproducibility_and_leakage(tmp_path):
    from interval_correction.model import Predictor, train
    cfg = Config()
    train_rows = [record([.45,.45,.1,.8,.8],truth=((0.,.2),),uid='a',group='a')]
    dev_rows = [record([.45,.45,.1,.8,.8],truth=((0.,.2),),uid='b',group='b',split='dev')]
    for r in train_rows+dev_rows:
        r.words = [dict(text='one',start=0.,end=.2,primary_type='quantity')]
    for semantic in (False,True):
        out = tmp_path/str(semantic)
        summary = train(train_rows,dev_rows,cfg,out,semantic=semantic,epochs=2)
        assert summary['epochs_completed'] == 2
        predictor = Predictor(out/'model.pt',cfg,semantic)
        pool = generate(dev_rows[0],cfg)
        values = predictor.predict(dev_rows[0],pool)
        assert values[0] == 0. and len(values) == len(pool) and np.isfinite(values).all()
        with pytest.raises(ValueError,match='mismatch'):
            Predictor(out/'model.pt',Config(merge_gap=.1),semantic)
    out2 = tmp_path/'repeat'
    train(train_rows,dev_rows,cfg,out2,epochs=2)
    assert (tmp_path/'False'/'training.jsonl').read_text() == (out2/'training.jsonl').read_text()
    with pytest.raises(ValueError,match='leakage'):
        train(train_rows,train_rows,cfg,tmp_path/'bad')
    dev_rows[0].split = 'test'
    with pytest.raises(ValueError,match='test'):
        train(train_rows,dev_rows,cfg,tmp_path/'bad')


def test_semantic_align_direct_call_rejects_bad_timestamp():
    from semantic.analysis import align, settings
    with pytest.raises(ValueError,match='invalid word interval'):
        align([dict(text='no',start=.2,end=.1)],np.array([[1.,0.]]),np.array([0]),.16,.16,settings())


def test_existing_annotator_output_is_d_compatible(tmp_path):
    from semantic.analysis import annotate_file, settings
    from interval_correction.model import features
    src = tmp_path/'raw.jsonl'
    dst = tmp_path/'annotated.jsonl'
    src.write_text(json.dumps(dict(uid='r',language='en',words=[dict(text='not',start=0.,end=.2,confidence=.8)]))+'\n')
    annotate_file(src,dst,settings())
    data = tmp_path/'records.jsonl'
    data.write_text(json.dumps(dict(uid='r',scores=[.4,.4,.1],dt=.1,seconds=.3,original=[],truth=[],source_group='g',split='train'))+'\n')
    r = load_records(data,semantic_path=dst)[0]
    assert features(r,generate(r,Config())[1],True)[-10] == 1.
