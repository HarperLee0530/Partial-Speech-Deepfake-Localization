"""Synthetic fixtures only: never presented as SAL experiment results."""
import csv
import json
from pathlib import Path
import sqlite3
import shutil
import subprocess
import sys
import tempfile
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
import numpy as np
from semantic.analysis import settings, write_json, read_json, path
from semantic.analysis import annotate, validate_record
from semantic.analysis import align
from semantic.analysis import Accumulator, fit_priors, lookup
from semantic.analysis import analyze, freeze, features


class SemanticsTest(unittest.TestCase):
    def setUp(self):
        self.config = settings()

    def words(self, *items):
        return annotate(dict(uid='u',language='en',words=[dict(text=t,start=s,end=e,confidence=.99) for t,s,e in items]),self.config)['words']

    def alignment(self, words, labels, pred, seconds=None, dt=1):
        logits = np.zeros((len(pred),2),np.float32)
        logits[np.arange(len(pred)),pred] = 2
        return align(words,logits,np.array(labels),len(pred)*dt if seconds is None else seconds,dt,self.config)

    def test_all_four_outcomes_and_denominators(self):
        words = self.words(('not',0,1),('three',1,2),('hello',2,3),('today',3,4))
        rows,audit = self.alignment(words,[0,0,1,1],[0,1,0,1])
        self.assertEqual([r['status'] for r in rows],['tp','fn','fp','tn'])
        self.assertEqual(audit['scope'],'partial')
        a = Accumulator()
        for row in rows: a.add(row,'partial')
        counts = {r['value']:r for r in a.rows() if r['scope']=='all' and r['dimension']=='primary_type'}
        self.assertEqual(counts['quantity']['miss_rate'],1)
        self.assertIsNone(counts['quantity']['false_alarm_rate'])
        self.assertEqual(counts['other']['false_alarm_rate'],1)

    def test_boundary_partial_is_not_forced_into_clean_denominator(self):
        rows,_ = self.alignment(self.words(('not',.5,1.5)),[0,1],[1,1])
        self.assertEqual(rows[0]['status'],'partial_truth')
        self.assertEqual(rows[0]['threshold_status'],'fn')
        self.assertAlmostEqual(rows[0]['false_negative_fraction'],.5)

    def test_unscored_audio_tail_is_not_assumed_real(self):
        rows,_ = self.alignment(self.words(('tail',.8,1.2)),[0],[0],seconds=1.2)
        self.assertEqual(rows[0]['status'],'unscored')
        self.assertAlmostEqual(rows[0]['scored_fraction'],.5)

    def test_last_grid_cell_clips_to_audio(self):
        rows,audit = self.alignment(self.words(('last',1,1.2)),[1,0],[1,1],seconds=1.2)
        self.assertEqual(rows[0]['status'],'fn')
        self.assertAlmostEqual(audit['false_negative_seconds'],.2)

    def test_complete_miss_is_not_any_partial_miss(self):
        rows,_ = self.alignment(self.words(('one',0,1),('two',1,2)),[0,0],[0,1])
        self.assertEqual(rows[1]['false_negative_fraction'],1)
        self.assertEqual(rows[1]['complete_miss_event_fraction'],0)

    def test_false_alarm_event_requires_no_true_overlap(self):
        rows,_ = self.alignment(self.words(('word',1,2)),[0,1],[0,0])
        self.assertEqual(rows[0]['false_positive_fraction'],1)
        self.assertEqual(rows[0]['no_overlap_false_alarm_event_fraction'],0)

    def test_ties_match_sal_argmax(self):
        rows,_ = align(self.words(('tie',0,1)),np.zeros((1,2)),np.array([0]),1,1,self.config)
        self.assertEqual(rows[0]['status'],'tp')

    def test_low_confidence_and_no_word_coverage_are_visible(self):
        words = self.words(('bad',0,1)); words[0]['confidence']=.1
        rows,audit = self.alignment(words,[0,0],[1,1])
        self.assertEqual(rows[0]['status'],'low_confidence')
        self.assertEqual(audit['false_negative_without_words_seconds'],1)

    def test_invalid_word_intervals_rejected(self):
        for words in [[dict(text='x',start=0,end=0)],
                      [dict(text='x',start=0,end=2),dict(text='y',start=1,end=3)]]:
            with self.assertRaises(ValueError): validate_record(dict(uid='u',words=words))

    def test_multi_tags_keep_one_primary_bucket(self):
        result=annotate(dict(uid='u',words=[dict(text='May',start=0,end=1,types=['person','time'])]),self.config)
        self.assertEqual(result['words'][0]['primary_type'],'time')
        self.assertIn('person',result['words'][0]['types'])

    def test_ner_char_alignment_with_stub(self):
        class Entity:
            start_char,end_char,label_=0,5,'PERSON'
        class Token:
            idx,text,pos_,lemma_=0,'Alice','PROPN','Alice'
        class Doc(list):
            ents=[Entity()]
        result=annotate(dict(uid='u',words=[dict(text='Alice',start=0,end=1)]),self.config,lambda text:Doc([Token()]))
        self.assertEqual(result['words'][0]['primary_type'],'person')
        self.assertEqual(result['words'][0]['pos'],'PROPN')

    def test_paths_do_not_depend_on_working_directory(self):
        self.assertEqual(path('semantic/config.json'),Path(__file__).resolve().parents[1]/'semantic/config.json')

    def test_relocated_package_runs_from_other_directory(self):
        with tempfile.TemporaryDirectory(prefix='semantic_portable_',dir=CODE) as td:
            new_code=Path(td)/'portable package'/'code'; new_code.mkdir(parents=True)
            shutil.copytree(CODE/'semantic',new_code/'semantic',ignore=shutil.ignore_patterns('__pycache__','.venv'))
            for name in ('project_config.py','config.json'):
                shutil.copy2(CODE/name,new_code/name)
            result=subprocess.run([sys.executable,str(new_code/'semantic'/'analysis.py'),'check'],cwd=td,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['prediction_runs'],[])

    def test_entity_type_precedes_numeric_surface(self):
        result=annotate(dict(uid='u',words=[dict(text='2026',start=0,end=1,types=['time'])]),self.config)
        self.assertEqual(result['words'][0]['primary_type'],'time')
        self.assertEqual(set(result['words'][0]['types']),{'time','quantity'})

    def test_backoff_and_missing_mask(self):
        rows=[dict(scope='all',dimension='primary_type',value='negation',tp=10,fn=10,fp=2,tn=18)]
        priors=fit_priors(rows,self.config)
        word=self.words(('never',0,1))[0]
        result=lookup(word,priors,.5)
        self.assertEqual(result['prior_sources'],['type']*3)
        word['confidence']=.1
        self.assertEqual(lookup(word,priors,.5)['prior_available'],[0,0,0])

    def test_end_to_end_and_no_test_prior_leakage(self):
        with tempfile.TemporaryDirectory(prefix='semantic_test_',dir=CODE) as td:
            root=Path(td); run=root/'run'; run.mkdir()
            write_json(run/'metrics.json',dict(split='eval',resolution_seconds=1))
            write_json(run/'status.json',dict(status='complete'))
            db=sqlite3.connect(run/'predictions.sqlite')
            db.execute('CREATE TABLE predictions (uid TEXT PRIMARY KEY, ordinal INTEGER, logits BLOB, labels BLOB, seconds REAL)')
            db.execute('INSERT INTO predictions VALUES (?,?,?,?,?)',('u',0,np.array([[0,2],[2,0]],np.float32).tobytes(),np.array([0,1],np.uint8).tobytes(),2))
            db.commit();db.close()
            transcript=root/'words.jsonl'
            transcript.write_text(json.dumps(dict(uid='u',words=self.words(('not',0,1),('two',1,2))))+'\n',encoding='utf-8')
            report=root/'report'; analyze(run,transcript,report,self.config)
            summary=read_json(report/'summary.json')
            self.assertEqual(summary['word_counts']['fn'],1)
            self.assertEqual(summary['word_counts']['fp'],1)
            self.assertFalse(summary['prior_eligible'])
            with self.assertRaises(ValueError): freeze(report,root/'unused',root/'priors.json')
            # Explicit train partition allows freezing; features cannot reuse fitting IDs.
            summary['split']='train'; write_json(report/'summary.json',summary)
            manifest=root/'partitions.jsonl'
            manifest.write_text('\n'.join(json.dumps(x) for x in [dict(uid='base',source_group='g1',role='base_train'),dict(uid='u',source_group='g2',role='correction_train')]),encoding='utf-8')
            frozen=root/'frozen.json'; freeze(report,manifest,frozen)
            with self.assertRaises(ValueError): features(transcript,frozen,root/'features.jsonl')
            manifest.write_text('\n'.join(json.dumps(x) for x in [dict(uid='base',source_group='g1',role='base_train'),dict(uid='u',source_group='g1',role='correction_train')]),encoding='utf-8')
            with self.assertRaises(ValueError): freeze(report,manifest,root/'leaky.json')


if __name__=='__main__':
    unittest.main()
