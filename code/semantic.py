# common
"""Portable I/O; CLI paths are relative to code/, independent of cwd."""
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from project_config import RESULT_ROOT, DATA_ROOT, artifact_relative


def path(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (CODE_ROOT / p).resolve()


def relative(value):
    return artifact_relative(value, CODE_ROOT, DATA_ROOT, RESULT_ROOT)


def read_json(value):
    return json.loads(Path(value).read_text(encoding='utf-8-sig'))


def write_json(value, obj):
    Path(value).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_jsonl(value):
    with Path(value).open(encoding='utf-8-sig') as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f'{value}:{lineno}: invalid JSON') from exc


def write_csv(value, rows, fields):
    with Path(value).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def digest(value):
    h = hashlib.sha256()
    with Path(value).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def database(run):
    db = sqlite3.connect((Path(run) / 'predictions.sqlite').as_uri() + '?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    return db


def settings(value=None):
    document = read_json(path(value) if value else CODE_ROOT/'config.json')
    if 'semantic' not in document:
        raise ValueError('Use the unified code/config.json with a semantic section')
    config = dict(document['semantic'])
    config.setdefault('data_profile', 'reduced')
    if config['data_profile'] != 'reduced':
        raise ValueError('Semantic analysis requires reduced data; full/fixed10 artifacts are historical only')
    if not 0 < config['word_spoof_overlap'] <= 1:
        raise ValueError('word_spoof_overlap must be in (0, 1]')
    if not 0 <= config['minimum_word_confidence'] <= 1:
        raise ValueError('minimum_word_confidence must be in [0, 1]')
    if config['minimum_count'] < 1 or config['smoothing_strength'] <= 0:
        raise ValueError('Invalid minimum_count or smoothing_strength')
    if set(config['type_priority']) != {'negation','quantity','time','person','place','organization','other'}:
        raise ValueError('type_priority must contain the seven supported categories exactly once')
    if len(config['type_priority']) != 7:
        raise ValueError('Duplicate category in type_priority')
    if config['type_priority'][-1] != 'other':
        raise ValueError('other must be last in type_priority')
    return config


# annotation
"""Word timestamps, lexical tags, optional POS / named entities. No labels used."""
import math
import re
import unicodedata

ENTITY_TYPES = {'PERSON': 'person', 'PER': 'person', 'GPE': 'place', 'LOC': 'place',
                'FAC': 'place', 'ORG': 'organization', 'DATE': 'time', 'TIME': 'time',
                'CARDINAL': 'quantity', 'ORDINAL': 'quantity', 'QUANTITY': 'quantity',
                'MONEY': 'quantity', 'PERCENT': 'quantity'}


def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower().replace('’', "'")
    return re.sub(r"^[^\w]+|[^\w]+$", '', text)


def validate_record(record):
    if not isinstance(record.get('uid'), str) or not record['uid']:
        raise ValueError('Every record requires a nonempty uid')
    if not isinstance(record.get('words'), list):
        raise ValueError(f"{record['uid']}: words must be a list (empty ASR output is allowed)")
    previous_end = 0.0
    for word in record['words']:
        if not isinstance(word.get('text'), str) or not word['text'].strip():
            raise ValueError('Each word needs nonempty text')
        start, end = float(word['start']), float(word['end'])
        if not all(map(math.isfinite, (start, end))) or start < 0 or end <= start:
            raise ValueError(f"{record['uid']}: invalid word interval {start}, {end}")
        if start < previous_end - 1e-6:
            raise ValueError(f"{record['uid']}: words overlap or are not time ordered")
        previous_end = end
        confidence = word.get('confidence')
        if confidence is not None and (not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1):
            raise ValueError('Word confidence must be null or in [0,1]')


def annotate(record, config, nlp=None):
    validate_record(record)
    result = dict(record)
    words = [dict(word) for word in record['words']]
    language = record.get('language', config['language'])
    if language not in ('en', 'eng') and nlp is None:
        raise ValueError('Built-in lexical rules are English only; supply a matching NLP model/annotations')
    offsets, cursor = [], 0
    text = ' '.join(word['text'].strip() for word in words)
    for word in words:
        length = len(word['text'].strip())
        offsets.append((cursor, cursor + length))
        cursor += length + 1
    doc = nlp(text) if nlp is not None and words else None
    lexicons = {k: set(v) for k, v in config['lexicons'].items()}
    for i, word in enumerate(words):
        word['start'], word['end'] = float(word['start']), float(word['end'])
        if word.get('confidence') is not None:
            word['confidence'] = float(word['confidence'])
        key = normalize(word['text'])
        tags = set(word.get('types', [])) - {'other'}
        if not tags <= set(config['type_priority']):
            raise ValueError(f'Unsupported semantic type: {tags}')
        sources = list(word.get('tag_sources', []))
        if language in ('en', 'eng'):
            for category, vocabulary in lexicons.items():
                if key in vocabulary:
                    tags.add(category)
                    sources.append('lexicon:' + category)
            if key.endswith("n't"):
                tags.add('negation')
                sources.append('lexicon:negation')
            if re.search(r'\d', key):
                tags.add('quantity')
                sources.append('rule:contains_digit')
        if doc is not None:
            start, end = offsets[i]
            tokens = [t for t in doc if t.idx < end and t.idx + len(t.text) > start]
            if tokens:
                token = max(tokens, key=lambda t: min(end, t.idx+len(t.text))-max(start, t.idx))
                word['pos'] = token.pos_ or 'UNKNOWN'
                word['lemma'] = token.lemma_ or key
            for entity in doc.ents:
                if entity.start_char < end and entity.end_char > start and entity.label_ in ENTITY_TYPES:
                    tags.add(ENTITY_TYPES[entity.label_])
                    sources.append('ner:' + entity.label_)
        word.update(normalized=key, types=sorted(tags) or ['other'],
                    primary_type=next(t for t in config['type_priority'] if t in tags or t == 'other'),
                    tag_sources=sorted(set(sources)) or ['unclassified'],
                    pos=word.get('pos', 'UNKNOWN'))
        # 'other' means unclassified under this extractor, not proof of no semantics.
    result.update(words=words, language=language,
                  annotation_method='lexicon+ner' if nlp is not None else record.get('annotation_method','lexicon+provided_types'))
    return result


# alignment
"""Align words to original saved grid; distinguish time errors from event errors."""
import numpy as np

EPS = 1e-8
REGIONS = ('true_spoof', 'true_bonafide', 'true_positive', 'false_negative',
           'false_positive', 'true_negative', 'complete_miss_event', 'no_overlap_false_alarm_event')


def event_ranges(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8))).reshape(-1, 2)


def masks(truth, predicted):
    regions = dict(true_spoof=truth, true_bonafide=~truth,
                   true_positive=truth & predicted, false_negative=truth & ~predicted,
                   false_positive=~truth & predicted, true_negative=~truth & ~predicted)
    missed, false_alarm = np.zeros(len(truth), bool), np.zeros(len(truth), bool)
    for start, end in event_ranges(truth):
        if not predicted[start:end].any():
            missed[start:end] = True
    for start, end in event_ranges(predicted):
        if not truth[start:end].any():
            false_alarm[start:end] = True
    regions.update(complete_miss_event=missed, no_overlap_false_alarm_event=false_alarm)
    return regions


def align(words, logits, labels, seconds, dt, config):
    if not np.isfinite(seconds) or seconds <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid audio duration or evaluation grid')
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.shape != (len(labels), 2) or not np.isfinite(logits).all() or not np.isin(labels, [0,1]).all():
        raise ValueError('Expected finite Nx2 logits and binary labels; 0=spoof')
    edges = np.minimum(np.arange(len(labels)+1)*dt, seconds)
    valid_cells = np.diff(edges) > EPS
    if not valid_cells.any():
        raise ValueError('No scored audio')
    n = int(valid_cells.sum())
    labels, logits, edges = labels[:n], logits[:n], edges[:n+1]
    truth, predicted = labels == 0, logits.argmax(axis=1) == 0
    region_masks = masks(truth, predicted)
    probabilities = 1/(1+np.exp(np.clip(logits[:,1].astype(float)-logits[:,0], -700, 700)))
    threshold = config['word_spoof_overlap']
    rows = []
    for i, word in enumerate(words):
        start, end = float(word['start']), float(word['end'])
        duration = end-start
        overlaps = np.maximum(0, np.minimum(edges[1:], end)-np.maximum(edges[:-1], start))
        scored = float(overlaps.sum())
        fractions = {name: float(overlaps[mask].sum()/duration) for name, mask in region_masks.items()}
        tf, pf = fractions['true_spoof'], float(overlaps[predicted].sum()/duration)
        confidence = word.get('confidence')
        full_grid = scored >= duration-EPS
        reliable = confidence is None or float(confidence) >= config['minimum_word_confidence']
        partial_truth = EPS < tf < 1-EPS
        partial_prediction = EPS < pf < 1-EPS
        truth_word, predicted_word = tf >= threshold-EPS, pf >= threshold-EPS
        outcome = ('tp' if predicted_word else 'fn') if truth_word else ('fp' if predicted_word else 'tn')
        # Partial GT coverage is reported, but not forced into the main word-rate denominator.
        status = 'unscored' if not full_grid else 'low_confidence' if not reliable else 'partial_truth' if partial_truth else outcome
        rows.append(dict(word_index=i, text=word['text'], normalized=word['normalized'],
                         primary_type=word['primary_type'], types='|'.join(word['types']), pos=word['pos'],
                         start=start, end=end, duration=duration,
                         confidence=confidence, confidence_missing=confidence is None,
                         scored_fraction=min(1.0, scored/duration),
                         predicted_spoof_fraction=pf, partial_truth=partial_truth,
                         partial_prediction=partial_prediction, status=status,
                         threshold_status=outcome if full_grid and reliable else status,
                         analysis_eligible=full_grid and reliable,
                         mean_spoof_score=float(np.dot(overlaps,probabilities)/scored) if scored else None,
                         tag_sources='|'.join(word['tag_sources']),
                         **{name+'_fraction': value for name,value in fractions.items()}))
    widths = np.diff(edges)
    covered = np.zeros(n, float)
    for word in words:
        covered += np.maximum(0, np.minimum(edges[1:],word['end'])-np.maximum(edges[:-1],word['start']))
    covered = np.minimum(covered, widths)
    scope = 'partial' if truth.any() and not truth.all() else 'full_spoof' if truth.all() else 'genuine'
    audit = dict(scope=scope, audio_seconds=seconds, scored_seconds=float(widths.sum()),
                 transcript_covered_seconds=float(covered.sum()), words=len(words),
                 **{name+'_seconds':float(widths[mask].sum()) for name,mask in region_masks.items()},
                 **{name+'_without_words_seconds':float((widths-covered)[mask].sum()) for name,mask in region_masks.items()})
    return rows, audit


# statistics
"""Descriptive counts and denominators; frozen priors are a separate artifact."""
from collections import Counter, defaultdict
import math

STATUSES = ('tp','fn','fp','tn','partial_truth','low_confidence','unscored')


def wilson(success, total):
    if not total:
        return None, None
    z = 1.959963984540054
    p, d = success/total, 1+z*z/total
    center = (p+z*z/(2*total))/d
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return max(0,center-half), min(1,center+half)


def ratios(row):
    fake, real = row['tp']+row['fn'], row['tn']+row['fp']
    for name, a, b in [('spoof_rate',fake,fake+real),('miss_rate',row['fn'],fake),('false_alarm_rate',row['fp'],real)]:
        row[name] = a/b if b else None
        row[name+'_denominator'] = b
        row[name+'_ci95_low'], row[name+'_ci95_high'] = wilson(a,b)
    return row


class Accumulator:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.regions = defaultdict(Counter)
        self.region_totals = defaultdict(Counter)

    def add(self, word, scope):
        dimensions = [('word',word['normalized']),('primary_type',word['primary_type']),('pos',word['pos']),
                      *[('multi_type',t) for t in word['types'].split('|')]]
        duration = word['duration']
        bucket = '<=0.16' if duration <= .16 else '0.16-0.32' if duration <= .32 else '0.32-0.64' if duration <= .64 else '>0.64'
        dimensions.append(('primary_type_word_duration',word['primary_type']+'|'+bucket))
        for section in ('all',scope):
            for dimension, value in dimensions:
                c = self.counts[section,dimension,value]
                c['total'] += 1
                c[word['status']] += 1
                c['confidence_missing'] += int(word['confidence_missing'])
                c['partial_prediction'] += int(word['partial_prediction'])
                if word['analysis_eligible']:
                    c['threshold_'+word['threshold_status']] += 1
            if word['analysis_eligible']:
                for region in REGIONS:
                    fraction = word[region+'_fraction']
                    if fraction <= 1e-8:
                        continue
                    key = section,region,word['primary_type']
                    self.regions[key]['touched_words'] += 1
                    self.regions[key]['fractional_word_count'] += fraction
                    self.regions[key]['overlap_seconds'] += fraction*duration
                    self.region_totals[section,region]['touched_words'] += 1
                    self.region_totals[section,region]['fractional_word_count'] += fraction

    def rows(self):
        result = []
        for (scope,dimension,value),c in sorted(self.counts.items()):
            row = dict(scope=scope,dimension=dimension,value=value,
                       **{k:c[k] for k in ('total',*STATUSES,'confidence_missing','partial_prediction')})
            ratios(row)
            # Sensitivity view includes partial-ground-truth words at the fixed overlap threshold.
            for status in ('tp','fn','fp','tn'):
                row['threshold_'+status] = c['threshold_'+status]
            result.append(row)
        return result

    def region_rows(self):
        result = []
        for (scope,region,category),c in sorted(self.regions.items()):
            total = self.region_totals[scope,region]
            result.append(dict(scope=scope,region=region,primary_type=category,**dict(c),
                               touched_word_share=c['touched_words']/total['touched_words'],
                               fractional_word_share=c['fractional_word_count']/total['fractional_word_count']))
        return result


COUNT_FIELDS = ['scope','dimension','value','total',*STATUSES,'confidence_missing','partial_prediction',
                *[x for rate in ('spoof_rate','miss_rate','false_alarm_rate')
                  for x in (rate,rate+'_denominator',rate+'_ci95_low',rate+'_ci95_high')],
                'threshold_tp','threshold_fn','threshold_fp','threshold_tn']
REGION_FIELDS = ['scope','region','primary_type','touched_words','fractional_word_count','overlap_seconds',
                 'touched_word_share','fractional_word_share']


def fit_priors(rows, config):
    """Caller must enforce correction-training provenance before saving these."""
    rows = [r for r in rows if r['scope']=='all' and r['dimension'] in ('word','primary_type')]
    def counts(r):
        return {k:int(r[k]) for k in ('tp','fn','fp','tn')}
    totals = Counter()
    for r in rows:
        if r['dimension']=='primary_type':
            totals.update(counts(r))
    def feature(c, parent):
        fake, real = c['tp']+c['fn'], c['tn']+c['fp']
        a = config['smoothing_strength']
        pairs = [(fake,fake+real),(c['fn'],fake),(c['fp'],real)]
        return dict(n=fake+real, denominators=[d for _,d in pairs],
                    rates=[(v+a*p)/(d+a) for (v,d),p in zip(pairs,parent)],
                    reliable=[d>=config['minimum_count'] for _,d in pairs])
    global_feature = feature(totals,[.5,.5,.5])
    result = dict(global_statistics=global_feature, word={}, primary_type={})
    for r in rows:
        # Word and type estimates shrink to a shared global mean; lookup falls back to type.
        result[r['dimension']][r['value']] = feature(counts(r),global_feature['rates'])
    return result


def lookup(word, priors, minimum_confidence):
    lexical = priors['word'].get(word['normalized'])
    typed = priors['primary_type'].get(word['primary_type'])
    confidence = word.get('confidence')
    quality = 0 if confidence is not None and confidence < minimum_confidence else 1
    rates, sources, masks, support_counts = [], [], [], []
    for i in range(3):
        chosen = next(((name,entry) for name,entry in [('word',lexical),('type',typed)]
                       if quality and entry is not None and entry['reliable'][i]), None)
        if chosen:
            name, entry = chosen
            rates.append(entry['rates'][i]); sources.append(name); masks.append(1)
            support_counts.append(entry['denominators'][i])
        else:
            # Zero is a missing-feature placeholder, not a claim that the true rate is zero.
            rates.append(0.0); sources.append('acoustic_fallback'); masks.append(0)
            support_counts.append(0)
    return dict(spoof_rate_prior=rates[0],miss_rate_prior=rates[1],false_alarm_rate_prior=rates[2],
                prior_available=masks,prior_sources=sources,prior_support_counts=support_counts,
                lexical_seen=lexical is not None,confidence_missing=confidence is None)


# adapters
"""Optional local ASR and spaCy adapters; importing does not load/download models."""
import json
import time
from pathlib import Path


def transcribe(manifest, output, model_directory, config, device='cpu', limit=0):
    from faster_whisper import WhisperModel
    model_path = path(model_directory)
    if not model_path.is_dir():
        raise FileNotFoundError('Provide a local CTranslate2 Whisper model directory')
    model = WhisperModel(str(model_path), device=device,
                         compute_type='int8' if device=='cpu' else 'float16', local_files_only=True)
    seen = set()
    with Path(output).open('x',encoding='utf-8') as f:
        for i, item in enumerate(read_jsonl(manifest)):
            if limit and i >= limit:
                break
            if item['uid'] in seen:
                raise ValueError('Duplicate uid in audio manifest')
            seen.add(item['uid'])
            started = time.perf_counter()
            segments, info = model.transcribe(str(path(item['audio'])), language=config['language'],
                                              word_timestamps=True, vad_filter=False, beam_size=5)
            words, zero_duration = [], 0
            for segment in segments:
                for word in segment.words or []:
                    if word.end <= word.start or not word.word.strip():
                        zero_duration += 1
                        continue
                    words.append(dict(text=word.word.strip(),start=word.start,end=word.end,confidence=word.probability))
            record = dict(uid=item['uid'],audio=item['audio'],language=info.language,words=words,
                          asr_model=relative(model_path), asr_seconds=time.perf_counter()-started,
                          discarded_zero_duration_words=zero_duration)
            validate_record(record)
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n'); f.flush()
            if (i+1)%10==0:
                print(f'Transcribed {i+1}',flush=True)


def annotate_file(transcripts, output, config, ner_model=None):
    nlp = None
    if ner_model:
        import spacy
        model_path = path(ner_model)
        nlp = spacy.load(str(model_path) if model_path.exists() else ner_model)
        if 'ner' not in nlp.pipe_names:
            raise ValueError('The selected spaCy pipeline has no named-entity recognizer')
    seen = set()
    with Path(output).open('x',encoding='utf-8') as f:
        for record in read_jsonl(transcripts):
            if record['uid'] in seen:
                raise ValueError('Duplicate uid in transcripts')
            seen.add(record['uid'])
            result = annotate(record,config,nlp)
            result['ner_model'] = ner_model
            f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')


# pipeline
"""Check readiness or run the reduced-data semantic pipeline; see the project README."""
import argparse
from collections import Counter
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import time

CODE = CODE_ROOT
if str(CODE) not in sys.path:
    sys.path.insert(0,str(CODE))


ROLES = ('train_detector', 'train_edit', 'train_review', 'dev', 'eval', 'lps')
ARTIFACT_SCHEMA = 3


def role_name(value):
    role = 'train_detector' if value == 'train' else value
    if role not in ROLES:
        raise ValueError('Expected an explicit reduced role: '+', '.join(ROLES))
    return role


def role_manifest(role):
    from data import read_manifest
    role = role_name(role)
    manifest = read_manifest(role)
    ids = manifest['utterance_ids']
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate IDs in reduced manifest: '+role)
    if not manifest.get('manifest_sha256'):
        raise ValueError('Reduced manifest has no identity hash: '+role)
    return manifest


def prediction_profile(metrics, expected=None):
    profile = metrics.get('data_profile')
    if profile != 'reduced' or expected not in (None, 'reduced'):
        raise ValueError('Expected reduced predictions; old full/fixed10 results cannot be reused. Run the new reduced pipeline.')
    return profile


def prediction_context(run, expected_role=None):
    metrics = read_json(run/'metrics.json')
    prediction_profile(metrics)
    role = role_name(metrics.get('partition_role', metrics.get('role', metrics.get('split'))))
    if expected_role is not None and role != role_name(expected_role):
        raise ValueError('Prediction role does not match requested role')
    manifest = role_manifest(role)
    observed = metrics.get('manifest_sha256', metrics.get('subset_manifest_sha256'))
    if observed != manifest['manifest_sha256']:
        raise ValueError('Prediction manifest differs from the current reduced role; rerun predictions')
    checkpoint = metrics.get('checkpoint_sha256')
    if not isinstance(checkpoint, str) or not re.fullmatch('[0-9a-fA-F]{64}', checkpoint):
        raise ValueError('Predictions require a SHA256 identity for the detector checkpoint')
    if (run/'status.json').exists() and read_json(run/'status.json').get('status') != 'complete':
        raise ValueError('Use a completed prediction run')
    context = dict(role=role, data_profile='reduced', manifest_sha256=manifest['manifest_sha256'],
                   official_split=manifest.get('official_split'), detector_checkpoint_sha256=checkpoint.lower(),
                   detector_training_profile=metrics.get('training_data_profile'),
                   detector_training_split=metrics.get('training_split'),
                   detector_training_manifest_sha256=metrics.get('training_manifest_sha256'),
                   checkpoint_kind=metrics.get('checkpoint_kind'),
                   source_grouping=manifest.get('source_grouping', 'Source-group separation is not verified.'))
    return metrics, manifest, context


def prediction_records(run, manifest):
    db = database(run)
    try:
        rows = list(db.execute('SELECT uid,seconds FROM predictions ORDER BY ordinal'))
    finally:
        db.close()
    ids = [uid for uid, _ in rows]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Prediction IDs are empty or duplicated')
    if set(ids) - set(manifest['utterance_ids']):
        raise ValueError('Predictions contain recordings outside the current role manifest')
    return rows


def check():
    runs = []
    for folder in sorted(RESULT_ROOT.glob('*')):
        if (folder/'predictions.sqlite').is_file() and (folder/'metrics.json').is_file():
            try:
                metrics, _, context = prediction_context(folder)
                runs.append(dict(run=relative(folder), role=context['role'], compatible=True,
                                 grid=metrics.get('resolution_seconds')))
            except (ValueError, KeyError, FileNotFoundError) as exc:
                runs.append(dict(run=relative(folder), compatible=False, reason=str(exc)))
    models = DATA_ROOT/'models'
    return dict(prediction_runs=runs,
                packages={name:importlib.util.find_spec(name) is not None for name in ('numpy','spacy','faster_whisper')},
                local_model_entries=[p.name for p in models.iterdir()] if models.exists() else [],
                transcript_files=[relative(p) for p in (DATA_ROOT/'semantic').glob('*.jsonl')],
                next_step='Use reduced manifests. ASR/POS/NER features need no labels; frozen risk statistics require train_edit. No downloads are started.')


def make_manifest(run, output, limit=0, expected_profile='reduced', role=None):
    prediction_profile(read_json(run/'metrics.json'), expected_profile)
    _, manifest, context = prediction_context(run, role)
    records = prediction_records(run, manifest)
    if limit:
        records = records[:limit]
    from data import dataset_paths
    audio_root, _ = dataset_paths(context['role'], 'reduced')
    audio = {}
    for uid, _ in records:
        candidates = [Path(audio_root)/(uid+suffix) for suffix in ('.wav', '.flac')]
        found = [p for p in candidates if p.is_file()]
        if len(found) != 1:
            raise FileNotFoundError('Expected one reduced audio file for '+uid)
        audio[uid] = found[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        for uid, seconds in records:
            f.write(json.dumps(dict(uid=uid, audio=relative(audio[uid]), seconds=seconds,
                                   role=context['role'], data_profile='reduced',
                                   manifest_sha256=context['manifest_sha256']), ensure_ascii=False)+'\n')
    return dict(recordings=len(records), audio_manifest=relative(output), **context,
                sampling='first saved ordinal; smoke test only' if limit else 'current reduced predictions')


WORD_FIELDS = ['uid','scope','language','word_index','text','normalized','primary_type','types','pos',
               'start','end','duration','confidence','confidence_missing','scored_fraction',
               'predicted_spoof_fraction','partial_truth','partial_prediction',
               'status','threshold_status','analysis_eligible','mean_spoof_score','tag_sources',
               *[name+'_fraction' for name in REGIONS]]


def analyze(run, transcripts, output, config):
    import csv
    import numpy as np
    metrics, manifest, context = prediction_context(run)
    prediction_profile(metrics, config.get('data_profile'))
    prediction_ids = {uid for uid, _ in prediction_records(run, manifest)}
    dt = float(metrics['resolution_seconds'])
    output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    metadata = dict(status='running',purpose='retrospective_diagnostics',prior_eligible=False,
                    schema_version=ARTIFACT_SCHEMA, source_run=relative(run), split=context['role'],
                    resolution_seconds=dt, **context,
                    transcripts=relative(transcripts),transcripts_sha256=digest(transcripts),config=config,
                    method='0=spoof; saved-logit argmax (ties=spoof); word overlap threshold; no resampling',
                    rate_population='fully scored, confidence accepted, ground-truth-pure words only',
                    region_population='fully scored, confidence accepted words; partial overlap retained',
                    notes=['Types are heuristics/NER predictions, not ground-truth semantics.',
                           'other means unclassified; multi_type overlaps and must not be stacked.',
                           'Wilson intervals are descriptive token-level intervals, not recording-bootstrap evidence.',
                           'ASR omission and timestamp errors can bias all lexical results.',
                           'Descriptive test statistics must not become training priors.'])
    write_json(output/'summary.json',metadata)
    db = database(run)
    accumulator, audits, seen, languages = Accumulator(), [], set(), set()
    try:
        with (output/'word_diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=WORD_FIELDS)
            writer.writeheader()
            for record in read_jsonl(transcripts):
                uid = record['uid']
                if uid in seen:
                    raise ValueError('Duplicate transcript uid: '+uid)
                if uid not in prediction_ids:
                    raise ValueError('Transcript outside current prediction role: '+uid)
                seen.add(uid)
                row = db.execute('SELECT logits,labels,seconds FROM predictions WHERE uid=?',(uid,)).fetchone()
                if row is None:
                    raise ValueError('Transcript uid absent from predictions: '+uid)
                record = annotate(record,config)
                languages.add(record['language'])
                if len(languages)>1:
                    raise ValueError('Analyze languages separately; do not merge lexical denominators')
                logits = np.frombuffer(row[0],dtype=np.float32).reshape(-1,2)
                labels = np.frombuffer(row[1],dtype=np.uint8)
                words,audit = align(record['words'],logits,labels,float(row[2]),dt,config)
                audit.update(uid=uid,language=record['language'],annotation_method=record['annotation_method'],
                             ner_model=record.get('ner_model'),asr_model=record.get('asr_model'))
                audits.append(audit)
                for word in words:
                    word.update(uid=uid,scope=audit['scope'],language=record['language'])
                    writer.writerow(word)
                    accumulator.add(word,audit['scope'])
                if len(seen)%1000==0:
                    print(f'Analyzed {len(seen)} recordings',flush=True)
        if not seen:
            raise ValueError('Transcript file is empty')
        all_recordings = db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        rows = accumulator.rows()
        write_csv(output/'type_statistics.csv',[r for r in rows if r['dimension']!='word'],COUNT_FIELDS)
        write_csv(output/'word_statistics.csv',[r for r in rows if r['dimension']=='word'],COUNT_FIELDS)
        write_csv(output/'region_type_distribution.csv',accumulator.region_rows(),REGION_FIELDS)
        write_csv(output/'recording_coverage.csv',audits,list(audits[0]))
        write_json(output/'analyzed_uids.json',sorted(seen))
        statuses = Counter()
        for r in rows:
            if r['scope']=='all' and r['dimension']=='primary_type':
                for key in ('total','tp','fn','fp','tn','partial_truth','low_confidence','unscored','confidence_missing'):
                    statuses[key] += r[key]
        metadata.update(status='complete',analyzed_recordings=len(seen),prediction_recordings=all_recordings,
                        missing_transcript_recordings=all_recordings-len(seen),word_counts=dict(statuses),
                        manifest_recordings=len(manifest['utterance_ids']),
                        prior_eligible=(context['role']=='train_edit' and seen==set(manifest['utterance_ids'])
                            and context['detector_training_profile']=='reduced'
                            and context['detector_training_split']=='train_detector'
                            and context['checkpoint_kind']!='released_reference'
                            and context['detector_training_manifest_sha256']==role_manifest('train_detector')['manifest_sha256']),
                        language=next(iter(languages)),elapsed_seconds=time.perf_counter()-started,
                        code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
        write_json(output/'summary.json',metadata)
    except Exception as exc:
        metadata.update(status='failed',error=str(exc))
        write_json(output/'summary.json',metadata)
        raise
    finally:
        db.close()
    return dict(output=relative(output),**{key:metadata[key] for key in ('analyzed_recordings','missing_transcript_recordings','word_counts')})


def protocol_manifests():
    manifests = {role: role_manifest(role) for role in ROLES}
    seen = set()
    for role, manifest in manifests.items():
        ids = set(manifest['utterance_ids'])
        if seen & ids:
            raise ValueError('Recording IDs overlap across reduced roles: '+role)
        seen.update(ids)
    return manifests


def partition_groups(partition_manifest, manifests):
    """Optional source-group evidence; missing provenance is reported, never invented."""
    if partition_manifest is None:
        return {}
    expected = {uid: role for role, manifest in manifests.items() for uid in manifest['utterance_ids']}
    groups, group_roles = {}, {}
    for item in read_jsonl(partition_manifest):
        uid, role, group = item['uid'], role_name(item['role']), item['source_group']
        if uid in groups or not group or expected.get(uid) != role:
            raise ValueError('Partition record differs from the authoritative reduced manifests')
        if group in group_roles and group_roles[group] != role:
            raise ValueError('Source-group leakage across reduced roles: '+group)
        groups[uid], group_roles[group] = group, role
    if set(groups) != set(expected):
        raise ValueError('Source-group partition must cover every current reduced recording')
    return groups


def freeze(report, partition_manifest, output):
    """Fit label-derived priors only on train_edit under the current reduced detector."""
    meta = read_json(report/'summary.json')
    if (meta.get('schema_version') != ARTIFACT_SCHEMA or meta.get('status') != 'complete'
            or meta.get('data_profile') != 'reduced' or meta.get('role') != 'train_edit'):
        raise ValueError('Only a current reduced train_edit report can be frozen; old train/full/fixed10 statistics must be refitted')
    manifests = protocol_manifests()
    if meta.get('manifest_sha256') != manifests['train_edit']['manifest_sha256']:
        raise ValueError('train_edit manifest changed; rerun semantic analysis')
    if (meta.get('detector_training_profile') != 'reduced'
            or meta.get('detector_training_split') != 'train_detector'
            or meta.get('detector_training_manifest_sha256') != manifests['train_detector']['manifest_sha256']
            or meta.get('checkpoint_kind') == 'released_reference'):
        raise ValueError('Frozen risk requires a detector trained on the current train_detector role; released/legacy checkpoint exposure is unsuitable')
    checkpoint = meta.get('detector_checkpoint_sha256')
    if not isinstance(checkpoint, str) or not re.fullmatch('[0-9a-f]{64}', checkpoint):
        raise ValueError('Missing detector checkpoint SHA256')
    ids = read_json(report/'analyzed_uids.json')
    if len(ids) != len(set(ids)) or set(ids) != set(manifests['train_edit']['utterance_ids']):
        raise ValueError('Frozen statistics require complete train_edit transcript coverage; partial smoke reports are diagnostic only')
    groups = partition_groups(partition_manifest, manifests)
    rows = []
    for name in ('type_statistics.csv', 'word_statistics.csv'):
        with (report/name).open(encoding='utf-8-sig', newline='') as f:
            rows.extend(csv.DictReader(f))
    priors = fit_priors(rows, meta['config'])
    if priors['global_statistics']['n'] == 0:
        raise ValueError('No eligible words to fit')
    artifact = dict(schema_version=ARTIFACT_SCHEMA, data_profile='reduced', source_role='train_edit',
                    source_report=relative(report), source_uids=ids,
                    manifest_sha256={role: m['manifest_sha256'] for role, m in manifests.items()},
                    detector_checkpoint_sha256=checkpoint, config=meta['config'],
                    language=meta['language'], priors=priors,
                    source_summary_sha256=digest(report/'summary.json'), source_groups=groups,
                    partition_sha256=digest(partition_manifest) if partition_manifest else None,
                    source_grouping='Verified against supplied source-group partition' if groups else
                        'Recording IDs are disjoint; source-recording provenance is unavailable, so source-group separation is not verified.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2, allow_nan=False)
    return dict(frozen_statistics=relative(output), training_recordings=len(ids), source_role='train_edit',
                source_grouping=artifact['source_grouping'])


def features(transcripts, frozen, output, split, run=None, config=None):
    """Default exports label-free features; risk priors require matching frozen detector provenance."""
    role = role_name(split)
    manifest = role_manifest(role)
    allowed_ids = set(manifest['utterance_ids'])
    artifact, training_ids = None, set()
    config = dict(config) if config is not None else settings()
    if frozen is not None:
        artifact = read_json(frozen)
        if (artifact.get('schema_version') != ARTIFACT_SCHEMA or artifact.get('source_role') != 'train_edit'
                or artifact.get('data_profile') != 'reduced'):
            raise ValueError('Legacy risk table rejected: refit on current train_edit; no full/fixed10 fallback is permitted')
        if role in ('train_detector', 'train_edit'):
            raise ValueError('Do not export label-derived risk into a fitting role; use label-free features or a separately audited cross-fitting experiment')
        if run is None:
            raise ValueError('Frozen-risk features require --run to verify the target detector checkpoint')
        _, _, context = prediction_context(run, role)
        allowed_ids &= {uid for uid, _ in prediction_records(run, manifest)}
        if context['detector_checkpoint_sha256'] != artifact.get('detector_checkpoint_sha256'):
            raise ValueError('Frozen risk and target predictions use different detector checkpoints; refit train_edit statistics')
        manifests = protocol_manifests()
        current_hashes = {name: item['manifest_sha256'] for name, item in manifests.items()}
        if artifact.get('manifest_sha256') != current_hashes:
            raise ValueError('Reduced manifests changed since risk fitting; refit train_edit statistics')
        training_ids = set(artifact['source_uids'])
        if training_ids != set(manifests['train_edit']['utterance_ids']):
            raise ValueError('Frozen fitting IDs do not match current train_edit')
        config = artifact['config']
    groups = artifact.get('source_groups', {}) if artifact else {}
    training_groups = {groups[uid] for uid in training_ids if uid in groups}
    frozen_digest = digest(frozen) if artifact else None
    records, seen = [], set()
    for record in read_jsonl(transcripts):
        uid = record['uid']
        if uid in seen or uid not in allowed_ids or uid in training_ids:
            raise ValueError('Duplicate, wrong-role, or fitting/feature recording: '+uid)
        if groups.get(uid) in training_groups:
            raise ValueError('Source-group overlap with semantic fitting data: '+uid)
        seen.add(uid)
        record = annotate(record, config)
        if artifact and record['language'] != artifact['language']:
            raise ValueError('Prior language differs from transcript language')
        # Whitelist every inference field; never copy labels or retrospective outcomes.
        words = [dict(text=w['text'], start=w['start'], end=w['end'], pos=w['pos'],
                      types=w['types'], primary_type=w['primary_type'], confidence=w.get('confidence'))
                 for w in record['words']]
        if artifact:
            for word, annotated in zip(words, record['words']):
                word.update(lookup(annotated, artifact['priors'], config['minimum_word_confidence']))
        records.append(dict(uid=uid, role=role, data_profile='reduced', words=words,
                            manifest_sha256=manifest['manifest_sha256'], feature_mode='frozen_risk' if artifact else 'label_free',
                            frozen_sha256=frozen_digest))
    if not records:
        raise ValueError('Transcript file is empty')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    return dict(feature_records=len(seen), output=relative(output), role=role,
                feature_mode='frozen_risk' if artifact else 'label_free')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Reduced-data ASR/POS/NER diagnostics and optional train_edit risk priors')
    parser.add_argument('--config', help='Unified config.json; relative paths are relative to code/')
    subs = parser.add_subparsers(dest='command')
    subs.add_parser('check')
    p = subs.add_parser('manifest'); p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    p.add_argument('--role', choices=ROLES); p.add_argument('--limit', type=int, default=0)
    p = subs.add_parser('transcribe'); p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    p.add_argument('--asr-model', required=True); p.add_argument('--device', choices=['cpu','cuda'], default='cpu'); p.add_argument('--limit', type=int, default=0)
    p = subs.add_parser('annotate'); p.add_argument('--transcripts', required=True); p.add_argument('--output', required=True); p.add_argument('--ner-model')
    p = subs.add_parser('analyze'); p.add_argument('--run', required=True); p.add_argument('--transcripts', required=True); p.add_argument('--output')
    p = subs.add_parser('freeze'); p.add_argument('--report', required=True); p.add_argument('--output', required=True)
    p.add_argument('--partition-manifest', help='Optional complete UID/role/source_group JSONL for additional source-group checks')
    p = subs.add_parser('features'); p.add_argument('--transcripts', required=True); p.add_argument('--output', required=True)
    p.add_argument('--role', '--split', dest='role', choices=ROLES, required=True)
    p.add_argument('--frozen', help='Omit for label-free ASR/POS/NER features'); p.add_argument('--run', help='Required with --frozen')
    args = parser.parse_args(argv)
    config = settings(args.config)
    if getattr(args, 'limit', 0) < 0:
        parser.error('--limit must be nonnegative')
    command = args.command or 'check'
    if command == 'check':
        result = check()
        result['default_data_profile'] = 'reduced'
    elif command == 'manifest':
        result = make_manifest(path(args.run), path(args.output), args.limit, role=args.role)
    elif command in ('transcribe', 'annotate'):
        path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if command == 'transcribe':
            transcribe(path(args.manifest), path(args.output), args.asr_model, config, args.device, args.limit)
        else:
            annotate_file(path(args.transcripts), path(args.output), config, args.ner_model)
        result = dict(output=args.output)
    elif command == 'analyze':
        destination = path(args.output) if args.output else RESULT_ROOT/('semantic_diagnostics_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        result = analyze(path(args.run), path(args.transcripts), destination, config)
    elif command == 'freeze':
        result = freeze(path(args.report), path(args.partition_manifest) if args.partition_manifest else None, path(args.output))
    else:
        result = features(path(args.transcripts), path(args.frozen) if args.frozen else None,
                          path(args.output), args.role, path(args.run) if args.run else None, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
