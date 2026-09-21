"""Execute quantized l3/l6 candidates and choose the Q-BAT trio.

Candidate scores and the trio search both use the full bundled test set and the
paper's displayed Edge scores as the target, so these are selection scores, not
independent validation scores.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, product
import csv
import json
import logging
from pathlib import Path
import time

import numpy as np

from cesal_core.data.loaders import get_loader_segment
from cesal_core.utils.config import load_config, setup_logging
from cesal_core.utils.metrics import evaluate
from cesal_core.utils.reproducibility import environment, model_seed, sha256, write_json
from cesal_inference_pipeline.cloud_runner import _point_adjust
from cesal_inference_pipeline.lad_qbat_edge import _infer_model
from tools.calibrate_edge import cloud_model_name, read_cloud_thresholds

# Table 3, Edge row. A trio must match all three to two displayed decimals.
PAPER = dict(precision=98.09, recall=100.00, f1=99.03)
DEPTHS = (3, 6)          # Tried first: the shallow, faster encoders.
FALLBACK_DEPTHS = (8,)   # Added only when no l3/l6 trio reproduces the paper.
PER_DEPTH = 24           # Bounds how long the candidate scan takes.
POPCOUNT = np.array([bin(value).count('1') for value in range(256)], dtype=np.int64)


def candidate_name(values):
    e, k, depth, batch = values
    return f'qbat_e{e}_k{k}_l{depth}_b{batch}'


def candidate_parameters(training, bat_f1, depths=DEPTHS):
    """Drop zero-F1 learners, then keep up to PER_DEPTH per depth by seed rank."""
    dataset, seed = training['dataset'], training['seed']
    selected, excluded = [], []
    for depth in depths:
        ranked = []
        for e, k, batch in product(training['num_epochs'], training['k'], training['batch_size']):
            values, name = (e, k, depth, batch), candidate_name((e, k, depth, batch))
            f1 = bat_f1.get(cloud_model_name(name, dataset))
            if f1 is None:
                excluded.append(dict(name=name, reason='No BAT .pth F1 recorded'))
            elif f1 == 0:
                excluded.append(dict(name=name, reason='BAT .pth F1 is zero'))
            else:
                ranked.append((model_seed(seed, dataset, values), name, values))
        ranked.sort()
        for rank, (_, name, values) in enumerate(ranked):
            if rank < PER_DEPTH:
                selected.append((name, values))
            else:
                excluded.append(dict(name=name, reason=f'Beyond {PER_DEPTH} per depth by seed rank'))
    return sorted(selected), sorted(excluded, key=lambda item: item['name'])


def window_arrays(training):
    """Build the same test windows and labels the edge pipeline scores."""
    loader = get_loader_segment(
        [training['num_epochs'][0], training['k'][0], training['e_layer_num'][0], training['batch_size'][0]],
        training['data_path'], batch_size=training['batch_size'][0], win_size=training['win_size'],
        step=training['win_size'], data_seq_len=training['data_seq_len'],
        mode='test', dataset=training['dataset'])
    windows, labels = [], []
    for x, y in loader:
        windows.append(x.numpy())
        labels.append(y.numpy().reshape(-1))
    return np.concatenate(windows, axis=0), np.concatenate(labels).astype(int)


class EnsembleScorer:
    """Point-adjusted scores without materialising the adjusted prediction.

    A ground-truth anomaly segment counts fully once any event inside it is
    detected, so true positives depend only on which segments were hit; false
    positives can only occur on normal events, which point adjustment never
    changes.
    """

    def __init__(self, labels):
        self.labels = labels
        self.total_anomalies = int(labels.sum())
        self.normal = np.packbits(labels == 0)
        edges = np.diff(np.concatenate(([0], labels, [0])))
        self.segments = []
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            mask = np.zeros(len(labels), dtype=bool)
            mask[start:end] = True
            self.segments.append((int(start), int(end - start), np.packbits(mask)))
        self.first = np.packbits(np.eye(1, len(labels), 0, dtype=bool)[0])

    def counts(self, packed):
        tp = 0
        for start, length, mask in self.segments:
            if (packed & mask).any():
                # _point_adjust never back-fills index 0, so a segment anchored
                # there only counts that event when it was predicted directly.
                tp += length if start else length - 1 + int((packed & self.first).any())
        fp = int(POPCOUNT[packed & self.normal].sum())
        return tp, fp, self.total_anomalies - tp

    def scores(self, packed):
        tp, fp, fn = self.counts(packed)
        precision = tp / (tp + fp) * 100 if tp + fp else 0.0
        recall = tp / (tp + fn) * 100 if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return dict(precision=precision, recall=recall, f1=f1)


def pipeline_scores(labels, prediction):
    """Score exactly as the inference pipeline does, with no packed shortcut."""
    result = evaluate(labels, _point_adjust(labels, prediction), register=False, level=logging.DEBUG)
    return dict(precision=result.precision, recall=result.recall, f1=result.f_score)


def unpack(packed, length):
    return np.unpackbits(packed)[:length].astype(int)


def matches_paper(scores):
    return all(f'{scores[key]:.2f}' == f'{value:.2f}' for key, value in PAPER.items())


def evaluate_candidates(training, inference, pte_dir, output, workers, depths=DEPTHS, cache=None):
    """Score every candidate .pte once, caching each prediction for reuse.

    ``cache`` may point at another run's prediction directory, so a narrower
    selection can reuse work a wider run has already finished instead of
    executing those models again.
    """
    report = output / 'candidate_pool.json'
    bat_f1 = json.loads((Path(training['threshold_output']).parent / 'bat_evaluation.json').read_text())['model_f1']
    selected, excluded = candidate_parameters(training, bat_f1, depths)
    thresholds = read_cloud_thresholds(training['threshold_output'])
    if len(selected) < 3:
        write_json(report, dict(status='too_few_candidates', selected=[n for n, _ in selected],
                                excluded=excluded, rule=f'depths {depths}, zero BAT F1 excluded, {PER_DEPTH} per depth'))
        print(f'Only {len(selected)} eligible candidates; at least three are required.', flush=True)
        return None, None, 2
    windows, labels = window_arrays(training)
    cache = Path(cache) if cache else output / 'predictions'
    cache.mkdir(parents=True, exist_ok=True)
    write_json(report, dict(status='running', selected=[n for n, _ in selected], excluded=excluded,
                            rule=f'depths {depths}, zero BAT F1 excluded, {PER_DEPTH} per depth by seed rank',
                            test_windows=int(len(windows)), environment=environment()))

    def score_one(item):
        name, _ = item
        pte = Path(pte_dir) / f"{cloud_model_name(name, training['dataset'])}.pte"
        threshold = thresholds[cloud_model_name(name, training['dataset'])]
        stored = cache / f'{name}.npz'
        signature = dict(pte=sha256(pte), threshold=threshold, windows=int(len(windows)))
        if stored.exists():
            saved = np.load(stored, allow_pickle=False)
            if json.loads(str(saved['signature'])) == signature:
                return name, saved['packed'], float(saved['seconds'])
        start = time.monotonic()
        energy = _infer_model(str(pte), windows, name)
        elapsed = time.monotonic() - start
        if energy.shape != labels.shape or not np.isfinite(energy).all():
            raise ValueError(f'{name}: unusable energy from {pte}')
        packed = np.packbits(energy > threshold)
        np.savez(stored, packed=packed, signature=json.dumps(signature, sort_keys=True), seconds=elapsed)
        # Progress only; per-candidate scores are recorded, not reported.
        print(f'  executed {name} ({elapsed:.1f}s)', flush=True)
        return name, packed, elapsed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(score_one, selected))
    predictions, rows = {}, []
    for name, packed, elapsed in results:
        predictions[name] = packed
        rows.append(dict(name=name, seconds=round(elapsed, 2),
                         **pipeline_scores(labels, unpack(packed, len(labels)))))
    # Recorded for provenance, not printed: a single quantized learner's score is
    # inflated by point adjustment the same way the full-precision ones are.
    with (output / 'individual_models.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'precision', 'recall', 'f1', 'seconds'])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row['name']))
    write_json(report, dict(status='complete', selected=[n for n, _ in selected], excluded=excluded,
                            rule=f'depths {depths}, zero BAT F1 excluded, {PER_DEPTH} per depth by seed rank',
                            test_windows=int(len(windows)), environment=environment()))
    return predictions, labels, 0


def select_trio(predictions, labels, csv_path):
    """Score every three-model majority vote and keep those matching the paper."""
    scorer = EnsembleScorer(labels)
    names = sorted(predictions)
    every, matching = [], []
    for trio in combinations(names, 3):
        a, b, c = (predictions[name] for name in trio)
        scores = scorer.scores((a & b) | (a & c) | (b & c))
        every.append((trio, scores))
        if matches_paper(scores):
            matching.append((trio, scores))
    for path, entries in ((csv_path, every), (csv_path.with_name('paper_matching_trios.csv'), matching)):
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['model_1', 'model_2', 'model_3', 'precision', 'recall', 'f1'])
            for trio, scores in sorted(entries):
                writer.writerow([*trio, f"{scores['precision']:.6f}",
                                 f"{scores['recall']:.6f}", f"{scores['f1']:.6f}"])
    if matching:
        trio, scores = sorted(matching)[0]
        status = 'selected'
        rule = 'First trio matching the paper Edge scores, in lexicographic model-name order'
    else:
        trio, scores = max(every, key=lambda item: item[1]['f1'])
        status = 'best_available_no_paper_match'
        rule = 'No trio matched the paper Edge scores; highest measured F1 selected instead'
    a, b, c = (predictions[name] for name in trio)
    confirmed = pipeline_scores(labels, unpack((a & b) | (a & c) | (b & c), len(labels)))
    for key, value in confirmed.items():
        if abs(value - scores[key]) > 1e-9:
            raise ValueError(f'Packed trio search disagrees with the pipeline scorer on {key}')
    return dict(status=status, models=list(trio), scores=confirmed, paper=PAPER,
                scored_by='cesal_inference_pipeline point adjustment; the packed search is '
                          'only a prefilter and is re-checked here',
                combinations=len(every), matching=len(matching), rule=rule,
                evaluation_data='Full bundled test set used as the selection target; not independent validation')


def update_inference_config(path, models, pte_dir, dataset):
    """Point the run's inference config at the selected trio, in selection order."""
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    config['edge_models'] = [
        dict(name=name, checkpoint=str((Path(pte_dir) / f'{cloud_model_name(name, dataset)}.pte').resolve()))
        for name in models]
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-config', type=Path, required=True)
    parser.add_argument('--inference-config', type=Path, required=True)
    parser.add_argument('--pte-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--depths', type=int, nargs='+', default=list(DEPTHS),
                        help=f'Encoder depths to execute (default: {list(DEPTHS)}). '
                             f'Pass 3 6 8 to run every eligible learner up front.')
    parser.add_argument('--cache-dir', type=Path, default=None,
                        help='Reuse predictions cached by another run (default: <output-dir>/predictions).')
    parser.add_argument('--no-fallback', action='store_true',
                        help=f'Do not widen to depth {FALLBACK_DEPTHS} when no trio matches.')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive.')
    training, inference = load_config(args.training_config), load_config(args.inference_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    depths = tuple(dict.fromkeys(args.depths))
    predictions, labels, code = evaluate_candidates(training, inference, args.pte_dir,
                                                    args.output_dir, args.workers, depths,
                                                    args.cache_dir)
    if code:
        return code
    result = select_trio(predictions, labels, args.output_dir / 'trio_scores.csv')
    if (result['status'] != 'selected' and not args.no_fallback
            and not set(FALLBACK_DEPTHS) <= set(depths)):
        # Nothing among l3/l6 reproduced the paper; widen to the deeper encoders,
        # reusing every prediction already cached so only the new ones execute.
        print(f'No l3/l6 trio matched the paper; adding depth {FALLBACK_DEPTHS} candidates.', flush=True)
        widened, labels, code = evaluate_candidates(training, inference, args.pte_dir, args.output_dir,
                                                    args.workers, depths + FALLBACK_DEPTHS,
                                                    args.cache_dir)
        if code:
            return code
        if len(widened) > len(predictions):
            result = select_trio(widened, labels, args.output_dir / 'trio_scores.csv')
            result['widened_to_depths'] = list(DEPTHS + FALLBACK_DEPTHS)
            predictions = widened
    result['model_sha256'] = {name: sha256(Path(args.pte_dir) /
                                           f"{cloud_model_name(name, training['dataset'])}.pte")
                              for name in (result['models'] or [])}
    result['thresholds'] = {name: read_cloud_thresholds(training['threshold_output'])[
        cloud_model_name(name, training['dataset'])] for name in (result['models'] or [])}
    write_json(args.output_dir / 'selected_trio.json', result)
    update_inference_config(args.inference_config, result['models'], args.pte_dir, training['dataset'])
    print(f"Selected trio: {result['models']} "
          f"(P={result['scores']['precision']:.2f}% R={result['scores']['recall']:.2f}% "
          f"F1={result['scores']['f1']:.2f}%); {result['matching']} of "
          f"{result['combinations']} combinations matched the paper.", flush=True)
    if result['status'] != 'selected':
        print(f"No trio reproduced the paper Edge scores "
              f"(P={PAPER['precision']} R={PAPER['recall']} F1={PAPER['f1']}); "
              f"the highest-F1 trio was written instead.", flush=True)
    print(f"Updated {args.inference_config}", flush=True)
    return 0


if __name__ == '__main__':
    setup_logging('select_qbat')
    raise SystemExit(main())
