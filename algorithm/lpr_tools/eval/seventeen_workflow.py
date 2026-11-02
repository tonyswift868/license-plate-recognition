"""Batch 17 adapters and frozen four-threshold candidate evaluation."""
import argparse
import gzip
import json
from pathlib import Path

from .detector_size_study import dump, key, paired_changes, sha, summary
from .runtime_acceptance import _rows, compare


def load_detector_384(model_path):
    """Keep verified checkpoint weights/threshold; apply the registered study size."""
    from lpr.plate_det import PlateDetector
    from lpr.recognize import load_corner_detector
    detector = load_corner_detector(Path(model_path))
    if not isinstance(detector, PlateDetector):
        raise ValueError('Batch 17 requires plate_det')
    return PlateDetector(detector.model,version=detector.version,image_size=384,
                         score_threshold=detector.score_threshold)


def load_box_reader_checked(model_path):
    from lpr.recognize import load_box_reader
    reader = load_box_reader(Path(model_path))
    if reader.box_pad != (.08, .12):
        raise ValueError('Batch 17 trace pad differs from checkpoint')
    return reader


def select(args):
    """Freeze fit optimum, then apply the preregistered full-data exception."""
    from lpr.arbitration import arbitrate
    from lpr.config import load_config

    from .four_threshold_search import replay_auto, score_records, search_fit, tighten_full
    from .route_split import assert_content_separation
    from .three_route_expected import export_expected
    directory = Path(args.trace_dir)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    workflow = json.loads((directory / 'workflow.json').read_text(encoding='utf-8'))
    if workflow.get('threshold_scoring_run') is not False:
        raise ValueError('candidate must come from the trace-only workflow')
    rows = _rows(directory / 'raw.jsonl.gz')
    features = {key(r['path']): r for r in _rows(directory / 'route_features.jsonl.gz')}
    split = json.loads((directory / 'split.json').read_text(encoding='utf-8'))
    parent = json.loads(Path(args.parent_split).read_text(encoding='utf-8'))
    if split['assignments'] != parent['assignments'] or split['assignment_sha256'] != parent['assignment_sha256']:
        raise ValueError('Batch 13 assignments changed')
    if len(rows) != 3100 or set(features) != {key(r['path']) for r in rows}:
        raise ValueError('trace/features must align on 3100 paths')
    assert_content_separation(split['assignments'], {p: r['image_sha256'] for p, r in features.items()})
    fit = [r for r in rows if split['assignments'][key(r['path'])] == 'fit']
    hold = [r for r in rows if split['assignments'][key(r['path'])] == 'hold']
    sources = [directory / 'raw.jsonl.gz', directory / 'route_features.jsonl.gz', directory / 'split.json',
               Path(args.config), Path(args.protocol), Path(args.supplement),
               Path(__file__), Path(__file__).with_name('four_threshold_search.py')]
    dump(output / 'selection-inputs.json', {'sources': {str(p): sha(p) for p in sources},
         'assignment_sha256': split['assignment_sha256'], 'fit': len(fit), 'hold': len(hold),
         'content_intersection': 0, 'full_gate_is_development_validation': True})
    def progress(value):
        print(json.dumps(value, ensure_ascii=False), flush=True)
    fitted = search_fit(fit, progress=progress)
    dump(output / 'fit-search.json', fitted)  # Must exist before inspecting full counts.
    tightened = tighten_full(rows, fit, fitted, progress=progress)
    dump(output / 'full-tightening.json', tightened)
    if not tightened['success']:
        dump(output / 'report.json', {'eligible': False, 'reason': tightened['reason']})
        return False
    thresholds = tightened['final']['thresholds']
    cfg = load_config(args.config)
    post = cfg.postprocess.model_copy(update=thresholds)
    # Validate the final values through the actual Pydantic config contract.
    type(post).model_validate(post.model_dump())
    fixed = {'arbitration': 'three_route', 'three_route_legacy_sr_agreement': False,
             'three_route_segment_solo': True, 'three_route_reader_solo': True,
             'three_route_province_support_min_routes': 0,
             'nonok_fallback': 'predicted_tilt_then_corrected', 'fallback_predicted_tilt_min_deg': 20.0}
    if any(getattr(post, k) != v for k, v in fixed.items()):
        raise ValueError('only the frozen 011 + T20_A family is supported')
    selected_rows = []
    with gzip.open(output / 'decisions.jsonl.gz', 'wt', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            candidates = features[key(row['path'])]['candidates']
            corners = candidates[0]['corners'] if candidates else None
            selected = arbitrate(row, post, predicted_corners=corners).as_dict()
            independent = replay_auto(row, thresholds)
            if (selected['status'] == 'OK') != (independent is not None):
                raise ValueError('independent auto/replay acceptance differs')
            if independent is not None and compare(selected, independent):
                raise ValueError('independent auto/replay fields differ')
            selected_rows.append({'path': row['path'], 'partition': row['split'],
                                  'subset': row['subset'], 'truth': row['truth'],
                                  'brightness': row['brightness'], **selected})
            stream.write(json.dumps({'path': row['path'], 'partition': row['split'], 'selected': selected},
                                    ensure_ascii=False, separators=(',', ':')) + '\n')
    expected = export_expected(directory / 'raw.jsonl.gz', output / 'decisions.jsonl.gz',
                               output / 'expected-runtime.jsonl.gz', preserve_errors=False)
    old = _rows(args.old_actual)
    old_report = json.loads(Path(args.old_report).read_text(encoding='utf-8'))
    if load_config(args.old_config).model_dump(mode='json') != old_report['config']['effective_config']:
        raise ValueError('old default differs from verified Batch 16 baseline')
    for r, previous in zip(selected_rows, old, strict=False):
        if any(r[k] != previous[k] for k in ('truth', 'partition', 'subset', 'brightness')):
            raise ValueError('old/new evaluation metadata differs')
    current_metrics = summary(selected_rows)
    limits = {'clear': .95, 'night': .85, 'tilt': .85, 'dirty': .85}
    gates = {k: current_metrics['composition_estimate'][k] >= v for k, v in limits.items()}
    report = {'eligible': all(gates.values()), 'composition_gates': gates, 'thresholds': thresholds,
              'auto': {'fit': score_records(fit, thresholds), 'hold': score_records(hold, thresholds),
                       'full': score_records(rows, thresholds)},
              'new': current_metrics, 'old': summary(old), 'paired': paired_changes(selected_rows, old),
              'expected': expected, 'independent_auto_replay_differences': 0,
              'old_sources': {str(p): sha(p) for p in (args.old_actual, args.old_report, args.old_config)}}
    dump(output / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('eligible', 'composition_gates', 'thresholds', 'auto')}, ensure_ascii=False), flush=True)
    return report['eligible']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    choice = sub.add_parser('select')
    for name in ('trace-dir', 'out', 'parent-split', 'config', 'protocol', 'supplement',
                 'old-actual', 'old-report', 'old-config'):
        choice.add_argument('--' + name, required=True)
    acceptance = sub.add_parser('accept')
    acceptance.add_argument('--manifest', action='append', required=True)
    acceptance.add_argument('--expected', required=True)
    acceptance.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.command == 'select':
        return 0 if select(args) else 2
    from .runtime_acceptance import run
    report = run(None, args.manifest, args.expected, args.out)
    print(json.dumps({'rows': report['rows'], 'difference_count': report['difference_count']}))
    return int(report['difference_count'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
