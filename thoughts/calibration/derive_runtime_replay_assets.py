"""Bind accepted saved centers to existing physical references, without new truth.

Produces input/settings/optics only. The production Rust CLI must independently
fit and validate the calibration; this tool never manufactures a fitted map.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--reference-input', type=Path, required=True)
    p.add_argument('--reference-optics', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--calibration-revision', type=int, required=True)
    args = p.parse_args()
    report = json.loads(args.summary.read_text())
    reference = json.loads(args.reference_input.read_text())
    optics = json.loads(args.reference_optics.read_text())
    revision = report['refiner_revision']
    by_id = {x['ball_id']: x for x in report['results']}
    assert len(by_id) == len(report['results']), 'duplicate ball IDs'
    scale = reference['source_width_px'] / report['input_image']['dimensions_px'][0]
    assert reference['source_height_px'] / report['input_image']['dimensions_px'][1] == scale
    for point in reference['correspondences']:
        name = point['provenance'].split(';')[0]
        observation = by_id[name]
        assert observation['status'] == 'accepted', f'{name} not accepted'
        x, y = observation['center_image_px']
        point['image_point_px'] = {'x_px': x * scale, 'y_px': y * scale}
        point['provenance'] = (f"{name}; actual detector box; refiner{revision} accepted; "
                               f"summary {args.summary}; trace SHA256 "
                               f"{observation['case_trace_sha256']}; "
                               'original physical measurement uncertainty unknown')
    reference['refiner_revision'] = revision
    reference['calibration_revision'] = args.calibration_revision
    optics['refiner_revision'] = revision
    width = reference['source_width_px']
    suffix = f'{width}-r{revision}.json'
    args.output_dir.mkdir(exist_ok=True, parents=True)
    settings = {'optics_profile_path': str(args.output_dir / ('optics-' + suffix)),
                'ball_center_calibration_path': str(args.output_dir / ('ball-center-' + suffix)),
                'parameters': {'batch_budget_ms': 150, 'proposal_budget_ms': 30}}
    for prefix, data in [('input-', reference), ('optics-', optics), ('settings-', settings)]:
        with (args.output_dir / (prefix + suffix)).open('x') as handle:
            json.dump(data, handle, indent=2)
            handle.write('\n')


if __name__ == '__main__':
    main()
