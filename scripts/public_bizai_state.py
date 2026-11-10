"""Benchmark licensed BizAI U-Net weights against the released masked states.

Imports the separately pinned MIT source; never uses shipped prediction files.
Processes one panel at a time to bound memory, with resumable per-panel CSVs.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def stream_profiles(panel, links, io):
    """Compute the upstream train-only weekday profiles without a dense train cache."""
    import numpy as np
    import pandas as pd
    sums = np.zeros((2, 2016, len(links)), dtype=np.float64)
    count = np.zeros((2016, len(links)), dtype=np.int64)
    positions = {str(link): i for i, link in enumerate(links)}
    for path in io.mainline_files(panel, 'train', False):
        frame = pd.read_parquet(path, columns=['timestamp', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible'])
        stamp = pd.to_datetime(frame.timestamp, utc=True)
        slot = (stamp.dt.dayofweek*288+stamp.dt.hour*12+stamp.dt.minute//5).to_numpy()
        col = frame.link_id.astype(str).map(positions).to_numpy()
        speed = frame.speed_kmh.to_numpy(np.float32)
        flow = frame.flow_vph.to_numpy(np.float32)
        valid = frame.is_score_eligible.to_numpy(bool) & np.isfinite(speed) & np.isfinite(flow)
        np.add.at(count, (slot[valid], col[valid]), 1)
        np.add.at(sums[0], (slot[valid], col[valid]), speed[valid])
        np.add.at(sums[1], (slot[valid], col[valid]), flow[valid])
    fallback = sums.sum(1)/np.maximum(count.sum(0), 1)
    return [np.where(count > 0, sums[c]/np.maximum(count, 1), fallback[c][None, :]) for c in (0, 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--kind', choices=['unet', 'unet2'], default='unet')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 123, 2031, 137])
    ap.add_argument('--panel', action='append')
    ap.add_argument('--batch', type=int, default=1)
    args = ap.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ['TFB_RELEASE'] = str(args.release_root.resolve())
    os.environ['TFB_CACHE'] = str(output / 'cache')
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    sys.path.insert(0, str(source / 'src'))
    import numpy as np
    import torch
    from tfb import cache, io
    from tfb.t1 import unet as U
    from tfb.t1.base import write_state, slot_of

    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    width, win, ctx = (32, 432, 72) if args.kind == 'unet' else (48, 576, 144)
    panels = args.panel or io.panels()
    weights = [source / 'artifacts' / 'weights' / args.kind / f'seed{s}' / 'weights.pt' for s in args.seeds]
    if not all(p.is_file() for p in weights):
        raise FileNotFoundError('Download and verify the upstream weights archive first')
    metadata = {'source': 'https://github.com/IzbassarO/tfbench-bizai-release',
                'license': 'MIT', 'kind': args.kind, 'seeds': args.seeds,
                'device': str(device), 'torch': torch.__version__,
                'inference_from_weights': True, 'shipped_predictions_used': False,
                'weights_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in weights}}
    (output / 'protocol.json').write_text(json.dumps(metadata, indent=2))
    start = time.monotonic()
    files = []
    for panel in panels:
        dest = output / f'{panel}.csv'
        files.append(dest)
        if dest.exists():
            print(f'{panel}: completed earlier', flush=True)
            continue
        print(f'{panel}: streaming train-only profiles', flush=True)
        reference = cache.load(panel, 'validation')
        profiles = stream_profiles(panel, reference.links, io)
        profile_links = reference.links.copy()
        del reference
        predictions = {}
        for split in ('validation', 'private'):
            pn = cache.load(panel, split)
            if not np.array_equal(pn.links, profile_links):
                raise ValueError('Link order differs between splits')
            ordered = U.T2.load(panel, split)
            U.T2.load.cache_clear()
            o = U.T2.downstream_order(panel, pn.links)
            d = dict(panel=panel, order=o, slot=slot_of(pn),
                prof_s=profiles[0][:, o].astype(np.float32), prof_f=profiles[1][:, o].astype(np.float32),
                ms=ordered['mspeed'], mf=ordered['mflow'], mo=pn.mocc[:, o],
                mpct=pn.mpct[:, o], melig=ordered['melig'], ron=ordered['ron'], roff=ordered['roff'],
                target=pn.target[:, o],
                **{k: ordered[k].astype(np.float32) for k in ('v_f', 'lanes', 'cap', 'length')})
            del pn, ordered
            days = np.arange(len(d['ms']) // U.STEP)
            mean_s = mean_f = None
            for seed, weight in zip(args.seeds, weights):
                print(f'{panel}/{split}: {args.kind} seed{seed}', flush=True)
                net = U.UNet(U.C_IN, width).to(device)
                net.load_state_dict(torch.load(weight, map_location=device, weights_only=True)['net'])
                sp, fl = U.predict(net, d, days, device, bs=args.batch, win=win, ctx=ctx)
                if mean_s is None:
                    mean_s = sp.astype(np.float64) / len(weights)
                    mean_f = fl.astype(np.float64) / len(weights)
                else:
                    mean_s += sp / len(weights)
                    mean_f += fl / len(weights)
                del net, sp, fl
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            # U.predict already restores the cache's original link order.
            predictions[(panel, split)] = (mean_s, mean_f)
            del d
            gc.collect()
        temporary = dest.with_suffix('.partial.csv')
        write_state(predictions, temporary)
        temporary.replace(dest)
        del profiles, predictions
        gc.collect()
        print(f'{panel}: wrote {dest}; elapsed {time.monotonic()-start:.0f}s', flush=True)
    final = output / 'state_submission.csv'
    with final.open('wb') as dst:
        for index, path in enumerate(files):
            with path.open('rb') as src:
                if index:
                    next(src)
                for block in iter(lambda: src.read(1 << 20), b''):
                    dst.write(block)
    metadata.update(seconds=time.monotonic()-start, output=str(final),
                    output_sha256=hashlib.sha256(final.read_bytes()).hexdigest())
    (output / 'result.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
