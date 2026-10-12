"""Extract, finalize and validate in order; stop at the first failed stage."""
from pathlib import Path
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 100:
        parser.error('--limit must be between 1 and 100')
    src = Path(__file__).resolve().parent
    common = ['--data-root', str(Path(args.data_root).resolve()),
              '--output-dir', args.output_dir]
    extract = [sys.executable, str(src / 'run.py'), *common]
    if args.limit is not None:
        extract += ['--limit', str(args.limit)]
    if args.device:
        extract += ['--device', args.device]
    if args.resume:
        extract += ['--resume']
    expected = args.limit if args.limit is not None else 100
    finalize = [sys.executable, str(src / 'finalize.py'), *common]
    if expected != 100:
        finalize += ['--allow-partial']
    validate = [sys.executable, str(src / 'validate.py'), *common,
                '--expected-count', str(expected)]
    for name, command in [('extract', extract), ('finalize', finalize), ('validate', validate)]:
        print(f'\n[stage: {name}]', flush=True)
        result = subprocess.run(command, cwd=src.parent)
        if result.returncode:
            print(f'Stopped: {name} failed (exit {result.returncode}); later stages were not run.',
                  file=sys.stderr, flush=True)
            return result.returncode
    print(f'Completed and validated {expected} samples.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
