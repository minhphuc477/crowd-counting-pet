#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print('+ ' + ' '.join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def download(url, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url} -> {output}')
    with urllib.request.urlopen(url) as response, open(output, 'wb') as handle:
        shutil.copyfileobj(response, handle)
    return output


def extract_archive(archive, extract_dir):
    archive = Path(archive)
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f'Extracting {archive} -> {extract_dir}')
    suffixes = ''.join(archive.suffixes).lower()
    if suffixes.endswith('.zip'):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
    elif suffixes.endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(archive) as tf:
            tf.extractall(extract_dir)
    else:
        raise SystemExit(f'Unsupported archive type: {archive}')


def find_nwpu_root(root):
    root = Path(root)
    candidates = [root]
    candidates.extend(path for path in root.rglob('*') if path.is_dir() and path.name in ('NWPU-Crowd', 'NWPU'))
    candidates.extend(path for path in root.rglob('*') if path.is_dir())
    for candidate in candidates:
        if (candidate / 'images').is_dir() and (
            (candidate / 'jsons').is_dir() or (candidate / 'mats').is_dir()
        ):
            return candidate
    return None


def normalize_layout(source_root, data_root):
    source_root = Path(source_root).resolve()
    data_root = Path(data_root).resolve()
    data_root.parent.mkdir(parents=True, exist_ok=True)
    if source_root == data_root:
        return data_root
    if data_root.exists():
        print(f'Using existing data root: {data_root}')
        return data_root
    print(f'Copying NWPU layout {source_root} -> {data_root}')
    shutil.copytree(source_root, data_root)
    return data_root


def maybe_merge_image_parts(data_root):
    data_root = Path(data_root)
    images_dir = data_root / 'images'
    if images_dir.is_dir():
        return
    part_dirs = sorted(
        path for path in data_root.iterdir()
        if path.is_dir() and path.name.lower().startswith('images_part')
    )
    if not part_dirs:
        return
    images_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for part_dir in part_dirs:
        for image in part_dir.iterdir():
            if image.is_file():
                shutil.copy2(image, images_dir / image.name)
                moved += 1
    print(f'Collected {moved} images from images_part* into {images_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Download/extract/validate NWPU-Crowd for PET on Ubuntu.'
    )
    parser.add_argument('--url', default=os.environ.get('NWPU_URL', ''))
    parser.add_argument('--archive', default=os.environ.get('NWPU_ARCHIVE', ''))
    parser.add_argument('--data_root', default=os.environ.get('NWPU_DATA', './data/NWPU-Crowd'))
    parser.add_argument('--work_dir', default='./data/downloads')
    parser.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT', ''))
    parser.add_argument('--skip_extract', action='store_true')
    parser.add_argument('--run_eval', action='store_true')
    parser.add_argument('--eval_split', default='val', choices=('val', 'test', 'train'))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    source_root = data_root if data_root.exists() else None

    if source_root is None:
        archive = Path(args.archive) if args.archive else None
        if archive is None and args.url:
            archive_name = Path(args.url.split('?', 1)[0]).name or 'NWPU-Crowd.zip'
            archive = download(args.url, Path(args.work_dir) / archive_name)
        if archive is None:
            raise SystemExit(
                'NWPU-Crowd official mirrors are often browser/login gated. '
                'Download the archive from the official page, then rerun:\n'
                '  python scripts/setup_nwpu_crowd.py --archive /path/to/NWPU-Crowd.zip\n'
                'Or provide a direct URL with --url or NWPU_URL.'
            )
        extract_dir = Path(args.work_dir) / 'NWPU-Crowd-extracted'
        if not args.skip_extract:
            extract_archive(archive, extract_dir)
        source_root = find_nwpu_root(extract_dir)
        if source_root is None:
            raise SystemExit(f'Could not find NWPU-Crowd layout under {extract_dir}')
        data_root = normalize_layout(source_root, data_root)

    maybe_merge_image_parts(data_root)

    run([
        sys.executable,
        REPO_ROOT / 'scripts' / 'check_nwpu_annotations.py',
        '--data_path',
        data_root,
        '--nwpu_eval_split',
        args.eval_split,
    ])

    print('\nDataset ready.')
    print(f'export NWPU_DATA="{data_root}"')
    if args.checkpoint:
        eval_cmd = [
            'CUDA_VISIBLE_DEVICES=0',
            'python',
            'eval.py',
            '--resume',
            args.checkpoint,
            '--dataset_file',
            'NWPU',
            '--data_path',
            str(data_root),
            '--device',
            'cuda',
            '--num_workers',
            '2',
            '--eval_max_size',
            '1536',
            '--nwpu_eval_split',
            args.eval_split,
            '--localization_protocol',
            'target_sigma',
            '--score_threshold',
            '0.55',
            '--split_threshold',
            '0.45',
            '--results_file',
            f'eval_results/NWPU/eval_{args.eval_split}_target_sigma.json',
        ]
        print('\nEval command:')
        print(' '.join(eval_cmd))
        if args.run_eval:
            run(eval_cmd[1:])


if __name__ == '__main__':
    main()
