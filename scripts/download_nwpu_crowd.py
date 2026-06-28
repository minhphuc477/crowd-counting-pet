#!/usr/bin/env python3
import argparse
import base64
import json
import re
import shutil
import sys
import time
import urllib.parse
import zipfile
from pathlib import Path

import requests


SHARE_URL = (
    'https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/'
    'gjy3035_mail_nwpu_edu_cn/EsubMp48wwJDiH0YlT82NYYBmY9L0s-FprrBcoaAJkI1rw?e=e2JLgD'
)
SERVER_RELATIVE_FOLDER = (
    '/personal/gjy3035_mail_nwpu_edu_cn/Documents/'
    '\u8bba\u6587\u5f00\u6e90\u6570\u636e/NWPU-Crowd'
)
DEFAULT_FILES = ('jsons.zip', 'mats.zip', 'train.txt', 'val.txt', 'test.txt', 'readme.md')
IMAGE_FILES = (
    'images_part1.zip',
    'images_part2.zip',
    'images_part3.zip',
    'images_part4.zip',
    'images_part5.zip',
)
OFFICIAL_VAL_GT_SIZE = 3238132
OFFICIAL_VAL_GT_BLOB_URL = (
    'https://api.github.com/repos/gjy3035/'
    'NWPU-Crowd-Sample-Code-for-Localization/git/blobs/'
    '7def184557a4cd708431239ccb518b94c55d355d'
)


def checked_get(session, url, **kwargs):
    last_error = None
    for attempt in range(5):
        try:
            response = session.get(url, timeout=kwargs.pop('timeout', 60), **kwargs)
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 + attempt * 2)
                last_error = RuntimeError(f'HTTP {response.status_code}: {url}')
                continue
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(2 + attempt * 2)
    raise last_error


def open_text(session, url):
    return checked_get(session, url, timeout=60).text


def parse_page_context(html):
    match = re.search(r'_spPageContextInfo=(\{.*?\});_spPageContextInfo', html)
    if not match:
        raise RuntimeError('Could not find _spPageContextInfo in SharePoint page')
    return json.loads(match.group(1))


def list_files(session, ctx):
    base = ctx['siteAbsoluteUrl']
    folder = urllib.parse.quote(SERVER_RELATIVE_FOLDER, safe='/')
    url = (
        f"{base}/_api/web/GetFolderByServerRelativeUrl(@v)/Files"
        f"?@v='{folder}'"
    )
    response = checked_get(
        session,
        url,
        headers={
            'Accept': 'application/json;odata=verbose',
            'User-Agent': 'Mozilla/5.0',
        },
        timeout=60,
    )
    payload = response.json()
    return {item['Name']: item for item in payload['d']['results']}


def download_file(session, ctx, server_relative_url, output_path, expected_size=0):
    base = ctx['siteAbsoluteUrl']
    encoded = urllib.parse.quote(server_relative_url, safe='/')
    url = (
        f"{base}/_api/web/GetFileByServerRelativeUrl(@v)/$value"
        f"?@v='{encoded}'"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = checked_get(session, url, stream=True, timeout=600)
    downloaded = 0
    start_time = time.time()
    last_print = 0.0
    total = int(expected_size or response.headers.get('content-length') or 0)
    with open(output_path, 'wb') as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= 2.0:
                    elapsed = max(now - start_time, 1e-6)
                    speed = downloaded / elapsed / (1024 * 1024)
                    if total > 0:
                        pct = downloaded * 100.0 / total
                        print(
                            f'  {downloaded / (1024 * 1024):.1f}/'
                            f'{total / (1024 * 1024):.1f} MB '
                            f'({pct:.1f}%) {speed:.2f} MB/s',
                            flush=True,
                        )
                    else:
                        print(
                            f'  {downloaded / (1024 * 1024):.1f} MB '
                            f'{speed:.2f} MB/s',
                            flush=True,
                        )
                    last_print = now
    if total > 0:
        elapsed = max(time.time() - start_time, 1e-6)
        speed = downloaded / elapsed / (1024 * 1024)
        print(
            f'  {downloaded / (1024 * 1024):.1f}/'
            f'{total / (1024 * 1024):.1f} MB '
            f'(100.0%) {speed:.2f} MB/s',
            flush=True,
        )
    return output_path.stat().st_size


def unzip_to(zip_path, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)


def download_official_val_gt(session, data_root):
    output_path = data_root / 'val_gt_loc.txt'
    if output_path.exists() and output_path.stat().st_size == OFFICIAL_VAL_GT_SIZE:
        print(f'{output_path}: already downloaded')
        return output_path

    print('Downloading official NWPU validation localization thresholds ...')
    response = checked_get(
        session,
        OFFICIAL_VAL_GT_BLOB_URL,
        headers={'Accept': 'application/vnd.github+json'},
        timeout=60,
    )
    payload = response.json()
    if payload.get('encoding') != 'base64' or not payload.get('content'):
        raise RuntimeError('GitHub returned an invalid val_gt_loc.txt blob response')
    content = base64.b64decode(payload['content'])
    if len(content) != OFFICIAL_VAL_GT_SIZE:
        raise RuntimeError(
            f'Invalid val_gt_loc.txt size: expected {OFFICIAL_VAL_GT_SIZE}, '
            f'got {len(content)}'
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    print(f'  saved {output_path} ({len(content)} bytes)')
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Download NWPU-Crowd files from the official public SharePoint mirror.')
    parser.add_argument('--data_root', default='data/NWPU-Crowd')
    parser.add_argument('--download_dir', default='data/downloads/NWPU-Crowd')
    parser.add_argument('--include_images', action='store_true',
                        help='also download images_part*.zip; these files are very large')
    parser.add_argument('--no_extract', action='store_true')
    parser.add_argument(
        '--official_localization_only',
        action='store_true',
        help='download only the official validation localization threshold file',
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    data_root = Path(args.data_root)
    if args.official_localization_only:
        download_official_val_gt(session, data_root)
        print('Done.')
        print(f'Dataset root: {data_root}')
        return

    html = open_text(session, SHARE_URL)
    ctx = parse_page_context(html)
    files = list_files(session, ctx)

    wanted = list(DEFAULT_FILES)
    if args.include_images:
        wanted.extend(IMAGE_FILES)

    download_dir = Path(args.download_dir)
    for name in wanted:
        if name not in files:
            print(f'WARNING: {name} not found in public mirror', file=sys.stderr)
            continue
        output = download_dir / name
        if output.exists() and output.stat().st_size == int(files[name].get('Length', 0)):
            print(f'{name}: already downloaded')
        else:
            print(f'Downloading {name} ...')
            size = download_file(
                session,
                ctx,
                files[name]['ServerRelativeUrl'],
                output,
                expected_size=int(files[name].get('Length', 0) or 0),
            )
            print(f'  saved {output} ({size} bytes)')

        if args.no_extract:
            continue
        if name == 'jsons.zip':
            unzip_to(output, data_root / 'jsons')
        elif name == 'mats.zip':
            unzip_to(output, data_root / 'mats')
        elif name.startswith('images_part') and name.endswith('.zip'):
            unzip_to(output, data_root / 'images')
        elif name.endswith('.txt') or name == 'readme.md':
            data_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output, data_root / name)

    if not args.no_extract:
        download_official_val_gt(session, data_root)

    print('Done.')
    print(f'Dataset root: {data_root}')


if __name__ == '__main__':
    main()
