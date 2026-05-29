#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def iter_images(root):
    root = Path(root)
    if root.is_file():
        if root.suffix.lower() in IMAGE_EXTENSIONS:
            yield root
        return
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_image(path):
    with Image.open(path) as img:
        img.load()
        return img.convert('RGB')


def repair_image(path, backup_suffix):
    img = load_image(path)
    backup = path.with_name(path.name + backup_suffix)
    if backup.exists():
        raise FileExistsError(f'backup already exists: {backup}')
    shutil.copy2(path, backup)
    save_kwargs = {}
    if path.suffix.lower() in {'.jpg', '.jpeg'}:
        save_kwargs.update({'quality': 95, 'subsampling': 0})
    img.save(path, **save_kwargs)
    return backup


def main():
    parser = argparse.ArgumentParser(description='Check and optionally re-encode readable crowd dataset images.')
    parser.add_argument('path', help='image file or dataset directory')
    parser.add_argument('--repair', action='store_true', help='re-encode images that fail PIL load')
    parser.add_argument('--backup_suffix', default='.bak', help='suffix for backups when --repair is used')
    parser.add_argument('--max_errors', default=20, type=int)
    args = parser.parse_args()

    total = 0
    bad = []
    repaired = []
    for path in iter_images(args.path):
        total += 1
        try:
            load_image(path)
        except Exception as exc:
            bad.append((path, exc))
            print(f'BAD: {path}: {exc}')
            if args.repair:
                try:
                    backup = repair_image(path, args.backup_suffix)
                    repaired.append((path, backup))
                    print(f'  repaired, backup={backup}')
                except Exception as repair_exc:
                    print(f'  repair failed: {repair_exc}')
            if len(bad) >= args.max_errors:
                print(f'stopping after --max_errors={args.max_errors}')
                break

    print(f'checked images: {total}')
    print(f'bad images: {len(bad)}')
    print(f'repaired images: {len(repaired)}')
    return 1 if bad and not args.repair else 0


if __name__ == '__main__':
    raise SystemExit(main())
