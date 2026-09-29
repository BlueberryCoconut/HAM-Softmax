import hashlib
import os
import tarfile
import time
import zipfile

import requests


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'augmentation_data'))
ARCHIVES = os.path.join(ROOT, 'archives')
CHUNK_SIZE = 8 * 1024 * 1024

FILES = (
    {
        'name': 'musan.tar.gz',
        'url': 'https://www.openslr.org/resources/17/musan.tar.gz',
        'md5': '0c472d4fc0c5141eca47ad1ffeb2a7df',
    },
    {
        'name': 'rirs_noises.zip',
        'url': 'https://www.openslr.org/resources/28/rirs_noises.zip',
        'md5': 'e6f48e257286e05de56413b4779d8ffb',
    },
)


def md5sum(path):
    digest = hashlib.md5()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(item):
    destination = os.path.join(ARCHIVES, item['name'])
    partial = destination + '.part'
    if os.path.exists(destination) and md5sum(destination) == item['md5']:
        print(f"Verified existing {item['name']}", flush=True)
        return destination

    for attempt in range(1, 51):
        downloaded = os.path.getsize(partial) if os.path.exists(partial) else 0
        headers = {'Range': f'bytes={downloaded}-'} if downloaded else {}
        try:
            with requests.get(item['url'], headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                append = downloaded > 0 and response.status_code == 206
                if not append:
                    downloaded = 0
                total = downloaded + int(response.headers.get('content-length', 0))
                mode = 'ab' if append else 'wb'
                last_report = 0.0
                with open(partial, mode) as handle:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_report >= 5:
                            percent = downloaded * 100 / total if total else 0
                            print(
                                f"{item['name']}: {percent:.1f}% "
                                f"({downloaded / 1024**3:.2f}/{total / 1024**3:.2f} GiB)",
                                flush=True,
                            )
                            last_report = now
            break
        except requests.RequestException as error:
            if attempt == 50:
                raise
            delay = min(60, attempt * 5)
            print(
                f"{item['name']}: connection interrupted ({type(error).__name__}); "
                f"retrying from {downloaded / 1024**3:.2f} GiB in {delay}s "
                f"[{attempt}/50]",
                flush=True,
            )
            time.sleep(delay)

    actual = md5sum(partial)
    if actual != item['md5']:
        raise RuntimeError(f"MD5 mismatch for {item['name']}: {actual}")
    os.replace(partial, destination)
    print(f"Downloaded and verified {item['name']}", flush=True)
    return destination


def safe_extract_tar(archive):
    target = os.path.abspath(ROOT)
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle.getmembers():
            member_path = os.path.abspath(os.path.join(target, member.name))
            if os.path.commonpath((target, member_path)) != target:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        bundle.extractall(target)


def safe_extract_zip(archive):
    target = os.path.abspath(ROOT)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = os.path.abspath(os.path.join(target, member.filename))
            if os.path.commonpath((target, member_path)) != target:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(target)


def main():
    os.makedirs(ARCHIVES, exist_ok=True)
    musan = download(FILES[0])
    rirs = download(FILES[1])

    musan_marker = os.path.join(ROOT, 'musan', 'noise')
    if not os.path.isdir(musan_marker):
        print('Extracting MUSAN...', flush=True)
        safe_extract_tar(musan)
    else:
        print('MUSAN already extracted', flush=True)

    rirs_marker = os.path.join(ROOT, 'RIRS_NOISES', 'simulated_rirs')
    if not os.path.isdir(rirs_marker):
        print('Extracting RIRS_NOISES...', flush=True)
        safe_extract_zip(rirs)
    else:
        print('RIRS_NOISES already extracted', flush=True)

    print('AUGMENTATION_DATA_READY', flush=True)


if __name__ == '__main__':
    main()
