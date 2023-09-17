import logging
import os
import shlex
import subprocess
from typing import List
from typing import Optional

import zstandard
from appdirs import user_cache_dir

LOGGER = logging.getLogger(__name__)


def shell(args: List[str]):
    """Executes the shell command in `args`."""
    cmd = shlex.join(args)
    LOGGER.info(f"Executing: {cmd}")
    exit_code = subprocess.call(args)
    if exit_code != 0:
        raise Exception(f"Failed with exit code {exit_code}: {cmd}")


def download_file(source_uri: str, target_path: str, unpack: bool = False,
                  unpack_type: Optional[str] = None, aws_profile: str = 'default'):
    """Download `source_uri` to `target_path` if it doesn't exist."""
    if os.path.exists(target_path):
        # Assume it's all good
        LOGGER.info(f"Not downloading {source_uri} because {target_path} already exists")
        return

    # Download
    tmp_path: str = f"{target_path}.tmp"
    if source_uri.startswith("s3://"):
        shell(['aws', 's3', 'cp', source_uri, tmp_path, '--profile', aws_profile])
    else:
        # gdown is used to download large files/zip folders from Google Drive.
        # It bypasses security warnings which wget cannot handle.
        downloader_executable: str = "gdown" if source_uri.startswith(
            "https://drive.google.com") else "wget"
        shell([downloader_executable, source_uri, "-O", tmp_path])

    # Unpack (if needed) and put it in the right location
    if unpack:
        if unpack_type is None:
            if source_uri.endswith(".tar") or source_uri.endswith(".tar.gz"):
                unpack_type = "untar"
            elif source_uri.endswith(".zip"):
                unpack_type = "unzip"
            elif source_uri.endswith(".zst"):
                unpack_type = "unzstd"
            else:
                raise Exception(
                    "Failed to infer the file format from source_uri. Please specify unpack_type.")

        tmp2_path = target_path + ".tmp2"
        os.makedirs(tmp2_path, exist_ok=True)
        if unpack_type == "untar":
            shell(["tar", "xf", tmp_path, "-C", tmp2_path])
        elif unpack_type == "unzip":
            shell(["unzip", tmp_path, "-d", tmp2_path])
        elif unpack_type == "unzstd":
            dctx = zstandard.ZstdDecompressor()
            with open(tmp_path, "rb") as ifh, open(os.path.join(tmp2_path, "data"), "wb") as ofh:
                dctx.copy_stream(ifh, ofh)
        else:
            raise Exception("Invalid unpack_type")
        files = os.listdir(tmp2_path)
        if len(files) == 1:
            # If contains one file, just get that one file
            shell(["mv", os.path.join(tmp2_path, files[0]), target_path])
            os.rmdir(tmp2_path)
        else:
            shell(["mv", tmp2_path, target_path])
        os.unlink(tmp_path)
    else:
        # Don't decompress if desired `target_path` ends with `.gz`.
        if source_uri.endswith(".gz") and not target_path.endswith(".gz"):
            gzip_path = f"{target_path}.gz"
            shell(["mv", tmp_path, gzip_path])
            # gzip writes its output to a file named the same as the input file, omitting the .gz extension
            shell(["gzip", "-d", gzip_path])
        else:
            shell(["mv", tmp_path, target_path])
    LOGGER.info(f"Finished downloading {source_uri} to {target_path}")


def get_cache_dir(dir_name=None, create=True):
    cache_root = user_cache_dir("enlighten_benchmark")
    if dir_name is not None:
        cache_dir = os.path.join(cache_root, dir_name)
    else:
        cache_dir = cache_root
    if create and not os.path.exists(cache_dir):
        LOGGER.info('Creating cache dir')
        os.makedirs(cache_dir)
    return cache_dir
