# -*- coding: utf-8 -*-
"""Extracts a downloaded encrypted file."""
import argparse
import subprocess
import threading
import time

from justice import path_util


def print_every_few_secs(fileobj):
    last_printed = time.time()
    n_other = 0
    for line in fileobj:
        n_other += 1
        if time.time() - last_printed > 0.5:
            filename = line.decode('utf-8').strip()
            print(f"   expanded {filename} and {n_other} other files")
            last_printed = time.time()
            n_other = 0


def main():
    try:
        subprocess.check_output(["which", "openssl"])
        subprocess.check_output(["which", "openssl"])
    except subprocess.CalledProcessError:
        print(
            "Please ensure `openssl` and `tar` are on your path, maybe try `apt "
            "install -y openssl` on Ubuntu (tar should always be installed)."
        )

    cmd_args = argparse.ArgumentParser()
    cmd_args.add_argument(
        "--key-file",
        required=True,
        help="Key file, usually .key extension, random bytes."
    )
    cmd_args.add_argument(
        "--encrypted-file", required=True, help="Encrypted file, usually myname.tar.enc"
    )
    args = cmd_args.parse_args()
    tar_extract = subprocess.Popen(["tar", "xv"],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   cwd=str(path_util.project_root))
    print_thread = threading.Thread(
        target=print_every_few_secs, args=(tar_extract.stdout, )
    )
    print_thread.start()
    print("Expanding ...")
    subprocess.check_call([
        "openssl", "aes-128-cbc", "-d", "-k", args.key_file, "-in", args.encrypted_file
    ],
        stdout=tar_extract.stdin)
    assert tar_extract.wait() == 0, "Tar failed!"
    print_thread.join()


if __name__ == '__main__':
    main()
