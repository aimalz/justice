# -*- coding: utf-8 -*-
"""Extracts a downloaded encrypted file."""
import argparse
import platform
import re
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


class PlatformSpecificExe(object):
    def __init__(self, **by_platform):
        assert "linux" in by_platform
        key = platform.system().lower()
        if key not in by_platform:
            key = "linux"
        self.exe_name, self.install_instruction = by_platform[key]

    def check_installed(self):
        try:
            subprocess.check_call(["which", self.exe_name])
        except subprocess.CalledProcessError:
            print(
                f"Please ensure `{self.exe_name}` is on your path. " +
                (f"Maybe try running `{self.install_instruction}`" if self.install_instruction else "")
            )
            exit(1)

    def override_and_check_installed(self, exe_name):
        if exe_name is not None:
            self.exe_name = exe_name
        self.check_installed()


def main():
    tar_exe = PlatformSpecificExe(
        darwin=("gtar", "brew install gnu-tar"),
        linux=("tar", None)
    )
    openssl_exe = PlatformSpecificExe(
        darwin=("/usr/local/opt/openssl/bin/openssl", "brew install openssl"),
        linux=("openssl", "apt install -y openssl"),
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
    cmd_args.add_argument("--tar-exe", help="Override for tar program.")
    cmd_args.add_argument("--openssl-exe", help="Override for openssl program.")
    args = cmd_args.parse_args()

    # Check programs exist.
    tar_exe.override_and_check_installed(args.tar_exe)
    openssl_exe.override_and_check_installed(args.openssl_exe)

    # Check OpenSSL version.
    openssl_version = subprocess.check_output([openssl_exe.exe_name, "version"]).strip().decode("utf-8")
    assert re.match(r"OpenSSL [1-9]\d*\.", openssl_version), "Expected OpenSSL >= v1."

    # Actually run extraction.
    print("Extracting ...")
    tar_extract = subprocess.Popen([tar_exe.exe_name, "xv"],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   cwd=str(path_util.project_root))
    print_thread = threading.Thread(
        target=print_every_few_secs, args=(tar_extract.stdout, )
    )
    print_thread.start()
    subprocess.check_call([
        openssl_exe.exe_name, "aes-128-cbc", "-d", "-kfile", args.key_file, "-in", args.encrypted_file
    ],
        stdout=tar_extract.stdin)
    assert tar_extract.wait() == 0, "Tar failed!"
    print_thread.join()


if __name__ == '__main__':
    main()
