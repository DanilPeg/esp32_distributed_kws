import argparse
import hashlib
import secrets


def make_hash(password: str, iters: int = 200_000) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iters)
    return f"pbkdf2_sha256${iters}${salt.hex()}${derived.hex()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--password', required=True)
    parser.add_argument('--iters', type=int, default=200_000)
    args = parser.parse_args()
    print(make_hash(args.password, args.iters))


if __name__ == '__main__':
    main()
