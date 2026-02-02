import requests, getpass, argparse
from pathlib import Path
import os

BASE_URL = os.getenv("BASE_URL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type = Path, default = "./.api-token")
    parser.add_argument("--user")
    args = parser.parse_args()

    user = args.user
    if not user: user = input('USERNAME: ')
    password = getpass.getpass("PASSCODE: ")

    response = requests.post(f'{BASE_URL}/token', data = {
        "username": user, 'password': password,
    })

    if not response.ok:
        print(response.json().get('message', response.text))
    else:
        token = response.json()['access_token']
        args.dest.parent.mkdir(exist_ok = True, parents = True)
        args.dest.write_text(token)
        print(f"Success! Token saved to {args.dest}")
