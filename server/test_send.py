#!/usr/bin/env python3
import argparse
import requests

def main():
    parser = argparse.ArgumentParser(description='Send WAV to /predict')
    parser.add_argument('wav', help='Path to WAV file to send')
    parser.add_argument('--url', default='http://localhost:5000/predict', help='Server predict URL')
    args = parser.parse_args()

    with open(args.wav, 'rb') as f:
        files = {'audio': ('recording.wav', f, 'audio/wav')}
        resp = requests.post(args.url, files=files)

    print('Status:', resp.status_code)
    print('Response:', resp.text)

if __name__ == '__main__':
    main()
