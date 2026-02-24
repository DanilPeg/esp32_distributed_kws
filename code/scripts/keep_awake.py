"""Keep-awake helper: nudges mouse cursor at a fixed interval.

Usage:
  python keep_awake.py --interval 120

Default interval: 120 seconds (2 minutes).
"""

import argparse
import ctypes
import time

MOUSEEVENTF_MOVE = 0x0001


def move_mouse(dx: int, dy: int) -> None:
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, dx, dy, 0, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=120,
                        help='Seconds between nudges (default: 120).')
    args = parser.parse_args()

    print(f'Keep-awake started. Interval: {args.interval}s. Ctrl+C to stop.')
    try:
        while True:
            move_mouse(1, 0)
            time.sleep(0.05)
            move_mouse(-1, 0)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('Stopped.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
