import argparse
import ctypes
import time
from ctypes import wintypes

user32 = ctypes.WinDLL('user32', use_last_error=True)


class POINT(ctypes.Structure):
    _fields_ = [('x', wintypes.LONG), ('y', wintypes.LONG)]


def get_pos():
    pt = POINT()
    if not user32.GetCursorPos(ctypes.byref(pt)):
        raise ctypes.WinError(ctypes.get_last_error())
    return pt.x, pt.y


def set_pos(x, y):
    if not user32.SetCursorPos(int(x), int(y)):
        raise ctypes.WinError(ctypes.get_last_error())


def screen_size():
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def nudge(step: int, pause: float):
    x, y = get_pos()
    width, _ = screen_size()
    if x + step < width:
        nx = x + step
    else:
        nx = max(0, x - step)
    set_pos(nx, y)
    time.sleep(pause)
    set_pos(x, y)


def main():
    parser = argparse.ArgumentParser(description='Keep session awake by nudging the mouse.')
    parser.add_argument('--interval-sec', type=int, default=120)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--pause-sec', type=float, default=0.05)
    args = parser.parse_args()

    print(f'keep_awake_mouse: interval={args.interval_sec}s step={args.step}px')
    try:
        while True:
            nudge(args.step, args.pause_sec)
            time.sleep(args.interval_sec)
    except KeyboardInterrupt:
        print('keep_awake_mouse: stopped')


if __name__ == '__main__':
    main()
