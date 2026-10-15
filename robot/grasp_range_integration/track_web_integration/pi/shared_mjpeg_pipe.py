"""Feed frozen grab_visual Cam from the task camera owner, as a JPEG pipe."""
import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', required=True)
    parser.add_argument('--size', required=True)
    parser.add_argument('--fps', type=float, required=True)
    args = parser.parse_args()
    url = os.environ.get('FINAL_WORK_CAMERA_BROKER_URL')
    if url != 'http://127.0.0.1:18881':
        parser.error('Shared camera owner is required; USB fallback is disabled')
    try:
        width, height = [int(n) for n in args.size.split('x')]
    except (ValueError, TypeError):
        parser.error('Invalid capture size')
    if not (16 <= width <= 4096 and 16 <= height <= 4096 and 0 < args.fps <= 60):
        parser.error('Invalid image geometry or frame rate')
    import cv2
    from .camera_proxy import SharedCapture, camera_index
    if camera_index(args.camera) is None:
        parser.error('Invalid camera node')
    capture = SharedCapture(args.camera, url=url, cv=cv2)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    last_frame = time.monotonic()
    try:
        while True:
            started = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                if time.monotonic() - last_frame > 10:
                    raise RuntimeError('Shared camera has no fresh frames')
                time.sleep(.05)
                continue
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError('Shared preview JPEG encoding failed')
            sys.stdout.buffer.write(encoded.tobytes())
            sys.stdout.buffer.flush()
            last_frame = time.monotonic()
            time.sleep(max(0., 1. / args.fps - (last_frame - started)))
    except BrokenPipeError:
        return 0
    finally:
        capture.release()


if __name__ == '__main__':
    raise SystemExit(main())
