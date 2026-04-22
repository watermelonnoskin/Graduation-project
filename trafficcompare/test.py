import argparse
import os
import sys
import numpy as np


def _summarize_array(arr, max_print=20, stats=False):
    print(f"type: {type(arr)}")
    if isinstance(arr, np.ndarray):
        print(f"dtype: {arr.dtype}")
        print(f"shape: {arr.shape}")
        if arr.size == 0:
            print("(empty array)")
            return

        if stats and np.issubdtype(arr.dtype, np.number):
            flat = arr.reshape(-1)
            print(f"min: {np.min(flat)}")
            print(f"max: {np.max(flat)}")
            print(f"mean: {np.mean(flat)}")
            print(f"std: {np.std(flat)}")

        to_show = min(int(max_print), arr.size)
        flat = arr.reshape(-1)
        print(f"first {to_show} elements (flattened):")
        print(flat[:to_show])
    else:
        print(arr)


def _print_npz(npz, max_print=20, stats=False):
    keys = list(npz.keys())
    print(f"npz keys ({len(keys)}): {keys}")
    for k in keys:
        print("\n" + "=" * 80)
        print(f"key: {k}")
        _summarize_array(npz[k], max_print=max_print, stats=stats)


def main():
    parser = argparse.ArgumentParser(description="Inspect .npy/.npz file contents")
    parser.add_argument("path", type=str, help="path to .npy or .npz file")
    parser.add_argument("--max_print", type=int, default=20, help="how many elements to print (flattened)")
    parser.add_argument("--stats", action="store_true", help="print numeric stats (min/max/mean/std) if possible")
    parser.add_argument(
        "--allow_pickle",
        action="store_true",
        help="allow loading object arrays (use only if you trust the file)",
    )
    args = parser.parse_args()

    path = args.path
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return 2

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        with np.load(path, allow_pickle=bool(args.allow_pickle)) as npz:
            _print_npz(npz, max_print=args.max_print, stats=args.stats)
        return 0

    if ext == ".npy":
        obj = np.load(path, allow_pickle = True)
        _summarize_array(obj, max_print=args.max_print, stats=args.stats)
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            try:
                print("\nobject array preview:")
                preview_n = min(5, obj.size)
                print(obj.reshape(-1)[:preview_n])
            except Exception:
                pass
        return 0

    print(f"Unsupported extension: {ext} (expected .npy or .npz)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
