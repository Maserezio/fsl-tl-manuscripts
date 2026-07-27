"""PREDICT — single entry point for both datasets (thin dispatcher).

  python predict.py diva    [args for predict_diva_seamcarve.py]
  python predict.py u_diads [args for predict_u_diads.py]

DIVA    -> predict_diva_seamcarve.py : U-Net -> word-merge + valley cuts -> PAGE-XML
           (--preset unified|playground; runs the Java evaluator unless --no-eval)
u_diads -> predict_u_diads.py       : cache sliding-window prob maps per manuscript
           (fusion + Zottin evaluation happen in evaluate_lines.py --family udiads)

Run with `diva --help` / `u_diads --help` for the full per-dataset options.
"""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("diva", "u_diads"):
        print(__doc__)
        sys.exit("first argument must be: diva | u_diads")
    dataset = sys.argv.pop(1)                      # underlying mains parse sys.argv themselves
    if dataset == "diva":
        from predict_diva_seamcarve import main as run
    else:
        from predict_u_diads import main as run
    run()


if __name__ == "__main__":
    main()
