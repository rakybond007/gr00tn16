"""평가 결과 요약: <run 폴더>/<task>/prediction.txt 를 읽어 태스크별·전체 성공률을 출력한다.

  python robocasa_finetune/eval/summarize.py eval_out/my_run [--csv out.csv]
"""
import argparse
import glob
import os
import re


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--csv", default="")
    a = p.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.run_dir, "*", "prediction.txt"))):
        res = [m.group(1) == "True" for m in re.finditer(r"is_success: \[\s*(True|False)\]", open(f).read())]
        if res:
            rows.append((os.path.basename(os.path.dirname(f)), len(res), sum(res)))
    if not rows:
        print("prediction.txt 가 없다"); return
    for t, n, s in rows:
        print(f"{t:24s} {s:3d}/{n:3d}  {s / n:.2f}")
    N, S = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(f"\n태스크 {len(rows)}개, 에피소드 {N}, 성공률 {S / N:.3f}")
    if a.csv:
        with open(a.csv, "w") as fo:
            fo.write("task,episodes,success,rate\n")
            for t, n, s in rows:
                fo.write(f"{t},{n},{s},{s / n:.4f}\n")


if __name__ == "__main__":
    main()
