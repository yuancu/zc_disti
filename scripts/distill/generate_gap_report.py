"""Compare teacher vs student positive-rank-1 rates and generate a markdown report."""

import argparse
import json
import os

DATASETS = [
    "aila-casedocs",
    "finqa",
    "cure_v1",
    "legalquad",
    "multi-cpr-video",
    "hc3finance",
    "financebench",
    "legal-summ",
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--teacher_dir", type=str, required=True,
    help="Directory containing teacher score JSONs (e.g. artifacts/teacher-score-gemma2)",
)
parser.add_argument(
    "--student_dir", type=str, required=True,
    help="Directory containing student score JSONs (e.g. artifacts/student-score-bge-m3)",
)
parser.add_argument(
    "--output", type=str, default="docs/teacher-student-gap.md",
    help="Output markdown file path",
)
parser.add_argument(
    "--datasets", type=str, nargs="+", default=DATASETS,
    help="Dataset names to include",
)
args = parser.parse_args()


def pos_rank1_rate(score_file: str) -> tuple[float, int]:
    """Return (rate, count) where rate is the fraction of samples whose
    positive doc (index 0) has the highest score."""
    with open(score_file) as f:
        samples = json.load(f)
    rank1 = sum(1 for s in samples if s["scores"][0] == max(s["scores"]))
    return rank1 / len(samples), len(samples)


rows = []
for dataset in args.datasets:
    teacher_file = os.path.join(args.teacher_dir, f"{dataset}.json")
    student_file = os.path.join(args.student_dir, f"{dataset}.json")

    if not os.path.exists(teacher_file):
        print(f"Skipping {dataset}: teacher file not found ({teacher_file})")
        continue
    if not os.path.exists(student_file):
        print(f"Skipping {dataset}: student file not found ({student_file})")
        continue

    t_rate, t_n = pos_rank1_rate(teacher_file)
    s_rate, s_n = pos_rank1_rate(student_file)
    gap = t_rate - s_rate

    rows.append((dataset, s_rate, t_rate, gap, t_n))
    print(f"{dataset:20s}  student={s_rate:.1%}  teacher={t_rate:.1%}  gap={gap:+.1%}  (n={t_n})")

# Sort by gap descending (largest room for improvement first)
rows.sort(key=lambda r: r[3], reverse=True)

teacher_name = os.path.basename(args.teacher_dir.rstrip("/"))
student_name = os.path.basename(args.student_dir.rstrip("/"))

lines = [
    "# Teacher vs Student: Positive-Rank-1 Gap",
    "",
    f"**Teacher:** `{teacher_name}`  ",
    f"**Student:** `{student_name}`",
    "",
    "For each query the positive document is scored alongside hard negatives.",
    '"Pos rank 1" is the rate at which the positive document receives the highest',
    "score. The gap (T − S) shows how much more often the teacher ranks the positive",
    "first, indicating room for the student to learn from the teacher's ranking signal.",
    "",
    "| Dataset | Student pos rank 1 | Teacher pos rank 1 | Gap (T − S) |",
    "|---|---|---|---|",
]

for dataset, s_rate, t_rate, gap, _ in rows:
    gap_pp = gap * 100
    gap_str = f"{gap_pp:.1f}pp" if gap_pp >= 0 else f"\u2212{abs(gap_pp):.1f}pp"
    lines.append(
        f"| {dataset} | {s_rate:.1%} | {t_rate:.1%} | **{gap_str}** |"
    )

lines.append("")

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
with open(args.output, "w") as f:
    f.write("\n".join(lines))

print(f"\nReport written to {args.output}")
