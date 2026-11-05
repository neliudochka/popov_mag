from pathlib import Path
import pandas as pd

root_dir = Path(__file__).parent.resolve()
readme_path = root_dir / "../../README.md"
summary_path = root_dir / "reports/summary.csv"

if not summary_path.exists():
    raise FileNotFoundError(f"Summary CSV not found: {summary_path}")
if not readme_path.exists():
    raise FileNotFoundError(f"README.md not found: {readme_path}")

df = pd.read_csv(summary_path, delimiter=",")
table_md = df.to_markdown(index=False)
table_lines = [line + "\n" for line in table_md.splitlines()]

with open(readme_path, "r") as f:
    lines = f.readlines()

target_header = "segmentation evaluation"
try:
    summary_idx = next(
        i for i, line in enumerate(lines) if target_header in line.lower()
    )
except StopIteration:
    raise RuntimeError(f"Could not find a '{target_header}' section in {readme_path}")


lines = (
    lines[:summary_idx + 1]
    + ["\n"]
    + table_lines
    + ["\n"]
)

with open(readme_path, "w") as f:
    f.writelines(lines)

print(f"README.md updated successfully at section '{target_header}'")
