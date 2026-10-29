"""Append a test-count table for JUnit XML reports to the GitHub job summary.

    python tools/junit_summary.py "<title>" report.xml [more.xml ...]

Writes to ``$GITHUB_STEP_SUMMARY`` when set, else to stdout. Missing reports are listed as such.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def counts(report: Path) -> dict[str, float]:
    root = ET.parse(report).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    total = {k: 0.0 for k in ("tests", "failures", "errors", "skipped", "time")}
    for suite in suites:
        for k in total:
            total[k] += float(suite.get(k, 0))
    total["passed"] = total["tests"] - total["failures"] - total["errors"] - total["skipped"]
    return total


def main(title: str, reports: list[str]) -> int:
    lines = [
        f"### {title}",
        "",
        "| report | passed | failed | errors | skipped | time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in reports:
        path = Path(name)
        if not path.exists():
            lines.append(f"| `{path.name}` | no report | | | | |")
            continue
        c = counts(path)
        mark = "pass" if c["failures"] + c["errors"] == 0 else "FAIL"
        lines.append(
            f"| `{path.name}` ({mark}) | {c['passed']:.0f} | {c['failures']:.0f} | "
            f"{c['errors']:.0f} | {c['skipped']:.0f} | {c['time']:.1f} s |"
        )
    text = "\n".join(lines) + "\n\n"
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2:]))
