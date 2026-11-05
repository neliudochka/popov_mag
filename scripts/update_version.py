#!/usr/bin/env python
"""Update version in pyproject.toml. Usage: python update_version.py <version>"""
import re
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python update_version.py <version>", file=sys.stderr)
        sys.exit(1)
    version = sys.argv[1]
    path = "pyproject.toml"
    text = open(path, encoding="utf-8").read()
    text = re.sub(r'version\s*=\s*"[^"]*"', f'version = "{version}"', text)
    open(path, "w", encoding="utf-8").write(text)
    print(f"Updated {path} to version {version}")

if __name__ == "__main__":
    main()
