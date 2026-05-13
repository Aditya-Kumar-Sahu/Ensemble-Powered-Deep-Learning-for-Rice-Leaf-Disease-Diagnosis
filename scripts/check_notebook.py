#!/usr/bin/env python3
import json
import sys
import os


def main():
    nb_path = os.path.join(os.getcwd(), "Rice_Leafs_Disease_15.ipynb")
    if not os.path.exists(nb_path):
        print(f"Notebook not found: {nb_path}", file=sys.stderr)
        sys.exit(2)
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print("Failed to read notebook:", e, file=sys.stderr)
        sys.exit(2)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code" and cell.get("outputs"):
            print("Notebook has outputs")
            sys.exit(1)
    print("No outputs found")
    sys.exit(0)


if __name__ == "__main__":
    main()
