"""Execute the original notebook's Python cells without saving notebook outputs."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from workflow_config import PROJECT_ROOT, load_settings, safe_error


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the GenUrban population, activity, and OD workflow.")
    parser.add_argument("--check", action="store_true", help="Check private configuration and inputs without network calls.")
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
    except (ValueError, OSError) as exc:
        print(safe_error(exc), file=sys.stderr)
        return 2
    packages = {"autogen": "autogen-agentchat", "numpy": "numpy", "pandas": "pandas",
                "geopandas": "geopandas", "sklearn": "scikit-learn", "shapely": "shapely",
                "matplotlib": "matplotlib", "seaborn": "seaborn", "requests": "requests",
                "overturemaps": "overturemaps"}
    missing = [dist for module, dist in packages.items() if importlib.util.find_spec(module) is None]
    if missing:
        print("Install the required packages: " + ", ".join(missing), file=sys.stderr)
        return 2
    for city in settings.cities:
        cache = settings.workspace / "Overture" / f"{city.replace(' ', '_')}_buildings.geojson"
        if not cache.is_file():
            print(f"{city}: building cache absent; execution will request Overture data.")
    print(f"Preflight passed: {len(settings.cities)} cities, {len(settings.llm_configs)} configured model(s).")
    if args.check:
        print("No model calls or data downloads were made.")
        return 0
    print("Starting the complete workflow. Model calls may incur charges.")
    if not settings.show_plots:
        os.environ.setdefault("MPLBACKEND", "Agg")
    original_directory = Path.cwd()
    namespace = {"__name__": "__main__", "__file__": str(PROJECT_ROOT / "MulitiCityStudy.ipynb")}
    try:
        os.chdir(PROJECT_ROOT)
        notebook = json.loads((PROJECT_ROOT / "MulitiCityStudy.ipynb").read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"MulitiCityStudy.ipynb:cell_{index}", "exec"), namespace)
        rows = namespace.get("comparison_results", [])
        if any(row.get("Status") == "failed" for row in rows):
            print("One or more workflow configurations failed; inspect the generated status column.", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:
        print("Workflow stopped: " + safe_error(exc), file=sys.stderr)
        return 1
    finally:
        os.chdir(original_directory)


if __name__ == "__main__":
    raise SystemExit(main())
