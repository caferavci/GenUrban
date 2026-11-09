"""Private runtime configuration for the original notebook workflow."""

from dataclasses import dataclass
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parent

# File names retain the original batch labels used by the notebook.
CITY_SPECS = {
    "NYC": ("New-york-city-boroughs", "new-york-city-boroughs", "36", "061",
            (-74.02, 40.70, -73.93, 40.80)),
    "Las Vegas": ("LasVegas", "lasvagas", "32", "003",
                  (-115.35, 36.00, -115.00, 36.30)),
    "San Francisco": ("San_Francisco", "sanfrancisco", "06", "075",
                      (-122.52, 37.70, -122.35, 37.83)),
    "Minneapolis": ("Minneapolis", "minneapolis", "27", "053",
                    (-93.29, 44.96, -93.24, 45.00)),
}


@dataclass(repr=False)
class WorkflowSettings:
    workspace: Path
    cities: dict
    llm_configs: list
    show_plots: bool


def selected_cities():
    requested = [name.strip() for name in os.getenv("GENURBAN_CITIES", "").split(",") if name.strip()]
    names = requested or list(CITY_SPECS)
    unknown = sorted(set(names) - set(CITY_SPECS))
    if unknown:
        raise ValueError("GENURBAN_CITIES contains an unknown city; see docs/INPUTS.md.")
    cities = {}
    for name in names:
        folder, stem, state, county, bbox = CITY_SPECS[name]
        cities[name] = {
            "train": [f"{folder}/{stem}.geojson Flow {batch}.csv"
                      for batch in ("6-9", "10-13", "14-17", "18-21")],
            "test": f"{folder}/{stem}.geojson Flow 22-25.csv",
            "state": state, "county": county, "bbox": bbox,
            "geojson": f"regions/{stem}.geojson",
        }
    return cities


def load_settings(validate_inputs=True):
    # Never override environment variables explicitly supplied by the caller.
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)

    key = os.getenv("LLM_API_KEY", "").strip()
    base = os.getenv("LLM_BASE_URL", "").strip()
    models = [name.strip() for name in os.getenv("GENURBAN_MODELS", "").split(",") if name.strip()]
    missing = [name for name, value in [("LLM_API_KEY", key), ("LLM_BASE_URL", base),
                                       ("GENURBAN_MODELS", models)] if not value]
    if missing:
        raise ValueError("Set these private environment variables before running: " + ", ".join(missing))
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("LLM_BASE_URL must be a service base URL without embedded credentials.")
    if any(any(char in name for char in "\\\n\r\t") for name in models):
        raise ValueError("GENURBAN_MODELS contains an invalid model identifier.")
    workspace = Path(os.getenv("GENURBAN_WORKSPACE", "private_workspace"))
    if not workspace.is_absolute():
        workspace = PROJECT_ROOT / workspace
    settings = WorkflowSettings(
        workspace=workspace.resolve(), cities=selected_cities(),
        llm_configs=[{"model": name, "api_key": key, "base_url": base} for name in dict.fromkeys(models)],
        show_plots=os.getenv("GENURBAN_PLOTS", "0").lower() in {"1", "true", "yes"},
    )
    if validate_inputs:
        errors = input_errors(settings)
        if errors:
            raise ValueError("Input preparation is incomplete:\n- " + "\n- ".join(errors))
    return settings


def input_errors(settings):
    """Check file presence and basic contracts without making network requests."""
    errors = []
    import csv
    import json
    for city, data in settings.cities.items():
        for relative in data["train"] + [data["test"]]:
            path = settings.workspace / relative
            if not path.is_file():
                errors.append(f"{city}: missing {relative}")
                continue
            with path.open(encoding="utf-8-sig", newline="") as handle:
                columns = csv.DictReader(handle).fieldnames or []
            if not {"Origin", "Destination"}.issubset(columns):
                errors.append(f"{city}: OD tables require Origin and Destination columns.")
            if "flow_count" not in columns and not any("Trips" in col for col in columns):
                errors.append(f"{city}: OD tables require flow_count or Trips columns.")
        regions = settings.workspace / data["geojson"]
        if not regions.is_file():
            errors.append(f"{city}: missing {data['geojson']}")
        else:
            try:
                payload = json.loads(regions.read_text(encoding="utf-8-sig"))
                if payload.get("type") != "FeatureCollection" or not payload.get("features"):
                    raise ValueError("Empty or invalid regions")
            except (ValueError, AttributeError):
                errors.append(f"{city}: regions must be a nonempty GeoJSON FeatureCollection.")
        acs = settings.workspace / "ACS" / f"{data['state']}_{data['county']}.csv"
        if acs.is_file():
            with acs.open(encoding="utf-8-sig", newline="") as handle:
                rows = csv.DictReader(handle)
                row = next(rows, {})
            try:
                if float(row["Estimate!!Total!!Total population"]) <= 0:
                    raise ValueError("Nonpositive total")
            except (ValueError, KeyError):
                errors.append(f"{city}: ACS input requires a positive total-population estimate.")
        elif not ((settings.workspace / "census_mapping.csv").is_file()
                  and os.getenv("CENSUS_API_BASE_URL", "").strip()):
            errors.append(f"{city}: supply its ACS CSV, or a private Census service configuration and mapping.")
    return errors


def safe_error(error):
    """Avoid exposing configured credentials or service addresses in run messages."""
    text = str(error)
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "CENSUS_API_KEY", "CENSUS_API_BASE_URL"):
        value = os.getenv(name, "")
        if value:
            text = text.replace(value, "[redacted]")
    text = re.sub(r"https?://[^\s'\"]+", "[service address redacted]", text)
    return text


def model_filename(model_name):
    """Make an export label portable without changing the provider model ID."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '-', model_name).strip(' .') or 'model'
