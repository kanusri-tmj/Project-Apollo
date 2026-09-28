"""Deployment-contract tests for the Render setup.

A deploy can fail in ways the unit tests never notice: a start command that
binds the wrong port, a health check pointing at a route that does not exist, a
runtime dependency set that is missing the WSGI server, or — most common of all
— trained artifacts that are gitignored and therefore absent from the build.

These tests pin those contracts down so the Render configuration cannot silently
drift from the code.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src import config, predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_VARIANTS = ("baseline", "ridge_l2", "lasso_l1")


# --------------------------------------------------------------------------- #
# Git-tracking rules
# --------------------------------------------------------------------------- #
def _gitignore_rules() -> list[str]:
    """Return the active (non-comment, non-empty) .gitignore patterns."""
    lines = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _is_ignored(relative_path: str, rules: list[str]) -> bool:
    """Approximate Git's last-match-wins semantics for the patterns in use."""
    ignored = False
    for rule in rules:
        negate = rule.startswith("!")
        pattern = rule[1:] if negate else rule
        if fnmatch.fnmatch(relative_path, pattern):
            ignored = not negate
    return ignored


def test_stable_artifacts_are_not_gitignored():
    """Render rebuilds from Git with an ephemeral disk, so these must be tracked."""
    rules = _gitignore_rules()
    for variant in DEPLOY_VARIANTS:
        assert not _is_ignored(f"models/{variant}_latest.joblib", rules), (
            f"models/{variant}_latest.joblib is gitignored; a Git-based deploy "
            "would boot without a model and serve 503 to every request."
        )
        assert not _is_ignored(f"models/{variant}_metadata.json", rules)


def test_timestamped_artifacts_stay_out_of_git():
    """The immutable audit trail is a duplicate copy and should not be committed."""
    rules = _gitignore_rules()
    assert _is_ignored("models/ridge_l2_v1.0.0_20260926T062633Z.joblib", rules)


def test_readme_images_are_not_gitignored():
    """A gitignored screenshot or figure renders as a broken image on GitHub."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    embedded = re.findall(r"!\[[^\]]*\]\(([^)\s]+)", readme)
    assert embedded, "README embeds no images - did the screenshots get removed?"

    rules = _gitignore_rules()
    for image in embedded:
        if image.startswith(("http://", "https://")):
            continue
        assert (PROJECT_ROOT / image).exists(), f"README references a missing image: {image}"
        assert not _is_ignored(image, rules), (
            f"README embeds {image}, but it is gitignored, so it would not exist "
            "in the repository and the image would be broken."
        )


def test_dataset_snapshots_are_not_required_at_runtime():
    """The pip-installed service must not depend on generated data files."""
    rules = _gitignore_rules()
    assert _is_ignored("data/processed/heart_disease_clean.csv", rules)


# --------------------------------------------------------------------------- #
# WSGI entry point
# --------------------------------------------------------------------------- #
def test_wsgi_entry_point_exposes_the_flask_app():
    flask_app = pytest.importorskip("app.flask_app")
    wsgi = pytest.importorskip("wsgi")
    assert wsgi.app is flask_app.app


# --------------------------------------------------------------------------- #
# render.yaml
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def blueprint() -> str:
    return (PROJECT_ROOT / "render.yaml").read_text(encoding="utf-8")


def test_blueprint_serves_the_app_with_gunicorn(blueprint):
    assert "wsgi:app" in blueprint
    assert "gunicorn" in blueprint


def test_blueprint_binds_the_injected_port(blueprint):
    # Binding a hardcoded port is the classic Render "no open ports detected" bug.
    assert "0.0.0.0:$PORT" in blueprint


def test_blueprint_health_check_matches_a_real_route(blueprint):
    """A health check pointing at a 404 route fails every deploy."""
    assert "healthCheckPath: /health" in blueprint
    flask_app = pytest.importorskip("app.flask_app")
    routes = {str(rule) for rule in flask_app.app.url_map.iter_rules()}
    assert "/health" in routes


def test_blueprint_installs_the_deploy_requirements(blueprint):
    assert "requirements-render.txt" in blueprint
    assert (PROJECT_ROOT / "requirements-render.txt").exists()


def test_blueprint_python_version_matches_python_version_file(blueprint):
    """The pinned interpreter must not drift from the runtime's pin."""
    pinned = (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip()
    assert pinned, ".python-version must pin a concrete Python version."
    assert f'value: "{pinned}"' in blueprint


# --------------------------------------------------------------------------- #
# Runtime dependency set
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def deploy_packages() -> set[str]:
    """The packages *actually* requested, ignoring comments and blank lines."""
    lines = (PROJECT_ROOT / "requirements-render.txt").read_text(encoding="utf-8").splitlines()
    return {
        line.split(">")[0].split("=")[0].strip().lower()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


@pytest.mark.parametrize("package", ["flask", "gunicorn", "scikit-learn", "joblib", "numpy", "pandas", "scipy"])
def test_deploy_requirements_cover_the_runtime_path(deploy_packages, package):
    assert package in deploy_packages


@pytest.mark.parametrize("package", ["matplotlib", "seaborn", "jupyter", "streamlit", "pytest"])
def test_deploy_requirements_exclude_development_only_packages(deploy_packages, package):
    assert package not in deploy_packages


def test_runtime_import_path_needs_no_development_only_packages():
    """The deployed service installs *only* requirements-render.txt.

    Import the real application in a fresh interpreter with every
    development-only package made unimportable. If a runtime module ever grows an
    ``import seaborn`` or ``import streamlit``, this fails here instead of
    taking the Render build up in smoke.
    """
    script = textwrap.dedent(
        """
        import importlib.abc, sys

        DEV_ONLY = {"matplotlib", "seaborn", "streamlit", "IPython", "jupyter",
                    "nbformat", "nbconvert", "pytest"}

        class BlockDevOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in DEV_ONLY:
                    raise ImportError("blocked development-only package: " + fullname)
                return None

        sys.meta_path.insert(0, BlockDevOnly())

        import wsgi

        client = wsgi.app.test_client()
        record = {"age": 57, "sex": 1, "chest_pain_type": 4, "resting_bp": 140,
                  "cholesterol": 260, "fasting_blood_sugar": 0, "resting_ecg": 1,
                  "max_heart_rate": 140, "exercise_angina": 1, "st_depression": 2.0,
                  "st_slope": 2, "num_major_vessels": 2, "thalassemia": 7}
        for path, payload in (("/health", None), ("/", None), ("/model-info", None)):
            assert client.get(path).status_code == 200, path
        for path in ("/predict", "/explain"):
            response = client.post(path, json={"features": record})
            assert response.status_code == 200, (path, response.get_data(as_text=True)[:200])
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "The Flask runtime depends on a development-only package that the "
        "deploy requirements do not install.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# --------------------------------------------------------------------------- #
# Artifacts the running service loads
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant", DEPLOY_VARIANTS)
def test_deployed_artifacts_exist_and_load(variant):
    """Every variant is loaded at startup so the Live Simulation can contrast them."""
    path = config.MODELS_DIR / f"{variant}_latest.joblib"
    assert path.exists(), f"{path.name} is missing; run `python -m src.train` and commit it."
    model, info = predict.load_model(path)
    assert info.get("version"), f"{path.name} carries no version metadata."
    assert hasattr(model, "predict_proba")


def test_primary_variant_is_available_for_the_headline_verdict():
    """`/explain` falls back through config.PRIMARY_VARIANT for its verdict."""
    assert (config.MODELS_DIR / f"{config.PRIMARY_VARIANT}_latest.joblib").exists()
