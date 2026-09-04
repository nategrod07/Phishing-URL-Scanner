"""Guards against the README drifting away from the code.

Documentation rots quietly: a flag gets renamed, a module moves, a
dependency is used but never declared, and the README goes on confidently
describing something that no longer exists. These tests turn that class of
mistake into a failing suite instead of something a reader discovers.

Deliberately not asserted here: measured numbers (accuracy, timings). Those
can only be checked by re-running the benchmark, so they are verified by
hand when the model changes.
"""
import re
from pathlib import Path

import pytest

try:  # tomllib is stdlib from 3.11; tomli is the backport for 3.9/3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on interpreter version
    import tomli as tomllib

from phishing_ml import predict, train
from phishing_ml.pipeline import CLASSIFIERS

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
README_TEXT = README.read_text()

# Fenced code blocks hold the commands a reader will actually run.
CODE_BLOCKS = "\n".join(re.findall(r"```(?:bash|python)?\n(.*?)```", README_TEXT, re.DOTALL))


def parser_flags(parser) -> set:
    return {opt for action in parser._actions for opt in action.option_strings}


ALL_FLAGS = parser_flags(train.build_parser()) | parser_flags(predict.build_parser())


class TestDocumentedFlags:
    def test_every_flag_in_the_readme_exists(self):
        documented = set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", README_TEXT))
        unknown = documented - ALL_FLAGS
        assert not unknown, f"README documents flags that no parser defines: {sorted(unknown)}"

    def test_readme_mentions_every_flag_the_cli_accepts(self):
        # --help is argparse's own; everything else is ours to document.
        undocumented = ALL_FLAGS - set(re.findall(r"--[a-z][a-z0-9-]+", README_TEXT)) - {"-h", "--help"}
        assert not undocumented, f"CLI flags missing from the README: {sorted(undocumented)}"


class TestDocumentedNames:
    def test_classifier_names_are_real(self):
        for name in re.findall(r"--classifier\s+([a-z_]+)", CODE_BLOCKS):
            assert name in CLASSIFIERS, f"README names an unknown classifier: {name}"

    def test_every_advertised_classifier_is_mentioned(self):
        for name in CLASSIFIERS:
            assert name in README_TEXT, f"Classifier {name} is not documented"

    def test_linked_files_exist(self):
        for target in re.findall(r"\[[^\]]+\]\(((?!https?:)[^)]+)\)", README_TEXT):
            path = target.split("#")[0]
            assert (REPO_ROOT / path).exists(), f"README links a missing file: {path}"

    def test_referenced_modules_are_importable(self):
        for module in re.findall(r"python -m (phishing_ml\.[a-z_]+)", CODE_BLOCKS):
            __import__(module)


class TestDeclaredDependencies:
    @pytest.fixture(scope="class")
    def declared(self):
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        project = pyproject["project"]
        deps = list(project["dependencies"])
        for extra in project.get("optional-dependencies", {}).values():
            deps += list(extra)
        return {re.split(r"[<>=\[]", d)[0].strip().lower() for d in deps}

    def test_packages_imported_in_the_readme_are_declared(self, declared):
        for package in re.findall(r"^\s*import\s+(\w+)", CODE_BLOCKS, re.MULTILINE):
            assert package.lower() in declared, (
                f"README tells the reader to import {package!r}, but it is not a "
                f"declared dependency -- a fresh install would fail"
            )

    def test_requirements_txt_matches_pyproject(self, declared):
        listed = {
            re.split(r"[<>=\[]", line)[0].strip().lower()
            for line in (REPO_ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        assert listed <= declared, (
            f"requirements.txt lists packages absent from pyproject: {sorted(listed - declared)}"
        )
