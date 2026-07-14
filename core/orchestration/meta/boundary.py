"""Adversarial boundary enforcement for the meta-layer."""

from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent.parent
_META_ROOT = _PACKAGE_ROOT

_FORBIDDEN_IMPORT_PREFIXES = (
    "harness.gates",
    "execution.live_stub",
)


class MetaBoundaryViolation(Exception):
    """Raised when the meta-layer adversarial boundary check fails."""


def _iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.py"))


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def find_meta_boundary_violations(
    *,
    meta_root: Path | None = None,
) -> list[str]:
    """Return human-readable violations for forbidden meta-layer imports."""
    root = meta_root or _META_ROOT
    violations: list[str] = []
    for path in _iter_python_files(root):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for module in sorted(_imported_modules(tree)):
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in _FORBIDDEN_IMPORT_PREFIXES
            ):
                try:
                    rel = path.relative_to(_REPO_ROOT)
                except ValueError:
                    rel = path.relative_to(root)
                violations.append(f"{rel}: forbidden import {module!r}")
    return violations


def run_meta_boundary_check(*, meta_root: Path | None = None) -> None:
    """Assert the meta-layer cannot reach gates or the real-money path."""
    violations = find_meta_boundary_violations(meta_root=meta_root)
    if violations:
        msg = "Meta-layer adversarial boundary violated:\n" + "\n".join(violations)
        raise MetaBoundaryViolation(msg)
