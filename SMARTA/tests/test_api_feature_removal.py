"""Regression checks for the removed fleet simulator API surface."""

from api import app


def test_simulator_routes_are_not_registered_and_core_routes_remain():
    paths = {route.path for route in app.routes if hasattr(route, "path")}

    assert not any(path.startswith("/simulator") for path in paths)
    assert "/health" in paths
    assert "/chat" in paths
    assert "/predict" in paths
    assert "/predict/metadata" in paths
    assert "/experimental-pdm" not in paths  # Explicit chat command, not an HTTP route.
