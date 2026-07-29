"""
The app must at least start.

Both times the app was broken on startup it was an import or an API that no
longer existed, and nothing in the repository noticed. This test runs the real
script through the Streamlit test runner and fails on any exception.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "src" / "gui_streamlit.py"


@pytest.fixture
def app():
    """The application script, freshly started"""
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_app_starts_without_exceptions(app):
    """A missing dependency or a removed Streamlit API shows up right here"""
    assert not app.exception


def test_control_panel_is_rendered(app):
    """The controls the operator works with are on the screen"""
    assert app.title[0].value == "Filament Measurer"

    buttons = [button.label for button in app.button]
    assert "Play" in buttons
    assert "Stop" in buttons

    assert app.radio[0].label == "Input Source"
    assert app.selectbox[0].label == "Update Interval"


def test_reference_defaults_to_the_common_filament(app):
    """1.75 mm is the filament the machine normally runs"""
    assert app.number_input[0].value == pytest.approx(1.75)


def test_stop_on_idle_app_is_harmless(app):
    """Pressing Stop before anything was played must not raise"""
    app.button(key="stop_button").click().run()

    assert not app.exception
    assert app.session_state["play"] is False
