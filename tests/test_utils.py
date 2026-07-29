"""Rolling statistics and the bounded measurement buffers"""

from collections import deque

import numpy as np
import pytest
import streamlit as st

from config import config
from utils import (
    AppState,
    last_n,
    make_result_df,
    mean_rolling,
    measurement_buffer,
)


@pytest.fixture
def session():
    """A clean session state for every test"""
    st.session_state.clear()
    state = AppState()
    yield state
    st.session_state.clear()


def test_mean_rolling_survives_empty_data():
    """No measurements yet is a normal state right after the start"""
    assert mean_rolling([], fps=24) == 0.0
    assert mean_rolling(deque(), fps=24) == 0.0


def test_mean_rolling_averages_the_last_window():
    """Only the frames of the requested window take part in the average"""
    data = list(range(100))

    assert mean_rolling(data, fps=10, seconds=1) == pytest.approx(np.mean(data[-10:]))
    assert mean_rolling(data, fps=10, seconds=10) == pytest.approx(np.mean(data))


def test_mean_rolling_uses_everything_it_has():
    """A window longer than the history averages the whole history"""
    assert mean_rolling([1, 2, 3], fps=24, seconds=10) == pytest.approx(2)


def test_mean_rolling_works_on_the_bounded_buffer():
    """The measurements are stored in a deque, not in a list"""
    buffer = measurement_buffer()
    buffer.extend(range(100))

    assert mean_rolling(buffer, fps=10, seconds=1) == pytest.approx(94.5)


def test_zero_fps_still_gives_a_number():
    """FPS may be reported as zero before the first frames are timed"""
    assert mean_rolling([1, 2, 3], fps=0) == pytest.approx(3)


def test_measurement_buffer_stops_growing():
    """A long run must not eat the memory frame by frame"""
    buffer = measurement_buffer()
    buffer.extend(range(config.MAX_MEASUREMENT_HISTORY * 2))

    assert len(buffer) == config.MAX_MEASUREMENT_HISTORY
    assert buffer[-1] == config.MAX_MEASUREMENT_HISTORY * 2 - 1


@pytest.mark.parametrize("factory", [list, deque])
def test_last_n_handles_both_sequences(factory):
    """The tail is taken the same way from a list and from a deque"""
    data = factory(range(10))

    assert last_n(data, 3) == [7, 8, 9]
    assert last_n(data, 100) == list(range(10))
    assert last_n(data, 0) == []


def test_result_df_is_empty_before_the_first_measurement(session):
    """The plot is simply not drawn while there is nothing to draw"""
    df = make_result_df()

    assert df.empty
    assert set(df.columns) == {"frame", "seconds_count", "values"}


def test_result_df_keeps_only_the_recent_seconds(session):
    """The plot shows a moving window, not the whole history"""
    st.session_state.fps = 10
    st.session_state.measurements_total = 500
    st.session_state.mean_1.extend(range(500))
    st.session_state.mean_2.extend(range(500))

    df = make_result_df(num_seconds=2)

    # 10 fps by 2 seconds, two lines on the plot
    assert len(df) == 20 * 2
    assert df["frame"].max() == 499


def test_reset_measurements_clears_the_previous_run(session):
    """Playback restarts the video, so the old values must not leak into it"""
    st.session_state.mean_1.extend(range(10))
    st.session_state.mean_2.extend(range(10))
    st.session_state.width_list.extend(range(10))
    st.session_state.measurements_total = 10

    session.reset_measurements()

    assert not len(st.session_state.mean_1)
    assert not len(st.session_state.width_list)
    assert st.session_state.measurements_total == 0
