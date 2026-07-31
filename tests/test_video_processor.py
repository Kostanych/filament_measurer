"""The video loop must run a whole file through without falling over"""

import cv2
import numpy as np
import pytest
import streamlit as st

import frames
from utils import AppState
from video_processor import VideoProcessor


class UiStub:
    """Stands in for a Streamlit placeholder and records what was shown"""

    def __init__(self):
        self.updates = 0

    def image(self, *args, **kwargs):
        self.updates += 1

    def markdown(self, *args, **kwargs):
        self.updates += 1

    def altair_chart(self, *args, **kwargs):
        self.updates += 1


@pytest.fixture
def session():
    """Session state wired to stubbed UI placeholders"""
    st.session_state.clear()
    state = AppState()
    for area in (
        "vid_area",
        "width_pxl_area",
        "width_mm_area",
        "rolling_1s_markdown",
        "rolling_10s_markdown",
        "difference_markdown",
        "plot_area",
    ):
        st.session_state[area] = UiStub()
    yield state
    st.session_state.clear()


def write_video(path, frame_list, fps=25):
    """Write frames into a real file so the loop reads it through OpenCV"""
    height, width = frame_list[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    assert writer.isOpened(), "MJPG writer is not available"
    for frame in frame_list:
        writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture
def messy_video(tmp_path):
    """
    A recording of a camera behaving badly.

    Blank, dark and noisy frames are mixed into the normal ones the same way
    they appear when the camera is starting up or losing the scene.
    """
    frame_list = []
    for i in range(10):
        frame_list.extend(
            [
                frames.filament_frame(angle=i),
                frames.white_frame(),
                frames.black_frame(),
                frames.noisy_frame(),
            ]
        )
    return write_video(tmp_path / "messy.avi", frame_list)


def play(session, video_path):
    """Run the whole video through the processor"""
    st.session_state.source = "File"
    st.session_state.video_path = video_path
    st.session_state.play = True
    VideoProcessor().play_or_continue_video()


def test_messy_video_plays_to_the_end(session, messy_video):
    """Blank, dark and noisy frames must not stop the run"""
    play(session, messy_video)

    assert st.session_state.play is False
    assert st.session_state.cap is None
    assert len(st.session_state.width_list) > 0


def test_every_frame_produces_a_measurement(session, tmp_path):
    """
    Readings keep coming for the frames that do hold a filament.

    The calibration reads one frame of its own before the loop starts, so the
    measurements are counted from the second frame on.
    """
    video = write_video(
        tmp_path / "clean.avi", [frames.filament_frame() for _ in range(20)]
    )

    play(session, video)

    assert len(st.session_state.width_list) == 19
    assert all(np.isfinite(width) for width in st.session_state.width_list)
    assert min(st.session_state.width_list) > 0


def test_calibration_pins_the_reading_to_the_reference(session, tmp_path):
    """
    Calibration turns pixels into millimetres against a known standard, so a
    steady filament reads back as exactly the reference width.
    """
    video = write_video(
        tmp_path / "steady.avi", [frames.filament_frame() for _ in range(10)]
    )
    st.session_state.reference = 1.75

    play(session, video)

    assert st.session_state.width_list[-1] == pytest.approx(1.75, abs=0.01)


def test_blank_video_reports_zero_and_stops(session, tmp_path):
    """A camera showing nothing gives zero width instead of an error"""
    video = write_video(
        tmp_path / "blank.avi", [frames.white_frame() for _ in range(10)]
    )

    play(session, video)

    assert st.session_state.play is False
    assert set(st.session_state.width_list) == {0.0}


def test_missing_video_file_does_not_start_playback(session, tmp_path):
    """An unplugged camera or a deleted file only turns the playback off"""
    play(session, str(tmp_path / "there_is_no_such_file.avi"))

    assert st.session_state.play is False
    assert st.session_state.cap is None


def test_replay_does_not_mix_in_the_previous_run(session, tmp_path):
    """The video restarts from the beginning, so the measurements restart too"""
    video = write_video(
        tmp_path / "clean.avi", [frames.filament_frame() for _ in range(15)]
    )

    play(session, video)
    play(session, video)

    # 15 frames in the file, one of them spent on the calibration
    assert len(st.session_state.width_list) == 14
