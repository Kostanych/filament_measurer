"""Measurement must survive any frame a camera can produce"""

import time

import cv2
import numpy as np
import pytest

import frames
from config import config
from image_processor import (
    blank_mask,
    calculate_pixel_multiplier,
    fit_filament_line,
    keep_filament_contour,
    measure_angle,
    measure_filament,
    normalize_frame,
    process_image,
    to_display,
)

BROKEN_IDS = [name for name, _ in frames.BROKEN_FRAMES]
BROKEN_VALUES = [frame for _, frame in frames.BROKEN_FRAMES]


@pytest.mark.parametrize("frame", BROKEN_VALUES, ids=BROKEN_IDS)
def test_measure_filament_never_raises(frame):
    """A broken frame costs one measurement, not the whole run"""
    source, width_pxl, width_mm = measure_filament(frame, width_multiplier=0.005)

    assert source is not None
    assert np.isfinite(width_pxl)
    assert np.isfinite(width_mm)
    assert width_pxl >= 0


@pytest.mark.parametrize("frame", BROKEN_VALUES, ids=BROKEN_IDS)
def test_process_image_never_raises(frame):
    """process_image always returns a mask and a finite thickness"""
    mask, thickness = process_image(frame, add_info=True)

    assert mask is not None
    assert np.isfinite(thickness)


def test_unusable_frame_gives_zero_readings():
    """Nothing to measure means zeros on the screen, not a crash"""
    for frame in (None, frames.empty_frame()):
        _, width_pxl, width_mm = measure_filament(frame, width_multiplier=0.005)
        assert width_pxl == 0
        assert width_mm == 0


def test_white_frame_finds_no_filament():
    """An overexposed frame holds no dark pixels"""
    mask, thickness = process_image(frames.white_frame(), add_info=False)

    assert thickness == 0
    assert measure_angle(mask) is None


def test_measured_width_matches_the_drawn_filament():
    """A filament drawn 14 px wide must be measured as roughly 14 px"""
    _, width_pxl, _ = measure_filament(
        frames.filament_frame(thickness=14), width_multiplier=1
    )

    assert width_pxl == pytest.approx(14, abs=1.5)


@pytest.mark.parametrize("angle", [0, 10, 20, 30])
def test_measured_angle_matches_the_drawn_tilt(angle):
    """The fitted angle follows the tilt of the filament"""
    mask, _ = process_image(frames.filament_frame(angle=angle), add_info=False)

    assert measure_angle(mask) == pytest.approx(angle, abs=1.0)


def test_tilt_correction_is_applied_exactly_once():
    """
    A tilted filament is crossed by a vertical scan line along a longer chord,
    and the cosine of the tilt compensates for it. Applying that cosine twice
    used to shrink the reading the more the filament was tilted.
    """
    widths = [
        measure_filament(frames.filament_frame(angle=angle), width_multiplier=1)[1]
        for angle in (0, 10, 20, 30)
    ]

    assert max(widths) / min(widths) - 1 < 0.05


def test_width_in_mm_follows_width_in_pixels():
    """Millimetres are pixels times the calibration, with no extra corrections"""
    _, width_pxl, width_mm = measure_filament(
        frames.filament_frame(angle=20), width_multiplier=0.125
    )

    assert width_mm == pytest.approx(width_pxl * 0.125)


@pytest.mark.parametrize("spots,radius", [(12, 18), (30, 25)])
def test_dirt_is_not_measured_as_filament(spots, radius):
    """
    Dust on the glass is as dark as the filament. Counting it in used to
    inflate the reading several times over on a dirty frame.

    The first case is a plausibly dirty lens, the second one is deliberately
    far worse than the real setup ever gets and is here to keep the margin.
    """
    _, width_pxl, _ = measure_filament(
        frames.dirty_frame(thickness=14, spots=spots, spot_radius=radius),
        width_multiplier=1,
    )

    assert width_pxl == pytest.approx(14, abs=1.5)


def test_dirt_larger_than_the_filament_is_still_dropped():
    """
    A stress case rather than a picture of the real setup: it pins down why
    the filament is chosen by reaching both edges of the frame and not by
    being the biggest dark object. Specks touching each other merge into a
    blob that can outweigh a thin filament, and then size alone picks the dirt.
    """
    frame = frames.filament_frame(thickness=10)
    cv2.circle(frame, (frames.WIDTH // 2, 100), 70, (0, 0, 0), -1)
    blob_area = np.pi * 70**2
    assert blob_area > 10 * frames.WIDTH, "the blob must outweigh the filament"

    _, width_pxl, _ = measure_filament(frame, width_multiplier=1)

    assert width_pxl == pytest.approx(10, abs=1.5)


def test_dirt_does_not_skew_the_angle():
    """The tilt is fitted into the filament alone, not into the dirt around it"""
    mask, _ = process_image(frames.dirty_frame(angle=15), add_info=False)

    assert measure_angle(mask) == pytest.approx(15, abs=1.0)


def test_clean_frame_is_measured_the_same_with_and_without_filtering(monkeypatch):
    """Dropping the dirt must not change what a clean frame reads"""
    frame = frames.filament_frame(thickness=14, angle=10)

    _, filtered, _ = measure_filament(frame.copy(), width_multiplier=1)
    monkeypatch.setattr(config, "FILTER_FILAMENT_CONTOUR", False)
    _, unfiltered, _ = measure_filament(frame.copy(), width_multiplier=1)

    assert filtered == pytest.approx(unfiltered, abs=0.1)


def test_filtering_survives_a_frame_without_anything_dark():
    """An overexposed frame has no contours to choose from"""
    mask = blank_mask()

    assert keep_filament_contour(mask) is not None
    assert np.array_equal(keep_filament_contour(mask), mask)


def test_filtering_keeps_the_filament_and_drops_the_rest():
    """Only one dark object is left on the mask"""
    dirty_mask, _ = process_image(frames.dirty_frame(), add_info=False)

    contours, _ = cv2.findContours(
        cv2.bitwise_not(dirty_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    assert len(contours) == 1


def test_dark_frame_does_not_get_a_line_fitted():
    """
    A frame that is mostly dark is broken, not a filament. Fitting a line into
    every one of its pixels gives a meaningless angle and is slow enough to
    stall the video loop, so the fit is skipped.
    """
    mask, _ = process_image(frames.black_frame(), add_info=False)

    started = time.perf_counter()
    params = fit_filament_line(mask)
    elapsed = time.perf_counter() - started

    assert params is None
    assert elapsed < 0.05


def test_unknown_angle_keeps_the_width_unchanged():
    """Without a known tilt the width is reported as measured"""
    assert calculate_pixel_multiplier(None) == 1


@pytest.mark.parametrize("frame", BROKEN_VALUES, ids=BROKEN_IDS)
def test_normalize_frame_returns_bgr_or_nothing(frame):
    """Whatever arrives is either a usable BGR frame or an honest None"""
    normalized = normalize_frame(frame)

    if normalized is not None:
        assert normalized.dtype == np.uint8
        assert normalized.ndim == 3
        assert normalized.shape[2] == 3


def test_grayscale_and_bgra_frames_are_measured():
    """Frames with an unexpected channel count are converted, not dropped"""
    for frame in (frames.grayscale_frame(), frames.bgra_frame()):
        _, width_pxl, _ = measure_filament(frame, width_multiplier=1)
        assert width_pxl == pytest.approx(14, abs=1.5)


def test_to_display_swaps_red_and_blue():
    """OpenCV keeps frames in BGR while Streamlit shows them as RGB"""
    blue_in_bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    blue_in_bgr[0, 0] = (255, 0, 0)

    assert tuple(to_display(blue_in_bgr)[0, 0]) == (0, 0, 255)


def test_to_display_leaves_the_mask_alone():
    """A single channel mask has no channels to swap"""
    mask = blank_mask()

    assert to_display(mask).shape == mask.shape
