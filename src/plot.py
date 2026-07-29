"""Plotting utilities for visualization"""

import logging

import altair as alt
import streamlit as st

from config import config
from utils import get_logger

logging_level = logging.DEBUG

logger = get_logger("PLOT", level=logging_level)


def update_rolling_plot(plot_area):
    """
    Display plot based on data from session state

    Args:
        plot_area: Streamlit container to display the plot
    """
    df_points = st.session_state.df_points
    if df_points is None or df_points.empty:
        return

    try:
        min_value = df_points["values"].min()
        max_value = df_points["values"].max()

        points = (
            alt.Chart(df_points)
            .mark_line()
            .encode(
                x=alt.X("frame"),
                y=alt.Y(
                    "values:Q",
                    scale=alt.Scale(
                        domain=[
                            min_value - config.PLOT_Y_MARGIN,
                            max_value + config.PLOT_Y_MARGIN,
                        ]
                    ),
                ),
                color="seconds_count:N",
            )
            .properties(width=config.PLOT_WIDTH)
            .configure_axis(
                labelFontSize=config.FONT_SIZE_LARGE,
                titleFontSize=config.FONT_SIZE_LARGE,
            )
            .configure_legend(titleFontSize=config.FONT_SIZE_LARGE)
        )
        plot_area.altair_chart(points)
    except Exception as e:
        logger.warning(f"Could not draw the plot: {e!r}")
