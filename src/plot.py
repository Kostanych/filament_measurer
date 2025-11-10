"""Plotting utilities for visualization"""

import streamlit as st
import altair as alt
from config import config


def update_rolling_plot(plot_area):
    """
    Display plot based on data from session state

    Args:
        plot_area: Streamlit container to display the plot
    """
    try:
        min_value = st.session_state.df_points["values"].min()
        max_value = st.session_state.df_points["values"].max()

        points = (
            alt.Chart(st.session_state.df_points)
            .mark_line()
            .encode(
                x=alt.X("frame"),
                y=alt.Y(
                    "values:Q",
                    scale=alt.Scale(
                        domain=[min_value - config.PLOT_Y_MARGIN,
                               max_value + config.PLOT_Y_MARGIN]
                    ),
                ),
                color="seconds_count:N",
            )
            .properties(width=config.PLOT_WIDTH)
            .configure_axis(
                labelFontSize=config.FONT_SIZE_LARGE,
                titleFontSize=config.FONT_SIZE_LARGE
            )
            .configure_legend(titleFontSize=config.FONT_SIZE_LARGE)
        )
        plot_area.altair_chart(points)
    except Exception as e:
        print(repr(e))
