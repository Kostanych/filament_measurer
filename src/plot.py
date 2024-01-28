
import streamlit as st
import altair as alt


def update_rolling_plot(plot_area):
    """
    Display plot based on data from session state.
    Args:
        plot_area: place to display the plot.
    """
    try:
        min_value = st.session_state.df_points["values"].min()
        max_value = st.session_state.df_points["values"].max()
        # print(st.session_state.df_points)
        points = (
            alt.Chart(st.session_state.df_points)
            .mark_line()
            .encode(
                x=alt.X("frame"),
                y=alt.Y(
                    "values:Q",
                    scale=alt.Scale(domain=[min_value - 0.2, max_value + 0.2]),
                ),
                color="seconds_count:N",
            )
            .properties(width=1000)
            .configure_axis(labelFontSize=20, titleFontSize=20)
            .configure_legend(titleFontSize=20)
        )
        # Update plot every quarter of a second
        # if st.session_state.df_points["frame"].max() % 6 == 0:
        #     plot_area.altair_chart(points)
        plot_area.altair_chart(points)
    except Exception as e:
        print(repr(e))
