import logging


import streamlit as st

from plot import update_rolling_plot
from utils import check_variables, get_logger


logging_level = logging.DEBUG


def stop():
    """Stop the cap"""
    logger = get_logger("STOP VIDEO", level=logging_level)

    check_variables()
    logger.info(f"BUTTON Stop")
    st.session_state.play = False
    logger.info(f"st.session_state.play:   {st.session_state.play}")
    if st.session_state.cap:
        st.session_state.cap.release()
        logger.debug('Cap released')

    # st.session_state.width_list = []

    if st.session_state["width_list"]:
        update_rolling_plot(st.session_state["plot_area"])



