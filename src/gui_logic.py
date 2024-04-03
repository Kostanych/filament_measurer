import logging

import streamlit as st

from plot import update_rolling_plot
from image_processor import update_title_frame
from utils import get_logger

logging_level = logging.DEBUG


def stop(app_state):
    """Stop the cap"""
    logger = get_logger("STOP VIDEO", level=logging_level)
    logger.info(f"BUTTON Stop")

    # Update title frame by the last processed frame
    update_title_frame(st.session_state['last_frame'])

    st.session_state.play = False
    logger.info(f"st.session_state.play:   {st.session_state.play}")
    if st.session_state.cap:
        st.session_state.cap.release()
        logger.debug('Cap released')

    # st.session_state.width_list = []

    if app_state.state['width_list']:
        update_rolling_plot(st.session_state["plot_area"])



