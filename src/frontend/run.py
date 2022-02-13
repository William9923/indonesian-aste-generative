import logging
from pathlib import Path as P

import pandas as pd
import streamlit as st

from src.frontend.download import download
from src.frontend.utils import predict, visualize_triplet_opinion, print_memory_usage, load_model

from src.loader import parse
from src.constant import Path

PRED_BATCH_SIZE = 4

configuration = {"Test-model": "Test-description"}
mode_option = ["Single", "Multi"]

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

# TODO: == Load Model ==
# TODO: Model Loading & Configuration

def main():
    config_name = st.sidebar.selectbox("Choose a configuration:", list(configuration.keys()))
    prefix = st.sidebar.text_input("Prefix:")

    path = P(Path.RESOURCES, "details.md")
    st.markdown(path.read_text())
    my_expander = st.expander("Click here for description of configurations")
    with my_expander:
        st.json(configuration)
    mode = st.selectbox("Choose mode:", mode_option)

    if mode == "Single":
        st.markdown("## ✏️ Single Input Demo")
        input_text = st.text_area(
            "Review Text Input",
            value="Saya suka menggunakan hotel ini. Kamar mandi wangi dan tidak jorok.",
            height=150,
            max_chars=425,
        )
        if st.button("Generate Triplet Opinion"):
            print_memory_usage()
            with st.spinner("Generating... (This may take some time)"):
                predictions = predict(input_text)
            st.markdown("### 🔎 Predictions")
            st.table(visualize_triplet_opinion(predictions))
            st.metric("Total Triplets", len(predictions), delta=None, delta_color="normal")


    else:
        # TODO:add uploader... # TODO: add download sample...
        # TODO:show table (exploded df...)
        # TODO:add downloader...
        pass 