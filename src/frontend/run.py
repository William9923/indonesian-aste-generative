import logging
import os
from pathlib import Path as P

import pandas as pd
import streamlit as st

from src.frontend.download import download
from src.frontend.utils import predict, visualize_triplet_opinion, print_memory_usage, loader

from src.loader import parse
from src.constant import Path

PRED_BATCH_SIZE = 4

configs_map = {
    "T5-Filter"     : os.path.join(Path.RESOURCES, "t5-config-filter-eval.yaml"),
    "T5-Unfilter"   : os.path.join(Path.RESOURCES, "t5-config-eval.yaml")
}

configuration = {
    "T5-Filter"     : "T5 based fine tuned model on filtered dataset",
    "T5-Unfilter"   : "T5 based fine tuned model on unfiltered dataset",
}
mode_option = ["Single", "Multi"]

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

def main():
    config_name = st.sidebar.selectbox("Choose a configuration:", list(configuration.keys()))
    prefix = st.sidebar.text_input("Prefix:", value="t5-model")
    config_path = configs_map.get(config_name)
    tokenizer, model = loader(prefix=prefix, config_path=config_path)

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
                predictions = predict(input_text, tokenizer, model, config_path)
            st.markdown("### 🔎 Predictions")
            st.table(visualize_triplet_opinion(predictions))
            st.metric("Total Triplets", len(predictions), delta=None, delta_color="normal")

    else:
        # TODO:add uploader... # TODO: add download sample...
        # TODO:show table (exploded df...)
        # TODO:add downloader...
        pass 