import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from datx_reader import load_datx
from NDF_core import (
    block_mean_resample,
    nan_aware_gradient,
    compute_normals_and_angles,
    compute_ndf,
)

st.set_page_config(layout="wide")

st.title("Surface Normal (NDF) Analyzer")
st.write("Upload a Zygo .datx file to compute the NDF from surface normals.")


# Sidebar
st.sidebar.header("Settings")

scaling_factor = st.sidebar.number_input(
    "Sampling Scaling Factor (>=1)",
    min_value=1, max_value=50, value=1, step=1
)

diff_type = st.sidebar.radio(
    "Differencing Type",
    ["3-point", "5-point"],
    index=0
)

bins = st.sidebar.number_input(
    "NDF bins",
    min_value=20, max_value=400,
    value=100, step=20
)

uploaded_file = st.file_uploader("Upload Zygo .datx file", type=["datx"])
run_button = st.button("Run NDF Analysis")


# Main processing
if run_button:
    if uploaded_file is None:
        st.error("Please upload a .datx file first.")
        st.stop()

    with open("temp_input.datx", "wb") as f:
        f.write(uploaded_file.read())

    filepath = "temp_input.datx"

    st.subheader("Loading file...")

    try:
        height_data, xy_sampling, valid_mask_original = load_datx(filepath)
    except Exception as e:
        st.error(f"Error reading .datx: {e}")
        st.stop()

    # Basic counts
    N_total = height_data.size
    N_valid = np.count_nonzero(valid_mask_original)
    pct_valid = 100 * (N_valid / N_total)

    # Effective sampling after resampling
    xy_eff = xy_sampling * scaling_factor

    st.write(f"**Valid points:** {N_valid:,}/{N_total:,} ({pct_valid:.2f}%)")

    st.write(f"**Effective lateral sampling (µm):** {xy_eff[0]:.3f} × {xy_eff[1]:.3f}")

    # Resampling

    z_res, mask_res = block_mean_resample(height_data, valid_mask_original, scaling_factor)
    xy_res = xy_eff


    # Gradients

    dzdx, dzdy = nan_aware_gradient(z_res, xy_res, mode=diff_type)


    # Normals + angles
    st.subheader("Surface Normals")
    theta_map, phi_map, theta_valid, phi_valid, n_valid = compute_normals_and_angles(
        z_res, dzdx, dzdy, mask_res
    )

    st.write(f"Valid normals used for NDF: **{n_valid:,}**")


    # Spatial angle maps
    st.subheader("Spatial Angle Maps")
    c1, c2 = st.columns(2)

    # Polar map
    with c1:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(theta_map, cmap="turbo", origin="lower")
        ax.set_title("Polar Angle Map (θ, deg)")
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        plt.colorbar(im, ax=ax, label="θ (degrees)")
        st.pyplot(fig)

    # Azimuth map
    with c2:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(phi_map, cmap="hsv", origin="lower")
        ax.set_title("Azimuth Angle Map (φ, deg)")
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        plt.colorbar(im, ax=ax, label="φ (degrees)")
        st.pyplot(fig)


    # Angle histograms
    st.subheader("Angle Distributions")
    c3, c4 = st.columns(2)

    polar_edges = np.linspace(0, 90, bins + 1)
    az_edges = np.linspace(0, 360, bins + 1)

    # Polar histogram
    with c3:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hist(theta_valid, bins=polar_edges, color="coral")
        ax.set_xlim(0, 90)
        ax.set_xlabel("θ (degrees)")
        ax.set_ylabel("Count")
        ax.set_title("Polar Angle Histogram")
        st.pyplot(fig)

    # Azimuth histogram
    with c4:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hist(phi_valid, bins=az_edges, color="skyblue")
        ax.set_xlim(0, 360)
        ax.set_xlabel("φ (degrees)")
        ax.set_ylabel("Count")
        ax.set_title("Azimuth Angle Histogram")
        st.pyplot(fig)


    # Polar Stats
    st.subheader("Polar Angle Stats")

    mean_t = float(np.nanmean(theta_valid))
    med_t = float(np.nanmedian(theta_valid))

    hist_vals, _ = np.histogram(theta_valid, bins=polar_edges)
    mode_center = (polar_edges[np.argmax(hist_vals)] +
                   polar_edges[np.argmax(hist_vals) + 1]) / 2
    mode_t = float(mode_center)

    std_t = float(np.nanstd(theta_valid))

    st.write(f"**Mean θ:** {mean_t:.3f}°")
    st.write(f"**Median θ:** {med_t:.3f}°")
    st.write(f"**Mode θ:** {mode_t:.3f}°")
    st.write(f"**Std Dev θ:** {std_t:.3f}°")

    st.success("Analysis complete.")
