import h5py
import numpy as np

def load_datx(path):

    f = h5py.File(path, "r")


    meta = f["MetaData"][()]
    sources = np.array([row["Source"].decode() for row in meta])
    links   = np.array([row["Link"].decode()   for row in meta])
    dests   = np.array([row["Destination"].decode() for row in meta])

    meas_node = dests[(sources == "Root") & (links == "Measurement")][0]
    surf_guid = dests[(sources == meas_node) & (links == "Surface")][0]

    # full path to surface map
    surf_path = f"/Data/Surface/{surf_guid}"
    ds = f[surf_path]

    # load raw height
    Z = ds[...]                      # correct orientation
    Z = Z.astype(float)

    # remove No Data values
    if "No Data" in ds.attrs:
        nod = float(np.array(ds.attrs["No Data"]).reshape(-1)[0])
        Z[Z >= nod] = np.nan

    # Z conversion
    zc = ds.attrs["Z Converter"][0]
    base = zc["BaseUnit"].decode()
    params = np.array(zc["Parameters"], float)

    if base == "NanoMeters":
        scale = 1e-3
    elif base == "MicroMeters":
        scale = 1.0
    elif base == "MilliMeters":
        scale = 1e3
    elif base == "Meters":
        scale = 1e6
    elif base == "Fringes":
        S = params[2]
        O = params[3]
        W = params[1]
        scale = 1e6 * S * O * W
    else:
        scale = 1

    height_um = Z * scale

    # lateral sampling
    xc = ds.attrs["X Converter"][0]
    yc = ds.attrs["Y Converter"][0]

    xRes_um = 1e6 * float(xc["Parameters"][1])
    yRes_um = 1e6 * float(yc["Parameters"][1])

    xy_sampling = np.array([yRes_um, xRes_um])  # row spacing, col spacing

    valid_mask = np.isfinite(height_um)

    return height_um, xy_sampling, valid_mask
