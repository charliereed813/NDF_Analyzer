import numpy as np


def block_mean_resample(z, orig_valid_mask, scale):

    if scale <= 1:
        return z, orig_valid_mask

    rows, cols = z.shape
    new_rows = rows // scale
    new_cols = cols // scale

    z_new = np.full((new_rows, new_cols), np.nan)
    mask_new = np.zeros((new_rows, new_cols), dtype=bool)

    for i in range(new_rows):
        for j in range(new_cols):
            r1 = i * scale
            r2 = min(r1 + scale, rows)
            c1 = j * scale
            c2 = min(c1 + scale, cols)

            blockZ = z[r1:r2, c1:c2]
            blockM = orig_valid_mask[r1:r2, c1:c2]

            if np.any(blockM):
                z_new[i, j] = np.nanmean(blockZ)
                mask_new[i, j] = True
            else:
                z_new[i, j] = np.nan
                mask_new[i, j] = False

    return z_new, mask_new


def nan_aware_gradient(z, xy_sampling, mode="3-point"):

    dx = xy_sampling[0]
    dy = xy_sampling[1]

    rows, cols = z.shape
    dzdx = np.full_like(z, np.nan, dtype=float)
    dzdy = np.full_like(z, np.nan, dtype=float)

    def get(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return np.nan
        return z[r, c]

    for r in range(rows):
        for c in range(cols):
            center = get(r, c)
            if np.isnan(center):
                continue

            L1, R1 = get(r, c - 1), get(r, c + 1)
            L2, R2 = get(r, c - 2), get(r, c + 2)
            U1, D1 = get(r - 1, c), get(r + 1, c)
            U2, D2 = get(r - 2, c), get(r + 2, c)

            # X direction
            if mode == "5-point" and np.all(np.isfinite([L2, L1, R1, R2])):
                dzdx[r, c] = (-R2 + 8 * R1 - 8 * L1 + L2) / (12 * dx)
            elif np.isfinite(L1) and np.isfinite(R1):
                dzdx[r, c] = (R1 - L1) / (2 * dx)
            elif np.isfinite(R1):
                dzdx[r, c] = (R1 - center) / dx
            elif np.isfinite(L1):
                dzdx[r, c] = (center - L1) / dx

            # Y direction
            if mode == "5-point" and np.all(np.isfinite([U2, U1, D1, D2])):
                dzdy[r, c] = (-D2 + 8 * D1 - 8 * U1 + U2) / (12 * dy)
            elif np.isfinite(U1) and np.isfinite(D1):
                dzdy[r, c] = (D1 - U1) / (2 * dy)
            elif np.isfinite(D1):
                dzdy[r, c] = (D1 - center) / dy
            elif np.isfinite(U1):
                dzdy[r, c] = (center - U1) / dy

    return dzdx, dzdy


def compute_normals_and_angles(z, dzdx, dzdy, orig_valid_mask):

    nx = -dzdx.flatten()
    ny = -dzdy.flatten()
    nz = np.ones_like(nx)

    normals = np.vstack((nx, ny, nz)).T
    nrm = np.linalg.norm(normals, axis=1)
    bad = ~np.isfinite(nrm) | (nrm == 0)
    nrm[bad] = np.nan
    normals = normals / nrm[:, None]

    theta = np.degrees(np.arccos(normals[:, 2]))
    phi = np.degrees(np.arctan2(normals[:, 1], normals[:, 0]))
    phi[phi < 0] += 360.0

    theta_map = theta.reshape(z.shape)
    phi_map = phi.reshape(z.shape)

    theta_map[~orig_valid_mask] = np.nan
    phi_map[~orig_valid_mask] = np.nan

    theta_flat = theta_map.flatten()
    phi_flat = phi_map.flatten()
    valid = np.isfinite(theta_flat) & np.isfinite(phi_flat)

    return theta_map, phi_map, theta_flat[valid], phi_flat[valid], valid.sum()


def compute_ndf(theta_valid, phi_valid, bins=200):
    theta_edges = np.linspace(0, 90, bins + 1)
    phi_edges = np.linspace(0, 360, bins + 1)

    H, _, _ = np.histogram2d(theta_valid, phi_valid,
                             bins=[theta_edges, phi_edges])

    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    return H, theta_edges, phi_edges, theta_centers, phi_centers
