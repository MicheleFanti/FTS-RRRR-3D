from re import A
import sys
import cupy as np
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import LateralChains
_plot_counter = 0 
def initialize_alternative_rho(sequence, rho_class, rho0_per_class,
                               gridshape, spat_weights, ang_weights,
                               droplet_boost=1.0, droplet_sigma_frac=0.18,
                               n_noise_droplets=3, noise_boost_frac=0.2,
                               min_distance=0,
                               sc_frac=None): 

    Nx, Ny, Nz, Nang = gridshape
    X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    sigma = droplet_sigma_frac * min(Nx, Ny, Nz)
    center = (Nx // 2, Ny // 2, Nz//2)

    rho_sidechains = {}

    for res_key in rho_class:
        rho0 = rho0_per_class[res_key]
        if res_key not in ['pb', 'Nsc', 'Csc']:
            sc_val = LateralChains.SideChain(res_key).length
        else:
            sc_val = 0
        
        denom = 2.0 + sc_val
        base_backbone = rho0 * (2.0 / denom)
        base_sidechain = rho0 * (sc_val / denom)

        rho = np.ones((Nx, Ny, Nz, Nang), dtype=np.float64) * base_backbone
        rho_sc = np.zeros_like(rho)

        r2 = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2
        gaussian = np.exp(-r2 / (2 * sigma**2))
        gaussian /= np.sum(spat_weights*gaussian)
        rho += base_backbone * droplet_boost * gaussian[..., None]
        rho_sc += base_sidechain * gaussian[..., None]

        for _ in range(n_noise_droplets):
            while True:
                cx, cy, cz = np.random.randint(0, Nx), np.random.randint(0, Ny), np.random.randint(0, Nz)
                if np.sqrt((cx - center[0])**2 + (cy - center[1])**2 + (cz - center[2])**2) > min_distance * sigma:
                    break
            r2_noise = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
            gaussian_noise = np.exp(-r2_noise / (2 * (sigma / 2)**2))
            gaussian_noise /= np.sum(spat_weights*gaussian_noise)
            rho += base_backbone * droplet_boost * noise_boost_frac * gaussian_noise[..., None]
            rho_sc += base_sidechain * gaussian_noise[..., None]

        integral = np.sum(spat_weights[..., None] * ang_weights[None, None, None, :] * rho)
        if integral > 0:
            rho *= base_backbone * np.sum(spat_weights) * np.sum(ang_weights) / integral
        else:
            rho *= 0.0

        integral_sc = np.sum(spat_weights[..., None] * ang_weights[None, None, None, :] * rho_sc)
        if integral_sc > 0:
            rho_sc *= base_sidechain * np.sum(spat_weights) * np.sum(ang_weights) / integral_sc
        else:
            rho_sc *= 0.0

        rho_class[res_key] = rho
        rho_sidechains[res_key] = rho_sc

    return rho_class, rho_sidechains




from grid.angular import AngularGrid

def lebedev_grid(Nang):
    degree_to_N = {3:6,5:14,7:26,9:38,11:50,13:74,15:86,17:110,19:146,21:170,23:194,25:230,27:266,29:302,31:350}
    degrees = np.array(list(degree_to_N.keys()))
    Npoints = np.array(list(degree_to_N.values()))
    idx = np.searchsorted(Npoints, Nang)
    if idx >= len(degrees):
        raise ValueError(f"Nang={Nang} too large, max supported {int(Npoints[-1])}")
    degree = int(degrees[idx])
    grid = AngularGrid(degree=degree)
    u = np.asarray(grid.points)
    w = np.asarray(grid.weights)
    return u, w


def gaussian_realspace(grid, spat_weights, l, sigma):
    X, Y, Z = grid
    r = np.sqrt(X**2 + Y**2 + Z**2)
    if l != 0:
        f = np.exp(-((r - l)**2)/(2*sigma**2))
    else: 
        f = np.exp(-((r)**2)/(2*sigma**2))
        f[r < sigma] = 0
    return  f

def yukawa_realspace(grid, spat_weights, eps, decay, dx_min = 1e-1):
    X, Y, Z = grid
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r[r < dx_min] = dx_min  
    return eps * np.exp(-r/decay)/r

def make_3D_kernel_fft(realspace_func, grid, spat_weights, *params):
    V_real = realspace_func(grid, spat_weights, *params)
    V_k = np.fft.fftn(V_real)
    return V_k, V_real

def build_angular_kernel(N_ang, u_vectors, theta_0, ang_weights):
    sigma = 0.2
    kernel = np.zeros((N_ang, N_ang))
    for idxu, u in enumerate(u_vectors): 
        for idxv, v in enumerate(u_vectors): 
            kernel[idxu, idxv] = np.exp((np.tensordot(u, v, axes=([0], [0])) - np.cos(theta_0))**2/(2*sigma**2))
    kernel /= np.sum(np.sum(kernel*ang_weights, axis = -1)*ang_weights, axis = -1) # normalization
    return kernel

def build_exp_angular_kernel(N_ang, u_vectors, theta_0, ang_weights):
    sigma = 0.2
    kernel = np.zeros((N_ang, N_ang))
    for idxu, u in enumerate(u_vectors): 
        for idxv, v in enumerate(u_vectors): 
            kernel[idxu, idxv] = np.exp((np.tensordot(u, v, axes=([0], [0])) - np.cos(theta_0))**2/(2*sigma**2))
    kernel /= np.sum(np.sum(kernel*ang_weights, axis = -1)*ang_weights, axis = -1) # normalization
    return kernel

def build_identity_angular_kernel(N_ang, u_vectors, theta_0, ang_weights):
    kernel = np.eye(N_ang)  # identity
    kernel /= np.sum(np.sum(kernel*ang_weights, axis=-1)*ang_weights, axis=-1)  # normalization
    return kernel

def rescale_density(rho_to_scale, rho0_to_scale, spat_weights, ang_weights):
    if rho_to_scale.ndim == 4:
        mean = np.sum(np.sum(rho0_to_scale*ang_weights, axis = -1)*spat_weights)
    else:
        mean = np.sum(rho0_to_scale*spat_weights)
    if mean != 0:
        scaling_factor = rho0_to_scale/mean
    else:
        scaling_factor = 0
    return rho_to_scale*scaling_factor

def build_initials(sequence, l_chain, rhop0, gridshape, spat_weights, ang_weights):
    Nx, Ny, Nz, Nang = gridshape
    residue_counts = Counter(sequence)
    residue_classes = list(set(sequence)) 
    residue_classes_per_s = []

    for s in range(len(sequence)):
        for i in range(2):
            residue_classes_per_s.append(sequence[s])
        if s < len(sequence) - 1:
            residue_classes_per_s.append('pb')
    residue_classes.append('pb')
    
    rho0_per_class = {}
    for res in residue_classes:
        if res != 'pb':
            n_occ = residue_counts[res]
            rho0_per_class[res] = ((2/3)*0.38) / l_chain * rhop0 * n_occ


    pb_density = (1/3)*(0.38/l_chain)*rhop0  # density per pb bond
    rho0_per_class['pb'] = 0
    for i in range(len(sequence)-1):
        res_left = sequence[i]
        res_right = sequence[i+1]
        rho0_per_class[res_left] = rho0_per_class.get(res_left, 0) + 0.5 * pb_density
        rho0_per_class[res_right] = rho0_per_class.get(res_right, 0) + 0.5 * pb_density

    rho0_per_class['Nsc'] = rhop0*len(sequence)*0.15/l_chain
    rho0_per_class['Csc'] = rhop0*len(sequence)*0.15/l_chain

    for res in sequence:

        if LateralChains.SideChain(res).terminal == 'acceptor':
            rho0_per_class['Csc'] += rhop0*0.15/l_chain
        elif LateralChains.SideChain(res).terminal == 'donor':
            rho0_per_class['Nsc'] += rhop0*0.15/l_chain
        elif LateralChains.SideChain(res).terminal == 'both':
            rho0_per_class['Nsc'] += rhop0*0.15/l_chain 
            rho0_per_class['Csc'] += rhop0*0.15/l_chain 
        rho0_per_class[res] += rhop0*(LateralChains.SideChain(res).length*0.15)/l_chain

    rho_class = {res_key: np.zeros(gridshape) * rho0_per_class[res_key] for res_key in rho0_per_class}
    rho_class, rho_sc_rs = initialize_alternative_rho(sequence, rho_class, rho0_per_class, gridshape, spat_weights, ang_weights)
    print(f'\n\n---->> INITIAL DENSITIES: {rho0_per_class}')

    rho_backbone = {k: v for k, v in rho_class.items() if k not in ['Nsc', 'Csc']}
    rho_sidechains = {k: v for k, v in rho_class.items() if k in ['Nsc', 'Csc']}

    return residue_classes_per_s, residue_classes, rho0_per_class, rho_backbone, rho_sidechains, rho_sc_rs

def compute_constraint_violations(rho0_per_class, rho_class, spat_weights, ang_weights, V):
    total_current_rho = 0.0
    partial_diffs = []
    for c_key in rho0_per_class:
        mean_rho =  np.sum(np.sum(rho_class[c_key]*ang_weights, axis = -1) * spat_weights)
        total_current_rho += mean_rho
        partial_diffs.append(mean_rho-rho0_per_class[c_key])
    total_target_rho = sum(rho0_per_class.values())

    total_diff = abs(total_current_rho - total_target_rho)
    return total_diff, partial_diffs

def compute_has(sequence, charges, rho_class, residue_classes, ang_weights, gridshape):
    Nx, Ny, Nz = gridshape[:3]
    h_as = {}
    for a_key in charges: 
        h_a_val = np.zeros((Nx, Ny, Nz))
        for i, res_name in enumerate(sequence): 
            if res_name in rho_class:
                h_a_val += np.sum(rho_class[res_name]*ang_weights, axis = -1) * charges[a_key][i]
        h_as[a_key] = h_a_val   
    return h_as

def compute_c(sequence, charges, rho_class, rho_s, residue_classes, ang_weights, gridshape):
    c = np.zeros_like(rho_s['neutral'])
    for residue in rho_class:
        c += np.sum(rho_class[residue]*ang_weights, axis =-1) * charges[residue]
    for species in rho_s:
        c += rho_s[species] * charges[species]
    return c
    
def plot_densities(
    sequence, rho_class, rhoS, gridshape, it, vchi_pp, vchi_ps,
    gamma, relax_mu, rhop0, eps_yukawa, bjerrum_length, ang_weights, outdir,
    alpha=0.1
):
    import matplotlib.pyplot as plt
    import os
    import math
    import numpy as _np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    Nx, Ny, Nz = gridshape
    plot_classes = list(rho_class.keys())

    if outdir is None:
        outdir = "density_outputs_" + sequence
    density_plots_folder = os.path.join(outdir, "density_plots")
    os.makedirs(density_plots_folder, exist_ok=True)

    plotfile = os.path.join(
        density_plots_folder,
        f"eps_yukawa_{eps_yukawa}_bjlen_{bjerrum_length}_it:{it}_||||_vchi_ps_{vchi_ps}_rhop0_{rhop0}.png"
    )

    all_keys = plot_classes + ["neutral", "plus", "minus"]

    n_per_density = 4  # 1 scatter + 3 slices
    n_plots = (len(all_keys) + 1) * n_per_density
    ncols = 4
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 5 * nrows), subplot_kw={"projection": None}
    )
    axes = axes.flatten()

    total_density = None
    ax_idx = 0

    def process_density(name, rho):
        nonlocal total_density, ax_idx

        if total_density is None:
            total_density = _np.zeros_like(rho)
        total_density += rho

        mean_val = _np.mean(rho)
        coords = _np.indices(rho.shape).reshape(3, -1).T
        vals = rho.flatten()

        mask_above = vals >= mean_val
        mask_below = ~mask_above

        ax3d = fig.add_subplot(nrows, ncols, ax_idx + 1, projection="3d")
        if mask_above.any():
            ax3d.scatter(
                coords[mask_above, 0], coords[mask_above, 1], coords[mask_above, 2],
                c=vals[mask_above], cmap="viridis", alpha=alpha, s=3
            )
        if mask_below.any():
            ax3d.scatter(
                coords[mask_below, 0], coords[mask_below, 1], coords[mask_below, 2],
                c=vals[mask_below], cmap="plasma", alpha=alpha, s=3
            )
        ax3d.set_title(f"{name} scatter\nmean={mean_val:.4f}")
        ax_idx += 1

        midx, midy, midz = Nx // 2, Ny // 2, Nz // 2
        cuts = [
            (rho[:, :, midz].T, "XY mid"),
            (rho[:, midy, :].T, "XZ mid"),
            (rho[midx, :, :].T, "YZ mid"),
        ]
        for cut, label in cuts:
            ax2d = axes[ax_idx]
            im = ax2d.imshow(cut, origin="lower", cmap="viridis")
            ax2d.set_title(f"{name} {label}")
            fig.colorbar(im, ax=ax2d, shrink=0.7)
            ax_idx += 1

    for c_key in all_keys:
        if c_key in ["neutral", "plus", "minus"]:
            rho_plot = np.asnumpy(rhoS[c_key])
            if rho_plot.ndim == 4 and rho_plot.shape[-1] == 1:
                rho_plot = rho_plot[..., 0]
        else:
            rho_plot = np.asnumpy(
                _np.sum(rho_class[c_key] * ang_weights[None, None, :], axis=-1)
            )
        process_density(c_key, rho_plot)

    process_density("total", total_density)

    for j in range(ax_idx, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        f"Densities 3D scatter + slices (it={it}, vchi_pp={vchi_pp}, "
        f"vchi_ps={vchi_ps}, gamma={gamma}, mu={relax_mu}, rhop0={rhop0})",
        fontsize=16, y=1.02,
    )

    plt.tight_layout()
    plt.savefig(plotfile, bbox_inches="tight")
    print(f"Saved 3D density plots to {plotfile}")
    plt.close(fig)
