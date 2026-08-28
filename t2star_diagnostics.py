import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import nibabel as nib



def load_echo_niftis(nifti_paths, TEs):
    TEs = np.asarray(TEs, dtype=float)
    assert len(nifti_paths) == len(TEs), \
        "Number of nifti files must match number of TEs"

    vols = []
    for p in nifti_paths:
        img = nib.load(p)
        data = np.asarray(img.dataobj)  # (ny, nx, nslices)
        if data.ndim != 3:
            raise ValueError(f"{p}: expected 3D (ny, nx, nslices), got shape {data.shape}")
        data = np.transpose(data, (1, 0, 2))
        print(data.shape)
        vols.append(data)  

    nx, ny, nslices = vols[0].shape
    print(nslices)
    for v, p in zip(vols, nifti_paths):
        if v.shape != (nx, ny, nslices):
            raise ValueError(f"{p}: shape {v.shape} does not match first echo {(nx, ny, nslices)}")

    combined_mag = np.stack(vols, axis=-1)               
    combined_mag = combined_mag[:, :, :, np.newaxis, :]   

    return combined_mag, TEs


def load_cord_mask_nifti(mask_path):
    img = nib.load(mask_path)
    data = np.asarray(img.dataobj)
    data = np.transpose(data, (1, 0, 2)) 
    return data > 0.5


def mono_exp_with_noise(TE, S0, T2star, C):
    """Mono-exponential decay with noise floor offset: S(TE) = S0 * exp(-TE / T2*) + C"""
    return S0 * np.exp(-TE / T2star) + C

def fit_t2star(TEs, signal, p0_t2star=40.0):
    """Fit 3-parameter mono-exponential T2* decay to a signal-vs-echo-time curve.

    :param TEs: echo times, ms, shape (neco,)
    :param signal: mean ROI signal per echo, shape (neco,)
    :param p0_t2star: initial guess for T2* (ms) used to seed the fit

    :returns: (S0, T2star, C, success) -- success is False if the fit failed
              or produced a non-physical result
    """
    TEs = np.asarray(TEs, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if np.any(signal <= 0) or not np.all(np.isfinite(signal)):
        return np.nan, np.nan, np.nan, False

    C_init = max(0.0, float(signal[-1]))
    S0_init = max(1e-3, float(signal[0]) - C_init)

    try:
        popt, _ = curve_fit(
            mono_exp_with_noise, TEs, signal,
            p0=[S0_init, p0_t2star, C_init],
            bounds=([0, 0.1, 0], [np.inf, 5000, np.max(signal)]),
            maxfev=5000,
        )
        S0, T2star, C = popt
        if not np.isfinite(T2star) or T2star <= 0:
            return S0, T2star, C, False
        return S0, T2star, C, True
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan, False


def _roi_signal_per_echo(mag_slice_bin, mask):
    """Mean signal within mask for each echo."""
    neco = mag_slice_bin.shape[-1]
    return np.array([mag_slice_bin[:, :, e][mask].mean() for e in range(neco)])


def run_t2star_diagnostics(combined_mag, cord_mask, TEs, rank, out_dir,
                           expected_t2star=None, show=False):
    """Run the full T2*-based over-sharing diagnostic for one reconstruction."""
    os.makedirs(out_dir, exist_ok=True)

    combined_mag = np.asarray(combined_mag)
    TEs = np.asarray(TEs, dtype=float)
    nx, ny, nslices, nbins, neco = combined_mag.shape
    assert len(TEs) == neco, "TEs length must match neco"

    if cord_mask.ndim == 2:
        cord_mask = np.repeat(cord_mask[:, :, None], nslices, axis=2)
    assert cord_mask.shape == (nx, ny, nslices), \
        "cord_mask must be (nx, ny, nslices) or (nx, ny) to broadcast"

    signal_per_slice_bin = np.zeros((nslices, nbins, neco))
    t2star_per_slice_bin = np.full((nslices, nbins), np.nan)
    s0_per_slice_bin = np.full((nslices, nbins), np.nan)
    c_per_slice_bin = np.full((nslices, nbins), np.nan)  

    for s in range(nslices):
        mask = cord_mask[:, :, s]
        if mask.sum() == 0:
            continue
        for b in range(nbins):
            sig = _roi_signal_per_echo(combined_mag[:, :, s, b, :], mask)
            signal_per_slice_bin[s, b, :] = sig
            S0, T2star, C, ok = fit_t2star(TEs, sig) 
            s0_per_slice_bin[s, b] = S0
            c_per_slice_bin[s, b] = C
            if ok:
                t2star_per_slice_bin[s, b] = T2star

    t2star_mean_per_bin = np.nanmean(t2star_per_slice_bin, axis=0)     # (nbins,)
    t2star_mean_per_slice = np.nanmean(t2star_per_slice_bin, axis=1)   # (nslices,)

    fig, ax = plt.subplots(figsize=(6, 5))
    TE_fine = np.linspace(TEs.min(), TEs.max(), 200)
    for b in range(nbins):
        mean_signal = np.nanmean(signal_per_slice_bin[:, b, :], axis=0)
        ax.plot(TEs, mean_signal, 'o', label=f'bin {b} data')
        S0, T2star, C, ok = fit_t2star(TEs, mean_signal)  # MODIFIED: Unpack C
        if ok:
            ax.plot(TE_fine, mono_exp_with_noise(TE_fine, S0, T2star, C), '-',
                    label=f'fit (T2*={T2star:.1f}ms, C={C:.1f})')
    if expected_t2star is not None:
        ax.axhspan(0, 0, alpha=0)
    ax.set_xlabel('Echo time (ms)')
    ax.set_ylabel('Mean cord signal (a.u.)')
    ax.set_title(f'T2* decay by bin (rank={rank}, averaged over slices)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'decay_by_bin.png'), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for b in range(nbins):
        ax.plot(range(nslices), t2star_per_slice_bin[:, b], 'o-', label=f'bin {b}')
    if expected_t2star is not None:
        ax.axhspan(expected_t2star[0], expected_t2star[1], color='gray',
                   alpha=0.2, label='expected range')
    ax.set_xlabel('Slice index')
    ax.set_ylabel('Fitted T2* (ms)')
    ax.set_title(f'T2* vs slice, by bin (rank={rank})')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 't2star_vs_slice_by_bin.png'), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for s in range(nslices):
        ax.plot(range(nbins), t2star_per_slice_bin[s, :], '-', color='lightgray', linewidth=1)
    ax.plot(range(nbins), t2star_mean_per_bin, 'o-', color='C0', linewidth=2,
            label='mean across slices')
    if expected_t2star is not None:
        ax.axhspan(expected_t2star[0], expected_t2star[1], color='gray',
                   alpha=0.2, label='expected range')
    ax.set_xlabel('Bin index')
    ax.set_ylabel('Fitted T2* (ms)')
    ax.set_title(f'T2* vs bin, all slices + mean (rank={rank})')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 't2star_vs_bin.png'), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    results = dict(
        rank=rank,
        TEs=TEs,
        t2star_per_slice_bin=t2star_per_slice_bin,
        t2star_mean_per_bin=t2star_mean_per_bin,
        t2star_mean_per_slice=t2star_mean_per_slice,
        signal_per_slice_bin=signal_per_slice_bin,
        s0_per_slice_bin=s0_per_slice_bin,
        c_per_slice_bin=c_per_slice_bin,  
        out_dir=out_dir,
    )
    np.savez(os.path.join(out_dir, 'results.npz'),
             **{k: v for k, v in results.items() if k != 'out_dir'})

    return results


def compare_ranks(results_list, out_dir, expected_t2star=None, show=False):
    """Compare T2* estimates across multiple rank values (over-sharing check)."""
    os.makedirs(out_dir, exist_ok=True)
    ranks = [r['rank'] for r in results_list]

    overall_mean = [np.nanmean(r['t2star_per_slice_bin']) for r in results_list]
    overall_std = [np.nanstd(r['t2star_per_slice_bin']) for r in results_list]
    c_mean = [np.nanmean(r['c_per_slice_bin']) for r in results_list]  

    fig, ax1 = plt.subplots(figsize=(6, 5))

    color = 'tab:blue'
    ax1.errorbar(ranks, overall_mean, yerr=overall_std, fmt='o-', color=color, capsize=4,
                 label='mean T2* ± std')
    if expected_t2star is not None:
        ax1.axhspan(expected_t2star[0], expected_t2star[1], color='gray',
                    alpha=0.2, label='expected range')
    ax1.set_xlabel('SLR rank (r)')
    ax1.set_ylabel('Fitted T2* (ms)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(ranks, c_mean, 's--', color=color, label='mean Noise Offset C')
    ax2.set_ylabel('Noise Floor C (a.u.)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('T2* & Noise Floor C vs reconstruction rank')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 't2star_vs_rank.png'), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    print("Rank sensitivity summary:")
    for r, m, sd, c in zip(ranks, overall_mean, overall_std, c_mean):
        print(f"   rank={r}: T2*={m:.1f} ± {sd:.1f} ms | C={c:.2f}")

    return dict(ranks=ranks, overall_mean=overall_mean, overall_std=overall_std, c_mean=c_mean)