import numpy as np
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
import matplotlib.colors as colors
from matplotlib.collections import LineCollection


##############################################
##### SAVE AND LOAD ARRAYS AS COMPLEX32 ######
##############################################


def save_complex32(filename, arr):
    """Save a complex numpy array in a compressed format with float16 precision"""
    real_part = arr.real.astype(np.float16)
    imag_part = arr.imag.astype(np.float16)
    np.savez_compressed(filename, real=real_part, imag=imag_part)


def load_complex32(filename):
    """Load a complex numpy array from a compressed format with float16 precision"""
    data = np.load(filename)
    arr = data["real"].astype(np.float32) + 1j * data["imag"].astype(np.float32)
    return arr


def colorize(
    z,
    theme="dark",
    saturation=1.0,
    beta=1.4,
    transparent=False,
    alpha=1.0,
    max_threshold=1,
):
    r = np.abs(z)
    r /= max_threshold * np.max(np.abs(r))
    arg = np.angle(z)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1.0 / (1.0 + r**beta) if theme == "white" else 1.0 - 1.0 / (1.0 + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    if transparent:
        a = 1.0 - np.sum(c**2, axis=-1) / 3
        alpha_channel = a[..., None] ** alpha
        return np.concatenate([c, alpha_channel], axis=-1)
    else:
        return c


def get_disk_mask(shape, radius, center=None):
    """
    Generate a binary mask with value 1 inside a disk, 0 elsewhere
    :param shape: list of integer, shape of the returned array
    :radius: integer, radius of the disk
    :center: list of integers, position of the center
    :return: numpy array, the resulting binary mask
    """
    if not center:
        center = (shape[0] // 2, shape[1] // 2)
    X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    mask = (Y - center[0]) ** 2 + (X - center[1]) ** 2 < radius**2
    return mask.astype(int)


##############################################
########## CORRELATION COMPUTATION ###########
##############################################


def stack_correlation(A, B):
    """compute the correlation between two stacks of images
    A and B over the last dimension
    return the correlation between the two stacks
    """
    C = np.einsum("...jk,...k->...j", A, B.conjugate())
    Norm1 = np.einsum("...k,...k->...", A, A.conjugate())
    Norm2 = np.einsum("...k,...k->...", B, B.conjugate())
    Norm = np.sqrt(np.einsum("...k,...->...k", Norm1, Norm2.conjugate()))
    return np.abs(C / Norm)


def cpx_corr(Y1, Y2):
    corr = np.sum((Y1) * (Y2).conj())
    norm_fac = np.sqrt(np.sum(np.abs(Y1) ** 2) * np.sum(np.abs(Y2) ** 2))
    return corr / norm_fac


def seq_cpx_corr(Yseq, Yref=None, mask=None, remove_mean=False):
    if Yref is None:
        Yref = Yseq[0]
    if mask is not None:
        Yref *= mask.flatten()
        Yseq *= mask.flatten()
    if remove_mean:
        Yseq = Yseq - np.mean(Yseq, axis=-1, keepdims=True)
        Yref = Yref - np.mean(Yref)
    corr = np.sum(Yseq * Yref.conj(), axis=-1)
    norm_ref = np.sum(np.abs(Yref) ** 2)
    norm_seq = np.sum(np.abs(Yseq) ** 2, axis=-1)
    return corr / np.sqrt(norm_ref * norm_seq)


def tr_prod(A, B):
    prod = np.abs(A @ np.swapaxes(B, axis1=-2, axis2=-1).conjugate()) ** 2
    return np.trace(prod, axis1=-2, axis2=-1)


def fidelity(A, B):
    return tr_prod(A, B) / (np.sqrt(tr_prod(A, A) * tr_prod(B, B)))


# Plotting functions


def get_color_map(n):
    cdict = {
        "red": ((0.0, 1.0, 1.0), (1.0, 0.3, 0.3)),
        "green": ((0.0, 0.0, 0.0), (1.0, 0.1, 0.1)),
        "blue": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    }
    cmap = colors.LinearSegmentedColormap("custom", cdict, N=n)
    return cmap


def generate_color_shades(color, n):
    # Generate a list of `n` different shades of the given color
    cmap = plt.get_cmap(color)
    dc = 100
    shades = [cmap(int(dc + i * (255 - dc) / (n - 1))) for i in range(n)]
    return shades


def logplotTM(
    array,
    fig,
    ax,
    degenerate_mask=None,
    min_val=1e-2,
    pola_quadrant=True,
    lw=1,
    c="r",
    cmap="inferno",
    shrink_cb=1,
):
    array /= np.max(array)
    pcm = ax.matshow(array, norm=colors.LogNorm(vmin=min_val, vmax=1), cmap=cmap)
    ax.axis("off")
    fig.colorbar(pcm, ax=ax, shrink=shrink_cb)  # , extend='max')
    if pola_quadrant:
        ax.axvline(array.shape[1] // 2 - 0.5, c=c, lw=lw)
        ax.axhline(array.shape[0] // 2 - 0.5, c=c, lw=lw)
    if degenerate_mask is not None:
        plot_outlines(np.tile(degenerate_mask.T, (2, 2)), ax=ax, lw=lw, color=c)


def get_all_edges(bool_img):
    """
    Get a list of all edges (where the value changes from True to False) in the 2D boolean image.
    The returned array edges has he dimension (n, 2, 2).
    Edge i connects the pixels edges[i, 0, :] and edges[i, 1, :].
    Note that the indices of a pixel also denote the coordinates of its lower left corner.
    """
    edges = []
    ii, jj = np.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == bool_img.shape[1] - 1 or not bool_img[i, j + 1]:
            edges.append(np.array([[i, j + 1], [i + 1, j + 1]]))
        # East
        if i == bool_img.shape[0] - 1 or not bool_img[i + 1, j]:
            edges.append(np.array([[i + 1, j], [i + 1, j + 1]]))
        # South
        if j == 0 or not bool_img[i, j - 1]:
            edges.append(np.array([[i, j], [i + 1, j]]))
        # West
        if i == 0 or not bool_img[i - 1, j]:
            edges.append(np.array([[i, j], [i, j + 1]]))

    if not edges:
        return np.zeros((0, 2, 2))
    else:
        return np.array(edges)


def close_loop_edges(edges):
    """
    Combine thee edges defined by 'get_all_edges' to closed loops around objects.
    If there are multiple disconnected objects a list of closed loops is returned.
    Note that it's expected that all the edges are part of exactly one loop (but not necessarily the same one).
    """

    loop_list = []
    while edges.size != 0:
        loop = [edges[0, 0], edges[0, 1]]  # Start with first edge
        edges = np.delete(edges, 0, axis=0)

        while edges.size != 0:
            # Get next edge (=edge with common node)
            ij = np.nonzero((edges == loop[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                loop.append(loop[0])
                # Uncomment to to make the start of the loop invisible when plotting
                # loop.append(loop[1])
                break

            loop.append(edges[i, (j + 1) % 2, :])
            edges = np.delete(edges, i, axis=0)

        loop_list.append(np.array(loop))

    return loop_list


def plot_outlines(bool_img, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    edges = get_all_edges(bool_img=bool_img)
    edges = (
        edges - 0.5
    )  # convert indices to coordinates; TODO adjust according to image extent
    outlines = close_loop_edges(edges=edges)
    cl = LineCollection(outlines, **kwargs)
    ax.add_collection(cl)
