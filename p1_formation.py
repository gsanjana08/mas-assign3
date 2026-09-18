"""
Problem 1 - Formation control that spells "SANJANA".

Setup
-----
N = 20 single-integrator agents  p_i(t) in R^2, dot p_i = u_i.
Communication graph: Erdos-Renyi G(N, p_edge), redrawn until connected.

Controller (formation control, Algorithm 2 of MAS_Algorithms_Lecture):
    u_i = sum_{j in N_i} a_ij [ (p_j - p_i) - (r_j - r_i) ]
    ==> stacked form   u = -L (p - r)
so p_i - p_j -> r_i - r_j (the desired shape r, up to a common translation).

For each capital letter of "SANJANA" I precompute 20 target offsets
r^{letter} sampling the letter's strokes with counts proportional to
stroke length.  Then I run K discrete-time steps of
    p[k+1] = p[k] + dt * u[k].

Plain formation control keeps the centroid frozen (L1=0), so if the
initial centroid is off-screen the letters would drift.  To keep every
letter centred on the origin I use a one-agent "leader pin" (agent 0
sees the desired centroid 0) as in the Leader-Follower slide:
    u_0 += (0 + r_0 - p_0).

Outputs
-------
* p1_formation.mp4  -- animation of the whole sequence
* p1_snapshots.png  -- final formation for each letter
* p1_graph.png      -- the communication graph
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx


# ----------------------------- parameters -----------------------------
N          = 20            # number of agents
P_EDGE     = 0.28          # Erdos-Renyi edge probability
DT         = 0.05          # integration step
STEPS      = 350           # simulation steps per letter
SEED       = 11019
NAME       = "SANJANA"     # 7 letters
rng = np.random.default_rng(SEED)


# --------------------- connected Erdos-Renyi graph --------------------
def connected_er(n, p, rng):
    while True:
        G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(1_000_000_000)))
        if nx.is_connected(G):
            return G


G   = connected_er(N, P_EDGE, rng)
A   = nx.to_numpy_array(G)
L   = np.diag(A.sum(axis=1)) - A          # graph Laplacian


# --------------------------- letter templates -------------------------
# Each letter is a list of line segments (stroke endpoints) in a 2x3 box.
LETTERS = {
    "S": [((2, 3), (0, 3)), ((0, 3), (0, 1.5)),
          ((0, 1.5), (2, 1.5)), ((2, 1.5), (2, 0)),
          ((2, 0), (0, 0))],
    "A": [((0, 0), (1, 3)), ((1, 3), (2, 0)),
          ((0.5, 1.2), (1.5, 1.2))],
    "N": [((0, 0), (0, 3)), ((0, 3), (2, 0)),
          ((2, 0), (2, 3))],
    "J": [((0, 3), (2, 3)), ((1.4, 3), (1.4, 0.5)),
          ((1.4, 0.5), (0.8, 0)), ((0.8, 0), (0.2, 0.4))],
}


def sample_letter(ch, n):
    """Return n offsets in R^2 sampling the strokes of `ch`."""
    strokes = LETTERS[ch]
    lens = np.array([np.linalg.norm(np.subtract(b, a)) for a, b in strokes])
    counts = np.maximum(2, np.round(n * lens / lens.sum()).astype(int))
    # trim / pad so counts sum to exactly n
    while counts.sum() > n:
        counts[np.argmax(counts)] -= 1
    while counts.sum() < n:
        counts[np.argmin(counts)] += 1
    pts = []
    for (a, b), c in zip(strokes, counts):
        a, b = np.asarray(a, float), np.asarray(b, float)
        for k in range(c):
            t = k / max(1, c - 1)
            pts.append(a + t * (b - a))
    pts = np.asarray(pts[:n])
    pts -= pts.mean(axis=0)                    # centre the letter
    return pts


# ---------------------------- simulation ------------------------------
p = rng.uniform(-3.0, 3.0, size=(N, 2))        # random initial positions
traj = [p.copy()]

# leader-pinning weights: only agent 0 is pinned to the centre
pin = np.zeros(N)
pin[0] = 1.0

for ch in NAME:
    r = sample_letter(ch, N)
    for _ in range(STEPS):
        u = -L @ (p - r)                       # formation control
        u += pin[:, None] * (r - p)            # centroid anchor
        p = p + DT * u
        traj.append(p.copy())

traj = np.asarray(traj)
print(f"trajectory: frames={traj.shape[0]}  N={N}  edges={G.number_of_edges()}")


# --------------------------- static plots -----------------------------
fig, axs = plt.subplots(1, len(NAME), figsize=(2.2 * len(NAME), 2.6),
                        sharey=True)
for k, ch in enumerate(NAME):
    frame = (k + 1) * STEPS
    pts = traj[frame]
    axs[k].scatter(pts[:, 0], pts[:, 1], s=20, c="steelblue")
    axs[k].set_title(ch)
    axs[k].set_aspect("equal")
    axs[k].set_xlim(-1.6, 1.6)
    axs[k].set_ylim(-2.0, 2.0)
    axs[k].grid(alpha=0.3)
fig.suptitle("SANJANA -- converged formation for each letter")
fig.tight_layout()
fig.savefig("p1_snapshots.png", dpi=160, bbox_inches="tight")
plt.close(fig)

figG, axG = plt.subplots(figsize=(4.2, 4.2))
nx.draw_spring(G, ax=axG, node_color="#7db1d1", edge_color="#9aa0a6",
               node_size=380, font_size=9, with_labels=True, seed=SEED)
axG.set_title(f"Erd\u0151s\u2013R\u00e9nyi graph  ($N={N}$, $p={P_EDGE}$)")
figG.tight_layout()
figG.savefig("p1_graph.png", dpi=160, bbox_inches="tight")
plt.close(figG)


# ------------------------------ animation -----------------------------
fig2, ax2 = plt.subplots(figsize=(5, 5))
sc = ax2.scatter(traj[0, :, 0], traj[0, :, 1], s=32, c="crimson")
edges = []
for i, j in G.edges():
    ln, = ax2.plot([traj[0, i, 0], traj[0, j, 0]],
                   [traj[0, i, 1], traj[0, j, 1]],
                   color="lightgray", lw=0.6, zorder=1)
    edges.append((i, j, ln))
ax2.set_xlim(-3.2, 3.2)
ax2.set_ylim(-3.2, 3.2)
ax2.set_aspect("equal")
ax2.grid(alpha=0.3)
title = ax2.set_title("")


def _update(k):
    pts = traj[k]
    sc.set_offsets(pts)
    for i, j, ln in edges:
        ln.set_data([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]])
    letter_idx = min(k // STEPS, len(NAME) - 1)
    title.set_text(f"forming '{NAME[letter_idx]}'   (frame {k})")
    return sc,


ani = animation.FuncAnimation(fig2, _update,
                              frames=range(0, traj.shape[0], 3),
                              interval=30, blit=False)
try:
    ani.save("p1_formation.mp4", fps=30, dpi=140)
    print("saved p1_formation.mp4")
except Exception:
    ani.save("p1_formation.gif", fps=30, dpi=100, writer="pillow")
    print("saved p1_formation.gif")
