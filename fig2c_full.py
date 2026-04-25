"""Simulate, save, and plot Fig. 2(c)-style dynamics.

Defaults are the paper-scale dynamics parameters:
    16 transmon eigenstates, 300 resonator Fock states, 200 trajectories,
    and evolution to kappa * t = 2.

This is intentionally standalone so the expensive trajectory run can happen
outside the notebook and the saved data can be replotted later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable


np = None
qt = None
plt = None


def require_runtime_dependencies():
    """Import heavy runtime dependencies only after argparse handles --help."""
    global np, qt, plt
    if np is None:
        import numpy as _np

        np = _np
    if qt is None:
        import qutip as _qt

        qt = _qt
    if plt is None:
        import matplotlib.pyplot as _plt

        plt = _plt


def transmon_hamiltonian(EJ: float, EC: float, ng: float = 0.0, charge_trunc: int = 15):
    """Charge-basis transmon Hamiltonian, in GHz."""
    dim = 2 * charge_trunc + 1
    ns = np.arange(-charge_trunc, charge_trunc + 1, dtype=float)
    H_C = qt.qdiags(4 * EC * (ns - ng) ** 2, 0)
    H_J = qt.qdiags(np.ones(dim - 1), 1) + qt.qdiags(np.ones(dim - 1), -1)
    return H_C - 0.5 * EJ * H_J, ns


def build_dynamics_hamiltonian(args: argparse.Namespace):
    """Build the driven, dissipative Hamiltonian used for Fig. 2(c)."""
    H_tmon_charge, ns = transmon_hamiltonian(
        args.EJ, args.EC, ng=args.ng, charge_trunc=args.charge_trunc
    )
    tmon_evals_full, tmon_evecs_full = H_tmon_charge.eigenstates()
    tmon_evals_full = np.array(tmon_evals_full)
    tmon_evals_full -= tmon_evals_full[0]

    n_charge = qt.qdiags(ns - args.ng, 0)
    U = qt.Qobj(
        np.column_stack(
            [tmon_evecs_full[i].full()[:, 0] for i in range(args.tmon_levels)]
        ),
        dims=[[2 * args.charge_trunc + 1], [args.tmon_levels]],
    )

    n_t = U.dag() * n_charge * U
    H_t = qt.qdiags(tmon_evals_full[: args.tmon_levels], 0)

    a = qt.destroy(args.cav_trunc)
    I_t = qt.qeye(args.tmon_levels)
    I_c = qt.qeye(args.cav_trunc)
    a_full = qt.tensor(I_t, a)
    num_full = a_full.dag() * a_full

    H0_cycles = (
        qt.tensor(H_t, I_c)
        + args.wr * qt.tensor(I_t, a.dag() * a)
        - 1j * args.g * qt.tensor(n_t, a - a.dag())
    )

    # QuTiP time evolution uses exp(-i H t). With t in ns and static
    # parameters in GHz = cycles/ns, convert Hamiltonian terms to rad/ns.
    H0 = 2 * np.pi * H0_cycles

    eps_d = 2 * np.pi * args.eps_d_over_2pi
    wd = 2 * np.pi * args.wd_over_2pi
    kappa = 2 * np.pi * args.kappa_over_2pi

    H_drive = -1j * eps_d * (a_full - a_full.dag())

    def drive_coeff(t, args=None):
        return np.sin(wd * t)

    H = [H0, [H_drive, drive_coeff]]
    c_ops = [np.sqrt(kappa) * a_full]

    return {
        "H": H,
        "H0_cycles": H0_cycles,
        "c_ops": c_ops,
        "num_full": num_full,
        "I_t": I_t,
        "kappa": kappa,
    }


def relabel_dressed_branches(H0_cycles: qt.Qobj, args: argparse.Namespace):
    """Relabel dressed eigenstates into transmon/resonator branches."""
    dim = args.tmon_levels * args.cav_trunc
    print(f"Diagonalizing dynamics Hamiltonian with dimension {dim}...")
    dressed_evals, dressed_evecs = H0_cycles.eigenstates()

    bare_eigenstates = []
    for i in range(args.tmon_levels):
        for n in range(args.cav_trunc):
            bare_eigenstates.append(
                qt.tensor(qt.basis(args.tmon_levels, i), qt.basis(args.cav_trunc, n))
            )

    create_op = qt.tensor(qt.qeye(args.tmon_levels), qt.create(args.cav_trunc))
    relabeled_states = []
    relabeled_indices = []
    assigned = np.zeros(dim)

    print("Relabeling dressed states into branches...")
    for i in range(args.tmon_levels):
        for n in range(args.cav_trunc):
            if n == 0:
                compare_state = bare_eigenstates[i * args.cav_trunc]
            else:
                compare_state = create_op * relabeled_states[i * args.cav_trunc + (n - 1)]
                compare_state = compare_state.unit()

            cur_overlap = -1.0
            cur_state = -1
            for k in range(dim):
                if assigned[k] == 0:
                    overlap = abs(dressed_evecs[k].overlap(compare_state)) ** 2
                    if overlap > cur_overlap:
                        cur_overlap = overlap
                        cur_state = k

            assigned[cur_state] = 1
            relabeled_indices.append(cur_state)
            relabeled_states.append(dressed_evecs[cur_state])

    return np.array(dressed_evals), relabeled_states, np.array(relabeled_indices, dtype=int)


def branch_population_callback(branch_states: list[qt.Qobj]) -> Callable:
    """Return an e_ops callback that computes population in one dressed branch."""

    def population(_, state):
        if state.isket:
            return float(sum(abs(psi.overlap(state)) ** 2 for psi in branch_states))

        # This fallback is slower and should rarely be used for mcsolve, but it
        # keeps the script correct if a solver passes a density matrix.
        return float(sum(np.real(qt.expect(psi * psi.dag(), state)) for psi in branch_states))

    return population


def run_simulation(args: argparse.Namespace):
    require_runtime_dependencies()
    built = build_dynamics_hamiltonian(args)
    dressed_evals, relabeled_states, relabeled_indices = relabel_dressed_branches(
        built["H0_cycles"], args
    )

    if max(args.highlight_branches) >= args.tmon_levels:
        raise ValueError(
            "All highlighted branches must be below --tmon-levels. "
            f"Got highlights {args.highlight_branches} with {args.tmon_levels} levels."
        )

    t_final = args.kappa_t_final / built["kappa"]
    tlist = np.linspace(0, t_final, args.num_times)

    psi0 = relabeled_states[args.initial_branch * args.cav_trunc + args.initial_fock]

    e_ops = [built["num_full"]]
    for branch in args.highlight_branches:
        start = branch * args.cav_trunc
        end = (branch + 1) * args.cav_trunc
        e_ops.append(branch_population_callback(relabeled_states[start:end]))

    fock_projectors = [
        qt.tensor(
            built["I_t"],
            qt.basis(args.cav_trunc, n) * qt.basis(args.cav_trunc, n).dag(),
        )
        for n in range(args.cav_trunc)
    ]
    e_ops.extend(fock_projectors)

    print(
        "Running mcsolve with "
        f"{args.tmon_levels} transmon levels, {args.cav_trunc} resonator states, "
        f"{args.ntraj} trajectories, {args.num_times} time points..."
    )
    result = qt.mcsolve(
        built["H"],
        psi0,
        tlist,
        built["c_ops"],
        e_ops=e_ops,
        ntraj=args.ntraj,
        options={"store_states": False, "progress_bar": args.progress_bar},
    )

    n_branches = len(args.highlight_branches)
    kappa_t = built["kappa"] * tlist
    Nr_t = np.real(result.expect[0])
    branch_population_array = np.array(
        [np.real(result.expect[1 + idx]) for idx in range(n_branches)]
    )
    final_fock_distribution = np.array(
        [
            np.real(result.expect[1 + n_branches + n][-1])
            for n in range(args.cav_trunc)
        ]
    )

    data = {
        "kappa_t": kappa_t,
        "Nr_t": Nr_t,
        "highlight_branches": np.array(args.highlight_branches, dtype=int),
        "branch_population_array": branch_population_array,
        "fock_numbers": np.arange(args.cav_trunc),
        "final_fock_distribution": final_fock_distribution,
        "relabeled_dressed_indices": relabeled_indices,
        "dressed_evals": dressed_evals,
        "EC": args.EC,
        "EJ": args.EJ,
        "ng": args.ng,
        "charge_trunc": args.charge_trunc,
        "wr": args.wr,
        "g": args.g,
        "tmon_levels": args.tmon_levels,
        "cav_trunc": args.cav_trunc,
        "ntraj": args.ntraj,
        "num_times": args.num_times,
        "eps_d_over_2pi": args.eps_d_over_2pi,
        "wd_over_2pi": args.wd_over_2pi,
        "kappa_over_2pi": args.kappa_over_2pi,
        "kappa_t_final": args.kappa_t_final,
        "initial_branch": args.initial_branch,
        "initial_fock": args.initial_fock,
    }

    np.savez(args.out, **data)
    print(f"Saved data to {args.out.resolve()}")
    return data


def load_data(path: Path):
    require_runtime_dependencies()
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def plot_data(data: dict, output: Path | None = None):
    require_runtime_dependencies()
    highlight_colors = {
        0: "blue",
        1: "red",
        7: "green",
        10: "orange",
        11: "pink",
    }

    kappa_t = data["kappa_t"]
    Nr_t = data["Nr_t"]
    branches = data["highlight_branches"].astype(int)
    branch_populations = data["branch_population_array"]
    fock_numbers = data["fock_numbers"]
    final_fock_distribution = data["final_fock_distribution"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for idx, branch in enumerate(branches):
        ax.plot(
            kappa_t,
            np.maximum(branch_populations[idx], 1e-15),
            color=highlight_colors.get(int(branch), "gray"),
            label=rf"${int(branch)}_t$",
        )

    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.1)
    ax.set_xlim(0, kappa_t[-1])
    ax.set_xlabel(r"$\kappa t$")
    ax.set_ylabel(r"$P(B_{i_t})$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    tick_kappa_t = np.linspace(0, kappa_t[-1], 5)
    tick_Nr = np.interp(tick_kappa_t, kappa_t, Nr_t)
    ax_top.set_xticks(tick_kappa_t)
    ax_top.set_xticklabels([f"{x:.0f}" for x in tick_Nr])
    ax_top.set_xlabel(r"$\langle N_r\rangle(t)$")

    inset = ax.inset_axes([0.17, 0.48, 0.34, 0.34])
    inset.plot(fock_numbers, np.maximum(final_fock_distribution, 1e-15), color="black")
    inset.set_yscale("log")
    inset.set_xlim(0, max(fock_numbers))
    inset.set_ylim(1e-5, 1)
    inset.set_xlabel(r"$n_r$", fontsize=9)
    inset.set_ylabel(r"$P(n_r)$", fontsize=9)
    inset.tick_params(labelsize=8)

    fig.tight_layout()

    if output is not None:
        fig.savefig(output, dpi=200)
        print(f"Saved plot to {output.resolve()}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="Load saved data and plot only.")
    parser.add_argument("--no-plot", action="store_true", help="Run/save without displaying a plot.")
    parser.add_argument("--out", type=Path, default=Path("fig2c_full_data.npz"))
    parser.add_argument("--plot-out", type=Path, default=None)

    parser.add_argument(
        "--preset",
        choices=["paper", "smoke"],
        default="paper",
        help="paper uses the full defaults; smoke is a quick low-resolution test.",
    )

    parser.add_argument("--EC", type=float, default=0.22)
    parser.add_argument("--EJ", type=float, default=None)
    parser.add_argument("--ng", type=float, default=0.0)
    parser.add_argument("--charge-trunc", type=int, default=15)
    parser.add_argument("--wr", type=float, default=7.5)
    parser.add_argument("--g", type=float, default=0.120)

    parser.add_argument("--tmon-levels", type=int, default=None)
    parser.add_argument("--cav-trunc", type=int, default=None)
    parser.add_argument("--ntraj", type=int, default=None)
    parser.add_argument("--num-times", type=int, default=None)
    parser.add_argument("--kappa-t-final", type=float, default=2.0)

    parser.add_argument("--eps-d-over-2pi", type=float, default=0.180)
    parser.add_argument("--wd-over-2pi", type=float, default=7.515)
    parser.add_argument("--kappa-over-2pi", type=float, default=0.00795)

    parser.add_argument("--initial-branch", type=int, default=1)
    parser.add_argument("--initial-fock", type=int, default=0)
    parser.add_argument("--highlight-branches", type=int, nargs="+", default=[0, 1, 7, 10, 11])
    parser.add_argument("--progress-bar", default="text")

    args = parser.parse_args()

    if args.EJ is None:
        args.EJ = 110 * args.EC

    if args.preset == "paper":
        args.tmon_levels = 16 if args.tmon_levels is None else args.tmon_levels
        args.cav_trunc = 300 if args.cav_trunc is None else args.cav_trunc
        args.ntraj = 200 if args.ntraj is None else args.ntraj
        args.num_times = 301 if args.num_times is None else args.num_times
    else:
        args.tmon_levels = 12 if args.tmon_levels is None else args.tmon_levels
        args.cav_trunc = 60 if args.cav_trunc is None else args.cav_trunc
        args.ntraj = 5 if args.ntraj is None else args.ntraj
        args.num_times = 101 if args.num_times is None else args.num_times

    return args


def main():
    args = parse_args()
    args.out = args.out.resolve()

    if args.plot_only:
        data = load_data(args.out)
    else:
        data = run_simulation(args)

    if not args.no_plot:
        plot_data(data, output=args.plot_out)


if __name__ == "__main__":
    main()
