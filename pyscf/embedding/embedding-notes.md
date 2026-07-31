# `pyscf/embedding` — residual notes

**Branch:** `pyframe_interface` · **Last updated:** 2026-09-10
*(This is the trimmed remnant of a full code review of the PyFraME interface.
Every reviewable finding is closed; what follows is only what is still open,
plus the rationales worth not rediscovering. Note this file is gitignored —
deleting it loses it for good.)*

## Development environment

Work in the `kosmos` conda env, which has PyFraME `0.5a0` installed editable
from `~/PycharmProjects/pyframe` (`git@gitlab.com:pyframe-project/pyframe.git`).
No MPI setup is needed — as of PyFraME `071f551` the tests run in a plain shell.

```
~/miniconda3/envs/kosmos/bin/python -m pytest pyscf/embedding/test/test_embedding.py -q
```

**Last verified 2026-09-08:** 30 passed, against pyframe `309e04c`, pyscf 2.14.0.

## [ ] Open: citation placeholders

`pyscf/embedding/embedding.py:24-27` still reads

```
TODO: fill in before merging.
GitHub:      XXX
Code:        Zenodo.XXX
Publication: XXX
```

These are the repository, the archived-code DOI and the paper for the PyFraME
embedding implementation. The Zenodo DOI is minted per release, so there is
nothing to fill in until 0.5 exists. **Must be resolved before the PR is
merged.** While there, decide whether the copyright header should credit the
PyFraME authors as well as the PySCF developers — the file is an interface to
their work but carries only the standard PySCF header.

## [R] Blocked: CI and packaging

CI exercises none of this module, and cannot be made to. `pyscf.embedding`
imports `pyframe.embedding`, a subpackage that exists in no released PyFraME —
PyPI still tops out at 0.4.0 (potential *generation* only), confirmed
2026-09-08. So:

- `.github/workflows/run_ci.sh:20` has `pip install "pyframe>=0.5"` present but
  commented, with the reason inline. Uncommenting it today would install 0.4.0,
  the import would fail and the tests would skip — green CI exercising nothing.
  Pinning `>=0.5` for real would fail the install step outright, and the script
  runs under `set -e`.
- Installing from git instead is not an option either: the PyFraME remote is
  private over SSH, so CI has no credentials for it.
- `setup.py:24` declares `'embedding': ['pyframe>=0.5']` — the honest
  requirement even though it cannot resolve yet.

This stays a **merge consideration for pyscf-forge**: until the release, no
reviewer or user can install the dependency and exercise the module.

### On release, in order

1. Uncomment the `run_ci.sh` install and confirm the embedding tests **run**
   rather than skip — the log should show 30 passed, not 30 skipped.
2. Tighten the `setup.py` pin if 0.5 turns out not to be the version that
   carries the subpackage.
3. Re-run the suite against the *released* PyFraME rather than the local working
   copy, since the two may diverge. Re-check in particular
   `_check_vdw_parameters` (it probes four PyFraME properties by name), the vdw
   cross-check test (it encodes PyFraME's 6-12 convention and Lorentz-Berthelot
   combination as read from `engine/computation.cpp`), and `reset` (it relies on
   `QuantumSubsystem.coordinates` being the only geometry-dependent cache).
4. Decide whether the module docstring should state a minimum PyFraME version,
   and add the requirement to the CHANGELOG entry.

## [-] Won't fix — rationales, so they are not re-litigated

- **The quadrupole gradient stays a per-site `rinv` Python loop.** It cannot be
  batched into the `aux_e2`/`fakemol` route the order-0/1 branches use: that
  needs a three-index intor for the *third* derivative, and the available
  three-centre derivatives stop at second order (`int3c2e_ip1`, `ip2`, `ip1ip2`,
  `ipip1`, `ipip2`, `ipspsp1`, `ipvip1` — there is no `int3c2e_ipipip1`).
  Re-deriving it in terms of second-derivative integrals is a genuine
  derivation, not a review fix. Cost stays `O(nsites · 27 · nao²)`, the dominant
  term for a potential with quadrupolar sites. A comment in the code records
  why the branch is shaped unlike its neighbours.
- **PyFraME prints to stdout, bypassing `mol.stdout` and `verbose`.** `Creating
  from Path or str.` and `Calculation of environment energy in serial.` are
  `print` calls inside PyFraME and are not fixable from here; capturing them
  would mean redirecting stdout around every PyFraME call, which risks
  swallowing output that matters. Worth an upstream issue. PyFraME's
  Lennard-Jones *warnings* are separate and **are** routed into the PySCF
  logger, since they go through `warnings.warn`.
- **The import guards.** `embedding.py:40` and `embedding_gradient.py:27` catch
  `ImportError` and re-raise `from err`; the test guard at
  `test_embedding.py:32` probes `pyscf.embedding.embedding`, not `import
  pyframe`. The probe target matters: `import pyframe` does not pull in mpi4py,
  but `from pyframe.embedding import ...` does — so probing the package would
  report success on a machine where the tests cannot actually run. (Historically
  these caught bare `Exception`, because pre-`071f551` PyFraME raised
  `RuntimeError` out of mpi4py. That is fixed upstream and the guards are back
  to `ImportError`.)

## [i] Structure, against `pyscf.solvent`

Recorded 2026-09-10, from a read of `pyscf/solvent/{__init__,_attach_solvent,
pol_embed,ddcosmo}.py` and `pyscf/solvent/grad/pcm.py` in pyscf 2.14.0. Not
findings — a map of what this module borrows from solvent, what it does
differently, and why, so that a reviewer asking "why is this not like solvent?"
has an answer.

|                      | `pyscf.solvent`                            | `pyscf.embedding`                                   |
|----------------------|--------------------------------------------|-----------------------------------------------------|
| Environment state    | a `StreamObject` model (`PolEmbed`, `PCM`) | a `StreamObject` model (`EmbeddingBase` subclasses) |
| Attribute on method  | `with_solvent`                             | `with_embedding`                                    |
| Entry point          | `solvent.PE(...)`, per-model `*_for_scf`   | `embedding.polarizable` / `.electrostatic`          |
| Injection point      | `get_veff` → tagged array → `get_fock`     | as solvent                                          |
| Mixin tag class      | `_Solvation`                               | `_Embedding`                                        |

The environment responds to the density, so it belongs on solvent's
`get_veff`/tagged-array route rather than in a static one-electron term.

### Cloned from `_attach_solvent`, near line for line

`_attach_embedding.SCFWithEmbedding` is `SCFWithSolvent` with the names
changed — same `_keys`, same `__dict__.update` constructor, same `undo_…` via
`lib.view` + `lib.drop_class`, same `dump_flags`/`reset` delegation, same
`get_veff` → `lib.tag_array` (carrying the same direct-SCF caveat), same
add-`v`-before-DIIS trick in `get_fock`, same `energy_elec` accumulation into
`scf_summary`, same `Gradients = nuc_grad_method`. `embedding_gradient`'s
`make_grad_object`/`EmbeddingGrad` likewise mirror `solvent/grad/pcm.py`'s
`make_grad_object`/`WithSolventGrad`, down to `vac_grad.base = base_method`, the
name-mangled `set_class` and the disabled `_finalize`. Keep it that way: the
value of the wrapper layer is that it behaves exactly as a `pyscf.solvent` user
expects.

### Deliberately different

- **Model composition by inheritance.** `EmbeddingBase` → `ElectrostaticEmbedding`
  → `PolarizableEmbedding`, with `_compute_static_contributions`,
  `_density_dependent_contributions` and `_energy_terms` as declared extension
  points, each calling up to its parent so terms accumulate. `solvent` has no
  such hierarchy — its models are independent classes with duplicated
  boilerplate, and the one generic special case it needed (SMD's `e_cds`) is
  hardcoded into `SCFWithSolvent.energy_elec`. That hook is exactly what
  `_energy_terms` exists to avoid.
- **Option validation.** `_valid_option_keys`/`_check_option_keys` reject
  unknown keys; `PolEmbed` forwards the dict to CPPE unchecked.
- **`reset` distinguishes new-potential from new-geometry.** `PolEmbed.reset`
  unconditionally rebuilds the whole CPPE state; ours rebuilds the MM subsystems
  only when the potential changed, otherwise runs `_sync_quantum_subsystem`, and
  defers the static contributions via `_static_contributions_stale`. This is
  what makes a geometry optimisation affordable with a realistic potential.
- **Memory-aware integral driver.** `EmbeddingIntegralDriver` + `_blocked_range`
  batch against `max_memory` as a separate collaborator; `PolEmbed` inlines the
  same work behind a manual `n_chunks` argument.
- **Caching lives in the model.** `SCFWithSolvent.get_veff` assigns
  `with_solvent.e/.v` itself, guarded by `frozen`; ours lets `EmbeddingBase.kernel`
  own `e`, `v` and `_dm`. `_dm` is kept so the gradient can tell whether the
  induced dipoles held by the classical subsystem belong to the density matrix
  it was handed — `PolEmbed` keeps a `_dm` too but does not use it for that.
- **Lazy package import.** `embedding/__init__.py` uses PEP 562 `__getattr__` so
  the package imports without PyFraME (see the import-guard note above);
  `solvent/__init__.py` imports its models eagerly and defers only `pol_embed`.
- **`to_gpu` refuses rather than half-works.** Both wrappers raise and point at
  `.undo_embedding().to_gpu()`, where `WithSolventGrad.to_gpu` instead asserts
  the model is PCM. Refusing avoids handing back a GPU method whose potential
  has silently vanished.
- **Type-checked model selection.** `_embedding` rejects an
  `ElectrostaticEmbedding` object passed to `polarizable()` and vice versa,
  since both models read the same potential file and the call would otherwise
  read as the other one. `solvent.PE` only asserts non-`None`.
- **Per-term energy report** from `energy_contributions()` in `_finalize`,
  against solvent's single `Solvent Energy` line.

### Gaps against `solvent` — scope, not structure

- **SCF only.** `_attach_solvent` provides `_for_casscf`, `_for_casci`,
  `_for_post_scf`, `_for_tdscf`, plus `gen_response`, `Hessian`, `stability` and
  `TDA`/`MP2`/`CCSD`/`CASCI` overrides. We have `_for_scf`; `make_grad_object`
  raises `NotImplementedError` for anything else.
- **No `frozen` / `dm=` frozen-potential mode**, and none of `state_id`,
  `max_cycle`, `conv_tol`, `equilibrium_solvation`. These exist in `solvent`
  only to serve the post-SCF and TDDFT attachments, so they follow from the
  point above rather than being independent omissions.
- **Not registered on the method classes.** `solvent` gets `mf.PE()`/`mf.PCM()`
  shortcuts; embedding is reachable only as `embedding.polarizable(mf, ...)`.
  Being in forge, there is no `scf.hf.SCF` hook to hang it on — revisit if the
  module moves into main pyscf.
