# `pyscf/embedding` — notes

**Branch:** `pyframe_interface` · **Last updated:** 2026-09-28

*(What is still open, what the module relies on in PyFraME, and a map of the
module against `pyscf.solvent`. It is not a history: rationales for design
choices live in comments next to the code, and the reasons for test
tolerances in the tests. This file is tracked and ships with the branch, so
write it for a reviewer.)*

## Tasks left

Each is written up in the section named; this is the index. In rough order of
urgency — the first gates CI, the next three gate the merge.

- [ ] **Push this branch and watch the first CI run.** *Open: landing the
  branch*.
- [ ] **Fill in the citation placeholders** and settle the copyright header.
  *Open: citation placeholders*.
- [ ] **Take Python 3.8 out of the CI matrix, or guard the PyFraME install.**
  *Open: CI and packaging*.
- [ ] **PyFraME 0.5.0 has to be tagged** before CI can pass: `run_ci.sh`
  installs the tag. *Open: landing the branch*.
- [ ] **On a PyFraME release:** pin CI to it, decide on a `setup.py` extra,
  re-run against it, fix the stated minimum version. *On release, in order*.
- [ ] **Correlated and excited-state gradients** (post-SCF, CASCI/CASSCF,
  TDSCF). Parity item 9.
- Blocked: method-class shortcuts (`mf.PE()`), parity item 10, unless the
  module moves into main pyscf.

## Development environment

Work in the `kosmos` conda env, which has PyFraME 0.5.0 (its `release-0.5`
branch, before the tag) installed editable from
`~/PycharmProjects/pyframe`
(`git@gitlab.com:pyframe-project/pyframe.git`). The suite runs in a plain shell,
without `module load mpi`: PyFraME imports `mpi4py.MPI` only where a
communicator is used, and this module never passes one.

```
~/miniconda3/envs/kosmos/bin/python -m pytest pyscf/embedding/test/test_embedding.py -q
```

**Check which copy of the module is loaded** before trusting a run:
`python -c "import pyscf.embedding.embedding as e; print(e.__file__)"`, from
where the tests run.
- `pyscf_forge` is installed editable, pointing at
  `~/PycharmProjects/pyscf-forge`. Run from the root of that checkout, or of a
  `git worktree`, the tests load that tree, since the current directory comes
  first.
- A *non-editable* install must not come back: its
  `site-packages/pyscf/embedding/` comes first on `pyscf.__path__` and
  silently shadows every working tree.
- An empty leftover of one, `site-packages/pyscf/embedding/test/__pycache__/`,
  is still there as of 2026-09-24. A *script* has its own directory on
  `sys.path` instead of the current one, so it finds the leftover first, gets
  an empty namespace package, and fails with `no attribute 'polarizable'`.
  That includes the examples. Fix:
  `rm -r ~/miniconda3/envs/kosmos/lib/python3.12/site-packages/pyscf/embedding`.
- Without any install, the plugin is found only through the current
  directory, by `pkgutil.extend_path` in `pyscf/__init__.py`, so run the
  examples as `PYSCF_EXT_PATH=$PWD python examples/embedding/00-scf_energy.py`.

**Last verified 2026-09-28:** 87 passed (22 subtests), against PyFraME 0.5.0
as on its `release-0.5` branch before the tag, pyscf 2.14.0. The three
examples ran the same day, from the checkout with `PYTHONPATH` set, with
PyFraME's warnings made errors.

**`qrunch` conflicts with this module.** `qrunch 1.5.0a8` requires
`pyframe~=0.4.0`, and 0.4.0 is the released PyFraME, which has no
`pyframe.embedding`. Installing qrunch into the env silently removes this
module's dependency, and the whole suite skips. If the suite reports
everything skipped, check `pip show pyframe` first.

PyFraME writes nothing to stdout. It reports through a `pyframe` logger that
carries a `NullHandler` (`pyframe/log.py`). Its warnings -- the molecule
against the potential, sites without Lennard-Jones parameters -- go through
`warnings.warn`, and `_relayed_warnings` re-emits those raised while a model
is built or given new nuclei through the PySCF logger.

## [ ] Open: landing the branch

This module needs PyFraME 0.5.0: its embedding models, including `relax`,
`density_matrix`, `update_geometry`, and `nuclear_terms`, which came in with
the release. It is not tagged yet, the last tag, `0.5a0.dev1`, lacks the
models, and nothing is on PyPI. `run_ci.sh` installs the tag `0.5.0`, so CI
fails at the install until the tag exists, and the `ImportError` in
`embedding.py` names the release. `origin/pyframe_interface` is `5779f30`,
which pins the commit `2632ce6`; what is on top of it needs 0.5.0.

- [ ] **Push, and watch the first CI run.** The run should show the embedding
  tests *running*, not skipped. It is the first run against the new PyFraME
  off this machine. It is also the first with `mpi4py>=4.1` as a hard install
  requirement of PyFraME. The binary wheel should install on the GitHub runner
  without a system MPI, and nothing serial loads MPI, but only the CI log will
  confirm it. If the wheel does not install there, the fix belongs in
  `run_ci.sh`: install an MPI first, e.g. `apt-get install libopenmpi-dev`.

## [x] Moving host-agnostic work into PyFraME

PyFraME stays host-agnostic and does everything between a host's integrals,
its nuclei as plain arrays, and its density matrices. This module keeps only
what is PySCF's:
- the integral driver (`integral_driver.py`);
- the wrappers that attach a model to PySCF's methods (`_attach_embedding`,
  `make_grad_object`, `make_hess_object`);
- the CPHF and response plumbing: spin blocks, `equilibrium_solvation`, the
  first-order Fock matrix;
- the conversion of the `Mole` to arrays, and the output.

**Adopted, 2026-09-26, against `2632ce6`.** `PolarizableEmbedding` and
`ElectrostaticEmbedding` here are `StreamObject`s holding one of PyFraME's
models (`pyframe.embedding.model`, imported as `pyframe_model`) as `model`:
- they hand it the options unchanged, less `"potential"`
  (`EmbeddingOptions.from_dict`);
- they convert the `Mole` for `check_nuclei`, `update_nuclei`, and
  `update_geometry`, and make one `EmbeddingIntegralDriver` per geometry;
- `kernel` is `energy_and_fock`, `_B_dot_x` is `response`, `frozen` is
  `freeze`/`unfreeze`, `relax` is `relax`, and the gradient, Hessian and Fock
  derivatives are the model's;
- `energy_contributions` puts this module's labels on the model's terms.

Everything the fourteen items of this section named went with that:
`_read_subsystems`, `_describe_potential`, the option parsing and checking,
`_check_vdw_parameters` (and with it the Good-Hope gap, which
`has_lennard_jones_pairs` closes), the checks in `_sync_quantum_subsystem`,
`_frozen_energy`, `induce_dipoles`, `field_nuclear_derivatives`,
`perturbed_induced_dipoles`, `_potential_interaction_gradient`,
`_energy_gradient`, the sum in `embedding_hessian.kernel`, the `V_ind(mu^A)`
loop in `fock_gradient`, and the drivers' `_memo`. The three drivers became
one, which implements the template's batched forms; the gradient and Hessian
modules are now the PySCF wrappers and a reshape. The finite-difference tests
of the removed code stay, and test PyFraME's version: `mu^A`, the Fock
derivative, the Hessian at fixed density, the frozen gradient.

**What changed for a user**, all of it PyFraME's behaviour now:
- `polarizable` refuses a potential without polarizabilities, naming
  `electrostatic`, where it used to solve for dipoles that are zero;
- setting `frozen = True` with no density to freeze at raises there, rather
  than at the next `kernel`;
- a frozen model's `energy_contributions` add up to its energy, the induction
  term linearized at the frozen density (see *Deliberately different*);
- the Lennard-Jones terms are computed at the first `kernel`, not when the
  object is built, so `energy_contributions` has nothing before it.

**Seven more, adopted 2026-09-28, against PyFraME 0.5.0**, which took them
in after they were found while adopting the models:
- the macro iterations of `PostSCFWithEmbedding.kernel` and
  `CASCIWithEmbedding.kernel` are the model's `relax`, through
  `EmbeddingBase.relax`, which keeps `e` and `v` on the potential held;
- `_dm` is the model's `density_matrix`, and `frozen = True` freezes there;
- a geometry step in `reset` is one `update_geometry`;
- `fock_gradient` hands an electrostatic model no density, rather than a zero
  matrix;
- the tests read the density-independent terms from `nuclear_terms`, rather
  than from a kernel on a zero density.

Two came for free: the warning about nuclei without Lennard-Jones parameters
is logged once rather than at every geometry step, and freezing at the
density the model was just polarized by holds its potential exactly, so
`test_frozen_potential_does_not_follow_the_density` is back at 1e-12.

**What changed for a user:**
- `conv_tol` is on the potential, not the energy: the macro iteration stops
  when no element of `v` changes by `conv_tol` (default 1e-6) or more. That
  is the self-consistency itself; a correlated energy also carries the
  method's own convergence, e.g. CCSD's, which kept moving by 1e-8 per cycle
  once the potential had settled to 2e-10;
- a post-SCF method in an `ElectrostaticEmbedding` runs once, where the loop
  ran it three times over a potential that cannot change;
- the macro iteration warns through the PyFraME warning, relayed to the
  output, when it runs out of cycles.

## [ ] Open: citation placeholders

`pyscf/embedding/embedding.py:24-27` still reads

```
TODO: fill in before merging.
GitHub:      XXX
Code:        Zenodo.XXX
Publication: XXX
```

These are the repository, the archived-code DOI and the paper for the PyFraME
embedding implementation. The repository URL and the paper can go in now. Only
the Zenodo DOI waits, since it is minted per release and a git tag such as
`0.5a0.dev1` does not get one. **Must be resolved before the PR is merged.**
While there, decide whether the copyright header should credit the PyFraME
authors as well as the PySCF developers. The file is an interface to their
work but carries only the standard PySCF header.

## [ ] Open: CI and packaging

`pyscf.embedding` needs `pyframe.embedding`, which no PyFraME on PyPI has: PyPI
still stops at 0.4.0, potential generation only (re-checked 2026-09-24 with
`pip index versions pyframe`). Until a release carries it, the dependency
cannot be declared the ordinary way:

- **`setup.py` declares no `embedding` extra, and must not grow one back.** An
  extra pinning `pyframe>=0.5` cannot resolve against PyPI. It would turn
  `pip install pyscf_forge[embedding]` into a hard failure while promising a
  dependency nobody can get.
- **`.github/workflows/run_ci.sh:21` installs PyFraME from git, at the tag
  `0.5.0`**, which does not exist yet. A fixed pin makes a CI run a statement
  about a fixed PyFraME rather than about the `master` of that day. Move the
  pin deliberately, re-running the suite locally against the new PyFraME
  first.
- [ ] **The install fails on Python 3.8, which `ci.yml` still builds.** PyFraME
  sets `python_requires = >=3.10` (its `setup.cfg:41`), so pip refuses the git
  URL on 3.8 with a Requires-Python error. `run_ci.sh` runs under `set -e`, so
  the whole Install step fails, and with it the 3.8 job. The matrix is
  `["3.8", "3.10", "3.12"]` (`ci.yml:20`). Either 3.8 leaves the matrix (it is
  long out of support) or the install needs a version guard. It is left alone
  here, since it changes shared CI rather than this module.
- Installing an unreleased dependency from a git tag is a **merge
  consideration for pyscf-forge**, and the thing a reviewer will ask about
  first.

### On release, in order

1. Replace the git install in `run_ci.sh` with the released package, and
   confirm the embedding tests **run** rather than skip: the log should show
   them passed, not skipped.
2. Decide whether `setup.py` should then carry an `embedding` extra after
   all, pinned at the version that actually ships the subpackage.
3. Re-run the suite against the *released* PyFraME rather than the local
   working copy, since the two may diverge. Re-check in particular:
   - the vdw cross-check test, which encodes PyFraME's 6-12 convention and
     Lorentz-Berthelot combination, as read from `engine/computation.cpp`;
   - `reset` and `_build_model`, which rely on `update_geometry` and
     `update_nuclei` carrying over
     everything but the positions and charges of the nuclei (fragments,
     formal charge, Lennard-Jones parameters, `kind`), and on
     `check_nuclei` comparing atomic numbers rather than charges;
   - the energy report, which relies on the names of the terms of
     `EnergyContributions` (`_term_labels`).
4. The CHANGELOG entry and the examples say "PyFraME 0.5 or newer", which is
   the release that ships the models. Decide whether the module docstring
   should state it too.

## [ ] Open: parity with `pyscf.solvent`

`solvent` attaches to SCF, CASCI/CASSCF, post-SCF and TDSCF, and so does this
module. What is still short of it is the correlated and excited-state
gradients (9) and the method-class shortcuts (10). The list is in the order
the work was done, which was dependency order. Where a done item departs from
`solvent`, the reason is under *Deliberately different*.

- [x] **1. `_B_dot_x`**, the environment's response to a trial density.
  Zero for a model without induced dipoles; for `PolarizableEmbedding`, the
  dipoles the trial density's electronic field alone induces, as in
  `PolEmbed._B_dot_x` with CPPE's `elec_only`.
- [x] **2. `gen_response`**, on `SCFWithEmbedding`, adding `_B_dot_x` to the
  bare response. It is gated on `equilibrium_solvation and not frozen`, read
  inside `vind`. This is what SOSCF, stability, polarizabilities and the
  Hessian use.
- [x] **3. `frozen`**, through `dm=` on `polarizable` / `electrostatic`.
  `EmbeddingBase.kernel` holds the potential of that density.
- [x] **4. `stability`**, as `_attach_solvent.py:173-182`: with
  `equilibrium_solvation=not frozen` for the duration.
- [x] **5. Post-SCF** (`MP2`, `CISD`, `CCSD`), through `PostSCFWithEmbedding`.
  Unfrozen, the environment is relaxed against the correlated density by a
  macro iteration, as in `PolEmbed`, until the potential changes by less
  than `conv_tol`, in at most `max_cycle` runs.
- [x] **6. CASCI and CASSCF**, through `CASCIWithEmbedding` and
  `CASSCFWithEmbedding`, with `state_id` for multi-root CASCI. The potential
  goes into `get_hcore`, and the `Tr(D v)` that adds to every energy is taken
  back out where each energy is reported, as in `_attach_solvent.py:232-514`.
- [x] **7. TDSCF** (`TDA`, `TDHF`, `TDDFT`), through `TDSCFWithEmbedding`,
  after `pol_embed.pe_for_tdscf`. Non-equilibrium is the default: a vertical
  excitation sees the environment only through the ground state's orbitals.
  `equilibrium_solvation` turns the linear response on, on a shallow copy of
  the ground state's model, so that the flag does not reach it.
- [x] **8. Hessian**, for both models; a frozen potential is refused. See
  *How the Hessian is built* below.
- [~] **9. Gradients.** The SCF gradient is done, for both models, frozen
  included. What remains is the **correlated** gradient: on
  `PostSCFWithEmbedding`, the two CAS classes and `TDSCFWithEmbedding` (for
  which `solvent` does have one, in `grad/ddcosmo_tdscf_grad.py`). All four
  raise.
  - Relaxed, the environment is a functional of a density that is not the
    SCF's, so the gradient carries the `DM * V[d/dX DM] + V[DM] * d/dX DM`
    terms that `_attach_solvent` warns about and evaluates anyway.
  - Those terms vanish for `ElectrostaticEmbedding`, whose potential does not
    depend on the density. There the obstacle is only plumbing: each method's
    gradient builds `hcore` from `._scf` and would miss `dv/dR`, and the Pulay
    terms are built from a generalized Fock matrix that has to be the one the
    wavefunction was optimized with.
- [R] **10. Register on the method classes.** `solvent` gets `mf.PE()` /
  `mf.PCM()` shortcuts; this module is reachable only as
  `embedding.polarizable(mf, …)`. Being in forge, there is no `scf.hf.SCF`
  hook to hang it on. **Blocked**; revisit only if the module moves into main
  pyscf.

### How the Hessian is built

`embedding_hessian.py` follows the shape of `solvent/hessian/pcm.py`:
- `EmbeddingHess.kernel` runs the vacuum Hessian with `equilibrium_solvation`
  on, so the environment is in the orbital Hessian the CPHF equations use.
- It then adds the second derivative of the embedding energy at fixed
  density, the model's `hessian`, reshaped to PySCF's layout.
- `make_h1` adds the potential's derivative to the first-order Fock matrix,
  the model's `fock_derivatives`.

**PyFraME assembles every term** (`pyframe.embedding.totals`), calling this
module's `EmbeddingIntegralDriver` for the electronic ones, induction and the
Lennard-Jones terms included. The response of the induced dipoles to the
nuclei, `mu^A`, which both the Hessian and the Fock derivative need, is
solved once per density and kept by the model, so a polarizable Hessian does
its `3 natm` perturbed solves once.

**The driver** rests on the second AO derivatives of the multipole potential.
At order L that means L + 2 derivatives in all:
- charges batch over sites with a fakemol;
- dipoles and quadrupoles re-centre rinv per site and need the third- and
  fourth-derivative intors (`int1e_ipipiprinv`, `int1e_ipiprinvip`,
  `int1e_ipipipiprinv`, `int1e_ipipiprinvip`, `int1e_ipiprinvipip`).

The induced dipoles' contribution, `mu·F_el^{AB}`, comes from the same
integrals: the induced-dipole operator is the order-1 multipole operator of
`-mu`, the minus sign being the Taylor coefficient the permanent dipoles carry
and the induced ones do not (`_dipole_multipoles`, as the template's
`induced_dipoles_potential_integrals` documents). The integrals serve every
nucleus at once, so the driver implements the template's batched forms
(`all_electronic_field_nuclear_derivatives`,
`full_electronic_electrostatic_energy_hessian`,
`all_electronic_electrostatic_fock_gradients`,
`all_electronic_induction_fock_gradients`), which PyFraME calls in preference
to the per-nucleus ones.

For induction, at fixed density:

    d2E_ind/dR_A dR_B = -mu·F^{AB} - F^{A}·mu^{B},    mu^{B} = B^-1 F^{B}

with `F = F_el(D,R) + F_nuc(R) + F_mult` and `B` fixed, since the sites do not
move.

The tests check each piece against finite differences of its first
derivative, and then the whole:
- the Hessian integrals at every multipole order;
- the driver methods PyFraME calls;
- the classical terms;
- the Fock derivative;
- the second derivative at fixed density;
- the whole RHF Hessian, against finite differences of the analytic gradient,
  to 2.9e-6. That tolerance is set by noise in the differenced gradients;
  the bare RHF Hessian of the molecule differs from its own finite
  differences by the same amount. Leaving out either induction term moves
  the Hessian by 6e-4, so the tolerance cannot hide one.
- closed-shell UHF against RHF.

## What this module relies on in PyFraME

A reference for reviewing, and for when PyFraME moves. The history of how each
point came about is in PyFraME's `CHANGELOG.md`.

- **PyFraME 0.5.0 or newer**, for `pyframe.embedding.model` and what it
  rests on (`options`, `totals`, the batched template methods,
  `read_input.subsystems_from`, `QuantumSubsystem.check_nuclei`).
  `embedding.py` checks for `EmbeddingModel.relax`, the newest of it, at
  import and raises an `ImportError` naming the release. The test module skips only when `pyframe.embedding`
  itself is missing, e.g. 0.4.0 under the qrunch pin. A PyFraME that is
  merely too old fails collection with that message, so that it cannot empty
  the suite unnoticed.
- **The models do the embedding.** `EmbeddingBase.model` is a
  `pyframe.embedding.model.PolarizableEmbedding` or `ElectrostaticEmbedding`;
  everything but the PySCF side goes through its public methods. The only
  things read from it by name are the terms of `energy_contributions()`, for
  the report's labels.
- **The options are PyFraME's**, read by `EmbeddingOptions.from_dict` against
  the model's `option_keys`, which refuses unknown keys. The `"potential"`
  key is this module's own and is taken out first.
- **A JSON potential states its units.** `read_input.reader` refuses one
  without a `units` block and converts everything to atomic units.
  `test/butadiene_water.json` carries the atomic-units block PyFraME itself
  writes (`pyframe.units.ATOMIC_DOCUMENT_UNITS`). A potential written by an
  older PyFraME has to be regenerated, or given the block by hand if it is
  known to be in atomic units.
- **The potential comes in every form `read_input.subsystems_from` takes**:
  a JSON path or document, a `MolecularSystem` or `Snapshot`, or
  `Subsystems`, which are copied. `TestPotentialSources` checks that every
  form gives the energy the file does.
  `examples/embedding/02-potential_from_pyframe.py` builds the potential and
  the `Mole` from one PyFraME system.
- **The quantum subsystem is the terminated core region**, and the `Mole` has
  to be that molecule, link and capping atoms included.
  `QuantumSubsystem.check_nuclei` refuses a wrong atom count, saying how many
  link and capping atoms the potential expects, and warns on an element
  mismatch, on another charge, and on a capping atom with its element's full
  charge. It warns through `warnings`, which `_relayed_warnings` passes on to
  the PySCF output, as it does the Lennard-Jones warnings.
- **The environment sees the nuclei as PySCF has them.** `_sync_nuclei` gives
  the model the `Mole`'s coordinates and `atom_charges()` through
  `update_nuclei`. An atom with an ECP interacts through its effective
  charge, as in `pyscf.qmmm`, and a capping atom *is* one. A ghost atom
  interacts with nothing. Tested against `pyscf.qmmm` with `bfd-pp` on the
  carbons, energy and gradient.
- **The induced-dipole threshold is relative** to the size of the dipoles,
  and every solve starts from the previous one. The `threshold` option keeps
  PyFraME's default of 1e-8. An SCF wants that warm start, which saves a
  third to a half of the iterations.
- **A response solve stores nothing** on the subsystem. `_B_dot_x`, i.e.
  `model.response`, relies on that to leave the ground state's dipoles alone,
  and the TDSCF attachment on it to share the ground state's model.
- **The driver follows `IntegralDriverTemplate`** and subclasses it, which
  checks the three abstract methods when it is made and derives each
  per-nucleus method from the batched one implemented here.
- **Lennard-Jones parameters are per interaction and in atomic units.** A
  potential built with TIP3P Lennard-Jones parameters by a PyFraME before
  `2cdb94b` had them in nm and kJ/mol, and has to be rebuilt.
- **No simulation box is passed to the dipole solvers**: the environment is a
  cluster, with no minimum image convention. A box in the potential is kept
  by the model as `simulation_box` and logged.
- **The construction side** of PyFraME is used only by `make_pyframe_system`
  in the test module and by example 02. Both use only `MolecularSystem`,
  `get_fragments_by_name`, `set_core_region`, `add_region` and
  `Project.create_embedding_potential`. Re-run them when that side of PyFraME
  moves.

## [i] Structure, against `pyscf.solvent`

Recorded 2026-09-10, from a read of `pyscf/solvent/{__init__,_attach_solvent,
pol_embed,ddcosmo}.py` and `pyscf/solvent/grad/pcm.py` in pyscf 2.14.0. Not
findings, but a map of what this module borrows from solvent, what it does
differently, and why, so that a reviewer asking "why is this not like
solvent?" has an answer.

|                      | `pyscf.solvent`                            | `pyscf.embedding`                                   |
|----------------------|--------------------------------------------|-----------------------------------------------------|
| Environment state    | a `StreamObject` model (`PolEmbed`, `PCM`) | a `StreamObject` holding a PyFraME model            |
| Attribute on method  | `with_solvent`                             | `with_embedding`                                    |
| Entry point          | `solvent.PE(...)`, per-model `*_for_scf`   | `embedding.polarizable` / `.electrostatic`          |
| Injection point      | `get_veff` → tagged array → `get_fock`     | as solvent                                          |
| Mixin tag class      | `_Solvation`                               | `_Embedding`                                        |

The environment responds to the density, so it belongs on solvent's
`get_veff`/tagged-array route rather than in a static one-electron term.

### Cloned from `_attach_solvent`, near line for line

`_attach_embedding.SCFWithEmbedding` is `SCFWithSolvent` with the names
changed:
- the same `_keys` and `__dict__.update` constructor;
- the same `undo_…` via `lib.view` + `lib.drop_class`;
- the same `dump_flags` / `reset` delegation;
- the same `get_veff` → `lib.tag_array`, with the same direct-SCF caveat;
- the same add-`v`-before-DIIS trick in `get_fock`;
- the same `energy_elec` accumulation into `scf_summary`;
- the same `Gradients = nuc_grad_method`.

`embedding_gradient`'s `make_grad_object` / `EmbeddingGrad` likewise mirror
`solvent/grad/pcm.py`'s `make_grad_object` / `WithSolventGrad`, down to
`vac_grad.base = base_method`, the name-mangled `set_class` and the disabled
`_finalize`. Keep it that way: the value of the wrapper layer is that it
behaves exactly as a `pyscf.solvent` user expects.

### Deliberately different

- **The model is PyFraME's, not this module's.** `PolEmbed` is the whole
  model in pyscf, calling CPPE for the environment's side. Here
  `EmbeddingBase` holds a `pyframe.embedding.model` object and does only
  what is PySCF's; the two classes differ in nothing but which PyFraME model
  they hold, and the energy is reported term by term from the model's
  `energy_contributions`, so that a new term needs nothing here. `solvent`'s
  one generic special case (SMD's `e_cds`) is hardcoded into
  `SCFWithSolvent.energy_elec` instead.
- **Option validation.** PyFraME's `EmbeddingOptions.from_dict` rejects
  unknown keys; `PolEmbed` forwards the dict to CPPE unchecked.
- **The potential need not be a file.** `PolEmbed` takes a CPPE potential
  file. This module takes a PyFraME JSON file or document, or the PyFraME
  system that built the potential, in memory
  (`read_input.subsystems_from`).
- **The nuclei the environment sees are the `Mole`'s**, coordinates and
  charges both, so an ECP atom interacts through its effective charge, as in
  `pyscf.qmmm`. The potential's own nuclei only have to agree with the
  molecule in number and element.
- **`reset` distinguishes new-potential from new-geometry.** `PolEmbed.reset`
  unconditionally rebuilds the whole CPPE state. This module rebuilds the
  model only when the potential changed, and otherwise hands it the new
  nuclei and a new driver (`update_geometry`), which
  recompute what depends on them when next asked. That is what makes a
  geometry optimization affordable with a realistic potential. And it does
  not even that where nothing moved: a scanner resets on every call, and the
  post-SCF macro iteration calls its scanner on the same molecule every
  cycle, so a new driver there would throw away the integrals and, frozen,
  re-solve the dipoles of the frozen density each pass. `reset` compares the
  arrays the integrals are built from (`_integral_key`: `_atm`, `_bas`,
  `_env`, `_ecpbas`, `cart`) rather than the `Mole` object, which a scanner
  moves in place with `set_geom_`. `pyscf.solvent` has no counterpart:
  `PolEmbed.reset` rebuilds everything, and `ddcosmo.reset` nothing.
- **A memory-aware integral driver.** `EmbeddingIntegralDriver` caches the
  three-index integrals within a budget and batches against `max_memory`
  beyond it (`_blocked_range`). `PolEmbed` inlines the same work behind a
  manual `n_chunks` argument.
- **Caching lives in the model.** `SCFWithSolvent.get_veff` assigns
  `with_solvent.e/.v` itself, guarded by `frozen`. Here `EmbeddingBase.kernel`
  owns `e` and `v`, and PyFraME's model keeps what does not depend on the
  density and knows which density its induced dipoles belong to, so a
  gradient or Hessian right after the SCF solves nothing again. The density
  the potential belongs to is the model's too (`density_matrix`), and
  `frozen = True` freezes there.
- **Lazy package import.** `embedding/__init__.py` uses PEP 562 `__getattr__`,
  so the package imports without PyFraME and a test module can skip rather
  than fail at import. `solvent/__init__.py` imports its models eagerly and
  defers only `pol_embed`.
- **`to_gpu` refuses rather than half-works.** The wrappers raise and point at
  `.undo_embedding().to_gpu()`, where `WithSolventGrad.to_gpu` instead asserts
  the model is PCM. Refusing avoids handing back a GPU method whose potential
  has silently vanished.
- **Type-checked model selection.** `_embedding` rejects an
  `ElectrostaticEmbedding` object passed to `polarizable()` and vice versa,
  since both models read the same potential and the call would otherwise
  silently mean the other one. `solvent.PE` only asserts non-`None`.
- **Per-term energy report** from `energy_contributions()` in `_finalize`,
  against solvent's single `Solvent Energy` line.
- **`gen_response` asks whether the trial density is one the environment can
  see.** Shared by the SCF and TDSCF attachments, through
  `_add_embedding_response`. `_couples_to_the_density` reproduces the test the
  response function itself applies to its Coulomb term
  (`pyscf/scf/_response_functions.py`), and the environment, a functional of
  the total charge density, is added only where that term is.
  `_attach_solvent.gen_response` adds the solvent unconditionally, which is
  wrong for the RHF triplet kernel: its trial density is a spin density, the
  alpha and beta blocks cancel, the total charge density does not change, and
  the environment must not respond. RHF internal stability asks for exactly
  that kernel (`scf/stability.py:370`, `singlet=False`), so this is not
  hypothetical. The `hermi == 2` half of the test is only an economy:
  `_B_dot_x` returns zero there anyway, the field integrals being symmetric.
- **A frozen potential still follows the nuclei.** `frozen` means the
  environment does not respond to the *density*; the geometry is a separate
  matter, and a potential held at the geometry it was built for would be
  wrong in a scanner, which calls `reset(mol)` on every step. So `reset` keeps
  the frozen density and clears `e` and `v`, and the next `kernel` rebuilds
  the potential for that density at the new geometry, as PyFraME's
  `update_nuclei` does for a frozen model. `ddcosmo.reset` clears nothing, so
  a frozen solvent survives a scanner step carrying the previous geometry's
  potential. Freezing with no density to freeze at raises rather than
  leaving a model that would return `(None, None)`.
- **A frozen potential is held, its energy is not.** `frozen` fixes the
  potential at one density `D0`; the energy `kernel` reports is still that of
  the density it is handed: the environment's own energy plus
  `Tr((D - D0) v)`. `_attach_solvent` reports `e(D0)` for every density
  instead, which is not the functional the SCF makes stationary, since its
  Fock matrix carries `v(D0)`. That is not cosmetic: the gradient of what
  solvent reports needs a CPHF solve, and on this system the missing term is
  2 percent of the gradient. With the term, the frozen energy is stationary,
  its gradient is the ordinary one, and both models agree with finite
  differences to 2e-7. The CAS classes apply the same correction for the same
  reason. Two consequences: the energy converges before the density does, so
  a test comparing densities has to lean on `conv_tol_grad` rather than
  `conv_tol`; and at `D = D0` nothing changes, so freezing at a converged
  density reproduces it. This is PyFraME's `EmbeddingModel.freeze`, with the
  reasoning in its docstrings. It reports the induction term linearized,
  `E_ind(D0) + Tr((D - D0) V_ind(mu0))`, so that the terms of
  `energy_contributions` add up to the frozen energy.
- **`frozen` and `equilibrium_solvation` cannot contradict each other.**
  `gen_response` adds the environment only when `equilibrium_solvation and not
  frozen`: a frozen potential is a fixed external term, so its response to
  any first-order density is zero however the other flag is set. `ddcosmo`
  says as much in a comment (*this attribute has no effects if .frozen is
  enabled*), but `_attach_solvent.gen_response` reads only
  `equilibrium_solvation`, and only `stability` reconciles the two. Setting
  both by hand elsewhere gets a response out of a potential that cannot
  respond.
- **Every post-SCF pass re-converges the SCF, through the scanner.** Both
  branches of `PostSCFWithEmbedding.kernel` go through `_run_once`, which runs
  the bare method's scanner over the embedded SCF and copies the result back.
  `_attach_solvent` uses the scanner for its macro iteration but lets the
  frozen branch call `super().kernel()` on whatever orbitals the object was
  built with. A potential frozen at a density other than the SCF's *changes
  the SCF problem*, so those orbitals are generally not the ones that go with
  it: freezing a correlated method at its own relaxed density and re-running
  it came out 5.8 mHa off the calculation it should have reproduced. The cost
  is an SCF restart from the converged density, and the `*args` of `kernel`:
  an initial guess cannot be handed through a scanner, so one is refused
  rather than ignored.
- **The macro iteration converges on the potential, and is PyFraME's.**
  `_attach_solvent` cycles to a change of the total energy below `conv_tol`;
  here `relax` cycles to a change of `v` below it, which is what
  self-consistency means and does not inherit the method's own convergence
  noise. It also knows an environment that cannot respond, and runs the
  method once for it. The last run is made in the potential of the density
  before it, so a polarizable post-SCF method runs once more in the potential
  it settled on, as `_attach_solvent` does; CASCI's extra pass is the one that
  canonicalizes the orbitals.
- **The post-SCF hooks import the package that defines them.** `MP2`, `CISD`
  and `CCSD` are attached to the SCF classes by `pyscf.mp`, `pyscf.ci` and
  `pyscf.cc` rather than defined on them, so `super().MP2()` fails with
  `'super' object has no attribute 'MP2'` unless something imported the
  package first. Each hook imports its own package; `_attach_solvent` does
  not.
- **CASSCF polarizes the environment with the orbitals it was handed.**
  `mc1step.kernel` keeps the orbitals of the macro iteration in a local and
  puts them on the object only when it returns, so `self.mo_coeff` inside the
  `casci` override is still the starting guess. `_attach_solvent` builds the
  polarizing density there as `self.make_rdm1(ci=fcivec)`: the starting
  orbitals with the current CI vector, a density belonging to neither. This
  module passes `mo_coeff` through explicitly. On CASSCF(4,4)/STO-3G with the
  bundled potential, solvent's version moved the total energy by 8.3e-5 Eh
  and left the converged object's environment 0.01 off in the density. With
  the orbitals right, the environment the method ends with belongs to its
  own converged density to 1e-14.
- **The multi-root correction is one scalar, and that is not an oversight.**
  With several roots, every root's energy carries its own `Tr(D_r v)` from
  `get_hcore`, but only one density polarizes the environment. What has to be
  added to each root is the part of the embedding energy that does *not* come
  from the density it acts on, `e - Tr(D_state v)`: the same number for every
  root, each root keeping its own `Tr(D_r v)` as its interaction with the
  environment the chosen state made. `_attach_solvent` does the same; the
  reasoning is written down because the expression invites the opposite
  conclusion.
- **Casida stays available.** `_attach_solvent` sets `CasidaTDDFT =
  NotImplemented`. The environment kernel is real, symmetric and frequency
  independent, so it enters A and B alike and leaves A - B untouched, which is
  what `(A-B)^(1/2)(A+B)(A-B)^(1/2)` rests on. A test checks that
  `CasidaTDDFT` and the full non-Hermitian solver agree, to 1e-12 on
  PBE/STO-3G with the response switched on.
- **`dm` is refused on the TDSCF attachment.** Everywhere else it chooses the
  density the potential is held at. A TDSCF object has no potential of its
  own to hold, since the ground state's reaches it through the orbitals of
  `._scf`, so all it could do is set an `e` and a `v` that nothing reads.
  `solvent` accepts it, and it is inert there too. Refusing it also keeps the
  shallow copy of the ground state's model safe: nothing on this path calls
  `kernel`, so the MM subsystems it shares with the ground state's model are
  never written to.
- **Every second derivative is checked on its own**, not only through the
  total; see *How the Hessian is built*.
- **ROHF is named explicitly.** `_attach_solvent` decides how to fold the spin
  blocks with `isinstance(self, scf.uhf.UHF)`, but `ROHF` subclasses `RHF`,
  not `UHF`, while taking the UHF-style response, so solvent hands it a
  per-spin environment response instead of one built from the total density.
  This module tests for `(scf.uhf.UHF, scf.rohf.ROHF)`, matching how
  `_response_functions.py` assigns the two kernels.

Everything missing against `solvent` is scope, not structure, and is tracked
under *Open: parity with `pyscf.solvent`*.
