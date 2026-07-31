# `pyscf/embedding` — residual notes

**Branch:** `pyframe_interface` · **Last updated:** 2026-09-24
*(This is the trimmed remnant of a full code review of the PyFraME interface.
Every reviewable finding is closed; what follows is only what is still open,
plus a map of the module against `pyscf.solvent`. Rationales for individual
design choices live in comments next to the code they explain, not here. This
file is tracked and ships with the branch, so write it for a reviewer, not just
for yourself.)*

## Tasks left

Each is written up in the section named; this is the index. In rough order of
urgency — the first gates CI, the next three gate the merge.

- [ ] **Push this branch and watch the first CI run.** PyFraME's side is
  pushed and tagged `0.5a0.dev1`; this side is not. *Open: landing the two
  sides together*, task 1.
- [ ] **Fill in the citation placeholders** and settle the copyright header.
  *Open: citation placeholders*.
- [ ] **Take Python 3.8 out of the CI matrix, or guard the PyFraME install.**
  *Open: CI and packaging*.
- [ ] **On a PyFraME release:** pin CI to it, decide on a `setup.py` extra,
  re-run against the release, fix the stated minimum version. *On release, in
  order*.
- [ ] **Induction Hessian** for `PolarizableEmbedding`, parity item 8; the
  recipe is under *The induction Hessian, the rest of 8*.
- [ ] **Correlated and excited-state gradients** (post-SCF, CASCI/CASSCF,
  TDSCF), parity item 9.
- Blocked: method-class shortcuts (`mf.PE()`), parity item 10 — only if the
  module moves into main pyscf.

## Development environment

Work in the `kosmos` conda env, which has PyFraME `0.5a0.dev1` installed
editable from `~/PycharmProjects/pyframe`
(`git@gitlab.com:pyframe-project/pyframe.git`). No MPI setup is needed — the
suite runs in a plain shell, without `module load mpi`. PyFraME lists `mpi4py`
as an install requirement again as of `0.5a0.dev1`, but imports `mpi4py.MPI`
only where a communicator is used, which this module never passes.

```
~/miniconda3/envs/kosmos/bin/python -m pytest pyscf/embedding/test/test_embedding.py -q
```

**Check what you are testing first.** On 2026-09-24 at 10:28 `pyscf_forge`
1.1.1 was installed into `kosmos` again, *non-editable*, from this checkout at
`f93a2a5`. Its `site-packages/pyscf/embedding/` comes first on `pyscf.__path__`
and shadows the working tree, so the command above tests that frozen copy
whatever the working tree holds.
`python -c "import pyscf.embedding.embedding as e; print(e.__file__)"` shows
which copy is loaded. Either `pip uninstall pyscf_forge`, which restores what
the next paragraph describes, or reinstall it with `pip install -e .`. Until
then, put the working tree first by hand, i.e.
`pyscf.__path__.insert(0, 'pyscf')` before importing `pyscf.embedding`, which
is how the 2026-09-24 verification below was run.

**Run it from the repository root.** Without that install, the plugin is found by `pkgutil.extend_path` in
`pyscf/__init__.py`, which scans `sys.path` for a `pyscf/` directory, and the
repository root is on it only as the working directory. From the root, pytest
and `python -c` import the working tree (check with
`python -c "import pyscf.embedding as e; print(e.__file__)"`). A script run
from anywhere else — the examples included, whose own directory is what goes on
`sys.path` — does not find the working tree. Run the examples as
`PYSCF_EXT_PATH=$PWD python examples/embedding/00-scf_energy.py`.

**Last verified 2026-09-24:** 83 passed (22 subtests), against PyFraME
`0.5a0.dev1` (`9dc0e0f`, pushed and tagged, the tree CI now installs), pyscf
2.14.0, in a shell with no MPI module loaded. The examples were last run
2026-09-23, against the same PyFraME code before it was committed.

PyFraME `0.5a0` and `qrunch` cannot both be satisfied: `qrunch 1.5.0a8`
requires `pyframe~=0.4.0`, and 0.4.0 is the released version that has no
`pyframe.embedding`. Installing qrunch therefore silently takes this module's
dependency away and the whole suite skips. If the suite reports 0 run and
everything skipped, check `pip show pyframe` first.

`test_vdw_gradient_matches_explicit_lennard_jones` used to be mildly flaky: it
asserted that two SCF runs differing only in a term that never enters the Fock
matrix give the same density to 1e-9, and one run in three landed at 2e-9. The
runs are numerically independent, so the tolerance was marginal; nothing was
wrong with the vdw terms. The density is compared to 1e-8 since 2026-09-24. The
gradient check it guards is unchanged, at 1e-10.

PyFraME does not write to stdout. It reports through one `pyframe` logger
carrying a `NullHandler` (`pyframe/log.py`), so nothing is emitted unless the
caller configures handlers. Its Lennard-Jones warnings go through
`warnings.warn` and are re-emitted through the PySCF logger by
`_check_vdw_parameters`. Nothing to do here.

## Following PyFraME: units, terminated core regions, potentials in memory

PyFraME changed under this interface in `2cdb94b` ("Unite potential
construction and embedding parts", 2026-09-19) and `abc34b6` ("Make fragments
and atoms immutable", 2026-09-20), both on `master`; the *Unreleased* section
of its `CHANGELOG.md` is the list. (The first of the two was rewritten out of
the branch it was developed on, so the commits an earlier version of these
notes cited — `7045ac4`..`d6598bd` — no longer exist.) What each change meant
here:

- **A JSON potential has to state its units.** `read_input.reader` refuses a
  document without a `units` block (keys `length`, `energy`, `angle`,
  `multipoles`, `polarizabilities`) and converts everything to atomic units
  from it. This was the only change that broke the suite: every test failed at
  the first read. `test/butadiene_water.json` now carries the atomic-units
  block PyFraME itself writes (`pyframe.units.ATOMIC_DOCUMENT_UNITS`); its
  values were in atomic units already, so no number changed. Older PyFraME
  readers ignore the key, so the file still reads there. A potential written by
  an older PyFraME has to be regenerated, or given the block by hand if it is
  known to be in atomic units; the error PyFraME raises shows the block.
- **The potential can be handed over in memory.**
  `MolecularSystem.to_subsystems()` and `Snapshot.to_subsystems()` build the
  subsystems with no file in between, which is what that PyFraME work was for.
  `polarizable` / `electrostatic` / the model classes now take, as the
  potential, a JSON path, a JSON document (`pyframe.writers.potential_document`),
  a system or snapshot, or a `read_input.Subsystems`; in an options dictionary
  it goes under the key `"potential"`, which replaces `"json_file"` (renamed
  2026-09-19; the old key is refused as unknown, like any other). See
  `EmbeddingBase._read_subsystems`. A
  `Subsystems` is deep-copied, since the model writes into its subsystems and
  one object may be handed to several models; a system is asked for fresh
  subsystems each time. `dump_flags` names a potential held in memory rather
  than printing it. `TestPotentialSources` checks that every form gives the
  energy the file does, and `examples/embedding/02-potential_from_pyframe.py`
  builds the potential and the `Mole` from one PyFraME system.
- **The quantum subsystem is the terminated core region.** Its nuclei are the
  atoms of the core region followed, per fragment, by the hydrogen link atoms or
  capping atoms that terminate cut bonds, each recorded as `Nucleus.kind`
  (`atom`/`link`/`cap`), with the charge of the terminated molecule on the
  subsystem. The `Mole` has to be that molecule. `_sync_quantum_subsystem` now
  says how many link and capping atoms the potential expects when the atom
  count is off, warns when the subsystem's `charge` is not `mol.charge`, and
  warns about a capping atom that carries no ECP in the `Mole`.
- **The environment sees the nuclei as PySCF has them.** The charges of the
  quantum subsystem are now taken from `mol.atom_charges()` — as the
  coordinates always were from `mol.atom_coords()` — where they used to be the
  potential's, with only a warning on a mismatch. The potential gives every
  nucleus the full charge of its element, which was wrong for any atom with an
  ECP: its core electrons are not in the density, so the nuclear-electrostatic,
  nuclear-field and induction terms counted the environment's interaction
  with charge that is not there. Capping atoms made this the normal case
  rather than an edge case, since a capping atom *is* an ECP. A ghost atom now
  interacts with nothing, as it should (tested). The element check that remains
  compares elements, not charges. Tested against `pyscf.qmmm` with `bfd-pp` on
  the carbons, energy and gradient.
- **Sites near the boundary are adjusted** by PyFraME (a capped atom's site
  removed, a link atom's neighbours' charge redistributed). That changes the
  potential, not its form; nothing to do here.
- **Lennard-Jones parameters are per interaction and in atomic units**, and
  TIP3P's used to reach `pyframe.embedding` in nm and kJ/mol. The `vdw` path
  reads the `rep_lj`/`disp_lj` arrays of the subsystems, which are unchanged; a
  potential built with TIP3P Lennard-Jones parameters before this change is
  wrong and has to be rebuilt.
- **A simulation box is in angstrom** on the PyFraME side, with
  `SimulationBox.box_in_bohr` for the solvers. This interface never passes a
  box to the dipole solvers (no minimum image convention: the environment is a
  cluster), so nothing changed; the box is now kept on the model as
  `simulation_box` and its presence is logged. If periodic treatment is ever
  wanted, it is `box_in_bohr` that the solvers take.
- `compute_electrostatic_interaction` no longer takes a list of classical
  subsystems, and `Atom` stores its polarizabilities as blocks keyed by orders.
  Neither is used here.
- **The fragments and atoms of a structure are immutable** (`abc34b6`), which
  reworked the potential-*construction* side: a `Topology` holds the bonds, a
  region owns its capped fragments and concaps, an `Atom` carries a `kind`, and
  `Fragment.copy_with` and friends replace mutation. Nothing in
  `pyframe.embedding` changed with it, so nothing in this module did either.
  The two places here that drive the construction side are
  `make_pyframe_system` in the test module and
  `examples/embedding/02-potential_from_pyframe.py`, and both use only
  `MolecularSystem`, `get_fragments_by_name`, `set_core_region`, `add_region`
  and `Project.create_embedding_potential`, which the rework left alone. They
  are what to re-run when that side of PyFraME moves again.
- **PyFraME writes American English** as of `abc34b6`, which renamed
  `LennardJonesParameters.parameterised` / `.unparameterised` to `…ized`. This
  module names neither (it reads `rep_lj`/`disp_lj`), so nothing broke; the
  prose here and the `parameterised` keyword of the test helpers are our own
  and are still British.

## Following PyFraME: read-only arrays, immutable particles, a relative threshold

A second round of PyFraME work (2026-09-21 to 2026-09-23, *Settled upstream*
below) changed things this module used. It is pushed upstream as `da663f1` and
`9dc0e0f`, tagged `0.5a0.dev1`. What it meant here:

- **`_sync_quantum_subsystem` is one `with_nuclei` call.** A subsystem cannot be
  changed, and the arrays it gathers are read-only, so writing `nucleus.charge[0]`
  and filling `subsystem.charges` in place now raises rather than quietly
  working; `subsystem.coordinates` cannot be set either. The replacement returns
  a new subsystem whose nuclei and gathered arrays cannot disagree, which is what
  the old pair of writes was working around. It carries over the fragments, the
  formal charge and the per-nucleus Lennard-Jones parameters, and refuses a
  charge a nucleus of its element cannot carry; the `bfd-pp` and ghost-atom tests
  are what exercise that. Only `self.quantum_subsystem` is rebound, and every
  reader here goes through that attribute, so nothing holds the old subsystem —
  the shallow copy the TDSCF attachment makes included, which resets its own
  model. The element check compares
  `Nucleus.atomic_number` with the Mole's elements, where it used to compare
  charges that are now deliberately different for an ECP or a ghost.
- **Nothing is left of a perturbed solve.** `_B_dot_x` and
  `embedding_gradient.perturbed_induced_dipoles` kept the length of PyFraME's
  cache and truncated it afterwards; the cache is gone and the three lines with
  it. Neither loop has a related earlier solve to pass as `starting_guess`.
- **The induced-dipole threshold is relative**, and `solve_induced_dipoles`
  starts from its previous solve. The `threshold` option keeps its name and its
  default of 1e-8, and means a relative accuracy now. For this module's test
  potential the default is the *tighter* of the two: the embedding energy of
  butadiene in water lands 1e-12 from its fully converged value where it used to
  land 1e-10 from it. Three tolerances in the suite moved as a consequence, each
  with what was measured:
  - `test_dft_matches_unrestricted_dft` compared the embedding energies of two
    separately converged SCFs. That energy is linear in the density, where the
    total energy is stationary in it, so the 4e-8 the two densities differ by
    reaches it as 2e-9. It now compares the two code paths on one density, which
    is what it is about.
  - `test_frozen_potential_still_follows_the_nuclei` compares a frozen model
    after a `reset` with a fresh one at the same density and geometry. The frozen
    model solves from the dipoles it had at the geometry before, so the two stop
    within the threshold of each other: 5e-12 at the default, 4e-16 at 1e-12.
  - `test_frozen_post_scf_is_the_plain_method_on_the_embedded_scf` compares two
    MP2 correlation energies that differ by 1.7e-10 however tightly the dipoles
    are converged — measured with PyFraME's older solvers too. The old default
    masked it to 1.6e-11 with a solver error of the opposite sign, and the
    tolerance had been set to that.
  A calculation that wants its energies not to depend on the solves before them,
  e.g. one differencing them over geometries, passes `start_from_previous=False`
  to `solve_induced_dipoles`, which this module does not do: an SCF wants the
  warm start, which is a third to a half fewer iterations.
- **The driver's Hessian-side methods are renamed.** PyFraME's
  `compute_electronic_field_gradients` and `compute_electronic_field_hessian` are
  `electronic_field_nuclear_derivatives` and `electronic_field_nuclear_hessian`.
  This module implements neither yet — it works its own Hessian out — but it
  will when it takes the Hessian from PyFraME, and its own
  `electronic_field_nuclear_gradients` is a different thing: every nucleus at
  once, for use here.
- **Nothing else in this module had to change.** It never wrote to an array a
  subsystem gave it, never used `Atom.induced_dipole`, `charge_to_element` or
  `element_to_charge`, and implements exactly the three driver methods an energy
  needs.

## [ ] Open: landing the two sides together

PyFraME's side landed on 2026-09-24: `da663f1` ("Make embedding data read-only
and fix the dipole solvers") and `9dc0e0f` ("Require mpi4py again and make
MDAnalysis optional") are on `master` and tagged `0.5a0.dev1`, which is not on
PyPI. This branch still has to follow. `origin/pyframe_interface` is still
`6a8dac5`, which writes `nucleus.charge[0]` and fails against `0.5a0.dev1`,
whose charges are read-only.

1. [ ] **Push this branch and watch the first CI run.** The run should show the
   embedding tests *running*, 83 passed. It is the first run against the new
   PyFraME off this machine. It is also the first with `mpi4py>=4.1` as a hard
   install requirement of PyFraME. The binary wheel should install on the
   GitHub runner without a system MPI, and nothing serial loads MPI, but only
   the CI log will confirm it. If the wheel does not install there, the fix
   belongs in `run_ci.sh`: install an MPI first, e.g. `apt-get install
   libopenmpi-dev`.
2. [x] **Fail clearly on a PyFraME that is too old.** Done 2026-09-23.
   `embedding.py` checks `QuantumSubsystem.with_nuclei`, the newest API it
   uses, at import. If it is missing, it raises an `ImportError` that names
   `0.5a0.dev1` and gives the pip line for the tag. The alternative was an
   `AttributeError` from deep inside the first model built. The test module
   now decides whether to skip by probing `pyframe.embedding` alone, and
   imports this module outside the `try`. So a PyFraME with no embedding
   subpackage, such as 0.4.0 under the qrunch pin, still skips all 83 tests,
   and one that has the subpackage but is too old fails collection with that
   message instead of skipping. A suite that skips when PyFraME is wrong is how
   the qrunch pin once emptied it without anyone noticing. `has_in_memory` and
   its two skips are gone; the check implies `to_subsystems`. Both paths were
   checked by hiding the subpackage and by deleting `with_nuclei` before
   collection.
3. [x] **Subclass `IntegralDriverTemplate`.** Done 2026-09-24.
   `EmbeddingIntegralDriver` subclasses it, so a rename of one of the three
   abstract methods stops the driver being made instead of failing mid-SCF.
   The gradient and Hessian drivers are not shaped like the template and do
   not subclass it: `electronic_field_nuclear_gradients` covers every nucleus
   at once, and `multipole_potential_hessian_integrals` has no counterpart
   there.

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
`0.5a0.dev1` does not get one. **Must be resolved before the PR is
merged.** While there, decide whether the copyright header should credit the
PyFraME authors as well as the PySCF developers — the file is an interface to
their work but carries only the standard PySCF header.

## [ ] Open: CI and packaging

`pyscf.embedding` imports `pyframe.embedding`, a subpackage that exists in no
PyFraME on PyPI — PyPI still tops out at 0.4.0, potential *generation* only,
re-checked 2026-09-24 with `pip index versions pyframe`. Until there is a
release carrying it, the dependency cannot be expressed the ordinary way, and
the two places that would express it are set up deliberately:

- **`setup.py` declares no `embedding` extra, and must not grow one back.** An
  extra pinning `pyframe>=0.5` cannot resolve against PyPI, so it would turn
  `pip install pyscf_forge[embedding]` into a hard failure while promising a
  dependency nobody can get. No extra is the honest state until the release.
- **`.github/workflows/run_ci.sh:21` installs PyFraME from git**, over HTTPS at
  the tag `0.5a0.dev1` (`9dc0e0f`). It was pinned there on 2026-09-24. Before
  that it followed `master`, and before 2026-09-19 `trajectory_embedding`,
  which is superseded and should not be pointed at again. A tag makes a CI run
  a statement about a fixed PyFraME rather than about the `master` of that
  day. Move the pin deliberately, re-running the suite locally against the new
  tag first. The tag is readable anonymously over HTTPS (verified 2026-09-24
  with `git ls-remote --tags`), so CI can install it.

There is no slack for an older PyFraME. Every model calls
`QuantumSubsystem.with_nuclei` when it is built, so `pyscf.embedding.embedding`
refuses at import any PyFraME before `0.5a0.dev1`. The test module fails
collection with that `ImportError` rather than skipping (task 2 of *Open:
landing the two sides together*). The bundled potential is still readable by
readers that ignore its `units` key, but nothing depends on that any more.

- **The install fails outright on Python 3.8, which `ci.yml` still builds.**
  PyFraME sets `python_requires = >=3.10` (`setup.cfg:41`), so pip refuses the
  git URL on 3.8 with a Requires-Python error rather than skipping it, and
  `run_ci.sh` runs under `set -e` — the whole Install step, and with it the 3.8
  job, goes red. The matrix is `["3.8", "3.10", "3.12"]` (`ci.yml:20`). Either
  3.8 leaves the matrix (it is long out of support) or the install needs a
  version guard; left alone here, since it is a change to shared CI rather than
  to this module.
- Installing an unreleased dependency from a git tag is a **merge consideration
  for pyscf-forge**, and the thing a reviewer will ask about first.

### On release, in order

1. Replace the tag install in `run_ci.sh` with the released package, and confirm
   the embedding tests **run** rather than skip — the log should show 83 passed,
   not 83 skipped. Change the `ImportError` message in `embedding.py` to name
   the release rather than the tag.
2. Decide whether `setup.py` should then carry an `embedding` extra after all,
   pinned at the version that actually ships the subpackage.
3. Re-run the suite against the *released* PyFraME rather than the local working
   copy, since the two may diverge. Re-check in particular
   `_check_vdw_parameters` (it probes four PyFraME properties by name), the vdw
   cross-check test (it encodes PyFraME's 6-12 convention and Lorentz-Berthelot
   combination as read from `engine/computation.cpp`), and `reset` and
   `_sync_quantum_subsystem` (they rely on `QuantumSubsystem.with_nuclei`
   carrying over everything but the positions and charges of the nuclei — fragments,
   formal charge, Lennard-Jones parameters, `kind` — and on
   `Nucleus.atomic_number` naming the element whatever charge it carries).
4. The CHANGELOG entry and the examples now say "PyFraME 0.5 or newer"; make
   that the version that actually ships the units block, `to_subsystems` and
   `with_nuclei`, and decide whether the module docstring should state it too.
   `0.5a0.dev1` is the first tag with all three, but it is a pre-release. By
   PEP 440 it sorts *before* 0.5, so "0.5 or newer" does not cover it, strictly
   read.

## [ ] Open: parity with `pyscf.solvent`

`solvent` attaches to SCF, CASCI/CASSCF, post-SCF and TDSCF, and so do we now.
What is still short of it is the correlated and excited-state gradients (9),
the induction Hessian (8) and the method-class shortcuts (10). The list is kept
in the order the work was done in, which was dependency order.

- [x] **1. `_B_dot_x`.** Done. `EmbeddingBase._B_dot_x` returns zeros — the
  answer for a model without induced dipoles, not a stub — and
  `PolarizableEmbedding` overrides it with the electronic-field-only response.
  It goes through PyFraME's `solve_perturbed_induced_dipoles`, which induces
  dipoles with the field it is handed and nothing else (no permanent multipole
  fields, no nuclear field) and returns them without storing them on the
  subsystem, so the dipoles belonging to the density the model was last run on
  survive the call.
  That is the same split `PolEmbed._B_dot_x` gets from CPPE's `elec_only`.
- [x] **2. `gen_response`.** Done, on `SCFWithEmbedding`, wrapping
  `undo_embedding().gen_response()` and adding `_B_dot_x(dm1)` — the spin blocks
  summed for the UHF-style response, which ROHF takes too. It is gated on
  `equilibrium_solvation`, which came with it rather than waiting for 3 below,
  since the gate is the half of that task this one needs; the flag is read
  inside `vind` so that a caller may set it around the call that *uses* the
  response function. This unlocks SOSCF, stability, polarizabilities and the
  Hessian. See *Deliberately different* for where it parts from
  `_attach_solvent.py:146-171`.
- [x] **3. `frozen`.** Done. `dm=` threads through `polarizable` /
  `electrostatic` / `embedding_for_scf` to `_for_scf`, which runs `kernel(dm)`
  once and sets `frozen = True`; from then on `EmbeddingBase.kernel` ignores the
  density it is handed and answers for the one it was frozen at. The guard sits
  in `kernel` rather than in `get_veff`, since that is where our caching lives.
  Two points where the behaviour had to be decided rather than copied, both
  written up under *Deliberately different*: what a reset does to a frozen
  potential, and what the gradient does with one. `equilibrium_solvation` landed
  with 2 above.
- [x] **4. `stability`.** Done, as `_attach_solvent.py:173-182` —
  `lib.temporary_env(self.with_embedding, equilibrium_solvation=not frozen)`
  around `super().stability()`, since the flag is off by default and the
  environment relaxes along with the density, so its response belongs in the
  Hessian of the solution being tested. Folded in with it: `gen_response` now
  requires `equilibrium_solvation and not frozen`, so the two flags cannot
  contradict each other — see *Deliberately different*. The internal kernels
  (`singlet=None`/`with_j=True`, `hermi=1`) carry the environment; the external
  ones (`singlet=False`, `with_j=False`, `hermi=2`) do not, which is what 2's
  gate is for.
- [x] **5. `_for_post_scf` + `PostSCFWithEmbedding`.** Done, with the `MP2`,
  `CISD` and `CCSD` hooks on `SCFWithEmbedding` and an `embedding_for_post_scf`
  entry that `polarizable` / `electrostatic` dispatch to. The question solvent
  answers at `_attach_solvent.py:517-650` is answered the same way: unfrozen,
  the environment is relaxed against the correlated density by a macro
  iteration to `EmbeddingBase.conv_tol` (new, with `max_cycle`, both as in
  `PolEmbed`); frozen, it stays at whatever density it was given and the
  correlated method never has to know about it. `CASCI`/`CASSCF` and TDSCF carry
  an `_scf` too and would otherwise be taken for correlated methods, so they are
  refused by name until 6 and 7 land. Gradients raise; see 9.
- [x] **6. `_for_casci` / `_for_casscf`.** Done, with `CASCIWithEmbedding` and
  `CASSCFWithEmbedding` over a shared `_CASWithEmbedding`, the `CASCI` /
  `CASSCF` hooks on `SCFWithEmbedding`, and `state_id` on `EmbeddingBase` for
  multi-root CASCI. As in `_attach_solvent.py:232-514`: a CAS method has no
  `get_veff` to hang the potential on, so it goes into `get_hcore`, and the
  `Tr(D v)` that puts into every energy the method computes is taken back out
  where each of them reports its total. CASCI cycles the environment around a
  converged solution; CASSCF rides its own macro iterations instead, through
  `update_casdm` and the `casci` override. One correction to solvent's version
  of the latter, under *Deliberately different*.
- [x] **7. `_for_tdscf` + `TDSCFWithEmbedding`.** Done, with the `TDA` / `TDHF`
  / `TDDFT` hooks, after `pol_embed.pe_for_tdscf` rather than
  `_attach_solvent.py:653-802` — the latter's non-equilibrium path swaps the
  dielectric constant for an optical one, which a discrete environment has no
  equivalent of. Non-equilibrium is the default and means the environment
  contributes nothing beyond the orbitals the ground state gave it, which is the
  right description of a vertical excitation; `equilibrium_solvation` turns the
  response on, on a *copy* of the ground state's model so the flag does not
  reach it. This is a linear-response environment, not a state-specific one.
  Three departures from solvent under *Deliberately different*; gradients raise,
  see 9.
- [~] **8. `Hessian`.** Done for `ElectrostaticEmbedding`, refused for
  `PolarizableEmbedding`. `embedding_hessian.py` follows the shape of
  `solvent/hessian/pcm.py`: `EmbeddingHess.kernel` runs the vacuum Hessian with
  `equilibrium_solvation` on, so the environment is in the orbital Hessian the
  CPHF equations use, and adds the second derivative at fixed density;
  `make_h1` adds the potential's derivative to the first-order Fock matrix. The
  new integrals are the second AO derivatives of the multipole potential, which
  at order L means L + 2 derivatives in all: charges batch over sites with a
  fakemol, dipoles and quadrupoles re-centre rinv per site and need the third-
  and fourth-derivative intors (`int1e_ipipiprinv`, `int1e_ipiprinvip`,
  `int1e_ipipipiprinv`, `int1e_ipipiprinvip`, `int1e_ipiprinvipip`). The
  classical terms — the nuclear electrostatic Hessian and the Lennard-Jones
  one, both block diagonal over the nuclei — come from PyFraME. **What is left
  is induction**, written up under *The induction Hessian* after this list.
- [~] **9. Gradients for whichever of the above land.** The **frozen SCF
  gradient** is done, for both models, once the energy it differentiates was
  made the one the SCF actually minimizes — see *A frozen potential is held,
  its energy is not* under *Deliberately different*. It brought
  `perturbed_induced_dipoles` with it, which is `mu^A = B^-1 F^A`: the response
  of the induced dipoles to a nuclear displacement, item 1 and 2 of *The
  induction Hessian* below, now written and tested. What remains is the
  **correlated** gradient, on `PostSCFWithEmbedding`, the two CAS classes and
  `TDSCFWithEmbedding` (for which `solvent` does have one, in
  `grad/ddcosmo_tdscf_grad.py`); all four raise. Relaxed, the environment is a
  functional of a density that is not the SCF's, so the gradient carries the
  `DM * V[d/dX DM] + V[DM] * d/dX DM` terms that `_attach_solvent` warns about
  and evaluates anyway. Note those terms vanish for `ElectrostaticEmbedding`,
  whose potential does not depend on the density at all — the obstacle there is
  only plumbing, since each method's gradient builds `hcore` from `._scf` and
  would miss `dv/dR`, and the Pulay terms are built from a generalized Fock
  matrix that has to be the one the wavefunction was optimized with.
- [R] **10. Register on the method classes.** `solvent` gets `mf.PE()` /
  `mf.PCM()` shortcuts; we are reachable only as `embedding.polarizable(mf, …)`.
  Being in forge, there is no `scf.hf.SCF` hook to hang it on. **Blocked** —
  revisit only if the module moves into main pyscf.

### [ ] The induction Hessian, the rest of 8

The missing piece of 8, stated as precisely as it can be without writing it.
With the environment's response matrix `B` fixed (the sites do not move) and
`F = F_el(D,R) + F_nuc(R) + F_mult`, the induction energy is `-1/2 F·B^-1·F`,
so at fixed density

    d2E_ind/dR_A dR_B = -mu·F^{AB} - F^{B}·mu^{A},    mu^{A} = B^-1 F^{A}

and `make_h1` needs, on top of the integral derivative it already has,
`-IDPI(mu^{A})` — the induced-dipole potential of the *perturbed* dipoles,
which is the term that makes `dv/dR` the mixed second derivative rather than
just the integrals moving. Three things are needed for that:

1. [x] `F^{A}`, the field gradient at the sites with respect to a nuclear
   coordinate:
   `EmbeddingGradientIntegralDriver.electronic_field_nuclear_gradients`,
   written for 9 and checked against finite differences of
   `compute_electronic_fields`. Watch the sign of the nuclear part —
   `compute_nuclear_field_gradients` differentiates the field with respect to
   the *sites*, which is the negative of the derivative with respect to a
   nuclear coordinate, as PyFraME now documents.
2. [x] `mu^{A} = solve_perturbed_induced_dipoles(external_fields=F^{A})`:
   `embedding_gradient.perturbed_induced_dipoles`, likewise checked against
   finite differences of the dipoles themselves.
3. `mu·F^{AB}`. This one is free: `mu·F_el` is a linear functional of the
   density whose operator is exactly the order-1 multipole operator with the
   induced dipoles in place of permanent ones, so its second derivative is
   `multipole_potential_hessian_integrals` at order 1 — already written and
   tested. Only the nuclear part, `mu·F_nuc^{AB}`, is new, and PyFraME has it:
   `compute_nuclear_field_hessian`, which is correct and tested upstream; a
   second derivative with respect to the sites is also the second derivative
   with respect to the nucleus.

**Building it on PyFraME's Hessian layer is possible, but not the recommended
route.** `compute_induction_energy_hessian` and
`compute_electronic_electrostatic_energy_hessian` are both tested upstream now
(the latter serially and over MPI, item 5 of *Settled upstream*), and the five
integral-driver methods that the two and the `compute_electronic_*_fock_gradient`
pair call are declared, concrete and raising, in
`integral_driver_template.py` under their current names:
`electronic_field_nuclear_derivatives`, `electronic_field_nuclear_hessian`,
`electronic_electrostatic_energy_hessian`,
`electronic_electrostatic_fock_gradient` and
`electronic_induction_fock_gradient`. (The first two were
`compute_electronic_field_gradients` / `compute_electronic_field_hessian`
until this round.) `electronic_field_nuclear_hessian` is `mu·F_el^{AB}`,
already contracted, so it is item 3's electronic half under another name.
Using that layer would still mean implementing those five on a driver here and
validating them through a consumer whose tests run on PyFraME's own mock
integrals, so a disagreement could come from either side. The route above,
through pieces each already checked here, stays the recommended one. The three
PyFraME Hessian routines this module *does* use —
`compute_electrostatic_nuclear_hessian`, `compute_repulsion_interactions_hessian`
and `compute_dispersion_interactions_hessian` — are tested there, and each is
checked here against finite differences of its own gradient as well.

## [x] Settled upstream, in PyFraME (2026-09-24)

Every item this section listed is done in PyFraME, along with a dozen more
found while doing them. The work is on `master` as `da663f1` and `9dc0e0f`,
tagged `0.5a0.dev1`. Its own `CHANGELOG.md` carries a line per user-visible
change. The per-task working notes (`PYSCF_UPSTREAM_PLAN.md`) were never
committed and no longer exist; the measurements that matter here are quoted
below and in the comments of the tests they moved.

What became of the eight items, in the order they were written:

1. *`solve_perturbed_induced_dipoles` handed back its cached dipoles.* The cache
   is gone entirely (6 below), and what a solve returns is the caller's own.
2. *The solvers converged to an absolute threshold.* The threshold is relative to
   the size of the dipoles now, so the response to a small field is as accurate
   as the response to a large one, which is what `_B_dot_x` asks for: a field
   scaled by 1e-8 used to come back 3e-3 off at the default threshold. What it
   cost here is under *Following PyFraME: read-only arrays…* above.
3. *The charges of the quantum subsystem could not be set.* `QuantumSubsystem.with_nuclei`
   returns a subsystem with the nuclei moved, given other charges, or both, and
   `Nucleus` keeps what it is apart from what it interacts with: `element` and
   `atomic_number` beside a `charge` that may be an effective one or zero.
   `_sync_quantum_subsystem` is one call to it.
4. *The driver template asked for more than an energy uses.* Three abstract
   methods, `electronic_fields`, `multipole_potential_integrals` and
   `induced_dipoles_potential_integrals`, which is what this module implements;
   the gradient and Hessian methods are concrete and raise
   `NotImplementedError`; three methods nothing called are deleted, as is
   `QuantumSubsystem.compute_electronic_field_gradients`. The template also says
   a driver need not subclass it; ours do not.
5. *`compute_electronic_electrostatic_energy_hessian` had no tests.* It has them,
   serial and over MPI, and takes the quantum subsystem in place of `nuc_list`,
   which its two branches had read differently. This module will call it when it
   takes its Hessian from PyFraME.
6. *The perturbed-dipole cache grew without bound.* Removed, with a
   `starting_guess` argument in its place. The two loops here that truncated it
   by hand no longer do.
7. *`compute_induction_energy_gradient` was a Python double loop.* One `einsum`,
   which reproduces the loop to 1.8e-16.
8. *Cut a release.* Partly: `0.5a0.dev1` is a git tag that CI pins to, but
   nothing newer than 0.4.0 is on PyPI, so everything under *On release* still
   waits. The qrunch pin (`pyframe~=0.4.0`) is unchanged and
   still removes `pyframe.embedding` if qrunch is installed into this env.

Found and fixed upstream while doing those, none of which this module had hit:
the induced dipoles a solve hands out are read-only, as is everything a subsystem
gathers, and particles and the records of a solve cannot be changed at all; the
divide-and-conquer solvers say which atom's polarizability they cannot invert
rather than `'Singular Matrix.'`; the solvers raise `TypeError` naming the
argument; a reader given both a flat list of nuclei and its fragments makes one
set of nuclei of them; `PolytensorStatics` is the singleton its docstring
claimed, which also took 18 s out of PyFraME's own suite; every split of a range
over MPI ranks goes through one helper, where 25 copies had been written by
hand; and `environment_energy` answers for the combination rule it is asked for,
where it used to return the first answer it ever gave.

One of those changes broke this module in a way its own suite caught and
PyFraME's did not: a relative threshold cannot judge dipoles that are exactly
zero, so a depolarized potential — `TestElectrostaticEmbedding` and
`test_charges_only_matches_qmmm` — ran to the iteration limit and raised. PyFraME
now treats zero dipoles as converged when they stop moving, and has a test for
it.

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
- **The potential need not be a file.** `PolEmbed` takes a CPPE potential file;
  we take a PyFraME JSON file or document, or the PyFraME system that built the
  potential, in memory (`_read_subsystems`).
- **The nuclei the environment sees are the `Mole`'s.** Coordinates and charges
  both come from the `Mole` (`_sync_quantum_subsystem`), so an ECP atom
  interacts through its effective charge, as in `pyscf.qmmm`; the potential's
  own nuclei only have to agree with the molecule in number and element.
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
  the package imports without PyFraME and a test module can skip rather than
  fail at import (the package docstring says so);
  `solvent/__init__.py` imports its models eagerly and defers only `pol_embed`.
- **`to_gpu` refuses rather than half-works.** Both wrappers raise and point at
  `.undo_embedding().to_gpu()`, where `WithSolventGrad.to_gpu` instead asserts
  the model is PCM. Refusing avoids handing back a GPU method whose potential
  has silently vanished.
- **Type-checked model selection.** `_embedding` rejects an
  `ElectrostaticEmbedding` object passed to `polarizable()` and vice versa,
  since both models read the same potential and the call would otherwise read
  as the other one. `solvent.PE` only asserts non-`None`.
- **Per-term energy report** from `energy_contributions()` in `_finalize`,
  against solvent's single `Solvent Energy` line.
- **`gen_response` asks whether the trial density is one the environment can
  see.** Shared by the SCF and TDSCF attachments, through
  `_add_embedding_response`. `_couples_to_the_density` reproduces the test the
  response function itself applies to its Coulomb term
  (`pyscf/scf/_response_functions.py`), and
  the environment — a functional of the total charge density — is added only
  where that term is. `_attach_solvent.gen_response` adds the solvent
  unconditionally, which is wrong for the RHF triplet kernel: its trial density
  is a spin density, the alpha and beta blocks cancel, the total charge density
  does not change, and the environment must not respond. RHF internal stability
  asks for exactly that kernel (`scf/stability.py:370`, `singlet=False`), so
  this is not a hypothetical. The `hermi == 2` half of the test is only an
  economy — `_B_dot_x` returns zero there anyway, the field integrals being
  symmetric — but it saves solving for dipoles known in advance to vanish.
- **A frozen potential still follows the nuclei.** `frozen` means the
  environment does not respond to the *density*; the geometry is a separate
  matter, and a potential held at the geometry it was built for would be wrong
  in a scanner — which calls `reset(mol)` on every step, ours clearing `e` and
  `v`. So `reset` keeps `_dm` while frozen and the next `kernel` rebuilds the
  potential for that density at the new geometry. `ddcosmo.reset` instead clears
  nothing, so a frozen solvent survives a scanner step carrying the previous
  geometry's potential. A frozen model that has no `_dm` at all raises rather
  than returning `(None, None)`.
- **A frozen potential is held, its energy is not.** `frozen` fixes the
  potential at one density; the energy `kernel` reports is still that of the
  density it is handed, which for a frozen model is the environment's own energy
  plus `Tr((D - D0) v)`. `_attach_solvent` reports `e(D0)` for every density
  instead, and that is not the functional the SCF makes stationary: its Fock
  matrix carries `v(D0)`, so the two differ by exactly the term above. The
  consequence is not cosmetic — the gradient of what solvent reports needs the
  response of the orbitals to a nuclear displacement, a CPHF solve, and on this
  system the missing term is 2 percent of the gradient. With the term put back,
  the frozen energy is stationary and its gradient is the ordinary one; both
  models now have it, and both agree with finite differences to 2e-7. It is the
  same correction the CAS classes apply for the same reason. Two things follow
  from the energy being stationary: it converges before the density does, so a
  test comparing densities has to lean on `conv_tol_grad` rather than
  `conv_tol`; and at `D = D0` nothing changes, so freezing at a converged
  density still reproduces it.
- **`frozen` and `equilibrium_solvation` cannot contradict each other.**
  `gen_response` adds the environment only when `equilibrium_solvation and not
  frozen`: a frozen potential is a fixed external term, so its response to any
  first-order density is zero however the other flag is set. `ddcosmo` says as
  much in a comment — *this attribute has no effects if .frozen is enabled* —
  but `_attach_solvent.gen_response` reads only `equilibrium_solvation`, and it
  is `stability` alone that reconciles them, so setting both by hand elsewhere
  gets a response out of a potential that cannot respond. Our `stability` still
  passes `not frozen`, now as a statement of intent rather than the thing doing
  the work.
- **Every pass re-converges the SCF, through the scanner.** Both branches of
  `PostSCFWithEmbedding.kernel` go through `_run_once`, which runs the bare
  method's scanner over the embedded SCF and copies the result back.
  `_attach_solvent` uses the scanner for its macro iteration but lets the frozen
  branch call `super().kernel()` on whatever orbitals the object was built with
  — and a potential frozen at some density other than the SCF's *changes the SCF
  problem*, so those orbitals are generally not the ones that go with it. The
  result is a silently wrong energy: freezing a correlated method at its own
  relaxed density and re-running it came out 5.8 mHa off the calculation it
  should have reproduced exactly. Running the scanner on both branches makes the
  orbitals always the ones belonging to the potential in force, at the cost of
  an SCF restart that begins from the converged density anyway. It also costs
  the `*args` of `kernel`: an initial guess cannot be handed through a scanner,
  so one is refused rather than ignored.
- **The post-SCF hooks import the package that defines them.** `MP2`, `CISD` and
  `CCSD` are attached to the SCF classes by `pyscf.mp`, `pyscf.ci` and
  `pyscf.cc` rather than defined on them, so `super().MP2()` fails with
  `'super' object has no attribute 'MP2'` unless something imported the package
  first — and our override is what makes the attribute appear to exist at all.
  Each hook imports its own package. `_attach_solvent` does not.
- **CASSCF polarizes the environment with the orbitals it was handed.**
  `mc1step.kernel` keeps the orbitals of the macro iteration in a local and puts
  them on the object only when it returns, so `self.mo_coeff` inside the `casci`
  override is still the starting guess. `_attach_solvent` builds the polarizing
  density there as `self.make_rdm1(ci=fcivec)` — the starting orbitals with the
  current CI vector, a density belonging to neither — where we pass `mo_coeff`
  through explicitly. It is not cosmetic: on CASSCF(4,4)/STO-3G for the bundled
  potential it moved the total energy by 8.3e-5 Eh, and left the converged
  object holding an environment 0.01 off in the density and 8e-5 Eh off in the
  energy it reports. With the orbitals right, the environment the method ends
  with is the one belonging to its own converged density to 1e-14.
- **The multi-root correction is one scalar, and that is not an oversight.**
  With several roots, every root's energy carries its own `Tr(D_r v)` from
  `get_hcore`, but only one density polarizes the environment. What has to be
  added to each root is the part of the embedding energy that does *not* come
  from the density it acts on, `e - Tr(D_state v)` — the same number for every
  root, with each root keeping its own `Tr(D_r v)` as the interaction of that
  state with the environment the chosen state made. `_attach_solvent` does the
  same; the reasoning is written down here because the shape of the expression
  invites the opposite conclusion.
- **Casida stays available.** `_attach_solvent` sets `CasidaTDDFT =
  NotImplemented`. The environment kernel is real, symmetric and frequency
  independent, so it enters A and B alike and leaves A - B untouched — which is
  the assumption `(A-B)^(1/2)(A+B)(A-B)^(1/2)` rests on. Checked rather than
  assumed: CasidaTDDFT and the full non-Hermitian solver agree to 1e-12 on
  PBE/STO-3G with the response switched on, and the test that says so is kept.
- **`dm` is refused on the TDSCF attachment.** Everywhere else it chooses the
  density the potential is held at. A TDSCF object has no potential of its own
  to hold — the ground state's reaches it through the orbitals of `._scf` — so
  all it could do is set an `e` and a `v` that nothing reads. `solvent` accepts
  it and it is inert there too. Refusing it also keeps the shallow copy above
  safe: nothing on this path calls `kernel`, so the MM subsystems shared with
  the ground state's model are never written to.
- **Every second derivative is checked on its own.** The Hessian is assembled
  from pieces that each have a first derivative the gradient tests already tie
  to the energy, so each is checked against finite differences of that
  derivative rather than only through the total: the multipole potential
  integrals at every order, the nuclear electrostatic and Lennard-Jones terms,
  and the Fock matrix derivative `make_h1` contributes. All agree to 1e-11 or
  better. The total Hessian is then checked against finite differences of the
  analytic gradient, where the tolerance is set by noise in the gradients being
  differenced — the *bare* RHF Hessian of the same molecule differs from its own
  finite differences by the same amount at the same step.
- **ROHF is named explicitly.** `_attach_solvent` decides how to fold the spin
  blocks with `isinstance(self, scf.uhf.UHF)`, but `ROHF` subclasses `RHF`, not
  `UHF`, while taking the UHF-style response — so solvent hands it a per-spin
  environment response instead of one built from the total density. We test for
  `(scf.uhf.UHF, scf.rohf.ROHF)` instead, matching how
  `_response_functions.py` assigns the two kernels.

### Gaps against `solvent`

Scope, not structure — everything above is a choice, everything missing is
unwritten. They are tracked as tasks under *Open: parity with `pyscf.solvent`*
above rather than listed here.
