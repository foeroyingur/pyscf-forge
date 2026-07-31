# `pyscf/embedding` — residual notes

**Branch:** `pyframe_interface` · **Last updated:** 2026-09-12
*(This is the trimmed remnant of a full code review of the PyFraME interface.
Every reviewable finding is closed; what follows is only what is still open,
plus a map of the module against `pyscf.solvent`. Rationales for individual
design choices live in comments next to the code they explain, not here. This
file is tracked and ships with the branch, so write it for a reviewer, not just
for yourself.)*

## Development environment

Work in the `kosmos` conda env, which has PyFraME `0.5a0` installed editable
from `~/PycharmProjects/pyframe` (`git@gitlab.com:pyframe-project/pyframe.git`).
No MPI setup is needed — as of PyFraME `071f551` the tests run in a plain shell.

```
~/miniconda3/envs/kosmos/bin/python -m pytest pyscf/embedding/test/test_embedding.py -q
```

**Careful: that runs the installed copy, not the working tree.** `pyscf_forge`
is installed into `site-packages` non-editably, i.e. as a copy of `pyscf/`
merged into the `pyscf` package, and since that is a regular package it wins
over the repository whatever `PYTHONPATH` says. Reinstall before trusting the
command above, or force the working-tree copy in — import
`pyscf/embedding/__init__.py` by path under the name `pyscf.embedding`, with
`submodule_search_locations` pointing at the repository, before pytest collects
anything.

**Last verified 2026-09-12:** 77 passed, against pyframe `675c070`, pyscf 2.14.0.

PyFraME `0.5a0` and `qrunch` cannot both be satisfied: `qrunch 1.5.0a8`
requires `pyframe~=0.4.0`, and 0.4.0 is the released version that has no
`pyframe.embedding`. Installing qrunch therefore silently takes this module's
dependency away and the whole suite skips. If the suite reports 0 run and 62
skipped, check `pip show pyframe` first.

`test_vdw_gradient_matches_explicit_lennard_jones` is mildly flaky: it asserts
that two SCF runs differing only in a term that never enters the Fock matrix
give the same density to 1e-9, and one run in three has been seen to land at
2e-9 instead. The runs are numerically independent, so this is the tolerance
being marginal rather than anything about the vdw terms; tighten it to 8 places,
or compare the converged energies instead, if it becomes a nuisance.

PyFraME no longer writes to stdout. It reports through one `pyframe` logger
carrying a `NullHandler` (`pyframe/log.py`), so nothing is emitted unless the
caller configures handlers, and an SCF + gradient run under
`redirect_stdout` captures the empty string. Its Lennard-Jones warnings go
through `warnings.warn` and so reach the PySCF logger. Nothing to do here; this
used to be a won't-fix.

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

## [ ] Open: CI and packaging

`pyscf.embedding` imports `pyframe.embedding`, a subpackage that exists in no
released PyFraME — PyPI still tops out at 0.4.0, potential *generation* only,
confirmed 2026-09-08. Until there is a release carrying it, the dependency
cannot be expressed the ordinary way, and the two places that would express it
are set up deliberately:

- **`setup.py` declares no `embedding` extra, and must not grow one back.** An
  extra pinning `pyframe>=0.5` cannot resolve against PyPI, so it would turn
  `pip install pyscf_forge[embedding]` into a hard failure while promising a
  dependency nobody can get. No extra is the honest state until the release.
- **`.github/workflows/run_ci.sh:21` installs PyFraME from git**, over HTTPS at
  branch `trajectory_embedding`. **Leave it there until the PR is opened.** The
  remote turns out to be readable anonymously over HTTPS — verified 2026-09-11
  with `git ls-remote https://gitlab.com/pyframe-project/pyframe.git` and no
  credentials — so CI can install it, and the earlier note here that the remote
  was private over SSH and therefore out of reach is simply wrong.

So CI does now exercise the module, rather than skipping it. Two things about
that install are worth keeping in view:

- A branch is a moving target. The suite is verified against the *local* working
  copy of PyFraME (`d4b9567`), which sits one unpushed commit ahead of the public
  tip of `trajectory_embedding` (`16638a9`), so CI installs slightly less than
  what was tested. That commit adds optional `charge`/`basis` on the quantum
  subsystem, flattening of `quantum_fragments` into nuclei, and exclusion-list
  validation; `butadiene_water.json` gives a flat `nuclei` list and neither
  `charge` nor `basis`, and this module names none of those attributes, so the
  older commit reads the bundled potential too. That is the current margin, not
  a guarantee — re-check it whenever either side moves.
- **The install fails outright on Python 3.8, which `ci.yml` still builds.**
  PyFraME sets `python_requires = >=3.10` (`setup.cfg:41`), so pip refuses the
  git URL on 3.8 with a Requires-Python error rather than skipping it, and
  `run_ci.sh` runs under `set -e` — the whole Install step, and with it the 3.8
  job, goes red. The matrix is `["3.8", "3.10", "3.12"]` (`ci.yml:20`). Either
  3.8 leaves the matrix (it is long out of support) or the install needs a
  version guard; left alone here, since it is a change to shared CI rather than
  to this module.
- Installing an unreleased dependency from a branch is a **merge consideration
  for pyscf-forge**, and the thing a reviewer will ask about first.

### On release, in order

1. Replace the git install in `run_ci.sh` with the released package, and confirm
   the embedding tests **run** rather than skip — the log should show 77 passed,
   not 77 skipped.
2. Decide whether `setup.py` should then carry an `embedding` extra after all,
   pinned at the version that actually ships the subpackage.
3. Re-run the suite against the *released* PyFraME rather than the local working
   copy, since the two may diverge. Re-check in particular
   `_check_vdw_parameters` (it probes four PyFraME properties by name), the vdw
   cross-check test (it encodes PyFraME's 6-12 convention and Lorentz-Berthelot
   combination as read from `engine/computation.cpp`), and `reset` (it relies on
   `QuantumSubsystem.coordinates` being the only geometry-dependent cache).
4. Decide whether the module docstring should state a minimum PyFraME version,
   and add the requirement to the CHANGELOG entry.

## [ ] Open: parity with `pyscf.solvent`

`solvent` attaches to SCF, CASCI/CASSCF, post-SCF and TDSCF; we attach to SCF
only (`_attach_embedding._for_scf`, and `make_grad_object` raises
`NotImplementedError` for anything else). None of this is blocked by the
embedding design — the model already computes everything the missing pieces
need. Roughly in dependency order; each is independently shippable.

- [x] **1. `_B_dot_x`.** Done. `EmbeddingBase._B_dot_x` returns zeros — the
  answer for a model without induced dipoles, not a stub — and
  `PolarizableEmbedding` overrides it with the electronic-field-only response.
  It goes through PyFraME's `solve_perturbed_induced_dipoles`, which induces
  dipoles with the field it is handed and nothing else (no permanent multipole
  fields, no nuclear field) and returns them rather than storing them, so the
  dipoles belonging to the density the model was last run on survive the call.
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
  induction Hessian* above, now written and tested. What remains is the
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
   `compute_nuclear_field_gradients` returns the *negative* of the derivative of
   `compute_nuclear_fields`, PyFraME's induction gradient consuming it with that
   sign built in (item 2 of *Open: upstream, in PyFraME*).
2. [x] `mu^{A} = solve_perturbed_induced_dipoles(external_fields=F^{A})`:
   `embedding_gradient.perturbed_induced_dipoles`, likewise checked against
   finite differences of the dipoles themselves.
3. `mu·F^{AB}`. This one is free: `mu·F_el` is a linear functional of the
   density whose operator is exactly the order-1 multipole operator with the
   induced dipoles in place of permanent ones, so its second derivative is
   `multipole_potential_hessian_integrals` at order 1 — already written and
   tested. Only the nuclear part, `mu·F_nuc^{AB}`, is new, and PyFraME has it:
   `compute_nuclear_field_hessian`, which has no test upstream but was checked
   here against finite differences and is correct — with the *natural* sign,
   unlike the gradients (item 2 of *Open: upstream, in PyFraME*).

**Do not build this on PyFraME's Hessian layer.**
`compute_induction_energy_hessian`,
`compute_electronic_electrostatic_energy_hessian` and the five integral-driver
methods they call (`compute_electronic_field_gradients`,
`compute_electronic_field_hessian`, `electronic_electrostatic_energy_hessian`,
`electronic_electrostatic_fock_gradient`, `electronic_induction_fock_gradient`)
have no callers, no tests, and no entry in `integral_driver_template.py`, and
`QuantumSubsystem.compute_nuclear_field_hessian`, which the first of them needs,
has its test commented out (`subsystem_test.py:114`). Using them would mean
validating our integrals and their untested consumers at once, with no way to
tell which of the two a disagreement came from. The three PyFraME Hessian
routines this module *does* use — `compute_electrostatic_nuclear_hessian`,
`compute_repulsion_interactions_hessian` and
`compute_dispersion_interactions_hessian` — are the three that are tested
there, and each is checked here against finite differences of its own gradient
as well. See items 4 and 7 of *Open: upstream, in PyFraME* for what would
have to change for that advice to be withdrawn.

## [ ] Open: upstream, in PyFraME

Things this interface ran into that are better fixed on the PyFraME side than
worked around here. Written for someone with the PyFraME tree open; each says
what we do about it in the meantime. Line numbers are as of `675c070`.

**Before tasks 8-10.** Checked 2026-09-14, each item re-verified against the
tree rather than carried over: *none of these has to land upstream first.*
Task 10 does not touch PyFraME at all. Everything the rest of 8 and 9 need from
PyFraME exists — the dipole solvers, the nuclear field gradients and, since
item 2 was checked, the nuclear field Hessian — and our call sites stay clear
of the defects below by construction. What the check did settle:

- Item 2 needed *checking* before the induction Hessian could use
  `compute_nuclear_field_hessian`, and that is done: the values are right, the
  documented shape is not, and its sign is the natural one where
  `compute_nuclear_field_gradients` is flipped. Usable, with that written down.
- **Recommended first, not required:** settling that sign convention (item 2).
  The induction Hessian is the first code to consume both routines, and
  writing it against the mismatch means encoding a second compensating sign
  that a later upstream cleanup would break in two places. A test pins each.
- **Fix upstream soon, independent of 8-10:** item 1, a silent wrong-answer
  path for any host that reuses a field buffer. The fix is a copy in each of
  the two solvers — and it has to be the copy, not merely stopping the
  mutation, which was tested and does not remove the wrong answer.
- Item 3 is a real defect whose effect on our results was measured and found
  negligible, so it does not gate anything either.

### Correctness and API traps

1. **Both dipole solvers keep the caller's field array rather than a copy,
   and a host that reuses a buffer gets the previous field's dipoles back.**
   `solve_induced_dipoles` aliases (`subsystem.py:882-884`,
   `static_fields = external_fields`, then `static_fields +=
   self.multipole_fields`), so the caller's array comes back with the
   permanent multipole fields added and that same object is stored as
   `InducedDipoles.external_fields` — contrary to the comment directly above
   it. The residue-norm shortcut at `subsystem.py:891-896` then compares the
   next call's field against the stored one and returns early when they are
   equal. For a host that preallocates one field buffer and refills it each
   iteration, the natural SCF-loop pattern, the stored record *is* the buffer,
   so after the refill the comparison is the buffer against itself, the
   difference is zero, and the solver returns the dipoles of the previous
   field without solving: measured here at 30 percent of their size, with no
   warning. `solve_perturbed_induced_dipoles` does the same with its cache
   (`subsystem.py:1030-1034`), and on the return side hands back the cached
   dipoles themselves, so a caller that modifies them corrupts the cache.
   **Fix: store a copy of the field.** Not mutating is the obvious change and
   is worth making, but on its own it fixes nothing that matters — patched in
   memory and run against the reused buffer, `static_fields = external_fields
   + self.multipole_fields` still returns the stale dipoles, because the record
   still *is* the buffer. It is `external_fields = external_fields.copy()` at
   the top that removes the wrong answer; the same goes for the perturbed
   solver's record, and for what it returns. *(An earlier version of this item
   recommended only the non-mutating line, and said the shortcut never fires;
   it does — both sides of its comparison carry the multipole fields — and
   that is precisely what makes the reused buffer dangerous.)* All four of
   our call sites pass a fresh array or a slice that is never refilled
   (`embedding.py` `_density_dependent_contributions` and `_B_dot_x`,
   `embedding_gradient.py` `_energy_gradient` and
   `perturbed_induced_dipoles`); new ones must too.
2. **The nuclear field derivatives disagree on their sign, and one of them on
   its shape.** Measured against second differences of `compute_nuclear_fields`
   itself: `compute_nuclear_field_gradients` returns the *negative* of the
   first derivative, `compute_nuclear_field_hessian` returns the second
   derivative with its *natural* sign (to 1e-9). Neither says so. The gradient
   docstring (`subsystem.py:383`) says only "electric field gradient from the
   nuclei", and `compute_induction_energy_gradient` consumes it with the flip
   built in ("sign has been changed from other branch"), so the convention is
   invisible until something else uses it — it cost a debugging cycle here,
   the perturbed induced dipoles coming out fifty times too large with the sign
   taken at face value. The Hessian carries `# FIXME doc string is wrong`
   (`subsystem.py:406`) and rightly: it documents
   `(nuclei, 3, 3, sites, 3)`, twice, "or maybe ... depending on
   implementation", where it returns `(nuclei, sites, 10)` in the packing
   `compute_induction_energy_hessian` unpacks. Its test is commented out and
   was never more than a print (`subsystem_test.py:114-150`). It is *correct*,
   though, which is what matters for the induction Hessian: checked here
   against finite differences of the gradients to 5e-11. Fix: pick one sign for
   both, update the one internal consumer of the gradients, fix the Hessian
   docstring, and turn the commented test into an assertion.
3. **The dipole solvers converge to an absolute threshold, so the response to
   a small field is inaccurate in proportion.** `induced_dipoles_jacobi` stops
   when the change in the dipoles between iterations is below `threshold`
   (`solvers.py:129`), unscaled. The dipoles are linear in the field, so for a
   field of size `s` the relative error is about `threshold / s`: `_B_dot_x`
   applied to a trial density scaled by 1e-8 is off by 7e-3 at the default
   threshold of 1e-8, and by 4e-7 at 1e-12. Whether that matters depends on the
   solver calling it, and it was measured for the three that do here. Davidson
   (`stability`) normalizes its trial vectors, which stay above 0.5, so it is
   untouched. SOSCF applies the Hessian to its step, which reaches 2.6e-11, so
   the response is noise at the end — it converges anyway, the gradient being
   exact. CPHF (`cphf.solve`, which the Hessian and any correlated gradient
   use) goes through `lib.krylov`, whose basis is "orthogonal but not
   normalized" (`linalg_helper.py:1281`) and shrinks to 2.6e-7; the static
   polarizability it produces moves by 9e-9 relative between thresholds of 1e-8
   and 1e-12, so this does not gate 8 or 9. Fix: a relative criterion, or
   document the threshold as absolute. The same absolute `1e-6` decides reuse
   of an old solve as a starting guess (`subsystem.py:1040`), which for small
   fields picks an unrelated one.

### The integral-driver contract

4. **`IntegralDriverTemplate` stops at the gradient.** Every method the Hessian
   layer calls is missing from it: `compute_electronic_field_gradients`,
   `compute_electronic_field_hessian`,
   `electronic_electrostatic_energy_hessian`,
   `electronic_electrostatic_fock_gradient` and
   `electronic_induction_fock_gradient`. A host program cannot implement them
   from the template, and there is nothing to check an implementation against.
5. **Two names for neighbouring quantities, one documented.** The template has
   `electronic_field_gradients(coordinates, density_matrix)` returning
   `(nuclei, sites, 6)`, while the induction Hessian calls
   `compute_electronic_field_gradients(coordinates, density_matrix, i)` — a
   different signature under a nearly identical name. The `6` of the documented
   one also implies a symmetric 3x3, which the nuclear derivative of the
   *electronic* field is not; that packing suits the nuclear field, whose
   derivative is symmetric, and it is `compute_nuclear_field_gradients` that
   uses it. Our own driver method is called
   `electronic_field_nuclear_gradients` to stay out of the way of both.
6. **The template asks for more than the model uses.**
   `electronic_multipole_interaction_energy` and
   `electronic_potential_integrals` are declared abstract, but nothing in
   `PolarizableEmbedding` calls either — this module implements neither and
   works. A host reading the template implements two routines it does not need.

### Untested code

7. **The electronic Hessian layer has no callers and no tests.**
   `compute_electronic_electrostatic_energy_hessian` and
   `compute_induction_energy_hessian` are referenced only from their own module
   docstrings. Their classical siblings —
   `compute_electrostatic_nuclear_hessian`,
   `compute_repulsion_interactions_hessian`,
   `compute_dispersion_interactions_hessian` — are tested, and those are the
   three this module uses. Until the electronic ones are exercised against
   something, a host that builds on them is validating two unknowns at once,
   which is why `embedding_hessian.py` derives its own instead.

### Performance

8. **`solve_perturbed_induced_dipoles` grows a list without bound.** Every call
   appends to `self.perturbed_induced_dipoles` and scans the whole list first
   (`subsystem.py:1029-1045`), to reuse an earlier solve as a starting guess.
   Trial densities are never repeated, so a response calculation pays the scan
   and the memory once per trial vector and reuses nothing. `_B_dot_x` and
   `perturbed_induced_dipoles` both truncate the list back to what it was. A cap
   on the cache, or a flag to skip it, would do.
9. **`compute_induction_energy_gradient` is a Python triple loop**
   (`induction_interactions.py:92-124`), marked
   `# TODO move into c++ layer?`. It is `O(nuclei x sites)` scalar iterations
   where an `einsum` over the packed field-gradient tensor would do; harmless on
   nine sites, not on a realistic potential.

### Packaging

10. **Cut a release that carries `pyframe.embedding`.** Everything in *Open: CI
    and packaging* above waits on it: CI installs from a branch, `setup.py`
    declares no extra, and a consumer pinning `pyframe~=0.4.0` silently removes
    the subpackage this module needs — which is how the suite came to skip
    entirely mid-session.

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
