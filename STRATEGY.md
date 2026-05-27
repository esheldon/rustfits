# Distribution strategy — rustfits and the fitsio handoff

This document captures the longer-term plan for how rustfits relates
to [fitsio](https://github.com/esheldon/fitsio).  Both projects are
maintained by Erin Sheldon; the question this doc answers is "how
do existing fitsio users get to rustfits without breaking their
pipelines."

See `CLAUDE.md` for code-architecture decisions and conventions.
This doc is project-management context: distribution, brand, and
the migration path.

## Constraints driving the plan

- fitsio is integral to important pipelines (e.g. DESI).  Any
  change that subtly breaks DESI-grade code is unacceptable —
  these projects have multi-year baked-in usage of fitsio's
  exact behavior.
- rustfits has a modern API with byte-exact interop, full
  compression support, and significantly better performance on
  some workloads.  It's where new development happens.
- The project currently has one maintainer.  That's likely to
  change as rustfits matures; the plan should be sustainable
  for one person today AND friendly to new contributors as
  they show up.  The detailed `CLAUDE.md` in particular means
  any contributor who uses Claude Code can come up to speed on
  the architecture (visibility rules, axis-order conventions,
  protected-key policy, taint discipline, etc.) and start
  submitting useful PRs without a long onboarding ramp.  That
  lowers the bar substantially compared to a typical Rust+PyO3
  codebase.

## The plan — "Option D + A" (freeze + shim)

Two parts, working together:

**Part D — fitsio enters long-term maintenance.**

- One or two final fitsio releases clean up known issues.  After
  that, fitsio is *frozen*: bug fixes and Python compatibility
  patches only.  No new features.
- fitsio remains installable indefinitely.  Existing pipelines
  pin against fitsio and stay working forever.  The behavior
  contract that DESI-grade users depend on is preserved by NOT
  changing it.
- Public announcement (in both repos' READMEs, plus the usual
  community channels) frames fitsio as stable / maintenance,
  rustfits as the modern successor.

**Part A — `rustfits.fitsio` migration shim.**

- A submodule of rustfits exposing a fitsio-shaped surface
  (`from rustfits import fitsio; with fitsio.FITS(path) as f:
  ...`).  Implementation calls into rustfits internals.
- Scope: narrow.  Cover what fitsio users actually do — open,
  read by ext, write a record array, headers, compression
  basics.  Don't try to mirror every corner of fitsio's API.
- The shim is a migration aid, not a long-term commitment.  We
  watch usage and may deprecate after a few years once the
  community has migrated.

**Headers are the hard part.**  fitsio returns dict-like
records (value via `[]`, comment via `get_comment()`, iteration
yields `(name, value, comment)` tuples).  rustfits returns
`FITSHeader` with `__getitem__` returning the value and
`.comment_of()` for comments.  The shim will need a thin
facade around `FITSHeader` that exposes the fitsio shape; full
behavioral parity isn't a goal.  Differences are documented in
the migration guide.

## Why not the alternatives

- **Merge rustfits into the fitsio repo (Option C).**  Locks in
  one of two bad outcomes: keep fitsio's 15-year-old API
  forever (constrains rustfits's clean design), or break the
  API in fitsio v2 (DESI-grade users break).  Also forces a
  build-system reconciliation between fitsio's setuptools+C
  extension and rustfits's maturin+PyO3 — ugly either way.
- **fitsio v2 vendors rustfits, with old code as fitsio_v1
  (Option B).**  Two implementations co-existing in one repo,
  permanent fitsio_v1 fallback as compatibility tax.  The
  subtle behavior diffs become *the fitsio maintainer's*
  problem when a pipeline breaks.  With D, the pipeline can
  just stay on fitsio 1.x and the issue doesn't exist.
- **Standalone rustfits + spread the word, no shim
  (D without A).**  Workable but leaves active fitsio users
  with a "rewrite or stay on fitsio forever" choice, which is
  hostile to the users who'd most benefit from migrating.

## The commitment question

The one thing this plan demands is a clear public commitment on
how long fitsio is maintained.  "Maintained forever" is
appealing but only honest if we actually intend to keep
patching it.  Two reasonable framings:

- **"Frozen indefinitely; security/compat patches only."**
  Strongest guarantee.  Requires us to keep at least patching
  Python deprecations for the foreseeable future.
- **"Frozen now; sunset in N years."**  Gives existing
  pipelines a window to migrate.  DESI-grade users will plan
  for the migration if we tell them well in advance.

The right framing depends on willingness to keep building
fitsio against future Python versions.  Decide before the
announcement — backtracking later (saying "we said forever but
actually no") damages trust in a way that's hard to recover.

## Concrete TODO

- [ ] Pick fitsio's long-term-stable version.  Ship 1-2 cleanup
      releases.
- [ ] Write the public announcement (READMEs of both repos,
      community channels).
- [ ] Build `rustfits.fitsio` shim.  Start narrow; expand only
      as users prompt.
- [ ] Write the migration doc (lives at
      `docs/tutorial/migration.rst`).  Document behavior diffs
      honestly — headers most importantly.
- [ ] Decide on the maintenance-window commitment ("forever" vs
      "N years").  Publish it.

When this plan changes — or when stages 1-5 complete — update
this doc.  CLAUDE.md should keep a short pointer to STRATEGY.md
near the top so contributors and future Claude sessions can
find this context.
