# VLA shape variants — mental model

Reference diagrams for the three VLA shapes that show up when
discussing the "Missing" items 1 and 2 under the table-read list
in `CLAUDE.md` (currently both deferred until a real user need
shows up).

The "today" diagram is what rustfits supports now; the other two
are what items #3 (`repeat > 1`) and #4 (`TDIMn` on VLA) would
unlock.  Drawn 2026-05-25 during a design discussion that ended
in "defer both."

## Today — `1Pt(maxlen)` (one descriptor per row, 1-D cell)

```
                  main rows                       heap
              ┌──────────────┐         ┌──────────────────────┐
   row 0   →  │ (nlen=3, off)│ ──────→ │ a  b  c              │
   row 1   →  │ (nlen=5, off)│ ──────→ │ x  y  z  w  v        │
   row 2   →  │ (nlen=1, off)│ ──────→ │ p                    │
              └──────────────┘         └──────────────────────┘

  numpy:  Object dtype, shape (nrows,)
          arr[0] = np.array([a, b, c])             # shape (3,)
          arr[1] = np.array([x, y, z, w, v])       # shape (5,)
```

## #3 — `3Pt(maxlen)` (THREE descriptors per row → 3 sibling VLAs)

```
                main rows (per-row × N=3)              heap
              ┌─────────────────────────────┐    ┌────────────┐
   row 0   →  │ (n=3,off) (n=2,off) (n=4,off)│ →→ │ abc de fghi │
   row 1   →  │ (n=2,off) (n=4,off) (n=0,off)│ →→ │ xy zwvu   ∅ │
   row 2   →  │ (n=1,off) (n=0,off) (n=2,off)│ →→ │ p ∅ qr      │
              └─────────────────────────────┘    └────────────┘

  numpy candidate A:  Object dtype, shape (nrows, N)
          arr[0, 0] = np.array([a, b, c])
          arr[0, 1] = np.array([d, e])
          arr[0, 2] = np.array([f, g, h, i])

  numpy candidate B:  Object dtype shape (nrows,), each is Object[N]
          arr[0] = np.array([
              np.array([a,b,c]), np.array([d,e]), np.array([f,g,h,i])
          ], dtype=object)

  (decision before coding — A reads more naturally for fancy
  indexing; numpy axis order would put N first per row)
```

## #4 — `1Pt(maxlen) + TDIMn='(8, 0)'` (VLA-of-images, ONE variable axis)

A 0 in TDIM marks the axis whose length is computed per-row from
`nelements / product_of_rest`.  FITS only allows exactly one
zero — see the limitation note below.

```
                 main rows                              heap
              ┌──────────────┐               ┌─────────────────────┐
   row 0   →  │ (nlen=24, off)│ ──→ 24 elts → │ reshape to (3, 8)  │ ← 24/8 = 3 rows
   row 1   →  │ (nlen=40, off)│ ──→ 40 elts → │ reshape to (5, 8)  │ ← 40/8 = 5 rows
   row 2   →  │ (nlen=16, off)│ ──→ 16 elts → │ reshape to (2, 8)  │ ← 16/8 = 2 rows
              └──────────────┘               └─────────────────────┘

  numpy:  Object dtype, shape (nrows,)
          arr[0]  # shape (3, 8)
          arr[1]  # shape (5, 8)
          arr[2]  # shape (2, 8)

         (today: arr[0] would be shape (24,) — a flat 1-D array)
```

## Why fully-variable `(n, m)` per row isn't possible

The VLA descriptor only stores ONE value — `nelements` — alongside
the heap offset.  So you have ONE number to recover the shape
from.  With one variable axis you can solve for it
(`var_axis = nelements / product_of_fixed_axes`); with two
variable axes you'd have one equation and two unknowns.

cfitsio and astropy both accept `TDIM = '(8, 0)'` or `'(0, 8)'`
— exactly one zero.  Two zeros is undefined.

**Workarounds for fully-variable `(n, m)`:**

1. **Companion shape column**: keep the data as a flat 1-D VLA
   plus an additional fixed `i4[2]` (or `i8[2]`) column storing
   `(n, m)` per row.  Reshape after read using the companion
   column.  Two reads, both cheap.  This is what most code that
   "fakes" 2-D VLA-of-images is already doing.

2. **Pack shape into the data itself**: a non-spec convention
   where the first 2 elements of each cell are `(n, m)` and the
   rest is the flattened data.  Cheaper storage but breaks tools
   that don't know the convention.

3. **Object-of-Object at the numpy level**: no FITS analogue;
   wouldn't round-trip.

## Mental model summary

- **#3 widens the row** — multiple sibling VLAs per row, each
  with its own descriptor and heap cell.  Useful when one row
  logically carries several independent variable-length arrays.
- **#4 reshapes the cell** — one descriptor per row, but the
  cell isn't flat 1-D; one TDIM axis = 0 marks the variable
  one and the rest is fixed.  Useful for image-shaped data with
  exactly one variable axis.
- The two are orthogonal — `3PE(maxlen) + TDIM='(8,0)'` would
  give 3 reshaped 2-D images per row.  Neither alone nor together
  gives fully-variable `(n, m)`.
