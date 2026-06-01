// AsciiTableHDU column metadata: the `AsciiColumn` struct, TFORM/TBCOL
// parsing, and the TSCAL/TZERO scaling classifier.
//
// ASCII tables (XTENSION='TABLE') are MUCH simpler than BINTABLE:
//   - no variable-length / heap (PCOUNT=0 always)
//   - no subarray TDIM
//   - no bit-packed X columns
//   - no complex C/M columns
//   - no L (boolean) columns
//   - just text fields with FORTRAN-style formats
//
// Supported TFORM letters: A (string), I (integer), F (fixed-point),
// E (exponential), D (double-precision exponential).
//
// Per-column positioning: TBCOLn is the 1-based starting byte of the
// nth column within each row.  Columns MAY have gaps (filler bytes
// between them) but MUST NOT overlap (rejected at parse time as
// almost-certainly-corrupt; no sane writer produces overlap).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::{parse_keyword, parse_keyword_float, parse_string_keyword};

// One ASCII-table column.  `byte_offset` is 0-based (TBCOLn - 1);
// `byte_width` is the format's total field width.
//
// For F/E/D formats: `decimals` is the digit count after the decimal
// point (the `d` in `Fw.d`).  Unused for A and I (kept None).
//
// TNULL on ASCII tables is a STRING sentinel, not an integer pattern.
// When the trimmed field text matches the trimmed TNULL string verbatim,
// the cell is considered null.  Stored as a String here, trimmed on
// parse so the per-cell mask check is a single &[u8] comparison.
#[derive(Debug, Clone)]
pub(crate) struct AsciiColumn {
    pub(crate) name: String,
    pub(crate) tform_letter: char,
    pub(crate) byte_offset: usize,
    pub(crate) byte_width: usize,
    // Decimal places from Fw.d / Ew.d / Dw.d.  Drives the f4-vs-f8
    // output-dtype decision on read (see `F_E_F4_MAX_DECIMALS` in
    // read.rs) and output formatting on write.
    pub(crate) decimals: Option<usize>,
    pub(crate) tscal: f64,
    pub(crate) tzero: f64,
    // TNULLn string sentinel (ASCII TNULL is a string compared verbatim
    // against the trimmed cell text, unlike BINTABLE's integer TNULL).
    // Used by `mask_null=True` on the read side.
    pub(crate) tnull: Option<String>,
    pub(crate) tunit: Option<String>,
}

// TSCAL/TZERO classification — parallels hdu_table's ScalingKind but
// scoped to the ASCII letters.  ASCII tables only carry numeric I/F/E/D
// columns (no L/B/X/C/M), so the variants reduce to:
//
//   None         — TSCAL=1, TZERO=0, OR letter is A (text), OR
//                  scale=False
//   UnsignedTrick — I column with TZERO equal to a sign-bias for some
//                   integer width.  Always reads as u8 with the
//                   matching bias subtracted, since rustfits maps
//                   Iw → i8 always (per the design decision); the
//                   unsigned trick on ASCII tables turns Iw → u8.
//   General      — anything else; promotes to f8.
//
// Implementation note: TZERO sign-bias values for the unsigned trick
// on Iw are the same as for BINTABLE B/I/J/K (128, 32768, 2^31,
// 2^63) — but since rustfits reads all integers as i8 internally,
// we accept any of those (the only thing that matters is that the
// reverse transform is bias subtraction, and the result fits in u8).
// We use 2^63 as the bias for the i8 case to mirror the BINTABLE K
// trick.  TODO: revisit when concrete test files surface — the
// FITS standard doesn't define the unsigned trick for ASCII tables,
// so this is a rustfits convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AsciiScalingKind {
    None,
    UnsignedTrick,
    General,
}

pub(crate) fn ascii_scaling_kind(col: &AsciiColumn) -> AsciiScalingKind {
    if col.tscal == 1.0 && col.tzero == 0.0 {
        return AsciiScalingKind::None;
    }
    // A columns ignore scaling (text doesn't scale meaningfully).
    if col.tform_letter == 'A' {
        return AsciiScalingKind::None;
    }
    // Unsigned-int trick on I: TSCAL=1, TZERO = 2^63 (matches the K
    // sign-bias because we always read I as i8).  Other biases land
    // in the General bucket; conversion is well-defined either way.
    if col.tform_letter == 'I'
        && col.tscal == 1.0
        && col.tzero == 9223372036854775808.0
    {
        return AsciiScalingKind::UnsignedTrick;
    }
    AsciiScalingKind::General
}

// The numpy dtype string the column reads into after scaling.  Only
// valid when kind != None.
pub(crate) fn ascii_scaled_output_dtype(
    letter: char, kind: AsciiScalingKind,
) -> &'static str {
    match kind {
        AsciiScalingKind::UnsignedTrick => match letter {
            'I' => "u8",
            _ => unreachable!(
                "unsigned-trick scaling on unexpected ASCII letter '{}'",
                letter
            ),
        },
        AsciiScalingKind::General => "f8",
        AsciiScalingKind::None => unreachable!(
            "ascii_scaled_output_dtype called with None"
        ),
    }
}

// Parsed pieces of an ASCII-table TFORM string.  `Aw` / `Iw` have
// `decimals=None`; `Fw.d` / `Ew.d` / `Dw.d` carry the trailing digit
// count.
struct AsciiTformInfo {
    letter: char,
    width: usize,
    decimals: Option<usize>,
}

// Parse one ASCII-table TFORM.  Examples:
//   "A10"     -> ('A', 10, None)
//   "I5"      -> ('I', 5, None)
//   "F8.2"    -> ('F', 8, Some(2))
//   "E12.5"   -> ('E', 12, Some(5))
//   "D25.17"  -> ('D', 25, Some(17))
fn parse_ascii_tform(tform: &str, col_index: usize) -> PyResult<AsciiTformInfo> {
    let trimmed = tform.trim();
    let mut chars = trimmed.chars();
    let letter = chars.next().ok_or_else(|| {
        PyValueError::new_err(format!(
            "column {}: TFORM='{}' is empty", col_index, tform
        ))
    })?;
    let rest: String = chars.collect();
    let letter = letter.to_ascii_uppercase();
    if !matches!(letter, 'A' | 'I' | 'F' | 'E' | 'D') {
        return Err(PyValueError::new_err(format!(
            "column {}: TFORM='{}' uses unsupported ASCII-table type \
             letter '{}'; expected one of A, I, F, E, D",
            col_index, tform, letter,
        )));
    }
    // Width is the digits before any '.'.  For A/I there must be no
    // '.'; for F/E/D the '.' is required.
    let (width_str, decimals_str) = match rest.find('.') {
        Some(i) => (&rest[..i], Some(&rest[i + 1..])),
        None => (rest.as_str(), None),
    };
    let width: usize = width_str.trim().parse().map_err(|_| {
        PyValueError::new_err(format!(
            "column {}: TFORM='{}' width '{}' is not a positive integer",
            col_index, tform, width_str,
        ))
    })?;
    if width == 0 {
        return Err(PyValueError::new_err(format!(
            "column {}: TFORM='{}' has zero width", col_index, tform
        )));
    }
    let decimals: Option<usize> = match (letter, decimals_str) {
        ('A', Some(_)) | ('I', Some(_)) => {
            return Err(PyValueError::new_err(format!(
                "column {}: TFORM='{}' has decimal modifier on letter '{}' \
                 (only F/E/D take Fw.d form)",
                col_index, tform, letter,
            )));
        }
        ('F' | 'E' | 'D', None) => {
            return Err(PyValueError::new_err(format!(
                "column {}: TFORM='{}' is missing decimal modifier (need \
                 '{}w.d' form)", col_index, tform, letter,
            )));
        }
        ('F' | 'E' | 'D', Some(d)) => {
            let n: usize = d.trim().parse().map_err(|_| {
                PyValueError::new_err(format!(
                    "column {}: TFORM='{}' decimals '{}' is not a \
                     non-negative integer", col_index, tform, d,
                ))
            })?;
            Some(n)
        }
        _ => None,
    };
    Ok(AsciiTformInfo { letter, width, decimals })
}

// Walk the header cards and produce a Vec<AsciiColumn> describing each
// column.  Reads TFIELDS, then per-column TBCOLn (required) + TFORMn
// (required) + TTYPEn / TUNITn / TSCALn / TZEROn / TNULLn (optional).
//
// Validation:
//   - TBCOLn must be >= 1 (1-based per spec).
//   - byte_offset + byte_width <= NAXIS1 (column fits within a row).
//   - Columns must not overlap (sorted by tbcol, each column's start
//     >= prior column's end).  Gaps are allowed.
pub(crate) fn parse_ascii_columns(
    cards: &[String],
) -> PyResult<Vec<AsciiColumn>> {
    let tfields = parse_keyword(cards, "TFIELDS").ok_or_else(|| {
        PyValueError::new_err("ASCII table missing required TFIELDS keyword")
    })?;
    if tfields < 0 {
        return Err(PyValueError::new_err(format!(
            "ASCII table TFIELDS={} is negative", tfields
        )));
    }
    let n = tfields as usize;
    let naxis1 = parse_keyword(cards, "NAXIS1").unwrap_or(0).max(0) as usize;

    let mut columns: Vec<AsciiColumn> = Vec::with_capacity(n);

    for i in 1..=n {
        let tbcol_i = parse_keyword(cards, &format!("TBCOL{}", i))
            .ok_or_else(|| PyValueError::new_err(format!(
                "ASCII table missing required TBCOL{} keyword", i
            )))?;
        if tbcol_i < 1 {
            return Err(PyValueError::new_err(format!(
                "column {}: TBCOL{}={} must be >= 1 (1-based per spec)",
                i, i, tbcol_i,
            )));
        }
        let byte_offset = (tbcol_i - 1) as usize;

        let tform_key = format!("TFORM{}", i);
        let tform = parse_string_keyword(cards, &tform_key).ok_or_else(|| {
            PyValueError::new_err(format!(
                "ASCII table missing required {} keyword", tform_key
            ))
        })?;
        let AsciiTformInfo { letter, width, decimals } =
            parse_ascii_tform(&tform, i)?;

        if naxis1 > 0 && byte_offset + width > naxis1 {
            return Err(PyValueError::new_err(format!(
                "column {}: TBCOL{}={} + TFORM width {} extends past \
                 NAXIS1={} (row width)", i, i, tbcol_i, width, naxis1,
            )));
        }

        let name = parse_string_keyword(cards, &format!("TTYPE{}", i))
            .unwrap_or_else(|| format!("COL{}", i));
        let tscal = parse_keyword_float(cards, &format!("TSCAL{}", i))
            .unwrap_or(1.0);
        let tzero = parse_keyword_float(cards, &format!("TZERO{}", i))
            .unwrap_or(0.0);
        // TNULL on ASCII tables is a string sentinel.  Store trimmed
        // so the per-cell compare matches the read-side trimmed field
        // text byte-for-byte.
        let tnull = parse_string_keyword(cards, &format!("TNULL{}", i))
            .map(|s| s.trim().to_string());
        let tunit = parse_string_keyword(cards, &format!("TUNIT{}", i));

        columns.push(AsciiColumn {
            name,
            tform_letter: letter,
            byte_offset,
            byte_width: width,
            decimals,
            tscal,
            tzero,
            tnull,
            tunit,
        });
    }

    // Validate non-overlap.  Walk in TBCOL order; the user may have
    // declared columns out of order in the cards, but on-disk they
    // are positioned by TBCOL.  Sort a parallel list of indices so
    // error messages can name the actual TFORMn that overlapped.
    let mut indexed: Vec<(usize, usize, usize)> = columns.iter()
        .enumerate()
        .map(|(idx, c)| (c.byte_offset, c.byte_width, idx))
        .collect();
    indexed.sort_by_key(|&(off, _, _)| off);
    for w in indexed.windows(2) {
        let (off_a, width_a, idx_a) = w[0];
        let (off_b, _width_b, idx_b) = w[1];
        if off_a + width_a > off_b {
            return Err(PyValueError::new_err(format!(
                "ASCII table columns {} (TBCOL={}, width={}) and {} \
                 (TBCOL={}, ...) overlap on disk; this is rejected as \
                 likely-corrupt (no sane writer produces overlap)",
                idx_a + 1, off_a + 1, width_a, idx_b + 1, off_b + 1,
            )));
        }
    }

    Ok(columns)
}
