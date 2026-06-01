// Write-side setup: AsciiWriteColumn metadata + dtype-to-format
// mapping + numpy structured dtype -> ASCII-table header card builder.
//
// dtype -> TFORM letter + width mappings:
//   - S<w>    -> A<w>
//   - i? (any signed width)   -> I20
//   - u? (any unsigned width) -> I20 + TZERO=2^63 (unsigned-int trick)
//   - f4      -> E15.7
//   - f8      -> D25.17
//   - U<w>    -> A<w>  (UTF-32-LE source; ASCII-validated per cell)
//   - b1      -> rejected (no native bool TFORM letter in ASCII tables)
//   - i1      -> rejected (cfitsio doesn't pick an I width for this;
//                  user can up-cast to i16/i32/i64 explicitly)
//
// User overrides via `formats={"col_name": "F12.4"}` (parsed via the
// same TFORM parser the read side uses).
//
// Column positions on disk pack flush (no inter-column space byte) —
// matches cfitsio's `fits_create_ascii_tbl` convention so files
// rustfits writes and files cfitsio writes have the same TBCOLn
// scheme (and `funpack`/astropy/fitsio all read either identically).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::header::{card_int, card_string, card_uint, pad_to_card};

// Per-column write spec used only at create-time to emit header
// cards.  `byte_offset` is the 0-based TBCOL - 1.  After create,
// subsequent write / append paths drive off the AsciiColumn list
// re-parsed from the on-disk header (via AsciiTableMeta.columns),
// so this struct is intentionally minimal.
pub(crate) struct AsciiWriteColumn {
    pub(crate) name: String,
    pub(crate) tform_letter: char,
    pub(crate) width: usize,
    pub(crate) decimals: Option<usize>,
    pub(crate) byte_offset: usize,
    pub(crate) tzero: Option<u64>,
    pub(crate) tunit: Option<String>,
}

// Parse one TFORM-letter format string (e.g. "F12.4", "I20", "A8")
// into (letter, width, decimals).  Mirrors the read-side parser but
// is local to write_setup because format= overrides validate without
// touching the read parser's `column index` arg.
fn parse_format_override(
    s: &str, col_name: &str,
) -> PyResult<(char, usize, Option<usize>)> {
    let trimmed = s.trim();
    let mut chars = trimmed.chars();
    let letter = chars.next().ok_or_else(|| {
        PyValueError::new_err(format!(
            "formats['{}']: '{}' is empty", col_name, s
        ))
    })?;
    let rest: String = chars.collect();
    let letter = letter.to_ascii_uppercase();
    if !matches!(letter, 'A' | 'I' | 'F' | 'E' | 'D') {
        return Err(PyValueError::new_err(format!(
            "formats['{}']: '{}' uses unsupported letter '{}' \
             (expected A, I, F, E, D)",
            col_name, s, letter,
        )));
    }
    let (width_str, decimals_str) = match rest.find('.') {
        Some(i) => (&rest[..i], Some(&rest[i + 1..])),
        None => (rest.as_str(), None),
    };
    let width: usize = width_str.trim().parse().map_err(|_| {
        PyValueError::new_err(format!(
            "formats['{}']: '{}' width '{}' is not a positive integer",
            col_name, s, width_str,
        ))
    })?;
    if width == 0 {
        return Err(PyValueError::new_err(format!(
            "formats['{}']: '{}' has zero width", col_name, s
        )));
    }
    let decimals: Option<usize> = match (letter, decimals_str) {
        ('A' | 'I', Some(_)) => {
            return Err(PyValueError::new_err(format!(
                "formats['{}']: '{}' has decimals on letter '{}' \
                 (only F/E/D take Fw.d form)",
                col_name, s, letter,
            )));
        }
        ('F' | 'E' | 'D', None) => {
            return Err(PyValueError::new_err(format!(
                "formats['{}']: '{}' missing decimals modifier (need \
                 '{}w.d' form)", col_name, s, letter,
            )));
        }
        ('F' | 'E' | 'D', Some(d)) => Some(d.trim().parse().map_err(|_| {
            PyValueError::new_err(format!(
                "formats['{}']: '{}' decimals '{}' is not a non-negative \
                 integer", col_name, s, d,
            ))
        })?),
        _ => None,
    };
    Ok((letter, width, decimals))
}

// Auto-pick a (TFORM letter, width, decimals, tzero) from a numpy
// field's kind + itemsize.  See the file header for the mapping table.
fn auto_format_for_dtype(
    kind: &str, itemsize: usize, col_name: &str,
) -> PyResult<(char, usize, Option<usize>, Option<u64>)> {
    match (kind, itemsize) {
        // Signed ints: always I20 (max width for i64 base-10).
        ("i", 2) | ("i", 4) | ("i", 8) => Ok(('I', 20, None, None)),
        // Unsigned ints: I20 + TZERO=2^63 (unsigned trick).  The
        // u<n>-to-i64 conversion happens per-cell in the row writer.
        ("u", 1) | ("u", 2) | ("u", 4) | ("u", 8) => {
            Ok(('I', 20, None, Some(1u64 << 63)))
        }
        ("f", 4) => Ok(('E', 15, Some(7), None)),
        ("f", 8) => Ok(('D', 25, Some(17), None)),
        ("S", n) if n > 0 => Ok(('A', n, None, None)),
        // numpy U is UCS-4 (4 bytes/char); A field width = char count.
        ("U", n) if n > 0 && n % 4 == 0 => Ok(('A', n / 4, None, None)),
        ("b", 1) => Err(PyValueError::new_err(format!(
            "column '{}': bool (b1) has no native ASCII-table TFORM \
             letter (FITS ASCII tables don't define a boolean type). \
             Convert to int explicitly (e.g. .astype('i4')) before \
             passing to create_ascii_table_hdu.",
            col_name,
        ))),
        ("i", 1) => Err(PyValueError::new_err(format!(
            "column '{}': int8 (i1) has no clear ASCII-table TFORM \
             width.  Up-cast to i2/i4/i8 explicitly.",
            col_name,
        ))),
        ("S", 0) | ("U", 0) => Err(PyValueError::new_err(format!(
            "column '{}': zero-length string column rejected",
            col_name,
        ))),
        _ => Err(PyValueError::new_err(format!(
            "column '{}': numpy dtype kind '{}' itemsize {} not supported \
             for ASCII tables (supported: i2/i4/i8/u1/u2/u4/u8/f4/f8/S*/U*)",
            col_name, kind, itemsize,
        ))),
    }
}

// Walk a numpy structured dtype, emit per-column write specs in
// field order.  Honors `units=` (informational TUNITn) and
// `formats=` overrides (per-column TFORM string).
//
// `formats` keys are matched case-insensitively against the dtype's
// field names; unmatched keys raise (likely-typo guard).
pub(crate) fn dtype_to_ascii_write_columns(
    dtype: &Bound<'_, PyAny>,
    units: Option<&Bound<'_, PyDict>>,
    formats: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<AsciiWriteColumn>> {
    let names_attr = dtype.getattr("names")?;
    if names_attr.is_none() {
        return Err(PyValueError::new_err(
            "create_ascii_table_hdu: dtype must be a numpy structured \
             dtype with named fields (got a plain dtype)",
        ));
    }
    let names: Vec<String> = names_attr.extract()?;
    if names.is_empty() {
        return Err(PyValueError::new_err(
            "create_ascii_table_hdu: dtype has no fields",
        ));
    }

    // Build the override lookup table (case-insensitive).  Track
    // matched names so we can complain about unused keys at the end.
    let mut format_overrides: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    if let Some(d) = formats {
        for (key, val) in d.iter() {
            let k: String = key.extract().map_err(|_| {
                PyValueError::new_err("formats= keys must be strings")
            })?;
            let v: String = val.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "formats['{}'] must be a TFORM string", k
                ))
            })?;
            format_overrides.insert(k.to_ascii_uppercase(), v);
        }
    }
    let mut matched_overrides: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    let fields = dtype.getattr("fields")?;
    let mut out: Vec<AsciiWriteColumn> = Vec::with_capacity(names.len());
    let mut cursor: usize = 0;
    for name in &names {
        let entry = fields.get_item(name.as_str())?;
        let entry_tup = entry.cast::<PyTuple>()?;
        let field_dtype = entry_tup.get_item(0)?;
        // ASCII tables don't support subarray fields — every column
        // is a scalar text field.  Reject subarray inputs up front.
        let subdtype = field_dtype.getattr("subdtype")?;
        if !subdtype.is_none() {
            return Err(PyValueError::new_err(format!(
                "column '{}': ASCII tables do not support subarray \
                 fields (every column must be a scalar dtype). \
                 Use BINTABLE (create_table_hdu) for subarrays.",
                name,
            )));
        }
        let kind: String = field_dtype.getattr("kind")?.extract()?;
        let itemsize: usize = field_dtype.getattr("itemsize")?.extract()?;

        // Resolve format: explicit override beats auto.  An override
        // does NOT change tzero (that comes from the input dtype).
        let key_upper = name.to_ascii_uppercase();
        let (letter, width, decimals, tzero_auto) = if let Some(s) =
            format_overrides.get(&key_upper)
        {
            matched_overrides.insert(key_upper.clone());
            let (l, w, d) = parse_format_override(s, name)?;
            // Validate override is compatible with input kind.
            validate_format_kind(l, &kind, name)?;
            // Re-derive tzero from input dtype (auto would have set it
            // for u*; honor the same when the override letter is I).
            let tz = if l == 'I' && kind == "u" {
                Some(1u64 << 63)
            } else {
                None
            };
            (l, w, d, tz)
        } else {
            auto_format_for_dtype(&kind, itemsize, name)?
        };

        let tunit: Option<String> = units.and_then(|d| {
            d.get_item(name.as_str()).ok().flatten()
                .and_then(|v| v.extract::<String>().ok())
        });

        out.push(AsciiWriteColumn {
            name: name.clone(),
            tform_letter: letter,
            width,
            decimals,
            byte_offset: cursor,
            tzero: tzero_auto,
            tunit,
        });
        cursor += width;
    }

    // Surface stale format keys (typo guard).
    for key in format_overrides.keys() {
        if !matched_overrides.contains(key) {
            return Err(PyValueError::new_err(format!(
                "formats= contains entry '{}' that does not match any \
                 column in the dtype", key,
            )));
        }
    }

    Ok(out)
}

// Reject an override letter that's incompatible with the input numpy
// kind (e.g. F format on an integer column).  Strings + A: always
// fine.  Numerics paired with the wrong letter would still emit
// something sensible per the format functions, but the user's intent
// is unclear so we surface it.
fn validate_format_kind(
    letter: char, input_kind: &str, col_name: &str,
) -> PyResult<()> {
    let compatible = match letter {
        'A' => matches!(input_kind, "S" | "U"),
        'I' => matches!(input_kind, "i" | "u"),
        'F' | 'E' | 'D' => matches!(input_kind, "f"),
        _ => false,
    };
    if !compatible {
        return Err(PyValueError::new_err(format!(
            "column '{}': formats= override letter '{}' is incompatible \
             with input dtype kind '{}'",
            col_name, letter, input_kind,
        )));
    }
    Ok(())
}

// Build the ASCII-table header card sequence (structural keys +
// EXTNAME/EXTVER + per-column TBCOLn/TFORMn/TTYPEn/TUNITn/TZEROn +
// END).
pub(crate) fn build_ascii_table_header_cards(
    write_columns: &[AsciiWriteColumn],
    nrows: i64,
    extname: Option<&str>,
    extver: Option<i64>,
) -> Vec<String> {
    let row_width: usize = write_columns.iter()
        .map(|c| c.width).sum();
    let mut cards: Vec<String> = vec![card_string(
        "XTENSION", "TABLE", "ASCII table extension",
    )];
    cards.push(card_int("BITPIX", 8, "8-bit bytes"));
    cards.push(card_int("NAXIS", 2, "2-dimensional ASCII table"));
    cards.push(card_int("NAXIS1", row_width as i64, "width of table in bytes"));
    cards.push(card_int("NAXIS2", nrows, "number of rows in table"));
    cards.push(card_int("PCOUNT", 0, "size of special data area"));
    cards.push(card_int("GCOUNT", 1, "one data group (required keyword)"));
    cards.push(card_int(
        "TFIELDS", write_columns.len() as i64, "number of columns",
    ));
    if let Some(name) = extname {
        cards.push(card_string("EXTNAME", name, "name of this HDU"));
    }
    if let Some(ver) = extver {
        cards.push(card_int("EXTVER", ver, "extension version"));
    }
    for (i, col) in write_columns.iter().enumerate() {
        let n = i + 1;
        let tform = match (col.tform_letter, col.decimals) {
            ('A' | 'I', _) => format!("{}{}", col.tform_letter, col.width),
            (_, Some(d)) => format!(
                "{}{}.{}", col.tform_letter, col.width, d
            ),
            (_, None) => format!("{}{}", col.tform_letter, col.width),
        };
        cards.push(card_string(
            &format!("TTYPE{}", n), &col.name, "label for column",
        ));
        cards.push(card_int(
            &format!("TBCOL{}", n),
            (col.byte_offset + 1) as i64,
            "starting column of field",
        ));
        cards.push(card_string(
            &format!("TFORM{}", n), &tform, "data format of column",
        ));
        if let Some(tz) = col.tzero {
            cards.push(card_uint(
                &format!("TZERO{}", n), tz,
                "offset for unsigned integer (unsigned-int trick)",
            ));
        }
        if let Some(unit) = &col.tunit {
            cards.push(card_string(
                &format!("TUNIT{}", n), unit, "physical unit of column",
            ));
        }
    }
    cards.push(pad_to_card("END"));
    cards
}

// Used by FITS.create_ascii_table_hdu: normalize the user-supplied
// dtype (which may be a numpy.dtype OR a descr list of tuples), then
// emit cards.  Returns (cards, row_width).
pub(crate) fn normalize_and_build_ascii_table_header(
    py: Python<'_>,
    dtype_in: &Bound<'_, PyAny>,
    nrows: i64,
    extname: Option<&str>,
    extver: Option<i64>,
    units: Option<&Bound<'_, PyDict>>,
    formats: Option<&Bound<'_, PyDict>>,
) -> PyResult<(Vec<String>, u64, Vec<AsciiWriteColumn>)> {
    let np = py.import("numpy")?;
    let np_dtype = np.getattr("dtype")?.call1((dtype_in,))?;
    let write_columns = dtype_to_ascii_write_columns(
        &np_dtype, units, formats,
    )?;
    let row_width: u64 = write_columns.iter()
        .map(|c| c.width as u64).sum();
    let cards = build_ascii_table_header_cards(
        &write_columns, nrows, extname, extver,
    );
    Ok((cards, row_width, write_columns))
}
