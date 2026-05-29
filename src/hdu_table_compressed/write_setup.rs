// Compressed-table write setup: algorithm/config resolution + per-dtype
// defaults, per-column prep (ColPrep), tile-slab encode helpers, and
// the ZTABLE header build.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::io::{Seek, SeekFrom};
use std::sync::Arc;

use crate::common::{
    lock_file,
    FileHandle, FileLayout, RawBuffer, TaintFlag,
};
use crate::zimage::compression_config::CompressionConfigKind;
use crate::hdu_table::{
    apply_transform_cell,
    bytes_per_element, column_expected_shape,
    column_transform, Column, WriteTransform,
};
use crate::zimage::CompressionAlgorithm;

use super::write::{grow_file_to_at_least};

fn default_table_algorithm(letter: char) -> CompressionAlgorithm {
    match letter {
        // Complex (C/M) defaults to GZIP_1, not GZIP_2: cfitsio's table
        // compressor defaults complex to GZIP_2 but can't read its own
        // GZIP_2-complex output (funpack errors with "error
        // uncompressing image"), so GZIP_2-complex is non-interoperable.
        // GZIP_1-complex round-trips in both rustfits and cfitsio.
        'B' | 'L' | 'A' | 'X' | 'C' | 'M' => CompressionAlgorithm::Gzip1,
        'J' => CompressionAlgorithm::Rice1,
        'I' | 'K' | 'E' | 'D' => CompressionAlgorithm::Gzip2,
        // Unknown letters land at Gzip1 (universally allowed).
        // parse_columns would have rejected anything truly bad
        // upstream; this is a safety net.
        _ => CompressionAlgorithm::Gzip1,
    }
}

fn algorithm_allowed_for_letter(
    letter: char, algo: CompressionAlgorithm,
) -> bool {
    use CompressionAlgorithm::*;
    match algo {
        Gzip1 => true,  // universally allowed
        Gzip2 => !matches!(letter, 'L' | 'A' | 'X'),
        Rice1 => matches!(letter, 'B' | 'I' | 'J'),
        // Hcompress1 and Plio1 are image-only — caller filters
        // them out before reaching this function.
        _ => false,
    }
}

fn allowed_algorithm_names_for_letter(letter: char) -> &'static str {
    match letter {
        'B' => "GZIP_1, RICE_1",
        'I' | 'J' => "GZIP_1, GZIP_2, RICE_1",
        'K' | 'E' | 'D' | 'C' | 'M' => "GZIP_1, GZIP_2",
        'L' | 'A' | 'X' => "GZIP_1",
        _ => "GZIP_1",
    }
}

// Resolve the user's compress= argument into a per-column config
// list.  Returns None when no compression was requested
// (compress=None / False), Some(Vec) otherwise.  Cell types are
// validated against the chosen algorithm before any file mutation.
//
// Accepted shapes:
//   - None / False       -> None (caller falls back to uncompressed)
//   - True               -> defaults per column
//   - str / class        -> same algorithm across all columns
//                          (must be allowed for every column)
//   - dict<str, ...>     -> per-column overrides; unspecified
//                          columns use defaults; values are
//                          strings or config-class instances
pub(crate) fn resolve_compress_arg(
    py: Python<'_>,
    compress: Option<&Bound<'_, PyAny>>,
    columns: &[Column],
) -> PyResult<Option<Vec<CompressionConfigKind>>> {
    let Some(arg) = compress else {
        return Ok(None);
    };
    if arg.is_none() {
        return Ok(None);
    }
    // bool: False -> uncompressed; True -> defaults
    if let Ok(b) = arg.extract::<bool>() {
        if !b {
            return Ok(None);
        }
        return Ok(Some(default_per_column_configs(columns)));
    }

    // dict<col_name, algo>
    if let Ok(dict) = arg.cast::<PyDict>() {
        let mut out: Vec<CompressionConfigKind> = columns.iter()
            .map(|c| build_default_config_for_letter(c.tform_letter))
            .collect();
        // Walk dict items and apply per-column overrides.
        for (key, val) in dict.iter() {
            let name: String = key.extract().map_err(|_| {
                PyValueError::new_err(
                    "compress= dict keys must be strings (column names)")
            })?;
            let pos = columns.iter()
                .position(|c| c.name.eq_ignore_ascii_case(&name))
                .ok_or_else(|| PyValueError::new_err(format!(
                    "compress= dict key '{}' does not match any column \
                     in the table",
                    name)))?;
            let cfg = CompressionConfigKind::from_pyany(&val)?;
            check_table_algorithm_allowed(
                &columns[pos], algorithm_of(&cfg),
            )?;
            out[pos] = cfg;
        }
        return Ok(Some(out));
    }

    // Otherwise treat as a single algorithm (string or class) and
    // apply it everywhere, with per-column validation.
    let cfg = CompressionConfigKind::from_pyany(arg)?;
    let _ = py;  // silence unused in the all-config path
    let algo = algorithm_of(&cfg);
    let mut out = Vec::with_capacity(columns.len());
    for col in columns {
        check_table_algorithm_allowed(col, algo)?;
        out.push(cfg.clone());
    }
    Ok(Some(out))
}

fn default_per_column_configs(columns: &[Column]) -> Vec<CompressionConfigKind> {
    columns.iter()
        .map(|c| build_default_config_for_letter(c.tform_letter))
        .collect()
}

fn build_default_config_for_letter(letter: char) -> CompressionConfigKind {
    let name = match default_table_algorithm(letter) {
        CompressionAlgorithm::Gzip1 => "GZIP_1",
        CompressionAlgorithm::Gzip2 => "GZIP_2",
        CompressionAlgorithm::Rice1 => "RICE_1",
        // Defaults for tables only use the three algorithms above.
        _ => "GZIP_1",
    };
    CompressionConfigKind::from_str(name)
        .expect("default algorithm name is always recognized")
}

fn algorithm_of(cfg: &CompressionConfigKind) -> CompressionAlgorithm {
    match cfg {
        CompressionConfigKind::Gzip1(_) => CompressionAlgorithm::Gzip1,
        CompressionConfigKind::Gzip2(_) => CompressionAlgorithm::Gzip2,
        CompressionConfigKind::Rice1(_) => CompressionAlgorithm::Rice1,
        CompressionConfigKind::Hcompress1(_) => CompressionAlgorithm::Hcompress1,
        CompressionConfigKind::Plio1(_) => CompressionAlgorithm::Plio1,
    }
}

fn check_table_algorithm_allowed(
    col: &Column, algo: CompressionAlgorithm,
) -> PyResult<()> {
    use CompressionAlgorithm::*;
    if matches!(algo, Hcompress1 | Plio1) {
        return Err(PyValueError::new_err(format!(
            "compress= column '{}': {} is an image-only algorithm and \
             cannot be used for tables (the FITS Tile Compression \
             Convention only allows GZIP_1, GZIP_2, and RICE_1 for ZTABLE)",
            col.name,
            match algo {
                Hcompress1 => "HCOMPRESS_1",
                Plio1 => "PLIO_1",
                _ => "?",
            },
        )));
    }
    if !algorithm_allowed_for_letter(col.tform_letter, algo) {
        let algo_name = match algo {
            Gzip1 => "GZIP_1",
            Gzip2 => "GZIP_2",
            Rice1 => "RICE_1",
            _ => "?",
        };
        return Err(PyValueError::new_err(format!(
            "compress= column '{}' (TFORM letter '{}'): {} is not \
             a valid algorithm for this column type.  Allowed for \
             this dtype: {}.  Pass `compress=True` for cfitsio \
             defaults or change this column's algorithm.",
            col.name, col.tform_letter, algo_name,
            allowed_algorithm_names_for_letter(col.tform_letter),
        )));
    }
    Ok(())
}

// Default ZTILELEN, picked the way cfitsio's fits_compress_table
// does (imcompress.c line 8135ish): rowspertile = max(1,
// min(nrows, 10_000_000 / row_width)).
pub(crate) fn default_ztilelen(nrows: usize, row_width: usize) -> usize {
    if nrows == 0 {
        return 1;
    }
    let cap = 10_000_000usize / row_width.max(1);
    cap.max(1).min(nrows)
}

// ---------------------------------------------------------------------------
// Encode one column's per-tile slab
// ---------------------------------------------------------------------------
//
// Input is the column's bytes for this tile, in native order
// (numpy's default).  Output is the compressed blob ready to land
// in the heap.  We don't byteswap to BE first — instead the
// per-algorithm encoder does it (RICE encodes from BE; GZIP_1 and
// GZIP_2 expect BE bytes too because that's what the read side
// reverses).  So caller passes `bytes_be: &[u8]` of length
// `n_pixels * elem_size`.
pub(crate) fn encode_table_column_slab(
    algo: CompressionAlgorithm,
    bytes_be: &[u8],
    n_pixels: usize,
    elem_size: usize,
    rice_blocksize: u32,
    gzip_level: Option<u32>,
) -> PyResult<Vec<u8>> {
    use crate::zimage::gzip::{encode_gzip1, encode_gzip2};
    use crate::zimage::rice::encode_rice;
    match algo {
        CompressionAlgorithm::Gzip1 => encode_gzip1(bytes_be, gzip_level),
        CompressionAlgorithm::Gzip2 => encode_gzip2(
            bytes_be, elem_size as u32, gzip_level,
        ),
        CompressionAlgorithm::Rice1 => encode_rice(
            bytes_be, n_pixels, elem_size as u32, rice_blocksize,
        ),
        _ => Err(PyValueError::new_err("internal: non-table algorithm reached encode_table_column_slab".to_string())),
    }
}

// Pull the gzip level (if set) and rice blocksize from a per-
// column config so the encoder gets the user's chosen params.
pub(crate) fn gzip_level_of(cfg: &CompressionConfigKind) -> Option<u32> {
    match cfg {
        CompressionConfigKind::Gzip1(g) => g.level,
        CompressionConfigKind::Gzip2(g) => g.level,
        _ => None,
    }
}

// Per-column setup shared between the bulk write and append paths.
// Holds everything the per-tile encode loop needs:
//   - the source ndarray's raw byte buffer (`buf`) + per-row stride
//     (`src_total_size`) + the per-cell `WriteTransform` derived from
//     the column's TFORM letter and the input dtype;
//   - encoder-side params (`elem_size` / `per_row_bytes` /
//     `per_row_pixels`) for the slab→blob call;
//   - per-column algorithm params (`rice_blocksize`, `gzip_level`)
//     pulled from the user's compression config.
//
// `contig_arr` pins the ndarray for the buf's lifetime; numpy could
// otherwise free the underlying buffer mid-encode.  Field is held,
// not read.
pub(crate) struct ColPrep<'py> {
    pub(crate) buf: RawBuffer,
    pub(crate) src_total_size: usize,
    pub(crate) transform: WriteTransform,
    #[allow(dead_code)]
    pub(crate) contig_arr: Bound<'py, PyAny>,
    pub(crate) elem_size: usize,
    pub(crate) per_row_bytes: usize,
    pub(crate) per_row_pixels: usize,
    pub(crate) rice_blocksize: u32,
    pub(crate) gzip_level: Option<u32>,
}

// Build a ColPrep from one input ndarray + column metadata + the
// per-column compression config (None when the HDU was reopened,
// in which case algorithm-level defaults apply).  Validates the
// input shape against the column's expected per-cell shape and
// derives the per-cell WriteTransform via the shared classifier;
// failures here surface before any file mutation.
pub(crate) fn prepare_fixed_column<'py>(
    np: &Bound<'py, PyAny>,
    ndarray: &Bound<'py, PyAny>,
    arr: &Bound<'py, PyAny>,
    col: &Column,
    nrows: usize,
    cfg: Option<&CompressionConfigKind>,
) -> PyResult<ColPrep<'py>> {
    if !arr.is_instance(ndarray)? {
        return Err(PyValueError::new_err(format!(
            "compressed table: column '{}' value must be a numpy ndarray",
            col.name)));
    }
    let arr_shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if arr_shape.is_empty() || arr_shape[0] != nrows {
        return Err(PyValueError::new_err(format!(
            "compressed table: column '{}' shape {:?} does not have \
             first axis == {}", col.name, arr_shape, nrows)));
    }
    let per_cell_shape: Vec<usize> = arr_shape[1..].to_vec();
    let expected_shape = column_expected_shape(col);
    if per_cell_shape != expected_shape {
        return Err(PyValueError::new_err(format!(
            "compressed table: column '{}' per-cell shape {:?} does \
             not match expected {:?}",
            col.name, per_cell_shape, expected_shape)));
    }
    let dtype = arr.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let input_elem_size: usize = dtype.getattr("itemsize")?.extract()?;
    let transform = column_transform(col, &kind, input_elem_size)?;
    let cell_elements: usize = per_cell_shape.iter()
        .product::<usize>().max(1);
    let src_total_size = input_elem_size * cell_elements;
    let contig = np.call_method1("ascontiguousarray", (arr,))?;
    let buf = RawBuffer::acquire(&contig)?;
    // X (bit-packed) columns are byte-flat on disk: byte_width =
    // ceil(repeat/8).  The encoders only see bytes here; per_row_pixels
    // is the byte-count rather than the bit-count so `n_pixels *
    // elem_size = slab.len()` for the byte-shuffle / RICE arithmetic
    // (only GZIP_1 is actually allowed for X per the table-allowed
    // matrix, and GZIP_1 ignores both fields).
    let (inner_elem_size, per_row_pixels) = if matches!(
        col.tform_letter, 'X' | 'C' | 'M'
    ) {
        // Byte-flat: X is bit-packed; complex (C/M) is NOT byte-shuffled
        // by cfitsio (its GZIP_2 shuffle skips complex), so encode it
        // unshuffled too -- bytepix 1 makes GZIP_2's shuffle a no-op and
        // keeps the on-disk form cfitsio/funpack-readable (issue #8).
        (1usize, col.byte_width)
    } else {
        let n = bytes_per_element(col.tform_letter)
            .ok_or_else(|| PyValueError::new_err(format!(
                "column '{}': unsupported TFORM letter '{}' on \
                 compressed write", col.name, col.tform_letter)))?;
        (n, col.repeat)
    };
    Ok(ColPrep {
        buf, src_total_size, transform, contig_arr: contig,
        elem_size: inner_elem_size,
        per_row_bytes: col.byte_width,
        per_row_pixels,
        rice_blocksize: cfg.map(rice_blocksize_of).unwrap_or(32),
        gzip_level: cfg.and_then(gzip_level_of),
    })
}

// Take a pre-built FITS big-endian slab (one column × `n_pixels`
// elements), encode it per algorithm, write the compressed blob to
// the heap, and fill the descriptor table entry for this
// (tile_idx, col_idx).  Used by both the write path (after building
// the slab via per-cell transforms) and the append-merge path
// (slab is already in hand from decoded-old + new-transformed bytes).
// Returns the updated heap_cursor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_be_slab_to_heap_and_record(
    slab: &[u8],
    n_pixels: usize,
    algo: CompressionAlgorithm,
    elem_size: usize,
    rice_blocksize: u32,
    gzip_level: Option<u32>,
    tile_idx: usize,
    col_idx: usize,
    col_name: &str,
    descriptor_row_width: usize,
    heap_start_offset: u64,
    mut heap_cursor: u64,
    desc_table: &mut [u8],
    file: &FileHandle,
    layout: &Arc<FileLayout>,
    data_offset: u64,
    tainted: &TaintFlag,
) -> PyResult<u64> {
    use std::io::Write;
    use std::sync::atomic::Ordering;
    let blob = encode_table_column_slab(
        algo, slab, n_pixels, elem_size, rice_blocksize, gzip_level,
    )?;
    let want_total =
        heap_start_offset + heap_cursor + blob.len() as u64 - data_offset;
    grow_file_to_at_least(file, layout, data_offset, want_total, tainted)?;
    {
        let mut g = lock_file(file)?;
        let f = g.as_mut()
            .ok_or_else(|| PyIOError::new_err("file is closed"))?;
        f.seek(SeekFrom::Start(heap_start_offset + heap_cursor))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        f.write_all(&blob).map_err(|e| {
            tainted.store(true, Ordering::Release);
            PyIOError::new_err(format!(
                "compressed table write: heap write failed at \
                 tile {} col '{}': {}", tile_idx, col_name, e))
        })?;
    }
    let desc_off = tile_idx * descriptor_row_width + col_idx * 16;
    let nelems_be = (blob.len() as i64).to_be_bytes();
    let off_be = (heap_cursor as i64).to_be_bytes();
    desc_table[desc_off..desc_off + 8].copy_from_slice(&nelems_be);
    desc_table[desc_off + 8..desc_off + 16].copy_from_slice(&off_be);
    heap_cursor += blob.len() as u64;
    Ok(heap_cursor)
}

// Build the per-tile per-column BE slab from `prep`'s native-order
// source bytes (applying the per-cell WriteTransform — byteswap,
// unsigned-int trick XOR, bool→ASCII, etc.) and hand off to
// `encode_be_slab_to_heap_and_record`.  Used by both write (with
// `source_row_offset = tile_row_start`) and append's new-tile branch
// (with `source_row_offset` past the merged rows).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_and_encode_tile_col(
    prep: &ColPrep,
    col: &Column,
    algo: CompressionAlgorithm,
    tile_idx: usize,
    col_idx: usize,
    rows_in_tile: usize,
    source_row_offset: usize,
    descriptor_row_width: usize,
    heap_start_offset: u64,
    heap_cursor: u64,
    desc_table: &mut [u8],
    file: &FileHandle,
    layout: &Arc<FileLayout>,
    data_offset: u64,
    tainted: &TaintFlag,
) -> PyResult<u64> {
    let src_bytes = prep.buf.as_slice();
    let mut slab = vec![0u8; rows_in_tile * prep.per_row_bytes];
    for r in 0..rows_in_tile {
        let src_row = source_row_offset + r;
        let src_off = src_row * prep.src_total_size;
        let src = &src_bytes
            [src_off..src_off + prep.src_total_size];
        let dst_off = r * prep.per_row_bytes;
        let dst = &mut slab[dst_off..dst_off + prep.per_row_bytes];
        apply_transform_cell(
            &prep.transform, src, dst, &col.name, src_row,
        )?;
    }
    let n_pixels = rows_in_tile * prep.per_row_pixels;
    encode_be_slab_to_heap_and_record(
        &slab, n_pixels, algo, prep.elem_size,
        prep.rice_blocksize, prep.gzip_level,
        tile_idx, col_idx, &col.name, descriptor_row_width,
        heap_start_offset, heap_cursor, desc_table,
        file, layout, data_offset, tainted,
    )
}

pub(crate) fn rice_blocksize_of(cfg: &CompressionConfigKind) -> u32 {
    match cfg {
        CompressionConfigKind::Rice1(r) => r.blocksize,
        _ => 32,
    }
}

// ---------------------------------------------------------------------------
// ZTABLE header construction
// ---------------------------------------------------------------------------
//
// Build the cards for a freshly-created compressed table.  Mirrors
// the cfitsio `fits_compress_table` header layout but produced
// directly from the user's structured dtype (the original
// uncompressed schema) — no copy from an existing BINTABLE.
//
// Result: a Vec<String> of cards ready to serialize, plus the
// computed (n_tiles, descriptor_row_width) the caller needs to
// reserve the data section.
pub(crate) fn build_compressed_table_header(
    cards_in: &[String],            // Pre-built uncompressed header
    row_width: u64,                 // From normalize_and_build_table_header
    nrows: i64,
    ztilelen: usize,
    algorithms: &[CompressionAlgorithm],
    columns: &[Column],
) -> PyResult<(Vec<String>, usize, u64)> {
    use crate::header::{card_int, card_logical, card_string, pad_to_card};

    let ncols = columns.len();
    let n_tiles_u = if nrows <= 0 {
        0usize
    } else {
        let n = nrows as usize;
        n.div_ceil(ztilelen.max(1))
    };
    // Each compressed-table row holds N descriptors; each is 1QB
    // (Q kind, 16 bytes).  Phase 5 only emits 1QB regardless of
    // input — Q-format heap supports arbitrarily large compressed
    // heaps and is what fpack always writes.
    let descriptor_row_width = ncols * 16;

    let mut out: Vec<String> = Vec::with_capacity(cards_in.len() + 16);

    // Structural keys: rewrite NAXIS1/NAXIS2/PCOUNT/TFIELDS into
    // the compressed shape, replace TFORMn with '1QB', drop the
    // input PCOUNT (we set it to 0 for now), and rewrite the
    // commentary lines so the user sees compressed-table semantics.
    for card in cards_in {
        if card.len() < 8 {
            out.push(card.clone());
            continue;
        }
        let kw = card[..8].trim_end();
        if kw == "NAXIS1" {
            out.push(card_int(
                "NAXIS1", descriptor_row_width as i64,
                "width of one compressed-table row in bytes"));
        } else if kw == "NAXIS2" {
            out.push(card_int(
                "NAXIS2", n_tiles_u as i64,
                "number of tiles"));
        } else if kw == "PCOUNT" {
            out.push(card_int(
                "PCOUNT", 0,
                "size of heap in bytes (filled on write)"));
        } else if let Some(suffix) = kw.strip_prefix("TFORM") {
            if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                if let Ok(n) = suffix.parse::<usize>() {
                    if n >= 1 && n <= ncols {
                        out.push(card_string(
                            &format!("TFORM{}", n), "1QB",
                            "compressed data descriptor"));
                        continue;
                    }
                }
            }
            out.push(card.clone());
        } else if kw == "END" {
            // Skip — we'll add the END after our Z-prefix cards.
        } else {
            out.push(card.clone());
        }
    }

    // ZTABLE / ZTILELEN / Z*-shape cards, ZFORM, ZCTYP.
    out.push(card_logical("ZTABLE", true, "this is a compressed table"));
    out.push(card_int(
        "ZTILELEN", ztilelen as i64, "number of rows in each tile"));
    out.push(card_int(
        "ZNAXIS1", row_width as i64,
        "original (uncompressed) row width in bytes"));
    out.push(card_int(
        "ZNAXIS2", nrows, "original (uncompressed) row count"));
    out.push(card_int(
        "ZPCOUNT", 0,
        "original heap size (0 for fixed-only tables)"));

    // ZFORMn (the original TFORMn).  Build from the Column list —
    // cfitsio copies via fits_read_card from the pre-compress
    // header, but constructing from the columns is equivalent and
    // doesn't require parsing the input cards twice.
    //   - Fixed columns: `<repeat><letter>` (e.g. '6E', '10A').
    //   - VLA columns: `1P<inner>` or `1Q<inner>` (e.g. '1PE',
    //     '1QJ').  parse_columns puts `tform_letter` = inner and
    //     `var_kind` = Some('P' | 'Q').
    for (i, col) in columns.iter().enumerate() {
        let n = i + 1;
        let tform = match col.var_kind {
            Some(desc) => format!("1{}{}", desc, col.tform_letter),
            None => format!("{}{}", col.repeat, col.tform_letter),
        };
        out.push(card_string(
            &format!("ZFORM{}", n), &tform,
            "original column TFORM"));
    }
    for (i, &algo) in algorithms.iter().enumerate() {
        let n = i + 1;
        let name = match algo {
            CompressionAlgorithm::Gzip1 => "GZIP_1",
            CompressionAlgorithm::Gzip2 => "GZIP_2",
            CompressionAlgorithm::Rice1 => "RICE_1",
            _ => return Err(PyValueError::new_err("internal: non-table algorithm in build_compressed_table_header".to_string())),
        };
        out.push(card_string(
            &format!("ZCTYP{}", n), name,
            "compression algorithm for this column"));
    }
    out.push(pad_to_card("END"));

    let data_size = (n_tiles_u as u64).saturating_mul(descriptor_row_width as u64);
    Ok((out, n_tiles_u, data_size))
}

