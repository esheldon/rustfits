// Compressed-image checksum: ZHECKSUM/ZDATASUM over the equivalent
// uncompressed-image bytes.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::common::{
    parse_keyword, parse_string_keyword,
};

use super::hdu::CompressedImageHDU;
use super::read::read_compressed_image_data;

// Read the entire uncompressed image as FITS-big-endian bytes
// (the conceptual data section the equivalent uncompressed HDU
// would carry).  Padded to BLOCK_SIZE.  Scaling is NOT applied
// — the checksum is over the BITPIX-native stored bytes, same
// representation a `BITPIX=ZBITPIX` uncompressed HDU would
// hold.  For quantized-float HDUs the result is the lossy
// dequantized floats (cfitsio convention).
fn read_uncompressed_image_be_bytes(
    slf: &PyRef<'_, CompressedImageHDU>,
    py: Python<'_>,
) -> PyResult<Vec<u8>> {
    // Get the equivalent (BITPIX-native, native-endian) data via
    // the existing read path with scaling off.
    let super_ = slf.as_super().as_super();
    let meta = slf.meta(super_)?;
    let arr_native = read_compressed_image_data(
        py, &meta, super_.offsets.data_offset(),
        &super_.file, &super_.tainted, &slf.cache,
        false, // scale=False — we want stored-space BITPIX-native
        false, // mask_blank=False — checksum is over raw bytes
    )?;
    let arr = arr_native.bind(py);
    let zbitpix = meta.zbitpix;
    let be_dtype = match zbitpix {
        8 => ">u1",
        16 => ">i2",
        32 => ">i4",
        64 => ">i8",
        -32 => ">f4",
        -64 => ">f8",
        other => {
            return Err(PyValueError::new_err(format!(
                "compressed checksum: unsupported ZBITPIX {}",
                other
            )))
        }
    };
    let np = py.import("numpy")?;
    let be = np.call_method1("ascontiguousarray", (arr, be_dtype))?;
    let raw_bytes: Vec<u8> =
        be.call_method0("tobytes")?.extract()?;
    // Pad to FITS block.
    let mut padded = raw_bytes;
    let pad = crate::hdu_image::round_up_to_block(padded.len() as u64)
        - padded.len() as u64;
    padded.extend(std::iter::repeat(0u8).take(pad as usize));
    Ok(padded)
}

// Build the synthetic header bytes of the *equivalent uncompressed
// image HDU* — i.e., what the header would look like if the same
// image were stored without compression.  Used to compute
// ZHECKSUM: we sum (synthetic_uncompressed_header + uncompressed
// data) and encode the complement.
//
// Cards included (minimum required for a valid IMAGE extension
// header + the cards a reader would care about for round-trip):
//   XTENSION = 'IMAGE'  (compressed HDUs can't be primary)
//   BITPIX   = <ZBITPIX>
//   NAXIS    = <ZNAXIS>
//   NAXISn   = <ZNAXISn>
//   PCOUNT   = 0
//   GCOUNT   = 1
//   BSCALE / BZERO / BLANK   (if present in the BINTABLE header
//                              — unsigned-int trick / BLANK
//                              sentinel propagate to the
//                              uncompressed equivalent)
//   EXTNAME / EXTVER         (if present)
//   DATASUM  = <ZDATASUM placeholder / value>
//   CHECKSUM = <ZHECKSUM placeholder / value>
//   END
fn build_equivalent_uncompressed_header(
    cards: &[String],
    datasum_value: &str,
    checksum_value: &str,
) -> PyResult<Vec<String>> {
    use crate::header::{card_int, card_string, pad_to_card};
    let mut out: Vec<String> = Vec::new();

    let zbitpix: i64 = parse_keyword(cards, "ZBITPIX").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing ZBITPIX")
    })?;
    let znaxis: i64 = parse_keyword(cards, "ZNAXIS").ok_or_else(|| {
        PyValueError::new_err("compressed HDU missing ZNAXIS")
    })?;

    out.push(card_string("XTENSION", "IMAGE", "image extension"));
    out.push(card_int("BITPIX", zbitpix, "number of bits per data pixel"));
    out.push(card_int("NAXIS", znaxis, "number of data axes"));
    for i in 1..=znaxis {
        let key = format!("ZNAXIS{}", i);
        let v: i64 = parse_keyword(cards, &key).ok_or_else(|| {
            PyValueError::new_err(format!(
                "compressed HDU missing {}", key,
            ))
        })?;
        out.push(card_int(
            &format!("NAXIS{}", i), v,
            &format!("length of data axis {}", i),
        ));
    }
    out.push(card_int("PCOUNT", 0, "required keyword; must = 0"));
    out.push(card_int("GCOUNT", 1, "required keyword; must = 1"));

    // Propagate optional integer-scaling cards.
    for key in &["BSCALE", "BZERO", "BLANK"] {
        if let Some(idx) =
            cards.iter().position(|c|
                c.len() >= key.len()
                    && c[..key.len()].trim() == *key)
        {
            // Take the card verbatim — preserves the value
            // formatting (signed int, unsigned int, etc.).
            out.push(cards[idx].trim_end().to_string());
        }
    }
    for key in &["EXTNAME", "EXTVER"] {
        if let Some(idx) =
            cards.iter().position(|c|
                c.len() >= key.len()
                    && c[..key.len()].trim() == *key)
        {
            out.push(cards[idx].trim_end().to_string());
        }
    }

    out.push(card_string(
        "DATASUM", datasum_value, "data unit checksum",
    ));
    out.push(card_string(
        "CHECKSUM", checksum_value, "HDU checksum",
    ));
    out.push(pad_to_card("END"));
    Ok(out)
}

pub(crate) fn compressed_add_datasum(
    slf: PyRef<'_, CompressedImageHDU>, py: Python<'_>,
) -> PyResult<()> {
    let data_bytes = read_uncompressed_image_be_bytes(&slf, py)?;
    let sum = crate::checksum::compute_datasum_of(&data_bytes);
    let super_ = slf.into_super().into_super();
    let cards = super_.header_snapshot()?;
    let new_cards =
        crate::checksum::cards_with_datasum(&cards, sum, "ZDATASUM");
    crate::hdu_image::commit_header_update(&super_, new_cards)
}

pub(crate) fn compressed_add_checksum(
    slf: PyRef<'_, CompressedImageHDU>, py: Python<'_>,
) -> PyResult<()> {
    let data_bytes = read_uncompressed_image_be_bytes(&slf, py)?;
    let datasum = crate::checksum::compute_datasum_of(&data_bytes);
    let super_ = slf.into_super().into_super();
    let cards = super_.header_snapshot()?;
    // ZHECKSUM is computed against the *equivalent uncompressed*
    // header bytes (per the FITS Tile Compression Convention),
    // not the BINTABLE header.  Build that synthetic header,
    // sum it + the uncompressed data, encode the complement,
    // then store the encoded value as ZHECKSUM on the BINTABLE.
    let datasum_str = crate::checksum::format_datasum(datasum);
    let synth_zero = build_equivalent_uncompressed_header(
        &cards, &datasum_str, "0000000000000000",
    )?;
    let synth_bytes =
        crate::hdu_image::serialize_header_to_disk_bytes(&synth_zero);
    let hsum = crate::checksum::compute_checksum_bytes(0, &synth_bytes);
    let total = crate::checksum::ones_complement_add(hsum, datasum);
    let encoded = crate::checksum::encode_checksum_ascii(total, true);
    let encoded_str = std::str::from_utf8(&encoded)
        .expect("encode_checksum_ascii produces printable ASCII");
    // Update the BINTABLE's ZDATASUM and ZHECKSUM cards.
    let mut new_cards = cards.clone();
    crate::checksum::set_or_insert_string_card(
        &mut new_cards, "ZDATASUM", &datasum_str,
        "checksum of uncompressed data",
    );
    crate::checksum::set_or_insert_string_card(
        &mut new_cards, "ZHECKSUM", encoded_str,
        "checksum of equivalent uncompressed HDU",
    );
    crate::hdu_image::commit_header_update(&super_, new_cards)
}

pub(crate) fn compressed_verify_datasum(
    slf: PyRef<'_, CompressedImageHDU>, py: Python<'_>,
) -> PyResult<Option<bool>> {
    let super_ref = slf.as_super().as_super();
    let cards = super_ref.header_snapshot()?;
    let Some(expected_str) = parse_string_keyword(&cards, "ZDATASUM")
    else {
        return Ok(None);
    };
    let Some(expected) =
        crate::checksum::parse_datasum(expected_str.trim())
    else {
        return Ok(None);
    };
    let data_bytes = read_uncompressed_image_be_bytes(&slf, py)?;
    let computed = crate::checksum::compute_datasum_of(&data_bytes);
    Ok(Some(computed == expected))
}

pub(crate) fn compressed_verify_checksum(
    slf: PyRef<'_, CompressedImageHDU>, py: Python<'_>,
) -> PyResult<Option<bool>> {
    let super_ref = slf.as_super().as_super();
    let cards = super_ref.header_snapshot()?;
    let Some(_zhecksum) = parse_string_keyword(&cards, "ZHECKSUM")
    else {
        return Ok(None);
    };
    let Some(zdatasum_str) = parse_string_keyword(&cards, "ZDATASUM")
    else {
        // Spec requires ZDATASUM for the invariant to hold.
        return Ok(Some(false));
    };
    // Re-run the equivalent-uncompressed-HDU sum and check the
    // invariant total == 0xFFFFFFFF.
    let zhecksum_str = parse_string_keyword(&cards, "ZHECKSUM").unwrap();
    let synth = build_equivalent_uncompressed_header(
        &cards, zdatasum_str.trim(), zhecksum_str.trim(),
    )?;
    let synth_bytes =
        crate::hdu_image::serialize_header_to_disk_bytes(&synth);
    let data_bytes = read_uncompressed_image_be_bytes(&slf, py)?;
    let hsum = crate::checksum::compute_checksum_bytes(0, &synth_bytes);
    let total =
        crate::checksum::compute_checksum_bytes(hsum, &data_bytes);
    Ok(Some(total == 0xFFFF_FFFF))
}

