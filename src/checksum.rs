// FITS Checksum Convention (Pence & Seaman 2010 / ADASS 1994).
//
// Two header cards record HDU integrity:
//   DATASUM   ASCII decimal representation of the 32-bit 1's-
//             complement checksum of the data section.
//   CHECKSUM  16-char ASCII encoding chosen so the 1's-complement
//             sum of all the HDU's bytes (header + data, after
//             CHECKSUM is written) equals 0xFFFFFFFF.  Readers
//             verify by summing the whole HDU and comparing.
//
// Tile-compressed images use ZHECKSUM / ZDATASUM with the same
// algorithm and encoding, computed against the conceptual
// UNCOMPRESSED image (so the user can verify data integrity
// through decompression).
//
// Routines here are direct ports of cfitsio's `checksum.c`
// (ffcsum / ffesum / ffdsum) — byte-exact agreement makes
// astropy / fitsio cross-verification trivial.

// ---------- core checksum primitive ----------

// 32-bit 1's-complement sum of `bytes`, treating them as a
// sequence of big-endian u32 chunks.  Last partial chunk
// (bytes.len() % 4 != 0) is zero-padded on the right (the bytes
// occupy the high-order positions of the synthesised u32).
//
// `seed` lets the caller accumulate across multiple calls (pass
// 0 to start fresh).  Returns the running sum.
//
// The high/low split mirrors cfitsio's `ffcsum`: summing
// 16-bit halves separately means many additions fit in u32
// without overflow, and the carry-fold-back loop at the end
// implements the end-around-carry of 1's-complement arithmetic.
pub(crate) fn compute_checksum_bytes(seed: u32, bytes: &[u8]) -> u32 {
    let mut hi: u32 = seed >> 16;
    let mut lo: u32 = seed & 0xFFFF;

    let mut iter = bytes.chunks_exact(4);
    for chunk in &mut iter {
        // BE u32 → upper u16 contributes to hi, lower u16 to lo.
        let upper = u16::from_be_bytes([chunk[0], chunk[1]]) as u32;
        let lower = u16::from_be_bytes([chunk[2], chunk[3]]) as u32;
        hi += upper;
        lo += lower;
    }
    // Tail: pad bytes into the high-order positions of a final u32.
    let tail = iter.remainder();
    if !tail.is_empty() {
        let mut pad = [0u8; 4];
        pad[..tail.len()].copy_from_slice(tail);
        let upper = u16::from_be_bytes([pad[0], pad[1]]) as u32;
        let lower = u16::from_be_bytes([pad[2], pad[3]]) as u32;
        hi += upper;
        lo += lower;
    }
    // End-around carry (port of cfitsio's ffcsum tail loop).
    loop {
        let hicarry = hi >> 16;
        let locarry = lo >> 16;
        if hicarry == 0 && locarry == 0 {
            break;
        }
        hi = (hi & 0xFFFF) + locarry;
        lo = (lo & 0xFFFF) + hicarry;
    }
    (hi << 16) | lo
}

// ---------- 16-char ASCII encoding (cfitsio `ffesum`) ----------

// ASCII punctuation chars that are excluded by the FITS Checksum
// Convention (so the encoded CHECKSUM string contains only
// letters and digits).
const EXCLUDE: [u8; 13] = [
    0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40,
    0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x60,
];

// Encode a 32-bit checksum (or its complement) as a 16-byte
// printable-ASCII string, per the FITS Checksum Convention.
// `complement = true` encodes the complement of `sum` — what
// the CHECKSUM card stores so that the total HDU checksum lands
// on 0xFFFFFFFF.  Direct port of cfitsio's `ffesum`.
pub(crate) fn encode_checksum_ascii(sum: u32, complement: bool) -> [u8; 16] {
    let value = if complement { !sum } else { sum };
    let offset = 0x30u8;
    let mut asc = [0u8; 16];
    let masks: [u32; 4] = [0xFF00_0000, 0x00FF_0000, 0x0000_FF00, 0x0000_00FF];

    for ii in 0..4 {
        let byte = ((value & masks[ii]) >> (24 - (8 * ii))) as i32;
        let quotient = byte / 4 + offset as i32;
        let remainder = byte % 4;
        let mut ch = [quotient; 4];
        ch[0] += remainder;

        // Avoid ASCII punctuation: rebalance offending pairs
        // (ch[0..2] and ch[2..4]) by +1/-1 until none match.
        loop {
            let mut adjusted = false;
            for kk in 0..EXCLUDE.len() {
                for jj in (0..4).step_by(2) {
                    if (ch[jj] as u32) & 0xFF == EXCLUDE[kk] as u32
                        || (ch[jj + 1] as u32) & 0xFF == EXCLUDE[kk] as u32
                    {
                        ch[jj] += 1;
                        ch[jj + 1] -= 1;
                        adjusted = true;
                    }
                }
            }
            if !adjusted {
                break;
            }
        }
        for jj in 0..4 {
            asc[4 * jj + ii] = ch[jj] as u8;
        }
    }

    // Shift one position to the right (asc[15] → ascii[0]) per
    // cfitsio's final byte-rotation.
    let mut ascii = [0u8; 16];
    for ii in 0..16 {
        ascii[ii] = asc[(ii + 15) % 16];
    }
    ascii
}

// Decode a 16-char ASCII CHECKSUM string back to the 32-bit
// sum.  When `complement = true`, returns the un-complemented
// value (i.e., the original sum the card encodes).  Direct port
// of cfitsio's `ffdsum`.
pub(crate) fn decode_checksum_ascii(ascii: &[u8], complement: bool) -> u32 {
    debug_assert!(ascii.len() >= 16);
    let mut cbuf = [0i32; 16];
    // Un-shift: ascii[1] → cbuf[0], ascii[0] → cbuf[15].
    for ii in 0..16 {
        cbuf[ii] = ascii[(ii + 1) % 16] as i32 - 0x30;
    }
    let mut hi: u32 = 0;
    let mut lo: u32 = 0;
    for ii in (0..16).step_by(4) {
        hi += ((cbuf[ii] as u32) << 8) + cbuf[ii + 1] as u32;
        lo += ((cbuf[ii + 2] as u32) << 8) + cbuf[ii + 3] as u32;
    }
    loop {
        let hicarry = hi >> 16;
        let locarry = lo >> 16;
        if hicarry == 0 && locarry == 0 {
            break;
        }
        hi = (hi & 0xFFFF) + locarry;
        lo = (lo & 0xFFFF) + hicarry;
    }
    let sum = (hi << 16) | lo;
    if complement {
        !sum
    } else {
        sum
    }
}

// ---------- DATASUM card formatting ----------

// DATASUM card value is the 32-bit checksum rendered as a
// decimal string (per the FITS Checksum Convention).  Up to
// 10 digits (u32::MAX = 4294967295).
pub(crate) fn format_datasum(sum: u32) -> String {
    sum.to_string()
}

// Parse a DATASUM card value back to u32.  Returns None if the
// string isn't a valid u32 decimal.
pub(crate) fn parse_datasum(s: &str) -> Option<u32> {
    s.trim().parse::<u32>().ok()
}

// ---------- shared card-management helpers ----------

// Update an existing standard-key card to a new string value, or
// insert it just before END if absent.  Bypasses the protected-
// keys guard since CHECKSUM/DATASUM/ZHECKSUM/ZDATASUM are
// internally-managed cards.
pub(crate) fn set_or_insert_string_card(
    cards: &mut Vec<String>,
    key: &str,
    value: &str,
    comment: &str,
) {
    let card = crate::header::card_string(key, value, comment)
        .trim_end()
        .to_string();
    let key_uc = key.to_uppercase();
    let needle_len = key.len();
    if let Some(idx) = cards.iter().position(|c| {
        c.len() >= needle_len
            && c[..needle_len].trim().to_uppercase() == key_uc
    }) {
        cards[idx] = card;
    } else {
        // Insert before END (always last).
        let pos = cards.len().saturating_sub(1);
        cards.insert(pos, card);
    }
}

// ---------- DATASUM / CHECKSUM computation against an HDU ----------

// Compute DATASUM from `data_bytes` (the padded data section as
// it lives on disk, in FITS big-endian order).  Returns the u32
// sum ready for format_datasum.
pub(crate) fn compute_datasum_of(data_bytes: &[u8]) -> u32 {
    compute_checksum_bytes(0, data_bytes)
}

// Streaming accumulator over the cfitsio-byte-exact checksum.
// `compute_checksum_bytes` requires every intermediate call's
// bytes to be a multiple of 4 (its tail-handling zero-pads
// partial trailing groups, which produces wrong results when
// followed by more bytes).  This accumulator buffers up to 3
// leftover bytes between feeds so callers can stream arbitrary
// chunk sizes safely — necessary whenever the data section is
// too large to materialize in RAM as a single Vec<u8>.
//
// Used by the uncompressed-HDU checksum path (which reads the
// data section in 1 MiB chunks) and by the compressed-table
// checksum path (which walks tiles, decoding each per-(tile, col)
// blob and feeding the assembled per-tile main rows).  In both
// cases peak memory is bounded at a per-chunk constant
// independent of file size.
pub(crate) struct ChecksumStream {
    seed: u32,
    carry: Vec<u8>,
}

impl ChecksumStream {
    pub(crate) fn new(seed: u32) -> Self {
        Self { seed, carry: Vec::with_capacity(4) }
    }

    pub(crate) fn feed(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        // If carry is non-empty, stitch with the prefix of bytes
        // up to a 4-byte boundary, process that one chunk, then
        // continue with the remaining input.
        if !self.carry.is_empty() {
            let need = 4 - self.carry.len();
            if bytes.len() < need {
                self.carry.extend_from_slice(bytes);
                return;
            }
            self.carry.extend_from_slice(&bytes[..need]);
            self.seed = compute_checksum_bytes(self.seed, &self.carry);
            self.carry.clear();
            let rest = &bytes[need..];
            let n_full = rest.len() / 4 * 4;
            if n_full > 0 {
                self.seed = compute_checksum_bytes(
                    self.seed, &rest[..n_full],
                );
            }
            if rest.len() > n_full {
                self.carry.extend_from_slice(&rest[n_full..]);
            }
        } else {
            let n_full = bytes.len() / 4 * 4;
            if n_full > 0 {
                self.seed = compute_checksum_bytes(
                    self.seed, &bytes[..n_full],
                );
            }
            if bytes.len() > n_full {
                self.carry.extend_from_slice(&bytes[n_full..]);
            }
        }
    }

    pub(crate) fn finish(mut self) -> u32 {
        // Any remaining carry is the final tail; let
        // compute_checksum_bytes apply its zero-pad rule.
        if !self.carry.is_empty() {
            self.seed = compute_checksum_bytes(self.seed, &self.carry);
        }
        self.seed
    }
}

// Given a cards Vec, set DATASUM (overwrites or inserts) and
// return the resulting Vec.  Caller is expected to follow the
// "disk-write-before-commit" pattern: rewrite header on disk,
// then commit the new cards.
pub(crate) fn cards_with_datasum(
    cards: &[String], sum: u32, key: &str,
) -> Vec<String> {
    let mut new_cards: Vec<String> = cards.to_vec();
    set_or_insert_string_card(
        &mut new_cards,
        key,
        &format_datasum(sum),
        "data unit checksum",
    );
    new_cards
}

// Given a cards Vec with DATASUM already in place, compute and
// set the CHECKSUM card so the total HDU checksum equals
// 0xFFFFFFFF.  Algorithm (per cfitsio's ffpcks):
//   1. Set CHECKSUM card placeholder to all-zeros (16 chars).
//   2. Compute checksum of the header bytes with the placeholder.
//   3. Add the data checksum (= DATASUM).
//   4. Encode the complement of the total → final 16 chars.
//   5. Replace the placeholder card with the encoded value.
// Caller supplies the data checksum (so we don't re-read the
// data section just to recompute DATASUM — the caller usually
// just ran add_datasum which left the value in hand).
pub(crate) fn cards_with_checksum(
    cards: &[String], datasum: u32, key: &str,
) -> Vec<String> {
    let mut new_cards: Vec<String> = cards.to_vec();
    // Step 1: placeholder.
    set_or_insert_string_card(
        &mut new_cards,
        key,
        "0000000000000000",
        "HDU checksum",
    );
    // Steps 2 + 3: checksum the placeholder-header + data.
    let header_bytes = crate::hdu_image::serialize_header_to_disk_bytes(
        &new_cards,
    );
    let hsum = compute_checksum_bytes(0, &header_bytes);
    let total = ones_complement_add(hsum, datasum);
    // Step 4: encoded complement.
    let encoded = encode_checksum_ascii(total, true);
    let encoded_str = std::str::from_utf8(&encoded)
        .expect("encode_checksum_ascii guarantees printable ASCII");
    // Step 5.
    set_or_insert_string_card(
        &mut new_cards, key, encoded_str, "HDU checksum",
    );
    new_cards
}

// 1's-complement add of two 32-bit values: a + b with end-around
// carry.  Used to compose the header and data checksums into a
// total HDU checksum without re-summing all bytes.
pub(crate) fn ones_complement_add(a: u32, b: u32) -> u32 {
    // Reuse compute_checksum_bytes: it's effectively a seeded
    // accumulator.  Pack `b` into 4 BE bytes and add against
    // seed `a`.
    compute_checksum_bytes(a, &b.to_be_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Anchor: 4 zero bytes → checksum 0.
    #[test]
    fn checksum_of_zeros_is_zero() {
        assert_eq!(compute_checksum_bytes(0, &[0; 16]), 0);
    }

    // Anchor: 4 0xFF bytes = 0xFFFFFFFF, then carry-fold-back
    // gives 0xFFFFFFFF (-0 in 1's complement).
    #[test]
    fn checksum_of_one_ff_word() {
        assert_eq!(compute_checksum_bytes(0, &[0xFF; 4]), 0xFFFFFFFF);
    }

    // Anchor: 8 bytes of 0xFF — two 0xFFFFFFFF words summed in
    // 1's complement = 0xFFFFFFFF (the "negative zero" identity).
    #[test]
    fn checksum_one_complement_negative_zero_stable() {
        assert_eq!(compute_checksum_bytes(0, &[0xFF; 8]), 0xFFFFFFFF);
    }

    // Tail-padding: 1 byte 0x12 → pads to (0x12, 0, 0, 0) →
    // u32 0x12000000.
    #[test]
    fn checksum_partial_tail() {
        assert_eq!(compute_checksum_bytes(0, &[0x12]), 0x12000000);
    }

    // Round-trip: encode then decode reproduces the input.
    #[test]
    fn encode_decode_round_trip_no_complement() {
        for sum in [0u32, 1, 0x1234_5678, 0xFFFF_FFFE, 0xDEAD_BEEF] {
            let encoded = encode_checksum_ascii(sum, false);
            let decoded = decode_checksum_ascii(&encoded, false);
            assert_eq!(
                decoded, sum,
                "round-trip failed for {:#010x}",
                sum
            );
        }
    }

    #[test]
    fn encode_decode_round_trip_complement() {
        for sum in [0u32, 1, 0x1234_5678, 0xFFFF_FFFE] {
            let encoded = encode_checksum_ascii(sum, true);
            let decoded = decode_checksum_ascii(&encoded, true);
            assert_eq!(decoded, sum);
        }
    }

    // Encoded output is printable ASCII (letters + digits only,
    // no punctuation).
    #[test]
    fn encoded_chars_are_alnum() {
        for sum in [0u32, 0xDEAD_BEEF, 0x1234_5678] {
            let encoded = encode_checksum_ascii(sum, false);
            for &c in encoded.iter() {
                assert!(
                    c.is_ascii_alphanumeric(),
                    "non-alnum char {:#x} in encoded output for {:#x}",
                    c,
                    sum
                );
            }
        }
    }

    // Anchor against astropy: the empirical test above wrote
    // a (8,8) i4 arange image and astropy reported DATASUM=2016.
    // The data is 64 i4 BE values = arange(64) → sum of u32
    // values = 0+1+2+...+63 = 2016.  Our compute should match.
    #[test]
    fn datasum_matches_astropy_anchor() {
        let mut bytes: Vec<u8> = Vec::with_capacity(64 * 4);
        for i in 0..64u32 {
            bytes.extend_from_slice(&i.to_be_bytes());
        }
        let sum = compute_checksum_bytes(0, &bytes);
        assert_eq!(sum, 2016);
        assert_eq!(format_datasum(sum), "2016");
    }

    // Byte-exact agreement with cfitsio's ffesum.  Anchor
    // values come from running cfitsio's encoder over the same
    // inputs (see commit message / dev notes for the
    // /tmp/test_encode harness).
    #[test]
    fn encode_matches_cfitsio() {
        let cases: &[(u32, bool, &str)] = &[
            (0x12345678, false, "N6AGN49EN4AEN49E"),
            (0x12345678, true,  "QleaTkbTQkbZQkbZ"),
            (0xdeadbeef, false, "kiafngVZkgadkgUZ"),
            (0xdeadbeef, true,  "49FH48D948DG48D9"),
            (0xFFFFFFFE, false, "orrrqooooooooooo"),
            (0xFFFFFFFE, true,  "0000100000000000"),
            (0x1,        false, "0000100000000000"),
            (0x1,        true,  "orrrqooooooooooo"),
        ];
        for (sum, complm, expected) in cases {
            let got = encode_checksum_ascii(*sum, *complm);
            let got_s = std::str::from_utf8(&got).unwrap();
            assert_eq!(
                got_s, *expected,
                "encode_checksum_ascii({:#010x}, {}) = {:?} expected {:?}",
                sum, complm, got_s, expected
            );
        }
    }

    // (The full CHECKSUM-card invariant — that the total HDU
    // sum equals 0xFFFFFFFF after CHECKSUM is written — requires
    // the 16 encoded chars to be placed at specific byte offsets
    // within an 80-byte CHECKSUM card, not summed in isolation.
    // That's tested end-to-end via the ImageHDU.add_checksum +
    // verify_checksum path in tests/test_checksum.py.)
}
