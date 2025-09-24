//! Parse rust's char to c_char (with mapping).

use core::ffi::c_char;
use std::collections::BTreeSet;

static MAP62: &[u8; 62] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/// Map rust's indices to c_char array for TBLIS.
///
/// Rules for this function:
/// 1. If all indices are in ASCII range, directly cast to c_char.
/// 2. If there are no more than 62 characters in all indices, map them to [a-zA-Z0-9].
/// 3. If there are no more than 128 characters in all indices, map them to extended ASCII (0-127).
/// 4. Otherwise will panic.
pub fn char_parse(indices: &[&str]) -> Result<Vec<Vec<c_char>>, String> {
    let all_chars = indices.iter().flat_map(|s| s.chars()).collect::<BTreeSet<char>>();
    let all_chars_len = all_chars.len();
    // rule 1
    if all_chars.iter().all(|&c| c.is_ascii()) {
        return Ok(indices.iter().map(|s| s.chars().map(|c| c as c_char).collect()).collect());
    }
    match all_chars_len {
        0..=62 => {
            // rule 2
            let char_map = all_chars
                .into_iter()
                .zip(MAP62.iter().map(|&b| b as c_char))
                .collect::<std::collections::BTreeMap<char, c_char>>();
            Ok(indices.iter().map(|s| s.chars().map(|c| char_map[&c]).collect()).collect())
        },
        63..=128 => {
            // rule 3
            let char_map = all_chars
                .into_iter()
                .zip((0..=127).map(|b| b as c_char))
                .collect::<std::collections::BTreeMap<char, c_char>>();
            Ok(indices.iter().map(|s| s.chars().map(|c| char_map[&c]).collect()).collect())
        },
        _ => Err(format!(
            "Too many unique characters in indices ({all_chars_len} > 128). Your indices are: {indices:#?}"
        )),
    }
}
