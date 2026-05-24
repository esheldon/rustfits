// Bytes-bounded LRU cache shared between CompressedImageHDU and
// CompressedTableHDU.  Both hold decompressed tile (or per-(tile,
// col)) byte buffers wrapped in `Arc<Vec<u8>>` so callers can clone
// the Arc out under the lock and then work on the bytes without
// holding it.  Eviction policy: standard LRU triggered by a byte
// budget rather than entry count — many small tiles can coexist
// with a few large ones, capped by the same total.
//
// Concurrency model: a single inner Mutex serializes get/put/clear.
// `get` holds the lock just long enough to clone the Arc; decode
// work (the expensive part) happens without the lock, so multi-
// reader workloads don't serialize through long decodes.  A race
// where two callers both miss the same key and both decode is
// acceptable: each gets a correct Arc, and `put` is idempotent
// (replaces any existing entry's accounting before re-inserting).

use lru::LruCache;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub(crate) struct BytesBoundLruCache<K: Hash + Eq> {
    inner: Mutex<Inner<K>>,
    max_bytes: AtomicU64,
}

struct Inner<K: Hash + Eq> {
    // Unbounded entry count — we evict by byte budget on insert.
    lru: LruCache<K, Arc<Vec<u8>>>,
    cur_bytes: u64,
}

impl<K: Hash + Eq> BytesBoundLruCache<K> {
    pub(crate) fn new(max_bytes: u64) -> Self {
        BytesBoundLruCache {
            inner: Mutex::new(Inner {
                lru: LruCache::unbounded(),
                cur_bytes: 0,
            }),
            max_bytes: AtomicU64::new(max_bytes),
        }
    }

    pub(crate) fn capacity(&self) -> u64 {
        self.max_bytes.load(Ordering::Relaxed)
    }

    pub(crate) fn used_bytes(&self) -> u64 {
        self.inner.lock().map(|g| g.cur_bytes).unwrap_or(0)
    }

    // Look up a key.  On hit, marks the entry MRU and returns a
    // clone of the Arc — caller works without holding the lock.
    pub(crate) fn get(&self, key: &K) -> Option<Arc<Vec<u8>>> {
        let mut guard = self.inner.lock().ok()?;
        guard.lru.get(key).cloned()
    }

    // Insert.  If `bytes` alone is larger than the cap, silently
    // drop it — no point evicting everything else for one giant
    // value that won't fit.  If cap == 0, caching is disabled —
    // also a no-op.
    pub(crate) fn put(&self, key: K, bytes: Arc<Vec<u8>>) {
        let cap = self.capacity();
        if cap == 0 {
            return;
        }
        let size = bytes.len() as u64;
        if size > cap {
            return;
        }
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        // Replace any existing entry (free its bytes from the
        // accounting) before evicting more.
        if let Some(old) = guard.lru.pop(&key) {
            guard.cur_bytes =
                guard.cur_bytes.saturating_sub(old.len() as u64);
        }
        while guard.cur_bytes + size > cap {
            match guard.lru.pop_lru() {
                Some((_, val)) => {
                    guard.cur_bytes = guard.cur_bytes
                        .saturating_sub(val.len() as u64);
                }
                None => break,
            }
        }
        guard.lru.put(key, bytes);
        guard.cur_bytes += size;
    }

    // Change the capacity.  Evicts LRU entries if the new cap is
    // smaller than current usage.  Setting cap = 0 effectively
    // empties the cache.
    pub(crate) fn set_capacity(&self, max_bytes: u64) {
        self.max_bytes.store(max_bytes, Ordering::Relaxed);
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        while guard.cur_bytes > max_bytes {
            match guard.lru.pop_lru() {
                Some((_, val)) => {
                    guard.cur_bytes = guard.cur_bytes
                        .saturating_sub(val.len() as u64);
                }
                None => break,
            }
        }
    }

    pub(crate) fn clear(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.lru.clear();
            guard.cur_bytes = 0;
        }
    }
}
