use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use rand::seq::SliceRandom;
use rand::Rng;
use std::fs;
use std::path::Path;

/// Map file extension to a content-type tag.
fn ext_to_type(ext: &str) -> &str {
    match ext {
        "md" | "txt" => "docs",
        "rs" | "py" | "js" | "ts" | "go" | "c" | "cpp" | "h" => "code",
        _ => "docs",
    }
}

/// Known text-file extensions we scan for.
const TEXT_EXTENSIONS: &[&str] = &[
    "md", "txt", "rs", "py", "js", "ts", "go", "c", "cpp", "h",
];

/// Build the inline header bytes: `source:{name}\ntype:{dtype}\n\n`
fn make_header(source: &str, dtype: &str) -> Vec<u8> {
    format!("source:{source}\ntype:{dtype}\n\n").into_bytes()
}

/// Build the inline header bytes with a pass tag for re-encounter training.
fn make_pass_header(source: &str, dtype: &str, pass: u8) -> Vec<u8> {
    format!("source:{source}\ntype:{dtype}\npass:{pass}\n\n").into_bytes()
}

/// Byte-level dataset that produces 256-byte chunks with inline headers.
pub struct ByteDataset {
    chunks: Vec<Vec<u8>>,
    context_window: usize,
}

impl ByteDataset {
    /// Scan `dir` for text files, chunk them into `context_window`-byte windows.
    pub fn from_directory(dir: &Path, context_window: usize) -> Result<Self> {
        if !dir.is_dir() {
            bail!("{} is not a directory", dir.display());
        }

        let mut chunks = Vec::new();

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            if !TEXT_EXTENSIONS.contains(&ext) {
                continue;
            }

            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            let dtype = ext_to_type(ext);
            let header = make_header(filename, dtype);

            let data = fs::read(&path)?;
            Self::chunk_with_header(&data, &header, context_window, &mut chunks);
        }

        let mut rng = rand::thread_rng();
        chunks.shuffle(&mut rng);

        Ok(Self {
            chunks,
            context_window,
        })
    }

    /// Create chunks from a single byte stream with explicit metadata.
    pub fn from_bytes(
        data: &[u8],
        source: &str,
        dtype: &str,
        context_window: usize,
    ) -> Result<Self> {
        let header = make_header(source, dtype);
        let mut chunks = Vec::new();
        Self::chunk_with_header(data, &header, context_window, &mut chunks);
        Ok(Self {
            chunks,
            context_window,
        })
    }

    /// Number of chunks in the dataset.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Return (input, target) tensors for the given chunk indices.
    ///
    /// - input:  bytes\[0..cw-1\] → \[batch, cw-1\] u32
    /// - target: bytes\[1..cw\]   → \[batch, cw-1\] u32
    pub fn get_batch(
        &self,
        indices: &[usize],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = self.context_window - 1; // 255
        let mut input_data: Vec<u32> = Vec::with_capacity(indices.len() * seq_len);
        let mut target_data: Vec<u32> = Vec::with_capacity(indices.len() * seq_len);

        for &idx in indices {
            if idx >= self.chunks.len() {
                bail!(
                    "index {} out of range (dataset has {} chunks)",
                    idx,
                    self.chunks.len()
                );
            }
            let chunk = &self.chunks[idx];
            for i in 0..seq_len {
                input_data.push(chunk[i] as u32);
                target_data.push(chunk[i + 1] as u32);
            }
        }

        let batch = indices.len();
        let input = Tensor::from_vec(input_data, (batch, seq_len), device)?
            .to_dtype(DType::U32)?;
        let target = Tensor::from_vec(target_data, (batch, seq_len), device)?
            .to_dtype(DType::U32)?;
        Ok((input, target))
    }

    /// Sample `batch_size` random chunks and return (input, target).
    pub fn get_random_batch(
        &self,
        batch_size: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        if self.chunks.is_empty() {
            bail!("dataset is empty");
        }
        let mut rng = rand::thread_rng();
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.gen_range(0..self.chunks.len()))
            .collect();
        self.get_batch(&indices, device)
    }

    /// Create re-encounter batch pairs for Phase 4 memory training.
    ///
    /// Each element is (first_input, first_target, second_input, second_target)
    /// where both passes contain the same content but with `pass:1` / `pass:2`
    /// headers so the model can learn to exploit trie reads on the second pass.
    pub fn create_reencounter_batches(
        &self,
        batch_size: usize,
        device: &Device,
    ) -> Result<Vec<(Tensor, Tensor, Tensor, Tensor)>> {
        if self.chunks.is_empty() {
            bail!("dataset is empty");
        }

        let mut rng = rand::thread_rng();
        let mut batches = Vec::new();

        // Process all chunks in batch_size groups.
        let mut indices: Vec<usize> = (0..self.chunks.len()).collect();
        indices.shuffle(&mut rng);

        for group in indices.chunks(batch_size) {
            let seq_len = self.context_window - 1;
            let batch = group.len();

            let mut first_in = Vec::with_capacity(batch * seq_len);
            let mut first_tgt = Vec::with_capacity(batch * seq_len);
            let mut second_in = Vec::with_capacity(batch * seq_len);
            let mut second_tgt = Vec::with_capacity(batch * seq_len);

            for &idx in group {
                let orig = &self.chunks[idx];

                // Extract raw content after the original header.
                // The header ends at the first occurrence of "\n\n".
                let content_start = find_header_end(orig).unwrap_or(0);
                let content = &orig[content_start..];

                // Reconstruct source/type from the original header.
                let (source, dtype) = parse_header(orig);

                let pass1_header = make_pass_header(&source, &dtype, 1);
                let chunk1 = build_chunk(content, &pass1_header, self.context_window);

                let pass2_header = make_pass_header(&source, &dtype, 2);
                let chunk2 = build_chunk(content, &pass2_header, self.context_window);

                for i in 0..seq_len {
                    first_in.push(chunk1[i] as u32);
                    first_tgt.push(chunk1[i + 1] as u32);
                    second_in.push(chunk2[i] as u32);
                    second_tgt.push(chunk2[i + 1] as u32);
                }
            }

            let fi = Tensor::from_vec(first_in, (batch, seq_len), device)?
                .to_dtype(DType::U32)?;
            let ft = Tensor::from_vec(first_tgt, (batch, seq_len), device)?
                .to_dtype(DType::U32)?;
            let si = Tensor::from_vec(second_in, (batch, seq_len), device)?
                .to_dtype(DType::U32)?;
            let st = Tensor::from_vec(second_tgt, (batch, seq_len), device)?
                .to_dtype(DType::U32)?;

            batches.push((fi, ft, si, st));
        }

        Ok(batches)
    }

    // ── internal helpers ──

    /// Split `data` into `context_window`-sized chunks, each prefixed with `header`.
    /// Content is truncated or zero-padded to exactly `context_window` bytes.
    fn chunk_with_header(
        data: &[u8],
        header: &[u8],
        context_window: usize,
        out: &mut Vec<Vec<u8>>,
    ) {
        if header.len() >= context_window {
            // Header alone fills the window — nothing useful we can do.
            return;
        }
        let content_capacity = context_window - header.len();
        if data.is_empty() {
            return;
        }

        let mut offset = 0;
        while offset < data.len() {
            let end = (offset + content_capacity).min(data.len());
            let mut chunk = Vec::with_capacity(context_window);
            chunk.extend_from_slice(header);
            chunk.extend_from_slice(&data[offset..end]);
            // Pad to exactly context_window bytes.
            chunk.resize(context_window, 0u8);
            out.push(chunk);
            offset = end;
        }
    }
}

/// Find the byte offset just past the first `\n\n` in `buf`.
fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(2)
        .position(|w| w == b"\n\n")
        .map(|p| p + 2)
}

/// Best-effort extraction of source and type from an existing chunk header.
fn parse_header(chunk: &[u8]) -> (String, String) {
    let end = find_header_end(chunk).unwrap_or(chunk.len().min(64));
    let header_str = String::from_utf8_lossy(&chunk[..end]);
    let mut source = String::from("unknown");
    let mut dtype = String::from("docs");
    for line in header_str.lines() {
        if let Some(val) = line.strip_prefix("source:") {
            source = val.to_string();
        } else if let Some(val) = line.strip_prefix("type:") {
            dtype = val.to_string();
        }
    }
    (source, dtype)
}

/// Assemble a single chunk: header + content, padded/truncated to `context_window`.
fn build_chunk(content: &[u8], header: &[u8], context_window: usize) -> Vec<u8> {
    let mut chunk = Vec::with_capacity(context_window);
    chunk.extend_from_slice(header);
    let remaining = context_window.saturating_sub(header.len());
    let take = remaining.min(content.len());
    chunk.extend_from_slice(&content[..take]);
    chunk.resize(context_window, 0u8);
    chunk
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_ext_to_type() {
        assert_eq!(ext_to_type("rs"), "code");
        assert_eq!(ext_to_type("md"), "docs");
        assert_eq!(ext_to_type("xyz"), "docs");
    }

    #[test]
    fn test_from_bytes_chunk_size() {
        let data = vec![b'A'; 600];
        let ds = ByteDataset::from_bytes(&data, "test.txt", "docs", 256).unwrap();
        assert!(ds.len() >= 3); // 600 bytes / ~200 content per chunk
        for chunk in &ds.chunks {
            assert_eq!(chunk.len(), 256);
        }
    }

    #[test]
    fn test_get_batch_shapes() {
        let data = vec![b'X'; 512];
        let ds = ByteDataset::from_bytes(&data, "f.rs", "code", 256).unwrap();
        assert!(ds.len() >= 1);
        let (inp, tgt) = ds.get_batch(&[0], &Device::Cpu).unwrap();
        assert_eq!(inp.dims(), &[1, 255]);
        assert_eq!(tgt.dims(), &[1, 255]);
    }

    #[test]
    fn test_autoregressive_shift() {
        let data = vec![b'Z'; 256];
        let ds = ByteDataset::from_bytes(&data, "t.txt", "docs", 256).unwrap();
        let (inp, tgt) = ds.get_batch(&[0], &Device::Cpu).unwrap();
        let inp_vec: Vec<u32> = inp.flatten_all().unwrap().to_vec1().unwrap();
        let tgt_vec: Vec<u32> = tgt.flatten_all().unwrap().to_vec1().unwrap();
        // target[i] == chunk[i+1], input[i] == chunk[i]
        // So target[i] should equal input[i+1] for all valid i.
        for i in 0..254 {
            assert_eq!(tgt_vec[i], inp_vec[i + 1]);
        }
    }

    #[test]
    fn test_reencounter_batches() {
        let data = vec![b'R'; 512];
        let ds = ByteDataset::from_bytes(&data, "re.txt", "docs", 256).unwrap();
        let batches = ds.create_reencounter_batches(4, &Device::Cpu).unwrap();
        assert!(!batches.is_empty());
        let (fi, ft, si, st) = &batches[0];
        assert_eq!(fi.dims()[1], 255);
        assert_eq!(ft.dims()[1], 255);
        assert_eq!(si.dims()[1], 255);
        assert_eq!(st.dims()[1], 255);
    }
}
