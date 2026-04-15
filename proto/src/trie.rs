use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::config::ZenoConfig;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrieNodeSnapshot {
    pub address: Vec<u8>,
    pub depth: usize,
    pub has_value: bool,
    pub has_summary: bool,
    pub has_residual: bool,
    pub populated_children: usize,
    pub structural_density: usize,
    pub subtree_values: usize,
    pub children: Vec<u8>,
    pub render: Option<TrieTensorSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrieEdgeSnapshot {
    pub parent: Vec<u8>,
    pub child: Vec<u8>,
    pub branch: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseTrieSnapshot {
    pub focus: Vec<Vec<u8>>,
    pub nodes: Vec<TrieNodeSnapshot>,
    pub edges: Vec<TrieEdgeSnapshot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrieTensorSource {
    Value,
    Summary,
    Residual,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrieTensorStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub l2_norm: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrieTopChannel {
    pub index: usize,
    pub value: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrieTensorSnapshot {
    pub source: TrieTensorSource,
    pub shape: [usize; 2],
    pub values: Vec<f32>,
    pub stats: TrieTensorStats,
    pub top_channels: Vec<TrieTopChannel>,
}

#[derive(Default)]
struct TrieNode {
    value: Option<Tensor>,
    summary: Option<Tensor>,
    residual: Option<Tensor>,
    children: BTreeMap<u8, TrieNode>,
    populated_children: usize,
    subtree_values: usize,
}

impl TrieNode {
    fn snapshot(&self, address: Vec<u8>) -> TrieNodeSnapshot {
        TrieNodeSnapshot {
            depth: address.len(),
            address,
            has_value: self.value.is_some(),
            has_summary: self.summary.is_some(),
            has_residual: self.residual.is_some(),
            populated_children: self.populated_children,
            structural_density: self.structural_density(),
            subtree_values: self.subtree_values,
            children: self.children.keys().copied().collect(),
            render: self.preferred_tensor_snapshot(),
        }
    }

    fn preferred_tensor_snapshot(&self) -> Option<TrieTensorSnapshot> {
        self.value
            .as_ref()
            .and_then(|value| Self::tensor_snapshot(value, TrieTensorSource::Value))
            .or_else(|| {
                self.summary
                    .as_ref()
                    .and_then(|summary| Self::tensor_snapshot(summary, TrieTensorSource::Summary))
            })
            .or_else(|| {
                self.residual.as_ref().and_then(|residual| {
                    Self::tensor_snapshot(residual, TrieTensorSource::Residual)
                })
            })
    }

    fn tensor_snapshot(tensor: &Tensor, source: TrieTensorSource) -> Option<TrieTensorSnapshot> {
        let values = tensor
            .flatten_all()
            .ok()?
            .to_dtype(DType::F32)
            .ok()?
            .to_vec1::<f32>()
            .ok()?;
        if values.is_empty() {
            return None;
        }

        let min = values
            .iter()
            .fold(f32::INFINITY, |current, &value| current.min(value));
        let max = values
            .iter()
            .fold(f32::NEG_INFINITY, |current, &value| current.max(value));
        let sum = values.iter().sum::<f32>();
        let value_count = values.len();
        let l2_norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        let mut top_channels = values
            .iter()
            .enumerate()
            .map(|(index, &value)| TrieTopChannel { index, value })
            .collect::<Vec<_>>();
        top_channels.sort_by(|left, right| {
            right
                .value
                .abs()
                .partial_cmp(&left.value.abs())
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.index.cmp(&right.index))
        });
        top_channels.truncate(5);

        Some(TrieTensorSnapshot {
            source,
            shape: tensor_shape(values.len()),
            values,
            stats: TrieTensorStats {
                min,
                max,
                mean: sum / value_count as f32,
                l2_norm,
            },
            top_channels,
        })
    }

    fn structural_density(&self) -> usize {
        if self.populated_children > 0 {
            self.populated_children
        } else {
            self.value.is_some() as usize
        }
    }

    fn refresh_summary(&mut self) -> Result<()> {
        let mut weighted_parts = Vec::new();
        let mut total_weight = 0usize;

        if let Some(value) = &self.value {
            weighted_parts.push(value.clone());
            total_weight += 1;
        }

        for child in self.children.values() {
            if let Some(summary) = &child.summary {
                weighted_parts.push((summary * child.subtree_values as f64)?);
                total_weight += child.subtree_values;
            }
        }

        self.summary = match weighted_parts.len() {
            0 => None,
            1 if total_weight == 1 => weighted_parts.into_iter().next(),
            _ => {
                let summed = Tensor::stack(&weighted_parts, 0)?.sum(0)?;
                Some((summed / total_weight as f64)?)
            }
        };

        Ok(())
    }

    fn refresh_metadata(&mut self) -> Result<()> {
        self.populated_children = self.children.len();
        self.subtree_values = self.value.is_some() as usize
            + self
                .children
                .values()
                .map(|c| c.subtree_values)
                .sum::<usize>();
        self.refresh_summary()
    }

    fn refresh_child_residuals(&mut self) -> Result<()> {
        for child in self.children.values_mut() {
            child.residual = match (&child.summary, &self.summary) {
                (Some(child_summary), Some(parent_summary)) => {
                    Some((child_summary - parent_summary)?)
                }
                _ => None,
            };
        }
        Ok(())
    }
}

fn tensor_shape(len: usize) -> [usize; 2] {
    if len == 96 {
        [12, 8]
    } else {
        let cols = len.clamp(1, 12);
        let rows = len.div_ceil(cols);
        [cols, rows]
    }
}

/// Hierarchical L0 trie memory for the prototype.
///
/// Each node stores:
/// - an exact value for its address (optional)
/// - a coarse subtree summary
/// - a residual relative to its parent summary
/// - structural metadata for density/introspection
///
/// Writes detach their payload before storage and cascade summary/residual updates
/// back up the path so prefix reads reflect coarse-to-fine structure.
pub struct ProtoTrie {
    root: TrieNode,
    d_model: usize,
    trie_depth: usize,
}

impl ProtoTrie {
    pub fn new(config: &ZenoConfig) -> Self {
        Self {
            root: TrieNode::default(),
            d_model: config.d_model,
            trie_depth: config.trie_depth,
        }
    }

    /// Look up a single address. Returns the node summary at that prefix or zeros.
    pub fn read(&self, address: &[u8], device: &Device) -> Result<Tensor> {
        match self
            .find_node(address)
            .and_then(|node| node.summary.clone())
        {
            Some(v) => Ok(v),
            None => Tensor::zeros(self.d_model, DType::F32, device),
        }
    }

    /// Read the residual stored for an address relative to its parent summary.
    pub fn read_residual(&self, address: &[u8], device: &Device) -> Result<Tensor> {
        match self
            .find_node(address)
            .and_then(|node| node.residual.clone())
        {
            Some(v) => Ok(v),
            None => Tensor::zeros(self.d_model, DType::F32, device),
        }
    }

    /// Blended write: `stored = (1-α) × old + α × new`.
    /// The incoming value is detached before storage to break the compute graph.
    pub fn write(&mut self, address: &[u8], value: &Tensor, strength: f64) -> Result<()> {
        let depth = address.len().min(self.trie_depth);
        let detached = value.detach();
        Self::write_recursive(&mut self.root, &address[..depth], &detached, strength)?;
        self.root.residual = None;
        Ok(())
    }

    fn write_recursive(
        node: &mut TrieNode,
        address: &[u8],
        value: &Tensor,
        strength: f64,
    ) -> Result<()> {
        if address.is_empty() {
            node.value = Some(match &node.value {
                Some(old) => ((old * (1.0 - strength))? + (value * strength)?)?,
                None => value.clone(),
            });
        } else {
            let child = node.children.entry(address[0]).or_default();
            Self::write_recursive(child, &address[1..], value, strength)?;
        }

        node.refresh_metadata()?;
        node.refresh_child_residuals()?;
        Ok(())
    }

    /// Structural density at a prefix.
    ///
    /// Internal nodes return their populated child count. Exact leaves return `1`.
    pub fn density_at_prefix(&self, prefix: &[u8]) -> usize {
        self.find_node(prefix)
            .map(TrieNode::structural_density)
            .unwrap_or(0)
    }

    /// For address `[a0, a1, …, aN]` return
    /// `[density([a0]), density([a0,a1]), …, density([a0,…,aN])]`.
    pub fn density_chain(&self, address: &[u8]) -> Vec<usize> {
        let depth = address.len().min(self.trie_depth);
        (1..=depth)
            .map(|i| self.density_at_prefix(&address[..i]))
            .collect()
    }

    /// Read multiple addresses and stack into `[n_addresses, d_model]`.
    pub fn batch_read(&self, addresses: &[Vec<u8>], device: &Device) -> Result<Tensor> {
        let vecs: Vec<Tensor> = addresses
            .iter()
            .map(|a| self.read(a, device))
            .collect::<Result<Vec<_>>>()?;
        Tensor::stack(&vecs, 0)
    }

    pub fn snapshot(&self, address: &[u8]) -> Option<TrieNodeSnapshot> {
        self.find_node(address)
            .map(|node| node.snapshot(address.to_vec()))
    }

    pub fn child_snapshots(&self, address: &[u8]) -> Vec<TrieNodeSnapshot> {
        self.find_node(address)
            .map(|node| {
                let mut child_address = address.to_vec();
                node.children
                    .iter()
                    .map(|(&byte, child)| {
                        child_address.push(byte);
                        let snapshot = child.snapshot(child_address.clone());
                        child_address.pop();
                        snapshot
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Capture a sparse, populated-only snapshot around the provided focus addresses.
    ///
    /// The snapshot includes:
    /// - all populated ancestor nodes from the root to each focus address
    /// - descendants beneath each populated focus node up to `descendant_depth`
    /// - only populated nodes/edges, never the full 256-way fanout
    pub fn sparse_snapshot(
        &self,
        focus_addresses: &[Vec<u8>],
        descendant_depth: usize,
    ) -> SparseTrieSnapshot {
        let mut node_addresses = BTreeSet::new();
        let mut focus = Vec::new();

        for address in focus_addresses {
            for depth in 0..=address.len().min(self.trie_depth) {
                let prefix = address[..depth].to_vec();
                if self.find_node(&prefix).is_some() {
                    node_addresses.insert(prefix.clone());
                } else {
                    break;
                }
            }

            if self.find_node(address).is_some() {
                focus.push(address.clone());
            }
        }

        let mut descendant_seen = BTreeSet::new();
        for root in &focus {
            let mut queue = VecDeque::from([(root.clone(), 0usize)]);
            while let Some((address, rel_depth)) = queue.pop_front() {
                if !descendant_seen.insert(address.clone()) {
                    continue;
                }
                node_addresses.insert(address.clone());
                if rel_depth >= descendant_depth {
                    continue;
                }
                for child in self.child_snapshots(&address) {
                    queue.push_back((child.address, rel_depth + 1));
                }
            }
        }

        let nodes: Vec<TrieNodeSnapshot> = node_addresses
            .iter()
            .filter_map(|address| self.snapshot(address))
            .collect();
        let node_set: BTreeSet<Vec<u8>> = nodes.iter().map(|node| node.address.clone()).collect();

        let mut edges = Vec::new();
        for node in &nodes {
            for &branch in &node.children {
                let mut child = node.address.clone();
                child.push(branch);
                if node_set.contains(&child) {
                    edges.push(TrieEdgeSnapshot {
                        parent: node.address.clone(),
                        child,
                        branch,
                    });
                }
            }
        }

        SparseTrieSnapshot {
            focus,
            nodes,
            edges,
        }
    }

    /// Clear all entries.
    pub fn reset(&mut self) {
        self.root = TrieNode::default();
    }

    /// Number of exact values stored in the trie.
    pub fn len(&self) -> usize {
        self.root.subtree_values
    }

    /// Whether the trie is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn find_node(&self, address: &[u8]) -> Option<&TrieNode> {
        let depth = address.len().min(self.trie_depth);
        let mut current = &self.root;
        for &byte in &address[..depth] {
            current = current.children.get(&byte)?;
        }
        Some(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ZenoConfig {
        ZenoConfig::default_proto()
    }

    fn scalar(value: &Tensor) -> f32 {
        value.to_scalar().unwrap()
    }

    fn mean(value: &Tensor) -> f32 {
        scalar(&value.mean_all().unwrap())
    }

    #[test]
    fn read_missing_returns_zeros() {
        let trie = ProtoTrie::new(&cfg());
        let v = trie.read(&[0, 1, 2], &Device::Cpu).unwrap();
        assert_eq!(v.dims(), &[96]);
        assert_eq!(scalar(&v.sum_all().unwrap()), 0.0);
    }

    #[test]
    fn write_then_read_leaf() {
        let mut trie = ProtoTrie::new(&cfg());
        let val = Tensor::ones(96, DType::F32, &Device::Cpu).unwrap();
        trie.write(&[1, 2, 3], &val, 1.0).unwrap();

        let out = trie.read(&[1, 2, 3], &Device::Cpu).unwrap();
        assert!((scalar(&out.sum_all().unwrap()) - 96.0).abs() < 1e-5);
    }

    #[test]
    fn internal_reads_return_coarse_summaries() {
        let mut trie = ProtoTrie::new(&cfg());
        let ones = Tensor::ones(96, DType::F32, &Device::Cpu).unwrap();
        let threes = (Tensor::ones(96, DType::F32, &Device::Cpu).unwrap() * 3.0).unwrap();

        trie.write(&[1, 2, 3], &ones, 1.0).unwrap();
        trie.write(&[1, 2, 4], &threes, 1.0).unwrap();

        let branch = trie.read(&[1, 2], &Device::Cpu).unwrap();
        let root_branch = trie.read(&[1], &Device::Cpu).unwrap();

        assert!((mean(&branch) - 2.0).abs() < 1e-5);
        assert!((mean(&root_branch) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn parent_updates_cascade_after_overwrite() {
        let mut trie = ProtoTrie::new(&cfg());
        let ones = Tensor::ones(96, DType::F32, &Device::Cpu).unwrap();
        let threes = (Tensor::ones(96, DType::F32, &Device::Cpu).unwrap() * 3.0).unwrap();
        let fives = (Tensor::ones(96, DType::F32, &Device::Cpu).unwrap() * 5.0).unwrap();

        trie.write(&[1, 2, 3], &ones, 1.0).unwrap();
        trie.write(&[1, 2, 4], &threes, 1.0).unwrap();
        assert!((mean(&trie.read(&[1], &Device::Cpu).unwrap()) - 2.0).abs() < 1e-5);

        trie.write(&[1, 2, 3], &fives, 1.0).unwrap();
        assert!((mean(&trie.read(&[1, 2], &Device::Cpu).unwrap()) - 4.0).abs() < 1e-5);
        assert!((mean(&trie.read(&[1], &Device::Cpu).unwrap()) - 4.0).abs() < 1e-5);
    }

    #[test]
    fn residual_tracks_parent_relative_signal() {
        let mut trie = ProtoTrie::new(&cfg());
        let ones = Tensor::ones(96, DType::F32, &Device::Cpu).unwrap();
        let threes = (Tensor::ones(96, DType::F32, &Device::Cpu).unwrap() * 3.0).unwrap();

        trie.write(&[7, 0], &ones, 1.0).unwrap();
        trie.write(&[7, 1], &threes, 1.0).unwrap();

        let residual = trie.read_residual(&[7, 1], &Device::Cpu).unwrap();
        assert!((mean(&residual) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn density_chain_uses_structural_density() {
        let mut trie = ProtoTrie::new(&cfg());
        let v = Tensor::zeros(96, DType::F32, &Device::Cpu).unwrap();

        trie.write(&[1, 0], &v, 1.0).unwrap();
        trie.write(&[1, 1], &v, 1.0).unwrap();
        trie.write(&[1, 1, 5], &v, 1.0).unwrap();

        assert_eq!(trie.density_at_prefix(&[1]), 2);
        assert_eq!(trie.density_at_prefix(&[1, 1]), 1);
        assert_eq!(trie.density_at_prefix(&[1, 1, 5]), 1);
        assert_eq!(trie.density_chain(&[1, 1, 5]), vec![2, 1, 1]);
    }

    #[test]
    fn batch_read_shape() {
        let mut trie = ProtoTrie::new(&cfg());
        let v = Tensor::ones(96, DType::F32, &Device::Cpu).unwrap();
        trie.write(&[0], &v, 1.0).unwrap();

        let out = trie
            .batch_read(&[vec![0], vec![1], vec![2]], &Device::Cpu)
            .unwrap();
        assert_eq!(out.dims(), &[3, 96]);
    }

    #[test]
    fn snapshots_expose_sparse_structure() {
        let mut trie = ProtoTrie::new(&cfg());
        let v = Tensor::ones(96, DType::F32, &Device::Cpu).unwrap();

        trie.write(&[3, 4], &v, 1.0).unwrap();
        trie.write(&[3, 8, 1], &v, 1.0).unwrap();

        let snapshot = trie.snapshot(&[3]).unwrap();
        assert_eq!(snapshot.address, vec![3]);
        assert_eq!(snapshot.depth, 1);
        assert_eq!(snapshot.populated_children, 2);
        assert_eq!(snapshot.subtree_values, 2);
        assert_eq!(snapshot.children, vec![4, 8]);

        let children = trie.child_snapshots(&[3]);
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].address, vec![3, 4]);
        assert!(children[0].has_value);
        assert_eq!(children[1].address, vec![3, 8]);
        assert!(!children[1].has_value);
        assert!(children[1].has_summary);
    }

    #[test]
    fn reset_clears() {
        let mut trie = ProtoTrie::new(&cfg());
        let v = Tensor::zeros(96, DType::F32, &Device::Cpu).unwrap();
        trie.write(&[0], &v, 1.0).unwrap();
        assert_eq!(trie.len(), 1);
        trie.reset();
        assert!(trie.is_empty());
        assert!(trie.snapshot(&[0]).is_none());
    }

    #[test]
    fn sparse_snapshot_captures_paths_and_local_children() {
        let mut trie = ProtoTrie::new(&cfg());
        let v = Tensor::ones(96, DType::F32, &Device::Cpu).unwrap();

        trie.write(&[1, 2, 3], &v, 1.0).unwrap();
        trie.write(&[1, 2, 4], &v, 1.0).unwrap();
        trie.write(&[9, 7], &v, 1.0).unwrap();

        let sparse = trie.sparse_snapshot(&[vec![1, 2]], 1);
        let addresses: Vec<Vec<u8>> = sparse
            .nodes
            .iter()
            .map(|node| node.address.clone())
            .collect();

        assert_eq!(sparse.focus, vec![vec![1, 2]]);
        assert_eq!(
            addresses,
            vec![vec![], vec![1], vec![1, 2], vec![1, 2, 3], vec![1, 2, 4]]
        );
        assert_eq!(sparse.edges.len(), 4);
        assert!(sparse
            .edges
            .iter()
            .any(|edge| edge.parent == vec![1, 2] && edge.child == vec![1, 2, 3]));
        assert!(sparse
            .edges
            .iter()
            .all(|edge| edge.parent.first().copied() != Some(9)));
    }

    #[test]
    fn node_snapshots_include_render_payload() {
        let mut trie = ProtoTrie::new(&cfg());
        let values = (0..96).map(|value| value as f32).collect::<Vec<_>>();
        let tensor = Tensor::from_vec(values.clone(), 96, &Device::Cpu).unwrap();

        trie.write(&[4, 2], &tensor, 1.0).unwrap();

        let snapshot = trie.snapshot(&[4, 2]).unwrap();
        let render = snapshot.render.expect("render payload");
        assert_eq!(render.source, TrieTensorSource::Value);
        assert_eq!(render.shape, [12, 8]);
        assert_eq!(render.values, values);
        assert_eq!(render.top_channels[0].index, 95);
        assert_eq!(render.top_channels[0].value, 95.0);
    }

    #[test]
    fn internal_snapshots_fall_back_to_summary_payloads() {
        let mut trie = ProtoTrie::new(&cfg());
        let ones = Tensor::ones(96, DType::F32, &Device::Cpu).unwrap();
        let threes = (Tensor::ones(96, DType::F32, &Device::Cpu).unwrap() * 3.0).unwrap();

        trie.write(&[1, 2, 3], &ones, 1.0).unwrap();
        trie.write(&[1, 2, 4], &threes, 1.0).unwrap();

        let snapshot = trie.snapshot(&[1, 2]).unwrap();
        let render = snapshot.render.expect("summary payload");
        assert_eq!(render.source, TrieTensorSource::Summary);
        assert!((render.stats.mean - 2.0).abs() < 1e-5);
    }
}
