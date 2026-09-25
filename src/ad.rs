//! Eager tape for the host tensor model.
//!
//! A straight-line network is a composition of geometric tensor operations.
//! Evaluating the tape is the forward pass. The reverse sweep is the cotangent
//! (adjoint) that stochastic gradient descent consumes. Routing decisions are
//! recorded as constants, which is the coproduct elimination: the branch that
//! ran receives the cotangent and the others stay zero.

use std::f32::consts::PI;

#[derive(Clone, Debug)]
enum Op {
    Leaf,
    Add(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Scale(usize, f32),
    Matmul(usize, usize),
    RmsNorm {
        x: usize,
        w: usize,
        eps: f32,
    },
    Silu(usize),
    SoftmaxLast(usize),
    WindowSoftmax {
        x: usize,
        window: usize,
    },
    Rope {
        x: usize,
        pos_axis: usize,
        cos: Vec<f32>,
        sin: Vec<f32>,
    },
    Concat {
        xs: Vec<usize>,
        dim: usize,
    },
    Slice {
        x: usize,
        dim: usize,
        start: usize,
        len: usize,
    },
    Reshape(usize),
    Permute {
        x: usize,
        axes: Vec<usize>,
    },
    RepeatKv {
        x: usize,
        n_rep: usize,
    },
    Embed {
        table: usize,
        ids: Vec<u32>,
        batch: usize,
        seq: usize,
    },
    CrossEntropy {
        logits: usize,
        targets: Vec<u32>,
    },
    MeanAxis0(usize),
    SumLast(usize),
    DotConst {
        x: usize,
        c: Vec<f32>,
    },
}

struct Node {
    shape: Vec<usize>,
    value: Vec<f32>,
    grad: Vec<f32>,
    op: Op,
}

pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn shape(&self, id: usize) -> &[usize] {
        &self.nodes[id].shape
    }

    pub fn value(&self, id: usize) -> &[f32] {
        &self.nodes[id].value
    }

    pub fn grad(&self, id: usize) -> &[f32] {
        &self.nodes[id].grad
    }

    pub fn scalar(&self, id: usize) -> f32 {
        assert_eq!(self.nodes[id].value.len(), 1, "not a scalar");
        self.nodes[id].value[0]
    }

    pub fn leaf(&mut self, shape: &[usize], value: Vec<f32>) -> usize {
        assert_eq!(value.len(), numel(shape), "leaf length");
        self.push(shape.to_vec(), value, Op::Leaf)
    }

    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let (v, s) = ewise(&self.nodes[a], &self.nodes[b], |x, y| x + y);
        self.push(s, v, Op::Add(a, b))
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let (v, s) = ewise(&self.nodes[a], &self.nodes[b], |x, y| x * y);
        self.push(s, v, Op::Mul(a, b))
    }

    pub fn div(&mut self, a: usize, b: usize) -> usize {
        let (v, s) = ewise(&self.nodes[a], &self.nodes[b], |x, y| x / y);
        self.push(s, v, Op::Div(a, b))
    }

    pub fn scale(&mut self, a: usize, s: f32) -> usize {
        let v: Vec<f32> = self.nodes[a].value.iter().map(|x| x * s).collect();
        let shape = self.nodes[a].shape.clone();
        self.push(shape, v, Op::Scale(a, s))
    }

    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
        let (v, s) = matmul_values(
            &self.nodes[a].value,
            &self.nodes[a].shape,
            &self.nodes[b].value,
            &self.nodes[b].shape,
        );
        self.push(s, v, Op::Matmul(a, b))
    }

    pub fn rmsnorm(&mut self, x: usize, w: usize, eps: f32) -> usize {
        let v = rmsnorm_values(&self.nodes[x].value, &self.nodes[x].shape, &self.nodes[w].value, eps);
        let shape = self.nodes[x].shape.clone();
        self.push(shape, v, Op::RmsNorm { x, w, eps })
    }

    pub fn silu(&mut self, x: usize) -> usize {
        let v: Vec<f32> = self.nodes[x].value.iter().copied().map(silu).collect();
        let shape = self.nodes[x].shape.clone();
        self.push(shape, v, Op::Silu(x))
    }

    pub fn softmax_last(&mut self, x: usize) -> usize {
        let v = softmax_last_values(&self.nodes[x].value, &self.nodes[x].shape);
        let shape = self.nodes[x].shape.clone();
        self.push(shape, v, Op::SoftmaxLast(x))
    }

    /// Causal sliding window: key `k` is visible to query `q` when `k <= q` and `q - k < window`.
    pub fn window_softmax(&mut self, x: usize, window: usize) -> usize {
        let v = window_softmax_values(&self.nodes[x].value, &self.nodes[x].shape, window);
        let shape = self.nodes[x].shape.clone();
        self.push(shape, v, Op::WindowSoftmax { x, window })
    }

    pub fn rope(&mut self, x: usize, pos_axis: usize, theta: f32) -> usize {
        let shape = self.nodes[x].shape.clone();
        let dim = *shape.last().expect("rope rank");
        let n_pos = shape[pos_axis];
        let (cos, sin) = rope_tables(n_pos, dim, theta);
        let v = rotate(&self.nodes[x].value, &shape, pos_axis, &cos, &sin, false);
        self.push(shape, v, Op::Rope { x, pos_axis, cos, sin })
    }

    pub fn concat(&mut self, xs: &[usize], dim: usize) -> usize {
        let shapes: Vec<Vec<usize>> = xs.iter().map(|i| self.nodes[*i].shape.clone()).collect();
        let values: Vec<&[f32]> = xs.iter().map(|i| self.nodes[*i].value.as_slice()).collect();
        let (v, s) = concat_values(&values, &shapes, dim);
        self.push(s, v, Op::Concat { xs: xs.to_vec(), dim })
    }

    pub fn slice(&mut self, x: usize, dim: usize, start: usize, len: usize) -> usize {
        let (v, s) = slice_values(&self.nodes[x].value, &self.nodes[x].shape, dim, start, len);
        self.push(s, v, Op::Slice { x, dim, start, len })
    }

    pub fn reshape(&mut self, x: usize, shape: &[usize]) -> usize {
        assert_eq!(numel(&self.nodes[x].shape), numel(shape), "reshape numel");
        let v = self.nodes[x].value.clone();
        self.push(shape.to_vec(), v, Op::Reshape(x))
    }

    /// `axes[new] = old`.
    pub fn permute(&mut self, x: usize, axes: &[usize]) -> usize {
        let (v, s) = permute_values(&self.nodes[x].value, &self.nodes[x].shape, axes);
        self.push(s, v, Op::Permute { x, axes: axes.to_vec() })
    }

    /// Repeat key/value heads. Input `[B, H, T, D]` becomes `[B, H * n_rep, T, D]`,
    /// with the `n_rep` copies of each head adjacent.
    pub fn repeat_kv(&mut self, x: usize, n_rep: usize) -> usize {
        let shape = &self.nodes[x].shape;
        assert_eq!(shape.len(), 4, "repeat_kv rank");
        let (b, h, t, d) = (shape[0], shape[1], shape[2], shape[3]);
        let mut y = vec![0f32; b * h * n_rep * t * d];
        let src = &self.nodes[x].value;
        for bi in 0..b {
            for hi in 0..h {
                for r in 0..n_rep {
                    let oh = hi * n_rep + r;
                    for ti in 0..t {
                        let s0 = ((bi * h + hi) * t + ti) * d;
                        let d0 = ((bi * (h * n_rep) + oh) * t + ti) * d;
                        y[d0..d0 + d].copy_from_slice(&src[s0..s0 + d]);
                    }
                }
            }
        }
        let out = vec![b, h * n_rep, t, d];
        self.push(out, y, Op::RepeatKv { x, n_rep })
    }

    pub fn embed(&mut self, table: usize, ids: &[u32], batch: usize, seq: usize) -> usize {
        let dim = self.nodes[table].shape[1];
        assert_eq!(ids.len(), batch * seq, "embed ids");
        let mut y = vec![0f32; batch * seq * dim];
        let tab = &self.nodes[table].value;
        for (i, id) in ids.iter().enumerate() {
            let row = *id as usize * dim;
            y[i * dim..(i + 1) * dim].copy_from_slice(&tab[row..row + dim]);
        }
        self.push(
            vec![batch, seq, dim],
            y,
            Op::Embed {
                table,
                ids: ids.to_vec(),
                batch,
                seq,
            },
        )
    }

    pub fn cross_entropy(&mut self, logits: usize, targets: &[u32]) -> usize {
        let v = self.nodes[logits].shape[1];
        assert_eq!(self.nodes[logits].value.len(), targets.len() * v, "ce shape");
        let (loss, _) = cross_entropy(&self.nodes[logits].value, v, targets);
        self.push(
            vec![],
            vec![loss],
            Op::CrossEntropy {
                logits,
                targets: targets.to_vec(),
            },
        )
    }

    pub fn mean_axis0(&mut self, x: usize) -> usize {
        let shape = &self.nodes[x].shape;
        assert_eq!(shape.len(), 2, "mean_axis0");
        let (n, e) = (shape[0], shape[1]);
        let mut y = vec![0f32; e];
        let src = &self.nodes[x].value;
        for i in 0..n {
            for j in 0..e {
                y[j] += src[i * e + j];
            }
        }
        let inv = 1.0 / n as f32;
        for z in &mut y {
            *z *= inv;
        }
        self.push(vec![e], y, Op::MeanAxis0(x))
    }

    /// Sum the last axis and keep it as length 1, so `[N, E]` becomes `[N, 1]`.
    pub fn sum_last(&mut self, x: usize) -> usize {
        let shape = &self.nodes[x].shape;
        let d = *shape.last().expect("sum_last");
        let rows = self.nodes[x].value.len() / d;
        let mut y = vec![0f32; rows];
        let src = &self.nodes[x].value;
        for r in 0..rows {
            let mut s = 0f32;
            for c in 0..d {
                s += src[r * d + c];
            }
            y[r] = s;
        }
        let mut osh = shape.clone();
        *osh.last_mut().unwrap() = 1;
        self.push(osh, y, Op::SumLast(x))
    }

    pub fn dot_const(&mut self, x: usize, c: &[f32]) -> usize {
        assert_eq!(self.nodes[x].value.len(), c.len(), "dot length");
        let s = self.nodes[x]
            .value
            .iter()
            .zip(c)
            .map(|(a, b)| a * b)
            .sum();
        self.push(vec![], vec![s], Op::DotConst { x, c: c.to_vec() })
    }

    pub fn backward(&mut self, loss: usize) {
        assert_eq!(self.nodes[loss].value.len(), 1, "loss is a scalar");
        self.backward_cotangent(loss, &[1.0]);
    }

    /// Seed `id` with an upstream cotangent and sweep the tape.
    /// Parameter updates read the resulting leaf gradients.
    pub fn backward_cotangent(&mut self, id: usize, cot: &[f32]) {
        assert_eq!(cot.len(), self.nodes[id].value.len(), "cotangent width");
        for n in &mut self.nodes {
            n.grad = vec![0.0; n.value.len()];
        }
        self.nodes[id].grad.copy_from_slice(cot);
        for i in (0..self.nodes.len()).rev() {
            self.back_one(i);
        }
    }

    /// Bytes of values and gradients currently held on the tape.
    pub fn bytes(&self) -> u64 {
        self.nodes
            .iter()
            .map(|n| (n.value.len() + n.grad.len()) as u64 * 4)
            .sum()
    }

    fn push(&mut self, shape: Vec<usize>, value: Vec<f32>, op: Op) -> usize {
        assert_eq!(value.len(), numel(&shape), "value/shape");
        let id = self.nodes.len();
        self.nodes.push(Node {
            shape,
            value,
            grad: Vec::new(),
            op,
        });
        id
    }

    fn back_one(&mut self, i: usize) {
        let grad = self.nodes[i].grad.clone();
        if grad.iter().all(|g| *g == 0.0) {
            return;
        }
        match self.nodes[i].op.clone() {
            Op::Leaf => {}
            Op::Add(a, b) => {
                let ga = unbroadcast(&grad, &self.nodes[i].shape, &self.nodes[a].shape);
                let gb = unbroadcast(&grad, &self.nodes[i].shape, &self.nodes[b].shape);
                acc(&mut self.nodes[a].grad, &ga);
                acc(&mut self.nodes[b].grad, &gb);
            }
            Op::Mul(a, b) => {
                let si = self.nodes[i].shape.clone();
                let sa = self.nodes[a].shape.clone();
                let sb = self.nodes[b].shape.clone();
                let av = broadcast_to(&self.nodes[a].value, &sa, &si);
                let bv = broadcast_to(&self.nodes[b].value, &sb, &si);
                let ga_b: Vec<f32> = grad.iter().zip(&bv).map(|(g, y)| g * y).collect();
                let gb_b: Vec<f32> = grad.iter().zip(&av).map(|(g, x)| g * x).collect();
                let ga = unbroadcast(&ga_b, &si, &sa);
                let gb = unbroadcast(&gb_b, &si, &sb);
                acc(&mut self.nodes[a].grad, &ga);
                acc(&mut self.nodes[b].grad, &gb);
            }
            Op::Div(a, b) => {
                let si = self.nodes[i].shape.clone();
                let sa = self.nodes[a].shape.clone();
                let sb = self.nodes[b].shape.clone();
                let av = broadcast_to(&self.nodes[a].value, &sa, &si);
                let bv = broadcast_to(&self.nodes[b].value, &sb, &si);
                let ga_b: Vec<f32> = grad.iter().zip(&bv).map(|(g, y)| g / y).collect();
                let gb_b: Vec<f32> = grad
                    .iter()
                    .zip(&av)
                    .zip(&bv)
                    .map(|((g, x), y)| -g * x / (y * y))
                    .collect();
                let ga = unbroadcast(&ga_b, &si, &sa);
                let gb = unbroadcast(&gb_b, &si, &sb);
                acc(&mut self.nodes[a].grad, &ga);
                acc(&mut self.nodes[b].grad, &gb);
            }
            Op::Scale(a, s) => {
                let g: Vec<f32> = grad.iter().map(|x| x * s).collect();
                acc(&mut self.nodes[a].grad, &g);
            }
            Op::Matmul(a, b) => {
                let (da, db) = matmul_backward(
                    &self.nodes[a].value,
                    &self.nodes[a].shape,
                    &self.nodes[b].value,
                    &self.nodes[b].shape,
                    &grad,
                );
                acc(&mut self.nodes[a].grad, &da);
                acc(&mut self.nodes[b].grad, &db);
            }
            Op::RmsNorm { x, w, eps } => {
                let (dx, dw) = rmsnorm_backward(
                    &self.nodes[x].value,
                    &self.nodes[x].shape,
                    &self.nodes[w].value,
                    &grad,
                    eps,
                );
                acc(&mut self.nodes[x].grad, &dx);
                acc(&mut self.nodes[w].grad, &dw);
            }
            Op::Silu(x) => {
                let g: Vec<f32> = self.nodes[x]
                    .value
                    .iter()
                    .zip(&grad)
                    .map(|(v, dy)| dy * silu_prime(*v))
                    .collect();
                acc(&mut self.nodes[x].grad, &g);
            }
            Op::SoftmaxLast(x) => {
                let g = softmax_backward(&self.nodes[i].value, &self.nodes[i].shape, &grad, None);
                acc(&mut self.nodes[x].grad, &g);
            }
            Op::WindowSoftmax { x, window } => {
                let g = softmax_backward(
                    &self.nodes[i].value,
                    &self.nodes[i].shape,
                    &grad,
                    Some(window),
                );
                acc(&mut self.nodes[x].grad, &g);
            }
            Op::Rope { x, pos_axis, cos, sin } => {
                let g = rotate(&grad, &self.nodes[i].shape, pos_axis, &cos, &sin, true);
                acc(&mut self.nodes[x].grad, &g);
            }
            Op::Concat { xs, dim } => {
                let mut offset = 0;
                for id in xs {
                    let len = self.nodes[id].shape[dim];
                    let g = slice_values(&grad, &self.nodes[i].shape, dim, offset, len).0;
                    acc(&mut self.nodes[id].grad, &g);
                    offset += len;
                }
            }
            Op::Slice { x, dim, start, len } => {
                let parent = self.nodes[x].shape.clone();
                let mut full = vec![0f32; numel(&parent)];
                paste_into(&mut full, &parent, &grad, dim, start, len);
                acc(&mut self.nodes[x].grad, &full);
            }
            Op::Reshape(x) => acc(&mut self.nodes[x].grad, &grad),
            Op::Permute { x, axes } => {
                let mut inv = vec![0; axes.len()];
                for (new_a, old_a) in axes.iter().enumerate() {
                    inv[*old_a] = new_a;
                }
                let g = permute_values(&grad, &self.nodes[i].shape, &inv).0;
                acc(&mut self.nodes[x].grad, &g);
            }
            Op::RepeatKv { x, n_rep } => {
                let shape = &self.nodes[x].shape;
                let (b, h, t, d) = (shape[0], shape[1], shape[2], shape[3]);
                let mut g = vec![0f32; b * h * t * d];
                for bi in 0..b {
                    for hi in 0..h {
                        for r in 0..n_rep {
                            let oh = hi * n_rep + r;
                            for ti in 0..t {
                                let s0 = ((bi * h + hi) * t + ti) * d;
                                let d0 = ((bi * (h * n_rep) + oh) * t + ti) * d;
                                for di in 0..d {
                                    g[s0 + di] += grad[d0 + di];
                                }
                            }
                        }
                    }
                }
                acc(&mut self.nodes[x].grad, &g);
            }
            Op::Embed { table, ids, batch, seq } => {
                assert_eq!(ids.len(), batch * seq, "embed ids");
                let dim = self.nodes[table].shape[1];
                for (i, id) in ids.iter().enumerate() {
                    let row = *id as usize * dim;
                    for d in 0..dim {
                        self.nodes[table].grad[row + d] += grad[i * dim + d];
                    }
                }
            }
            Op::CrossEntropy { logits, targets } => {
                let v = self.nodes[logits].shape[1];
                let (_, g) = cross_entropy(&self.nodes[logits].value, v, &targets);
                let up = grad[0];
                let g: Vec<f32> = g.into_iter().map(|x| x * up).collect();
                acc(&mut self.nodes[logits].grad, &g);
            }
            Op::MeanAxis0(x) => {
                let (n, e) = (self.nodes[x].shape[0], self.nodes[x].shape[1]);
                let inv = 1.0 / n as f32;
                let mut g = vec![0f32; n * e];
                for i in 0..n {
                    for j in 0..e {
                        g[i * e + j] = grad[j] * inv;
                    }
                }
                acc(&mut self.nodes[x].grad, &g);
            }
            Op::SumLast(x) => {
                let d = *self.nodes[x].shape.last().unwrap();
                let rows = grad.len();
                let mut g = vec![0f32; rows * d];
                for r in 0..rows {
                    for c in 0..d {
                        g[r * d + c] = grad[r];
                    }
                }
                acc(&mut self.nodes[x].grad, &g);
            }
            Op::DotConst { x, c } => {
                let up = grad[0];
                let g: Vec<f32> = c.iter().map(|z| z * up).collect();
                acc(&mut self.nodes[x].grad, &g);
            }
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

pub fn rope_tables(n_pos: usize, dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    assert!(dim % 2 == 0, "rope dim");
    let pairs = dim / 2;
    let mut cos = vec![0f32; n_pos * pairs];
    let mut sin = vec![0f32; n_pos * pairs];
    for p in 0..pairs {
        let freq = theta.powf(-(2.0 * p as f32) / dim as f32);
        for t in 0..n_pos {
            let ang = t as f32 * freq;
            cos[t * pairs + p] = ang.cos();
            sin[t * pairs + p] = ang.sin();
        }
    }
    (cos, sin)
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn unravel(mut i: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0; shape.len()];
    for d in (0..shape.len()).rev() {
        let s = shape[d].max(1);
        idx[d] = i % s;
        i /= s;
    }
    idx
}

fn ravel(idx: &[usize], shape: &[usize]) -> usize {
    let mut o = 0;
    for (c, s) in idx.iter().zip(shape) {
        o = o * s.max(&1) + c;
    }
    o
}

fn dim_at(shape: &[usize], i: usize, rank: usize) -> usize {
    let off = rank - shape.len();
    if i < off {
        1
    } else {
        shape[i - off]
    }
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for i in 0..rank {
        let av = dim_at(a, i, rank);
        let bv = dim_at(b, i, rank);
        assert!(
            av == bv || av == 1 || bv == 1,
            "cannot broadcast {a:?} with {b:?}"
        );
        out[i] = av.max(bv);
    }
    out
}

fn batch_map(index: usize, from: &[usize], to: &[usize]) -> usize {
    if to.is_empty() {
        return 0;
    }
    let coord = if from.is_empty() {
        vec![]
    } else {
        unravel(index, from)
    };
    let rank_f = from.len();
    let rank_t = to.len();
    let mut tcoord = vec![0usize; rank_t];
    for d in 0..rank_t {
        let fd = d + rank_f - rank_t;
        let c = if fd < rank_f && !coord.is_empty() {
            coord[fd]
        } else {
            0
        };
        tcoord[d] = if to[d] == 1 { 0 } else { c };
    }
    ravel(&tcoord, to)
}

fn broadcast_to(value: &[f32], from: &[usize], to: &[usize]) -> Vec<f32> {
    if from == to {
        return value.to_vec();
    }
    let n = numel(to).max(1);
    let mut out = vec![0f32; n];
    for i in 0..n {
        out[i] = value[batch_map(i, to, from)];
    }
    out
}

fn unbroadcast(grad: &[f32], from: &[usize], to: &[usize]) -> Vec<f32> {
    if from == to {
        return grad.to_vec();
    }
    let mut out = vec![0f32; numel(to).max(1)];
    let n = numel(from).max(1);
    for i in 0..n {
        let j = batch_map(i, from, to);
        out[j] += grad[i];
    }
    out
}

fn ewise(a: &Node, b: &Node, f: impl Fn(f32, f32) -> f32) -> (Vec<f32>, Vec<usize>) {
    let shape = broadcast_shape(&a.shape, &b.shape);
    let av = broadcast_to(&a.value, &a.shape, &shape);
    let bv = broadcast_to(&b.value, &b.shape, &shape);
    let v = av.iter().zip(&bv).map(|(x, y)| f(*x, *y)).collect();
    (v, shape)
}

fn acc(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len(), "accumulate");
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

fn matmul_values(a: &[f32], ash: &[usize], b: &[f32], bsh: &[usize]) -> (Vec<f32>, Vec<usize>) {
    assert!(ash.len() >= 2 && bsh.len() >= 2, "matmul rank");
    let k = ash[ash.len() - 1];
    assert_eq!(k, bsh[bsh.len() - 2], "matmul inner");
    let m = ash[ash.len() - 2];
    let n = bsh[bsh.len() - 1];
    let cb = broadcast_shape(&ash[..ash.len() - 2], &bsh[..bsh.len() - 2]);
    let batch = numel(&cb).max(1);
    let mut out = vec![0f32; batch * m * n];
    for bi in 0..batch {
        let ai = batch_map(bi, &cb, &ash[..ash.len() - 2]);
        let bbi = batch_map(bi, &cb, &bsh[..bsh.len() - 2]);
        gemm_into(a, ai, b, bbi, &mut out, bi, m, n, k, false);
    }
    let mut shape = cb;
    shape.push(m);
    shape.push(n);
    (out, shape)
}

fn gemm_into(
    a: &[f32],
    ai: usize,
    b: &[f32],
    bi: usize,
    c: &mut [f32],
    ci: usize,
    m: usize,
    n: usize,
    k: usize,
    accumulate_ab: bool,
) {
    let _ = accumulate_ab;
    let a0 = ai * m * k;
    let b0 = bi * k * n;
    let c0 = ci * m * n;
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f32;
            for t in 0..k {
                s += a[a0 + i * k + t] * b[b0 + t * n + j];
            }
            c[c0 + i * n + j] = s;
        }
    }
}

fn matmul_backward(
    a: &[f32],
    ash: &[usize],
    b: &[f32],
    bsh: &[usize],
    dc: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let k = ash[ash.len() - 1];
    let m = ash[ash.len() - 2];
    let n = bsh[bsh.len() - 1];
    let ab = &ash[..ash.len() - 2];
    let bb = &bsh[..bsh.len() - 2];
    let cb = broadcast_shape(ab, bb);
    let batch = numel(&cb).max(1);
    let mut da = vec![0f32; a.len()];
    let mut db = vec![0f32; b.len()];
    for bi in 0..batch {
        let ai = batch_map(bi, &cb, ab);
        let bbi = batch_map(bi, &cb, bb);
        let a0 = ai * m * k;
        let b0 = bbi * k * n;
        let c0 = bi * m * n;
        for i in 0..m {
            for t in 0..k {
                let mut s = 0f32;
                for j in 0..n {
                    s += dc[c0 + i * n + j] * b[b0 + t * n + j];
                }
                da[a0 + i * k + t] += s;
            }
        }
        for t in 0..k {
            for j in 0..n {
                let mut s = 0f32;
                for i in 0..m {
                    s += a[a0 + i * k + t] * dc[c0 + i * n + j];
                }
                db[b0 + t * n + j] += s;
            }
        }
    }
    (da, db)
}

fn rmsnorm_values(x: &[f32], shape: &[usize], w: &[f32], eps: f32) -> Vec<f32> {
    let d = *shape.last().unwrap();
    assert_eq!(w.len(), d, "rmsnorm width");
    let rows = x.len() / d;
    let mut y = vec![0f32; x.len()];
    for r in 0..rows {
        let mut ms = 0f32;
        for i in 0..d {
            let v = x[r * d + i];
            ms += v * v;
        }
        ms /= d as f32;
        let inv = (ms + eps).sqrt().recip();
        for i in 0..d {
            y[r * d + i] = x[r * d + i] * inv * w[i];
        }
    }
    y
}

fn rmsnorm_backward(x: &[f32], shape: &[usize], w: &[f32], dy: &[f32], eps: f32) -> (Vec<f32>, Vec<f32>) {
    let d = *shape.last().unwrap();
    let rows = x.len() / d;
    let mut dx = vec![0f32; x.len()];
    let mut dw = vec![0f32; d];
    for r in 0..rows {
        let mut ms = 0f32;
        for i in 0..d {
            let v = x[r * d + i];
            ms += v * v;
        }
        ms /= d as f32;
        let inv = (ms + eps).sqrt().recip();
        let mut dot = 0f32;
        for i in 0..d {
            let dxhat = dy[r * d + i] * w[i];
            dw[i] += x[r * d + i] * inv * dy[r * d + i];
            dot += x[r * d + i] * dxhat;
        }
        for i in 0..d {
            let dxhat = dy[r * d + i] * w[i];
            dx[r * d + i] = inv * (dxhat - x[r * d + i] * dot / (d as f32 * (ms + eps)));
        }
    }
    (dx, dw)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn silu_prime(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

fn softmax_last_values(x: &[f32], shape: &[usize]) -> Vec<f32> {
    let d = *shape.last().unwrap();
    let rows = x.len() / d;
    let mut y = vec![0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * d..(r + 1) * d];
        let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for i in 0..d {
            let e = (row[i] - maxv).exp();
            y[r * d + i] = e;
            sum += e;
        }
        for i in 0..d {
            y[r * d + i] /= sum;
        }
    }
    y
}

fn allow_key(q: usize, k: usize, window: usize) -> bool {
    k <= q && q - k < window
}

fn window_softmax_values(x: &[f32], shape: &[usize], window: usize) -> Vec<f32> {
    assert!(shape.len() >= 2, "window softmax rank");
    let kdim = shape[shape.len() - 1];
    let qdim = shape[shape.len() - 2];
    let rows = x.len() / (qdim * kdim);
    let mut y = vec![0f32; x.len()];
    for r in 0..rows {
        for q in 0..qdim {
            let base = (r * qdim + q) * kdim;
            let mut maxv = f32::NEG_INFINITY;
            for k in 0..kdim {
                if allow_key(q, k, window) {
                    maxv = maxv.max(x[base + k]);
                }
            }
            let mut sum = 0f32;
            for k in 0..kdim {
                if allow_key(q, k, window) {
                    let e = (x[base + k] - maxv).exp();
                    y[base + k] = e;
                    sum += e;
                }
            }
            for k in 0..kdim {
                if allow_key(q, k, window) {
                    y[base + k] /= sum;
                }
            }
        }
    }
    y
}

fn softmax_backward(y: &[f32], shape: &[usize], dy: &[f32], window: Option<usize>) -> Vec<f32> {
    let d = *shape.last().unwrap();
    let rows_outer = if shape.len() >= 2 {
        y.len() / (shape[shape.len() - 2] * d)
    } else {
        1
    };
    let qdim = if shape.len() >= 2 { shape[shape.len() - 2] } else { 1 };
    let mut dx = vec![0f32; y.len()];
    match window {
        None => {
            let rows = y.len() / d;
            for r in 0..rows {
                let mut dot = 0f32;
                for i in 0..d {
                    dot += y[r * d + i] * dy[r * d + i];
                }
                for i in 0..d {
                    let p = y[r * d + i];
                    dx[r * d + i] = p * (dy[r * d + i] - dot);
                }
            }
        }
        Some(window) => {
            for r in 0..rows_outer {
                for q in 0..qdim {
                    let base = (r * qdim + q) * d;
                    let mut dot = 0f32;
                    for k in 0..d {
                        if allow_key(q, k, window) {
                            dot += y[base + k] * dy[base + k];
                        }
                    }
                    for k in 0..d {
                        if allow_key(q, k, window) {
                            let p = y[base + k];
                            dx[base + k] = p * (dy[base + k] - dot);
                        }
                    }
                }
            }
        }
    }
    dx
}

fn rotate(
    x: &[f32],
    shape: &[usize],
    pos_axis: usize,
    cos: &[f32],
    sin: &[f32],
    inverse: bool,
) -> Vec<f32> {
    let dim = *shape.last().unwrap();
    let pairs = dim / 2;
    let mut y = vec![0f32; x.len()];
    for i in (0..x.len()).step_by(2) {
        let coord = unravel(i, shape);
        let t = coord[pos_axis];
        let p = coord[shape.len() - 1] / 2;
        let c = cos[t * pairs + p];
        let s = sin[t * pairs + p];
        let x0 = x[i];
        let x1 = x[i + 1];
        if inverse {
            y[i] = x0 * c + x1 * s;
            y[i + 1] = -x0 * s + x1 * c;
        } else {
            y[i] = x0 * c - x1 * s;
            y[i + 1] = x0 * s + x1 * c;
        }
    }
    y
}

fn concat_values(xs: &[&[f32]], shapes: &[Vec<usize>], dim: usize) -> (Vec<f32>, Vec<usize>) {
    let mut shape = shapes[0].clone();
    shape[dim] = shapes.iter().map(|s| s[dim]).sum();
    let mut out = vec![0f32; numel(&shape)];
    let mut start = 0;
    for (v, s) in xs.iter().zip(shapes) {
        paste_into(&mut out, &shape, v, dim, start, s[dim]);
        start += s[dim];
    }
    (out, shape)
}

fn slice_values(x: &[f32], shape: &[usize], dim: usize, start: usize, len: usize) -> (Vec<f32>, Vec<usize>) {
    let mut osh = shape.to_vec();
    osh[dim] = len;
    let mut out = vec![0f32; numel(&osh)];
    // Copy elements whose coordinate on `dim` lies in the slice.
    for i in 0..x.len() {
        let mut coord = unravel(i, shape);
        if coord[dim] >= start && coord[dim] < start + len {
            coord[dim] -= start;
            let j = ravel(&coord, &osh);
            out[j] = x[i];
        }
    }
    (out, osh)
}

fn paste_into(dst: &mut [f32], dst_shape: &[usize], src: &[f32], dim: usize, start: usize, len: usize) {
    let mut src_shape = dst_shape.to_vec();
    src_shape[dim] = len;
    for i in 0..src.len() {
        let mut coord = unravel(i, &src_shape);
        coord[dim] += start;
        let j = ravel(&coord, dst_shape);
        dst[j] = src[i];
    }
}

fn permute_values(x: &[f32], shape: &[usize], axes: &[usize]) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(axes.len(), shape.len(), "permute rank");
    let out_shape: Vec<usize> = axes.iter().map(|a| shape[*a]).collect();
    let mut y = vec![0f32; x.len()];
    for i in 0..x.len() {
        let coord = unravel(i, shape);
        let mut oc = vec![0; shape.len()];
        for a in 0..shape.len() {
            oc[a] = coord[axes[a]];
        }
        y[ravel(&oc, &out_shape)] = x[i];
    }
    (y, out_shape)
}

fn cross_entropy(logits: &[f32], v: usize, targets: &[u32]) -> (f32, Vec<f32>) {
    let n = targets.len();
    let mut loss = 0f32;
    let mut grad = vec![0f32; n * v];
    let inv_n = 1.0 / n as f32;
    for i in 0..n {
        let row = &logits[i * v..(i + 1) * v];
        let maxv = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        let mut exps = vec![0f32; v];
        for j in 0..v {
            exps[j] = (row[j] - maxv).exp();
            sum += exps[j];
        }
        let t = targets[i] as usize;
        loss += -((row[t] - maxv) - sum.ln()) * inv_n;
        for j in 0..v {
            let p = exps[j] / sum;
            let ind = if j == t { 1.0 } else { 0.0 };
            grad[i * v + j] = (p - ind) * inv_n;
        }
    }
    (loss, grad)
}

/// Deterministic normal sampler for parameter initialisation.
pub struct Rng(pub u64);

impl Rng {
    fn u(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    fn uniform(&mut self) -> f32 {
        let x = (self.u() >> 40) as u32;
        (x as f32) / (1u32 << 24) as f32
    }

    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().clamp(1e-7, 1.0);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fd(f: impl Fn(f32) -> f32, x: f32) -> f32 {
        let e = 1e-3;
        (f(x + e) - f(x - e)) / (2.0 * e)
    }

    #[test]
    fn matmul_matches_outer_product_gradient() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [0.5f32, -1.0, 1.5, 0.25, 2.0, -0.5];
        let (c, _) = matmul_values(&a, &[2, 3], &b, &[3, 2]);
        let ones = vec![1f32; c.len()];
        let (da, db) = matmul_backward(&a, &[2, 3], &b, &[3, 2], &ones);
        // dB_kj = sum_i A_ik
        assert!((db[0] - (a[0] + a[3])).abs() < 1e-5);
        assert!((da[0] - (b[0] + b[1])).abs() < 1e-5);
    }

    #[test]
    fn broadcast_weight_gradient_sums_batch() {
        let a = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
        let b = [2.0f32, 0.0, 0.0, 3.0];
        let (c, s) = matmul_values(&a, &[2, 2, 2], &b, &[2, 2]);
        assert_eq!(s, vec![2, 2, 2]);
        let (da, db) = matmul_backward(&a, &[2, 2, 2], &b, &[2, 2], &vec![1.0; c.len()]);
        assert_eq!(da.len(), a.len());
        assert_eq!(db.len(), b.len());
        assert!(db.iter().any(|z| z.abs() > 0.0));
    }

    #[test]
    fn rmsnorm_silu_and_rope_match_finite_differences() {
        let mut g = Graph::new();
        let x = g.leaf(&[2, 4], vec![0.5, -1.0, 0.25, 1.5, -0.5, 0.75, 0.1, -0.2]);
        let w = g.leaf(&[4], vec![1.0, 0.8, 1.2, 0.9]);
        let y = g.rmsnorm(x, w, 1e-5);
        let s = g.silu(y);
        let r = g.rope(s, 0, 10_000.0);
        let flat = g.reshape(r, &[8]);
        let loss = g.dot_const(flat, &[0.2, -0.1, 0.3, 0.0, 0.5, -0.4, 0.1, 0.2]);
        g.backward(loss);
        let analytic = g.grad(x)[3];
        let base = g.value(x).to_vec();
        let numeric = fd(
            |h| {
                let mut g = Graph::new();
                let mut v = base.clone();
                v[3] = h;
                let x = g.leaf(&[2, 4], v);
                let w = g.leaf(&[4], vec![1.0, 0.8, 1.2, 0.9]);
                let y = g.rmsnorm(x, w, 1e-5);
                let s = g.silu(y);
                let r = g.rope(s, 0, 10_000.0);
                let flat = g.reshape(r, &[8]);
                let loss = g.dot_const(flat, &[0.2, -0.1, 0.3, 0.0, 0.5, -0.4, 0.1, 0.2]);
                g.scalar(loss)
            },
            base[3],
        );
        assert!((analytic - numeric).abs() < 2e-3, "{analytic} vs {numeric}");
    }

    #[test]
    fn window_softmax_blocks_far_keys() {
        let mut g = Graph::new();
        let x = g.leaf(&[1, 4, 4], vec![0.0; 16]);
        let y = g.window_softmax(x, 2);
        let v = g.value(y);
        // Query 3 may see keys 2 and 3 only.
        assert!(v[3 * 4 + 0].abs() < 1e-6);
        assert!(v[3 * 4 + 1].abs() < 1e-6);
        assert!((v[3 * 4 + 2] - 0.5).abs() < 1e-5);
        assert!((v[3 * 4 + 3] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn renorm_mask_adjoint_matches_finite_difference() {
        let logits0 = vec![0.2f32, -0.4, 0.7, 0.1];
        let mask = vec![1.0f32, 0.0, 1.0, 0.0];
        let run = |logits: &[f32]| {
            let mut g = Graph::new();
            let z = g.leaf(&[1, 4], logits.to_vec());
            let p = g.softmax_last(z);
            let m = g.leaf(&[1, 4], mask.clone());
            let masked = g.mul(p, m);
            let denom = g.sum_last(masked);
            let w = g.div(masked, denom);
            let flat = g.reshape(w, &[4]);
            let loss = g.dot_const(flat, &[0.3, 0.0, -0.2, 0.5]);
            g.scalar(loss)
        };
        let mut g = Graph::new();
        let z = g.leaf(&[1, 4], logits0.clone());
        let p = g.softmax_last(z);
        let m = g.leaf(&[1, 4], mask.clone());
        let masked = g.mul(p, m);
        let denom = g.sum_last(masked);
        let w = g.div(masked, denom);
        let flat = g.reshape(w, &[4]);
        let loss = g.dot_const(flat, &[0.3, 0.0, -0.2, 0.5]);
        g.backward(loss);
        let analytic = g.grad(z)[2];
        let numeric = fd(|h| {
            let mut logits = logits0.clone();
            logits[2] = h;
            run(&logits)
        }, logits0[2]);
        assert!((analytic - numeric).abs() < 2e-3, "{analytic} vs {numeric}");
    }
}
