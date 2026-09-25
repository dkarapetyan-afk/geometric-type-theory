//! Mixtral 8x7B, the open mixture-of-experts transformer from Jiang et al.,
//! arXiv:2401.04088, as published in `mistralai/Mixtral-8x7B-v0.1`.
//!
//! Each block is grouped-query attention with rotary embeddings and a sliding
//! window, then a top-2 router over eight SwiGLU experts. The training loss is
//! next-token cross-entropy plus the Switch load-balancing term. The adjoint
//! of that scalar is the gradient stochastic gradient descent applies.
//!
//! The published model has 46,702,788,608 parameters. The runnable network is
//! the same block at demonstration width, so the forward pass and the adjoint
//! can be checked by finite differences and a descent step.

use crate::ad::{Graph, Rng};

#[derive(Clone, Debug)]
pub struct Config {
    pub vocab: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub intermediate: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub sliding_window: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub aux_loss_coef: f32,
    pub init_std: f32,
}

impl Config {
    /// Hyperparameters of Mixtral-8x7B-v0.1.
    pub fn mixtral_8x7b() -> Self {
        Self {
            vocab: 32_000,
            dim: 4096,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 8,
            intermediate: 14_336,
            n_experts: 8,
            top_k: 2,
            sliding_window: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            aux_loss_coef: 0.02,
            init_std: 0.02,
        }
    }

    /// Same block, every expert selected, so the loss is smooth.
    pub fn demo_smooth() -> Self {
        Self {
            vocab: 48,
            dim: 32,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            intermediate: 48,
            n_experts: 4,
            top_k: 4,
            sliding_window: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            aux_loss_coef: 0.02,
            init_std: 0.02,
        }
    }

    /// Top-2 of 8 experts, the published routing ratio, at demonstration width.
    pub fn demo_sparse() -> Self {
        Self {
            vocab: 48,
            dim: 32,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            intermediate: 32,
            n_experts: 8,
            top_k: 2,
            sliding_window: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            aux_loss_coef: 0.02,
            init_std: 0.02,
        }
    }

    pub(crate) fn head_dim(&self) -> usize {
        assert!(self.dim % self.n_heads == 0, "dim divides into heads");
        self.dim / self.n_heads
    }

    fn n_rep(&self) -> usize {
        assert!(self.n_heads % self.n_kv_heads == 0, "kv heads divide query heads");
        self.n_heads / self.n_kv_heads
    }
}

#[derive(Clone, Copy)]
enum Init {
    Normal,
    Ones,
}

pub(crate) struct Slot {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    init: Init,
}

pub(crate) struct LayerSpec {
    pub(crate) attn_norm: usize,
    pub(crate) wq: usize,
    pub(crate) wk: usize,
    pub(crate) wv: usize,
    pub(crate) wo: usize,
    pub(crate) ffn_norm: usize,
    pub(crate) gate: usize,
    pub(crate) experts: Vec<[usize; 3]>,
}

pub(crate) struct Arch {
    pub(crate) embed: usize,
    pub(crate) layers: Vec<LayerSpec>,
    pub(crate) norm: usize,
    pub(crate) head: usize,
    pub(crate) slots: Vec<Slot>,
}

pub(crate) fn arch(cfg: &Config) -> Arch {
    let mut slots = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, init: Init| {
        let id = slots.len();
        slots.push(Slot { name, shape, init });
        id
    };
    let embed = add("embed".into(), vec![cfg.vocab, cfg.dim], Init::Normal);
    let hd = cfg.head_dim();
    let mut layers = Vec::new();
    for layer in 0..cfg.n_layers {
        let p = format!("layers.{layer}");
        let attn_norm = add(format!("{p}.attn_norm"), vec![cfg.dim], Init::Ones);
        let wq = add(format!("{p}.wq"), vec![cfg.dim, cfg.n_heads * hd], Init::Normal);
        let wk = add(format!("{p}.wk"), vec![cfg.dim, cfg.n_kv_heads * hd], Init::Normal);
        let wv = add(format!("{p}.wv"), vec![cfg.dim, cfg.n_kv_heads * hd], Init::Normal);
        let wo = add(format!("{p}.wo"), vec![cfg.n_heads * hd, cfg.dim], Init::Normal);
        let ffn_norm = add(format!("{p}.ffn_norm"), vec![cfg.dim], Init::Ones);
        let gate = add(format!("{p}.gate"), vec![cfg.dim, cfg.n_experts], Init::Normal);
        let mut experts = Vec::new();
        for e in 0..cfg.n_experts {
            let w1 = add(format!("{p}.experts.{e}.w1"), vec![cfg.dim, cfg.intermediate], Init::Normal);
            let w3 = add(format!("{p}.experts.{e}.w3"), vec![cfg.dim, cfg.intermediate], Init::Normal);
            let w2 = add(format!("{p}.experts.{e}.w2"), vec![cfg.intermediate, cfg.dim], Init::Normal);
            experts.push([w1, w3, w2]);
        }
        layers.push(LayerSpec {
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            ffn_norm,
            gate,
            experts,
        });
    }
    let norm = add("norm".into(), vec![cfg.dim], Init::Ones);
    let head = add("lm_head".into(), vec![cfg.dim, cfg.vocab], Init::Normal);
    Arch {
        embed,
        layers,
        norm,
        head,
        slots,
    }
}

pub fn parameter_count(cfg: &Config) -> u64 {
    arch(cfg)
        .slots
        .iter()
        .map(|s| s.shape.iter().product::<usize>() as u64)
        .sum()
}

pub fn active_parameter_count(cfg: &Config) -> u64 {
    let total = parameter_count(cfg);
    let expert: u64 = (cfg.n_layers * cfg.n_experts * 3 * cfg.dim * cfg.intermediate) as u64;
    let active_expert = expert * cfg.top_k as u64 / cfg.n_experts as u64;
    total - expert + active_expert
}

pub(crate) fn each_init(cfg: &Config, seed: u64, mut f: impl FnMut(usize, &[f32])) {
    let spec = arch(cfg);
    let mut rng = Rng(seed);
    for (i, s) in spec.slots.iter().enumerate() {
        let n: usize = s.shape.iter().product();
        let mut v = Vec::with_capacity(n);
        match s.init {
            Init::Normal => {
                for _ in 0..n {
                    v.push(rng.normal() * cfg.init_std);
                }
            }
            Init::Ones => v.resize(n, 1.0),
        }
        f(i, &v);
    }
}

fn init_params(cfg: &Config, seed: u64) -> Vec<f32> {
    let mut out = Vec::new();
    each_init(cfg, seed, |_, v| out.extend_from_slice(v));
    out
}

fn offsets(slots: &[Slot]) -> Vec<usize> {
    let mut o = Vec::with_capacity(slots.len());
    let mut c = 0;
    for s in slots {
        o.push(c);
        c += s.shape.iter().product::<usize>();
    }
    o
}

struct Bound {
    g: Graph,
    nodes: Vec<usize>,
}

fn bind(arch: &Arch, params: &[f32]) -> Bound {
    let off = offsets(&arch.slots);
    let mut g = Graph::new();
    let mut nodes = Vec::with_capacity(arch.slots.len());
    for (s, o) in arch.slots.iter().zip(&off) {
        let n: usize = s.shape.iter().product();
        nodes.push(g.leaf(&s.shape, params[*o..*o + n].to_vec()));
    }
    Bound { g, nodes }
}

pub struct Batch {
    pub tokens: Vec<u32>,
    pub batch: usize,
    pub seq: usize,
}

pub fn sample_batch(cfg: &Config, batch: usize, seq: usize, seed: u64) -> Batch {
    let mut rng = Rng(seed);
    let mut tokens = Vec::with_capacity(batch * seq);
    for _ in 0..batch * seq {
        tokens.push((rng.normal().abs() * (cfg.vocab as f32)).floor() as u32 % cfg.vocab as u32);
    }
    Batch { tokens, batch, seq }
}

pub(crate) fn swiglu(g: &mut Graph, x: usize, w1: usize, w3: usize, w2: usize) -> usize {
    let gate = g.matmul(x, w1);
    let gate = g.silu(gate);
    let up = g.matmul(x, w3);
    let hidden = g.mul(gate, up);
    g.matmul(hidden, w2)
}

pub(crate) struct AttnW {
    pub(crate) norm: usize,
    pub(crate) q: usize,
    pub(crate) k: usize,
    pub(crate) v: usize,
    pub(crate) o: usize,
}

pub(crate) fn attention(cfg: &Config, g: &mut Graph, w: AttnW, x: usize, seq: usize) -> usize {
    let hd = cfg.head_dim();
    let h = g.rmsnorm(x, w.norm, cfg.rms_norm_eps);
    let batch = g.shape(h)[0];
    let q = g.matmul(h, w.q);
    let q = g.reshape(q, &[batch, seq, cfg.n_heads, hd]);
    let k = g.matmul(h, w.k);
    let k = g.reshape(k, &[batch, seq, cfg.n_kv_heads, hd]);
    let v = g.matmul(h, w.v);
    let v = g.reshape(v, &[batch, seq, cfg.n_kv_heads, hd]);
    let q = g.rope(q, 1, cfg.rope_theta);
    let k = g.rope(k, 1, cfg.rope_theta);
    let q = g.permute(q, &[0, 2, 1, 3]);
    let k = g.permute(k, &[0, 2, 1, 3]);
    let v = g.permute(v, &[0, 2, 1, 3]);
    let k = g.repeat_kv(k, cfg.n_rep());
    let v = g.repeat_kv(v, cfg.n_rep());
    let scale = (hd as f32).sqrt().recip();
    let kt = g.permute(k, &[0, 1, 3, 2]);
    let scores = g.matmul(q, kt);
    let scores = g.scale(scores, scale);
    let window = cfg.sliding_window.max(1);
    let probs = g.window_softmax(scores, window);
    let ctx = g.matmul(probs, v);
    let ctx = g.permute(ctx, &[0, 2, 1, 3]);
    let flat = g.reshape(ctx, &[batch, seq, cfg.n_heads * hd]);
    let proj = g.matmul(flat, w.o);
    g.add(x, proj)
}

pub(crate) fn topk_mask(probs: &[f32], n: usize, experts: usize, k: usize) -> (Vec<f32>, Vec<u32>) {
    let mut mask = vec![0f32; n * experts];
    let mut counts = vec![0u32; experts];
    for row in 0..n {
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&i, &j| {
            probs[row * experts + j]
                .partial_cmp(&probs[row * experts + i])
                .unwrap()
        });
        for &e in order.iter().take(k) {
            mask[row * experts + e] = 1.0;
            counts[e] += 1;
        }
    }
    (mask, counts)
}

pub(crate) struct Router {
    pub flat: usize,
    pub probs: usize,
    pub weights: usize,
    pub counts: Vec<u32>,
}

pub(crate) fn router(cfg: &Config, g: &mut Graph, norm: usize, gate: usize, x: usize) -> Router {
    let batch = g.shape(x)[0];
    let seq = g.shape(x)[1];
    let ntok = batch * seq;
    let h = g.rmsnorm(x, norm, cfg.rms_norm_eps);
    let flat = g.reshape(h, &[ntok, cfg.dim]);
    let logits = g.matmul(flat, gate);
    let probs = g.softmax_last(logits);
    let (mask, counts) = topk_mask(g.value(probs), ntok, cfg.n_experts, cfg.top_k);
    let mask_id = g.leaf(&[ntok, cfg.n_experts], mask);
    let masked = g.mul(probs, mask_id);
    let denom = g.sum_last(masked);
    let weights = g.div(masked, denom);
    Router {
        flat,
        probs,
        weights,
        counts,
    }
}

pub(crate) fn moe(
    cfg: &Config,
    g: &mut Graph,
    norm: usize,
    gate: usize,
    experts: &[[usize; 3]],
    x: usize,
) -> (usize, usize, Vec<u32>) {
    let batch = g.shape(x)[0];
    let seq = g.shape(x)[1];
    let ntok = batch * seq;
    let route = router(cfg, g, norm, gate, x);
    let mut acc = g.leaf(&[ntok, cfg.dim], vec![0.0; ntok * cfg.dim]);
    for (e, ws) in experts.iter().enumerate() {
        if route.counts[e] == 0 {
            continue;
        }
        let hidden = swiglu(g, route.flat, ws[0], ws[1], ws[2]);
        let column = g.slice(route.weights, 1, e, 1);
        let scaled = g.mul(hidden, column);
        acc = g.add(acc, scaled);
    }
    let y = g.reshape(acc, &[batch, seq, cfg.dim]);
    (g.add(x, y), route.probs, route.counts)
}

fn forward(cfg: &Config, arch: &Arch, params: &[f32], batch: &Batch) -> (Graph, Vec<usize>, usize, Vec<Vec<u32>>) {
    let mut b = bind(arch, params);
    let mut x = b.g.embed(b.nodes[arch.embed], &batch.tokens, batch.batch, batch.seq);
    let mut probs = Vec::new();
    let mut counts = Vec::new();
    for layer in &arch.layers {
        let aw = AttnW {
            norm: b.nodes[layer.attn_norm],
            q: b.nodes[layer.wq],
            k: b.nodes[layer.wk],
            v: b.nodes[layer.wv],
            o: b.nodes[layer.wo],
        };
        x = attention(cfg, &mut b.g, aw, x, batch.seq);
        let experts: Vec<[usize; 3]> = layer.experts.iter().map(|e| [b.nodes[e[0]], b.nodes[e[1]], b.nodes[e[2]]]).collect();
        let norm = b.nodes[layer.ffn_norm];
        let gate = b.nodes[layer.gate];
        let (y, p, c) = moe(cfg, &mut b.g, norm, gate, &experts, x);
        x = y;
        probs.push(p);
        counts.push(c);
    }
    let hidden = b.g.rmsnorm(x, b.nodes[arch.norm], cfg.rms_norm_eps);
    let logits = b.g.matmul(hidden, b.nodes[arch.head]);
    let shifted = b.g.slice(logits, 1, 0, batch.seq - 1);
    let pred = b.g.reshape(shifted, &[batch.batch * (batch.seq - 1), cfg.vocab]);
    let mut targets = Vec::with_capacity(batch.batch * (batch.seq - 1));
    for bi in 0..batch.batch {
        for t in 1..batch.seq {
            targets.push(batch.tokens[bi * batch.seq + t]);
        }
    }
    let ce = b.g.cross_entropy(pred, &targets);
    let cat = if probs.len() == 1 {
        probs[0]
    } else {
        b.g.concat(&probs, 0)
    };
    let mean_p = b.g.mean_axis0(cat);
    let rows = b.g.shape(cat)[0];
    let mut fraction = vec![0f32; cfg.n_experts];
    for c in &counts {
        for (e, n) in c.iter().enumerate() {
            fraction[e] += *n as f32;
        }
    }
    for f in &mut fraction {
        *f /= rows as f32;
    }
    let dot = b.g.dot_const(mean_p, &fraction);
    let aux = b.g.scale(dot, cfg.n_experts as f32);
    let weighted = b.g.scale(aux, cfg.aux_loss_coef);
    let loss = b.g.add(ce, weighted);
    (b.g, b.nodes, loss, counts)
}

pub struct Eval {
    pub loss: f32,
    pub grads: Vec<f32>,
    pub expert_counts: Vec<Vec<u32>>,
}

pub fn evaluate(cfg: &Config, params: &[f32], batch: &Batch) -> Eval {
    let spec = arch(cfg);
    let (mut g, nodes, loss, counts) = forward(cfg, &spec, params, batch);
    let loss_v = g.scalar(loss);
    g.backward(loss);
    let mut grads = vec![0f32; params.len()];
    let off = offsets(&spec.slots);
    for (s, (id, o)) in spec.slots.iter().zip(nodes.iter().zip(&off)) {
        let n: usize = s.shape.iter().product();
        grads[*o..*o + n].copy_from_slice(g.grad(*id));
    }
    Eval {
        loss: loss_v,
        grads,
        expert_counts: counts,
    }
}

fn grad_norm(g: &[f32]) -> f32 {
    g.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub(crate) fn sgd(params: &mut [f32], grads: &[f32], lr: f32) {
    for (p, g) in params.iter_mut().zip(grads) {
        *p -= lr * g;
    }
}

pub struct CompileReport {
    pub paper_parameters: u64,
    pub active_parameters: u64,
    pub demo_parameters: u64,
    pub loss_before: f32,
    pub loss_after: f32,
    pub learning_rate: f32,
    pub max_abs_fd_error: f32,
    pub sparse_loss: f32,
    pub unused_expert_grad: f32,
    pub used_expert_grad: f32,
    pub full_model_stages: usize,
    pub full_model_peak: u64,
    pub full_model_budget: u64,
    pub full_model_code_peak: u64,
    pub full_model_code_budget: u64,
}

/// Finite-difference check, then one SGD step, on the Mixtral block.
pub fn compile_mixtral() -> CompileReport {
    let paper = Config::mixtral_8x7b();
    let smooth = Config::demo_smooth();
    let sparse = Config::demo_sparse();
    let params0 = init_params(&smooth, 0x4D49_5854);
    let batch = sample_batch(&smooth, 2, 6, 7);
    let base = evaluate(&smooth, &params0, &batch);
    let names = arch(&smooth).slots.into_iter().map(|s| s.name).collect::<Vec<_>>();
    let off = {
        let slots = arch(&smooth).slots;
        offsets(&slots)
    };
    let probes = [
        "embed",
        "layers.0.wq",
        "layers.1.experts.0.w2",
        "layers.1.gate",
        "lm_head",
    ];
    let mut max_err = 0f32;
    for name in probes {
        let idx = names.iter().position(|n| n == name).expect(name);
        let at = off[idx] + 3;
        let analytic = base.grads[at];
        let eps = 1e-3;
        let mut up = params0.clone();
        let mut down = params0.clone();
        up[at] += eps;
        down[at] -= eps;
        let lp = evaluate(&smooth, &up, &batch).loss;
        let lm = evaluate(&smooth, &down, &batch).loss;
        let numeric = (lp - lm) / (2.0 * eps);
        max_err = max_err.max((analytic - numeric).abs());
    }

    let mut lr_used = 0.0;
    let mut loss_after = base.loss;
    for lr in [0.2, 0.05, 0.01, 0.002] {
        let mut stepped = params0.clone();
        sgd(&mut stepped, &base.grads, lr);
        let next = evaluate(&smooth, &stepped, &batch).loss;
        if next.is_finite() && next < base.loss {
            lr_used = lr;
            loss_after = next;
            break;
        }
    }

    let sparse_params = init_params(&sparse, 0x5350_4152);
    // Three tokens and top-2 routing cannot cover eight experts, so at least
    // one expert is idle and its adjoint must stay zero.
    let sparse_batch = sample_batch(&sparse, 1, 3, 9);
    let sparse_eval = evaluate(&sparse, &sparse_params, &sparse_batch);
    let slots = arch(&sparse).slots;
    let soff = offsets(&slots);
    let mut unused = 0f32;
    let mut used = 0f32;
    let mut saw_unused = false;
    let mut saw_used = false;
    for layer in 0..sparse.n_layers {
        for e in 0..sparse.n_experts {
            let name = format!("layers.{layer}.experts.{e}.w1");
            let idx = slots.iter().position(|s| s.name == name).unwrap();
            let n = slots[idx].shape.iter().product::<usize>();
            let g = &sparse_eval.grads[soff[idx]..soff[idx] + n];
            let peak = g.iter().map(|z| z.abs()).fold(0f32, f32::max);
            let count = sparse_eval.expert_counts[layer][e];
            if count == 0 {
                unused = unused.max(peak);
                saw_unused = true;
            } else {
                used = used.max(peak);
                saw_used = true;
            }
        }
    }
    assert!(saw_used, "a routed expert should receive tokens");
    assert!(saw_unused, "top-2 of 8 should leave an expert idle on this batch");
    let _ = grad_norm(&base.grads);
    let budget = crate::stage::Memory {
        code_bytes: 1 << 20,
        buffer_bytes: 8 << 30,
    };
    let plan = crate::stage::schedule(&paper, 1, 512, budget).expect("8 GiB schedule");

    CompileReport {
        paper_parameters: parameter_count(&paper),
        active_parameters: active_parameter_count(&paper),
        demo_parameters: parameter_count(&smooth),
        loss_before: base.loss,
        loss_after,
        learning_rate: lr_used,
        max_abs_fd_error: max_err,
        sparse_loss: sparse_eval.loss,
        unused_expert_grad: unused,
        used_expert_grad: used,
        full_model_stages: plan.stages.len(),
        full_model_peak: plan.peak_bytes,
        full_model_budget: plan.budget,
        full_model_code_peak: plan.peak_code_bytes,
        full_model_code_budget: plan.code_budget,
    }
}
