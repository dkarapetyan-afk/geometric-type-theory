# Geometric type theory

A kernel for the two calculi in Mitchell Riley, [Geometric Type Theory, Done Two Ways](https://topos.institute/blog/2026-06-15-geometric-type-theory-done-two-ways/) (Topos Institute, 2026-06-15). One checker implements both presentations. They share a host Martin-Löf fragment and the same classifying-topos transport. Style A is not elaborated into Style B. The section “Compiling networks to NVIDIA kernels” specifies how a compiler on this kernel turns a geometric tensor term into NVIDIA forward and backward kernels.

```
cargo test --offline
cargo run --offline --quiet -- check examples/theories/monoid.gtt
```

`gtt check` prints `path: ok` and exits 0, or prints a spanned error and exits 1. The crate has no dependencies.

## Host

The host is predicative Martin-Löf type theory. `U` is a type and is not an element of itself. A term of type `U` is a type. The formers are `Π`, `Σ`, identity (with `J`), `Empty`, `Unit`, sums, `Nat`, and `U`, with the usual β and η rules. Identity is written `==` or `Id`. `Bool` in the examples is the definition `Unit + Unit`.

## Style B

Theories and models are judgements, not a type of theories. The formers are `One`, theory-`Σ`, `Sort`, `Ax`, and mixed `▹`. `ty`/`sort` and `unax`/`ax` are definitional inverses. The second projection of a theory-`Σ` instantiates the family by model substitution.

A pointed object, and the natural numbers as a model of it:

```
theory Pointed : Theory = (X :: Sort) * Ax (ty X)

model natPoint :: Pointed = (sort Nat, ax zero)

check unax (pr2 natPoint) : Nat
defeq unax (pr2 natPoint) = zero : Nat
```

`unax (pr2 natPoint)` is `zero` because `pr₂` substitutes the first component for the bound model. See `examples/emtt/`.

## Style A

A straight-line theory is a list of fields. Each field has an external telescope and a construction context. Constructions are the geometric fragment: `Σ`, identity, `Empty`, `Unit`, sums, propositional truncation, `Nat`, and `Ax`. There is no function type and no universe in the construction language. Models interpret constructions back into the host.

The theory of a monoid, with the carrier required to be a 0-type:

```
theory Monoid where
  sort X ;
  term e : X ;
  term mul (x : X, y : X) : X ;
  term unitL (x : X) : Id X (mul(x, e)) x ;
  term unitR (x : X) : Id X (mul(e, x)) x ;
  term assoc (x : X, y : X, z : X) : Id X (mul(mul(x, y), z)) (mul(x, mul(y, z))) ;
  term isset (x : X, y : X, p : Id X x y, q : Id X x y) : Id (Id X x y) p q ;
end
```

`examples/theories/monoid_model.gtt` gives the booleans under disjunction as a model, including the unit laws, associativity, and 0-truncation. The other files in that directory are the strict interval, a ring of characteristic zero, a ring of finite characteristic, a module over a postulated host ring, and a prime filter.

External `Nat`-induction into a construction is `nind`, as in the characteristic axiom. `let ax n := c in …` unwraps `Ax`. A field parameter list without `;` is a construction telescope. `sort P(r : R ;)` is an external telescope. Both nonempty telescopes are split by `;`.

Coproduct elimination in a construction follows the post’s rule with an empty extra telescope: the motive may use the scrutinee and the ambient context. The same restriction applies to the other construction eliminators. Propositional truncation can be formed (`Trunc (P + Q)` in the interval). There is no truncation eliminator.

## Classifying contexts

A context may bind `x : A` and `u :: T`. Weakening past a term is silent. Weakening past a model and substitution of a model are written explicitly, as `t ^ u` and `t{m / u}` (`↑` is also accepted). Both push through the geometric constructors, including `Nat`, sums, identity, `Sort`, `Ax`, `One`, theory-`Σ`, and `▹`. Both stick on `Π` and `U`, including when those types are closed. In particular `Nat ^ u` is `Nat`, and `(Nat -> Bool) ^ u` is not `(Nat ^ u) -> (Bool ^ u)`. Looking up `u :: T` yields `T ^ u`. Cancelling a weakening against the matching substitution gives the original term back. Defined models are constants, so they are not weakened by the variable rule; a classifying variable is.

`examples/classify/` checks the object classifier and the stuck arrow. `examples/negative/` rejects a bare carrier used across a model binder, an arrow inside a construction, and a field used at the wrong arity.

## Compiling networks to NVIDIA kernels

A network specification is a straight-line geometric term. The CUDA forward pass and the CUDA backward pass are two models of that term. The checker is the front end. Reverse mode and kernel emission are a second interpretation of the same construction.

`gtt compile` builds that pass for Mixtral 8x7B. It checks the geometric signature, runs the forward loss and its adjoint, takes one gradient step, and writes CUDA kernels whose forward GEMM, activation gradient, and weight gradient match the host formulas on the GPU.

### What compiles, and what is sealed afterwards

Style A is a signature of sorts and operations, indexed by an external host telescope, with no function type and no universe in the construction language. That is the specification language for a tensor program. A dense layer, a residual branch, a convolution, an attention block, and an unrolled recurrence are terms built from those operations by substitution. `examples/theories/module.gtt` is the same pattern: `sigma (r : R ; v : M)` takes its scalar arguments from the host and its vector argument from the theory.

The host Martin-Löf fragment is where a finished kernel is a function. Transport pushes through sums, products, identity, `Nat`, `Ax`, and theory fields, and it sticks on `Π` and `U`. The compiler finishes the geometric term first — shape arithmetic, the forward DAG, the cotangent DAG — and only then seals each DAG as a host function that launches a kernel. A specification written as a host lambda, or as `Ax` of a function type, does not instantiate when a CUDA model is substituted for the generic one, because that substitution sticks.

An arbitrary specification is an arbitrary construction in a fixed theory of tensor operations: field applications, pairs, coproduct elimination, external `nind`, and `let ax` around parameter buffers. Data-dependent dispatch is a sum match. Its cotangent runs the branch that was taken and sends zero to the other branch. Recurrence is `nind`, reversed from the successor case back to zero. A higher-order architecture, a layer that takes another layer as a value, is specialized to a straight-line term before this pass.

### Mixtral 8x7B

The compiled network is Mixtral of Experts (Jiang et al., arXiv:2401.04088), the open checkpoint `mistralai/Mixtral-8x7B-v0.1`. Thirty-two blocks each contain grouped-query attention and a sparse MoE. Attention has 32 query heads and 8 key/value heads, rotary embeddings with θ = 10^6, and a sliding window of 4096: query `q` sees key `k` when `k ≤ q` and `q − k < 4096`. The feed-forward is a softmax router over 8 SwiGLU experts,

```
expert(x) = W2 (SiLU(W1 x) * W3 x)
```

with the top 2 experts per token. Their routing weights are renormalized to sum to 1. The block residual sits outside both sublayers. The training loss is next-token cross-entropy plus `0.02` times the Switch load-balancing term `E Σ_i f_i P_i`, where `f_i` is the fraction of tokens sent to expert `i` and `P_i` is that expert’s mean router probability. The routing choice is a coproduct elimination: an expert that received no tokens has a zero adjoint, and the selected experts plus the softmax Jacobian receive the cotangent.

Counted from those tensors, including the final RMSNorm and the untied output head, the published model has 46,702,792,704 parameters, of which 12,879,925,248 are active at top-2. `gtt compile` checks that count, then runs the same block at demonstration width. On the smooth configuration every expert is selected, so the loss is differentiable, and the adjoint matches a central difference on the embedding, an attention matrix, an expert down-projection, the router, and the output head. One SGD step, `θ ← θ − η ∇L`, lowers the loss. On the sparse configuration, top-2 of 8, an idle expert’s parameter gradient is zero and a routed expert’s is not.

`examples/mixtral/signature.gtt` is the straight-line theory the checker accepts. `examples/mixtral/kernels.cu` is the CUDA the compiler writes: GEMM together with its two adjoints, RMSNorm, and SiLU. The driver is compiled with `nvcc -ccbin g++-11 -arch=sm_75` and checked on the GPU.

### The theory the kernels are models of

Primitives are fields of one theory. Shapes are host `Nat`s. Normalization computes a concrete grid size whenever the shape reduces. A stuck shape becomes a kernel argument.

```
theory Tensor where
  sort T(m : Nat, n : Nat ;) ;
  term gemm(m : Nat, n : Nat, k : Nat ; a : T(m, k), b : T(k, n)) : T(m, n) ;
  term add(m : Nat, n : Nat ; a : T(m, n), b : T(m, n)) : T(m, n) ;
  term relu(m : Nat, n : Nat ; a : T(m, n)) : T(m, n) ;
  term transpose(m : Nat, n : Nat ; a : T(m, n)) : T(n, m) ;
end
```

A network is a construction whose free variables are weights and inputs, not a new primitive:

```
-- h = relu(gemm(w1, x))
-- y = gemm(w2, h)
gemm(10, batch, 128 ; w2, relu(128, batch ; gemm(128, batch, 784 ; w1, x)))
```

`elab_record_model` already turns each field into a host function of its telescope, and `con_to_htm` already folds a construction into applications of those functions. That fold is the compiler hook. `con_to_htm` sends `Trunc` and most eliminators to `Embed(U)`, a dummy universe code, so the kernel backend uses its own walk of `Con`. The walk is the same recursion as `subst_con`, and it produces a DAG instead of a host term.

Two models of `Tensor` are required.

| Model | What a field means |
| --- | --- |
| `Ref` | An array function. Equations are checked here by running them. |
| `Cuda` | A device kernel template: pointer arguments, launch bounds, and the operation it implements. |

Instantiating the generic network at `Cuda` is model substitution. For a Style A record that is the `impl_lvls` vector `con_to_htm` already threads through interpretation. For a classifying variable `u :: Tensor` it is `{cuda / u}`. Substitution sticks if any operation was hidden inside a `Π`.

### Reverse mode is another construction

The backward pass is another straight-line term in the same theory. The output cotangent is an extra variable of the output sort. Each primitive has a fixed cotangent, and substitution composes them. That substitution is the chain rule.

For the two-layer network, with output cotangent `dy`:

```
dW2 = gemm(dy, transpose(h))
dH  = gemm(transpose(w2), dy)
dZ  = drelu(z, dH)          -- z = gemm(w1, x), the saved pre-activation
dW1 = gemm(dZ, transpose(x))
dX  = gemm(transpose(w1), dZ)
```

`h` and `z` are primal values the forward DAG keeps. The translation walks the construction once, binds each intermediate as a device buffer, and emits the cotangent term in reverse topological order. Coproduct branches are translated separately and rejoined with `match` on the same scrutinee. `nind` becomes a loop whose backward loop runs the translated successor case from `n` down to zero. `let ax w := params in body` unwraps a host parameter buffer; the cotangent of `w` is the gradient written back to that buffer.

The cotangent laws are equations of `Tensor`, proved in `Ref` by computation:

```
term gemmLeft(m : Nat, n : Nat, k : Nat ; a : T(m, k), b : T(k, n), dy : T(m, n))
  : Id T(m, k)
      (dLeft(m, n, k ; a, b, dy))
      (gemm(m, k, n ; dy, transpose(n, k ; b))) ;
```

`Cuda` is checked against the same types. Definitional equality does not identify a kernel with the reference function: a launch is a host `Π`, and transport stops there. The equational check is the reference model. The CUDA model is also tested by launching both on concrete shapes.

### From the DAG to a kernel

Normalize shapes with the evaluator before any launch decision. Then rewrite the DAG with the theory's equations. The rewrite that matters on NVIDIA is epilogue fusion: `relu(gemm(a, b))` becomes one GEMM whose epilogue is ReLU, and the backward pass is the fused derivative of ReLU plus the two GEMMs. Each remaining node is one launch.

Emit CUDA C++, not hand-written PTX. A GEMM node is a cuBLAS or CUTLASS call. A fused pointwise node is a generated kernel: one thread per element, grid `ceil(numel / 256)`, the body a straight transcription of that node. The forward kernel and the backward kernel are two functions. The host wrapper, the sealed `Π`, allocates the saved activations, launches forward, then launches backward with `dy`.

```cuda
__global__ void relu_bwd(int n, const float* z, const float* dh, float* dz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dz[i] = z[i] > 0.f ? dh[i] : 0.f;
}
```

Buffer assignment is the SSA of the construction: one device pointer per bound subterm, live from its definition to its last cotangent use. Weights unwrapped with `let ax` are the caller's buffers.

### Where it sits in this repository

`src/mixtral.rs` is the Mixtral model. `src/ad.rs` is the tape: forward values and the reverse adjoint. `src/cuda_emit.rs` prints the kernels. `gtt compile` runs the demonstration and writes `examples/mixtral/kernels.cu`. `cargo test --offline --test mixtral` checks the parameter count, the finite-difference adjoint, the idle-expert gradient, and the GPU kernels.

The cotangent is an ordinary reverse sweep over the tape. It is not the `Const` modality, and it is not a translation of Style A into Style B. Epilogue fusion and cuBLAS launches are the lowering described above; the kernels that ship are the GEMM, RMSNorm, and SiLU adjoints the Mixtral tape uses.

## Not in this kernel

Section 3 of the note is not implemented. That is the synthetic quasicoherence eliminator, slice theories `u ↓ T`, the `Const` modality, the multimodal sketch, and the “not-not-equal” argument. Also omitted, as the note leaves them unsettled or as future work: a general higher-inductive schema, an elaborator that inserts weakenings, a translation between the two styles, and a mechanized semantics. The kernel is the syntax and the definitional theory through the end of §2. The Mixtral compiler above is a model of that kernel: it does not add function types to the construction language, and it does not implement §3.
