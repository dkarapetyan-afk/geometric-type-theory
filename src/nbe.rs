//! Syntax, values, and normalization for the host, for EMTT-style theories,
//! and for the geometric transports `↑` and `{m/u}`.
//!
//! Inverse-image transport pushes through limits, colimits, identity, `ℕ`,
//! truncation's host image, `Sort`, `Ax`, `▹`, and theory `Σ`. It sticks on
//! `Π` and `U`, including when those types are closed. Bound variables of a
//! family are rigid: they are punched out as holes, the body is transported,
//! and the argument is written back afterwards so it is not transported a
//! second time.

#![allow(dead_code)]

use std::cell::Cell;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Syntax
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum HTm {
    Var(u32),
    U,
    Pi {
        binder: u32,
        dom: Box<HTm>,
        cod: Box<HTm>,
    },
    Lam {
        binder: u32,
        body: Box<HTm>,
    },
    App(Box<HTm>, Box<HTm>),
    Sigma {
        binder: u32,
        dom: Box<HTm>,
        cod: Box<HTm>,
    },
    Pair(Box<HTm>, Box<HTm>),
    Fst(Box<HTm>),
    Snd(Box<HTm>),
    Sum(Box<HTm>, Box<HTm>),
    Inl(Box<HTm>),
    Inr(Box<HTm>),
    Match {
        scrut: Box<HTm>,
        motive_b: u32,
        motive: Box<HTm>,
        inl_b: u32,
        inl: Box<HTm>,
        inr_b: u32,
        inr: Box<HTm>,
    },
    Id(Box<HTm>, Box<HTm>, Box<HTm>),
    Refl(Box<HTm>),
    J {
        ty: Box<HTm>,
        a: Box<HTm>,
        b: Box<HTm>,
        path: Box<HTm>,
        y_b: u32,
        p_b: u32,
        motive: Box<HTm>,
        refl_case: Box<HTm>,
    },
    Nat,
    Z,
    S(Box<HTm>),
    NatInd {
        scrut: Box<HTm>,
        k_b: u32,
        motive: Box<HTm>,
        zcase: Box<HTm>,
        m_b: u32,
        ih_b: u32,
        scase: Box<HTm>,
    },
    Empty,
    Exfalso {
        motive: Box<HTm>,
        scrut: Box<HTm>,
    },
    Unit,
    Tt,
    /// A value spliced into syntax (inferred types, definitional transports).
    Embed(Box<HVal>),
    Ty(Box<MTm>),
    Unax(Box<MTm>),
    Weak {
        tm: Box<HTm>,
        past: u32,
    },
    MSub {
        tm: Box<HTm>,
        repl: MRepl,
        past: u32,
    },
}

#[derive(Clone, Debug)]
pub enum MRepl {
    Syn(Box<MTm>),
    Val(Box<MVal>),
}

#[derive(Clone, Debug)]
pub enum MTm {
    Var(u32),
    Embed(Box<MVal>),
    Tt,
    Pair(Box<MTm>, Box<MTm>),
    Pr1(Box<MTm>),
    Pr2(Box<MTm>),
    Lam {
        binder: u32,
        body: Box<MTm>,
    },
    App(Box<MTm>, Box<HTm>),
    Sort(Box<HTm>),
    Ax(Box<HTm>),
    IndSum {
        scrut: Box<HTm>,
        l_b: u32,
        left: Box<MTm>,
        r_b: u32,
        right: Box<MTm>,
    },
    IndNat {
        scrut: Box<HTm>,
        zcase: Box<MTm>,
        m_b: u32,
        ih_b: u32,
        scase: Box<MTm>,
    },
    Weak {
        tm: Box<MTm>,
        past: u32,
    },
    MSub {
        tm: Box<MTm>,
        repl: MRepl,
        past: u32,
    },
}

#[derive(Clone, Debug)]
pub enum Con {
    Var(u32),
    Field {
        idx: usize,
        host: Vec<HTm>,
        args: Vec<Con>,
    },
    Sigma {
        binder: u32,
        dom: Box<Con>,
        cod: Box<Con>,
    },
    Pair(Box<Con>, Box<Con>),
    Fst(Box<Con>),
    Snd(Box<Con>),
    Id(Box<Con>, Box<Con>, Box<Con>),
    Refl(Box<Con>),
    Empty,
    Exfalso {
        motive: Box<Con>,
        scrut: Box<Con>,
    },
    Unit,
    Tt,
    Sum(Box<Con>, Box<Con>),
    Inl(Box<Con>),
    Inr(Box<Con>),
    Match {
        scrut: Box<Con>,
        x: u32,
        extra: Vec<(u32, Con)>,
        motive: Box<Con>,
        inl_b: u32,
        inl_extra: Vec<u32>,
        inl: Box<Con>,
        inr_b: u32,
        inr_extra: Vec<u32>,
        inr: Box<Con>,
        theta: Vec<Con>,
    },
    Trunc(Box<Con>),
    TIn(Box<Con>),
    TElim {
        scrut: Box<Con>,
        x: u32,
        motive: Box<Con>,
        p_b: u32,
        q_b: u32,
        proof: Box<Con>,
        a_b: u32,
        into: Box<Con>,
    },
    Nat,
    Z,
    S(Box<Con>),
    NInd {
        scrut: Box<HTm>,
        k: u32,
        motive: Box<Con>,
        zcase: Box<Con>,
        m_b: u32,
        ih_b: u32,
        scase: Box<Con>,
    },
    Ax(Box<HTm>),
    AxIn(Box<HTm>),
    LetAx {
        binder: u32,
        scrut: Box<Con>,
        /// Body is a construction in the external context extended by `binder : A`.
        body: Box<Con>,
    },
    J {
        y_b: u32,
        p_b: u32,
        motive: Box<Con>,
        refl_case: Box<Con>,
        path: Box<Con>,
    },
}

#[derive(Clone, Debug)]
pub enum RecKind {
    Sort,
    Term(Con),
}

#[derive(Clone, Debug)]
pub struct RecField {
    pub name: String,
    pub delta: Vec<(u32, String, HTm)>,
    pub phi: Vec<(u32, String, Con)>,
    pub kind: RecKind,
}

#[derive(Clone, Debug)]
pub struct RecordTheory {
    pub fields: Vec<RecField>,
}

#[derive(Clone, Debug)]
pub enum TTm {
    One,
    Sort,
    Ax(Box<HTm>),
    Sigma {
        binder: u32,
        fst: Box<TTm>,
        snd: Box<TTm>,
    },
    Rtri {
        binder: u32,
        dom: Box<HTm>,
        cod: Box<TTm>,
    },
    IndSum {
        scrut: Box<HTm>,
        l_b: u32,
        left: Box<TTm>,
        r_b: u32,
        right: Box<TTm>,
    },
    IndNat {
        scrut: Box<HTm>,
        zcase: Box<TTm>,
        m_b: u32,
        scase: Box<TTm>,
    },
    Record(RecordTheory),
    /// A theory value spliced back into syntax (named theories, transports).
    Embed(Box<TVal>),
    Weak {
        tm: Box<TTm>,
        past: u32,
    },
    MSub {
        tm: Box<TTm>,
        repl: MRepl,
        past: u32,
    },
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Env {
    slots: Vec<(u32, Slot)>,
}

#[derive(Clone, Debug)]
pub enum Slot {
    H(HVal),
    M(MVal),
}

impl Env {
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    pub fn get(&self, lvl: u32) -> Option<&Slot> {
        self.slots
            .iter()
            .rev()
            .find(|(l, _)| *l == lvl)
            .map(|(_, s)| s)
    }

    pub fn insert(&self, lvl: u32, slot: Slot) -> Env {
        let mut e = self.clone();
        e.slots.retain(|(l, _)| *l != lvl);
        e.slots.push((lvl, slot));
        e
    }

    /// Drop bindings introduced at `past` or later. Levels are allocated in
    /// context order, so this is the prefix before a model binder.
    pub fn truncate_before(&self, past: u32) -> Env {
        Env {
            slots: self
                .slots
                .iter()
                .filter(|(l, _)| *l < past)
                .cloned()
                .collect(),
        }
    }

    pub fn remove_level(&mut self, lvl: u32) {
        self.slots.retain(|(l, _)| *l != lvl);
    }
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct Plain<T> {
    pub env: Env,
    pub binder: u32,
    pub body: T,
}

#[derive(Clone, Debug)]
pub enum HClos {
    Plain(Plain<HTm>),
    TransWeak {
        inner: Plain<HTm>,
        past: u32,
    },
    TransSub {
        inner: Plain<HTm>,
        past: u32,
        repl: Box<MVal>,
    },
}

#[derive(Clone, Debug)]
pub struct TPlain {
    pub env: Env,
    pub binder: u32,
    pub body: TTm,
    /// Theory `Σ` binds a model. `▹` binds a host element.
    pub model_binder: bool,
}

#[derive(Clone, Debug)]
pub enum TClos {
    Plain(TPlain),
    TransWeak {
        inner: TPlain,
        past: u32,
    },
    TransSub {
        inner: TPlain,
        past: u32,
        repl: Box<MVal>,
    },
}

#[derive(Clone, Debug)]
pub enum MClos {
    Plain(Plain<MTm>),
    TransWeak {
        inner: Plain<MTm>,
        past: u32,
    },
    TransSub {
        inner: Plain<MTm>,
        past: u32,
        repl: Box<MVal>,
    },
}

#[derive(Clone, Debug)]
pub enum Head {
    Var {
        lvl: u32,
        ty: Box<HVal>,
    },
    App {
        fun: Box<Head>,
        arg: Box<HVal>,
        ty: Box<HVal>,
    },
    Fst {
        of: Box<Head>,
        ty: Box<HVal>,
    },
    Snd {
        of: Box<Head>,
        ty: Box<HVal>,
    },
    Match {
        scrut: Box<Head>,
        motive: Box<HClos>,
        inl: Box<HClos>,
        inr: Box<HClos>,
        ty: Box<HVal>,
    },
    J {
        path: Box<Head>,
        env: Env,
        y_b: u32,
        p_b: u32,
        motive: HTm,
        refl_case: Box<HVal>,
        ty: Box<HVal>,
    },
    NatInd {
        scrut: Box<Head>,
        env: Env,
        k_b: u32,
        motive: HTm,
        zcase: Box<HVal>,
        m_b: u32,
        ih_b: u32,
        scase: HTm,
        ty: Box<HVal>,
    },
    Exfalso {
        scrut: Box<Head>,
        ty: Box<HVal>,
    },
}

#[derive(Clone, Debug)]
pub enum HVal {
    U,
    Pi(Box<HVal>, Box<HClos>),
    Sigma(Box<HVal>, Box<HClos>),
    Sum(Box<HVal>, Box<HVal>),
    Id(Box<HVal>, Box<HVal>, Box<HVal>),
    Nat,
    Empty,
    Unit,
    Lam(Box<HClos>),
    Pair(Box<HVal>, Box<HVal>),
    Inl(Box<HVal>),
    Inr(Box<HVal>),
    Refl(Box<HVal>),
    Z,
    S(Box<HVal>),
    Tt,
    Ty(Box<MVal>),
    Unax(Box<MVal>),
    Neu(Box<Head>),
    /// A transported non-geometric value. The inner value is *not* pre-transported.
    StuckWeak(Box<HVal>, u32),
    StuckSub(Box<HVal>, Box<MVal>, u32),
    /// Rigid placeholder for a binder while a family is transported.
    Hole(u32),
    /// Already lives in the target context; transport must not enter.
    Frozen(Box<HVal>),
}

#[derive(Clone, Debug)]
pub enum MHead {
    Var { lvl: u32 },
    Pr1(Box<MHead>),
    Pr2(Box<MHead>),
    App { fun: Box<MHead>, arg: Box<HVal> },
    IndSum { scrut: Box<HVal> },
    IndNat { scrut: Box<HVal> },
}

#[derive(Clone, Debug)]
pub enum MVal {
    Tt,
    Pair(Box<MVal>, Box<MVal>),
    Lam(Box<MClos>),
    Sort(Box<HVal>),
    Ax(Box<HVal>),
    Neu { head: Box<MHead>, thy: Box<TVal> },
    StuckWeak(Box<MVal>, u32),
    StuckSub(Box<MVal>, Box<MVal>, u32),
    MHole(u32),
    Frozen(Box<MVal>),
}

#[derive(Clone, Debug)]
pub enum TVal {
    One,
    Sort,
    Ax(Box<HVal>),
    Sigma(Box<TVal>, Box<TClos>),
    Rtri(Box<HVal>, Box<TClos>),
    IndSum {
        scrut: Box<HVal>,
        left: Box<TClos>,
        right: Box<TClos>,
    },
    IndNat {
        scrut: Box<HVal>,
        zero: Box<TVal>,
        succ: Box<TClos>,
    },
    Record(RecordTheory),
    StuckWeak(Box<TVal>, u32),
    StuckSub(Box<TVal>, Box<MVal>, u32),
}

#[derive(Clone)]
enum Trav {
    Weak { past: u32 },
    Sub { past: u32, repl: MVal },
}

pub struct Nbe {
    fresh: Cell<u32>,
    depth: Cell<u32>,
}

impl Nbe {
    pub fn new(fresh: u32) -> Self {
        Self {
            fresh: Cell::new(fresh.max(1_000_000)),
            depth: Cell::new(0),
        }
    }

    fn bump(&self) {
        let depth = self.depth.get() + 1;
        if depth > 20_000 {
            panic!("normalization exceeded its recursion budget");
        }
        self.depth.set(depth);
    }

    fn drop_depth(&self) {
        self.depth.set(self.depth.get().saturating_sub(1));
    }

    fn fresh_id(&self) -> u32 {
        let id = self.fresh.get();
        self.fresh.set(id + 1);
        id
    }

    pub fn fresh_neu(&self, ty: &HVal) -> HVal {
        let lvl = self.fresh_id();
        HVal::Neu(Box::new(Head::Var {
            lvl,
            ty: Box::new(ty.clone()),
        }))
    }

    // ----- eval -----------------------------------------------------------

    pub fn eval_h(&self, tm: &HTm, env: &Env) -> HVal {
        self.bump();
        let v = match tm {
            HTm::Var(l) => match env.get(*l) {
                Some(Slot::H(v)) => v.clone(),
                _ => HVal::Neu(Box::new(Head::Var {
                    lvl: *l,
                    ty: Box::new(HVal::U),
                })),
            },
            HTm::U => HVal::U,
            HTm::Pi { binder, dom, cod } => HVal::Pi(
                Box::new(self.eval_h(dom, env)),
                Box::new(HClos::Plain(Plain {
                    env: env.clone(),
                    binder: *binder,
                    body: (**cod).clone(),
                })),
            ),
            HTm::Lam { binder, body } => HVal::Lam(Box::new(HClos::Plain(Plain {
                env: env.clone(),
                binder: *binder,
                body: (**body).clone(),
            }))),
            HTm::App(f, a) => {
                let fv = self.eval_h(f, env);
                let av = self.eval_h(a, env);
                self.apply_h(fv, av)
            }
            HTm::Sigma { binder, dom, cod } => HVal::Sigma(
                Box::new(self.eval_h(dom, env)),
                Box::new(HClos::Plain(Plain {
                    env: env.clone(),
                    binder: *binder,
                    body: (**cod).clone(),
                })),
            ),
            HTm::Pair(a, b) => {
                HVal::Pair(Box::new(self.eval_h(a, env)), Box::new(self.eval_h(b, env)))
            }
            HTm::Fst(t) => self.fst_h(self.eval_h(t, env)),
            HTm::Snd(t) => self.snd_h(self.eval_h(t, env)),
            HTm::Sum(a, b) => {
                HVal::Sum(Box::new(self.eval_h(a, env)), Box::new(self.eval_h(b, env)))
            }
            HTm::Inl(t) => HVal::Inl(Box::new(self.eval_h(t, env))),
            HTm::Inr(t) => HVal::Inr(Box::new(self.eval_h(t, env))),
            HTm::Match {
                scrut,
                motive_b,
                motive,
                inl_b,
                inl,
                inr_b,
                inr,
            } => {
                let s = self.eval_h(scrut, env);
                let mot = HClos::Plain(Plain {
                    env: env.clone(),
                    binder: *motive_b,
                    body: (**motive).clone(),
                });
                let lc = HClos::Plain(Plain {
                    env: env.clone(),
                    binder: *inl_b,
                    body: (**inl).clone(),
                });
                let rc = HClos::Plain(Plain {
                    env: env.clone(),
                    binder: *inr_b,
                    body: (**inr).clone(),
                });
                self.eval_match(s, mot, lc, rc)
            }
            HTm::Id(t, a, b) => HVal::Id(
                Box::new(self.eval_h(t, env)),
                Box::new(self.eval_h(a, env)),
                Box::new(self.eval_h(b, env)),
            ),
            HTm::Refl(t) => HVal::Refl(Box::new(self.eval_h(t, env))),
            HTm::J {
                ty,
                a,
                b,
                path,
                y_b,
                p_b,
                motive,
                refl_case,
            } => {
                let pv = self.eval_h(path, env);
                let rc = self.eval_h(refl_case, env);
                let tyv = self.eval_h(ty, env);
                let av = self.eval_h(a, env);
                let bv = self.eval_h(b, env);
                self.eval_j(pv, rc, tyv, av, bv, *y_b, *p_b, motive, env)
            }
            HTm::Nat => HVal::Nat,
            HTm::Z => HVal::Z,
            HTm::S(t) => HVal::S(Box::new(self.eval_h(t, env))),
            HTm::NatInd {
                scrut,
                k_b,
                motive,
                zcase,
                m_b,
                ih_b,
                scase,
            } => {
                let s = self.eval_h(scrut, env);
                self.eval_natind(s, *k_b, motive, zcase, *m_b, *ih_b, scase, env)
            }
            HTm::Empty => HVal::Empty,
            HTm::Exfalso { motive, scrut } => {
                let mot = self.eval_h(motive, env);
                let s = self.eval_h(scrut, env);
                self.eval_exfalso(mot, s)
            }
            HTm::Unit => HVal::Unit,
            HTm::Tt => HVal::Tt,
            HTm::Embed(v) => (**v).clone(),
            HTm::Ty(m) => self.ty_of(self.eval_m(m, env)),
            HTm::Unax(m) => self.unax_of(self.eval_m(m, env)),
            HTm::Weak { tm, past } => {
                let prefix = env.truncate_before(*past);
                let v = self.eval_h(tm, &prefix);
                self.transport_h(v, &Trav::Weak { past: *past })
            }
            HTm::MSub { tm, repl, past } => {
                let prefix = env.truncate_before(*past);
                let mv = self.eval_repl(repl, &prefix);
                let v = self.eval_h(tm, env);
                self.transport_h(
                    v,
                    &Trav::Sub {
                        past: *past,
                        repl: mv,
                    },
                )
            }
        };
        self.drop_depth();
        v
    }

    fn eval_repl(&self, repl: &MRepl, env: &Env) -> MVal {
        match repl {
            MRepl::Syn(tm) => self.eval_m(tm, env),
            MRepl::Val(v) => (**v).clone(),
        }
    }

    pub fn eval_m(&self, tm: &MTm, env: &Env) -> MVal {
        self.bump();
        let v = match tm {
            MTm::Embed(v) => (**v).clone(),
            MTm::Var(l) => match env.get(*l) {
                Some(Slot::M(v)) => v.clone(),
                _ => MVal::Neu {
                    head: Box::new(MHead::Var { lvl: *l }),
                    thy: Box::new(TVal::Sort),
                },
            },
            MTm::Tt => MVal::Tt,
            MTm::Pair(a, b) => {
                MVal::Pair(Box::new(self.eval_m(a, env)), Box::new(self.eval_m(b, env)))
            }
            MTm::Pr1(t) => self.pr1(self.eval_m(t, env)),
            MTm::Pr2(t) => self.pr2(self.eval_m(t, env)),
            MTm::Lam { binder, body } => MVal::Lam(Box::new(MClos::Plain(Plain {
                env: env.clone(),
                binder: *binder,
                body: (**body).clone(),
            }))),
            MTm::App(f, a) => {
                let fv = self.eval_m(f, env);
                let av = self.eval_h(a, env);
                self.apply_m(fv, av)
            }
            MTm::Sort(a) => self.sort_of(self.eval_h(a, env)),
            MTm::Ax(a) => self.ax_of(self.eval_h(a, env)),
            MTm::IndSum {
                scrut,
                l_b,
                left,
                r_b,
                right,
            } => {
                let s = self.eval_h(scrut, env);
                let lc = MClos::Plain(Plain {
                    env: env.clone(),
                    binder: *l_b,
                    body: (**left).clone(),
                });
                let rc = MClos::Plain(Plain {
                    env: env.clone(),
                    binder: *r_b,
                    body: (**right).clone(),
                });
                self.eval_ind_m(s, lc, rc)
            }
            MTm::IndNat {
                scrut,
                zcase,
                m_b,
                ih_b,
                scase,
            } => {
                let s = self.eval_h(scrut, env);
                self.eval_ind_nat_m(s, zcase, *m_b, *ih_b, scase, env)
            }
            MTm::Weak { tm, past } => {
                let prefix = env.truncate_before(*past);
                let v = self.eval_m(tm, &prefix);
                self.transport_m(v, &Trav::Weak { past: *past })
            }
            MTm::MSub { tm, repl, past } => {
                let prefix = env.truncate_before(*past);
                let mv = self.eval_repl(repl, &prefix);
                let v = self.eval_m(tm, env);
                self.transport_m(
                    v,
                    &Trav::Sub {
                        past: *past,
                        repl: mv,
                    },
                )
            }
        };
        self.drop_depth();
        v
    }

    pub fn eval_t(&self, tm: &TTm, env: &Env) -> TVal {
        self.bump();
        let v = match tm {
            TTm::One => TVal::One,
            TTm::Sort => TVal::Sort,
            TTm::Ax(a) => TVal::Ax(Box::new(self.eval_h(a, env))),
            TTm::Sigma { binder, fst, snd } => TVal::Sigma(
                Box::new(self.eval_t(fst, env)),
                Box::new(TClos::Plain(TPlain {
                    env: env.clone(),
                    binder: *binder,
                    body: (**snd).clone(),
                    model_binder: true,
                })),
            ),
            TTm::Rtri { binder, dom, cod } => TVal::Rtri(
                Box::new(self.eval_h(dom, env)),
                Box::new(TClos::Plain(TPlain {
                    env: env.clone(),
                    binder: *binder,
                    body: (**cod).clone(),
                    model_binder: false,
                })),
            ),
            TTm::IndSum {
                scrut,
                l_b,
                left,
                r_b,
                right,
            } => {
                let s = self.eval_h(scrut, env);
                let lc = TClos::Plain(TPlain {
                    env: env.clone(),
                    binder: *l_b,
                    body: (**left).clone(),
                    model_binder: false,
                });
                let rc = TClos::Plain(TPlain {
                    env: env.clone(),
                    binder: *r_b,
                    body: (**right).clone(),
                    model_binder: false,
                });
                self.eval_ind_t(s, lc, rc)
            }
            TTm::IndNat {
                scrut,
                zcase,
                m_b,
                scase,
            } => {
                let s = self.eval_h(scrut, env);
                let z = self.eval_t(zcase, env);
                let sc = TClos::Plain(TPlain {
                    env: env.clone(),
                    binder: *m_b,
                    body: (**scase).clone(),
                    model_binder: false,
                });
                self.eval_ind_nat_t(s, z, sc)
            }
            TTm::Record(r) => TVal::Record(r.clone()),
            TTm::Embed(v) => (**v).clone(),
            TTm::Weak { tm, past } => {
                let prefix = env.truncate_before(*past);
                let v = self.eval_t(tm, &prefix);
                self.transport_t(v, &Trav::Weak { past: *past })
            }
            TTm::MSub { tm, repl, past } => {
                let prefix = env.truncate_before(*past);
                let mv = self.eval_repl(repl, &prefix);
                let v = self.eval_t(tm, env);
                self.transport_t(
                    v,
                    &Trav::Sub {
                        past: *past,
                        repl: mv,
                    },
                )
            }
        };
        self.drop_depth();
        v
    }

    fn ty_of(&self, m: MVal) -> HVal {
        match m {
            MVal::Sort(a) => self.thaw_h(*a),
            MVal::Frozen(m) => HVal::Frozen(Box::new(self.ty_of(*m))),
            other => HVal::Ty(Box::new(other)),
        }
    }

    fn unax_of(&self, m: MVal) -> HVal {
        match m {
            MVal::Ax(a) => self.thaw_h(*a),
            MVal::Frozen(m) => HVal::Frozen(Box::new(self.unax_of(*m))),
            other => HVal::Unax(Box::new(other)),
        }
    }

    fn sort_of(&self, a: HVal) -> MVal {
        match self.thaw_h(a) {
            HVal::Ty(m) => *m,
            other => MVal::Sort(Box::new(other)),
        }
    }

    fn ax_of(&self, a: HVal) -> MVal {
        match self.thaw_h(a) {
            HVal::Unax(m) => *m,
            other => MVal::Ax(Box::new(other)),
        }
    }

    pub fn apply_h(&self, f: HVal, arg: HVal) -> HVal {
        match f {
            HVal::Lam(c) => self.apply_hclos(*c, arg),
            HVal::Neu(h) => {
                let ty = head_ty(&h);
                let res_ty = match &ty {
                    HVal::Pi(_, cod) => self.apply_hclos((**cod).clone(), arg.clone()),
                    _ => HVal::U,
                };
                HVal::Neu(Box::new(Head::App {
                    fun: h,
                    arg: Box::new(arg),
                    ty: Box::new(res_ty),
                }))
            }
            HVal::Frozen(v) => HVal::Frozen(Box::new(self.apply_h(*v, arg))),
            other => HVal::Neu(Box::new(Head::App {
                fun: Box::new(Head::Var {
                    lvl: self.fresh_id(),
                    ty: Box::new(other),
                }),
                arg: Box::new(arg),
                ty: Box::new(HVal::U),
            })),
        }
    }

    pub fn apply_hclos(&self, c: HClos, arg: HVal) -> HVal {
        match c {
            HClos::Plain(p) => {
                let env = p.env.insert(p.binder, Slot::H(arg));
                self.eval_h(&p.body, &env)
            }
            HClos::TransWeak { inner, past } => {
                let hole = self.fresh_id();
                let env = inner.env.insert(inner.binder, Slot::H(HVal::Hole(hole)));
                let v = self.eval_h(&inner.body, &env);
                let tv = self.transport_h(v, &Trav::Weak { past });
                let tv = self.subst_hole_h(tv, hole, &arg);
                self.thaw_h(tv)
            }
            HClos::TransSub { inner, past, repl } => {
                let hole = self.fresh_id();
                let env = inner.env.insert(inner.binder, Slot::H(HVal::Hole(hole)));
                let v = self.eval_h(&inner.body, &env);
                let tv = self.transport_h(
                    v,
                    &Trav::Sub {
                        past,
                        repl: (*repl).clone(),
                    },
                );
                let tv = self.subst_hole_h(tv, hole, &arg);
                self.thaw_h(tv)
            }
        }
    }

    pub fn apply_m(&self, f: MVal, arg: HVal) -> MVal {
        match f {
            MVal::Lam(c) => self.apply_mclos(*c, arg),
            MVal::Neu { head, thy } => {
                let res = match &*thy {
                    TVal::Rtri(_, cod) => self.apply_t_host(cod, arg.clone()),
                    _ => TVal::One,
                };
                MVal::Neu {
                    head: Box::new(MHead::App {
                        fun: head,
                        arg: Box::new(arg),
                    }),
                    thy: Box::new(res),
                }
            }
            MVal::Frozen(m) => MVal::Frozen(Box::new(self.apply_m(*m, arg))),
            other => other,
        }
    }

    fn apply_mclos(&self, c: MClos, arg: HVal) -> MVal {
        match c {
            MClos::Plain(p) => {
                let env = p.env.insert(p.binder, Slot::H(arg));
                self.eval_m(&p.body, &env)
            }
            MClos::TransWeak { inner, past } => {
                let hole = self.fresh_id();
                let env = inner.env.insert(inner.binder, Slot::H(HVal::Hole(hole)));
                let v = self.eval_m(&inner.body, &env);
                let tv = self.transport_m(v, &Trav::Weak { past });
                let tv = self.subst_hole_m(tv, hole, &arg);
                self.thaw_m(tv)
            }
            MClos::TransSub { inner, past, repl } => {
                let hole = self.fresh_id();
                let env = inner.env.insert(inner.binder, Slot::H(HVal::Hole(hole)));
                let v = self.eval_m(&inner.body, &env);
                let tv = self.transport_m(
                    v,
                    &Trav::Sub {
                        past,
                        repl: (*repl).clone(),
                    },
                );
                let tv = self.subst_hole_m(tv, hole, &arg);
                self.thaw_m(tv)
            }
        }
    }

    pub fn apply_t_host(&self, c: &TClos, arg: HVal) -> TVal {
        match c {
            TClos::Plain(p) => {
                let env = p.env.insert(p.binder, Slot::H(arg));
                self.eval_t(&p.body, &env)
            }
            TClos::TransWeak { inner, past } => {
                let hole = self.fresh_id();
                let env = inner.env.insert(inner.binder, Slot::H(HVal::Hole(hole)));
                let v = self.eval_t(&inner.body, &env);
                let tv = self.transport_t(v, &Trav::Weak { past: *past });
                let tv = self.subst_hole_t(tv, hole, &arg);
                self.thaw_t(tv)
            }
            TClos::TransSub { inner, past, repl } => {
                let hole = self.fresh_id();
                let env = inner.env.insert(inner.binder, Slot::H(HVal::Hole(hole)));
                let v = self.eval_t(&inner.body, &env);
                let tv = self.transport_t(
                    v,
                    &Trav::Sub {
                        past: *past,
                        repl: (**repl).clone(),
                    },
                );
                let tv = self.subst_hole_t(tv, hole, &arg);
                self.thaw_t(tv)
            }
        }
    }

    /// Instantiate a theory-`Σ` codomain at a model, using model substitution
    /// for the bound model variable (not ordinary term substitution).
    pub fn inst_sigma(&self, c: &TClos, arg: &MVal) -> TVal {
        match c {
            TClos::Plain(p) => {
                let neu = MVal::Neu {
                    head: Box::new(MHead::Var { lvl: p.binder }),
                    thy: Box::new(TVal::One),
                };
                let env = p.env.insert(p.binder, Slot::M(neu));
                let v = self.eval_t(&p.body, &env);
                self.transport_t(
                    v,
                    &Trav::Sub {
                        past: p.binder,
                        repl: arg.clone(),
                    },
                )
            }
            TClos::TransWeak { inner, past } => {
                let id = self.fresh_id();
                let env = inner.env.insert(inner.binder, Slot::M(MVal::MHole(id)));
                let v = self.eval_t(&inner.body, &env);
                let tv = self.transport_t(v, &Trav::Weak { past: *past });
                // The bound model is rigid under this weakening, then written back.
                let tv = self.subst_mhole_t(tv, id, arg);
                // Also substitute the bound variable if the body mentioned it as a neu
                // of its binder level — MHole covers the eval we just did.
                self.thaw_t(tv)
            }
            TClos::TransSub { inner, past, repl } => {
                let id = self.fresh_id();
                let env = inner.env.insert(inner.binder, Slot::M(MVal::MHole(id)));
                let v = self.eval_t(&inner.body, &env);
                let tv = self.transport_t(
                    v,
                    &Trav::Sub {
                        past: *past,
                        repl: (**repl).clone(),
                    },
                );
                let tv = self.subst_mhole_t(tv, id, arg);
                self.thaw_t(tv)
            }
        }
    }

    fn eval_match(&self, s: HVal, mot: HClos, lc: HClos, rc: HClos) -> HVal {
        match s {
            HVal::Inl(v) => self.apply_hclos(lc, *v),
            HVal::Inr(v) => self.apply_hclos(rc, *v),
            HVal::Frozen(v) => HVal::Frozen(Box::new(self.eval_match(*v, mot, lc, rc))),
            HVal::Neu(h) => {
                let ty = self.apply_hclos(mot.clone(), HVal::Neu(h.clone()));
                HVal::Neu(Box::new(Head::Match {
                    scrut: h,
                    motive: Box::new(mot),
                    inl: Box::new(lc),
                    inr: Box::new(rc),
                    ty: Box::new(ty),
                }))
            }
            other => HVal::Neu(Box::new(Head::Match {
                scrut: Box::new(Head::Var {
                    lvl: self.fresh_id(),
                    ty: Box::new(other),
                }),
                motive: Box::new(mot),
                inl: Box::new(lc),
                inr: Box::new(rc),
                ty: Box::new(HVal::U),
            })),
        }
    }

    fn eval_j(
        &self,
        path: HVal,
        refl_case: HVal,
        ty: HVal,
        a: HVal,
        b: HVal,
        y_b: u32,
        p_b: u32,
        motive: &HTm,
        env: &Env,
    ) -> HVal {
        match path {
            HVal::Refl(_) => refl_case,
            HVal::Frozen(v) => self.eval_j(*v, refl_case, ty, a, b, y_b, p_b, motive, env),
            HVal::Neu(h) => {
                let mut e = env.insert(y_b, Slot::H(b.clone()));
                e = e.insert(p_b, Slot::H(HVal::Neu(h.clone())));
                let tyv = self.eval_h(motive, &e);
                let _ = (ty, a);
                HVal::Neu(Box::new(Head::J {
                    path: h,
                    env: env.clone(),
                    y_b,
                    p_b,
                    motive: motive.clone(),
                    refl_case: Box::new(refl_case),
                    ty: Box::new(tyv),
                }))
            }
            _ => refl_case,
        }
    }

    fn eval_natind(
        &self,
        s: HVal,
        k_b: u32,
        motive: &HTm,
        zcase: &HTm,
        m_b: u32,
        ih_b: u32,
        scase: &HTm,
        env: &Env,
    ) -> HVal {
        match s {
            HVal::Z => self.eval_h(zcase, env),
            HVal::S(n) => {
                let ih = self.eval_natind(*n.clone(), k_b, motive, zcase, m_b, ih_b, scase, env);
                let e = env.insert(m_b, Slot::H(*n)).insert(ih_b, Slot::H(ih));
                self.eval_h(scase, &e)
            }
            HVal::Frozen(v) => self.eval_natind(*v, k_b, motive, zcase, m_b, ih_b, scase, env),
            HVal::Neu(h) => {
                let e = env.insert(k_b, Slot::H(HVal::Neu(h.clone())));
                let ty = self.eval_h(motive, &e);
                HVal::Neu(Box::new(Head::NatInd {
                    scrut: h,
                    env: env.clone(),
                    k_b,
                    motive: motive.clone(),
                    zcase: Box::new(self.eval_h(zcase, env)),
                    m_b,
                    ih_b,
                    scase: scase.clone(),
                    ty: Box::new(ty),
                }))
            }
            _ => HVal::Neu(Box::new(Head::NatInd {
                scrut: Box::new(Head::Var {
                    lvl: self.fresh_id(),
                    ty: Box::new(HVal::Nat),
                }),
                env: env.clone(),
                k_b,
                motive: motive.clone(),
                zcase: Box::new(self.eval_h(zcase, env)),
                m_b,
                ih_b,
                scase: scase.clone(),
                ty: Box::new(HVal::U),
            })),
        }
    }

    fn eval_exfalso(&self, motive: HVal, scrut: HVal) -> HVal {
        match scrut {
            HVal::Neu(h) => HVal::Neu(Box::new(Head::Exfalso {
                scrut: h,
                ty: Box::new(motive),
            })),
            HVal::Frozen(v) => HVal::Frozen(Box::new(self.eval_exfalso(motive, *v))),
            _ => HVal::Neu(Box::new(Head::Exfalso {
                scrut: Box::new(Head::Var {
                    lvl: self.fresh_id(),
                    ty: Box::new(HVal::Empty),
                }),
                ty: Box::new(motive),
            })),
        }
    }

    fn eval_ind_m(&self, s: HVal, left: MClos, right: MClos) -> MVal {
        match s {
            HVal::Inl(v) => self.apply_mclos_host_as_binder(left, *v),
            HVal::Inr(v) => self.apply_mclos_host_as_binder(right, *v),
            HVal::Neu(h) => MVal::Neu {
                head: Box::new(MHead::IndSum {
                    scrut: Box::new(HVal::Neu(h)),
                }),
                thy: Box::new(TVal::One),
            },
            _ => MVal::Neu {
                head: Box::new(MHead::IndSum { scrut: Box::new(s) }),
                thy: Box::new(TVal::One),
            },
        }
    }

    fn apply_mclos_host_as_binder(&self, c: MClos, arg: HVal) -> MVal {
        // Ind branches bind a host element. Reuse apply_mclos.
        self.apply_mclos(c, arg)
    }

    fn eval_ind_nat_m(
        &self,
        s: HVal,
        zcase: &MTm,
        m_b: u32,
        ih_b: u32,
        scase: &MTm,
        env: &Env,
    ) -> MVal {
        match s {
            HVal::Z => self.eval_m(zcase, env),
            HVal::S(n) => {
                let ih = self.eval_ind_nat_m(*n.clone(), zcase, m_b, ih_b, scase, env);
                let e = env.insert(m_b, Slot::H(*n)).insert(ih_b, Slot::M(ih));
                self.eval_m(scase, &e)
            }
            other => MVal::Neu {
                head: Box::new(MHead::IndNat {
                    scrut: Box::new(other),
                }),
                thy: Box::new(TVal::One),
            },
        }
    }

    fn eval_ind_t(&self, s: HVal, left: TClos, right: TClos) -> TVal {
        match s {
            HVal::Inl(v) => self.apply_t_host(&left, *v),
            HVal::Inr(v) => self.apply_t_host(&right, *v),
            other => TVal::IndSum {
                scrut: Box::new(other),
                left: Box::new(left),
                right: Box::new(right),
            },
        }
    }

    fn eval_ind_nat_t(&self, s: HVal, zero: TVal, succ: TClos) -> TVal {
        match s {
            HVal::Z => zero,
            HVal::S(n) => self.apply_t_host(&succ, *n),
            other => TVal::IndNat {
                scrut: Box::new(other),
                zero: Box::new(zero),
                succ: Box::new(succ),
            },
        }
    }

    pub fn fst_h(&self, v: HVal) -> HVal {
        match v {
            HVal::Pair(a, _) => *a,
            HVal::Frozen(v) => HVal::Frozen(Box::new(self.fst_h(*v))),
            HVal::Neu(h) => {
                let ty = match head_ty(&h) {
                    HVal::Sigma(d, _) => *d,
                    other => other,
                };
                HVal::Neu(Box::new(Head::Fst {
                    of: h,
                    ty: Box::new(ty),
                }))
            }
            other => other,
        }
    }

    pub fn snd_h(&self, v: HVal) -> HVal {
        match v {
            HVal::Pair(_, b) => *b,
            HVal::Frozen(v) => HVal::Frozen(Box::new(self.snd_h(*v))),
            HVal::Neu(h) => {
                let ty = match head_ty(&h) {
                    HVal::Sigma(_, cod) => {
                        let fst = HVal::Neu(Box::new(Head::Fst {
                            of: h.clone(),
                            ty: Box::new(HVal::U),
                        }));
                        self.apply_hclos(*cod, fst)
                    }
                    other => other,
                };
                HVal::Neu(Box::new(Head::Snd {
                    of: h,
                    ty: Box::new(ty),
                }))
            }
            other => other,
        }
    }

    pub fn pr1(&self, m: MVal) -> MVal {
        match m {
            MVal::Pair(a, _) => *a,
            MVal::Frozen(m) => MVal::Frozen(Box::new(self.pr1(*m))),
            MVal::Neu { head, thy } => {
                let fthy = match *thy {
                    TVal::Sigma(fst, _) => fst,
                    other => Box::new(other),
                };
                MVal::Neu {
                    head: Box::new(MHead::Pr1(head)),
                    thy: fthy,
                }
            }
            other => other,
        }
    }

    pub fn pr2(&self, m: MVal) -> MVal {
        match m {
            MVal::Pair(_, b) => *b,
            MVal::Frozen(m) => MVal::Frozen(Box::new(self.pr2(*m))),
            MVal::Neu { head, thy } => {
                let sthy = match &*thy {
                    TVal::Sigma(_, cod) => {
                        let p1 = MVal::Neu {
                            head: Box::new(MHead::Pr1(head.clone())),
                            thy: Box::new(TVal::One),
                        };
                        self.inst_sigma(cod, &p1)
                    }
                    _ => TVal::One,
                };
                MVal::Neu {
                    head: Box::new(MHead::Pr2(head)),
                    thy: Box::new(sthy),
                }
            }
            other => other,
        }
    }

    // ----- transport ------------------------------------------------------

    fn transport_h(&self, v: HVal, trav: &Trav) -> HVal {
        self.bump();
        let out = match v {
            HVal::Hole(i) => HVal::Hole(i),
            HVal::Frozen(v) => HVal::Frozen(v),
            HVal::U => self.stick_h(HVal::U, trav),
            HVal::Pi(d, c) => self.stick_h(HVal::Pi(d, c), trav),
            HVal::Lam(c) => self.stick_h(HVal::Lam(c), trav),
            HVal::Nat | HVal::Empty | HVal::Unit | HVal::Z | HVal::Tt => v,
            HVal::S(n) => HVal::S(Box::new(self.transport_h(*n, trav))),
            HVal::Sum(a, b) => HVal::Sum(
                Box::new(self.transport_h(*a, trav)),
                Box::new(self.transport_h(*b, trav)),
            ),
            HVal::Sigma(d, c) => HVal::Sigma(
                Box::new(self.transport_h(*d, trav)),
                Box::new(self.transport_hclos(*c, trav)),
            ),
            HVal::Id(t, a, b) => HVal::Id(
                Box::new(self.transport_h(*t, trav)),
                Box::new(self.transport_h(*a, trav)),
                Box::new(self.transport_h(*b, trav)),
            ),
            HVal::Pair(a, b) => HVal::Pair(
                Box::new(self.transport_h(*a, trav)),
                Box::new(self.transport_h(*b, trav)),
            ),
            HVal::Inl(a) => HVal::Inl(Box::new(self.transport_h(*a, trav))),
            HVal::Inr(a) => HVal::Inr(Box::new(self.transport_h(*a, trav))),
            HVal::Refl(a) => HVal::Refl(Box::new(self.transport_h(*a, trav))),
            HVal::Ty(m) => self.ty_of(self.transport_m(*m, trav)),
            HVal::Unax(m) => self.unax_of(self.transport_m(*m, trav)),
            HVal::Neu(h) => self.transport_neu(*h, trav),
            HVal::StuckWeak(inner, p) => match trav {
                Trav::Sub { past, .. } if *past == p => *inner,
                Trav::Weak { past } => HVal::StuckWeak(Box::new(HVal::StuckWeak(inner, p)), *past),
                Trav::Sub { past, repl } => HVal::StuckSub(
                    Box::new(HVal::StuckWeak(inner, p)),
                    Box::new(repl.clone()),
                    *past,
                ),
            },
            HVal::StuckSub(inner, m, p) => match trav {
                Trav::Weak { past } => {
                    HVal::StuckWeak(Box::new(HVal::StuckSub(inner, m, p)), *past)
                }
                Trav::Sub { past, repl } => HVal::StuckSub(
                    Box::new(HVal::StuckSub(inner, m, p)),
                    Box::new(repl.clone()),
                    *past,
                ),
            },
        };
        self.drop_depth();
        out
    }

    fn stick_h(&self, v: HVal, trav: &Trav) -> HVal {
        match trav {
            Trav::Weak { past } => HVal::StuckWeak(Box::new(v), *past),
            Trav::Sub { past, repl } => HVal::StuckSub(Box::new(v), Box::new(repl.clone()), *past),
        }
    }

    fn transport_hclos(&self, c: HClos, trav: &Trav) -> HClos {
        let inner = match c {
            HClos::Plain(p) => p,
            // Compose by stacking apply-time transports: keep the inner plain
            // term and rely on nested Trans constructors via the value they
            // produce. A clos that is already transported is applied through
            // its own constructor; wrapping again would drop the first
            // transport. Represent composition by a plain clos whose body is
            // the original and whose env has already been... we instead store
            // nested structure by making Trans* contain HClos. For the
            // closures built here, `c` is plain because eval only makes plain
            // clos, and transport wraps once.
            HClos::TransWeak { inner, past } => {
                return match trav {
                    Trav::Weak { past: p2 } => HClos::TransWeak {
                        inner: self.pre_weak_plain(inner, past),
                        past: *p2,
                    },
                    Trav::Sub { past: p2, repl } => HClos::TransSub {
                        inner: self.pre_weak_plain(inner, past),
                        past: *p2,
                        repl: Box::new(repl.clone()),
                    },
                };
            }
            HClos::TransSub { inner, past, repl } => {
                return match trav {
                    Trav::Weak { past: p2 } => HClos::TransWeak {
                        inner: self.pre_sub_plain(inner, past, *repl),
                        past: *p2,
                    },
                    Trav::Sub { past: p2, repl: r2 } => HClos::TransSub {
                        inner: self.pre_sub_plain(inner, past, *repl),
                        past: *p2,
                        repl: Box::new(r2.clone()),
                    },
                };
            }
        };
        match trav {
            Trav::Weak { past } => HClos::TransWeak { inner, past: *past },
            Trav::Sub { past, repl } => HClos::TransSub {
                inner,
                past: *past,
                repl: Box::new(repl.clone()),
            },
        }
    }

    fn pre_weak_plain(&self, inner: Plain<HTm>, past: u32) -> Plain<HTm> {
        // Encode the pending weak as a syntax node around the body so a second
        // transport still sees it. The body is evaluated in `inner.env`.
        Plain {
            env: inner.env,
            binder: inner.binder,
            body: HTm::Weak {
                tm: Box::new(inner.body),
                past,
            },
        }
    }

    fn pre_sub_plain(&self, inner: Plain<HTm>, past: u32, repl: MVal) -> Plain<HTm> {
        Plain {
            env: inner.env,
            binder: inner.binder,
            body: HTm::MSub {
                tm: Box::new(inner.body),
                repl: MRepl::Val(Box::new(repl)),
                past,
            },
        }
    }

    fn transport_neu(&self, h: Head, trav: &Trav) -> HVal {
        let ty = head_ty(&h);
        // Sigma elements commute with transport via η, even when a factor does not.
        if let HVal::Sigma(dom, cod) = &ty {
            let fst = HVal::Neu(Box::new(Head::Fst {
                of: Box::new(h.clone()),
                ty: dom.clone(),
            }));
            let snd_ty = self.apply_hclos((**cod).clone(), fst.clone());
            let snd = HVal::Neu(Box::new(Head::Snd {
                of: Box::new(h),
                ty: Box::new(snd_ty),
            }));
            return HVal::Pair(
                Box::new(self.transport_h(fst, trav)),
                Box::new(self.transport_h(snd, trav)),
            );
        }
        let ty2 = self.transport_h(ty.clone(), trav);
        if self.eq_ty(ty2, ty) {
            HVal::Neu(Box::new(h))
        } else {
            self.stick_h(HVal::Neu(Box::new(h)), trav)
        }
    }

    fn transport_m(&self, m: MVal, trav: &Trav) -> MVal {
        self.bump();
        let out = match m {
            MVal::MHole(i) => MVal::MHole(i),
            MVal::Frozen(v) => MVal::Frozen(v),
            MVal::Tt => MVal::Tt,
            MVal::Pair(a, b) => MVal::Pair(
                Box::new(self.transport_m(*a, trav)),
                Box::new(self.transport_m(*b, trav)),
            ),
            MVal::Lam(c) => MVal::Lam(Box::new(self.transport_mclos(*c, trav))),
            MVal::Sort(a) => self.sort_of(self.transport_h(*a, trav)),
            MVal::Ax(a) => self.ax_of(self.transport_h(*a, trav)),
            MVal::Neu { head, thy } => self.transport_mneu(*head, *thy, trav),
            MVal::StuckWeak(inner, p) => match trav {
                Trav::Sub { past, .. } if *past == p => *inner,
                Trav::Weak { past } => MVal::StuckWeak(Box::new(MVal::StuckWeak(inner, p)), *past),
                Trav::Sub { past, repl } => MVal::StuckSub(
                    Box::new(MVal::StuckWeak(inner, p)),
                    Box::new(repl.clone()),
                    *past,
                ),
            },
            MVal::StuckSub(inner, m, p) => match trav {
                Trav::Weak { past } => {
                    MVal::StuckWeak(Box::new(MVal::StuckSub(inner, m, p)), *past)
                }
                Trav::Sub { past, repl } => MVal::StuckSub(
                    Box::new(MVal::StuckSub(inner, m, p)),
                    Box::new(repl.clone()),
                    *past,
                ),
            },
        };
        self.drop_depth();
        out
    }

    fn transport_mneu(&self, head: MHead, thy: TVal, trav: &Trav) -> MVal {
        if let MHead::Var { lvl } = &head {
            if let Trav::Sub { past, repl } = trav {
                if *lvl == *past {
                    return repl.clone();
                }
            }
        }
        let thy2 = self.transport_t(thy.clone(), trav);
        if self.eq_t(&thy2, &thy) {
            MVal::Neu {
                head: Box::new(head),
                thy: Box::new(thy),
            }
        } else {
            match trav {
                Trav::Weak { past } => MVal::StuckWeak(
                    Box::new(MVal::Neu {
                        head: Box::new(head),
                        thy: Box::new(thy),
                    }),
                    *past,
                ),
                Trav::Sub { past, repl } => MVal::StuckSub(
                    Box::new(MVal::Neu {
                        head: Box::new(head),
                        thy: Box::new(thy),
                    }),
                    Box::new(repl.clone()),
                    *past,
                ),
            }
        }
    }

    fn transport_mclos(&self, c: MClos, trav: &Trav) -> MClos {
        match c {
            MClos::Plain(inner) => match trav {
                Trav::Weak { past } => MClos::TransWeak { inner, past: *past },
                Trav::Sub { past, repl } => MClos::TransSub {
                    inner,
                    past: *past,
                    repl: Box::new(repl.clone()),
                },
            },
            other => other,
        }
    }

    fn transport_t(&self, t: TVal, trav: &Trav) -> TVal {
        self.bump();
        let out = match t {
            TVal::One | TVal::Sort => t,
            TVal::Ax(a) => TVal::Ax(Box::new(self.transport_h(*a, trav))),
            TVal::Sigma(fst, cod) => TVal::Sigma(
                Box::new(self.transport_t(*fst, trav)),
                Box::new(self.transport_tclos(*cod, trav)),
            ),
            TVal::Rtri(dom, cod) => TVal::Rtri(
                Box::new(self.transport_h(*dom, trav)),
                Box::new(self.transport_tclos(*cod, trav)),
            ),
            TVal::IndSum { scrut, left, right } => {
                let s = self.transport_h(*scrut, trav);
                let l = self.transport_tclos(*left, trav);
                let r = self.transport_tclos(*right, trav);
                self.eval_ind_t(s, l, r)
            }
            TVal::IndNat { scrut, zero, succ } => {
                let s = self.transport_h(*scrut, trav);
                let z = self.transport_t(*zero, trav);
                let sc = self.transport_tclos(*succ, trav);
                self.eval_ind_nat_t(s, z, sc)
            }
            TVal::Record(r) => TVal::Record(self.transport_record(r, trav)),
            TVal::StuckWeak(inner, p) => match trav {
                Trav::Sub { past, .. } if *past == p => *inner,
                Trav::Weak { past } => TVal::StuckWeak(Box::new(TVal::StuckWeak(inner, p)), *past),
                Trav::Sub { past, repl } => TVal::StuckSub(
                    Box::new(TVal::StuckWeak(inner, p)),
                    Box::new(repl.clone()),
                    *past,
                ),
            },
            TVal::StuckSub(inner, m, p) => match trav {
                Trav::Weak { past } => {
                    TVal::StuckWeak(Box::new(TVal::StuckSub(inner, m, p)), *past)
                }
                Trav::Sub { past, repl } => TVal::StuckSub(
                    Box::new(TVal::StuckSub(inner, m, p)),
                    Box::new(repl.clone()),
                    *past,
                ),
            },
        };
        self.drop_depth();
        out
    }

    fn transport_tclos(&self, c: TClos, trav: &Trav) -> TClos {
        match c {
            TClos::Plain(inner) => match trav {
                Trav::Weak { past } => TClos::TransWeak { inner, past: *past },
                Trav::Sub { past, repl } => TClos::TransSub {
                    inner,
                    past: *past,
                    repl: Box::new(repl.clone()),
                },
            },
            other => other,
        }
    }

    fn transport_record(&self, r: RecordTheory, trav: &Trav) -> RecordTheory {
        RecordTheory {
            fields: r
                .fields
                .into_iter()
                .map(|f| RecField {
                    name: f.name,
                    delta: f
                        .delta
                        .into_iter()
                        .map(|(lvl, name, ty)| (lvl, name, wrap_h(ty, trav)))
                        .collect(),
                    phi: f
                        .phi
                        .into_iter()
                        .map(|(lvl, name, c)| (lvl, name, transport_con(c, trav)))
                        .collect(),
                    kind: match f.kind {
                        RecKind::Sort => RecKind::Sort,
                        RecKind::Term(c) => RecKind::Term(transport_con(c, trav)),
                    },
                })
                .collect(),
        }
    }

    // ----- holes / thaw ---------------------------------------------------

    fn subst_hole_h(&self, v: HVal, hole: u32, arg: &HVal) -> HVal {
        match v {
            HVal::Hole(h) if h == hole => HVal::Frozen(Box::new(arg.clone())),
            HVal::Hole(h) => HVal::Hole(h),
            HVal::Frozen(v) => HVal::Frozen(Box::new(self.subst_hole_h(*v, hole, arg))),
            HVal::U | HVal::Nat | HVal::Empty | HVal::Unit | HVal::Z | HVal::Tt => v,
            HVal::Pi(d, c) => HVal::Pi(
                Box::new(self.subst_hole_h(*d, hole, arg)),
                Box::new(self.subst_hole_hclos(*c, hole, arg)),
            ),
            HVal::Sigma(d, c) => HVal::Sigma(
                Box::new(self.subst_hole_h(*d, hole, arg)),
                Box::new(self.subst_hole_hclos(*c, hole, arg)),
            ),
            HVal::Sum(a, b) => HVal::Sum(
                Box::new(self.subst_hole_h(*a, hole, arg)),
                Box::new(self.subst_hole_h(*b, hole, arg)),
            ),
            HVal::Id(t, a, b) => HVal::Id(
                Box::new(self.subst_hole_h(*t, hole, arg)),
                Box::new(self.subst_hole_h(*a, hole, arg)),
                Box::new(self.subst_hole_h(*b, hole, arg)),
            ),
            HVal::Lam(c) => HVal::Lam(Box::new(self.subst_hole_hclos(*c, hole, arg))),
            HVal::Pair(a, b) => HVal::Pair(
                Box::new(self.subst_hole_h(*a, hole, arg)),
                Box::new(self.subst_hole_h(*b, hole, arg)),
            ),
            HVal::Inl(a) => HVal::Inl(Box::new(self.subst_hole_h(*a, hole, arg))),
            HVal::Inr(a) => HVal::Inr(Box::new(self.subst_hole_h(*a, hole, arg))),
            HVal::Refl(a) => HVal::Refl(Box::new(self.subst_hole_h(*a, hole, arg))),
            HVal::S(n) => HVal::S(Box::new(self.subst_hole_h(*n, hole, arg))),
            HVal::Ty(m) => HVal::Ty(Box::new(self.subst_hole_m(*m, hole, arg))),
            HVal::Unax(m) => HVal::Unax(Box::new(self.subst_hole_m(*m, hole, arg))),
            HVal::Neu(h) => HVal::Neu(Box::new(self.subst_hole_head(*h, hole, arg))),
            HVal::StuckWeak(i, p) => HVal::StuckWeak(Box::new(self.subst_hole_h(*i, hole, arg)), p),
            HVal::StuckSub(i, m, p) => HVal::StuckSub(
                Box::new(self.subst_hole_h(*i, hole, arg)),
                Box::new(self.subst_hole_m(*m, hole, arg)),
                p,
            ),
        }
    }

    fn subst_hole_hclos(&self, c: HClos, hole: u32, arg: &HVal) -> HClos {
        match c {
            HClos::Plain(p) => HClos::Plain(Plain {
                env: self.subst_hole_env(p.env, hole, arg),
                binder: p.binder,
                body: p.body,
            }),
            HClos::TransWeak { inner, past } => HClos::TransWeak {
                inner: Plain {
                    env: self.subst_hole_env(inner.env, hole, arg),
                    binder: inner.binder,
                    body: inner.body,
                },
                past,
            },
            HClos::TransSub { inner, past, repl } => HClos::TransSub {
                inner: Plain {
                    env: self.subst_hole_env(inner.env, hole, arg),
                    binder: inner.binder,
                    body: inner.body,
                },
                past,
                repl: Box::new(self.subst_hole_m(*repl, hole, arg)),
            },
        }
    }

    fn subst_hole_env(&self, env: Env, hole: u32, arg: &HVal) -> Env {
        Env {
            slots: env
                .slots
                .into_iter()
                .map(|(l, s)| {
                    let s = match s {
                        Slot::H(v) => Slot::H(self.subst_hole_h(v, hole, arg)),
                        Slot::M(v) => Slot::M(self.subst_hole_m(v, hole, arg)),
                    };
                    (l, s)
                })
                .collect(),
        }
    }

    fn subst_hole_head(&self, h: Head, hole: u32, arg: &HVal) -> Head {
        match h {
            Head::Var { lvl, ty } => Head::Var {
                lvl,
                ty: Box::new(self.subst_hole_h(*ty, hole, arg)),
            },
            Head::App { fun, arg: a, ty } => Head::App {
                fun: Box::new(self.subst_hole_head(*fun, hole, arg)),
                arg: Box::new(self.subst_hole_h(*a, hole, arg)),
                ty: Box::new(self.subst_hole_h(*ty, hole, arg)),
            },
            Head::Fst { of, ty } => Head::Fst {
                of: Box::new(self.subst_hole_head(*of, hole, arg)),
                ty: Box::new(self.subst_hole_h(*ty, hole, arg)),
            },
            Head::Snd { of, ty } => Head::Snd {
                of: Box::new(self.subst_hole_head(*of, hole, arg)),
                ty: Box::new(self.subst_hole_h(*ty, hole, arg)),
            },
            Head::Match {
                scrut,
                motive,
                inl,
                inr,
                ty,
            } => Head::Match {
                scrut: Box::new(self.subst_hole_head(*scrut, hole, arg)),
                motive: Box::new(self.subst_hole_hclos(*motive, hole, arg)),
                inl: Box::new(self.subst_hole_hclos(*inl, hole, arg)),
                inr: Box::new(self.subst_hole_hclos(*inr, hole, arg)),
                ty: Box::new(self.subst_hole_h(*ty, hole, arg)),
            },
            Head::J {
                path,
                env,
                y_b,
                p_b,
                motive,
                refl_case,
                ty,
            } => Head::J {
                path: Box::new(self.subst_hole_head(*path, hole, arg)),
                env: self.subst_hole_env(env, hole, arg),
                y_b,
                p_b,
                motive,
                refl_case: Box::new(self.subst_hole_h(*refl_case, hole, arg)),
                ty: Box::new(self.subst_hole_h(*ty, hole, arg)),
            },
            Head::NatInd {
                scrut,
                env,
                k_b,
                motive,
                zcase,
                m_b,
                ih_b,
                scase,
                ty,
            } => Head::NatInd {
                scrut: Box::new(self.subst_hole_head(*scrut, hole, arg)),
                env: self.subst_hole_env(env, hole, arg),
                k_b,
                motive,
                zcase: Box::new(self.subst_hole_h(*zcase, hole, arg)),
                m_b,
                ih_b,
                scase,
                ty: Box::new(self.subst_hole_h(*ty, hole, arg)),
            },
            Head::Exfalso { scrut, ty } => Head::Exfalso {
                scrut: Box::new(self.subst_hole_head(*scrut, hole, arg)),
                ty: Box::new(self.subst_hole_h(*ty, hole, arg)),
            },
        }
    }

    fn subst_hole_m(&self, m: MVal, hole: u32, arg: &HVal) -> MVal {
        match m {
            MVal::Tt | MVal::MHole(_) => m,
            MVal::Frozen(v) => MVal::Frozen(Box::new(self.subst_hole_m(*v, hole, arg))),
            MVal::Pair(a, b) => MVal::Pair(
                Box::new(self.subst_hole_m(*a, hole, arg)),
                Box::new(self.subst_hole_m(*b, hole, arg)),
            ),
            MVal::Lam(c) => MVal::Lam(Box::new(self.subst_hole_mclos(*c, hole, arg))),
            MVal::Sort(a) => MVal::Sort(Box::new(self.subst_hole_h(*a, hole, arg))),
            MVal::Ax(a) => MVal::Ax(Box::new(self.subst_hole_h(*a, hole, arg))),
            MVal::Neu { head, thy } => MVal::Neu {
                head,
                thy: Box::new(self.subst_hole_t(*thy, hole, arg)),
            },
            MVal::StuckWeak(i, p) => MVal::StuckWeak(Box::new(self.subst_hole_m(*i, hole, arg)), p),
            MVal::StuckSub(i, r, p) => MVal::StuckSub(
                Box::new(self.subst_hole_m(*i, hole, arg)),
                Box::new(self.subst_hole_m(*r, hole, arg)),
                p,
            ),
        }
    }

    fn subst_hole_mclos(&self, c: MClos, hole: u32, arg: &HVal) -> MClos {
        match c {
            MClos::Plain(p) => MClos::Plain(Plain {
                env: self.subst_hole_env(p.env, hole, arg),
                binder: p.binder,
                body: p.body,
            }),
            MClos::TransWeak { inner, past } => MClos::TransWeak {
                inner: Plain {
                    env: self.subst_hole_env(inner.env, hole, arg),
                    binder: inner.binder,
                    body: inner.body,
                },
                past,
            },
            MClos::TransSub { inner, past, repl } => MClos::TransSub {
                inner: Plain {
                    env: self.subst_hole_env(inner.env, hole, arg),
                    binder: inner.binder,
                    body: inner.body,
                },
                past,
                repl: Box::new(self.subst_hole_m(*repl, hole, arg)),
            },
        }
    }

    fn subst_hole_t(&self, t: TVal, hole: u32, arg: &HVal) -> TVal {
        match t {
            TVal::One | TVal::Sort => t,
            TVal::Ax(a) => TVal::Ax(Box::new(self.subst_hole_h(*a, hole, arg))),
            TVal::Sigma(fst, cod) => TVal::Sigma(
                Box::new(self.subst_hole_t(*fst, hole, arg)),
                Box::new(self.subst_hole_tclos(*cod, hole, arg)),
            ),
            TVal::Rtri(dom, cod) => TVal::Rtri(
                Box::new(self.subst_hole_h(*dom, hole, arg)),
                Box::new(self.subst_hole_tclos(*cod, hole, arg)),
            ),
            TVal::IndSum { scrut, left, right } => TVal::IndSum {
                scrut: Box::new(self.subst_hole_h(*scrut, hole, arg)),
                left: Box::new(self.subst_hole_tclos(*left, hole, arg)),
                right: Box::new(self.subst_hole_tclos(*right, hole, arg)),
            },
            TVal::IndNat { scrut, zero, succ } => TVal::IndNat {
                scrut: Box::new(self.subst_hole_h(*scrut, hole, arg)),
                zero: Box::new(self.subst_hole_t(*zero, hole, arg)),
                succ: Box::new(self.subst_hole_tclos(*succ, hole, arg)),
            },
            TVal::Record(r) => TVal::Record(r),
            TVal::StuckWeak(i, p) => TVal::StuckWeak(Box::new(self.subst_hole_t(*i, hole, arg)), p),
            TVal::StuckSub(i, m, p) => TVal::StuckSub(
                Box::new(self.subst_hole_t(*i, hole, arg)),
                Box::new(self.subst_hole_m(*m, hole, arg)),
                p,
            ),
        }
    }

    fn subst_hole_tclos(&self, c: TClos, hole: u32, arg: &HVal) -> TClos {
        match c {
            TClos::Plain(p) => TClos::Plain(TPlain {
                env: self.subst_hole_env(p.env, hole, arg),
                binder: p.binder,
                body: p.body,
                model_binder: p.model_binder,
            }),
            TClos::TransWeak { inner, past } => TClos::TransWeak {
                inner: TPlain {
                    env: self.subst_hole_env(inner.env, hole, arg),
                    binder: inner.binder,
                    body: inner.body,
                    model_binder: inner.model_binder,
                },
                past,
            },
            TClos::TransSub { inner, past, repl } => TClos::TransSub {
                inner: TPlain {
                    env: self.subst_hole_env(inner.env, hole, arg),
                    binder: inner.binder,
                    body: inner.body,
                    model_binder: inner.model_binder,
                },
                past,
                repl: Box::new(self.subst_hole_m(*repl, hole, arg)),
            },
        }
    }

    fn subst_mhole_t(&self, t: TVal, hole: u32, arg: &MVal) -> TVal {
        match t {
            TVal::One | TVal::Sort => t,
            TVal::Ax(a) => TVal::Ax(Box::new(self.subst_mhole_h(*a, hole, arg))),
            TVal::Rtri(dom, cod) => TVal::Rtri(
                Box::new(self.subst_mhole_h(*dom, hole, arg)),
                Box::new(self.subst_mhole_tclos(*cod, hole, arg)),
            ),
            TVal::Sigma(fst, cod) => TVal::Sigma(
                Box::new(self.subst_mhole_t(*fst, hole, arg)),
                Box::new(self.subst_mhole_tclos(*cod, hole, arg)),
            ),
            TVal::IndSum { scrut, left, right } => TVal::IndSum {
                scrut: Box::new(self.subst_mhole_h(*scrut, hole, arg)),
                left: Box::new(self.subst_mhole_tclos(*left, hole, arg)),
                right: Box::new(self.subst_mhole_tclos(*right, hole, arg)),
            },
            TVal::IndNat { scrut, zero, succ } => TVal::IndNat {
                scrut: Box::new(self.subst_mhole_h(*scrut, hole, arg)),
                zero: Box::new(self.subst_mhole_t(*zero, hole, arg)),
                succ: Box::new(self.subst_mhole_tclos(*succ, hole, arg)),
            },
            TVal::Record(r) => TVal::Record(r),
            TVal::StuckWeak(i, p) => {
                TVal::StuckWeak(Box::new(self.subst_mhole_t(*i, hole, arg)), p)
            }
            TVal::StuckSub(i, m, p) => TVal::StuckSub(
                Box::new(self.subst_mhole_t(*i, hole, arg)),
                Box::new(self.subst_mhole_m(*m, hole, arg)),
                p,
            ),
        }
    }

    fn subst_mhole_tclos(&self, c: TClos, hole: u32, arg: &MVal) -> TClos {
        let map_plain = |this: &Self, p: TPlain| TPlain {
            env: this.subst_mhole_env(p.env, hole, arg),
            binder: p.binder,
            body: p.body,
            model_binder: p.model_binder,
        };
        match c {
            TClos::Plain(p) => TClos::Plain(map_plain(self, p)),
            TClos::TransWeak { inner, past } => TClos::TransWeak {
                inner: map_plain(self, inner),
                past,
            },
            TClos::TransSub { inner, past, repl } => TClos::TransSub {
                inner: map_plain(self, inner),
                past,
                repl: Box::new(self.subst_mhole_m(*repl, hole, arg)),
            },
        }
    }

    fn subst_mhole_env(&self, env: Env, hole: u32, arg: &MVal) -> Env {
        Env {
            slots: env
                .slots
                .into_iter()
                .map(|(l, s)| {
                    let s = match s {
                        Slot::H(v) => Slot::H(self.subst_mhole_h(v, hole, arg)),
                        Slot::M(v) => Slot::M(self.subst_mhole_m(v, hole, arg)),
                    };
                    (l, s)
                })
                .collect(),
        }
    }

    fn subst_mhole_h(&self, v: HVal, hole: u32, arg: &MVal) -> HVal {
        match v {
            HVal::Ty(m) => HVal::Ty(Box::new(self.subst_mhole_m(*m, hole, arg))),
            HVal::Unax(m) => HVal::Unax(Box::new(self.subst_mhole_m(*m, hole, arg))),
            HVal::Frozen(v) => HVal::Frozen(Box::new(self.subst_mhole_h(*v, hole, arg))),
            HVal::Pi(d, c) => HVal::Pi(
                Box::new(self.subst_mhole_h(*d, hole, arg)),
                Box::new(self.subst_mhole_hclos(*c, hole, arg)),
            ),
            HVal::Sigma(d, c) => HVal::Sigma(
                Box::new(self.subst_mhole_h(*d, hole, arg)),
                Box::new(self.subst_mhole_hclos(*c, hole, arg)),
            ),
            HVal::Sum(a, b) => HVal::Sum(
                Box::new(self.subst_mhole_h(*a, hole, arg)),
                Box::new(self.subst_mhole_h(*b, hole, arg)),
            ),
            HVal::Id(t, a, b) => HVal::Id(
                Box::new(self.subst_mhole_h(*t, hole, arg)),
                Box::new(self.subst_mhole_h(*a, hole, arg)),
                Box::new(self.subst_mhole_h(*b, hole, arg)),
            ),
            HVal::Pair(a, b) => HVal::Pair(
                Box::new(self.subst_mhole_h(*a, hole, arg)),
                Box::new(self.subst_mhole_h(*b, hole, arg)),
            ),
            HVal::Lam(c) => HVal::Lam(Box::new(self.subst_mhole_hclos(*c, hole, arg))),
            HVal::Inl(a) => HVal::Inl(Box::new(self.subst_mhole_h(*a, hole, arg))),
            HVal::Inr(a) => HVal::Inr(Box::new(self.subst_mhole_h(*a, hole, arg))),
            HVal::Refl(a) => HVal::Refl(Box::new(self.subst_mhole_h(*a, hole, arg))),
            HVal::S(n) => HVal::S(Box::new(self.subst_mhole_h(*n, hole, arg))),
            HVal::Neu(h) => HVal::Neu(Box::new(self.subst_mhole_head(*h, hole, arg))),
            HVal::StuckWeak(i, p) => {
                HVal::StuckWeak(Box::new(self.subst_mhole_h(*i, hole, arg)), p)
            }
            HVal::StuckSub(i, m, p) => HVal::StuckSub(
                Box::new(self.subst_mhole_h(*i, hole, arg)),
                Box::new(self.subst_mhole_m(*m, hole, arg)),
                p,
            ),
            other => other,
        }
    }

    fn subst_mhole_hclos(&self, c: HClos, hole: u32, arg: &MVal) -> HClos {
        match c {
            HClos::Plain(p) => HClos::Plain(Plain {
                env: self.subst_mhole_env(p.env, hole, arg),
                binder: p.binder,
                body: p.body,
            }),
            HClos::TransWeak { inner, past } => HClos::TransWeak {
                inner: Plain {
                    env: self.subst_mhole_env(inner.env, hole, arg),
                    binder: inner.binder,
                    body: inner.body,
                },
                past,
            },
            HClos::TransSub { inner, past, repl } => HClos::TransSub {
                inner: Plain {
                    env: self.subst_mhole_env(inner.env, hole, arg),
                    binder: inner.binder,
                    body: inner.body,
                },
                past,
                repl: Box::new(self.subst_mhole_m(*repl, hole, arg)),
            },
        }
    }

    fn subst_mhole_head(&self, h: Head, hole: u32, arg: &MVal) -> Head {
        match h {
            Head::Var { lvl, ty } => Head::Var {
                lvl,
                ty: Box::new(self.subst_mhole_h(*ty, hole, arg)),
            },
            Head::App { fun, arg: a, ty } => Head::App {
                fun: Box::new(self.subst_mhole_head(*fun, hole, arg)),
                arg: Box::new(self.subst_mhole_h(*a, hole, arg)),
                ty: Box::new(self.subst_mhole_h(*ty, hole, arg)),
            },
            Head::Fst { of, ty } => Head::Fst {
                of: Box::new(self.subst_mhole_head(*of, hole, arg)),
                ty: Box::new(self.subst_mhole_h(*ty, hole, arg)),
            },
            Head::Snd { of, ty } => Head::Snd {
                of: Box::new(self.subst_mhole_head(*of, hole, arg)),
                ty: Box::new(self.subst_mhole_h(*ty, hole, arg)),
            },
            Head::Match {
                scrut,
                motive,
                inl,
                inr,
                ty,
            } => Head::Match {
                scrut: Box::new(self.subst_mhole_head(*scrut, hole, arg)),
                motive: Box::new(self.subst_mhole_hclos(*motive, hole, arg)),
                inl: Box::new(self.subst_mhole_hclos(*inl, hole, arg)),
                inr: Box::new(self.subst_mhole_hclos(*inr, hole, arg)),
                ty: Box::new(self.subst_mhole_h(*ty, hole, arg)),
            },
            Head::J {
                path,
                env,
                y_b,
                p_b,
                motive,
                refl_case,
                ty,
            } => Head::J {
                path: Box::new(self.subst_mhole_head(*path, hole, arg)),
                env: self.subst_mhole_env(env, hole, arg),
                y_b,
                p_b,
                motive,
                refl_case: Box::new(self.subst_mhole_h(*refl_case, hole, arg)),
                ty: Box::new(self.subst_mhole_h(*ty, hole, arg)),
            },
            Head::NatInd {
                scrut,
                env,
                k_b,
                motive,
                zcase,
                m_b,
                ih_b,
                scase,
                ty,
            } => Head::NatInd {
                scrut: Box::new(self.subst_mhole_head(*scrut, hole, arg)),
                env: self.subst_mhole_env(env, hole, arg),
                k_b,
                motive,
                zcase: Box::new(self.subst_mhole_h(*zcase, hole, arg)),
                m_b,
                ih_b,
                scase,
                ty: Box::new(self.subst_mhole_h(*ty, hole, arg)),
            },
            Head::Exfalso { scrut, ty } => Head::Exfalso {
                scrut: Box::new(self.subst_mhole_head(*scrut, hole, arg)),
                ty: Box::new(self.subst_mhole_h(*ty, hole, arg)),
            },
        }
    }

    fn subst_mhole_m(&self, m: MVal, hole: u32, arg: &MVal) -> MVal {
        match m {
            MVal::MHole(h) if h == hole => MVal::Frozen(Box::new(arg.clone())),
            MVal::MHole(h) => MVal::MHole(h),
            MVal::Tt => MVal::Tt,
            MVal::Frozen(v) => MVal::Frozen(Box::new(self.subst_mhole_m(*v, hole, arg))),
            MVal::Pair(a, b) => MVal::Pair(
                Box::new(self.subst_mhole_m(*a, hole, arg)),
                Box::new(self.subst_mhole_m(*b, hole, arg)),
            ),
            MVal::Lam(c) => MVal::Lam(Box::new(match *c {
                MClos::Plain(p) => MClos::Plain(Plain {
                    env: self.subst_mhole_env(p.env, hole, arg),
                    binder: p.binder,
                    body: p.body,
                }),
                MClos::TransWeak { inner, past } => MClos::TransWeak {
                    inner: Plain {
                        env: self.subst_mhole_env(inner.env, hole, arg),
                        binder: inner.binder,
                        body: inner.body,
                    },
                    past,
                },
                MClos::TransSub { inner, past, repl } => MClos::TransSub {
                    inner: Plain {
                        env: self.subst_mhole_env(inner.env, hole, arg),
                        binder: inner.binder,
                        body: inner.body,
                    },
                    past,
                    repl: Box::new(self.subst_mhole_m(*repl, hole, arg)),
                },
            })),
            MVal::Sort(a) => MVal::Sort(Box::new(self.subst_mhole_h(*a, hole, arg))),
            MVal::Ax(a) => MVal::Ax(Box::new(self.subst_mhole_h(*a, hole, arg))),
            MVal::Neu { head, thy } => MVal::Neu {
                head,
                thy: Box::new(self.subst_mhole_t(*thy, hole, arg)),
            },
            MVal::StuckWeak(i, p) => {
                MVal::StuckWeak(Box::new(self.subst_mhole_m(*i, hole, arg)), p)
            }
            MVal::StuckSub(i, r, p) => MVal::StuckSub(
                Box::new(self.subst_mhole_m(*i, hole, arg)),
                Box::new(self.subst_mhole_m(*r, hole, arg)),
                p,
            ),
        }
    }

    pub fn thaw_h(&self, v: HVal) -> HVal {
        match v {
            HVal::Frozen(v) => self.thaw_h(*v),
            HVal::Hole(i) => HVal::Hole(i),
            HVal::U | HVal::Nat | HVal::Empty | HVal::Unit | HVal::Z | HVal::Tt => v,
            HVal::Pi(d, c) => HVal::Pi(Box::new(self.thaw_h(*d)), Box::new(self.thaw_hclos(*c))),
            HVal::Sigma(d, c) => {
                HVal::Sigma(Box::new(self.thaw_h(*d)), Box::new(self.thaw_hclos(*c)))
            }
            HVal::Sum(a, b) => HVal::Sum(Box::new(self.thaw_h(*a)), Box::new(self.thaw_h(*b))),
            HVal::Id(t, a, b) => HVal::Id(
                Box::new(self.thaw_h(*t)),
                Box::new(self.thaw_h(*a)),
                Box::new(self.thaw_h(*b)),
            ),
            HVal::Lam(c) => HVal::Lam(Box::new(self.thaw_hclos(*c))),
            HVal::Pair(a, b) => HVal::Pair(Box::new(self.thaw_h(*a)), Box::new(self.thaw_h(*b))),
            HVal::Inl(a) => HVal::Inl(Box::new(self.thaw_h(*a))),
            HVal::Inr(a) => HVal::Inr(Box::new(self.thaw_h(*a))),
            HVal::Refl(a) => HVal::Refl(Box::new(self.thaw_h(*a))),
            HVal::S(n) => HVal::S(Box::new(self.thaw_h(*n))),
            HVal::Ty(m) => self.ty_of(self.thaw_m(*m)),
            HVal::Unax(m) => self.unax_of(self.thaw_m(*m)),
            HVal::Neu(h) => HVal::Neu(Box::new(self.thaw_head(*h))),
            HVal::StuckWeak(i, p) => HVal::StuckWeak(Box::new(self.thaw_h(*i)), p),
            HVal::StuckSub(i, m, p) => {
                HVal::StuckSub(Box::new(self.thaw_h(*i)), Box::new(self.thaw_m(*m)), p)
            }
        }
    }

    fn thaw_hclos(&self, c: HClos) -> HClos {
        match c {
            HClos::Plain(p) => HClos::Plain(Plain {
                env: self.thaw_env(p.env),
                binder: p.binder,
                body: p.body,
            }),
            // Pending transports keep Frozen arguments in their environments.
            other => other,
        }
    }

    fn thaw_env(&self, env: Env) -> Env {
        Env {
            slots: env
                .slots
                .into_iter()
                .map(|(l, s)| {
                    let s = match s {
                        Slot::H(v) => Slot::H(self.thaw_h(v)),
                        Slot::M(v) => Slot::M(self.thaw_m(v)),
                    };
                    (l, s)
                })
                .collect(),
        }
    }

    fn thaw_head(&self, h: Head) -> Head {
        match h {
            Head::Var { lvl, ty } => Head::Var {
                lvl,
                ty: Box::new(self.thaw_h(*ty)),
            },
            Head::App { fun, arg, ty } => Head::App {
                fun: Box::new(self.thaw_head(*fun)),
                arg: Box::new(self.thaw_h(*arg)),
                ty: Box::new(self.thaw_h(*ty)),
            },
            Head::Fst { of, ty } => Head::Fst {
                of: Box::new(self.thaw_head(*of)),
                ty: Box::new(self.thaw_h(*ty)),
            },
            Head::Snd { of, ty } => Head::Snd {
                of: Box::new(self.thaw_head(*of)),
                ty: Box::new(self.thaw_h(*ty)),
            },
            Head::Match {
                scrut,
                motive,
                inl,
                inr,
                ty,
            } => Head::Match {
                scrut: Box::new(self.thaw_head(*scrut)),
                motive: Box::new(self.thaw_hclos(*motive)),
                inl: Box::new(self.thaw_hclos(*inl)),
                inr: Box::new(self.thaw_hclos(*inr)),
                ty: Box::new(self.thaw_h(*ty)),
            },
            Head::J {
                path,
                env,
                y_b,
                p_b,
                motive,
                refl_case,
                ty,
            } => Head::J {
                path: Box::new(self.thaw_head(*path)),
                env: self.thaw_env(env),
                y_b,
                p_b,
                motive,
                refl_case: Box::new(self.thaw_h(*refl_case)),
                ty: Box::new(self.thaw_h(*ty)),
            },
            Head::NatInd {
                scrut,
                env,
                k_b,
                motive,
                zcase,
                m_b,
                ih_b,
                scase,
                ty,
            } => Head::NatInd {
                scrut: Box::new(self.thaw_head(*scrut)),
                env: self.thaw_env(env),
                k_b,
                motive,
                zcase: Box::new(self.thaw_h(*zcase)),
                m_b,
                ih_b,
                scase,
                ty: Box::new(self.thaw_h(*ty)),
            },
            Head::Exfalso { scrut, ty } => Head::Exfalso {
                scrut: Box::new(self.thaw_head(*scrut)),
                ty: Box::new(self.thaw_h(*ty)),
            },
        }
    }

    pub fn thaw_m(&self, m: MVal) -> MVal {
        match m {
            MVal::Frozen(v) => self.thaw_m(*v),
            MVal::Tt | MVal::MHole(_) => m,
            MVal::Pair(a, b) => MVal::Pair(Box::new(self.thaw_m(*a)), Box::new(self.thaw_m(*b))),
            MVal::Lam(c) => MVal::Lam(Box::new(match *c {
                MClos::Plain(p) => MClos::Plain(Plain {
                    env: self.thaw_env(p.env),
                    binder: p.binder,
                    body: p.body,
                }),
                other => other,
            })),
            MVal::Sort(a) => self.sort_of(self.thaw_h(*a)),
            MVal::Ax(a) => self.ax_of(self.thaw_h(*a)),
            MVal::Neu { head, thy } => MVal::Neu {
                head,
                thy: Box::new(self.thaw_t(*thy)),
            },
            MVal::StuckWeak(i, p) => MVal::StuckWeak(Box::new(self.thaw_m(*i)), p),
            MVal::StuckSub(i, r, p) => {
                MVal::StuckSub(Box::new(self.thaw_m(*i)), Box::new(self.thaw_m(*r)), p)
            }
        }
    }

    pub fn thaw_t(&self, t: TVal) -> TVal {
        match t {
            TVal::One | TVal::Sort | TVal::Record(_) => t,
            TVal::Ax(a) => TVal::Ax(Box::new(self.thaw_h(*a))),
            TVal::Sigma(fst, cod) => {
                TVal::Sigma(Box::new(self.thaw_t(*fst)), Box::new(self.thaw_tclos(*cod)))
            }
            TVal::Rtri(dom, cod) => {
                TVal::Rtri(Box::new(self.thaw_h(*dom)), Box::new(self.thaw_tclos(*cod)))
            }
            TVal::IndSum { scrut, left, right } => TVal::IndSum {
                scrut: Box::new(self.thaw_h(*scrut)),
                left: Box::new(self.thaw_tclos(*left)),
                right: Box::new(self.thaw_tclos(*right)),
            },
            TVal::IndNat { scrut, zero, succ } => TVal::IndNat {
                scrut: Box::new(self.thaw_h(*scrut)),
                zero: Box::new(self.thaw_t(*zero)),
                succ: Box::new(self.thaw_tclos(*succ)),
            },
            TVal::StuckWeak(i, p) => TVal::StuckWeak(Box::new(self.thaw_t(*i)), p),
            TVal::StuckSub(i, m, p) => {
                TVal::StuckSub(Box::new(self.thaw_t(*i)), Box::new(self.thaw_m(*m)), p)
            }
        }
    }

    fn thaw_tclos(&self, c: TClos) -> TClos {
        match c {
            TClos::Plain(p) => TClos::Plain(TPlain {
                env: self.thaw_env(p.env),
                binder: p.binder,
                body: p.body,
                model_binder: p.model_binder,
            }),
            other => other,
        }
    }

    // ----- equality -------------------------------------------------------

    pub fn eq_ty(&self, a: HVal, b: HVal) -> bool {
        let a = self.thaw_h(a);
        let b = self.thaw_h(b);
        match (a, b) {
            (HVal::U, HVal::U)
            | (HVal::Nat, HVal::Nat)
            | (HVal::Empty, HVal::Empty)
            | (HVal::Unit, HVal::Unit) => true,
            (HVal::Pi(d1, c1), HVal::Pi(d2, c2)) => {
                if !self.eq_ty(*d1.clone(), *d2) {
                    return false;
                }
                let x = self.fresh_neu(&d1);
                self.eq_ty(self.apply_hclos(*c1, x.clone()), self.apply_hclos(*c2, x))
            }
            (HVal::Sigma(d1, c1), HVal::Sigma(d2, c2)) => {
                if !self.eq_ty(*d1.clone(), *d2) {
                    return false;
                }
                let x = self.fresh_neu(&d1);
                self.eq_ty(self.apply_hclos(*c1, x.clone()), self.apply_hclos(*c2, x))
            }
            (HVal::Sum(a, b), HVal::Sum(c, d)) => self.eq_ty(*a, *c) && self.eq_ty(*b, *d),
            (HVal::Id(t1, a1, b1), HVal::Id(t2, a2, b2)) => {
                self.eq_ty(*t1.clone(), *t2)
                    && self.eq_tm(&t1, *a1, *a2)
                    && self.eq_tm(&t1, *b1, *b2)
            }
            (HVal::Ty(m1), HVal::Ty(m2)) => self.eq_m(*m1, *m2),
            (HVal::StuckWeak(v1, p1), HVal::StuckWeak(v2, p2)) => p1 == p2 && self.eq_ty(*v1, *v2),
            (HVal::StuckSub(v1, m1, p1), HVal::StuckSub(v2, m2, p2)) => {
                p1 == p2 && self.eq_m(*m1, *m2) && self.eq_ty(*v1, *v2)
            }
            (HVal::Neu(h1), HVal::Neu(h2)) => self.eq_head(&h1, &h2),
            _ => false,
        }
    }

    pub fn eq_tm(&self, ty: &HVal, a: HVal, b: HVal) -> bool {
        let ty = self.thaw_h(ty.clone());
        let a = self.thaw_h(a);
        let b = self.thaw_h(b);
        match &ty {
            HVal::Pi(dom, cod) => {
                let x = self.fresh_neu(dom);
                let ta = self.apply_h(a, x.clone());
                let tb = self.apply_h(b, x.clone());
                let tb_ty = self.apply_hclos((**cod).clone(), x);
                self.eq_tm(&tb_ty, ta, tb)
            }
            HVal::Sigma(dom, cod) => {
                let a1 = self.fst_h(a.clone());
                let b1 = self.fst_h(b.clone());
                if !self.eq_tm(dom, a1.clone(), b1) {
                    return false;
                }
                let cty = self.apply_hclos((**cod).clone(), a1);
                self.eq_tm(&cty, self.snd_h(a), self.snd_h(b))
            }
            HVal::Unit => true,
            HVal::U => self.eq_ty(a, b),
            HVal::Id(carrier, _, _) => match (&a, &b) {
                (HVal::Refl(p), HVal::Refl(q)) => self.eq_tm(carrier, (**p).clone(), (**q).clone()),
                _ => self.eq_structural(a, b),
            },
            HVal::Sum(lt, rt) => match (&a, &b) {
                (HVal::Inl(x), HVal::Inl(y)) => self.eq_tm(lt, (**x).clone(), (**y).clone()),
                (HVal::Inr(x), HVal::Inr(y)) => self.eq_tm(rt, (**x).clone(), (**y).clone()),
                _ => self.eq_structural(a, b),
            },
            _ => self.eq_structural(a, b),
        }
    }

    fn eq_structural(&self, a: HVal, b: HVal) -> bool {
        let a = self.thaw_h(a);
        let b = self.thaw_h(b);
        match (a, b) {
            (HVal::Z, HVal::Z)
            | (HVal::Tt, HVal::Tt)
            | (HVal::U, HVal::U)
            | (HVal::Nat, HVal::Nat)
            | (HVal::Empty, HVal::Empty)
            | (HVal::Unit, HVal::Unit) => true,
            (HVal::S(n), HVal::S(m)) => self.eq_structural(*n, *m),
            (HVal::Inl(a), HVal::Inl(b)) | (HVal::Inr(a), HVal::Inr(b)) => {
                self.eq_structural(*a, *b)
            }
            (HVal::Refl(a), HVal::Refl(b)) => self.eq_structural(*a, *b),
            (HVal::Pair(a1, b1), HVal::Pair(a2, b2)) => {
                self.eq_structural(*a1, *a2) && self.eq_structural(*b1, *b2)
            }
            (HVal::Ty(m1), HVal::Ty(m2)) => self.eq_m(*m1, *m2),
            (HVal::Unax(m1), HVal::Unax(m2)) => self.eq_m(*m1, *m2),
            (HVal::Neu(h1), HVal::Neu(h2)) => self.eq_head(&h1, &h2),
            (HVal::StuckWeak(v1, p1), HVal::StuckWeak(v2, p2)) => {
                p1 == p2 && self.eq_structural(*v1, *v2)
            }
            (HVal::StuckSub(v1, m1, p1), HVal::StuckSub(v2, m2, p2)) => {
                p1 == p2 && self.eq_m(*m1, *m2) && self.eq_structural(*v1, *v2)
            }
            (HVal::Lam(c1), HVal::Lam(c2)) => {
                let x = self.fresh_neu(&HVal::U);
                self.eq_structural(self.apply_hclos(*c1, x.clone()), self.apply_hclos(*c2, x))
            }
            _ => false,
        }
    }

    fn eq_match(
        &self,
        s1: &Head,
        s2: &Head,
        m1: &HClos,
        m2: &HClos,
        l1: &HClos,
        l2: &HClos,
        r1: &HClos,
        r2: &HClos,
    ) -> bool {
        if !self.eq_head(s1, s2) {
            return false;
        }
        let HVal::Sum(lt, rt) = self.thaw_h(head_ty(s1)) else {
            return false;
        };
        let left = self.fresh_neu(&lt);
        let right = self.fresh_neu(&rt);
        let mot_l = self.apply_hclos(m1.clone(), HVal::Inl(Box::new(left.clone())));
        let mot_l2 = self.apply_hclos(m2.clone(), HVal::Inl(Box::new(left.clone())));
        let mot_r = self.apply_hclos(m1.clone(), HVal::Inr(Box::new(right.clone())));
        let mot_r2 = self.apply_hclos(m2.clone(), HVal::Inr(Box::new(right.clone())));
        self.eq_ty(mot_l.clone(), mot_l2)
            && self.eq_ty(mot_r.clone(), mot_r2.clone())
            && self.eq_tm(
                &mot_l,
                self.apply_hclos(l1.clone(), left.clone()),
                self.apply_hclos(l2.clone(), left),
            )
            && self.eq_tm(
                &mot_r,
                self.apply_hclos(r1.clone(), right.clone()),
                self.apply_hclos(r2.clone(), right),
            )
    }

    /// Two stuck path inductions agree when the paths agree, the motives agree
    /// on a fresh endpoint of the carrier, and the reflexivity cases agree there.
    fn eq_j(
        &self,
        p1: &Head,
        p2: &Head,
        e1: &Env,
        e2: &Env,
        y1: u32,
        y2: u32,
        q1: u32,
        q2: u32,
        m1: &HTm,
        m2: &HTm,
        c1: &HVal,
        c2: &HVal,
    ) -> bool {
        if !self.eq_head(p1, p2) {
            return false;
        }
        let HVal::Id(carrier, left, _) = self.thaw_h(head_ty(p1)) else {
            return false;
        };
        let y = self.fresh_neu(&carrier);
        let path = self.fresh_neu(&HVal::Id(
            carrier.clone(),
            left.clone(),
            Box::new(y.clone()),
        ));
        let t1 = self.eval_h(
            m1,
            &e1.insert(y1, Slot::H(y.clone()))
                .insert(q1, Slot::H(path.clone())),
        );
        let t2 = self.eval_h(m2, &e2.insert(y2, Slot::H(y)).insert(q2, Slot::H(path)));
        if !self.eq_ty(t1, t2) {
            return false;
        }
        let at_refl = self.eval_h(
            m1,
            &e1.insert(y1, Slot::H((*left).clone()))
                .insert(q1, Slot::H(HVal::Refl(left.clone()))),
        );
        self.eq_tm(&at_refl, c1.clone(), c2.clone())
    }

    fn eq_natind(
        &self,
        s1: &Head,
        s2: &Head,
        e1: &Env,
        e2: &Env,
        k1: u32,
        k2: u32,
        m1: &HTm,
        m2: &HTm,
        z1: &HVal,
        z2: &HVal,
        n1: u32,
        n2: u32,
        i1: u32,
        i2: u32,
        c1: &HTm,
        c2: &HTm,
    ) -> bool {
        if !self.eq_head(s1, s2) {
            return false;
        }
        let k = self.fresh_neu(&HVal::Nat);
        let t1 = self.eval_h(m1, &e1.insert(k1, Slot::H(k.clone())));
        let t2 = self.eval_h(m2, &e2.insert(k2, Slot::H(k)));
        if !self.eq_ty(t1, t2) {
            return false;
        }
        let zmot = self.eval_h(m1, &e1.insert(k1, Slot::H(HVal::Z)));
        if !self.eq_tm(&zmot, z1.clone(), z2.clone()) {
            return false;
        }
        let n = self.fresh_neu(&HVal::Nat);
        let ih_ty = self.eval_h(m1, &e1.insert(k1, Slot::H(n.clone())));
        let ih = self.fresh_neu(&ih_ty);
        let smot = self.eval_h(m1, &e1.insert(k1, Slot::H(HVal::S(Box::new(n.clone())))));
        let b1 = self.eval_h(
            c1,
            &e1.insert(n1, Slot::H(n.clone()))
                .insert(i1, Slot::H(ih.clone())),
        );
        let b2 = self.eval_h(c2, &e2.insert(n2, Slot::H(n)).insert(i2, Slot::H(ih)));
        self.eq_tm(&smot, b1, b2)
    }

    fn eq_head(&self, a: &Head, b: &Head) -> bool {
        match (a, b) {
            (Head::Var { lvl: l1, .. }, Head::Var { lvl: l2, .. }) => l1 == l2,
            (
                Head::App {
                    fun: f1, arg: a1, ..
                },
                Head::App {
                    fun: f2, arg: a2, ..
                },
            ) => self.eq_head(f1, f2) && self.eq_structural((**a1).clone(), (**a2).clone()),
            (Head::Fst { of: o1, .. }, Head::Fst { of: o2, .. }) => self.eq_head(o1, o2),
            (Head::Snd { of: o1, .. }, Head::Snd { of: o2, .. }) => self.eq_head(o1, o2),
            (
                Head::Match {
                    scrut: s1,
                    motive: m1,
                    inl: l1,
                    inr: r1,
                    ..
                },
                Head::Match {
                    scrut: s2,
                    motive: m2,
                    inl: l2,
                    inr: r2,
                    ..
                },
            ) => self.eq_match(s1, s2, m1, m2, l1, l2, r1, r2),
            (
                Head::J {
                    path: p1,
                    env: e1,
                    y_b: y1,
                    p_b: q1,
                    motive: m1,
                    refl_case: c1,
                    ..
                },
                Head::J {
                    path: p2,
                    env: e2,
                    y_b: y2,
                    p_b: q2,
                    motive: m2,
                    refl_case: c2,
                    ..
                },
            ) => self.eq_j(p1, p2, e1, e2, *y1, *y2, *q1, *q2, m1, m2, c1, c2),
            (
                Head::NatInd {
                    scrut: s1,
                    env: e1,
                    k_b: k1,
                    motive: m1,
                    zcase: z1,
                    m_b: n1,
                    ih_b: i1,
                    scase: c1,
                    ..
                },
                Head::NatInd {
                    scrut: s2,
                    env: e2,
                    k_b: k2,
                    motive: m2,
                    zcase: z2,
                    m_b: n2,
                    ih_b: i2,
                    scase: c2,
                    ..
                },
            ) => self.eq_natind(
                s1, s2, e1, e2, *k1, *k2, m1, m2, z1, z2, *n1, *n2, *i1, *i2, c1, c2,
            ),
            (Head::Exfalso { scrut: s1, .. }, Head::Exfalso { scrut: s2, .. }) => {
                self.eq_head(s1, s2)
            }
            _ => false,
        }
    }

    pub fn eq_m(&self, a: MVal, b: MVal) -> bool {
        let a = self.thaw_m(a);
        let b = self.thaw_m(b);
        match (a, b) {
            (MVal::Tt, MVal::Tt) => true,
            (MVal::Pair(a1, b1), MVal::Pair(a2, b2)) => self.eq_m(*a1, *a2) && self.eq_m(*b1, *b2),
            (MVal::Pair(a1, b1), other) | (other, MVal::Pair(a1, b1)) => {
                self.eq_m(*a1, self.pr1(other.clone())) && self.eq_m(*b1, self.pr2(other))
            }
            (MVal::Lam(c1), MVal::Lam(c2)) => {
                let x = self.fresh_neu(&HVal::U);
                self.eq_m(self.apply_mclos(*c1, x.clone()), self.apply_mclos(*c2, x))
            }
            (MVal::Lam(c), other) | (other, MVal::Lam(c)) => {
                let x = self.fresh_neu(&HVal::U);
                self.eq_m(self.apply_mclos(*c, x.clone()), self.apply_m(other, x))
            }
            (MVal::Sort(a), MVal::Sort(b)) => self.eq_ty(*a, *b),
            (MVal::Ax(a), MVal::Ax(b)) => self.eq_structural(*a, *b),
            (MVal::Neu { head: h1, .. }, MVal::Neu { head: h2, .. }) => self.eq_mhead(&h1, &h2),
            (MVal::StuckWeak(v1, p1), MVal::StuckWeak(v2, p2)) => p1 == p2 && self.eq_m(*v1, *v2),
            (MVal::StuckSub(v1, m1, p1), MVal::StuckSub(v2, m2, p2)) => {
                p1 == p2 && self.eq_m(*m1, *m2) && self.eq_m(*v1, *v2)
            }
            _ => false,
        }
    }

    fn eq_mhead(&self, a: &MHead, b: &MHead) -> bool {
        match (a, b) {
            (MHead::Var { lvl: l1 }, MHead::Var { lvl: l2 }) => l1 == l2,
            (MHead::Pr1(a), MHead::Pr1(b)) | (MHead::Pr2(a), MHead::Pr2(b)) => self.eq_mhead(a, b),
            (MHead::App { fun: f1, arg: a1 }, MHead::App { fun: f2, arg: a2 }) => {
                self.eq_mhead(f1, f2) && self.eq_structural((**a1).clone(), (**a2).clone())
            }
            (MHead::IndSum { scrut: s1 }, MHead::IndSum { scrut: s2 })
            | (MHead::IndNat { scrut: s1 }, MHead::IndNat { scrut: s2 }) => {
                self.eq_structural((**s1).clone(), (**s2).clone())
            }
            _ => false,
        }
    }

    pub fn eq_t(&self, a: &TVal, b: &TVal) -> bool {
        match (a, b) {
            (TVal::One, TVal::One) | (TVal::Sort, TVal::Sort) => true,
            (TVal::Ax(a), TVal::Ax(b)) => self.eq_ty((**a).clone(), (**b).clone()),
            (TVal::Sigma(f1, c1), TVal::Sigma(f2, c2)) => {
                if !self.eq_t(f1, f2) {
                    return false;
                }
                let x = MVal::Neu {
                    head: Box::new(MHead::Var {
                        lvl: self.fresh_id(),
                    }),
                    thy: f1.clone(),
                };
                let a = self.inst_sigma(c1, &x);
                let b = self.inst_sigma(c2, &x);
                self.eq_t(&a, &b)
            }
            (TVal::Rtri(d1, c1), TVal::Rtri(d2, c2)) => {
                if !self.eq_ty((**d1).clone(), (**d2).clone()) {
                    return false;
                }
                let x = self.fresh_neu(d1);
                let a = self.apply_t_host(c1, x.clone());
                let b = self.apply_t_host(c2, x);
                self.eq_t(&a, &b)
            }
            (TVal::StuckWeak(v1, p1), TVal::StuckWeak(v2, p2)) => p1 == p2 && self.eq_t(v1, v2),
            (TVal::StuckSub(v1, m1, p1), TVal::StuckSub(v2, m2, p2)) => {
                p1 == p2 && self.eq_m((**m1).clone(), (**m2).clone()) && self.eq_t(v1, v2)
            }
            (TVal::Record(r1), TVal::Record(r2)) => eq_record(self, r1, r2),
            (
                TVal::IndSum {
                    scrut: s1,
                    left: l1,
                    right: r1,
                },
                TVal::IndSum {
                    scrut: s2,
                    left: l2,
                    right: r2,
                },
            ) => {
                self.eq_structural((**s1).clone(), (**s2).clone())
                    && {
                        let x = self.fresh_neu(&HVal::U);
                        let a = self.apply_t_host(l1, x.clone());
                        let b = self.apply_t_host(l2, x);
                        self.eq_t(&a, &b)
                    }
                    && {
                        let x = self.fresh_neu(&HVal::U);
                        let a = self.apply_t_host(r1, x.clone());
                        let b = self.apply_t_host(r2, x);
                        self.eq_t(&a, &b)
                    }
            }
            (
                TVal::IndNat {
                    scrut: s1,
                    zero: z1,
                    succ: c1,
                },
                TVal::IndNat {
                    scrut: s2,
                    zero: z2,
                    succ: c2,
                },
            ) => {
                self.eq_structural((**s1).clone(), (**s2).clone()) && self.eq_t(z1, z2) && {
                    let x = self.fresh_neu(&HVal::Nat);
                    let a = self.apply_t_host(c1, x.clone());
                    let b = self.apply_t_host(c2, x);
                    self.eq_t(&a, &b)
                }
            }
            _ => false,
        }
    }

    pub fn is_small(&self, v: &HVal) -> bool {
        match self.thaw_h_ref(v) {
            HVal::Nat | HVal::Empty | HVal::Unit => true,
            HVal::Sum(a, b) => self.is_small(a) && self.is_small(b),
            HVal::Sigma(a, _) | HVal::Pi(a, _) => self.is_small(a),
            HVal::Id(t, _, _) => self.is_small(t),
            HVal::Neu(h) => matches!(head_ty(h), HVal::U),
            _ => false,
        }
    }

    fn thaw_h_ref<'a>(&self, v: &'a HVal) -> &'a HVal {
        match v {
            HVal::Frozen(inner) => self.thaw_h_ref(inner),
            _ => v,
        }
    }

    pub fn is_type(&self, v: &HVal) -> bool {
        match self.thaw_h(v.clone()) {
            HVal::U
            | HVal::Nat
            | HVal::Empty
            | HVal::Unit
            | HVal::Pi(_, _)
            | HVal::Sigma(_, _)
            | HVal::Sum(_, _)
            | HVal::Id(_, _, _)
            | HVal::Ty(_) => true,
            HVal::StuckWeak(inner, _) | HVal::StuckSub(inner, _, _) => self.is_type(&inner),
            HVal::Neu(h) => matches!(head_ty(&h), HVal::U),
            _ => false,
        }
    }

    pub fn pp_h(&self, v: &HVal) -> String {
        pp_h(self, v, 0)
    }

    pub fn pp_t(&self, v: &TVal) -> String {
        pp_t(self, v, 0)
    }

    pub fn pp_m(&self, v: &MVal) -> String {
        pp_m(self, v, 0)
    }
}

fn head_ty(h: &Head) -> HVal {
    match h {
        Head::Var { ty, .. }
        | Head::App { ty, .. }
        | Head::Fst { ty, .. }
        | Head::Snd { ty, .. }
        | Head::Match { ty, .. }
        | Head::J { ty, .. }
        | Head::NatInd { ty, .. }
        | Head::Exfalso { ty, .. } => (**ty).clone(),
    }
}

fn wrap_h(tm: HTm, trav: &Trav) -> HTm {
    match trav {
        Trav::Weak { past } => HTm::Weak {
            tm: Box::new(tm),
            past: *past,
        },
        Trav::Sub { past, repl } => HTm::MSub {
            tm: Box::new(tm),
            repl: MRepl::Val(Box::new(repl.clone())),
            past: *past,
        },
    }
}

fn transport_con(c: Con, trav: &Trav) -> Con {
    match c {
        Con::Var(l) => Con::Var(l),
        Con::Field { idx, host, args } => Con::Field {
            idx,
            host: host.into_iter().map(|h| wrap_h(h, trav)).collect(),
            args: args.into_iter().map(|a| transport_con(a, trav)).collect(),
        },
        Con::Sigma { binder, dom, cod } => Con::Sigma {
            binder,
            dom: Box::new(transport_con(*dom, trav)),
            cod: Box::new(transport_con(*cod, trav)),
        },
        Con::Pair(a, b) => Con::Pair(
            Box::new(transport_con(*a, trav)),
            Box::new(transport_con(*b, trav)),
        ),
        Con::Fst(a) => Con::Fst(Box::new(transport_con(*a, trav))),
        Con::Snd(a) => Con::Snd(Box::new(transport_con(*a, trav))),
        Con::Id(t, a, b) => Con::Id(
            Box::new(transport_con(*t, trav)),
            Box::new(transport_con(*a, trav)),
            Box::new(transport_con(*b, trav)),
        ),
        Con::Refl(a) => Con::Refl(Box::new(transport_con(*a, trav))),
        Con::Empty => Con::Empty,
        Con::Exfalso { motive, scrut } => Con::Exfalso {
            motive: Box::new(transport_con(*motive, trav)),
            scrut: Box::new(transport_con(*scrut, trav)),
        },
        Con::Unit => Con::Unit,
        Con::Tt => Con::Tt,
        Con::Sum(a, b) => Con::Sum(
            Box::new(transport_con(*a, trav)),
            Box::new(transport_con(*b, trav)),
        ),
        Con::Inl(a) => Con::Inl(Box::new(transport_con(*a, trav))),
        Con::Inr(a) => Con::Inr(Box::new(transport_con(*a, trav))),
        Con::Match {
            scrut,
            x,
            extra,
            motive,
            inl_b,
            inl_extra,
            inl,
            inr_b,
            inr_extra,
            inr,
            theta,
        } => Con::Match {
            scrut: Box::new(transport_con(*scrut, trav)),
            x,
            extra: extra
                .into_iter()
                .map(|(l, c)| (l, transport_con(c, trav)))
                .collect(),
            motive: Box::new(transport_con(*motive, trav)),
            inl_b,
            inl_extra,
            inl: Box::new(transport_con(*inl, trav)),
            inr_b,
            inr_extra,
            inr: Box::new(transport_con(*inr, trav)),
            theta: theta.into_iter().map(|c| transport_con(c, trav)).collect(),
        },
        Con::Trunc(a) => Con::Trunc(Box::new(transport_con(*a, trav))),
        Con::TIn(a) => Con::TIn(Box::new(transport_con(*a, trav))),
        Con::TElim {
            scrut,
            x,
            motive,
            p_b,
            q_b,
            proof,
            a_b,
            into,
        } => Con::TElim {
            scrut: Box::new(transport_con(*scrut, trav)),
            x,
            motive: Box::new(transport_con(*motive, trav)),
            p_b,
            q_b,
            proof: Box::new(transport_con(*proof, trav)),
            a_b,
            into: Box::new(transport_con(*into, trav)),
        },
        Con::Nat => Con::Nat,
        Con::Z => Con::Z,
        Con::S(a) => Con::S(Box::new(transport_con(*a, trav))),
        Con::NInd {
            scrut,
            k,
            motive,
            zcase,
            m_b,
            ih_b,
            scase,
        } => Con::NInd {
            scrut: Box::new(wrap_h(*scrut, trav)),
            k,
            motive: Box::new(transport_con(*motive, trav)),
            zcase: Box::new(transport_con(*zcase, trav)),
            m_b,
            ih_b,
            scase: Box::new(transport_con(*scase, trav)),
        },
        Con::Ax(a) => Con::Ax(Box::new(wrap_h(*a, trav))),
        Con::AxIn(a) => Con::AxIn(Box::new(wrap_h(*a, trav))),
        Con::LetAx {
            binder,
            scrut,
            body,
        } => Con::LetAx {
            binder,
            scrut: Box::new(transport_con(*scrut, trav)),
            body: Box::new(transport_con(*body, trav)),
        },
        Con::J {
            y_b,
            p_b,
            motive,
            refl_case,
            path,
        } => Con::J {
            y_b,
            p_b,
            motive: Box::new(transport_con(*motive, trav)),
            refl_case: Box::new(transport_con(*refl_case, trav)),
            path: Box::new(transport_con(*path, trav)),
        },
    }
}

fn eq_record(nbe: &Nbe, a: &RecordTheory, b: &RecordTheory) -> bool {
    if a.fields.len() != b.fields.len() {
        return false;
    }
    let env = Env::new();
    for (f, g) in a.fields.iter().zip(b.fields.iter()) {
        if f.name != g.name || f.delta.len() != g.delta.len() || f.phi.len() != g.phi.len() {
            return false;
        }
        for ((_, _, ty1), (_, _, ty2)) in f.delta.iter().zip(g.delta.iter()) {
            let v1 = nbe.eval_h(ty1, &env);
            let v2 = nbe.eval_h(ty2, &env);
            if !nbe.eq_ty(v1, v2) {
                return false;
            }
        }
        if !eq_con(nbe, &f.phi, &g.phi) {
            return false;
        }
        match (&f.kind, &g.kind) {
            (RecKind::Sort, RecKind::Sort) => {}
            (RecKind::Term(c1), RecKind::Term(c2)) => {
                if !eq_con_one(nbe, c1, c2) {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

fn eq_con(nbe: &Nbe, a: &[(u32, String, Con)], b: &[(u32, String, Con)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|((_, _, c1), (_, _, c2))| eq_con_one(nbe, c1, c2))
}

fn eq_con_one(nbe: &Nbe, a: &Con, b: &Con) -> bool {
    match (a, b) {
        (Con::Var(i), Con::Var(j)) => i == j,
        (
            Con::Field {
                idx: i1,
                host: h1,
                args: a1,
            },
            Con::Field {
                idx: i2,
                host: h2,
                args: a2,
            },
        ) => {
            i1 == i2
                && h1.len() == h2.len()
                && h1.iter().zip(h2.iter()).all(|(x, y)| {
                    let env = Env::new();
                    nbe.eq_ty(nbe.eval_h(x, &env), nbe.eval_h(y, &env))
                        || nbe.eq_structural(nbe.eval_h(x, &env), nbe.eval_h(y, &env))
                })
                && a1.len() == a2.len()
                && a1.iter().zip(a2.iter()).all(|(x, y)| eq_con_one(nbe, x, y))
        }
        (Con::Empty, Con::Empty) | (Con::Unit, Con::Unit) | (Con::Tt, Con::Tt) => true,
        (Con::Nat, Con::Nat) | (Con::Z, Con::Z) => true,
        (Con::Ax(a), Con::Ax(b)) | (Con::AxIn(a), Con::AxIn(b)) => {
            let env = Env::new();
            let va = nbe.eval_h(a, &env);
            let vb = nbe.eval_h(b, &env);
            nbe.eq_ty(va.clone(), vb.clone()) || nbe.eq_structural(va, vb)
        }
        _ => format!("{a:?}") == format!("{b:?}"),
    }
}

fn pp_h(nbe: &Nbe, v: &HVal, d: u32) -> String {
    if d > 6 {
        return "…".into();
    }
    match v {
        HVal::U => "U".into(),
        HVal::Nat => "Nat".into(),
        HVal::Empty => "Empty".into(),
        HVal::Unit => "Unit".into(),
        HVal::Z => "zero".into(),
        HVal::Tt => "tt".into(),
        HVal::S(n) => format!("succ {}", pp_h(nbe, n, d + 1)),
        HVal::Pi(dom, _) => format!("(Π {} → …)", pp_h(nbe, dom, d + 1)),
        HVal::Sigma(dom, _) => format!("(Σ {} × …)", pp_h(nbe, dom, d + 1)),
        HVal::Sum(a, b) => format!("({} + {})", pp_h(nbe, a, d + 1), pp_h(nbe, b, d + 1)),
        HVal::Id(t, a, b) => format!(
            "Id {} {} {}",
            pp_h(nbe, t, d + 1),
            pp_h(nbe, a, d + 1),
            pp_h(nbe, b, d + 1)
        ),
        HVal::Lam(_) => "λ".into(),
        HVal::Pair(a, b) => format!("({}, {})", pp_h(nbe, a, d + 1), pp_h(nbe, b, d + 1)),
        HVal::Inl(a) => format!("inl {}", pp_h(nbe, a, d + 1)),
        HVal::Inr(a) => format!("inr {}", pp_h(nbe, a, d + 1)),
        HVal::Refl(a) => format!("refl {}", pp_h(nbe, a, d + 1)),
        HVal::Ty(m) => format!("ty {}", pp_m(nbe, m, d + 1)),
        HVal::Unax(m) => format!("unax {}", pp_m(nbe, m, d + 1)),
        HVal::Neu(h) => format!("#{}", head_lvl(h)),
        HVal::StuckWeak(i, p) => format!("({} ↑ {})", pp_h(nbe, i, d + 1), p),
        HVal::StuckSub(i, _, p) => format!("({} {{{}/}})", pp_h(nbe, i, d + 1), p),
        HVal::Hole(i) => format!("hole{i}"),
        HVal::Frozen(i) => pp_h(nbe, i, d),
    }
}

fn head_lvl(h: &Head) -> String {
    match h {
        Head::Var { lvl, .. } => lvl.to_string(),
        Head::App { fun, .. } => format!("app{}", head_lvl(fun)),
        Head::Fst { of, .. } => format!("fst{}", head_lvl(of)),
        Head::Snd { of, .. } => format!("snd{}", head_lvl(of)),
        Head::Match { scrut, .. } => format!("match{}", head_lvl(scrut)),
        Head::J { path, .. } => format!("j{}", head_lvl(path)),
        Head::NatInd { scrut, .. } => format!("nat{}", head_lvl(scrut)),
        Head::Exfalso { scrut, .. } => format!("ef{}", head_lvl(scrut)),
    }
}

fn pp_m(nbe: &Nbe, v: &MVal, d: u32) -> String {
    if d > 6 {
        return "…".into();
    }
    match v {
        MVal::Tt => "★".into(),
        MVal::Pair(a, b) => format!("({}, {})", pp_m(nbe, a, d + 1), pp_m(nbe, b, d + 1)),
        MVal::Lam(_) => "λ".into(),
        MVal::Sort(a) => format!("sort {}", pp_h(nbe, a, d + 1)),
        MVal::Ax(a) => format!("ax {}", pp_h(nbe, a, d + 1)),
        MVal::Neu { head, .. } => format!("m{}", pp_mhead(head)),
        MVal::StuckWeak(i, p) => format!("({} ↑ {})", pp_m(nbe, i, d + 1), p),
        MVal::StuckSub(i, _, p) => format!("({} {{{}}})", pp_m(nbe, i, d + 1), p),
        MVal::MHole(i) => format!("mhole{i}"),
        MVal::Frozen(i) => pp_m(nbe, i, d),
    }
}

fn pp_mhead(h: &MHead) -> String {
    match h {
        MHead::Var { lvl } => lvl.to_string(),
        MHead::Pr1(h) => format!("pr1{}", pp_mhead(h)),
        MHead::Pr2(h) => format!("pr2{}", pp_mhead(h)),
        MHead::App { fun, .. } => format!("app{}", pp_mhead(fun)),
        MHead::IndSum { .. } => "ind+".into(),
        MHead::IndNat { .. } => "indN".into(),
    }
}

fn pp_t(nbe: &Nbe, v: &TVal, d: u32) -> String {
    if d > 6 {
        return "…".into();
    }
    match v {
        TVal::One => "One".into(),
        TVal::Sort => "Sort".into(),
        TVal::Ax(a) => format!("Ax {}", pp_h(nbe, a, d + 1)),
        TVal::Sigma(fst, _) => format!("(Σ {} :: …)", pp_t(nbe, fst, d + 1)),
        TVal::Rtri(dom, _) => format!("({} ▹ …)", pp_h(nbe, dom, d + 1)),
        TVal::IndSum { .. } => "Ind+".into(),
        TVal::IndNat { .. } => "IndN".into(),
        TVal::Record(r) => format!(
            "record[{}]",
            r.fields
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>()
                .join(",")
        ),
        TVal::StuckWeak(i, p) => format!("({} ↑ {})", pp_t(nbe, i, d + 1), p),
        TVal::StuckSub(i, _, p) => format!("({} {{{}}})", pp_t(nbe, i, d + 1), p),
    }
}

/// Names currently in scope. Used by elaboration, not by normalization.
#[derive(Clone, Debug)]
pub struct NameEnv {
    pub theories: BTreeMap<String, TVal>,
}
