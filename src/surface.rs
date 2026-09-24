//! Concrete syntax, shared by the parser and the elaborator.

use crate::error::Span;

#[derive(Clone, Debug)]
pub struct E {
    pub sp: Span,
    pub kind: Ek,
}

#[derive(Clone, Debug)]
pub enum Ek {
    Name(String),
    Tt,
    U,
    Nat,
    Empty,
    Unit,
    One,
    Sort,
    Zero,
    Arrow {
        binder: Option<String>,
        dom: Box<E>,
        cod: Box<E>,
    },
    /// Host `Σ` when `model` is false, theory `Σ` when `model` is true.
    Prod {
        binder: Option<String>,
        model: bool,
        dom: Box<E>,
        cod: Box<E>,
    },
    Rtri {
        binder: Option<String>,
        dom: Box<E>,
        cod: Box<E>,
    },
    Lam {
        name: String,
        body: Box<E>,
    },
    App(Box<E>, Box<E>),
    Pair(Box<E>, Box<E>),
    /// `name(host… ; con…)` — also used for curried host calls.
    Call {
        name: String,
        host: Vec<E>,
        con: Vec<E>,
        semi: bool,
    },
    Weak {
        tm: Box<E>,
        name: String,
    },
    Subst {
        tm: Box<E>,
        repl: Box<E>,
        name: String,
    },
    Un(UnOp, Box<E>),
    Id {
        ty: Option<Box<E>>,
        a: Box<E>,
        b: Box<E>,
    },
    Sum(Box<E>, Box<E>),
    Match {
        scrut: Box<E>,
        binder: String,
        motive: Box<E>,
        inl_name: String,
        inl: Box<E>,
        inr_name: String,
        inr: Box<E>,
    },
    J {
        path: Box<E>,
        y: String,
        p: String,
        motive: Box<E>,
        refl_case: Box<E>,
    },
    NatInd {
        scrut: Box<E>,
        k: String,
        motive: Box<E>,
        zcase: Box<E>,
        m: String,
        ih: String,
        scase: Box<E>,
    },
    /// Large elimination into a theory.
    IndTy {
        scrut: Box<E>,
        lname: String,
        left: Box<E>,
        rname: String,
        right: Box<E>,
    },
    /// Elimination into a model, motive explicit.
    IndM {
        scrut: Box<E>,
        x: String,
        motive: Box<E>,
        lname: String,
        left: Box<E>,
        rname: String,
        right: Box<E>,
    },
    LetAx {
        name: String,
        scrut: Box<E>,
        body: Box<E>,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum UnOp {
    Fst,
    Snd,
    Pr1,
    Pr2,
    Inl,
    Inr,
    Succ,
    Refl,
    Ty,
    Unax,
    SortM,
    AxM,
    AxTy,
    Trunc,
    Abort,
}

#[derive(Clone, Debug)]
pub struct BinderS {
    pub sp: Span,
    pub name: String,
    pub model: bool,
    pub ty: E,
}

#[derive(Clone, Debug)]
pub struct TeleParam {
    pub name: String,
    pub ty: E,
}

#[derive(Clone, Debug)]
pub enum FieldS {
    Sort {
        sp: Span,
        name: String,
        delta: Vec<TeleParam>,
        phi: Vec<TeleParam>,
    },
    Term {
        sp: Span,
        name: String,
        delta: Vec<TeleParam>,
        phi: Vec<TeleParam>,
        ty: E,
    },
}

#[derive(Clone, Debug)]
pub enum Item {
    Postulate {
        sp: Span,
        name: String,
        ty: E,
    },
    Def {
        sp: Span,
        name: String,
        ty: E,
        body: E,
    },
    TheoryB {
        sp: Span,
        name: String,
        body: E,
    },
    TheoryA {
        sp: Span,
        name: String,
        fields: Vec<FieldS>,
    },
    ModelB {
        sp: Span,
        name: String,
        thy: String,
        body: E,
    },
    ModelA {
        sp: Span,
        name: String,
        thy: String,
        fields: Vec<(Span, String, E)>,
    },
    Context {
        sp: Span,
        binders: Vec<BinderS>,
        items: Vec<Item>,
    },
    CheckType {
        sp: Span,
        expr: E,
    },
    CheckTerm {
        sp: Span,
        expr: E,
        ty: E,
    },
    CheckModel {
        sp: Span,
        expr: E,
        thy: E,
    },
    Defeq {
        sp: Span,
        negate: bool,
        a: E,
        b: E,
        ty: Option<E>,
    },
}
