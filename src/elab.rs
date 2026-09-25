//! Elaboration from concrete syntax into the kernel.

use crate::error::{Error, Span};
use crate::nbe::*;
use crate::surface::*;
use std::collections::BTreeMap;

#[derive(Clone)]
struct Bnd {
    name: String,
    lvl: u32,
    kind: Kind,
}

#[derive(Clone)]
enum Kind {
    Term(HVal),
    Model(TVal),
}

struct Ctx {
    binders: Vec<Bnd>,
    env: Env,
    next: u32,
    theories: BTreeMap<String, TVal>,
    models: BTreeMap<String, (MVal, TVal)>,
    nbe: Nbe,
}

pub fn check_items(items: &[Item]) -> Result<(), Error> {
    let mut ctx = Ctx {
        binders: Vec::new(),
        env: Env::new(),
        next: 0,
        theories: BTreeMap::new(),
        models: BTreeMap::new(),
        nbe: Nbe::new(1_000_000),
    };
    ctx.items(items)
}

impl Ctx {
    fn fresh(&mut self) -> u32 {
        let l = self.next;
        self.next += 1;
        // Keep normalization holes out of the user-level range.
        if self.next > 900_000 {
            panic!("too many binders");
        }
        l
    }

    fn fail<T>(&self, sp: Span, msg: impl Into<String>) -> Result<T, Error> {
        Err(Error::at(sp, msg))
    }

    fn items(&mut self, items: &[Item]) -> Result<(), Error> {
        for it in items {
            self.item(it)?;
        }
        Ok(())
    }

    fn item(&mut self, it: &Item) -> Result<(), Error> {
        match it {
            Item::Postulate { name, ty, sp } => {
                let (_, v) = self.elab_type(ty)?;
                let lvl = self.fresh();
                let neu = HVal::Neu(Box::new(Head::Var {
                    lvl,
                    ty: Box::new(v.clone()),
                }));
                self.env = self.env.insert(lvl, Slot::H(neu));
                self.binders.push(Bnd {
                    name: name.clone(),
                    lvl,
                    kind: Kind::Term(v),
                });
                let _ = sp;
                Ok(())
            }
            Item::Def { name, ty, body, .. } => {
                let (_, tv) = self.elab_type(ty)?;
                let (_, val) = self.elab_term_check(body, &tv)?;
                let lvl = self.fresh();
                self.env = self.env.insert(lvl, Slot::H(val));
                self.binders.push(Bnd {
                    name: name.clone(),
                    lvl,
                    kind: Kind::Term(tv),
                });
                Ok(())
            }
            Item::TheoryB { name, body, .. } => {
                let (_, v) = self.elab_theory(body)?;
                self.theories.insert(name.clone(), v);
                Ok(())
            }
            Item::TheoryA { name, fields, .. } => {
                let rec = self.elab_record(fields)?;
                self.theories.insert(name.clone(), TVal::Record(rec));
                Ok(())
            }
            Item::ModelB {
                name,
                thy,
                body,
                sp,
            } => {
                let tval = self
                    .theories
                    .get(thy)
                    .cloned()
                    .ok_or_else(|| Error::at(*sp, format!("unknown theory `{thy}`")))?;
                let (_, val) = self.elab_model_check(body, &tval)?;
                self.models.insert(name.clone(), (val, tval));
                Ok(())
            }
            Item::ModelA {
                name,
                thy,
                fields,
                sp,
            } => {
                let tval = self
                    .theories
                    .get(thy)
                    .cloned()
                    .ok_or_else(|| Error::at(*sp, format!("unknown theory `{thy}`")))?;
                let TVal::Record(rec) = tval else {
                    return self.fail(*sp, format!("`{thy}` is not a straight-line theory"));
                };
                self.elab_record_model(&rec, fields)?;
                let _ = name;
                Ok(())
            }
            Item::Context { binders, items, .. } => {
                let n = self.binders.len();
                let env = self.env.clone();
                let theories = self.theories.clone();
                let models = self.models.clone();
                for b in binders {
                    self.elab_binder(b)?;
                }
                let r = self.items(items);
                self.binders.truncate(n);
                self.env = env;
                self.theories = theories;
                self.models = models;
                r
            }
            Item::CheckType { expr, .. } => {
                let _ = self.elab_type(expr)?;
                Ok(())
            }
            Item::CheckTerm { expr, ty, .. } => {
                let (_, tv) = self.elab_type(ty)?;
                let _ = self.elab_term_check(expr, &tv)?;
                Ok(())
            }
            Item::CheckModel { expr, thy, .. } => {
                let (_, tv) = self.elab_theory(thy)?;
                let _ = self.elab_model_check(expr, &tv)?;
                Ok(())
            }
            Item::Defeq {
                negate,
                a,
                b,
                ty,
                sp,
            } => self.defeq(*sp, *negate, a, b, ty.as_ref()),
        }
    }

    fn elab_binder(&mut self, b: &BinderS) -> Result<(), Error> {
        if b.model {
            let (_, thy) = self.elab_theory(&b.ty)?;
            let lvl = self.fresh();
            let neu = MVal::Neu {
                head: Box::new(MHead::Var { lvl }),
                thy: Box::new(thy.clone()),
            };
            self.env = self.env.insert(lvl, Slot::M(neu));
            self.binders.push(Bnd {
                name: b.name.clone(),
                lvl,
                kind: Kind::Model(thy),
            });
        } else {
            let (_, ty) = self.elab_type(&b.ty)?;
            let lvl = self.fresh();
            let neu = HVal::Neu(Box::new(Head::Var {
                lvl,
                ty: Box::new(ty.clone()),
            }));
            self.env = self.env.insert(lvl, Slot::H(neu));
            self.binders.push(Bnd {
                name: b.name.clone(),
                lvl,
                kind: Kind::Term(ty),
            });
        }
        Ok(())
    }

    fn defeq(&mut self, sp: Span, negate: bool, a: &E, b: &E, ty: Option<&E>) -> Result<(), Error> {
        let equal = if let Some(ty) = ty {
            let (_, tv) = self.elab_type(ty)?;
            let (_, va) = self.elab_term_check(a, &tv)?;
            let (_, vb) = self.elab_term_check(b, &tv)?;
            self.nbe.eq_tm(&tv, va, vb)
        } else if let (Ok((_, va)), Ok((_, vb))) = (self.elab_type(a), self.elab_type(b)) {
            let eq = self.nbe.eq_ty(va.clone(), vb.clone());
            if !eq && !negate {
                return self.fail(
                    sp,
                    format!(
                        "not definitionally equal\n  {}\n  {}",
                        self.nbe.pp_h(&va),
                        self.nbe.pp_h(&vb)
                    ),
                );
            }
            if eq && negate {
                return self.fail(
                    sp,
                    format!("definitionally equal\n  {}", self.nbe.pp_h(&va)),
                );
            }
            return Ok(());
        } else {
            let (_, va) = self.elab_theory(a)?;
            let (_, vb) = self.elab_theory(b)?;
            let eq = self.nbe.eq_t(&va, &vb);
            if !eq && !negate {
                return self.fail(
                    sp,
                    format!(
                        "not definitionally equal\n  {}\n  {}",
                        self.nbe.pp_t(&va),
                        self.nbe.pp_t(&vb)
                    ),
                );
            }
            if eq && negate {
                return self.fail(
                    sp,
                    format!("definitionally equal\n  {}", self.nbe.pp_t(&va)),
                );
            }
            return Ok(());
        };
        if equal == negate {
            return self.fail(
                sp,
                if negate {
                    "definitionally equal".to_string()
                } else {
                    "not definitionally equal".into()
                },
            );
        }
        Ok(())
    }

    fn in_prefix<R>(&mut self, past: u32, f: impl FnOnce(&mut Self) -> R) -> R {
        let old_b = self.binders.clone();
        let old_e = self.env.clone();
        self.binders.retain(|b| b.lvl < past);
        self.env = old_e.truncate_before(past);
        let r = f(self);
        self.binders = old_b;
        self.env = old_e;
        r
    }

    fn model_level(&self, name: &str, sp: Span) -> Result<u32, Error> {
        let mut found = None;
        for (i, b) in self.binders.iter().enumerate().rev() {
            if b.name == name {
                match &b.kind {
                    Kind::Model(_) => {
                        found = Some((i, b.lvl));
                        break;
                    }
                    Kind::Term(_) => {
                        return self.fail(sp, format!("`{name}` is a term, not a model"));
                    }
                }
            }
        }
        let Some((i, lvl)) = found else {
            return self.fail(sp, format!("unknown model `{name}`"));
        };
        for b in self.binders.iter().skip(i + 1) {
            if let Kind::Model(_) = b.kind {
                return self.fail(sp, format!("weaken past model `{}` as well", b.name));
            }
        }
        Ok(lvl)
    }

    fn lookup_term(&self, name: &str, sp: Span) -> Result<(u32, HVal), Error> {
        let mut blocked: Option<&str> = None;
        for b in self.binders.iter().rev() {
            if b.name == name {
                return match &b.kind {
                    Kind::Term(ty) => {
                        if let Some(m) = blocked {
                            self.fail(sp, format!("weaken `{name}` past model `{m}`"))
                        } else {
                            Ok((b.lvl, ty.clone()))
                        }
                    }
                    Kind::Model(_) => self.fail(sp, format!("`{name}` is a model")),
                };
            }
            if blocked.is_none() {
                if let Kind::Model(_) = b.kind {
                    blocked = Some(&b.name);
                }
            }
        }
        self.fail(sp, format!("unknown term `{name}`"))
    }

    fn lookup_model(&self, name: &str, sp: Span) -> Result<(u32, TVal), Error> {
        let mut blocked: Option<&str> = None;
        for b in self.binders.iter().rev() {
            if b.name == name {
                return match &b.kind {
                    Kind::Model(thy) => {
                        if let Some(m) = blocked {
                            self.fail(sp, format!("weaken `{name}` past model `{m}`"))
                        } else {
                            let transported =
                                weaken_theory(&self.nbe, &self.env, thy.clone(), b.lvl);
                            Ok((b.lvl, transported))
                        }
                    }
                    Kind::Term(_) => self.fail(sp, format!("`{name}` is a term")),
                };
            }
            if blocked.is_none() {
                if let Kind::Model(_) = b.kind {
                    blocked = Some(&b.name);
                }
            }
        }
        self.fail(sp, format!("unknown model `{name}`"))
    }

    fn elab_type(&mut self, e: &E) -> Result<(HTm, HVal), Error> {
        let tm = self.elab_type_tm(e)?;
        let val = self.nbe.eval_h(&tm, &self.env);
        if !self.nbe.is_type(&val) {
            return self.fail(e.sp, format!("expected a type\n  {}", self.nbe.pp_h(&val)));
        }
        Ok((tm, val))
    }

    fn elab_type_tm(&mut self, e: &E) -> Result<HTm, Error> {
        match &e.kind {
            Ek::U => Ok(HTm::U),
            Ek::Nat => Ok(HTm::Nat),
            Ek::Empty => Ok(HTm::Empty),
            Ek::Unit => Ok(HTm::Unit),
            Ek::Arrow { binder, dom, cod } => {
                let d = self.elab_type_tm(dom)?;
                let dv = self.nbe.eval_h(&d, &self.env);
                let lvl = self.fresh();
                let name = binder.clone().unwrap_or_else(|| "_".into());
                self.push_term(&name, lvl, dv);
                let c = self.elab_type_tm(cod);
                self.pop();
                Ok(HTm::Pi {
                    binder: lvl,
                    dom: Box::new(d),
                    cod: Box::new(c?),
                })
            }
            Ek::Prod {
                binder,
                model: false,
                dom,
                cod,
            } => {
                let d = self.elab_type_tm(dom)?;
                let dv = self.nbe.eval_h(&d, &self.env);
                let lvl = self.fresh();
                let name = binder.clone().unwrap_or_else(|| "_".into());
                self.push_term(&name, lvl, dv);
                let c = self.elab_type_tm(cod);
                self.pop();
                Ok(HTm::Sigma {
                    binder: lvl,
                    dom: Box::new(d),
                    cod: Box::new(c?),
                })
            }
            Ek::Sum(a, b) => Ok(HTm::Sum(
                Box::new(self.elab_type_tm(a)?),
                Box::new(self.elab_type_tm(b)?),
            )),
            Ek::Id { ty, a, b } => {
                if let Some(t) = ty {
                    let tt = self.elab_type_tm(t)?;
                    let tv = self.nbe.eval_h(&tt, &self.env);
                    let aa = self.elab_term_check(a, &tv)?.0;
                    let bb = self.elab_term_check(b, &tv)?.0;
                    Ok(HTm::Id(Box::new(tt), Box::new(aa), Box::new(bb)))
                } else {
                    let (aa, _, ty) = self.elab_term_infer(a)?;
                    let bb = self.elab_term_check(b, &ty)?.0;
                    let tt = quote_type_var(&ty);
                    // Rebuild via eval equality: synthesize Id from inferred type syntax
                    // by embedding the type value.
                    let _ = tt;
                    Ok(HTm::Id(
                        Box::new(HTm::Embed(Box::new(ty))),
                        Box::new(aa),
                        Box::new(bb),
                    ))
                }
            }
            Ek::Un(UnOp::Ty, m) => {
                let mt = self.elab_model_tm(m)?;
                Ok(HTm::Ty(Box::new(mt)))
            }
            Ek::Un(UnOp::Trunc, _) => self.fail(e.sp, "Trunc is a construction, not a host type"),
            Ek::Name(n) => {
                let (lvl, ty) = self.lookup_term(n, e.sp)?;
                if !self.nbe.eq_ty(ty, HVal::U) && !matches!(self.env.get(lvl), Some(Slot::H(_))) {
                    return self.fail(e.sp, format!("`{n}` is not a type"));
                }
                // Russell: a term of type U is a type. A defined small type also lives in U.
                let (_, got) = self.lookup_term(n, e.sp)?;
                if self.nbe.eq_ty(got, HVal::U) {
                    Ok(HTm::Var(lvl))
                } else {
                    self.fail(e.sp, format!("`{n}` is not a type"))
                }
            }
            Ek::Weak { tm, name } => {
                let past = self.model_level(name, e.sp)?;
                let inner = self.in_prefix(past, |ctx| ctx.elab_type_tm(tm))?;
                Ok(HTm::Weak {
                    tm: Box::new(inner),
                    past,
                })
            }
            Ek::Subst { tm, repl, name } => {
                let past = self.model_level(name, e.sp)?;
                let repl_tm = self.in_prefix(past, |ctx| ctx.elab_model_tm(repl))?;
                let body = self.elab_type_tm(tm)?;
                Ok(HTm::MSub {
                    tm: Box::new(body),
                    repl: MRepl::Syn(Box::new(repl_tm)),
                    past,
                })
            }
            Ek::Call { name, .. } => {
                // A named small type applied? Rare. Treat as a term and require it be a type.
                let (tm, val, ty) = self.elab_term_infer(e)?;
                if self.nbe.eq_ty(ty, HVal::U) || self.nbe.is_type(&val) {
                    let _ = name;
                    Ok(tm)
                } else {
                    self.fail(e.sp, "expected a type")
                }
            }
            _ => {
                // Fall back: elaborate as a term of type U (a small type code).
                match self.elab_term_infer(e) {
                    Ok((tm, val, ty))
                        if self.nbe.eq_ty(ty.clone(), HVal::U) || self.nbe.is_type(&val) =>
                    {
                        Ok(tm)
                    }
                    Ok((_, val, _)) => {
                        self.fail(e.sp, format!("expected a type\n  {}", self.nbe.pp_h(&val)))
                    }
                    Err(err) => Err(err),
                }
            }
        }
    }

    fn push_term(&mut self, name: &str, lvl: u32, ty: HVal) {
        let neu = HVal::Neu(Box::new(Head::Var {
            lvl,
            ty: Box::new(ty.clone()),
        }));
        self.env = self.env.insert(lvl, Slot::H(neu));
        self.binders.push(Bnd {
            name: name.to_string(),
            lvl,
            kind: Kind::Term(ty),
        });
    }

    fn push_model(&mut self, name: &str, lvl: u32, thy: TVal) {
        let neu = MVal::Neu {
            head: Box::new(MHead::Var { lvl }),
            thy: Box::new(thy.clone()),
        };
        self.env = self.env.insert(lvl, Slot::M(neu));
        self.binders.push(Bnd {
            name: name.to_string(),
            lvl,
            kind: Kind::Model(thy),
        });
    }

    fn pop(&mut self) {
        if let Some(b) = self.binders.pop() {
            self.env.remove_level(b.lvl);
        }
    }

    fn elab_term_infer(&mut self, e: &E) -> Result<(HTm, HVal, HVal), Error> {
        match &e.kind {
            Ek::Tt => Ok((HTm::Tt, HVal::Tt, HVal::Unit)),
            Ek::Zero => Ok((HTm::Z, HVal::Z, HVal::Nat)),
            Ek::Name(n) => {
                if let Ok((lvl, ty)) = self.lookup_term(n, e.sp) {
                    let tm = HTm::Var(lvl);
                    let val = self.nbe.eval_h(&tm, &self.env);
                    return Ok((tm, val, ty));
                }
                if n == "zero" {
                    return Ok((HTm::Z, HVal::Z, HVal::Nat));
                }
                self.fail(e.sp, format!("unknown term `{n}`"))
            }
            Ek::Un(UnOp::Succ, n) => {
                let (tm, _) = self.elab_term_check(n, &HVal::Nat)?;
                let tm = HTm::S(Box::new(tm));
                let val = self.nbe.eval_h(&tm, &self.env);
                Ok((tm, val, HVal::Nat))
            }
            Ek::Un(UnOp::Fst, p) => {
                let (pt, pv, pty) = self.elab_term_infer(p)?;
                let HVal::Sigma(dom, _) = &pty else {
                    return self.fail(e.sp, "fst expects a pair type");
                };
                let tm = HTm::Fst(Box::new(pt));
                let val = self.nbe.fst_h(pv);
                Ok((tm, val, (**dom).clone()))
            }
            Ek::Un(UnOp::Snd, p) => {
                let (pt, pv, pty) = self.elab_term_infer(p)?;
                let HVal::Sigma(_, cod) = &pty else {
                    return self.fail(e.sp, "snd expects a pair type");
                };
                let fst = self.nbe.fst_h(pv.clone());
                let ty = self.nbe.apply_hclos((**cod).clone(), fst);
                let tm = HTm::Snd(Box::new(pt));
                let val = self.nbe.snd_h(pv);
                Ok((tm, val, ty))
            }
            Ek::Un(UnOp::Refl, a) => {
                let (at, av, ty) = self.elab_term_infer(a)?;
                let tm = HTm::Refl(Box::new(at.clone()));
                let val = HVal::Refl(Box::new(av.clone()));
                let id = HVal::Id(Box::new(ty), Box::new(av.clone()), Box::new(av));
                Ok((tm, val, id))
            }
            Ek::Un(UnOp::Unax, m) => {
                let (mt, mv, thy) = self.elab_model_infer(m)?;
                let TVal::Ax(a) = thy else {
                    return self.fail(e.sp, format!("unax expects Ax\n  {}", self.nbe.pp_t(&thy)));
                };
                let tm = HTm::Unax(Box::new(mt));
                let val = self.nbe.eval_h(&tm, &self.env);
                let _ = mv;
                Ok((tm, val, *a))
            }
            Ek::App(f, a) => {
                let (ft, fv, fty) = self.elab_term_infer(f)?;
                let HVal::Pi(dom, cod) = fty else {
                    return self.fail(
                        e.sp,
                        format!("application expects a function\n  {}", self.nbe.pp_h(&fty)),
                    );
                };
                let (at, av) = self.elab_term_check(a, &dom)?;
                let ty = self.nbe.apply_hclos(*cod, av.clone());
                let tm = HTm::App(Box::new(ft), Box::new(at));
                let val = self.nbe.apply_h(fv, av);
                Ok((tm, val, ty))
            }
            Ek::Call {
                name,
                host,
                con,
                semi,
            } => {
                if *semi {
                    return self.fail(e.sp, "host calls do not take a ';'");
                }
                let mut args = host.clone();
                args.extend(con.clone());
                let head = E {
                    sp: e.sp,
                    kind: Ek::Name(name.clone()),
                };
                let mut cur = head;
                if args.is_empty() {
                    return self.elab_term_infer(&cur);
                }
                for a in args {
                    cur = E {
                        sp: e.sp,
                        kind: Ek::App(Box::new(cur), Box::new(a)),
                    };
                }
                self.elab_term_infer(&cur)
            }
            Ek::Pair(_, _) => self.fail(e.sp, "a pair needs an expected Σ type"),
            Ek::Un(UnOp::Inl, _) | Ek::Un(UnOp::Inr, _) => {
                self.fail(e.sp, "inl/inr need an expected sum type")
            }
            Ek::Lam { .. } => self.fail(e.sp, "a lambda needs an expected function type"),
            Ek::Weak { tm, name } => {
                let past = self.model_level(name, e.sp)?;
                let (inner, _, ty) = self.in_prefix(past, |ctx| ctx.elab_term_infer(tm))?;
                let tm = HTm::Weak {
                    tm: Box::new(inner),
                    past,
                };
                let val = self.nbe.eval_h(&tm, &self.env);
                let ty_tm = HTm::Weak {
                    tm: Box::new(HTm::Embed(Box::new(ty))),
                    past,
                };
                let tyv = self.nbe.eval_h(&ty_tm, &self.env);
                Ok((tm, val, tyv))
            }
            Ek::Subst { tm, repl, name } => {
                let past = self.model_level(name, e.sp)?;
                let repl_tm = self.in_prefix(past, |ctx| ctx.elab_model_tm(repl))?;
                let (body, _, ty) = self.elab_term_infer(tm)?;
                let tm = HTm::MSub {
                    tm: Box::new(body),
                    repl: MRepl::Syn(Box::new(repl_tm.clone())),
                    past,
                };
                let val = self.nbe.eval_h(&tm, &self.env);
                let ty_tm = HTm::MSub {
                    tm: Box::new(HTm::Embed(Box::new(ty))),
                    repl: MRepl::Syn(Box::new(repl_tm)),
                    past,
                };
                let tyv = self.nbe.eval_h(&ty_tm, &self.env);
                Ok((tm, val, tyv))
            }
            _ => {
                // Motives make match inferable. Handle match/j/natind here.
                self.elab_term_infer_more(e)
            }
        }
    }

    fn elab_term_infer_more(&mut self, e: &E) -> Result<(HTm, HVal, HVal), Error> {
        match &e.kind {
            Ek::Match {
                scrut,
                binder,
                motive,
                inl_name,
                inl,
                inr_name,
                inr,
            } => {
                let (st, sv, sty) = self.elab_term_infer(scrut)?;
                let HVal::Sum(lt, rt) = sty.clone() else {
                    return self.fail(e.sp, "match expects a sum");
                };
                let xb = self.fresh();
                let base = self.env.clone();
                self.push_term(binder, xb, sty);
                let mot_tm = self.elab_type_tm(motive);
                self.pop();
                let mot_tm = mot_tm?;
                let apply_mot = |ctx: &mut Ctx, arg: HVal| {
                    let env = base.insert(xb, Slot::H(arg));
                    ctx.nbe.eval_h(&mot_tm, &env)
                };
                let lb = self.fresh();
                self.push_term(inl_name, lb, (*lt).clone());
                let inl_ty = apply_mot(self, HVal::Inl(Box::new(self.neu(lb, (*lt).clone()))));
                let inl_tm = self.elab_term_check(inl, &inl_ty);
                self.pop();
                let inl_tm = inl_tm?.0;
                let rb = self.fresh();
                self.push_term(inr_name, rb, (*rt).clone());
                let inr_ty = apply_mot(self, HVal::Inr(Box::new(self.neu(rb, (*rt).clone()))));
                let inr_tm = self.elab_term_check(inr, &inr_ty);
                self.pop();
                let inr_tm = inr_tm?.0;
                let tm = HTm::Match {
                    scrut: Box::new(st),
                    motive_b: xb,
                    motive: Box::new(mot_tm.clone()),
                    inl_b: lb,
                    inl: Box::new(inl_tm),
                    inr_b: rb,
                    inr: Box::new(inr_tm),
                };
                let val = self.nbe.eval_h(&tm, &self.env);
                let ty = apply_mot(self, sv);
                Ok((tm, val, ty))
            }
            Ek::J {
                path,
                y,
                p,
                motive,
                refl_case,
            } => {
                let (pt, pv, pty) = self.elab_term_infer(path)?;
                let HVal::Id(a_ty, aa, bb) = pty.clone() else {
                    return self.fail(e.sp, "j expects an identity");
                };
                let yb = self.fresh();
                let pb = self.fresh();
                let base = self.env.clone();
                self.push_term(y, yb, (*a_ty).clone());
                self.push_term(
                    p,
                    pb,
                    HVal::Id(
                        a_ty.clone(),
                        aa.clone(),
                        Box::new(self.neu(yb, (*a_ty).clone())),
                    ),
                );
                let mot_tm = self.elab_type_tm(motive);
                self.pop();
                self.pop();
                let mot_tm = mot_tm?;
                let at_refl = {
                    let mut env = base.insert(yb, Slot::H((*aa).clone()));
                    env = env.insert(pb, Slot::H(HVal::Refl(aa.clone())));
                    self.nbe.eval_h(&mot_tm, &env)
                };
                let rc = self.elab_term_check(refl_case, &at_refl)?.0;
                let tm = HTm::J {
                    ty: Box::new(HTm::Embed(a_ty.clone())),
                    a: Box::new(HTm::Embed(aa.clone())),
                    b: Box::new(HTm::Embed(bb.clone())),
                    path: Box::new(pt),
                    y_b: yb,
                    p_b: pb,
                    motive: Box::new(mot_tm.clone()),
                    refl_case: Box::new(rc),
                };
                let val = self.nbe.eval_h(&tm, &self.env);
                let mut env = base.insert(yb, Slot::H((*bb).clone()));
                env = env.insert(pb, Slot::H(pv));
                let ty = self.nbe.eval_h(&mot_tm, &env);
                Ok((tm, val, ty))
            }
            Ek::NatInd {
                scrut,
                k,
                motive,
                zcase,
                m,
                ih,
                scase,
            } => {
                let (st, sv) = self.elab_term_check(scrut, &HVal::Nat)?;
                let kb = self.fresh();
                let base = self.env.clone();
                self.push_term(k, kb, HVal::Nat);
                let mot_tm = self.elab_type_tm(motive);
                self.pop();
                let mot_tm = mot_tm?;
                let zty = {
                    let env = base.insert(kb, Slot::H(HVal::Z));
                    self.nbe.eval_h(&mot_tm, &env)
                };
                let zt = self.elab_term_check(zcase, &zty)?.0;
                let mb = self.fresh();
                let ib = self.fresh();
                self.push_term(m, mb, HVal::Nat);
                let ih_ty = {
                    let env = base.insert(kb, Slot::H(self.neu(mb, HVal::Nat)));
                    self.nbe.eval_h(&mot_tm, &env)
                };
                self.push_term(ih, ib, ih_ty);
                let sty = {
                    let env = base.insert(kb, Slot::H(HVal::S(Box::new(self.neu(mb, HVal::Nat)))));
                    self.nbe.eval_h(&mot_tm, &env)
                };
                let sct = self.elab_term_check(scase, &sty);
                self.pop();
                self.pop();
                let sct = sct?.0;
                let tm = HTm::NatInd {
                    scrut: Box::new(st),
                    k_b: kb,
                    motive: Box::new(mot_tm.clone()),
                    zcase: Box::new(zt),
                    m_b: mb,
                    ih_b: ib,
                    scase: Box::new(sct),
                };
                let val = self.nbe.eval_h(&tm, &self.env);
                let env = base.insert(kb, Slot::H(sv));
                let ty = self.nbe.eval_h(&mot_tm, &env);
                Ok((tm, val, ty))
            }
            _ => self.fail(e.sp, "cannot infer this term"),
        }
    }

    fn neu(&self, lvl: u32, ty: HVal) -> HVal {
        HVal::Neu(Box::new(Head::Var {
            lvl,
            ty: Box::new(ty),
        }))
    }

    fn elab_term_check(&mut self, e: &E, expected: &HVal) -> Result<(HTm, HVal), Error> {
        if self.nbe.eq_ty(expected.clone(), HVal::U) {
            if let Ok(tm) = self.elab_type_tm(e) {
                let val = self.nbe.eval_h(&tm, &self.env);
                if self.nbe.is_small(&val) {
                    return Ok((tm, val));
                }
            }
        }
        match &e.kind {
            Ek::Lam { name, body } => {
                let HVal::Pi(dom, cod) = expected else {
                    return self.fail(
                        e.sp,
                        format!("lambda is not a function\n  {}", self.nbe.pp_h(expected)),
                    );
                };
                let lvl = self.fresh();
                let x = self.neu(lvl, (**dom).clone());
                let body_ty = self.nbe.apply_hclos((**cod).clone(), x);
                self.push_term(name, lvl, (**dom).clone());
                let bt = self.elab_term_check(body, &body_ty);
                self.pop();
                let tm = HTm::Lam {
                    binder: lvl,
                    body: Box::new(bt?.0),
                };
                let val = self.nbe.eval_h(&tm, &self.env);
                Ok((tm, val))
            }
            Ek::Pair(a, b) => {
                let HVal::Sigma(dom, cod) = expected else {
                    return self.fail(e.sp, "pair is not a Σ");
                };
                let (at, av) = self.elab_term_check(a, dom)?;
                let cty = self.nbe.apply_hclos((**cod).clone(), av.clone());
                let (bt, bv) = self.elab_term_check(b, &cty)?;
                let _ = (av, bv);
                let tm = HTm::Pair(Box::new(at), Box::new(bt));
                let val = self.nbe.eval_h(&tm, &self.env);
                Ok((tm, val))
            }
            Ek::Un(UnOp::Refl, a) => {
                let HVal::Id(ty, left, right) = expected else {
                    return self.fail(e.sp, "refl needs an identity type");
                };
                let (at, av) = self.elab_term_check(a, ty)?;
                if self.nbe.eq_tm(ty, (**left).clone(), av.clone())
                    && self.nbe.eq_tm(ty, (**right).clone(), av.clone())
                {
                    let tm = HTm::Refl(Box::new(at));
                    let val = self.nbe.eval_h(&tm, &self.env);
                    Ok((tm, val))
                } else {
                    self.fail(
                        e.sp,
                        format!(
                            "refl is not an identity at the endpoints\n  {}",
                            self.nbe.pp_h(&av)
                        ),
                    )
                }
            }
            Ek::Un(UnOp::Abort, scrut) => {
                let (st, _) = self.elab_term_check(scrut, &HVal::Empty)?;
                let tm = HTm::Exfalso {
                    motive: Box::new(HTm::Embed(Box::new(expected.clone()))),
                    scrut: Box::new(st),
                };
                let val = self.nbe.eval_h(&tm, &self.env);
                Ok((tm, val))
            }
            Ek::Un(UnOp::Inl, a) => {
                let HVal::Sum(lt, _) = expected else {
                    return self.fail(e.sp, "inl is not a sum");
                };
                let (at, _) = self.elab_term_check(a, lt)?;
                let tm = HTm::Inl(Box::new(at));
                let val = self.nbe.eval_h(&tm, &self.env);
                Ok((tm, val))
            }
            Ek::Un(UnOp::Inr, a) => {
                let HVal::Sum(_, rt) = expected else {
                    return self.fail(e.sp, "inr is not a sum");
                };
                let (at, _) = self.elab_term_check(a, rt)?;
                let tm = HTm::Inr(Box::new(at));
                let val = self.nbe.eval_h(&tm, &self.env);
                Ok((tm, val))
            }
            Ek::Tt => {
                if !self.nbe.eq_ty(expected.clone(), HVal::Unit) {
                    return self.fail(e.sp, "tt is a unit term");
                }
                Ok((HTm::Tt, HVal::Tt))
            }
            _ => {
                let (tm, val, got) = self.elab_term_infer(e)?;
                if self.nbe.eq_ty(got.clone(), expected.clone()) {
                    Ok((tm, val))
                } else {
                    self.fail(
                        e.sp,
                        format!(
                            "type mismatch\n  expected {}\n  got      {}",
                            self.nbe.pp_h(expected),
                            self.nbe.pp_h(&got)
                        ),
                    )
                }
            }
        }
    }

    fn elab_theory(&mut self, e: &E) -> Result<(TTm, TVal), Error> {
        let tm = self.elab_theory_tm(e)?;
        let val = self.nbe.eval_t(&tm, &self.env);
        Ok((tm, val))
    }

    fn elab_theory_tm(&mut self, e: &E) -> Result<TTm, Error> {
        match &e.kind {
            Ek::One => Ok(TTm::One),
            Ek::Sort => Ok(TTm::Sort),
            Ek::Un(UnOp::AxTy, a) => Ok(TTm::Ax(Box::new(self.elab_type_tm(a)?))),
            Ek::Prod {
                binder: Some(name),
                model: true,
                dom,
                cod,
            } => {
                let fst = self.elab_theory_tm(dom)?;
                let fval = self.nbe.eval_t(&fst, &self.env);
                let lvl = self.fresh();
                self.push_model(name, lvl, fval);
                let snd = self.elab_theory_tm(cod);
                self.pop();
                Ok(TTm::Sigma {
                    binder: lvl,
                    fst: Box::new(fst),
                    snd: Box::new(snd?),
                })
            }
            Ek::Rtri { binder, dom, cod } => {
                let d = self.elab_type_tm(dom)?;
                let dv = self.nbe.eval_h(&d, &self.env);
                let lvl = self.fresh();
                let name = binder.clone().unwrap_or_else(|| "_".into());
                self.push_term(&name, lvl, dv);
                let c = self.elab_theory_tm(cod);
                self.pop();
                Ok(TTm::Rtri {
                    binder: lvl,
                    dom: Box::new(d),
                    cod: Box::new(c?),
                })
            }
            Ek::Name(n) => {
                let v = self
                    .theories
                    .get(n)
                    .cloned()
                    .ok_or_else(|| Error::at(e.sp, format!("unknown theory `{n}`")))?;
                Ok(TTm::Embed(Box::new(v)))
            }
            Ek::IndTy {
                scrut,
                lname,
                left,
                rname,
                right,
            } => {
                let (st, _, sty) = self.elab_term_infer(scrut)?;
                let HVal::Sum(lt, rt) = sty else {
                    return self.fail(e.sp, "Ind expects a sum");
                };
                let lb = self.fresh();
                self.push_term(lname, lb, (*lt).clone());
                let ltm = self.elab_theory_tm(left);
                self.pop();
                let rb = self.fresh();
                self.push_term(rname, rb, (*rt).clone());
                let rtm = self.elab_theory_tm(right);
                self.pop();
                Ok(TTm::IndSum {
                    scrut: Box::new(st),
                    l_b: lb,
                    left: Box::new(ltm?),
                    r_b: rb,
                    right: Box::new(rtm?),
                })
            }
            Ek::Weak { tm, name } => {
                let past = self.model_level(name, e.sp)?;
                let inner = self.in_prefix(past, |ctx| ctx.elab_theory_tm(tm))?;
                Ok(TTm::Weak {
                    tm: Box::new(inner),
                    past,
                })
            }
            Ek::Subst { tm, repl, name } => {
                let past = self.model_level(name, e.sp)?;
                let repl_tm = self.in_prefix(past, |ctx| ctx.elab_model_tm(repl))?;
                let body = self.elab_theory_tm(tm)?;
                Ok(TTm::MSub {
                    tm: Box::new(body),
                    repl: MRepl::Syn(Box::new(repl_tm)),
                    past,
                })
            }
            _ => self.fail(e.sp, "expected a theory"),
        }
    }

    fn elab_model_tm(&mut self, e: &E) -> Result<MTm, Error> {
        Ok(self.elab_model_infer(e)?.0)
    }

    fn elab_model_infer(&mut self, e: &E) -> Result<(MTm, MVal, TVal), Error> {
        match &e.kind {
            Ek::Tt => Ok((MTm::Tt, MVal::Tt, TVal::One)),
            Ek::Name(n) => {
                if let Some((val, thy)) = self.models.get(n).cloned() {
                    return Ok((MTm::Embed(Box::new(val.clone())), val, thy));
                }
                let (lvl, thy) = self.lookup_model(n, e.sp)?;
                let tm = MTm::Var(lvl);
                let val = self.nbe.eval_m(&tm, &self.env);
                Ok((tm, val, thy))
            }
            Ek::Un(UnOp::SortM, a) => {
                let (at, av) = self.elab_type(a)?;
                let tm = MTm::Sort(Box::new(at));
                let val = self.nbe.eval_m(&tm, &self.env);
                let _ = av;
                Ok((tm, val, TVal::Sort))
            }
            Ek::Un(UnOp::Pr1, m) => {
                let (mt, mv, thy) = self.elab_model_infer(m)?;
                let TVal::Sigma(fst, _) = thy else {
                    return self.fail(e.sp, "pr1 expects a Σ theory");
                };
                let tm = MTm::Pr1(Box::new(mt));
                let val = self.nbe.pr1(mv);
                Ok((tm, val, *fst))
            }
            Ek::Un(UnOp::Pr2, m) => {
                let (mt, mv, thy) = self.elab_model_infer(m)?;
                let TVal::Sigma(_, cod) = &thy else {
                    return self.fail(e.sp, "pr2 expects a Σ theory");
                };
                let p1 = self.nbe.pr1(mv.clone());
                let sty = self.nbe.inst_sigma(cod, &p1);
                let tm = MTm::Pr2(Box::new(mt));
                let val = self.nbe.pr2(mv);
                Ok((tm, val, sty))
            }
            Ek::App(f, a) => {
                let (ft, fv, fthy) = self.elab_model_infer(f)?;
                let TVal::Rtri(dom, cod) = fthy else {
                    return self.fail(
                        e.sp,
                        format!("model application expects ▹\n  {}", self.nbe.pp_t(&fthy)),
                    );
                };
                let (at, av) = self.elab_term_check(a, &dom)?;
                let ty = self.nbe.apply_t_host(&cod, av.clone());
                let tm = MTm::App(Box::new(ft), Box::new(at));
                let val = self.nbe.apply_m(fv, av);
                Ok((tm, val, ty))
            }
            Ek::Weak { tm, name } => {
                let past = self.model_level(name, e.sp)?;
                let (inner, _, thy) = self.in_prefix(past, |ctx| ctx.elab_model_infer(tm))?;
                let tm = MTm::Weak {
                    tm: Box::new(inner),
                    past,
                };
                let val = self.nbe.eval_m(&tm, &self.env);
                let thy = weaken_theory(&self.nbe, &self.env, thy, past);
                Ok((tm, val, thy))
            }
            _ => self.fail(e.sp, "cannot infer this model"),
        }
    }

    fn elab_model_check(&mut self, e: &E, expected: &TVal) -> Result<(MTm, MVal), Error> {
        match &e.kind {
            Ek::Pair(a, b) => {
                let TVal::Sigma(fst, cod) = expected else {
                    return self.fail(e.sp, "pair is not a Σ theory");
                };
                let (at, av) = self.elab_model_check(a, fst)?;
                let sty = self.nbe.inst_sigma(cod, &av);
                let (bt, _) = self.elab_model_check(b, &sty)?;
                let tm = MTm::Pair(Box::new(at), Box::new(bt));
                let val = self.nbe.eval_m(&tm, &self.env);
                Ok((tm, val))
            }
            Ek::Lam { name, body } => {
                let TVal::Rtri(dom, cod) = expected else {
                    return self.fail(e.sp, "lambda is not a ▹ theory");
                };
                let lvl = self.fresh();
                let x = self.neu(lvl, (**dom).clone());
                let body_thy = self.nbe.apply_t_host(cod, x);
                self.push_term(name, lvl, (**dom).clone());
                let bt = self.elab_model_check(body, &body_thy);
                self.pop();
                let tm = MTm::Lam {
                    binder: lvl,
                    body: Box::new(bt?.0),
                };
                let val = self.nbe.eval_m(&tm, &self.env);
                Ok((tm, val))
            }
            Ek::Un(UnOp::AxM, a) => {
                let TVal::Ax(ty) = expected else {
                    return self.fail(e.sp, format!("ax is not Ax\n  {}", self.nbe.pp_t(expected)));
                };
                let (at, _) = self.elab_term_check(a, ty)?;
                let tm = MTm::Ax(Box::new(at));
                let val = self.nbe.eval_m(&tm, &self.env);
                Ok((tm, val))
            }
            Ek::Tt => {
                if !self.nbe.eq_t(expected, &TVal::One) {
                    return self.fail(e.sp, "tt is the unit model");
                }
                Ok((MTm::Tt, MVal::Tt))
            }
            Ek::IndM {
                scrut,
                x,
                motive,
                lname,
                left,
                rname,
                right,
            } => {
                let (st, sv, sty) = self.elab_term_infer(scrut)?;
                let HVal::Sum(lt, rt) = sty.clone() else {
                    return self.fail(e.sp, "ind expects a sum");
                };
                let xb = self.fresh();
                self.push_term(x, xb, sty);
                let mot = self.elab_theory_tm(motive);
                self.pop();
                let mot = mot?;
                let base = self.env.clone();
                let at = |ctx: &mut Ctx, arg: HVal| {
                    let env = base.insert(xb, Slot::H(arg));
                    ctx.nbe.eval_t(&mot, &env)
                };
                let got = at(self, sv.clone());
                if !self.nbe.eq_t(&got, expected) {
                    return self.fail(e.sp, "ind motive does not match the expected theory");
                }
                let lb = self.fresh();
                self.push_term(lname, lb, (*lt).clone());
                let lty = at(self, HVal::Inl(Box::new(self.neu(lb, (*lt).clone()))));
                let ltm = self.elab_model_check(left, &lty);
                self.pop();
                let rb = self.fresh();
                self.push_term(rname, rb, (*rt).clone());
                let rty = at(self, HVal::Inr(Box::new(self.neu(rb, (*rt).clone()))));
                let rtm = self.elab_model_check(right, &rty);
                self.pop();
                let tm = MTm::IndSum {
                    scrut: Box::new(st),
                    l_b: lb,
                    left: Box::new(ltm?.0),
                    r_b: rb,
                    right: Box::new(rtm?.0),
                };
                let val = self.nbe.eval_m(&tm, &self.env);
                Ok((tm, val))
            }
            _ => {
                let (tm, val, got) = self.elab_model_infer(e)?;
                if self.nbe.eq_t(&got, expected) {
                    Ok((tm, val))
                } else {
                    self.fail(
                        e.sp,
                        format!(
                            "model mismatch\n  expected {}\n  got      {}",
                            self.nbe.pp_t(expected),
                            self.nbe.pp_t(&got)
                        ),
                    )
                }
            }
        }
    }

    // ----- straight-line theories ------------------------------------------

    fn elab_record(&mut self, fields: &[FieldS]) -> Result<RecordTheory, Error> {
        let mut rec = Vec::new();
        for f in fields {
            match f {
                FieldS::Sort {
                    name,
                    delta,
                    phi,
                    sp,
                } => {
                    let (d, p) = self.elab_tele(*sp, delta, phi, &rec)?;
                    self.pop_n(d.len() + p.len());
                    rec.push(RecField {
                        name: name.clone(),
                        delta: d,
                        phi: p.into_iter().map(|(n, l, t)| (l, n, t)).collect(),
                        kind: RecKind::Sort,
                    });
                }
                FieldS::Term {
                    name,
                    delta,
                    phi,
                    ty,
                    sp,
                } => {
                    let (d, p) = self.elab_tele(*sp, delta, phi, &rec)?;
                    let ct = self.elab_con_type(ty, &rec, &p)?;
                    self.pop_n(d.len() + p.len());
                    rec.push(RecField {
                        name: name.clone(),
                        delta: d,
                        phi: p.into_iter().map(|(n, l, t)| (l, n, t)).collect(),
                        kind: RecKind::Term(ct),
                    });
                }
            }
        }
        Ok(RecordTheory { fields: rec })
    }

    /// Push telescope binders. Returns delta syntax and phi binders still in scope.
    fn elab_tele(
        &mut self,
        sp: Span,
        delta: &[TeleParam],
        phi: &[TeleParam],
        fields: &[RecField],
    ) -> Result<(Vec<(u32, String, HTm)>, Vec<(String, u32, Con)>), Error> {
        let _ = sp;
        let mut d = Vec::new();
        for p in delta {
            let tm = self.elab_type_tm(&p.ty)?;
            let val = self.nbe.eval_h(&tm, &self.env);
            if !self.nbe.is_type(&val) {
                return self.fail(p.ty.sp, "telescope entry is not a type");
            }
            let lvl = self.fresh();
            self.push_term(&p.name, lvl, val);
            d.push((lvl, p.name.clone(), tm));
        }
        let mut ph = Vec::new();
        for p in phi {
            let ct = self.elab_con_type(&p.ty, fields, &ph)?;
            let lvl = self.fresh();
            // Construction variables are not host values. Record them only in `ph`,
            // but also as dummy host binders so the name resolves as a construction
            // before a host name. We keep them out of the host env.
            ph.push((p.name.clone(), lvl, ct));
        }
        // Host delta binders were pushed; phi binders were not. Caller pops
        // d.len()+p.len(), so push placeholder phi binders to match.
        for (name, lvl, _) in &ph {
            self.binders.push(Bnd {
                name: name.clone(),
                lvl: *lvl,
                kind: Kind::Term(HVal::Unit),
            });
        }
        Ok((d, ph))
    }

    fn pop_n(&mut self, n: usize) {
        for _ in 0..n {
            self.pop();
        }
    }

    fn elab_con_type(
        &mut self,
        e: &E,
        fields: &[RecField],
        phi: &[(String, u32, Con)],
    ) -> Result<Con, Error> {
        match &e.kind {
            Ek::Empty => Ok(Con::Empty),
            Ek::Unit => Ok(Con::Unit),
            Ek::Nat => Ok(Con::Nat),
            Ek::Sum(a, b) => Ok(Con::Sum(
                Box::new(self.elab_con_type(a, fields, phi)?),
                Box::new(self.elab_con_type(b, fields, phi)?),
            )),
            Ek::Un(UnOp::Trunc, a) => Ok(Con::Trunc(Box::new(self.elab_con_type(a, fields, phi)?))),
            Ek::Un(UnOp::AxTy, a) => Ok(Con::Ax(Box::new(self.elab_type_tm(a)?))),
            Ek::Prod {
                binder,
                model: false,
                dom,
                cod,
            } => {
                let d = self.elab_con_type(dom, fields, phi)?;
                let mut phi2 = phi.to_vec();
                let lvl = self.fresh();
                let name = binder.clone().unwrap_or_else(|| "_".into());
                phi2.push((name, lvl, d.clone()));
                let c = self.elab_con_type(cod, fields, &phi2)?;
                Ok(Con::Sigma {
                    binder: lvl,
                    dom: Box::new(d),
                    cod: Box::new(c),
                })
            }
            Ek::Id { ty, a, b } => {
                let t = if let Some(t) = ty {
                    self.elab_con_type(t, fields, phi)?
                } else {
                    self.infer_con(a, fields, phi)?.1
                };
                let aa = self.check_con(a, &t, fields, phi)?;
                let bb = self.check_con(b, &t, fields, phi)?;
                Ok(Con::Id(Box::new(t), Box::new(aa), Box::new(bb)))
            }
            Ek::Name(n) => self.con_field_type(e.sp, n, &[], &[], false, fields, phi),
            Ek::Call {
                name,
                host,
                con,
                semi,
            } => self.con_field_type(e.sp, name, host, con, *semi, fields, phi),
            Ek::LetAx { name, scrut, body } => {
                let (sc, sty) = self.infer_con(scrut, fields, phi)?;
                let Con::Ax(aty) = sty else {
                    return self.fail(e.sp, "let ax expects an Ax construction");
                };
                let aval = self.nbe.eval_h(&aty, &self.env);
                let lvl = self.fresh();
                self.push_term(name, lvl, aval);
                let phi2 = phi.to_vec();
                // The bound name is a host variable, visible to host arguments.
                let body = self.elab_con_type(body, fields, &phi2);
                self.pop();
                let _ = phi2;
                Ok(Con::LetAx {
                    binder: lvl,
                    scrut: Box::new(sc),
                    body: Box::new(body?),
                })
            }
            _ => self.fail(e.sp, "expected a construction type"),
        }
    }

    fn con_field_type(
        &mut self,
        sp: Span,
        name: &str,
        host: &[E],
        con: &[E],
        semi: bool,
        fields: &[RecField],
        phi: &[(String, u32, Con)],
    ) -> Result<Con, Error> {
        if host.is_empty() && con.is_empty() {
            if let Some((_, _, _)) = phi.iter().find(|(n, _, _)| n == name) {
                return self.fail(sp, format!("`{name}` is a term, not a sort"));
            }
        }
        let (idx, field) = self.find_field(sp, name, fields)?;
        if !matches!(field.kind, RecKind::Sort) {
            return self.fail(sp, format!("`{name}` is a term field"));
        }
        let (hargs, cargs) = self.field_args(sp, field, host, con, semi, fields, phi)?;
        Ok(Con::Field {
            idx,
            host: hargs,
            args: cargs,
        })
    }

    fn find_field<'a>(
        &self,
        sp: Span,
        name: &str,
        fields: &'a [RecField],
    ) -> Result<(usize, &'a RecField), Error> {
        fields
            .iter()
            .enumerate()
            .find(|(_, f)| f.name == name)
            .ok_or_else(|| Error::at(sp, format!("unknown field `{name}`")))
    }

    fn field_args(
        &mut self,
        sp: Span,
        field: &RecField,
        host_e: &[E],
        con_e: &[E],
        semi: bool,
        fields: &[RecField],
        phi: &[(String, u32, Con)],
    ) -> Result<(Vec<HTm>, Vec<Con>), Error> {
        let (host_e, con_e) = if semi {
            (host_e.to_vec(), con_e.to_vec())
        } else if field.delta.is_empty() {
            (Vec::new(), con_e.to_vec())
        } else if field.phi.is_empty() {
            // Arguments were stored in the con vector by the parser.
            (con_e.to_vec(), Vec::new())
        } else {
            return self.fail(
                sp,
                "this field needs both host and construction arguments; use ';'",
            );
        };
        if host_e.len() != field.delta.len() || con_e.len() != field.phi.len() {
            return self.fail(
                sp,
                format!(
                    "field `{}` expects {} host and {} construction arguments",
                    field.name,
                    field.delta.len(),
                    field.phi.len()
                ),
            );
        }
        let mut hargs = Vec::new();
        let saved_env = self.env.clone();
        let saved_len = self.binders.len();
        for (i, arg) in host_e.iter().enumerate() {
            let dom = self.nbe.eval_h(&field.delta[i].2, &self.env);
            let (tm, val) = self.elab_term_check(arg, &dom)?;
            let lvl = field.delta[i].0;
            self.env = self.env.insert(lvl, Slot::H(val));
            hargs.push(tm);
        }
        let mut cargs = Vec::new();
        let mut subst_phi: BTreeMap<u32, Con> = BTreeMap::new();
        for (i, arg) in con_e.iter().enumerate() {
            let dom = subst_con(&field.phi[i].2, &subst_phi, &BTreeMap::new());
            let ct = self.check_con(arg, &dom, fields, phi)?;
            subst_phi.insert(field.phi[i].0, ct.clone());
            cargs.push(ct);
        }
        self.binders.truncate(saved_len);
        self.env = saved_env;
        Ok((hargs, cargs))
    }

    fn infer_con(
        &mut self,
        e: &E,
        fields: &[RecField],
        phi: &[(String, u32, Con)],
    ) -> Result<(Con, Con), Error> {
        match &e.kind {
            Ek::Name(n) => {
                if let Some((_, lvl, ty)) = phi.iter().rev().find(|(name, _, _)| name == n) {
                    return Ok((Con::Var(*lvl), ty.clone()));
                }
                self.infer_field_term(e.sp, n, &[], &[], false, fields, phi)
            }
            Ek::Call {
                name,
                host,
                con,
                semi,
            } => self.infer_field_term(e.sp, name, host, con, *semi, fields, phi),
            Ek::Tt => Ok((Con::Tt, Con::Unit)),
            Ek::Zero => Ok((Con::Z, Con::Nat)),
            Ek::Un(UnOp::Succ, n) => {
                let c = self.check_con(n, &Con::Nat, fields, phi)?;
                Ok((Con::S(Box::new(c)), Con::Nat))
            }
            Ek::Un(UnOp::Refl, a) => {
                let (c, ty) = self.infer_con(a, fields, phi)?;
                Ok((
                    Con::Refl(Box::new(c.clone())),
                    Con::Id(Box::new(ty), Box::new(c.clone()), Box::new(c)),
                ))
            }
            Ek::Un(UnOp::AxM, a) => {
                let (htm, _, _) = self.elab_term_infer(a)?;
                // ax in the construction language introduces Ax(A) for A the type of a.
                // We only have the term. Use its host type via a second check path:
                let (_, _, ty) = self.elab_term_infer(a)?;
                let _ = htm;
                let tm = self.elab_term_check(a, &ty)?.0;
                Ok((
                    Con::AxIn(Box::new(tm)),
                    Con::Ax(Box::new(HTm::Embed(Box::new(ty)))),
                ))
            }
            Ek::NatInd {
                scrut,
                k,
                motive,
                zcase,
                m,
                ih,
                scase,
            } => {
                let (st, _) = self.elab_term_check(scrut, &HVal::Nat)?;
                let kb = self.fresh();
                self.push_term(k, kb, HVal::Nat);
                let mot = self.elab_con_type(motive, fields, phi);
                self.pop();
                let mot = mot?;
                let z = self.check_con(zcase, &mot, fields, phi)?;
                let mb = self.fresh();
                let ib = self.fresh();
                self.push_term(m, mb, HVal::Nat);
                let mut phi2 = phi.to_vec();
                phi2.push((ih.clone(), ib, mot.clone()));
                let s = self.check_con(scase, &mot, fields, &phi2);
                self.pop();
                let s = s?;
                Ok((
                    Con::NInd {
                        scrut: Box::new(st),
                        k: kb,
                        motive: Box::new(mot.clone()),
                        zcase: Box::new(z),
                        m_b: mb,
                        ih_b: ib,
                        scase: Box::new(s),
                    },
                    mot,
                ))
            }
            _ => self.fail(e.sp, "cannot infer this construction"),
        }
    }

    fn infer_field_term(
        &mut self,
        sp: Span,
        name: &str,
        host: &[E],
        con: &[E],
        semi: bool,
        fields: &[RecField],
        phi: &[(String, u32, Con)],
    ) -> Result<(Con, Con), Error> {
        let (idx, field) = self.find_field(sp, name, fields)?;
        let RecKind::Term(ty) = &field.kind else {
            return self.fail(sp, format!("`{name}` is a sort"));
        };
        let (hargs, cargs) = self.field_args(sp, field, host, con, semi, fields, phi)?;
        let mut cmap = BTreeMap::new();
        for (i, c) in cargs.iter().enumerate() {
            cmap.insert(field.phi[i].0, c.clone());
        }
        let mut hmap = BTreeMap::new();
        for (i, h) in hargs.iter().enumerate() {
            hmap.insert(field.delta[i].0, h.clone());
        }
        let got = subst_con(ty, &cmap, &hmap);
        Ok((
            Con::Field {
                idx,
                host: hargs,
                args: cargs,
            },
            got,
        ))
    }

    fn check_con(
        &mut self,
        e: &E,
        expected: &Con,
        fields: &[RecField],
        phi: &[(String, u32, Con)],
    ) -> Result<Con, Error> {
        let (c, got) = self.infer_con(e, fields, phi)?;
        if con_eq(&got, expected) {
            Ok(c)
        } else {
            self.fail(
                e.sp,
                format!("construction mismatch\n  expected {expected:?}\n  got      {got:?}"),
            )
        }
    }

    fn elab_record_model(
        &mut self,
        rec: &RecordTheory,
        fields: &[(Span, String, E)],
    ) -> Result<(), Error> {
        if fields.len() != rec.fields.len() {
            return self.fail(
                fields.first().map(|f| f.0).unwrap_or(Span::new(0, 0)),
                format!(
                    "model provides {} fields, theory has {}",
                    fields.len(),
                    rec.fields.len()
                ),
            );
        }
        let mut impl_lvls: Vec<u32> = Vec::new();
        for (given, field) in fields.iter().zip(rec.fields.iter()) {
            if given.1 != field.name {
                return self.fail(
                    given.0,
                    format!("expected field `{}`, got `{}`", field.name, given.1),
                );
            }
            let expected_tm = self.field_host_type(field, &impl_lvls);
            let expected = self.nbe.eval_h(&expected_tm, &self.env);
            let (_, val) = self.elab_term_check(&given.2, &expected)?;
            let lvl = self.fresh();
            self.env = self.env.insert(lvl, Slot::H(val));
            impl_lvls.push(lvl);
        }
        Ok(())
    }

    fn field_host_type(&mut self, field: &RecField, impl_lvls: &[u32]) -> HTm {
        let ret = match &field.kind {
            RecKind::Sort => HTm::U,
            RecKind::Term(c) => con_to_htm(c, impl_lvls),
        };
        let mut binders: Vec<(u32, HTm)> = Vec::new();
        for (lvl, _, ty) in &field.delta {
            binders.push((*lvl, ty.clone()));
        }
        for (lvl, _, ty) in &field.phi {
            binders.push((*lvl, con_to_htm(ty, impl_lvls)));
        }
        let mut body = ret;
        for (lvl, dom) in binders.into_iter().rev() {
            body = HTm::Pi {
                binder: lvl,
                dom: Box::new(dom),
                cod: Box::new(body),
            };
        }
        body
    }
}

fn weaken_theory(nbe: &Nbe, env: &Env, thy: TVal, past: u32) -> TVal {
    let tm = TTm::Weak {
        tm: Box::new(TTm::Embed(Box::new(thy))),
        past,
    };
    nbe.eval_t(&tm, env)
}

fn quote_type_var(ty: &HVal) -> HTm {
    HTm::Embed(Box::new(ty.clone()))
}

fn subst_con(c: &Con, phi: &BTreeMap<u32, Con>, host: &BTreeMap<u32, HTm>) -> Con {
    match c {
        Con::Var(l) => phi.get(l).cloned().unwrap_or_else(|| Con::Var(*l)),
        Con::Field {
            idx,
            host: hs,
            args,
        } => Con::Field {
            idx: *idx,
            host: hs.iter().map(|h| subst_h(h, host)).collect(),
            args: args.iter().map(|a| subst_con(a, phi, host)).collect(),
        },
        Con::Sigma { binder, dom, cod } => Con::Sigma {
            binder: *binder,
            dom: Box::new(subst_con(dom, phi, host)),
            cod: Box::new(subst_con(cod, phi, host)),
        },
        Con::Pair(a, b) => Con::Pair(
            Box::new(subst_con(a, phi, host)),
            Box::new(subst_con(b, phi, host)),
        ),
        Con::Fst(a) => Con::Fst(Box::new(subst_con(a, phi, host))),
        Con::Snd(a) => Con::Snd(Box::new(subst_con(a, phi, host))),
        Con::Id(t, a, b) => Con::Id(
            Box::new(subst_con(t, phi, host)),
            Box::new(subst_con(a, phi, host)),
            Box::new(subst_con(b, phi, host)),
        ),
        Con::Refl(a) => Con::Refl(Box::new(subst_con(a, phi, host))),
        Con::Empty => Con::Empty,
        Con::Unit => Con::Unit,
        Con::Tt => Con::Tt,
        Con::Nat => Con::Nat,
        Con::Z => Con::Z,
        Con::S(a) => Con::S(Box::new(subst_con(a, phi, host))),
        Con::Sum(a, b) => Con::Sum(
            Box::new(subst_con(a, phi, host)),
            Box::new(subst_con(b, phi, host)),
        ),
        Con::Inl(a) => Con::Inl(Box::new(subst_con(a, phi, host))),
        Con::Inr(a) => Con::Inr(Box::new(subst_con(a, phi, host))),
        Con::Trunc(a) => Con::Trunc(Box::new(subst_con(a, phi, host))),
        Con::TIn(a) => Con::TIn(Box::new(subst_con(a, phi, host))),
        Con::Ax(a) => Con::Ax(Box::new(subst_h(a, host))),
        Con::AxIn(a) => Con::AxIn(Box::new(subst_h(a, host))),
        Con::LetAx {
            binder,
            scrut,
            body,
        } => Con::LetAx {
            binder: *binder,
            scrut: Box::new(subst_con(scrut, phi, host)),
            body: Box::new(subst_con(body, phi, host)),
        },
        Con::NInd {
            scrut,
            k,
            motive,
            zcase,
            m_b,
            ih_b,
            scase,
        } => Con::NInd {
            scrut: Box::new(subst_h(scrut, host)),
            k: *k,
            motive: Box::new(subst_con(motive, phi, host)),
            zcase: Box::new(subst_con(zcase, phi, host)),
            m_b: *m_b,
            ih_b: *ih_b,
            scase: Box::new(subst_con(scase, phi, host)),
        },
        other => other.clone(),
    }
}

fn subst_h(t: &HTm, host: &BTreeMap<u32, HTm>) -> HTm {
    match t {
        HTm::Var(l) => host.get(l).cloned().unwrap_or_else(|| HTm::Var(*l)),
        HTm::App(f, a) => HTm::App(Box::new(subst_h(f, host)), Box::new(subst_h(a, host))),
        HTm::Pi { binder, dom, cod } => HTm::Pi {
            binder: *binder,
            dom: Box::new(subst_h(dom, host)),
            cod: Box::new(subst_h(cod, host)),
        },
        other => other.clone(),
    }
}

fn con_eq(a: &Con, b: &Con) -> bool {
    match (a, b) {
        (Con::Var(i), Con::Var(j)) => i == j,
        (Con::Empty, Con::Empty)
        | (Con::Unit, Con::Unit)
        | (Con::Tt, Con::Tt)
        | (Con::Nat, Con::Nat)
        | (Con::Z, Con::Z) => true,
        (
            Con::Field {
                idx: i,
                host: h1,
                args: a1,
            },
            Con::Field {
                idx: j,
                host: h2,
                args: a2,
            },
        ) => {
            i == j
                && h1.len() == h2.len()
                && a1.len() == a2.len()
                && a1.iter().zip(a2).all(|(x, y)| con_eq(x, y))
                && h1.iter().zip(h2).all(|(x, y)| htm_eq(x, y))
        }
        (Con::Id(t1, a1, b1), Con::Id(t2, a2, b2)) => {
            con_eq(t1, t2) && con_eq(a1, a2) && con_eq(b1, b2)
        }
        (Con::Sum(a, b), Con::Sum(c, d)) => con_eq(a, c) && con_eq(b, d),
        (Con::S(a), Con::S(b)) | (Con::Trunc(a), Con::Trunc(b)) => con_eq(a, b),
        (Con::Ax(a), Con::Ax(b)) => htm_eq(a, b),
        _ => false,
    }
}

fn htm_eq(a: &HTm, b: &HTm) -> bool {
    format!("{a:?}") == format!("{b:?}")
}

fn con_to_htm(c: &Con, impls: &[u32]) -> HTm {
    match c {
        Con::Var(l) => HTm::Var(*l),
        Con::Field { idx, host, args } => {
            let mut f = HTm::Var(impls[*idx]);
            for h in host {
                f = HTm::App(Box::new(f), Box::new(h.clone()));
            }
            for a in args {
                f = HTm::App(Box::new(f), Box::new(con_to_htm(a, impls)));
            }
            f
        }
        Con::Id(t, a, b) => HTm::Id(
            Box::new(con_to_htm(t, impls)),
            Box::new(con_to_htm(a, impls)),
            Box::new(con_to_htm(b, impls)),
        ),
        Con::Empty => HTm::Empty,
        Con::Unit => HTm::Unit,
        Con::Nat => HTm::Nat,
        Con::Sum(a, b) => HTm::Sum(
            Box::new(con_to_htm(a, impls)),
            Box::new(con_to_htm(b, impls)),
        ),
        Con::Sigma { binder, dom, cod } => HTm::Sigma {
            binder: *binder,
            dom: Box::new(con_to_htm(dom, impls)),
            cod: Box::new(con_to_htm(cod, impls)),
        },
        Con::Ax(h) => (**h).clone(),
        Con::Trunc(_) => HTm::U,
        _ => HTm::Embed(Box::new(HVal::U)),
        // `other` is unused below; the wildcard above already covers it.
    }
}
