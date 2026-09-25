//! Recursive-descent parser for the concrete syntax.

use crate::error::{Error, Span};
use crate::surface::*;

pub struct Parser<'a> {
    src: &'a str,
    i: usize,
}

pub fn parse_file(src: &str) -> Result<Vec<Item>, Error> {
    let mut p = Parser { src, i: 0 };
    p.skip();
    let mut items = Vec::new();
    while !p.eof() {
        items.push(p.item()?);
        p.skip();
    }
    Ok(items)
}

impl<'a> Parser<'a> {
    fn eof(&self) -> bool {
        self.i >= self.src.len()
    }

    fn rest(&self) -> &'a str {
        &self.src[self.i..]
    }

    fn skip(&mut self) {
        loop {
            let s = self.rest();
            if s.starts_with("--") {
                self.i += 2;
                while let Some(c) = self.peek_char() {
                    self.i += c.len_utf8();
                    if c == '\n' {
                        break;
                    }
                }
                continue;
            }
            if let Some(c) = self.peek_char() {
                if c.is_whitespace() {
                    self.i += c.len_utf8();
                    continue;
                }
            }
            break;
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn starts(&self, s: &str) -> bool {
        self.rest().starts_with(s)
    }

    fn eat(&mut self, s: &str) -> bool {
        self.skip();
        if self.starts(s) {
            // Don't eat a longer identifier or arrow as a prefix by accident.
            let next = self.rest()[s.len()..].chars().next();
            let boundary_ok = match s {
                w if is_word(w) => next.map(|c| !is_ident_char(c)).unwrap_or(true),
                _ => true,
            };
            if boundary_ok {
                self.i += s.len();
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    fn expect(&mut self, s: &str) -> Result<(), Error> {
        if self.eat(s) {
            Ok(())
        } else {
            Err(self.err(format!("expected '{s}'")))
        }
    }

    fn err(&self, msg: String) -> Error {
        Error::at(Span::new(self.i, self.i), msg)
    }

    fn span_from(&self, start: usize) -> Span {
        Span::new(start, self.i)
    }

    fn ident(&mut self) -> Result<String, Error> {
        self.skip();
        let start = self.i;
        let mut chars = self.rest().chars();
        let Some(c) = chars.next() else {
            return Err(self.err("expected a name".into()));
        };
        if !(c.is_ascii_alphabetic() || c == '_') {
            return Err(self.err("expected a name".into()));
        }
        self.i += c.len_utf8();
        while let Some(c) = self.peek_char() {
            if is_ident_char(c) {
                self.i += c.len_utf8();
            } else {
                break;
            }
        }
        Ok(self.src[start..self.i].to_string())
    }

    fn e(&self, start: usize, kind: Ek) -> E {
        E {
            sp: self.span_from(start),
            kind,
        }
    }

    fn item(&mut self) -> Result<Item, Error> {
        self.skip();
        let start = self.i;
        if self.eat("postulate") {
            let name = self.ident()?;
            self.expect(":")?;
            let ty = self.expr()?;
            return Ok(Item::Postulate {
                sp: self.span_from(start),
                name,
                ty,
            });
        }
        if self.eat("def") {
            let name = self.ident()?;
            self.expect(":")?;
            let ty = self.expr()?;
            self.expect("=")?;
            let body = self.expr()?;
            return Ok(Item::Def {
                sp: self.span_from(start),
                name,
                ty,
                body,
            });
        }
        if self.eat("theory") {
            let name = self.ident()?;
            if self.eat(":") {
                self.expect("Theory")?;
                self.expect("=")?;
                let body = self.expr()?;
                return Ok(Item::TheoryB {
                    sp: self.span_from(start),
                    name,
                    body,
                });
            }
            self.expect("where")?;
            let mut fields = Vec::new();
            while !self.eat("end") {
                if self.eof() {
                    return Err(self.err("unclosed theory".into()));
                }
                fields.push(self.field()?);
            }
            return Ok(Item::TheoryA {
                sp: self.span_from(start),
                name,
                fields,
            });
        }
        if self.eat("model") {
            let name = self.ident()?;
            self.expect("::")?;
            let thy = self.ident()?;
            if self.eat("where") {
                let mut fields = Vec::new();
                while !self.eat("end") {
                    if self.eof() {
                        return Err(self.err("unclosed model".into()));
                    }
                    let fs = self.i;
                    let fname = self.ident()?;
                    self.expect("=")?;
                    let body = self.expr()?;
                    self.expect(";")?;
                    fields.push((self.span_from(fs), fname, body));
                }
                return Ok(Item::ModelA {
                    sp: self.span_from(start),
                    name,
                    thy,
                    fields,
                });
            }
            self.expect("=")?;
            let body = self.expr()?;
            return Ok(Item::ModelB {
                sp: self.span_from(start),
                name,
                thy,
                body,
            });
        }
        if self.eat("context") {
            let mut binders = Vec::new();
            while self.starts_paren() {
                binders.push(self.binder()?);
            }
            self.expect("in")?;
            let mut items = Vec::new();
            while !self.eat("end") {
                if self.eof() {
                    return Err(self.err("unclosed context".into()));
                }
                items.push(self.item()?);
            }
            return Ok(Item::Context {
                sp: self.span_from(start),
                binders,
                items,
            });
        }
        if self.eat("check") {
            let expr = self.expr()?;
            if self.eat("type") {
                return Ok(Item::CheckType {
                    sp: self.span_from(start),
                    expr,
                });
            }
            if self.eat("::") {
                let thy = self.expr()?;
                return Ok(Item::CheckModel {
                    sp: self.span_from(start),
                    expr,
                    thy,
                });
            }
            self.expect(":")?;
            let ty = self.expr()?;
            return Ok(Item::CheckTerm {
                sp: self.span_from(start),
                expr,
                ty,
            });
        }
        let negate = self.eat("not");
        if self.eat("defeq") {
            let a = self.expr()?;
            self.expect("=")?;
            let b = self.expr()?;
            let ty = if self.eat(":") {
                Some(self.expr()?)
            } else {
                None
            };
            return Ok(Item::Defeq {
                sp: self.span_from(start),
                negate,
                a,
                b,
                ty,
            });
        }
        if negate {
            return Err(self.err("expected defeq after not".into()));
        }
        Err(self.err("expected a declaration".into()))
    }

    fn starts_paren(&mut self) -> bool {
        self.skip();
        self.starts("(")
    }

    fn binder(&mut self) -> Result<BinderS, Error> {
        self.skip();
        let start = self.i;
        self.expect("(")?;
        let name = self.ident()?;
        let model = if self.eat("::") {
            true
        } else {
            self.expect(":")?;
            false
        };
        let ty = self.expr()?;
        self.expect(")")?;
        Ok(BinderS {
            sp: self.span_from(start),
            name,
            model,
            ty,
        })
    }

    fn field(&mut self) -> Result<FieldS, Error> {
        self.skip();
        let start = self.i;
        if self.eat("sort") {
            let name = self.ident()?;
            let (delta, phi) = if self.starts_paren() {
                self.params()?
            } else {
                (Vec::new(), Vec::new())
            };
            self.expect(";")?;
            Ok(FieldS::Sort {
                sp: self.span_from(start),
                name,
                delta,
                phi,
            })
        } else if self.eat("term") {
            let name = self.ident()?;
            let (delta, phi) = if self.starts_paren() {
                self.params()?
            } else {
                (Vec::new(), Vec::new())
            };
            self.expect(":")?;
            let ty = self.expr()?;
            self.expect(";")?;
            Ok(FieldS::Term {
                sp: self.span_from(start),
                name,
                delta,
                phi,
                ty,
            })
        } else {
            Err(self.err("expected sort or term".into()))
        }
    }

    fn params(&mut self) -> Result<(Vec<TeleParam>, Vec<TeleParam>), Error> {
        self.expect("(")?;
        if self.eat(")") {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut host = Vec::new();
        let mut con = Vec::new();
        let mut semi = false;
        loop {
            let name = self.ident()?;
            self.expect(":")?;
            let ty = self.expr()?;
            let p = TeleParam { name, ty };
            if semi {
                con.push(p);
            } else {
                host.push(p);
            }
            self.skip();
            if self.eat(";") {
                semi = true;
                if self.eat(")") {
                    break;
                }
                continue;
            }
            if self.eat(",") {
                continue;
            }
            self.expect(")")?;
            break;
        }
        if semi {
            Ok((host, con))
        } else {
            // No semicolon: these are construction parameters (the common case).
            Ok((Vec::new(), host))
        }
    }

    fn expr(&mut self) -> Result<E, Error> {
        self.skip();
        if self.binder_paren() {
            return self.binder_expr();
        }
        self.eq_expr()
    }

    /// `(name : …)` or `(name :: …)` at the cursor.
    fn binder_paren(&mut self) -> bool {
        self.skip();
        if !self.starts("(") {
            return false;
        }
        let saved = self.i;
        self.i += 1;
        self.skip();
        let ok = self.peek_ident().is_some() && {
            let _ = self.ident();
            self.skip();
            self.starts(":")
        };
        self.i = saved;
        ok
    }

    fn peek_ident(&mut self) -> Option<()> {
        self.skip();
        let c = self.peek_char()?;
        if c.is_ascii_alphabetic() || c == '_' {
            Some(())
        } else {
            None
        }
    }

    fn binder_expr(&mut self) -> Result<E, Error> {
        let start = self.i;
        self.expect("(")?;
        let name = self.ident()?;
        let model = if self.eat("::") {
            true
        } else {
            self.expect(":")?;
            false
        };
        let dom = self.expr()?;
        self.expect(")")?;
        if self.eat("->") || self.eat("→") {
            if model {
                return Err(Error::at(
                    self.span_from(start),
                    "a model binder is not the domain of a host function",
                ));
            }
            let cod = self.expr()?;
            return Ok(self.e(
                start,
                Ek::Arrow {
                    binder: Some(name),
                    dom: Box::new(dom),
                    cod: Box::new(cod),
                },
            ));
        }
        if self.eat("|>") || self.eat("▹") {
            if model {
                return Err(Error::at(
                    self.span_from(start),
                    "▹ binds a host element; use (x : A) ▹ T",
                ));
            }
            let cod = self.expr()?;
            return Ok(self.e(
                start,
                Ek::Rtri {
                    binder: Some(name),
                    dom: Box::new(dom),
                    cod: Box::new(cod),
                },
            ));
        }
        if self.eat("*") || self.eat("×") {
            let cod = self.expr()?;
            return Ok(self.e(
                start,
                Ek::Prod {
                    binder: Some(name),
                    model,
                    dom: Box::new(dom),
                    cod: Box::new(cod),
                },
            ));
        }
        Err(Error::at(
            self.span_from(start),
            "expected ->, |>, or * after a binder",
        ))
    }

    fn eq_expr(&mut self) -> Result<E, Error> {
        let start = self.i;
        let left = self.arrow()?;
        if self.eat("==") {
            let right = self.arrow()?;
            Ok(self.e(
                start,
                Ek::Id {
                    ty: None,
                    a: Box::new(left),
                    b: Box::new(right),
                },
            ))
        } else {
            Ok(left)
        }
    }

    fn arrow(&mut self) -> Result<E, Error> {
        let start = self.i;
        let left = self.sum()?;
        if self.eat("->") || self.eat("→") {
            let right = self.arrow()?;
            Ok(self.e(
                start,
                Ek::Arrow {
                    binder: None,
                    dom: Box::new(left),
                    cod: Box::new(right),
                },
            ))
        } else if self.eat("|>") || self.eat("▹") {
            let right = self.arrow()?;
            Ok(self.e(
                start,
                Ek::Rtri {
                    binder: None,
                    dom: Box::new(left),
                    cod: Box::new(right),
                },
            ))
        } else {
            Ok(left)
        }
    }

    fn sum(&mut self) -> Result<E, Error> {
        let start = self.i;
        let mut left = self.prod()?;
        while self.eat("+") {
            let right = self.prod()?;
            left = self.e(start, Ek::Sum(Box::new(left), Box::new(right)));
        }
        Ok(left)
    }

    fn prod(&mut self) -> Result<E, Error> {
        let start = self.i;
        let left = self.app()?;
        if self.eat("*") || self.eat("×") {
            let right = self.prod()?;
            Ok(self.e(
                start,
                Ek::Prod {
                    binder: None,
                    model: false,
                    dom: Box::new(left),
                    cod: Box::new(right),
                },
            ))
        } else {
            Ok(left)
        }
    }

    fn app(&mut self) -> Result<E, Error> {
        let start = self.i;
        let mut fun = self.post()?;
        loop {
            self.skip();
            if self.eof() || self.starts("=") || !self.can_start_atom() {
                break;
            }
            // `end`, `inl`, `inr`, `zero`, `succ`, `refl` as keywords stop application
            // when they begin a new clause. `zero` and names are valid arguments.
            if self.keyword_stops_app() {
                break;
            }
            let arg = self.atom()?;
            fun = self.e(start, Ek::App(Box::new(fun), Box::new(arg)));
        }
        Ok(fun)
    }

    fn keyword_stops_app(&mut self) -> bool {
        self.skip();
        const STOP: &[&str] = &[
            "end",
            "with",
            "inl",
            "inr",
            "zero",
            "succ",
            "refl",
            "in",
            "where",
            "as",
            "return",
            ":",
            "model",
            "theory",
            "context",
            "check",
            "defeq",
            "postulate",
            "def",
            "not",
            "sort",
            "term",
            "type",
        ];
        for s in STOP {
            if self.starts(s) {
                let next = self.rest()[s.len()..].chars().next();
                if next.map(|c| !is_ident_char(c)).unwrap_or(true) {
                    return true;
                }
            }
        }
        false
    }

    fn can_start_atom(&mut self) -> bool {
        self.skip();
        matches!(self.peek_char(), Some('(' | '\\' | 'λ' | '★')) || self.peek_ident().is_some()
    }

    fn post(&mut self) -> Result<E, Error> {
        let start = self.i;
        let mut e = self.prefix()?;
        loop {
            self.skip();
            if self.eat("^") || self.eat("↑") {
                let name = self.ident()?;
                e = self.e(
                    start,
                    Ek::Weak {
                        tm: Box::new(e),
                        name,
                    },
                );
            } else if self.eat("{") {
                let repl = self.expr()?;
                self.expect("/")?;
                let name = self.ident()?;
                self.expect("}")?;
                e = self.e(
                    start,
                    Ek::Subst {
                        tm: Box::new(e),
                        repl: Box::new(repl),
                        name,
                    },
                );
            } else {
                break;
            }
        }
        Ok(e)
    }

    fn prefix(&mut self) -> Result<E, Error> {
        self.skip();
        let start = self.i;
        let un = [
            ("fst", UnOp::Fst),
            ("snd", UnOp::Snd),
            ("pr1", UnOp::Pr1),
            ("pr2", UnOp::Pr2),
            ("inl", UnOp::Inl),
            ("inr", UnOp::Inr),
            ("succ", UnOp::Succ),
            ("refl", UnOp::Refl),
            ("unax", UnOp::Unax),
            ("ty", UnOp::Ty),
            ("sort", UnOp::SortM),
            ("ax", UnOp::AxM),
            ("Ax", UnOp::AxTy),
            ("Trunc", UnOp::Trunc),
            ("abort", UnOp::Abort),
        ];
        for (w, op) in un {
            if self.eat(w) {
                let arg = self.atom()?;
                return Ok(self.e(start, Ek::Un(op, Box::new(arg))));
            }
        }
        if self.eat("Id") {
            let t = self.atom()?;
            let a = self.atom()?;
            let b = self.atom()?;
            return Ok(self.e(
                start,
                Ek::Id {
                    ty: Some(Box::new(t)),
                    a: Box::new(a),
                    b: Box::new(b),
                },
            ));
        }
        if self.eat("match") {
            return self.match_expr(start);
        }
        if self.eat("j") {
            return self.j_expr(start);
        }
        if self.eat("natind") || self.eat("nind") {
            return self.natind_expr(start);
        }
        if self.eat("Ind") {
            return self.ind_ty(start);
        }
        if self.eat("ind") {
            return self.ind_m(start);
        }
        if self.eat("let") {
            return self.let_ax(start);
        }
        self.atom()
    }

    fn match_expr(&mut self, start: usize) -> Result<E, Error> {
        let scrut = self.eq_expr()?;
        self.expect("as")?;
        let binder = self.ident()?;
        self.expect("return")?;
        let motive = self.eq_expr()?;
        self.expect("with")?;
        self.expect("inl")?;
        let inl_name = self.ident()?;
        self.expect("=>")?;
        let inl = self.expr()?;
        self.expect("inr")?;
        let inr_name = self.ident()?;
        self.expect("=>")?;
        let inr = self.expr()?;
        self.expect("end")?;
        Ok(self.e(
            start,
            Ek::Match {
                scrut: Box::new(scrut),
                binder,
                motive: Box::new(motive),
                inl_name,
                inl: Box::new(inl),
                inr_name,
                inr: Box::new(inr),
            },
        ))
    }

    fn j_expr(&mut self, start: usize) -> Result<E, Error> {
        let path = self.eq_expr()?;
        self.expect("as")?;
        let y = self.ident()?;
        let p = self.ident()?;
        self.expect("return")?;
        let motive = self.eq_expr()?;
        self.expect("refl")?;
        self.expect("=>")?;
        let refl_case = self.expr()?;
        self.expect("end")?;
        Ok(self.e(
            start,
            Ek::J {
                path: Box::new(path),
                y,
                p,
                motive: Box::new(motive),
                refl_case: Box::new(refl_case),
            },
        ))
    }

    fn natind_expr(&mut self, start: usize) -> Result<E, Error> {
        let scrut = self.eq_expr()?;
        self.expect("as")?;
        let k = self.ident()?;
        self.expect("return")?;
        let motive = self.eq_expr()?;
        self.expect("zero")?;
        self.expect("=>")?;
        let zcase = self.expr()?;
        self.expect("succ")?;
        let m = self.ident()?;
        let ih = self.ident()?;
        self.expect("=>")?;
        let scase = self.expr()?;
        self.expect("end")?;
        Ok(self.e(
            start,
            Ek::NatInd {
                scrut: Box::new(scrut),
                k,
                motive: Box::new(motive),
                zcase: Box::new(zcase),
                m,
                ih,
                scase: Box::new(scase),
            },
        ))
    }

    fn ind_ty(&mut self, start: usize) -> Result<E, Error> {
        let scrut = self.eq_expr()?;
        self.expect("inl")?;
        let lname = self.ident()?;
        self.expect("=>")?;
        let left = self.expr()?;
        self.expect("inr")?;
        let rname = self.ident()?;
        self.expect("=>")?;
        let right = self.expr()?;
        self.expect("end")?;
        Ok(self.e(
            start,
            Ek::IndTy {
                scrut: Box::new(scrut),
                lname,
                left: Box::new(left),
                rname,
                right: Box::new(right),
            },
        ))
    }

    fn ind_m(&mut self, start: usize) -> Result<E, Error> {
        let scrut = self.eq_expr()?;
        self.expect("as")?;
        let x = self.ident()?;
        self.expect("return")?;
        let motive = self.expr()?;
        self.expect("inl")?;
        let lname = self.ident()?;
        self.expect("=>")?;
        let left = self.expr()?;
        self.expect("inr")?;
        let rname = self.ident()?;
        self.expect("=>")?;
        let right = self.expr()?;
        self.expect("end")?;
        Ok(self.e(
            start,
            Ek::IndM {
                scrut: Box::new(scrut),
                x,
                motive: Box::new(motive),
                lname,
                left: Box::new(left),
                rname,
                right: Box::new(right),
            },
        ))
    }

    fn let_ax(&mut self, start: usize) -> Result<E, Error> {
        self.expect("ax")?;
        let name = self.ident()?;
        self.expect(":=")?;
        let scrut = self.expr()?;
        self.expect("in")?;
        let body = self.expr()?;
        Ok(self.e(
            start,
            Ek::LetAx {
                name,
                scrut: Box::new(scrut),
                body: Box::new(body),
            },
        ))
    }

    fn atom(&mut self) -> Result<E, Error> {
        self.skip();
        let start = self.i;
        if self.eat("(") {
            if self.eat(")") {
                return Ok(self.e(start, Ek::Tt));
            }
            let first = self.expr()?;
            if self.eat(",") {
                let second = self.expr()?;
                self.expect(")")?;
                return Ok(self.e(start, Ek::Pair(Box::new(first), Box::new(second))));
            }
            self.expect(")")?;
            return Ok(first);
        }
        if self.eat("\\") || self.eat("λ") {
            let name = self.ident()?;
            // `\x y. body` is nested.
            let mut names = vec![name];
            while !self.eat(".") {
                if self.starts(":") || self.starts("(") {
                    return Err(self.err("expected '.' in a lambda".into()));
                }
                names.push(self.ident()?);
            }
            let mut body = self.expr()?;
            for name in names.into_iter().rev() {
                body = self.e(
                    start,
                    Ek::Lam {
                        name,
                        body: Box::new(body),
                    },
                );
            }
            return Ok(body);
        }
        if self.eat("★") {
            return Ok(self.e(start, Ek::Tt));
        }
        if self.peek_ident().is_some() {
            let name = self.ident()?;
            // `f(x, y)` is a call. `f (x)` is juxtaposition, so the parenthesis
            // must follow the name immediately.
            if self.starts("(") {
                self.expect("(")?;
                let (host, con, semi) = self.call_args()?;
                return Ok(self.e(
                    start,
                    Ek::Call {
                        name,
                        host,
                        con,
                        semi,
                    },
                ));
            }
            let kind = match name.as_str() {
                "tt" => Ek::Tt,
                "U" => Ek::U,
                "Nat" => Ek::Nat,
                "Empty" => Ek::Empty,
                "Unit" => Ek::Unit,
                "One" => Ek::One,
                "Sort" => Ek::Sort,
                _ => Ek::Name(name),
            };
            return Ok(self.e(start, kind));
        }
        Err(self.err("expected an expression".into()))
    }

    fn skip_peek(&mut self, s: &str) -> bool {
        self.skip();
        self.starts(s)
    }

    fn call_args(&mut self) -> Result<(Vec<E>, Vec<E>, bool), Error> {
        if self.eat(")") {
            return Ok((Vec::new(), Vec::new(), false));
        }
        let mut host = Vec::new();
        let mut con = Vec::new();
        let mut semi = false;
        loop {
            let e = self.expr()?;
            if semi {
                con.push(e);
            } else {
                host.push(e);
            }
            if self.eat(";") {
                semi = true;
                if self.eat(")") {
                    break;
                }
                continue;
            }
            if self.eat(",") {
                continue;
            }
            self.expect(")")?;
            break;
        }
        if semi {
            Ok((host, con, true))
        } else {
            Ok((Vec::new(), host, false))
        }
    }
}

fn is_word(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '\''
}
