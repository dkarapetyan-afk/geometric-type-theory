//! Source locations and type errors.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Error {
    pub msg: String,
    pub span: Option<Span>,
}

impl Error {
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            msg: msg.into(),
            span: None,
        }
    }

    pub fn at(span: Span, msg: impl Into<String>) -> Self {
        Self {
            msg: msg.into(),
            span: Some(span),
        }
    }

    pub fn render(&self, src: &str) -> String {
        let Some(sp) = self.span else {
            return self.msg.clone();
        };
        let (line, col) = line_col(src, sp.start);
        let snippet: String = src
            .lines()
            .nth(line.saturating_sub(1))
            .unwrap_or("")
            .trim_end()
            .to_string();
        format!(
            "{}:{}: {}\n  {}\n  {}^",
            line,
            col,
            self.msg,
            snippet,
            " ".repeat(col.saturating_sub(1))
        )
    }
}

pub fn line_col(src: &str, pos: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    for (i, c) in src.char_indices() {
        if i >= pos {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}
