use gtt::check_source;
use std::fs;
use std::path::PathBuf;

fn files(dir: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("gtt") {
            out.push(path);
        }
    }
    out.sort();
    out
}

#[test]
fn positive_examples() {
    for dir in ["examples/emtt", "examples/classify", "examples/theories"] {
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        let _ = entries;
        for path in files(dir) {
            let src = fs::read_to_string(&path).unwrap();
            check_source(&src).unwrap_or_else(|e| {
                panic!("{}:\n{}", path.display(), e.render(&src));
            });
        }
    }
}

#[test]
fn negative_examples() {
    for path in files("examples/negative") {
        let src = fs::read_to_string(&path).unwrap();
        assert!(
            check_source(&src).is_err(),
            "{} should fail",
            path.display()
        );
    }
}
