use std::{
    collections::BTreeMap,
    fs,
    ops::RangeInclusive,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use proc_macro2::TokenStream;
use scnr2_generate::{
    character_classes::CharacterClasses,
    dfa::Dfa,
    nfa::Nfa,
    scanner_data::{ScannerData, TransitionToNumericMode},
    scanner_mode::ScannerMode,
};
use syn::visit::Visit;

#[derive(Debug, Parser)]
#[command(name = "scnr2_viz")]
#[command(about = "Generate Graphviz DOT output for scanner! mode DFAs")]
struct Cli {
    /// Path to a Rust file that contains at least one scanner! macro invocation.
    #[arg(short, long)]
    input: PathBuf,

    /// Path of the generated .dot file.
    #[arg(short, long)]
    output: PathBuf,

    /// Zero-based scanner! macro index if the input file has multiple invocations.
    #[arg(long, default_value_t = 0)]
    macro_index: usize,
}

#[derive(Debug)]
struct BuildArtifacts {
    scanner_name: String,
    scanner_modes: Vec<ScannerMode>,
    dfas: Vec<Dfa>,
    character_classes: CharacterClasses,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let artifacts = build_artifacts(&cli.input, cli.macro_index)?;
    let dot = render_dot(&artifacts)?;

    write_output(&cli.output, &dot)?;

    println!(
        "Generated DOT for scanner '{}' with {} mode(s) at {}",
        artifacts.scanner_name,
        artifacts.scanner_modes.len(),
        cli.output.display()
    );

    Ok(())
}

fn build_artifacts(input_path: &Path, macro_index: usize) -> Result<BuildArtifacts> {
    let input = fs::read_to_string(input_path)
        .with_context(|| format!("Failed to read input file {}", input_path.display()))?;
    let syntax = syn::parse_file(&input)
        .with_context(|| format!("Failed to parse Rust file {}", input_path.display()))?;
    let scanner_macro_tokens = extract_scanner_macro_tokens(&syntax, macro_index)?;

    let scanner_data: ScannerData = syn::parse2(scanner_macro_tokens)
        .map_err(|e| anyhow!("Failed to parse scanner! macro payload: {e}"))?;

    let scanner_name = scanner_data.name.clone();
    let scanner_modes = scanner_data
        .build_scanner_modes()
        .map_err(|e| anyhow!("Failed to build scanner modes: {e}"))?;

    let (dfas, character_classes) = build_dfas_and_classes(&scanner_modes)?;

    Ok(BuildArtifacts {
        scanner_name,
        scanner_modes,
        dfas,
        character_classes,
    })
}

fn build_dfas_and_classes(scanner_modes: &[ScannerMode]) -> Result<(Vec<Dfa>, CharacterClasses)> {
    let mut nfas = scanner_modes
        .iter()
        .map(|mode| Nfa::build_from_patterns(&mode.patterns))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow!("Failed to build NFA: {e}"))?;

    let mut character_classes = CharacterClasses::new();
    for nfa in &nfas {
        nfa.collect_character_classes(&mut character_classes);
    }
    character_classes.create_disjoint_character_classes();
    for nfa in &mut nfas {
        nfa.convert_to_disjoint_character_classes(&character_classes);
    }

    let dfas = nfas
        .iter()
        .map(Dfa::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow!("Failed to convert NFA to DFA: {e}"))?;

    Ok((dfas, character_classes))
}

fn extract_scanner_macro_tokens(file: &syn::File, macro_index: usize) -> Result<TokenStream> {
    struct ScannerMacroCollector {
        macros: Vec<TokenStream>,
    }

    impl<'ast> Visit<'ast> for ScannerMacroCollector {
        fn visit_macro(&mut self, mac: &'ast syn::Macro) {
            let is_scanner = mac.path.is_ident("scanner")
                || mac
                    .path
                    .segments
                    .last()
                    .map(|seg| seg.ident == "scanner")
                    .unwrap_or(false);

            if is_scanner {
                self.macros.push(mac.tokens.clone());
            }
            syn::visit::visit_macro(self, mac);
        }
    }

    let mut collector = ScannerMacroCollector { macros: Vec::new() };
    collector.visit_file(file);

    if collector.macros.is_empty() {
        bail!("No scanner! macro invocations found in input Rust file")
    }
    if macro_index >= collector.macros.len() {
        bail!(
            "Requested --macro-index {} but only {} scanner! invocation(s) were found",
            macro_index,
            collector.macros.len()
        )
    }

    Ok(collector.macros.remove(macro_index))
}

fn render_dot(artifacts: &BuildArtifacts) -> Result<String> {
    if artifacts.scanner_modes.len() != artifacts.dfas.len() {
        bail!(
            "Inconsistent build result: {} modes but {} DFAs",
            artifacts.scanner_modes.len(),
            artifacts.dfas.len()
        );
    }

    let mut out = String::new();
    out.push_str("digraph scanner_modes {\n");
    out.push_str("  rankdir=LR;\n");
    out.push_str("  compound=true;\n");
    out.push_str("  labelloc=\"t\";\n");
    out.push_str(&format!(
        "  label=\"SCNR2 DFA Visualization: {}\";\n",
        escape_dot(&artifacts.scanner_name)
    ));
    out.push_str("  node [fontname=\"Helvetica\"];\n\n");

    for (mode_idx, mode) in artifacts.scanner_modes.iter().enumerate() {
        let dfa = &artifacts.dfas[mode_idx];
        out.push_str(&format!(
            "  mode_{mode_idx} [shape=box, style=\"rounded,filled\", fillcolor=\"#eef5ff\", label=\"mode {}\"];\n",
            escape_dot(&mode.name)
        ));

        out.push_str(&format!("  subgraph cluster_mode_{mode_idx} {{\n"));
        out.push_str("    style=rounded;\n");
        out.push_str("    color=\"#8aa1c7\";\n");
        out.push_str(&format!(
            "    label=\"DFA for mode {} (index {})\";\n",
            escape_dot(&mode.name),
            mode_idx
        ));

        for (state_idx, state) in dfa.states.iter().enumerate() {
            let node_name = dfa_node_name(mode_idx, state_idx);
            let is_accepting = !state.accept_data.is_empty();
            let shape = if is_accepting {
                "doublecircle"
            } else {
                "circle"
            };

            let mut label = format!("s{state_idx}");
            if is_accepting {
                let terminal_ids = state
                    .accept_data
                    .iter()
                    .map(|a| a.terminal_type.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                label.push_str("\\n");
                label.push_str(&format!("accept: {terminal_ids}"));
            }

            out.push_str(&format!(
                "    {node_name} [shape={shape}, label=\"{}\"];\n",
                escape_dot(&label)
            ));
        }

        if !dfa.states.is_empty() {
            out.push_str(&format!(
                "    mode_{mode_idx} -> {} [lhead=cluster_mode_{mode_idx}, label=\"start\"];\n",
                dfa_node_name(mode_idx, 0)
            ));
        }

        for (state_idx, state) in dfa.states.iter().enumerate() {
            let from = dfa_node_name(mode_idx, state_idx);
            let grouped = group_transitions_by_target(state, &artifacts.character_classes)?;
            for (target, labels) in grouped {
                let to = dfa_node_name(mode_idx, target);
                let label = labels.join("\\n");
                out.push_str(&format!(
                    "    {from} -> {to} [label=\"{}\"];\n",
                    escape_dot(&label)
                ));
            }
        }

        out.push_str("  }\n\n");
    }

    out.push_str("  mode_stack [shape=diamond, style=filled, fillcolor=\"#f8f8f8\", label=\"mode stack\"];\n\n");

    for (mode_idx, mode) in artifacts.scanner_modes.iter().enumerate() {
        for transition in &mode.transitions {
            match transition {
                TransitionToNumericMode::SetMode(token, target_mode_idx) => {
                    out.push_str(&format!(
                        "  mode_{mode_idx} -> mode_{target_mode_idx} [style=dashed, color=\"#1d6f42\", label=\"on {}: enter\"];\n",
                        token
                    ));
                }
                TransitionToNumericMode::PushMode(token, target_mode_idx) => {
                    out.push_str(&format!(
                        "  mode_{mode_idx} -> mode_{target_mode_idx} [style=dashed, color=\"#b06a00\", label=\"on {}: push\"];\n",
                        token
                    ));
                }
                TransitionToNumericMode::PopMode(token) => {
                    out.push_str(&format!(
                        "  mode_{mode_idx} -> mode_stack [style=dashed, color=\"#7a2f6d\", label=\"on {}: pop\"];\n",
                        token
                    ));
                }
            }
        }
    }

    out.push_str("}\n");
    Ok(out)
}

fn group_transitions_by_target(
    state: &scnr2_generate::dfa::DfaState,
    character_classes: &CharacterClasses,
) -> Result<BTreeMap<usize, Vec<String>>> {
    let mut grouped: BTreeMap<usize, Vec<String>> = BTreeMap::new();

    for transition in &state.transitions {
        let target = transition.target.as_usize();
        let class_idx = transition.elementary_interval_index.as_usize();
        let intervals = character_classes
            .intervals
            .get(class_idx)
            .ok_or_else(|| anyhow!("Invalid disjoint class index {class_idx} in DFA transition"))?;
        let rendered = render_interval_group(intervals);
        grouped.entry(target).or_default().push(rendered);
    }

    for labels in grouped.values_mut() {
        labels.sort();
    }

    Ok(grouped)
}

fn render_interval_group(intervals: &[RangeInclusive<char>]) -> String {
    let mut parts = intervals
        .iter()
        .map(|r| render_range(r.start(), r.end()))
        .collect::<Vec<_>>();
    parts.sort();
    parts.join(", ")
}

fn render_range(start: &char, end: &char) -> String {
    if start == end {
        printable_char(*start)
    } else {
        format!("{}-{}", printable_char(*start), printable_char(*end))
    }
}

fn printable_char(c: char) -> String {
    match c {
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        '"' => "\\\"".to_string(),
        '\\' => "\\\\".to_string(),
        ch if ch.is_control() => format!("\\u{{{:x}}}", ch as u32),
        ch => ch.to_string(),
    }
}

fn dfa_node_name(mode_idx: usize, state_idx: usize) -> String {
    format!("m{mode_idx}_s{state_idx}")
}

fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn write_output(path: &Path, dot: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create output directory {}", parent.display())
            })?;
        }
    }
    fs::write(path, dot)
        .with_context(|| format!("Failed to write DOT file {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_range_simple() {
        assert_eq!(render_range(&'a', &'a'), "a");
        assert_eq!(render_range(&'a', &'z'), "a-z");
    }

    #[test]
    fn test_printable_char_escapes_specials() {
        assert_eq!(printable_char('\n'), "\\n");
        assert_eq!(printable_char('"'), "\\\"");
        assert_eq!(printable_char('x'), "x");
    }

    #[test]
    fn test_escape_dot() {
        assert_eq!(escape_dot("a\\b\"c"), "a\\\\b\\\"c");
    }
}
