use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
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

    /// Optional path for generated SVG output (requires Graphviz `dot` in PATH).
    #[arg(long)]
    svg_output: Option<PathBuf>,

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
    token_names_by_id: BTreeMap<String, String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let artifacts = build_artifacts(&cli.input, cli.macro_index)?;
    let dot = render_dot(&artifacts)?;
    let classes_json = render_disjunct_classes_json(&artifacts.character_classes);
    let classes_output_path = disjunct_classes_output_path(&cli.output);

    write_output(&cli.output, &dot)?;
    write_output(&classes_output_path, &classes_json)?;
    if let Some(svg_output_path) = &cli.svg_output {
        write_svg_output(&cli.output, svg_output_path)?;
    }

    println!(
        "Generated DOT for scanner '{}' with {} mode(s) at {} and disjunct classes at {}{}",
        artifacts.scanner_name,
        artifacts.scanner_modes.len(),
        cli.output.display(),
        classes_output_path.display(),
        cli.svg_output
            .as_ref()
            .map(|path| format!(", and SVG at {}", path.display()))
            .unwrap_or_default()
    );

    Ok(())
}

fn build_artifacts(input_path: &Path, macro_index: usize) -> Result<BuildArtifacts> {
    let input = fs::read_to_string(input_path)
        .with_context(|| format!("Failed to read input file {}", input_path.display()))?;
    let syntax = syn::parse_file(&input).map_err(|e| {
        anyhow!(
            "Failed to parse Rust file {}: {e}\nHint: Input must be a Rust source file containing a `scanner! {{ ... }}` invocation.\nIf your file contains only scanner payload (e.g. `MyScanner {{ ... }}`), wrap it in `scanner! {{ ... }}` first.",
            input_path.display()
        )
    })?;
    let scanner_macro_tokens = extract_scanner_macro_tokens(&syntax, macro_index)?;

    let scanner_data: ScannerData = syn::parse2(scanner_macro_tokens)
        .map_err(|e| anyhow!("Failed to parse scanner! macro payload: {e}"))?;

    let scanner_name = scanner_data.name.clone();
    let token_names_by_id = extract_token_names_from_input(&input);
    let scanner_modes = scanner_data
        .build_scanner_modes()
        .map_err(|e| anyhow!("Failed to build scanner modes: {e}"))?;

    let (dfas, character_classes) = build_dfas_and_classes(&scanner_modes)?;

    Ok(BuildArtifacts {
        scanner_name,
        scanner_modes,
        dfas,
        character_classes,
        token_names_by_id,
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
                    .map(|a| {
                        let id = a.terminal_type.to_string();
                        artifacts.token_names_by_id.get(&id).cloned().unwrap_or(id)
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                label.push('\n');
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
                let label = labels.join("|");
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
                    let token_id = token.to_string();
                    let token_label = artifacts
                        .token_names_by_id
                        .get(&token_id)
                        .cloned()
                        .unwrap_or(token_id);
                    let from_states =
                        accepting_states_for_token_id(&artifacts.dfas[mode_idx], *token);
                    let to = mode_start_anchor(artifacts, *target_mode_idx);
                    if from_states.is_empty() {
                        out.push_str(&format!(
                            "  mode_{mode_idx} -> {to} [style=dashed, color=\"#1d6f42\", label=\"on {}: enter\"];\n",
                            escape_dot(&token_label)
                        ));
                    } else {
                        for state_idx in from_states {
                            let from = dfa_node_name(mode_idx, state_idx);
                            out.push_str(&format!(
                                "  {from} -> {to} [style=dashed, color=\"#1d6f42\", label=\"on {}: enter\"];\n",
                                escape_dot(&token_label)
                            ));
                        }
                    }
                }
                TransitionToNumericMode::PushMode(token, target_mode_idx) => {
                    let token_id = token.to_string();
                    let token_label = artifacts
                        .token_names_by_id
                        .get(&token_id)
                        .cloned()
                        .unwrap_or(token_id);
                    let from_states =
                        accepting_states_for_token_id(&artifacts.dfas[mode_idx], *token);
                    let to = mode_start_anchor(artifacts, *target_mode_idx);
                    if from_states.is_empty() {
                        out.push_str(&format!(
                            "  mode_{mode_idx} -> {to} [style=dashed, color=\"#b06a00\", label=\"on {}: push\"];\n",
                            escape_dot(&token_label)
                        ));
                    } else {
                        for state_idx in from_states {
                            let from = dfa_node_name(mode_idx, state_idx);
                            out.push_str(&format!(
                                "  {from} -> {to} [style=dashed, color=\"#b06a00\", label=\"on {}: push\"];\n",
                                escape_dot(&token_label)
                            ));
                        }
                    }
                }
                TransitionToNumericMode::PopMode(token) => {
                    let token_id = token.to_string();
                    let token_label = artifacts
                        .token_names_by_id
                        .get(&token_id)
                        .cloned()
                        .unwrap_or(token_id);
                    let from_states =
                        accepting_states_for_token_id(&artifacts.dfas[mode_idx], *token);
                    if from_states.is_empty() {
                        out.push_str(&format!(
                            "  mode_{mode_idx} -> mode_stack [style=dashed, color=\"#7a2f6d\", label=\"on {}: pop\"];\n",
                            escape_dot(&token_label)
                        ));
                    } else {
                        for state_idx in from_states {
                            let from = dfa_node_name(mode_idx, state_idx);
                            out.push_str(&format!(
                                "  {from} -> mode_stack [style=dashed, color=\"#7a2f6d\", label=\"on {}: pop\"];\n",
                                escape_dot(&token_label)
                            ));
                        }
                    }
                }
            }
        }
    }

    out.push_str("}\n");
    Ok(out)
}

fn accepting_states_for_token_id(dfa: &Dfa, token: usize) -> Vec<usize> {
    dfa.states
        .iter()
        .enumerate()
        .filter_map(|(state_idx, state)| {
            state
                .accept_data
                .iter()
                .any(|accept| accept.terminal_type.as_usize() == token)
                .then_some(state_idx)
        })
        .collect()
}

fn mode_start_anchor(artifacts: &BuildArtifacts, mode_idx: usize) -> String {
    if artifacts
        .dfas
        .get(mode_idx)
        .map(|dfa| !dfa.states.is_empty())
        .unwrap_or(false)
    {
        dfa_node_name(mode_idx, 0)
    } else {
        format!("mode_{mode_idx}")
    }
}

fn group_transitions_by_target(
    state: &scnr2_generate::dfa::DfaState,
    character_classes: &CharacterClasses,
) -> Result<BTreeMap<usize, Vec<String>>> {
    let mut grouped: BTreeMap<usize, Vec<String>> = BTreeMap::new();

    for transition in &state.transitions {
        let target = transition.target.as_usize();
        let class_idx = transition.elementary_interval_index.as_usize();
        character_classes
            .intervals
            .get(class_idx)
            .ok_or_else(|| anyhow!("Invalid disjoint class index {class_idx} in DFA transition"))?;
        let rendered = class_idx.to_string();
        grouped.entry(target).or_default().push(rendered);
    }

    for labels in grouped.values_mut() {
        labels.sort();
    }

    Ok(grouped)
}

fn extract_token_names_from_input(input: &str) -> BTreeMap<String, String> {
    let mut names = BTreeMap::new();

    for raw_line in input.lines() {
        let line = raw_line.trim();
        if !line.starts_with("token ") {
            continue;
        }

        let Some((lhs, rhs)) = line.split_once("=>") else {
            continue;
        };
        let _ = lhs;

        let Some((terminal_part, comment_part)) = rhs.split_once(";") else {
            continue;
        };
        let Some((_, comment_text)) = comment_part.split_once("//") else {
            continue;
        };

        let terminal_id = terminal_part.trim();
        if terminal_id.is_empty() {
            continue;
        }

        let mut token_name = comment_text.trim();
        if token_name.starts_with('"') && token_name.ends_with('"') && token_name.len() >= 2 {
            token_name = &token_name[1..token_name.len() - 1];
        }
        if token_name.is_empty() {
            continue;
        }

        names.insert(terminal_id.to_string(), token_name.to_string());
    }

    names
}

fn render_disjunct_classes_json(character_classes: &CharacterClasses) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"format\": \"scnr2_disjunct_character_classes_v1\",\n");
    out.push_str("  \"classes\": [\n");

    for (class_id, intervals) in character_classes.intervals.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"id\": {},\n", class_id));
        out.push_str("      \"ranges\": [\n");

        for (idx, range) in intervals.iter().enumerate() {
            let start = *range.start();
            let end = *range.end();
            out.push_str(&format!(
                "        {{\"start\": \"{}\", \"end\": \"{}\", \"start_u32\": {}, \"end_u32\": {}}}",
                escape_json_char(start),
                escape_json_char(end),
                start as u32,
                end as u32
            ));
            if idx + 1 != intervals.len() {
                out.push(',');
            }
            out.push('\n');
        }

        out.push_str("      ]\n");
        out.push_str("    }");
        if class_id + 1 != character_classes.intervals.len() {
            out.push(',');
        }
        out.push('\n');
    }

    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

fn disjunct_classes_output_path(dot_output_path: &Path) -> PathBuf {
    let mut path = dot_output_path.to_path_buf();
    path.set_extension("classes.json");
    path
}

fn escape_json_char(c: char) -> String {
    match c {
        '"' => "\\\"".to_string(),
        '\\' => "\\\\".to_string(),
        '\u{08}' => "\\b".to_string(),
        '\u{0C}' => "\\f".to_string(),
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        ch if ch.is_control() => format!("\\u{:04x}", ch as u32),
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
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory {}", parent.display()))?;
    }
    fs::write(path, dot).with_context(|| format!("Failed to write DOT file {}", path.display()))?;
    Ok(())
}

fn write_svg_output(dot_path: &Path, svg_path: &Path) -> Result<()> {
    if let Some(parent) = svg_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).with_context(|| {
            format!("Failed to create SVG output directory {}", parent.display())
        })?;
    }

    let status = Command::new("dot")
        .arg("-Tsvg")
        .arg(dot_path)
        .arg("-o")
        .arg(svg_path)
        .status()
        .with_context(|| {
            "Failed to execute Graphviz `dot`. Ensure Graphviz is installed and `dot` is in PATH."
                .to_string()
        })?;

    if !status.success() {
        bail!(
            "Graphviz `dot` failed while generating SVG (exit status: {})",
            status
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_dot() {
        assert_eq!(escape_dot("a\\b\"c"), "a\\\\b\\\"c");
    }

    #[test]
    fn test_disjunct_classes_output_path() {
        let path = disjunct_classes_output_path(Path::new("output.dot"));
        assert_eq!(path, PathBuf::from("output.classes.json"));
    }

    #[test]
    fn test_escape_json_char() {
        assert_eq!(escape_json_char('"'), "\\\"");
        assert_eq!(escape_json_char('\\'), "\\\\");
        assert_eq!(escape_json_char('\n'), "\\n");
        assert_eq!(escape_json_char('x'), "x");
    }

    #[test]
    fn test_extract_token_names_from_input() {
        let input = r#"
            token r"\\n" => 1; // "Newline"
            token r"[0-9]+" => 5; // "NUMBER"
            token r"\\+" => 6;
        "#;

        let names = extract_token_names_from_input(input);
        assert_eq!(names.get("1"), Some(&"Newline".to_string()));
        assert_eq!(names.get("5"), Some(&"NUMBER".to_string()));
        assert_eq!(names.get("6"), None);
    }
}
