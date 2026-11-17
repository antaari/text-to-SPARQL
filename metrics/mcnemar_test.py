#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics/mcnemar_test.py
========================================================================================
Calcolo del Test Esatto di McNemar (Exact Binomial Test a due code) per valutare
la significatività statistica delle differenze tra coppie di esecuzioni (Modello - Strategia).

Utilizza direttamente il dizionario 'breakdown_by_index' di 'method_comparison.json':
  - correct_indices: query corrette (Exact Match / Accuracy)
  - mismatch_indices: query con result mismatch (valide ma con tuple errate)
  - error_indices: query con execution error (errori di sintassi/runtime)

Metriche analizzate per ciascuna coppia appaiata:
  1. GENERALE:
     - correct_queries: confronto su correct_indices (status == CORRECT).
  2. AL DETTAGLIO:
     - execution_success: confronto su (correct_indices | mismatch_indices).
     - result_mismatch: confronto su mismatch_indices.

Nessun ricalcolo dei risultati o scansione di file singoli: sfrutta direttamente i set
di indici pre-calcolati e consolidati in method_comparison.json.

Eseguibile come modulo Python e direttamente come script per confrontare esclusivamente
le coppie di metodi specificate (nessuna modalità CLI con argparse).
========================================================================================
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple, Union, Sequence

# Risoluzione percorsi: cerca method_comparison.json sia nella cartella padre (scratches) sia in metrics/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_COMPARISON_FILE = PROJECT_ROOT / "method_comparison.json"
if not DEFAULT_COMPARISON_FILE.exists():
    fallback_file = SCRIPT_DIR / "method_comparison.json"
    if fallback_file.exists():
        DEFAULT_COMPARISON_FILE = fallback_file


# ==============================================================================
# CONFIGURAZIONE COPPIE PREDEFINITE DA CONFRONTARE
# ==============================================================================
# Inserire qui le coppie di metodi da confrontare quando lo script viene eseguito direttamente.
DEFAULT_PAIRS: List[Tuple[str, str]] = [
    ("zs4_pp12", "cot13_pp3"),
    ("zs4_pp12", "fs6_pp3"),
    ("fs6_pp3", "cot13_pp3"),
]


# ==============================================================================
# 1. FUNZIONE STATISTICA: TEST ESATTO DI MCNEMAR
# ==============================================================================

def compute_mcnemar_from_counts(
    a: int, b: int, c: int, d: int
) -> Dict[str, Any]:
    """
    Calcola il test esatto di McNemar (Exact Binomial a due code) a partire
    dalle celle della tabella di contingenza 2x2.

    Tabella 2x2:
                    Metodo 2 = 1    Metodo 2 = 0
    Metodo 1 = 1         a               b   (Metodo 1 vince)
    Metodo 1 = 0         c               d
                    (Metodo 2 vince)
    
    Ipotesi nulla H0: p_b == p_c (b ~ Binomial(n = b + c, p = 0.5))
    """
    n_pairs = a + b + c + d
    if n_pairs == 0:
        return {
            "n_pairs": 0, "a": 0, "b": 0, "c": 0, "d": 0,
            "rate1": 0.0, "rate2": 0.0, "diff_rate": 0.0,
            "discordant_total": 0, "odds_ratio": 1.0,
            "p_value": 1.0, "significance": "n.s."
        }

    rate1 = (a + b) / n_pairs
    rate2 = (a + c) / n_pairs
    diff_rate = rate1 - rate2
    discordant_total = b + c

    if c == 0:
        odds_ratio = float("inf") if b > 0 else 1.0
    else:
        odds_ratio = b / c

    if discordant_total == 0:
        p_value = 1.0
    else:
        k = min(b, c)
        prob_tail = sum(math.comb(discordant_total, i) for i in range(k + 1)) * (0.5 ** discordant_total)
        p_value = min(1.0, 2.0 * prob_tail)

    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "n.s."

    return {
        "n_pairs": n_pairs,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "rate1": rate1,
        "rate2": rate2,
        "diff_rate": diff_rate,
        "discordant_total": discordant_total,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
        "significance": sig
    }


# ==============================================================================
# 2. STRUTTURA METODO BASATA SU BREAKDOWN_BY_INDEX
# ==============================================================================

class MethodBreakdown:
    """Rappresenta una configurazione Modello-Strategia caricata da breakdown_by_index."""

    def __init__(
        self,
        name: str,
        model: str,
        strategy: str,
        correct_indices: Set[int],
        mismatch_indices: Set[int],
        error_indices: Set[int],
        evaluated_indices: Optional[Set[int]] = None
    ):
        self.name = name
        self.model = model or "unknown"
        self.strategy = strategy or name

        self.correct_set = set(correct_indices)
        self.mismatch_set = set(mismatch_indices)
        self.error_set = set(error_indices)

        # execution_success: unione di corrette e mismatch (tutte le query eseguibili)
        self.exec_success_set = self.correct_set | self.mismatch_set

        # Tutte le query valutate per questo metodo
        if evaluated_indices:
            self.evaluated_set = set(evaluated_indices) | self.correct_set | self.mismatch_set | self.error_set
        else:
            self.evaluated_set = self.correct_set | self.mismatch_set | self.error_set

    @property
    def total_queries(self) -> int:
        return len(self.evaluated_set)

    @property
    def correct_count(self) -> int:
        return len(self.correct_set)

    @property
    def exec_success_count(self) -> int:
        return len(self.exec_success_set)

    @property
    def mismatch_count(self) -> int:
        return len(self.mismatch_set)

    @property
    def error_count(self) -> int:
        return len(self.error_set)


def load_methods_from_comparison(file_path: Union[str, Path] = DEFAULT_COMPARISON_FILE) -> Dict[str, MethodBreakdown]:
    """Carica tutti i metodi leggendo direttamente breakdown_by_index in method_comparison.json."""
    path = Path(file_path)
    if not path.exists():
        alt_path = PROJECT_ROOT / file_path
        if alt_path.exists():
            path = alt_path
        else:
            print(f"[ERRORE] File '{file_path}' non trovato.")
            return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    methods = {}
    for m_key, m_val in data.get("methods", {}).items():
        cfg = m_val.get("configuration", {})
        model = cfg.get("model", "")
        strategy = cfg.get("generation_strategy", m_key)
        eval_list = cfg.get("evaluated_indices", [])

        bb = m_val.get("breakdown_by_index", {})
        corr_list = bb.get("correct_indices", [])
        mis_list = bb.get("mismatch_indices", [])
        err_list = bb.get("error_indices", [])

        # Se breakdown_by_index non fosse presente, fallback su detailed_results
        if not bb and "detailed_results" in m_val:
            for r in m_val["detailed_results"]:
                q_idx = r.get("question_index", r.get("id"))
                if q_idx is None:
                    continue
                st = r.get("status")
                if st == "CORRECT" or r.get("is_correct"):
                    corr_list.append(q_idx)
                elif st == "RESULT_MISMATCH":
                    mis_list.append(q_idx)
                else:
                    err_list.append(q_idx)

        methods[m_key] = MethodBreakdown(
            name=m_key,
            model=model,
            strategy=strategy,
            correct_indices=set(corr_list),
            mismatch_indices=set(mis_list),
            error_indices=set(err_list),
            evaluated_indices=set(eval_list) if eval_list else None
        )

    return methods


def list_available_methods(
    file_path: Union[str, Path] = DEFAULT_COMPARISON_FILE,
    print_list: bool = True
) -> Dict[str, MethodBreakdown]:
    """Elenca tutti i metodi disponibili nel file di confronto."""
    methods = load_methods_from_comparison(file_path)
    if print_list and methods:
        p_name = Path(file_path).name
        print(f"\nMetodi disponibili in '{p_name}':")
        print("-" * 88)
        for k, v in methods.items():
            print(f" - {k:<30} | Model: {v.model:<22} | Strategy: {v.strategy:<18} | "
                  f"Tot: {v.total_queries:<2} (C:{v.correct_count:<2} M:{v.mismatch_count:<2} E:{v.error_count:<2})")
        print("-" * 88 + "\n")
    return methods


# ==============================================================================
# 3. CONFRONTO APPAIATO MEDIANTE OPERAZIONI INSIEMISTICHE
# ==============================================================================

def compare_two_methods(
    m1: MethodBreakdown,
    m2: MethodBreakdown
) -> Dict[str, Any]:
    """
    Esegue il test di McNemar tra due metodi confrontando i set di indici
    sulle query appaiate (intersezione degli indici valutati).
    """
    common_indices = m1.evaluated_set & m2.evaluated_set
    n_common = len(common_indices)

    if n_common == 0:
        return {
            "method1": m1.name,
            "method2": m2.name,
            "error": "Nessuna query in comune tra le due esecuzioni."
        }

    def evaluate_metric(set1: Set[int], set2: Set[int], name: str, level: str):
        # Restringiamo ai soli indici in comune
        s1 = set1 & common_indices
        s2 = set2 & common_indices

        # Celle della tabella 2x2
        a = len(s1 & s2)            # Entrambi 1
        b = len(s1 - s2)            # Solo M1 = 1 (M1 vince)
        c = len(s2 - s1)            # Solo M2 = 1 (M2 vince)
        d = n_common - (a + b + c)  # Entrambi 0

        stats = compute_mcnemar_from_counts(a, b, c, d)
        disc_m1_wins = sorted(list(s1 - s2))
        disc_m2_wins = sorted(list(s2 - s1))

        return {
            "name": name,
            "level": level,
            "stats": stats,
            "discordant_m1_wins": disc_m1_wins,
            "discordant_m2_wins": disc_m2_wins
        }

    # 1. GENERALE: correct_queries (Accuracy / Exact Match)
    metric_correct = evaluate_metric(
        m1.correct_set, m2.correct_set,
        name="Correct Queries (Accuracy / Exact Match)",
        level="GENERALE"
    )

    # 2. DETTAGLIO: execution_success (Validità sintattica ed esecutiva)
    metric_exec = evaluate_metric(
        m1.exec_success_set, m2.exec_success_set,
        name="Execution Success (Syntactic & Runtime Validity)",
        level="DETTAGLIO"
    )

    # 3. DETTAGLIO: result_mismatch (Query eseguite ma con tuple errate)
    metric_mismatch = evaluate_metric(
        m1.mismatch_set, m2.mismatch_set,
        name="Result Mismatch (Valid Execution with Mismatched Tuples)",
        level="DETTAGLIO"
    )

    return {
        "method1": {
            "name": m1.name,
            "model": m1.model,
            "strategy": m1.strategy,
            "total_queries": m1.total_queries
        },
        "method2": {
            "name": m2.name,
            "model": m2.model,
            "strategy": m2.strategy,
            "total_queries": m2.total_queries
        },
        "paired_queries": n_common,
        "metrics": {
            "correct_queries": metric_correct,
            "execution_success": metric_exec,
            "result_mismatch": metric_mismatch
        }
    }


# ==============================================================================
# 4. REPORTISTICA ED ESPORTAZIONE
# ==============================================================================

def print_pair_report(comp: Dict[str, Any], show_discrepancies: bool = False):
    """Stampa a video il report dettagliato per una coppia di metodi."""
    if "error" in comp:
        print(f"[ERRORE] {comp['error']}")
        return

    m1 = comp["method1"]
    m2 = comp["method2"]
    n_common = comp["paired_queries"]

    print("=" * 86)
    print(" TEST ESATTO DI MCNEMAR (DA BREAKDOWN_BY_INDEX): CONFRONTO MODELLO-STRATEGIA")
    print("=" * 86)
    print(f" Metodo 1: {m1['name']} | Modello: {m1['model']} | Strategia: {m1['strategy']}")
    print(f" Metodo 2: {m2['name']} | Modello: {m2['model']} | Strategia: {m2['strategy']}")
    print(f" Query Appaiate (intersezione indici): {n_common}")
    print("-" * 86)

    for metric_key, m_data in comp["metrics"].items():
        st = m_data["stats"]
        level = m_data["level"]
        name = m_data["name"]

        print(f"\n >>> [{level}] {name}")
        print(f"     Tassi: Metodo 1 = {st['rate1']*100:5.2f}% ({st['a']+st['b']}/{n_common}) | "
              f"Metodo 2 = {st['rate2']*100:5.2f}% ({st['a']+st['c']}/{n_common}) | "
              f"Diff = {st['diff_rate']*100:+5.2f}%")

        col1_w = max(10, min(14, len(m1['name'])))
        col2_w = max(10, min(14, len(m2['name'])))
        name1 = m1['name'][:col1_w]
        name2 = m2['name'][:col2_w]

        print("     Tabella di Contingenza 2x2:")
        print(f"                    {name2:>12}=1  {name2:>12}=0")
        print(f"       {name1:>12}=1  {st['a']:>10}  {st['b']:>10}  (solo M1)")
        print(f"       {name1:>12}=0  {st['c']:>10}  {st['d']:>10}")
        print(f"                           (solo M2)")

        b, c = st["b"], st["c"]
        n_disc = st["discordant_total"]
        print(f"     Discordanti totali (b + c): {n_disc} (M1 vince: {b}, M2 vince: {c})")
        print(f"     Odds Ratio (b / c):         {st['odds_ratio']:.3f}")
        print(f"     Exact Binomial p-value:     {st['p_value']:.4f}")
        print(f"     --> Significativita':       {st['significance']}")

        if show_discrepancies:
            w1 = m_data["discordant_m1_wins"]
            w2 = m_data["discordant_m2_wins"]
            print(f"     - Indici dove solo Metodo 1 ha l'evento (n={len(w1)}): {w1}")
            print(f"     - Indici dove solo Metodo 2 ha l'evento (n={len(w2)}): {w2}")

    print("\n" + "=" * 86)
    print(" Legenda: *** p < 0.001 | ** p < 0.01 | * p < 0.05 | n.s. non significativo")
    print("=" * 86 + "\n")


def print_summary_table(comparisons: List[Dict[str, Any]]):
    """Stampa una tabella riassuntiva per confronti a coppie multipli."""
    valid_comps = [c for c in comparisons if "error" not in c]
    if not valid_comps:
        return

    print("\n" + "=" * 115)
    print(" RIASSUNTO TEST ESATTO DI MCNEMAR (DA BREAKDOWN_BY_INDEX)")
    print("=" * 115)
    header = (
        f"{'Coppia Metodi':<35} | "
        f"{'Correct (Acc) M1 vs M2':<22} | {'p (Corr)':<8} {'Sig':<4} | "
        f"{'ExecSuccess M1 vs M2':<20} | {'p (Exec)':<8} {'Sig':<4} | "
        f"{'Mismatch M1 vs M2':<18} | {'p (Mis)':<8} {'Sig':<4}"
    )
    print(header)
    print("-" * 115)

    for c in valid_comps:
        m1 = c["method1"]["name"]
        m2 = c["method2"]["name"]
        pair_str = f"{m1} vs {m2}"
        if len(pair_str) > 34:
            pair_str = pair_str[:31] + "..."

        m_corr = c["metrics"]["correct_queries"]["stats"]
        m_exec = c["metrics"]["execution_success"]["stats"]
        m_mis = c["metrics"]["result_mismatch"]["stats"]

        corr_str = f"{m_corr['rate1']*100:4.1f}% vs {m_corr['rate2']*100:4.1f}%"
        exec_str = f"{m_exec['rate1']*100:4.1f}% vs {m_exec['rate2']*100:4.1f}%"
        mis_str = f"{m_mis['rate1']*100:4.1f}% vs {m_mis['rate2']*100:4.1f}%"

        line = (
            f"{pair_str:<35} | "
            f"{corr_str:<22} | {m_corr['p_value']:<8.4f} {m_corr['significance']:<4} | "
            f"{exec_str:<20} | {m_exec['p_value']:<8.4f} {m_exec['significance']:<4} | "
            f"{mis_str:<18} | {m_mis['p_value']:<8.4f} {m_mis['significance']:<4}"
        )
        print(line)

    print("-" * 115)
    print(" Legenda: *** p < 0.001 | ** p < 0.01 | * p < 0.05 | n.s. non significativo")
    print("=" * 115 + "\n")


def save_results_to_json(results: List[Dict[str, Any]], output_path: Union[str, Path]):
    """Salva la lista dei risultati in formato JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Risultati salvati in JSON: {out_p.resolve()}")


def save_results_to_markdown(results: List[Dict[str, Any]], output_path: Union[str, Path]):
    """Salva un report leggibile dei risultati in formato Markdown."""
    out_md = Path(output_path)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Report Test Esatto di McNemar (da breakdown_by_index)\n\n")
        for c in results:
            if "error" in c:
                continue
            m1 = c["method1"]
            m2 = c["method2"]
            f.write(f"## Confronto: `{m1['name']}` vs `{m2['name']}`\n\n")
            f.write(f"- **Metodo 1**: `{m1['name']}` (Modello: `{m1['model']}`, Strategia: `{m1['strategy']}`)\n")
            f.write(f"- **Metodo 2**: `{m2['name']}` (Modello: `{m2['model']}`, Strategia: `{m2['strategy']}`)\n")
            f.write(f"- **Query Appaiate**: {c['paired_queries']}\n\n")

            f.write("| Metrica | Livello | Tasso M1 | Tasso M2 | Diff | Discordanti (b/c) | Odds Ratio | Exact p-value | Sig |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for m_k, m_d in c["metrics"].items():
                st = m_d["stats"]
                f.write(
                    f"| {m_d['name']} | {m_d['level']} | {st['rate1']*100:.1f}% | {st['rate2']*100:.1f}% | "
                    f"{st['diff_rate']*100:+.1f}% | {st['b']} / {st['c']} | {st['odds_ratio']:.2f} | "
                    f"{st['p_value']:.4f} | **{st['significance']}** |\n"
                )
            f.write("\n---\n\n")
    print(f"[OK] Report salvato in Markdown: {out_md.resolve()}")


# ==============================================================================
# 5. ESECUZIONE COME MODULO PYTHON SU COPPIE SPECIFICATE
# ==============================================================================

def run_mcnemar_test(
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    comparison_file: Union[str, Path] = DEFAULT_COMPARISON_FILE,
    show_discrepancies: bool = False,
    print_details: bool = True,
    print_summary: bool = True,
    output_json: Optional[Union[str, Path]] = None,
    output_md: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Esegue il Test Esatto di McNemar esclusivamente tra coppie di metodi specificate.

    Parametri:
      pairs: Sequenza di tuple contenenti le coppie da confrontare, es. [("zs4_pp12", "cot13_pp3")].
             Se omesso o None, utilizza la lista predefinita 'DEFAULT_PAIRS'.
      comparison_file: Percorso del file 'method_comparison.json'.
      show_discrepancies: Se True, mostra gli indici delle query discordanti.
      print_details: Se True, stampa il report dettagliato per ciascuna coppia.
      print_summary: Se True, stampa la tabella riassuntiva finale.
      output_json: Opzionale percorso per salvare i risultati in formato JSON.
      output_md: Opzionale percorso per salvare i risultati in formato Markdown.

    Ritorna:
      Lista di dizionari con i risultati del test per ciascuna coppia valida.
    """
    target_pairs = list(pairs) if pairs is not None else list(DEFAULT_PAIRS)
    if not target_pairs:
        print("[AVVISO] Nessuna coppia di metodi specificata per il confronto.")
        return []

    methods = load_methods_from_comparison(comparison_file)
    if not methods:
        return []

    results: List[Dict[str, Any]] = []

    for m1_name, m2_name in target_pairs:
        if m1_name not in methods:
            print(f"[ATTENZIONE] Metodo '{m1_name}' non trovato in {Path(comparison_file).name}. Coppia ({m1_name}, {m2_name}) saltata.")
            continue
        if m2_name not in methods:
            print(f"[ATTENZIONE] Metodo '{m2_name}' non trovato in {Path(comparison_file).name}. Coppia ({m1_name}, {m2_name}) saltata.")
            continue

        comp = compare_two_methods(methods[m1_name], methods[m2_name])
        if print_details:
            print_pair_report(comp, show_discrepancies=show_discrepancies)
        results.append(comp)

    if print_summary and len(results) > 0:
        print_summary_table(results)

    if output_json and results:
        save_results_to_json(results, output_json)

    if output_md and results:
        save_results_to_markdown(results, output_md)

    return results


def main():
    """Esecuzione predefinita dello script: confronta esclusivamente le coppie specificate in DEFAULT_PAIRS."""
    print(f"[INFO] Avvio Test di McNemar per le coppie specificate ({len(DEFAULT_PAIRS)} coppie)...")
    run_mcnemar_test(
        pairs=DEFAULT_PAIRS,
        comparison_file=DEFAULT_COMPARISON_FILE,
        show_discrepancies=False,
        print_details=True,
        print_summary=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_PAIRS",
    "DEFAULT_COMPARISON_FILE",
    "run_mcnemar_test",
    "compare_two_methods",
    "compute_mcnemar_from_counts",
    "load_methods_from_comparison",
    "list_available_methods",
    "print_pair_report",
    "print_summary_table",
    "save_results_to_json",
    "save_results_to_markdown",
    "MethodBreakdown",
]
