#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics/plot_error_histograms.py
========================================================================================
Modulo Python per l'analisi e la visualizzazione della tassonomia degli errori SPARQL.

Utilizzabile esclusivamente come MODULO IMPORTABILE in Python, con dichiarazione
esplicita dei target files da parte dell'utente:

    from metrics.plot_error_histograms import plot_error_histograms

    target_files = {
        "CoT v13 (14B)": "cot13_pp3.json",
        "FS v07 (Phi-4)": "fs7_phi4.json",
    }

    results = plot_error_histograms(
        target_files=target_files,
        output_dir="metrics"
    )

Caratteristiche:
  - Nessuna esecuzione automatica/hardcoded: richiede la definizione esplicita dei file target.
  - Estrazione rigorosa delle sole 14 categorie valide dell'ontologia dell'errore
    (nessun fallback euristico testuale né alias).
  - Generazione di grafici a barre aggregati ad alta risoluzione (300 DPI) per le categorie
    sintattiche, logiche, ontologiche, macro totali e pannello combinato.
========================================================================================
"""

import os
import re
import json
import tempfile
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Optional, Union
import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

# Optional plotting and analysis imports
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import pandas as pd
except ImportError:
    pd = None

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent  # Cartella text2SPARQL (dove ci sono i .json)
METRICS_DIR = Path(__file__).resolve().parent       # Cartella metrics (dove salva i grafici .png)

# Palette colori per i grafici
DEFAULT_PALETTE = [
    "#1D3557",  # Deep Navy
    "#E63946",  # Coral Red
    "#2A9D8F",  # Teal Green
    "#7B2CBF",  # Royal Purple
    "#F4A261",  # Amber / Orange
    "#457B9D",  # Steel Blue
    "#E76F51",  # Burnt Coral
    "#264653",  # Charcoal Green
    "#9B5DE5",  # Lavender
    "#00BBF9",  # Cyan Blue
]

METHOD_COLOR_MAP = {
    "Zero-Shot": "#E63946",
    "One-Shot":  "#F4A261",
    "Few-Shots": "#2A9D8F",
    "CoT v07":   "#457B9D",
    "CoT v12":   "#1D3557",
    "CoT v13":   "#7B2CBF",
    "ZS v04 (14B)": "#E63946",
    "FS v06 (14B)": "#2A9D8F",
    "CoT v13 (14B)": "#7B2CBF"
}

# 1. Official Taxonomy Descriptions (from error_analysis.xlsx)
TAXONOMY_DESCRIPTIONS = {
    # Syntax errors
    "syn_sql": "Uso di sintassi o costrutti SQL in una query SPARQL.",
    "syn_subq": "Subquery o funzione aggregata collocata dentro una clausola FILTER o BIND.",
    "syn_agg": "Parentesi mancanti o alias assente nelle proiezioni di funzioni aggregate (AVG, MIN, MAX, COUNT).",
    "syn": "Errore sintattico generico: struttura o clausole SPARQL non valide, oppure punteggiatura mancante.",
    "tot_syn": "Totale degli errori sintattici: somma aritmetica di syn_sql, syn_subq, syn_agg e syn.",

    # Logical errors
    "log_var": "Problemi di dichiarazione o differenziazione delle variabili, ad esempio variabili non dichiarate o ambigue.",
    "log_flip": "Inversione del soggetto e dell'oggetto in una tripla.",
    "log_hall": "Allucinazione di URI o prefissi CURIE, oppure contaminazione da esempi (shot leakage).",
    "log_curie": "CURIE incompleto che porta a una risoluzione errata dell'URI durante l'espansione del prefisso.",
    "log_idk": "Conoscenza insufficiente o mancato recupero di informazioni presenti nel grafo; include query rispondibili omesse.",
    "log": "Errore logico generico non ricondotto a una sottocategoria specifica, ad esempio selezione del ramo errato o logica della query non corrispondente alla domanda.",
    "tot_log": "Totale degli errori logici: somma aritmetica di log_var, log_flip, log_hall, log_curie, log_idk e log.",

    # Ontology errors
    "onto_ref": "Errore ontologico nella scelta tra proprietà di riferimento (ref_*) e proprietà identificative (*_id).",
    "onto_year": "Errore ontologico nel trattamento del tipo o del formato dell'anno, ad esempio come numero invece che stringa.",
    "onto": "Errore ontologico generico: interpretazione errata del modello o delle proprietà dell'ontologia.",
    "tot_onto": "Totale degli errori ontologici: somma aritmetica di onto_ref, onto_year e onto."
}

# --- Category Lists ---
SYNTAX_CATEGORIES = [
    "syn_sql",
    "syn_subq",
    "syn_agg",
    "syn"
]

SYNTAX_LABELS = {
    "syn_sql": "SQL Syntax/Constructs\n(syn_sql)",
    "syn_subq": "Subquery/Agg in Filter/Bind\n(syn_subq)",
    "syn_agg": "Agg Parentheses/Alias\n(syn_agg)",
    "syn": "Generic Syntax/Clause\n(syn)"
}

LOGICAL_CATEGORIES = [
    "log_var",
    "log_flip",
    "log_hall",
    "log_curie",
    "log_idk",
    "log"
]

LOGICAL_LABELS = {
    "log_var": "Var Decl / Ambiguity\n(log_var)",
    "log_flip": "Triple Subj-Obj Flip\n(log_flip)",
    "log_hall": "URI/CURIE Hallucination\n(log_hall)",
    "log_curie": "Incomplete CURIE\n(log_curie)",
    "log_idk": "Insufficient Knowledge\n(log_idk)",
    "log": "Generic Logic Error\n(log)"
}

ONTOLOGY_CATEGORIES = [
    "onto_ref",
    "onto_year",
    "onto"
]

ONTOLOGY_LABELS = {
    "onto_ref": "Ref Property vs ID\n(onto_ref)",
    "onto_year": "Year Datatype/Format\n(onto_year)",
    "onto": "Generic Ontology Error\n(onto)"
}

TOTAL_CATEGORIES = [
    "tot_syn",
    "tot_log",
    "tot_onto"
]

TOTAL_LABELS = {
    "tot_syn": "Total Syntax Errors\n(tot_syn)",
    "tot_log": "Total Logical Errors\n(tot_log)",
    "tot_onto": "Total Ontology Errors\n(tot_onto)"
}

VALID_LABELS = set(SYNTAX_CATEGORIES + LOGICAL_CATEGORIES + ONTOLOGY_CATEGORIES)


def classify_record(comment: str, record: dict = None, method_name: str = "") -> list:
    """
    Estrae unicamente le label valide della tassonomia esplicitamente presenti nel commento.
    Tutti i fallback euristici basati su testo/lingua naturale e gli alias sono rimossi.
    """
    c = (comment or "").strip()
    if not c:
        return []

    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", c.lower())
    cats = [t for t in tokens if t in VALID_LABELS]
    return list(dict.fromkeys(cats))


def collect_all_data(target_files: Dict[str, Union[str, Path]]) -> Dict[str, Counter]:
    """
    Esegue la scansione e l'aggregazione statistica dei record per i target_files dichiarati.
    """
    if not target_files:
        raise ValueError("È necessario specificare un dizionario non vuoto di target_files.")

    all_categories = SYNTAX_CATEGORIES + LOGICAL_CATEGORIES + ONTOLOGY_CATEGORIES
    data_by_method = {}
    
    for method_name, file_path in target_files.items():
        p = Path(file_path)
        if not p.is_absolute() and not p.exists():
            candidate = BASE_DIR / p
            if candidate.exists():
                p = candidate

                
        if not p.exists():
            raise FileNotFoundError(f"File target non trovato per '{method_name}': {p}")
            
        with open(p, "r", encoding="utf-8") as f:
            records = json.load(f)
            
        counts = Counter({cat: 0 for cat in all_categories})
        records_detail = []
        erroneous_count = 0
        
        for idx, rec in enumerate(records):
            comm = rec.get("comment", "")
            cats = classify_record(comm, rec, method_name)
            for c in cats:
                if c in counts:
                    counts[c] += 1
            if len(cats) > 0:
                erroneous_count += 1
            records_detail.append({
                "idx": idx,
                "id": rec.get("id", rec.get("question_index", idx)),
                "question_index": rec.get("question_index", idx),
                "comment": comm,
                "errors": cats
            })
                    
        # Calcolo totali macro
        counts["tot_syn"] = sum(counts[c] for c in SYNTAX_CATEGORIES)
        counts["tot_log"] = sum(counts[c] for c in LOGICAL_CATEGORIES)
        counts["tot_onto"] = sum(counts[c] for c in ONTOLOGY_CATEGORIES)
        counts["tot_all"] = counts["tot_syn"] + counts["tot_log"] + counts["tot_onto"]

        counts["total_queries"] = len(records)
        counts["error_queries"] = erroneous_count
        counts["correct_queries"] = len(records) - erroneous_count
        counts["error_rate"] = (erroneous_count / len(records)) if len(records) > 0 else 0.0
        counts["accuracy"] = ((len(records) - erroneous_count) / len(records)) if len(records) > 0 else 0.0
        counts["records_detail"] = records_detail
        
        data_by_method[method_name] = counts
    return data_by_method


def draw_grouped_bar_chart(ax, categories, labels_dict, data_by_method, title, color_map, ylabel="Error Count"):
    """
    Disegna un grafico a barre raggruppate per le categorie fornite.
    """
    if plt is None:
        raise ImportError("matplotlib è richiesto per le funzioni di plotting.")
    methods = list(data_by_method.keys())
    if not methods:
        return
    n_cats = len(categories)
    n_methods = len(methods)
    
    x = np.arange(n_cats)
    total_width = 0.76
    bar_width = total_width / max(1, n_methods)
    
    for i, m in enumerate(methods):
        y_vals = [data_by_method[m][cat] for cat in categories]
        offset = (i - n_methods / 2 + 0.5) * bar_width
        col = color_map.get(m, DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)])
        rects = ax.bar(
            x + offset,
            y_vals,
            width=bar_width * 0.92,
            label=m,
            color=col,
            edgecolor="white",
            linewidth=1.2,
            alpha=0.92
        )
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{int(height)}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#222222"
                )
                
    ax.set_xticks(x)
    ax.set_xticklabels([labels_dict[c] for c in categories], fontsize=9.5, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    all_vals = [data_by_method[m][c] for m in methods for c in categories]
    max_val = max(all_vals) if all_vals else 0
    ax.set_ylim(0, max_val + max(2, max_val * 0.2))


def plot_error_histograms(
    target_files: Dict[str, Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    output_prefix: str = "",
    save_plots: bool = True,
    color_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Funzione principale per l'analisi modulare della tassonomia degli errori SPARQL.
    
    Parametri:
      target_files: Dizionario obbligatorio avente per chiave il nome del metodo/modello
                    e per valore il percorso al file JSON (es. {"FS7 Phi-4": "fs7_phi4.json"}).
      output_dir: Directory in cui salvare i grafici (default: 'metrics/').
      output_prefix: Prefisso opzionale da anteporre ai nomi dei file salvati.
      save_plots: Se True, genera e salva le immagini PNG degli istogrammi a 300 DPI.
      color_map: Mappatura facoltativa dei colori personalizzati per ciascun metodo.

    Restituisce:
      Dizionario contenente:
        - "data_by_method": conteggi, distribuzioni e record dettagliati per metodo
        - "saved_plots": lista dei percorsi dei file grafici PNG salvati
    """
    if not target_files:
        raise ValueError("È necessario dichiarare esplicitamente un dizionario non vuoto per 'target_files'.")

    out_dir = Path(output_dir) if output_dir is not None else METRICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Calcolo ed estrazione errori
    data_by_method = collect_all_data(target_files)
    
    # Configurazione colori
    colors = dict(METHOD_COLOR_MAP)
    if color_map:
        colors.update(color_map)
    for i, m in enumerate(target_files.keys()):
        if m not in colors:
            colors[m] = DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]

    saved_plots = []

    # 2. Generazione grafici a barre
    if save_plots and plt is not None:
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        pfx = f"{output_prefix}_" if output_prefix else ""
        
        # 2.1 Syntax Errors Histogram
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        draw_grouped_bar_chart(
            ax,
            SYNTAX_CATEGORIES,
            SYNTAX_LABELS,
            data_by_method,
            "Syntax Errors by Prompting Strategy (Spider4SPARQL)",
            colors
        )
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5, loc="upper right")
        plt.tight_layout()
        syntax_path = out_dir / f"{pfx}syntax_errors_histogram.png"
        plt.savefig(syntax_path, dpi=300)
        plt.close()
        saved_plots.append(syntax_path)
        print(f"[OK] Saved {syntax_path}")
        
        # 2.2 Logical Errors Histogram
        fig, ax = plt.subplots(figsize=(12.5, 6), dpi=300)
        draw_grouped_bar_chart(
            ax,
            LOGICAL_CATEGORIES,
            LOGICAL_LABELS,
            data_by_method,
            "Logical & Reasoning Errors by Prompting Strategy (Spider4SPARQL)",
            colors
        )
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5, loc="upper right")
        plt.tight_layout()
        logical_path = out_dir / f"{pfx}logical_errors_histogram.png"
        plt.savefig(logical_path, dpi=300)
        plt.close()
        saved_plots.append(logical_path)
        print(f"[OK] Saved {logical_path}")
        
        # 2.3 Ontology Errors Histogram
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        draw_grouped_bar_chart(
            ax,
            ONTOLOGY_CATEGORIES,
            ONTOLOGY_LABELS,
            data_by_method,
            "Ontology & Schema Modeling Errors by Prompting Strategy (Spider4SPARQL)",
            colors
        )
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5, loc="upper right")
        plt.tight_layout()
        ontology_path = out_dir / f"{pfx}ontology_errors_histogram.png"
        plt.savefig(ontology_path, dpi=300)
        plt.close()
        saved_plots.append(ontology_path)
        print(f"[OK] Saved {ontology_path}")
        
        # 2.4 Macro Totals Histogram
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        draw_grouped_bar_chart(
            ax,
            TOTAL_CATEGORIES,
            TOTAL_LABELS,
            data_by_method,
            "Macro Error Totals by Prompting Strategy (Spider4SPARQL)",
            colors
        )
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5, loc="upper right")
        plt.tight_layout()
        totals_path = out_dir / f"{pfx}macro_totals_histogram.png"
        plt.savefig(totals_path, dpi=300)
        plt.close()
        saved_plots.append(totals_path)
        print(f"[OK] Saved {totals_path}")

        # 2.5 Combined 3-Panel Figure
        fig, axes = plt.subplots(3, 1, figsize=(13, 15), dpi=300)
        draw_grouped_bar_chart(
            axes[0],
            SYNTAX_CATEGORIES,
            SYNTAX_LABELS,
            data_by_method,
            "1. Syntax Errors (syn_sql, syn_subq, syn_agg, syn)",
            colors
        )
        axes[0].legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")
        
        draw_grouped_bar_chart(
            axes[1],
            LOGICAL_CATEGORIES,
            LOGICAL_LABELS,
            data_by_method,
            "2. Logical & Reasoning Errors (log_var, log_flip, log_hall, log_curie, log_idk, log)",
            colors
        )
        axes[1].legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")
        
        draw_grouped_bar_chart(
            axes[2],
            ONTOLOGY_CATEGORIES,
            ONTOLOGY_LABELS,
            data_by_method,
            "3. Ontology & Schema Errors (onto_ref, onto_year, onto)",
            colors
        )
        axes[2].legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper right")
        
        fig.suptitle("SPARQL Generation Error Taxonomy: Syntax vs Logical vs Ontology Errors", fontsize=15, fontweight="bold", y=0.995)
        plt.tight_layout()
        combined_path = out_dir / f"{pfx}error_taxonomy_histograms_combined.png"
        plt.savefig(combined_path, dpi=300)
        plt.close()
        saved_plots.append(combined_path)
        print(f"[OK] Saved {combined_path}")

    return {
        "data_by_method": data_by_method,
        "saved_plots": saved_plots
    }


__all__ = [
    "plot_error_histograms",
    "collect_all_data",
    "classify_record",
    "draw_grouped_bar_chart",
    "VALID_LABELS",
    "SYNTAX_CATEGORIES",
    "LOGICAL_CATEGORIES",
    "ONTOLOGY_CATEGORIES",
    "TOTAL_CATEGORIES",
    "SYNTAX_LABELS",
    "LOGICAL_LABELS",
    "ONTOLOGY_LABELS",
    "TOTAL_LABELS",
    "TAXONOMY_DESCRIPTIONS",
]
if __name__ == "__main__":
    target_files = {
        "CoT12 Qwen14b": "cot12_pp3.json",
        "CoT13 Qwen14B": "cot13_pp3_qwen14b_v2.json",
    }
    plot_error_histograms(target_files)