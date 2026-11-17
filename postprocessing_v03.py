import argparse
import datetime
import json
from collections import Counter
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

try:
    import rdflib
    from rdflib import URIRef, Literal, BNode
except ImportError:
    print("[ERRORE] 'rdflib' non è installato. Esegui: pip install rdflib")
    sys.exit(1)

# Patch di conformità SPARQL 1.1 per RDFLib: unescape di PN_LOCAL_ESC nei local name dei CURIE
try:
    from rdflib.plugins.sparql.sparql import Prologue
    _orig_resolvePName = Prologue.resolvePName

    def _patched_resolvePName(self, prefix: Optional[str], localname: Optional[str]) -> URIRef:
        if localname:
            localname = re.sub(r'\\([_~.\-!$&\'()*+,;=/?#@%])', r'\1', localname)
        return _orig_resolvePName(self, prefix, localname)

    Prologue.resolvePName = _patched_resolvePName
except Exception:
    pass

# Percorsi di riferimento base
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DEV_PATH = ROOT_DIR / "spider4sparql" / "dev.json"


# ==============================================================================
# CONFIGURAZIONE UTENTE (MODIFICA DA QUI PER IL TUO TEST)
# ==============================================================================

# 1. File JSON dei risultati generati (es. "one_shot_v01.json", "zero_shot_v01.json", "cot_v01.json")
JSON_PATH = "one_shot_v01.json"

# 2. ID della risposta da testare:
#    - Imposta un intero specifico (es. 1 oppure 2) per testare solo quella risposta
#    - Imposta una lista (es. [1, 2]) per testare più ID specifici
#    - Imposta None o "all" per testare tutte le risposte presenti nel file JSON
TARGET_GEN_ID: Optional[Union[int, List[int], str]] = 1

# 3. Indice domanda di default in dev.json (usato come fallback se non specificato nel record)
FALLBACK_QUESTION_INDEX = 25

# 4. Grafo completo personalizzato (None per risoluzione automatica basata sul kg_name della domanda)
CUSTOM_FULL_GRAPH_PATH: Optional[str] = None

# 5. Salva/Appendi l'esito dell'esecuzione direttamente nel file JSON dei risultati
UPDATE_JSON_FILE = True


# ==============================================================================
# CARICAMENTO DATI E RISOLUZIONE GRAFO RDF COMPLETO
# ==============================================================================

def load_question_from_dev(
    index: int = FALLBACK_QUESTION_INDEX,
    dev_path: Union[str, Path] = DEFAULT_DEV_PATH
) -> Tuple[str, str, str]:
    """
    Carica la query SPARQL gold, la domanda e il kg_name dal file dev.json all'indice specificato (0-based).
    """
    p = Path(dev_path)
    if not p.is_absolute():
        p = ROOT_DIR / p

    if not p.exists():
        alt = ROOT_DIR / "spider4sparql" / "dev.json"
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"File dev.json non trovato in: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if index < 0 or index >= len(data):
        raise IndexError(f"Indice {index} non valido per dev.json (lunghezza: {len(data)})")

    entry = data[index]
    return entry.get("query", ""), entry.get("question", ""), entry.get("kg_name", "concert_singer")


def resolve_kg_full_graph(
    kg_name: str,
    base_dir: Union[str, Path] = "spider4sparql"
) -> Path:
    """
    Risolve il percorso del file RDF (.ttl o .owl) del grafo completo per il kg_name specificato.
    Se non presente localmente, tenta l'estrazione da materialized_triples_dev.zip.
    """
    base_path = Path(base_dir)
    if not base_path.is_absolute():
        base_path = ROOT_DIR / base_path

    if kg_name.lower() in ("nis2onto", "nis2"):
        for cand in [ROOT_DIR / "NIS2Onto(v2).ttl", ROOT_DIR / "NIS2Onto(v2).owl"]:
            if cand.exists():
                return cand

    ttl_file = base_path / f"{kg_name}.ttl"
    if ttl_file.exists():
        return ttl_file

    zip_file = base_path / "materialized_triples_dev.zip"
    if zip_file.exists():
        with zipfile.ZipFile(zip_file, "r") as z:
            target_inside = f"materialized_triples/{kg_name}.ttl"
            if target_inside in z.namelist():
                z.extract(target_inside, base_path)
                extracted = base_path / "materialized_triples" / f"{kg_name}.ttl"
                if extracted.exists():
                    extracted.rename(ttl_file)
                    try:
                        (base_path / "materialized_triples").rmdir()
                    except Exception:
                        pass
                    return ttl_file

    return ttl_file


# ==============================================================================
# GESTIONE ED ESTRAZIONE PREFISSI DALL'ONTOLOGIA COMPLETA (V03)
# ==============================================================================

# Import diretto da preprocessing_v12
from preprocessing_v12 import extract_ontology_prefixes


def get_ontology_prefixes(kg_name: str, graph: rdflib.Graph) -> Dict[str, str]:
    """
    Recupera i prefissi per il Knowledge Graph specificato:
    1. Tenta prima la lettura istantanea dal file precalcolato '{kg_name}_prefixes.json' (preprocessing v12).
    2. Se il file non esiste, esegue il fallback dinamico su 'extract_ontology_prefixes(graph)' in memoria.
    """
    for cand in [f"{kg_name.lower()}_prefixes.json", f"{kg_name}_prefixes.json"]:
        pfx_file = ROOT_DIR / cand
        if pfx_file.exists():
            try:
                with open(pfx_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data:
                    return {k: str(v).strip() for k, v in data.items()}
            except Exception:
                pass

    return extract_ontology_prefixes(graph)


# Alias per retrocompatibilità
extract_prefixes_from_sources = get_ontology_prefixes


def build_prefix_header(prefixes: Dict[str, str]) -> str:
    """
    Costruisce l'header SPARQL PREFIX ordinato partendo dal dizionario di prefissi.
    """
    if not prefixes:
        return ""

    def prefix_sort_key(item):
        pfx, ns_uri = item
        ns_lower = ns_uri.lower()
        is_std = any(std in ns_lower for std in ["w3.org", "schema.org", "purl.org", "xmlns.com"])
        return (1 if is_std else 0, pfx)

    declarations = []
    for prefix, ns_uri in sorted(prefixes.items(), key=prefix_sort_key):
        pfx_name = str(prefix).strip().rstrip(":")
        if pfx_name:
            declarations.append(f"PREFIX {pfx_name}: <{ns_uri}>")
        elif ns_uri:
            declarations.append(f"PREFIX : <{ns_uri}>")

    return "\n".join(declarations) + ("\n\n" if declarations else "")


def attach_prefixes_to_query(
    query_str: str,
    prefixes: Dict[str, str]
) -> str:
    """
    Aggiunge le dichiarazioni PREFIX canoniche all'inizio della query SPARQL,
    dopo aver rimosso eventuali dichiarazioni PREFIX/BASE preesistenti.
    """
    if not query_str:
        return ""

    text = strip_sparql_prefixes(query_str)
    text = repair_curie_escapes(text)
    if not prefixes:
        return text

    header = build_prefix_header(prefixes).strip()
    if not header:
        return text

    return f"{header}\n\n{text}"


# ==============================================================================
# POSTPROCESSING SPARQL: CLEANING
# ==============================================================================

def strip_sparql_prefixes(query_str: str) -> str:
    """
    Rimuove tutte le dichiarazioni PREFIX e BASE (case-insensitive) dalla query generata,
    restituendo la sola query senza prefissi.
    """
    if not query_str:
        return ""

    lines = query_str.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^(?:PREFIX\s+(?:[a-zA-Z0-9_\-]*:)?\s*<[^>]*>|BASE\s+<[^>]*>)\s*;?\s*$', stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines).strip()

    prefix_inline = re.compile(
        r'^\s*(?:(?:PREFIX\s+(?:[a-zA-Z0-9_\-]*:)?\s*<[^>]*>|BASE\s+<[^>]*>)\s*;?\s*)+',
        re.IGNORECASE
    )
    text = prefix_inline.sub('', text).strip()
    return text


def repair_curie_escapes(query_str: str) -> str:
    """
    Ripara i CURIE nella query SPARQL assicurando che caratteri speciali
    come '#' all'interno dei local name siano propriamente escapati (es. :singer\\#name),
    in piena conformità con la specifica SPARQL 1.1 (PN_LOCAL_ESC).
    Preserva intatti i letterali stringa, gli URI completi <...> e i commenti reali.
    """
    if not query_str:
        return ""

    # Tokenizza preservando letterali e URI racchiusi in <...>
    token_pattern = re.compile(
        r'(?P<LITERAL>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|'
        r'(?P<IRI><[^>\s]+>)|'
        r'(?P<CURIE>(?:[a-zA-Z0-9_\-]*):(?:[a-zA-Z0-9_\-\.]*\\*#[a-zA-Z0-9_\-\.]+))'
    )

    def replace_token(m: re.Match) -> str:
        if m.group("LITERAL") is not None:
            return m.group("LITERAL")
        if m.group("IRI") is not None:
            return m.group("IRI")
        curie = m.group("CURIE")
        if curie is not None:
            # Normalizza rimuovendo eventuali backslash multipli prima di '#' e applicandone esattamente uno: '\#'
            return re.sub(r'\\*#', r'\#', curie)
        return m.group(0)

    return token_pattern.sub(replace_token, query_str)


def clean_generated_sparql(raw_text: str) -> str:
    """
    Estrae la query SPARQL pulita rimuovendo blocchi markdown (```sparql ... ``` o ``` ... ```),
    apici tripli (''' ... ''' o \"\"\" ... \"\"\") e prefissi testuali quali 'sparql:' o 'query:'.
    Se sono presenti più blocchi/query nel testo generato, estrae l'ULTIMA occorrenza valida non vuota.
    Rimuove tutte le dichiarazioni PREFIX ed eventuali BASE generate dal modello.
    Ripara i caratteri speciali nei CURIE applicando i corretti escape SPARQL 1.1 (es. \\#).
    Restituisce la sola query generata pulita (senza prefissi).
    """
    if not raw_text:
        return ""
    text = raw_text.strip()
    
    # 1. Trova tutti i blocchi racchiusi tra tre backtick obliqui ```
    matches_backticks = re.findall(r'```(?:(?!SELECT|ASK|CONSTRUCT|DESCRIBE|PREFIX|BASE)[a-zA-Z0-9_\-]+:?)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    valid_backticks = [m.strip() for m in matches_backticks if m.strip()]
    
    # 2. Trova tutti i blocchi racchiusi tra tre apici singoli '''
    matches_triples = re.findall(r"'''(?:(?!SELECT|ASK|CONSTRUCT|DESCRIBE|PREFIX|BASE)[a-zA-Z0-9_\-]+:?)?\s*(.*?)\s*'''", text, re.DOTALL | re.IGNORECASE)
    valid_triples = [m.strip() for m in matches_triples if m.strip()]
    
    # 3. Trova tutti i blocchi racchiusi tra tre doppi apici """
    matches_double = re.findall(r'"""(?:(?!SELECT|ASK|CONSTRUCT|DESCRIBE|PREFIX|BASE)[a-zA-Z0-9_\-]+:?)?\s*(.*?)\s*"""', text, re.DOTALL | re.IGNORECASE)
    valid_double = [m.strip() for m in matches_double if m.strip()]
    
    candidates = valid_backticks or valid_triples or valid_double
    
    if candidates:
        # Se ci sono più blocchi (es. passaggi intermedi CoT), privilegia l'ultimo blocco contenente una query SPARQL
        sparql_candidates = [c for c in candidates if re.search(r'\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b', c, re.IGNORECASE)]
        if sparql_candidates:
            text = sparql_candidates[-1]
        else:
            text = candidates[-1]
    else:
        # Fallback: cerca pattern markdown parziali o query SPARQL diretta
        m_fallback = re.search(r'```(?:(?!SELECT|ASK|CONSTRUCT|DESCRIBE|PREFIX|BASE)[a-zA-Z0-9_\-]+:?)?\s*(\b(?:SELECT|ASK|CONSTRUCT|DESCRIBE)\b[\s\S]*?)(?:```|$)', text, re.IGNORECASE)
        if m_fallback:
            text = m_fallback.group(1).strip()
        else:
            m_sparql = re.search(r'(\b(?:SELECT|ASK|CONSTRUCT|DESCRIBE)\b[\s\S]+)', text, re.IGNORECASE)
            if m_sparql:
                text = m_sparql.group(1).strip()

    # Rimuove eventuali tag/backtick nidificati residui e prefissi
    text = re.sub(r'^(?:```|"{3}|\'{3})(?:(?!SELECT|ASK|CONSTRUCT|DESCRIBE|PREFIX|BASE)[a-zA-Z0-9_\-]+:?)?\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'(?:```|"{3}|\'{3})$', '', text).strip()
    text = re.sub(r'^(?:sparql|query)\s*:\s*', '', text, flags=re.IGNORECASE).strip()
    # Rimuove tutte le dichiarazioni PREFIX e BASE
    text = strip_sparql_prefixes(text)
    # Ripara gli escape nei CURIE (es. :singer#name -> :singer\#name)
    text = repair_curie_escapes(text)
    return text


def extract_and_validate_prefixes(
    query_str: str = "",
    ontology_prefixes: Optional[Dict[str, str]] = None
) -> Tuple[str, Set[str]]:
    """
    Costruisce l'header PREFIX per i prefissi disponibili e verifica se nella query
    sono utilizzati prefissi non dichiarati.
    Restituisce: (prefix_header, set_prefissi_mancanti).
    """
    effective_prefixes = dict(ontology_prefixes or {})
    prefix_header = build_prefix_header(effective_prefixes)

    # Estrae prefissi usati nella query
    used_prefixes = set(re.findall(r'(?<![<"\'a-zA-Z0-9_\-])([a-zA-Z0-9_\-]*):(?![<"\'a-zA-Z0-9_\-])', query_str))
    query_declared = set(re.findall(r'(?i)PREFIX\s+([a-zA-Z0-9_\-]*):', query_str))
    available_prefixes = set(effective_prefixes.keys()).union(query_declared)

    missing_prefixes = set()
    for p in used_prefixes:
        if p not in available_prefixes:
            missing_prefixes.add(p if p else "(default prefix ':')")

    return prefix_header, missing_prefixes


def postprocess_sparql(
    raw_query: str,
    ontology_prefixes: Optional[Dict[str, str]] = None,
    graph: Optional[rdflib.Graph] = None,
    add_prefixes_to_output: bool = True
) -> Dict[str, Any]:
    """
    Esegue la pipeline di postprocessing senza euristiche:
    1. clean_generated_sparql: pulizia da markdown.
    2. Validazione prefissi.
    3. attach_prefixes_to_query: aggiunge i PREFIX in cima (se richiesto).
    """
    effective_prefixes = dict(ontology_prefixes or {})

    # 1. Pulizia markdown
    cleaned = clean_generated_sparql(raw_query)

    # 2. Validazione prefissi
    prefix_header, missing_prefixes = extract_and_validate_prefixes(
        query_str=cleaned,
        ontology_prefixes=effective_prefixes
    )

    # 3. Query con prefissi in testa (nessuna modifica al corpo della query)
    resolved = attach_prefixes_to_query(cleaned, effective_prefixes) if add_prefixes_to_output else cleaned

    return {
        "cleaned_query": cleaned,
        "resolved_query": resolved,
        "prefix_header": prefix_header,
        "missing_prefixes": list(missing_prefixes),
        "prefixes": effective_prefixes
    }


# ==============================================================================
# ESECUZIONE QUERY SUL GRAFO COMPLETO
# ==============================================================================

def execute_sparql_on_full_graph(
    graph: rdflib.Graph,
    query_str: str,
    prefix_header: str = ""
) -> Tuple[bool, List[List[str]], Optional[str]]:
    """
    Esegue la query SPARQL sul grafo completo RDFLib.
    Se la query contiene già i prefissi, viene eseguita direttamente.
    In caso di errore, tenta il fallback con l'aggiunta esplicita di prefix_header.
    Restituisce: (successo, lista_righe_risultati, eventuale_errore).
    """
    try:
        res = graph.query(query_str)
        rows = [[str(item) for item in row] for row in res]
        return True, rows, None
    except Exception as e_first:
        if prefix_header and prefix_header.strip() and not query_str.strip().upper().startswith("PREFIX"):
            try:
                res = graph.query(prefix_header + "\n" + query_str)
                rows = [[str(item) for item in row] for row in res]
                return True, rows, None
            except Exception as e_second:
                return False, [], str(e_second)
        return False, [], str(e_first)

def compare_results(pred_rows: list, gold_rows: list) -> bool:
    """Confronto multiset rigoroso tra le righe restituite dalla predizione e dal gold."""
    if pred_rows is None or gold_rows is None:
        return False
    if len(pred_rows) == 0 and len(gold_rows) == 0:
        return True
    if len(pred_rows) != len(gold_rows):
        return False
    def row_to_tuple(row):
        return tuple(str(x).strip() for x in row)
    pred_tuples = [row_to_tuple(r) for r in pred_rows]
    gold_tuples = [row_to_tuple(r) for r in gold_rows]
    return Counter(pred_tuples) == Counter(gold_tuples)

# ==============================================================================
# ELABORAZIONE RECORD SINGOLO (V03)
# ==============================================================================

def process_record(
    record: Dict[str, Any],
    dev_path: Union[str, Path] = DEFAULT_DEV_PATH,
    custom_full_graph: Optional[Union[str, Path]] = None,
    graph_cache: Optional[Dict[str, rdflib.Graph]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Elabora un singolo record:
    - Recupera kg_name e domanda da dev.json
    - Carica il Full Graph associato
    - Estrae direttamente i prefissi dall'ontologia completa / {kg_name}_prefixes.json
    - Esegue postprocessing (cleaned_query intatta con prefissi)
    - Esegue la query SPARQL sul Full Graph
    - Appende il risultato nel dizionario del record
    """
    rec_id = record.get("id", "N/A")
    raw_query = record.get("result", "")

    # Recupera indice domanda se presente, altrimenti usa il fallback
    q_idx = record.get("question_index", FALLBACK_QUESTION_INDEX)
    kg_name = "concert_singer"
    question_text = ""
    gold_query = ""

    try:
        gold_query, question_text, kg_name = load_question_from_dev(index=q_idx, dev_path=dev_path)
    except Exception:
        prompt = record.get("prompt", {})
        user_p = prompt.get("user", "") if isinstance(prompt, dict) else str(prompt)
        m_q = re.search(r'question:\s*(.*?)(?:,\s*query\s*SPARQL:|$)', user_p, re.DOTALL | re.IGNORECASE)
        if m_q:
            question_text = m_q.group(1).strip()

    # Risoluzione Full Graph
    if custom_full_graph:
        full_file = Path(custom_full_graph)
        if not full_file.is_absolute():
            full_file = ROOT_DIR / full_file
    else:
        full_file = resolve_kg_full_graph(kg_name)

    if not full_file.exists():
        raise FileNotFoundError(f"File Full Graph non trovato: {full_file}")

    # Gestione cache grafo
    graph_key = str(full_file)
    if graph_cache is not None and graph_key in graph_cache:
        g_full = graph_cache[graph_key]
    else:
        if verbose:
            print(f"--> Caricamento Grafo Completo RDF da: '{full_file}'...")
        fmt = "turtle" if full_file.suffix == ".ttl" else "xml"
        g_full = rdflib.Graph(bind_namespaces="none").parse(str(full_file), format=fmt)
        if graph_cache is not None:
            graph_cache[graph_key] = g_full

    # Estrazione prefissi: da file JSON con fallback sul grafo completo
    ontology_pref = get_ontology_prefixes(kg_name=kg_name, graph=g_full)

    if verbose:
        print("\n" + "=" * 80)
        print(f"   POSTPROCESSING & ESECUZIONE QUERY SU FULL GRAPH (V03) - RECORD ID: {rec_id}")
        print("=" * 80)
        print(f"[INFO] Record ID           : {rec_id}")
        print(f"[INFO] Indice Domanda      : {q_idx}")
        print(f"[INFO] Knowledge Graph     : {kg_name}")
        print(f"[INFO] Full Graph File     : '{full_file}' (Triple: {len(g_full)})")
        print(f"[INFO] Prefissi Ontologia  : {len(ontology_pref)} prefissi")
        if question_text:
            print(f"[INFO] Domanda (NL)        : \"{question_text}\"")

    # Postprocessing Query Generata
    pp = postprocess_sparql(raw_query, ontology_prefixes=ontology_pref, add_prefixes_to_output=True)
    clean_q = pp["cleaned_query"]
    resolved_q = pp["resolved_query"]
    prefix_hdr = pp["prefix_header"]
    missing_pref = pp["missing_prefixes"]

    if verbose:
        print("\n" + "-" * 80)
        print(" [Query SPARQL Pulita (senza prefissi)]")
        print(clean_q if clean_q else "(VUOTA)")
        print("\n [Query SPARQL con Prefissi (pronta all'esecuzione)]")
        print(resolved_q if resolved_q else "(VUOTA)")
        if missing_pref:
            print(f"\n[WARNING] Prefissi usati ma non definiti: {missing_pref}")
        print("-" * 80)

    # Esecuzione diretta della query con prefissi
    success, results_rows, error_msg = False, [], "Nessuna query valida da eseguire"
    if resolved_q:
        success, results_rows, error_msg = execute_sparql_on_full_graph(g_full, resolved_q, prefix_hdr)

    # Esecuzione Gold
    gold_rows: List[List[str]] = []
    gold_success = False
    gold_error = None
    if gold_query:
        pp_gold = postprocess_sparql(gold_query, ontology_prefixes=ontology_pref, add_prefixes_to_output=True)
        gold_resolved = pp_gold["resolved_query"]
        gold_prefix_hdr = pp_gold["prefix_header"]
        gold_success, gold_rows, gold_error = execute_sparql_on_full_graph(g_full, gold_resolved, gold_prefix_hdr)

    if verbose:
        print("\n" + "=" * 80)
        print("                      RISULTATI ESECUZIONE (V03)")
        print("=" * 80)
        print(f"  - Stato Esecuzione Predetta : {'SUCCESSO' if success else 'FALLITO'}")
        print(f"  - Risultati Query Predetta  : {len(results_rows)} righe")
        print(f"  - Risultati Query Gold      : {len(gold_rows)} righe (Stato: {'SUCCESSO' if gold_success else 'FALLITO'})")
        if error_msg:
            print(f"  - Errore Predetta           : {error_msg}")
        if gold_error:
            print(f"  - Errore Gold               : {gold_error}")

        if success and results_rows:
            print(f"\n  [Primi 10 Risultati Predetti]:")
            for idx_row, r in enumerate(results_rows[:10], start=1):
                print(f"    {idx_row}. {r}")
            if len(results_rows) > 10:
                print(f"    ... (+ altri {len(results_rows) - 10} risultati)")
        elif success and not results_rows:
            print("  [Risultato Predetto]: L'interrogazione ha prodotto un insieme vuoto (0 righe).")

        if gold_success and gold_rows:
            print(f"\n  [Primi 10 Risultati Gold]:")
            for idx_row, r in enumerate(gold_rows[:10], start=1):
                print(f"    {idx_row}. {r}")
            if len(gold_rows) > 10:
                print(f"    ... (+ altri {len(gold_rows) - 10} risultati)")
        elif gold_success and not gold_rows:
            print("  [Risultato Gold]: L'interrogazione Gold ha prodotto un insieme vuoto (0 righe).")
        print("=" * 80 + "\n")
    # Calcola se il risultato corrisponde al gold
    is_correct = compare_results(results_rows, gold_rows) if success else False

    # Struttura del risultato appeso al dizionario del record
    execution_info = {
        "cleaned_query": clean_q,
        "resolved_query": resolved_q,
        "full_graph_file": str(full_file.name),
        "execution_success": success,
        "is_correct": is_correct,
        "predicted_rows_count": len(results_rows) if results_rows is not None else 0,
        "gold_rows_count": len(gold_rows) if gold_rows is not None else 0,
        "results": results_rows,
        "gold_result": gold_rows,
        "error": error_msg
    }

    # Salva il commento esistente (se presente) o stringa vuota per assicurare che 'comment' sia l'ultima chiave
    comment = record.pop("comment", "")

    # Appende / aggiorna nel dizionario del record
    record["execution_result"] = execution_info
    record["comment"] = comment
    return record


# ==============================================================================
# RUNNER PRINCIPALE PIPELINE (V03)
# ==============================================================================

_GLOBAL_GRAPH_CACHE: Dict[str, rdflib.Graph] = {}


def run_pipeline(
    json_path: Union[str, Path] = JSON_PATH,
    target_id: Optional[Union[int, List[int], str]] = TARGET_GEN_ID,
    dev_path: Union[str, Path] = DEFAULT_DEV_PATH,
    custom_full_graph: Optional[Union[str, Path]] = CUSTOM_FULL_GRAPH_PATH,
    update_file: bool = UPDATE_JSON_FILE,
    verbose: bool = True,
    graph_cache: Optional[Dict[str, rdflib.Graph]] = None
) -> List[Dict[str, Any]]:
    """
    Carica il file JSON dei risultati, elabora i record selezionati da target_id,
    appende l'esito al dizionario corrispondente e riscrive il file JSON.
    """
    json_file = Path(json_path)
    if not json_file.is_absolute():
        json_file = ROOT_DIR / json_file

    if not json_file.exists():
        raise FileNotFoundError(f"File JSON dei risultati non trovato in: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        print(f"[ATTENZIONE] Il file '{json_file.name}' è vuoto o non contiene una lista di record.")
        return []

    # Normalizzazione target_id
    allowed_ids: Optional[Set[int]] = None
    if isinstance(target_id, int):
        allowed_ids = {target_id}
    elif isinstance(target_id, (list, tuple, set)):
        allowed_ids = {int(x) for x in target_id}
    elif isinstance(target_id, str) and target_id.lower() not in ("all", "none", "*", ""):
        try:
            allowed_ids = {int(target_id)}
        except ValueError:
            allowed_ids = {target_id}

    # Filtra i record da elaborare
    records_to_process = []
    for rec in records:
        rec_id = rec.get("id")
        if allowed_ids is None or (rec_id is not None and (rec_id in allowed_ids or str(rec_id) in allowed_ids)):
            records_to_process.append(rec)

    if not records_to_process:
        print(f"[ATTENZIONE] Nessun record trovato con ID: {target_id} in '{json_file.name}'.")
        print(f"  ID disponibili: {[r.get('id') for r in records]}")
        return []

    print("=" * 80)
    id_desc = f"ID = {target_id}" if allowed_ids is not None else f"TUTTI I RECORD (Totale: {len(records_to_process)})"
    print(f"  POSTPROCESSING & QUERYING V03 SU: '{json_file.name}' ({id_desc})")
    print("=" * 80)

    if graph_cache is None:
        graph_cache = _GLOBAL_GRAPH_CACHE
    processed_records = []

    for rec in records_to_process:
        updated_rec = process_record(
            record=rec,
            dev_path=dev_path,
            custom_full_graph=custom_full_graph,
            graph_cache=graph_cache,
            verbose=verbose
        )
        processed_records.append(updated_rec)

    # Aggiorna il file JSON su disco
    if update_file:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"--> [OK] Risultati salvati e appesi con successo nel file: '{json_file}'\n")

    return processed_records


# ==============================================================================
# MAIN & CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="postprocessing_v03: Postprocessing ed esecuzione su Full Graph con estrazione nativa prefissi dall'ontologia."
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=JSON_PATH,
        help=f"Percorso del file JSON dei risultati (default da codice: '{JSON_PATH}')"
    )
    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help="ID specifico della risposta da testare (sovrascrive la variabile TARGET_GEN_ID nel codice)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Testa tutte le risposte presenti nel file JSON (sovrascrive TARGET_GEN_ID)"
    )
    parser.add_argument(
        "--full-graph",
        type=str,
        default=CUSTOM_FULL_GRAPH_PATH,
        help="Percorso personalizzato del grafo completo RDF"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Non salvare l'esito nel file JSON dei risultati"
    )

    args = parser.parse_args()

    final_target_id = TARGET_GEN_ID
    if args.all:
        final_target_id = None
    elif args.id is not None:
        final_target_id = args.id

    final_json_path = args.json_path
    final_full_graph = args.full_graph
    final_update_file = not args.no_save if args.no_save else UPDATE_JSON_FILE

    run_pipeline(
        json_path=final_json_path,
        target_id=final_target_id,
        custom_full_graph=final_full_graph,
        update_file=final_update_file,
        verbose=True
    )


if __name__ == "__main__":
    main()
