import argparse
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Supporto encoding UTF-8 su console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import ollama
except ImportError:
    ollama = None

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

# Import preprocessing per l'estrazione dinamica del sottografo, tassonomia ed embedding
import preprocessing_v12 as preprocessing


def set_preprocessing_module(mod):
    """Permette l'iniezione dinamica del modulo di preprocessing da parte dei batch runner."""
    global preprocessing
    preprocessing = mod


MODEL_NAME = "qwen2.5-coder:7b"
TEMPERATURE = 0.0
DEFAULT_NUM_CTX = 8192
OUTPUT_JSON_FILE = "cot_v13.json"
DEV_JSON_PATH = os.path.join(os.path.dirname(__file__), "spider4sparql", "dev.json")
DEFAULT_QUESTION_INDEX = 25

# Cache globale in memoria
_DEV_DATA_CACHE = None
_VALID_KG_CACHE = {}
_TAXONOMY_CACHE: Dict[str, Dict[str, Any]] = {}

# Cache globale per modello LLM
_MODEL = None
_TOKENIZER = None
_LOADED_MODEL_NAME = None
_BACKEND = None  # 'ollama' o 'transformers'


SYSTEM_PROMPT = """You are an expert Semantic Web assistant specialized in text-to-SPARQL translation tasks.

CRITICAL CONSTRAINTS & RULES:
- STRICT ONTOLOGY BOUNDARY: Use EXCLUSIVELY the classes, properties, individuals, datatypes and prefixes defined in the provided target ontology subgraph and hierarchical taxonomy.
- Do NOT specify prefix declaration
- Use only SPARQL syntax. Do NOT use SQL operators or syntax.
- Always prefer flat and direct queries.

CHAIN-OF-THOUGHT REASONING & OUTPUT FORMAT:
First, analyze the question step-by-step:
- Step 1 (Intent & Target Concepts): Identify what information is requested, whether grouping/aggregations are needed, and the core entities or concepts involved.
- Step 2 (Schema Alignment & WHERE Patterns): Match concepts with exact CURIEs from the ontology subgraph and hierarchical taxonomy. Define the RDF triple patterns in the WHERE clause.
- Step 3 (Aggregations & Subqueries): Identify aggregate functions (COUNT, MAX, MIN, AVG, SUM). If comparing individual values against a global aggregate, define an inner sub-SELECT inside WHERE. Otherwise, plan solution modifiers outside WHERE.
- Step 4 (Filters & Solution Modifiers): Identify FILTER conditions, GROUP BY, ORDER BY, and LIMIT.
- Step 5 (SELECT Projection & SPARQL Assembly): Define the projected variables, verify that NO SQL operators exist and assemble the final valid SPARQL query, ensuring variable names and prefix bounds are fully consistent.

Output your step-by-step reasoning first, and conclude your response with the final valid SPARQL query enclosed in triple backticks with the 'sparql' language tag (```sparql ... ```)."""


# ==============================================================================
# GESTIONE DATASET DEV.JSON
# ==============================================================================

def load_dev_dataset(dev_path: str = DEV_JSON_PATH) -> List[Dict[str, Any]]:
    """Carica l'intero dataset dev.json."""
    global _DEV_DATA_CACHE
    if _DEV_DATA_CACHE is not None:
        return _DEV_DATA_CACHE

    if not os.path.exists(dev_path):
        alt_path = os.path.join("spider4sparql", "dev.json")
        if os.path.exists(alt_path):
            dev_path = alt_path
        else:
            raise FileNotFoundError(f"File dev.json non trovato in '{dev_path}'")

    with open(dev_path, "r", encoding="utf-8") as f:
        _DEV_DATA_CACHE = json.load(f)
    return _DEV_DATA_CACHE


def load_question_from_dev(
    index: int = DEFAULT_QUESTION_INDEX,
    dev_path: str = DEV_JSON_PATH
) -> Tuple[str, str, str]:
    """Carica domanda, query gold e kg_name all'indice specificato da dev.json."""
    data = load_dev_dataset(dev_path)
    if index < 0 or index >= len(data):
        raise IndexError(f"Indice {index} non valido per dev.json (totale elementi: {len(data)})")
    entry = data[index]
    return entry["question"], entry.get("query", ""), entry.get("kg_name", "")


# ==============================================================================
# SALVATAGGIO RISULTATI JSON
# ==============================================================================

def get_next_id(filepath: str = OUTPUT_JSON_FILE) -> int:
    """Restituisce il prossimo ID numerico incrementale leggendo i record presenti nel file JSON."""
    if not os.path.exists(filepath):
        return 1
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return 1
            data = json.loads(content)
            if not isinstance(data, list):
                return 1

            existing_ids = []
            for item in data:
                if isinstance(item, dict) and "id" in item:
                    item_id = item["id"]
                    if isinstance(item_id, int):
                        existing_ids.append(item_id)
                    elif isinstance(item_id, str) and item_id.isdigit():
                        existing_ids.append(int(item_id))

            if existing_ids:
                return max(existing_ids) + 1
            return len(data) + 1
    except Exception:
        return 1


def append_to_json(record: Dict[str, Any], filepath: str = OUTPUT_JSON_FILE) -> None:
    """Aggiunge un nuovo record al file JSON senza sovrascrivere."""
    data = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    if not isinstance(data, list):
                        data = [data]
        except Exception as e:
            print(f"[ATTENZIONE] Errore nella lettura del file esistente {filepath}: {e}. Verrà reinizializzata la lista.")
            data = []

    data.append(record)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"--> Record (ID: {record.get('id')}) salvato con successo in '{filepath}' (Totale record: {len(data)})")


# ==============================================================================
# GESTIONE SOTTOGRAFI, TASSONOMIA ED ESECUZIONE
# ==============================================================================

def resolve_ontology_file(kg_name: str, base_dir: str = "spider4sparql") -> str:
    """Risolve il percorso del file RDF (.ttl o .owl) per il KG specificato."""
    script_dir = Path(__file__).parent.resolve()
    base_path = script_dir / base_dir if not Path(base_dir).is_absolute() else Path(base_dir)

    ttl_file = base_path / f"{kg_name}.ttl"
    if ttl_file.exists() and ttl_file.stat().st_size > 0:
        return str(ttl_file)

    root_ttl = script_dir / f"{kg_name}.ttl"
    if root_ttl.exists() and root_ttl.stat().st_size > 0:
        return str(root_ttl)

    zip_file = base_path / "materialized_triples_dev.zip"
    if zip_file.exists():
        try:
            with zipfile.ZipFile(zip_file, "r") as z:
                target_inside = f"materialized_triples/{kg_name}.ttl"
                if target_inside in z.namelist():
                    info = z.getinfo(target_inside)
                    if info.file_size > 0:
                        z.extract(target_inside, base_path)
                        extracted = base_path / "materialized_triples" / f"{kg_name}.ttl"
                        if extracted.exists() and extracted.stat().st_size > 0:
                            extracted.rename(ttl_file)
                            try:
                                (base_path / "materialized_triples").rmdir()
                            except Exception:
                                pass
                            return str(ttl_file)
        except Exception as e:
            print(f"[ATTENZIONE] Errore durante l'estrazione da {zip_file}: {e}")

    if preprocessing is not None and hasattr(preprocessing, "get_concert_singer_dataset"):
        cs_ttl, _, _ = preprocessing.get_concert_singer_dataset(str(base_path))
        return cs_ttl

    return str(ttl_file)


def get_or_load_taxonomy(kg_name: str) -> Optional[Dict[str, Any]]:
    """Carica la tassonomia dal file JSON creato da preprocessing o la ricalcola come fallback."""
    if kg_name not in _TAXONOMY_CACHE:
        # 1. Tentativo prioritario da file JSON
        for p in [Path("spider4sparql") / f"{kg_name}.taxonomy_cache.json", Path(f"{kg_name}.taxonomy_cache.json")]:
            if p.exists() and p.stat().st_size > 0:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        _TAXONOMY_CACHE[kg_name] = json.load(f)
                    break
                except Exception:
                    pass

        # 2. Fallback: ricalcolo ex-novo con il modulo di preprocessing se il file non è presente
        if kg_name not in _TAXONOMY_CACHE and preprocessing is not None:
            onto_path = resolve_ontology_file(kg_name)
            if hasattr(preprocessing, "extract_taxonomy_hierarchy") and os.path.exists(onto_path):
                try:
                    tax = preprocessing.extract_taxonomy_hierarchy(onto_path)
                    if tax:
                        _TAXONOMY_CACHE[kg_name] = tax
                except Exception as e:
                    print(f"[ATTENZIONE] Ricalcolo tassonomia fallito per '{kg_name}': {e}")

    return _TAXONOMY_CACHE.get(kg_name)


def extract_subgraph_content(
    question_text: str,
    kg_name: str = "concert_singer",
    output_ttl_path: Optional[str] = None
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Estrae il sottografo RDF e recupera la tassonomia (da JSON se già presente)."""
    if preprocessing is None:
        raise ImportError("Modulo di preprocessing non disponibile.")

    taxonomy = get_or_load_taxonomy(kg_name)
    onto_path = resolve_ontology_file(kg_name)
    dense_dir = f"dense_space_{kg_name.lower()}" if os.path.exists(f"dense_space_{kg_name.lower()}") else f"dense_space_{kg_name}"

    res = preprocessing.run_pipeline_for_ontology(
        onto_name=kg_name,
        onto_path=onto_path,
        query_test=question_text,
        output_dir=dense_dir,
        output_subgraph_path=output_ttl_path
    )

    generated_path = res.get("subgraph_ttl_path", output_ttl_path or f"{kg_name.lower()}_subgraph.ttl")
    with open(generated_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not taxonomy:
        taxonomy = res.get("taxonomy")
        if taxonomy:
            _TAXONOMY_CACHE[kg_name] = taxonomy

    return content, generated_path, taxonomy


# ==============================================================================
# CARICAMENTO MODELLO LLM
# ==============================================================================

def load_model(model_name: str = MODEL_NAME) -> bool:
    """Carica e verifica la presenza del modello specificato su Ollama o Hugging Face Transformers."""
    global _MODEL, _TOKENIZER, _LOADED_MODEL_NAME, _BACKEND

    if ollama is not None:
        try:
            res = ollama.list()
            models_info = getattr(res, "models", []) if hasattr(res, "models") else res.get("models", [])
            downloaded_names = []
            for m in models_info:
                if hasattr(m, "model"):
                    downloaded_names.append(m.model)
                elif isinstance(m, dict):
                    downloaded_names.append(m.get("name", m.get("model", "")))
                else:
                    downloaded_names.append(str(m))

            name_without_tag = model_name.split(":")[0]
            is_present = any(
                model_name == d or d.startswith(f"{model_name}:") or name_without_tag == d.split(":")[0]
                for d in downloaded_names
            )

            if is_present:
                _LOADED_MODEL_NAME = model_name
                _BACKEND = "ollama"
                print(f"--> [OK] Modello Ollama '{model_name}' pronto all'uso.")
                return True
        except Exception as e:
            print(f"[ATTENZIONE] Fallback da Ollama a Transformers: {e}")

    # Fallback su Transformers
    if torch is not None and AutoModelForCausalLM is not None:
        try:
            print(f"--> Inizializzazione Transformers per '{model_name}'...")
            _TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if torch.cuda.is_available():
                _MODEL = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                _MODEL = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to("cpu")

            _MODEL.eval()
            _LOADED_MODEL_NAME = model_name
            _BACKEND = "transformers"
            print(f"--> [OK] Modello '{model_name}' caricato con successo su Transformers!")
            return True
        except Exception as e:
            print(f"[ERRORE] Impossibile caricare il modello '{model_name}' con Transformers: {e}")

    print(f"[ERRORE] Impossibile caricare il modello '{model_name}' né con Ollama né con Transformers.")
    return False


# ==============================================================================
# PIPELINE DI GENERAZIONE ZERO-SHOT CHAIN-OF-THOUGHT (CoT v13 + TAXONOMY)
# ==============================================================================

def run_cot(
    nl_query: str,
    subgraph: Optional[str] = None,
    call_id: Optional[Union[str, int]] = None,
    question_index: Optional[int] = None,
    kg_name: str = "concert_singer",
    model_name: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    taxonomy: Optional[Union[Dict[str, Any], str]] = None,
    include_taxonomy: bool = True,
    output_json_path: str = OUTPUT_JSON_FILE,
    dev_path: str = DEV_JSON_PATH
) -> Dict[str, Any]:
    """
    Esegue una chiamata Zero-Shot Chain-of-Thought (CoT v13) a 5 step
    con prompt ottimizzato anti-SQL leakage (FILTER NOT EXISTS al posto di EXCEPT, congiunzioni in WHERE al posto di INTERSECT),
    vincolo di binding rigoroso su variabili aggregate e proiezioni esplicite.
    """
    global _MODEL, _TOKENIZER, _BACKEND

    if call_id is None:
        call_id = get_next_id(output_json_path)

    def _read_if_file(val: Optional[str]) -> Optional[str]:
        if isinstance(val, str) and os.path.isfile(val):
            with open(val, "r", encoding="utf-8") as f:
                return f.read().strip()
        return val

    # Sottografo Target e Tassonomia
    target_tax = taxonomy
    if target_tax is None and include_taxonomy:
        target_tax = get_or_load_taxonomy(kg_name=kg_name)

    if subgraph is None:
        target_ttl_file = f"{kg_name.lower()}_t_subgraph.ttl"
        subgraph_content, _, extracted_tax = extract_subgraph_content(
            question_text=nl_query,
            kg_name=kg_name,
            output_ttl_path=target_ttl_file
        )
        if target_tax is None and include_taxonomy:
            target_tax = extracted_tax
    else:
        subgraph_content = _read_if_file(subgraph)

    # Costruzione sezione Tassonomia
    tax_section = ""
    if include_taxonomy and target_tax:
        tax_str = json.dumps(target_tax, indent=2, ensure_ascii=False) if isinstance(target_tax, (dict, list)) else str(target_tax)
        tax_section = f"\n\nTaxonomy (Hierarchical Format):\n{tax_str}"

    # Costruzione del prompt utente
    target_user_prompt = f"""Target Ontology Subgraph:
{subgraph_content}{tax_section}

Natural Language Question: {nl_query}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": target_user_prompt}
    ]

    print(f"\n--- Esecuzione Chiamata CoT v13 (Zero-Shot + Tassonomia) ID: {call_id} ---")
    if question_index is not None:
        print(f"Question Index Target : {question_index}")
    print(f"Question Target       : {nl_query}")
    print(f"Knowledge Graph       : {kg_name}")
    print(f"Modello LLM           : {model_name}")
    print(f"Context Length num_ctx: {num_ctx}")
    print(f"Tassonomia Inclusa    : {bool(include_taxonomy and target_tax)}")

    result_text = ""
    error_msg = None

    try:
        if _LOADED_MODEL_NAME != model_name or _BACKEND is None:
            success = load_model(model_name)
            if not success:
                raise RuntimeError(f"Impossibile inizializzare il modello '{model_name}'.")

        if _BACKEND == "ollama":
            response = ollama.chat(
                model=model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                }
            )
            result_text = response.get("message", {}).get("content", "").strip()
        else:
            if hasattr(_TOKENIZER, "apply_chat_template") and _TOKENIZER.chat_template:
                try:
                    formatted_input = _TOKENIZER.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    formatted_input = f"{SYSTEM_PROMPT}\n\n[USER]\n{target_user_prompt}\n\n[ASSISTANT]\n"
            else:
                formatted_input = f"{SYSTEM_PROMPT}\n\n[USER]\n{target_user_prompt}\n\n[ASSISTANT]\n"

            inputs = _TOKENIZER(formatted_input, return_tensors="pt").to(_MODEL.device)

            generate_kwargs = {
                "max_new_tokens": 1024,
                "pad_token_id": _TOKENIZER.eos_token_id if _TOKENIZER.eos_token_id is not None else _TOKENIZER.pad_token_id
            }
            if temperature == 0.0 or temperature == 0:
                generate_kwargs["do_sample"] = False
            else:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = temperature

            with torch.no_grad():
                outputs = _MODEL.generate(
                    **inputs,
                    **generate_kwargs
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            result_text = _TOKENIZER.decode(generated_tokens, skip_special_tokens=True).strip()

    except Exception as e:
        error_msg = str(e)
        print(f"[ERRORE] Chiamata al modello fallita: {e}")

    record = {
    "id": call_id,
    "pipeline": "cot_v13",
    "question_index": question_index,
    "timestamp": datetime.now().isoformat(),
    "model": model_name,
    "num_ctx": num_ctx,
    "include_taxonomy": bool(include_taxonomy and target_tax),
    "prompt": {
        "system": SYSTEM_PROMPT,
        "user": target_user_prompt
    },
    "result": result_text if not error_msg else f"ERROR: {error_msg}",
    "comment": ""
    }

    append_to_json(record, filepath=output_json_path)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esecuzione Zero-Shot Chain-of-Thought (CoT v13 + Tassonomia) Text-to-SPARQL")
    parser.add_argument("-i", "--index", type=int, default=DEFAULT_QUESTION_INDEX, help=f"Indice della domanda target in dev.json (default: {DEFAULT_QUESTION_INDEX})")
    parser.add_argument("--subgraph", type=str, default=None, help="File o percorso del sottografo RDF target")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help=f"Nome modello (default: {MODEL_NAME})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help=f"Temperatura del modello (default: {TEMPERATURE})")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help=f"Context length Ollama (default: {DEFAULT_NUM_CTX})")
    parser.add_argument("--no-taxonomy", action="store_true", default=False, help="Disabilita l'inclusione della tassonomia gerarchica")
    args = parser.parse_args()

    print(f"=== Avvio Esecuzione cot_v13.py (Domanda Indice: {args.index}) ===")

    test_query, gold_query, kg_name = load_question_from_dev(args.index)
    print(f"Knowledge Graph : {kg_name}")
    print(f"Domanda (NL)    : \"{test_query}\"")
    print(f"Query Gold      : {gold_query}")

    ready = load_model(args.model)
    if ready:
        result = run_cot(
            nl_query=test_query,
            subgraph=args.subgraph,
            question_index=args.index,
            kg_name=kg_name,
            model_name=args.model,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            include_taxonomy=not args.no_taxonomy
        )
        print("\n--- Risultato Ottenuto dal Modello ---")
        print(result["result"])
    else:
        print("[INFO] Il modello non è pronto. Verifica la configurazione.")
