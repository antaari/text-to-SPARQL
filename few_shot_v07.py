import argparse
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

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
    global preprocessing
    preprocessing = mod


MODEL_NAME = "qwen2.5-coder:7b"
OUTPUT_JSON_FILE = "few_shots_v07.json"
DEV_JSON_PATH = os.path.join(os.path.dirname(__file__), "spider4sparql", "dev.json")
DEFAULT_QUESTION_INDEX = 41

# 4 Shot Fissi da dev.json ordinati per livello/tipo di query:
# 1. ID 320: Easy - Query semplice di selezione/proiezione (KG: cre_Doc_Template_Mgt)
# 2. ID 147: Medium - Utilizzo di FILTER e JOIN tra classi (KG: car_1)
# 3. ID 933: Hard - Utilizzo di GROUP BY e HAVING con aggregazione (KG: dog_kennels)
# 4. ID 97: Extra Hard - Subquery nidificata con aggregazione AVG e FILTER (KG: car_1)
DEFAULT_EXAMPLE_INDICES = [320, 147, 933, 97]
EXAMPLE_DESCRIPTIONS = {
    320: "Easy (Simple query example)",
    147: "Medium (FILTER example)",
    933: "Hard (HAVING example)",
    97: "Extra Hard (subquery example)"
}

# Cache globale in memoria per dev.json
_DEV_DATA_CACHE = None

# Cache globale in memoria per i 4 shot fissi (sottografi ed elementi)
_FIXED_EXAMPLES_CACHE = {}

# Cache globale per modello LLM
_MODEL = None
_TOKENIZER = None
_LOADED_MODEL_NAME = None
_BACKEND = None  # 'ollama' o 'transformers'


def load_dev_dataset(dev_path: str = DEV_JSON_PATH) -> List[Dict[str, Any]]:
    """
    Carica l'intero dataset dev.json.
    """
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
    """
    Carica la domanda e la query gold dal file dev.json all'indice specificato (0-based).
    Restituisce (question, gold_query, kg_name).
    """
    data = load_dev_dataset(dev_path)
    if index < 0 or index >= len(data):
        raise IndexError(f"Indice {index} non valido per dev.json (totale elementi: {len(data)})")
    entry = data[index]
    return entry["question"], entry.get("query", ""), entry.get("kg_name", "")


SYSTEM_PROMPT = (
    "you're an expert in ontologies, sparql querying and text-to sparql. "
    "given a question in natural language and ontology triples translate the question into a valid sparql query. "
    "Use only the triples and the resources in the ontology and the taxonomy provided. "
    "Answer only with the sparql query enclosed between '''."
    "Use CURIE notation as in the provided ontology and taxonomy."
    "Do not specify prefixes"
)


def format_curie_items(items: Union[Set[str], List[str]], prefixes: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Formatta una lista/set di URI o termini nella notazione CURIE compatta
    utilizzando la funzione format_term_curie del modulo di preprocessing.
    """
    if not items:
        return []

    curie_list = []
    for item in sorted(list(items)):
        if not item:
            continue
        if prefixes and preprocessing is not None and hasattr(preprocessing, "format_term_curie"):
            curie_val = preprocessing.format_term_curie(item, prefixes)
            curie_list.append(curie_val)
        else:
            curie_list.append(str(item))
    return curie_list


def resolve_ontology_file(kg_name: str, base_dir: str = "spider4sparql") -> str:
    """
    Risolve il percorso del file RDF (.ttl o .owl) per il KG specificato.
    Se non presente localmente in spider4sparql/, lo estrae automaticamente da materialized_triples_dev.zip.
    """
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
                            # Sanitizza il file se l'ultima riga è troncata
                            try:
                                with open(ttl_file, "r", encoding="utf-8", errors="ignore") as tf:
                                    lines = tf.read().splitlines()
                                fixed = False
                                while lines and not lines[-1].strip().endswith("."):
                                    lines.pop()
                                fixed = True
                                if fixed:
                                    with open(ttl_file, "w", encoding="utf-8") as tf:
                                        tf.write("\n".join(lines) + "\n")
                            except Exception:
                                pass
                            return str(ttl_file)
        except Exception as e:
            print(f"[ATTENZIONE] Errore durante l'estrazione da {zip_file}: {e}")

    # Fallback concert_singer
    if preprocessing is not None and hasattr(preprocessing, "get_concert_singer_dataset"):
        cs_ttl, _, _ = preprocessing.get_concert_singer_dataset(str(base_path))
        return cs_ttl

    return str(ttl_file)


def extract_subgraph_and_elements(
    question_text: str,
    kg_name: str = "concert_singer",
    output_ttl_path: Optional[str] = None
) -> Tuple[str, str, List[str], Dict[str, Any]]:
    """
    Applica la pipeline di preprocessing alla domanda indicata per estrarre:
    - il sottografo RDF (.ttl)
    - il percorso del file .ttl salvato
    - le entity estratte (convertite in notazione CURIE)
    - la tassonomia gerarchica (Formato 1: Tree Dict)
    Restituisce: (contenuto_ttl, percorso_file, entities_curie, taxonomy)
    """
    if preprocessing is None:
        raise ImportError("Modulo di preprocessing non disponibile.")

    prep_module_name = preprocessing.__name__
    print(f"\n[PREPROCESSING] Estrazione sottografo ed elementi con {prep_module_name} per: '{question_text}' (KG: {kg_name})...", flush=True)
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

    if output_ttl_path and output_ttl_path != generated_path:
        with open(generated_path, "r", encoding="utf-8") as f_in:
            content = f_in.read().strip()
        with open(output_ttl_path, "w", encoding="utf-8") as f_out:
            f_out.write(content)
        final_path = output_ttl_path
    else:
        final_path = generated_path
        with open(final_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

    # Carica prefissi per formattazione CURIE
    prefixes = {}
    prefixes_file = f"{kg_name.lower()}_prefixes.json"
    if os.path.exists(prefixes_file):
        try:
            with open(prefixes_file, "r", encoding="utf-8") as pf:
                prefixes = json.load(pf)
        except Exception:
            prefixes = {}

    raw_entities = res.get("selected_entities", set())
    taxonomy = res.get("taxonomy", {})

    entities_curie = format_curie_items(raw_entities, prefixes)

    print(f"[OK] Sottografo salvato in: '{final_path}' ({len(res.get('subgraph_triples', []))} triple)", flush=True)
    print(f"[OK] Entità estratte   : {entities_curie}", flush=True)

    return content, final_path, entities_curie, taxonomy


def extract_subgraph_for_query(
    question_text: str,
    kg_name: str = "concert_singer",
    output_ttl_path: Optional[str] = None
) -> Tuple[str, str]:
    """Retrocompatibilità per estrazione solo del sottografo."""
    content, path, _, _ = extract_subgraph_and_elements(question_text, kg_name, output_ttl_path)
    return content, path


def build_target_user_prompt(
    subgraph_content: str,
    nl_query: str,
    taxonomy: Optional[Dict[str, Any]] = None,
    entities: Optional[List[str]] = None,
    include_taxonomy: bool = True
) -> str:
    """
    Costruisce il prompt dell'utente per la domanda target.
    In few_shots_v07 la domanda target include sia il sottografo ontologico che la tassonomia gerarchica.
    """
    if include_taxonomy and taxonomy:
        tax_str = json.dumps(taxonomy, indent=2, ensure_ascii=False)
        return (
            f"Given the following ontology subgraph:\n{subgraph_content}\n\n"
            f"Taxonomy (Hierarchical Format):\n{tax_str}\n\n"
            f"translate the question \"{nl_query}\" into SPARQL."
        )
    return f"Given the following ontology subgraph {subgraph_content}, translate the question \"{nl_query}\" into SPARQL."


def format_assistant_example_sparql(gold_sparql: str) -> str:
    """
    Formatta la query SPARQL gold dell'esempio selezionato nel formato atteso dal modello (racchiusa tra ''').
    """
    clean_query = gold_sparql.strip()
    return f"''' sparql:\n{clean_query}\n'''"


def load_model(model_name: str = MODEL_NAME) -> bool:
    """
    Inizializza il backend di generazione:
    1. Verifica e prepara Ollama se attivo.
    2. Fallback trasparente su HuggingFace Transformers se Ollama non è disponibile.
    """
    global _MODEL, _TOKENIZER, _LOADED_MODEL_NAME, _BACKEND

    if ollama is not None:
        try:
            models_info = ollama.list().get("models", [])
            downloaded_names = []
            for m in models_info:
                if isinstance(m, dict):
                    downloaded_names.append(m.get("name", ""))
                    downloaded_names.append(m.get("model", ""))
                elif hasattr(m, "model"):
                    downloaded_names.append(getattr(m, "model", ""))

            is_present = any(
                model_name == name or name.startswith(f"{model_name}:") or f"{model_name}:" in name
                for name in downloaded_names if name
            )

            if not is_present:
                print(f"--> Modello Ollama '{model_name}' non trovato localmente. Avvio pull automatico...")
                for progress in ollama.pull(model_name, stream=True):
                    status = progress.get("status", "")
                    completed = progress.get("completed") or 0
                    total = progress.get("total") or 0
                    if total > 0:
                        percent = round((completed / total) * 100, 1)
                        print(f"\r--> Ollama Pull: {status} ({percent}%)", end="", flush=True)
                    else:
                        print(f"\r--> Ollama Pull: {status}", end="", flush=True)
                print(f"\n--> [OK] Modello '{model_name}' pronto su Ollama!")
            else:
                print(f"--> [OK] Modello '{model_name}' pronto su Ollama.")
            _BACKEND = "ollama"
            _LOADED_MODEL_NAME = model_name
            return True
        except Exception:
            pass

    if AutoModelForCausalLM is not None and AutoTokenizer is not None and torch is not None:
        if _MODEL is not None and _TOKENIZER is not None and _LOADED_MODEL_NAME == model_name:
            print(f"--> [OK] Modello Transformers '{model_name}' già caricato in memoria.")
            _BACKEND = "transformers"
            return True

        print(f"=== Verifica ed Inizializzazione Modello Transformers: '{model_name}' ===")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"--> Dispositivo rilevato: {device}")
            print(f"--> Caricamento Tokenizer per '{model_name}'...")
            _TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"--> Caricamento Modello per '{model_name}'...")
            if device == "cuda":
                _MODEL = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                _MODEL = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                _MODEL.to("cpu")

            _MODEL.eval()
            _LOADED_MODEL_NAME = model_name
            _BACKEND = "transformers"
            print(f"--> [OK] Modello '{model_name}' caricato con successo su Transformers!")
            return True
        except Exception as e:
            print(f"[ERRORE] Impossibile caricare il modello '{model_name}' con Transformers: {e}")

    print(f"[ERRORE] Impossibile caricare il modello '{model_name}' né con Ollama né con Transformers.")
    return False


def get_next_id(filepath: str = OUTPUT_JSON_FILE) -> int:
    """
    Restituisce il prossimo ID numerico incrementale leggendo i record presenti nel file JSON.
    Se il file non esiste o è vuoto, parte da 1.
    """
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
    """
    Aggiunge un nuovo record al file JSON specificato senza sovrascrivere i risultati precedenti.
    """
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


def run_few_shot(
    nl_query: str,
    subgraph: Optional[str] = None,
    taxonomy: Optional[Dict[str, Any]] = None,
    entities: Optional[List[str]] = None,
    call_id: Optional[Union[str, int]] = None,
    question_index: Optional[int] = None,
    example_indices: Optional[List[int]] = None,
    kg_name: str = "concert_singer",
    model_name: str = MODEL_NAME,
    temperature: float = 0.0,
    num_ctx: Optional[int] = 18432,
    include_taxonomy: bool = True,
    include_shot_taxonomy: bool = False,
    output_json_path: str = OUTPUT_JSON_FILE,
    dev_path: str = DEV_JSON_PATH
) -> Dict[str, Any]:
    """
    Esegue una chiamata Few-Shot v07 con tecnica SCHEMA-AGNOSTIC:
    - Gli shot (dimostrazioni) sono privi di tassonomia e contengono SOLO il sottografo ontologico e la domanda.
    - La domanda target invece contiene sia la tassonomia gerarchica sia il sottografo ontologico.

    I 4 esempi predefiniti provengono da dev.json:
    1. ID 320: Easy - Query semplice (KG: cre_Doc_Template_Mgt)
    2. ID 147: Medium - Utilizzo di FILTER (KG: car_1)
    3. ID 933: Hard - Utilizzo di HAVING (KG: dog_kennels)
    4. ID 97: Extra Hard - Subquery nidificata (KG: car_1)
    """
    global _FIXED_EXAMPLES_CACHE, _LOADED_MODEL_NAME, _BACKEND

    if call_id is None:
        call_id = get_next_id(output_json_path)

    def _read_if_file(val: Optional[str]) -> Optional[str]:
        if isinstance(val, str) and os.path.isfile(val):
            print(f"--> Lettura del file subgraph da: '{val}'")
            with open(val, "r", encoding="utf-8") as f:
                return f.read().strip()
        return val

    # 1. Caricamento dei 4 esempi fissi da dev.json nell'ordine definito
    chosen_indices = example_indices if example_indices is not None and len(example_indices) > 0 else DEFAULT_EXAMPLE_INDICES
    dev_data = load_dev_dataset(dev_path)

    examples_details = []
    example_messages = []

    print(f"\n--> Caricamento ed elaborazione di {len(chosen_indices)} Shot Fissi (Schema-Agnostic, senza tassonomia):")
    for rank, ex_idx in enumerate(chosen_indices, start=1):
        if ex_idx < 0 or ex_idx >= len(dev_data):
            raise IndexError(f"Indice esempio {ex_idx} non valido per dev.json (lunghezza: {len(dev_data)})")

        ex_entry = dev_data[ex_idx]
        ex_question = ex_entry.get("question", "")
        ex_gold_query = ex_entry.get("query", "")
        ex_kg = ex_entry.get("kg_name", "")
        ex_role = EXAMPLE_DESCRIPTIONS.get(ex_idx, f"Example #{rank}")

        # Recupera il sottografo e gli elementi dalla cache o estraili
        if ex_idx in _FIXED_EXAMPLES_CACHE:
            cached_data = _FIXED_EXAMPLES_CACHE[ex_idx]
            ex_subgraph_content = cached_data["subgraph"]
            ex_taxonomy = cached_data.get("taxonomy", {})
            print(f"  [{rank}] Indice {ex_idx} ({ex_role}) [CACHED] | KG: {ex_kg} | Q: \"{ex_question}\"")
        else:
            print(f"  [{rank}] Indice {ex_idx} ({ex_role}) | KG: {ex_kg} | Q: \"{ex_question}\"")
            ex_ttl_file = f"{ex_kg.lower()}_ex_subgraph.ttl"
            ex_subgraph_content, _, ex_ent, ex_taxonomy = extract_subgraph_and_elements(
                question_text=ex_question,
                kg_name=ex_kg,
                output_ttl_path=ex_ttl_file
            )
            _FIXED_EXAMPLES_CACHE[ex_idx] = {
                "subgraph": ex_subgraph_content,
                "taxonomy": ex_taxonomy,
                "entities": ex_ent
            }

        # Costruzione del prompt di input per l'esempio:
        # Schema-Agnostic: gli shot NON contengono tassonomia (solo subgraph + question)
        if include_shot_taxonomy and ex_taxonomy:
            ex_tax_str = json.dumps(ex_taxonomy, indent=2, ensure_ascii=False)
            ex_user_content = (
                f"ontology subgraph:\n{ex_subgraph_content}\n\n"
                f"taxonomy:\n{ex_tax_str}\n\n"
                f"question: {ex_question}"
            )
        else:
            ex_user_content = f"ontology subgraph: {ex_subgraph_content}, question: {ex_question}"

        ex_assistant_content = format_assistant_example_sparql(ex_gold_query)

        examples_details.append({
            "rank": rank,
            "example_index": ex_idx,
            "example_role": ex_role,
            "example_kg": ex_kg,
            "example_question": ex_question,
            "example_gold_query": ex_gold_query,
            "taxonomy": ex_taxonomy if include_shot_taxonomy else None,
            "user_prompt": ex_user_content,
            "assistant_prompt": ex_assistant_content
        })

        example_messages.append({"role": "user", "content": ex_user_content})
        example_messages.append({"role": "assistant", "content": ex_assistant_content})

    # 2. Sottografo Target, Entity, Property e Tassonomia
    target_entities = entities or []
    target_taxonomy = taxonomy or {}

    if subgraph is None:
        target_ttl_file = f"{kg_name.lower()}_t_subgraph.ttl"
        subgraph_content, _, extracted_ent, extracted_tax = extract_subgraph_and_elements(
            question_text=nl_query,
            kg_name=kg_name,
            output_ttl_path=target_ttl_file
        )
        if not target_entities:
            target_entities = extracted_ent
        if not target_taxonomy:
            target_taxonomy = extracted_tax
    else:
        subgraph_content = _read_if_file(subgraph)
        if not target_taxonomy:
            try:
                _, _, extracted_ent, extracted_tax = extract_subgraph_and_elements(
                    question_text=nl_query,
                    kg_name=kg_name
                )
                if not target_entities:
                    target_entities = extracted_ent
                if not target_taxonomy:
                    target_taxonomy = extracted_tax
            except Exception as e:
                print(f"[ATTENZIONE] Estrazione automatica entità/proprietà/tassonomia per subgraph personalizzato fallita: {e}")

    # 3. Costruzione prompt Few-Shot per il target (con tassonomia e subgraph)
    target_user_prompt = build_target_user_prompt(
        subgraph_content=subgraph_content,
        nl_query=nl_query,
        taxonomy=target_taxonomy,
        entities=target_entities,
        include_taxonomy=include_taxonomy
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(example_messages)
    messages.append({"role": "user", "content": target_user_prompt})

    print(f"\n--- Esecuzione Chiamata Few-Shot v07 (Schema-Agnostic Shots) ID: {call_id} ---")
    if question_index is not None:
        print(f"Question Index Target   : {question_index}")
    print(f"Question Target         : {nl_query}")
    print(f"Numero Esempi           : {len(examples_details)}")
    print(f"Indici Esempi           : {[ex['example_index'] for ex in examples_details]}")
    print(f"Shots Include Taxonomy  : {include_shot_taxonomy} (Schema-Agnostic)")
    print(f"Target Include Taxonomy : {include_taxonomy}")
    print(f"Target Entities         : {target_entities}")
    if num_ctx is not None:
        print(f"Context Length (num_ctx): {num_ctx}")

    result_text = ""
    error_msg = None

    try:
        if _LOADED_MODEL_NAME != model_name or _BACKEND is None:
            ready = load_model(model_name)
            if not ready:
                raise RuntimeError(f"Impossibile inizializzare il modello '{model_name}'.")

        if _BACKEND == "ollama":
            chat_options = {}
            if temperature is not None:
                chat_options["temperature"] = temperature
            if num_ctx is not None:
                chat_options["num_ctx"] = num_ctx

            response = ollama.chat(
                model=model_name,
                messages=messages,
                options=chat_options if chat_options else None
            )
            result_text = response.get("message", {}).get("content", "").strip()
        else:
            # Fallback Transformers
            if hasattr(_TOKENIZER, "apply_chat_template") and _TOKENIZER.chat_template:
                try:
                    formatted_input = _TOKENIZER.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    formatted_input = f"{SYSTEM_PROMPT}\n\n" + "\n\n".join(
                        f"[{m['role'].upper()}]\n{m['content']}" for m in messages[1:]
                    ) + "\n\n[ASSISTANT]\n"
            else:
                formatted_input = f"{SYSTEM_PROMPT}\n\n" + "\n\n".join(
                    f"[{m['role'].upper()}]\n{m['content']}" for m in messages[1:]
                ) + "\n\n[ASSISTANT]\n"

            inputs = _TOKENIZER(formatted_input, return_tensors="pt").to(_MODEL.device)

            generate_kwargs = {
                "max_new_tokens": 512,
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
    "pipeline": "few_shots_v07",
    "question_index": question_index,
    "num_examples": len(examples_details),
    "include_shot_taxonomy": include_shot_taxonomy,
    "include_target_taxonomy": include_taxonomy,
    "examples": [
        {
            "rank": ex["rank"],
            "example_index": ex["example_index"],
            "example_role": ex["example_role"],
            "example_kg": ex["example_kg"],
            "example_question": ex["example_question"],
            "example_gold_query": ex["example_gold_query"],
            "taxonomy": ex.get("taxonomy")
        }
        for ex in examples_details
    ],
    "model": model_name,
    "num_ctx": num_ctx,
    "extracted_entities": target_entities,
    "taxonomy": target_taxonomy if include_taxonomy else None,
    "prompt": {
        "system": SYSTEM_PROMPT,
        "examples": [
            {
                "rank": ex["rank"],
                "example_index": ex["example_index"],
                "example_role": ex["example_role"],
                "user": ex["user_prompt"],
                "assistant": ex["assistant_prompt"]
            }
            for ex in examples_details
        ],
        "user": target_user_prompt
    },
    "result": result_text if not error_msg else f"ERROR: {error_msg}",
    "comment": ""
    }

    append_to_json(record, filepath=output_json_path)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Esecuzione Few-Shot v07 Schema-Agnostic (Shot privi di tassonomia con solo subgraph, Target con tassonomia e subgraph)"
    )
    parser.add_argument("-i", "--index", type=int, default=DEFAULT_QUESTION_INDEX, help=f"Indice della domanda target in dev.json (default: {DEFAULT_QUESTION_INDEX})")
    parser.add_argument("--subgraph", type=str, default=None, help="File del sottografo target (se omesso, estratto con preprocessing)")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help=f"Modello LLM (default: {MODEL_NAME})")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperatura del modello LLM (default: 0.0)")
    parser.add_argument("--include-taxonomy", action="store_true", default=True, help="Include la tassonomia nella domanda target (default: True)")
    parser.add_argument("--no-taxonomy", dest="include_taxonomy", action="store_false", help="Esclude la tassonomia dalla domanda target")
    parser.add_argument("--output-json", type=str, default=OUTPUT_JSON_FILE, help=f"File JSON di output (default: {OUTPUT_JSON_FILE})")
    args = parser.parse_args()

    print(f"=== Avvio Esecuzione few_shots_v07.py Schema-Agnostic (Target Indice: {args.index}, Shots: {DEFAULT_EXAMPLE_INDICES}) ===")

    # 1. Caricamento domanda target da dev.json
    test_query, gold_query, kg_name = load_question_from_dev(args.index)
    print(f"\nKnowledge Graph Target  : {kg_name}")
    print(f"Domanda Target (NL)     : \"{test_query}\"")
    print(f"Query Gold Target       : {gold_query}")

    # 2. Caricamento / verifica modello
    ready = load_model(args.model)

    if ready:
        result = run_few_shot(
            nl_query=test_query,
            subgraph=args.subgraph,
            question_index=args.index,
            kg_name=kg_name,
            model_name=args.model,
            temperature=args.temperature,
            include_taxonomy=args.include_taxonomy,
            include_shot_taxonomy=False,
            output_json_path=args.output_json
        )
        print("\n--- Risultato Ottenuto dal Modello ---")
        print(result["result"])
    else:
        print("[INFO] Il modello non è pronto. Verifica che Ollama sia installato e attivo.")

