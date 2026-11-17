import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

try:
    import ollama
except ImportError:
    ollama = None

try:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
except ImportError:
    torch = None
    transformers = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except Exception:
    pass


MODEL_NAME = "qwen2.5-coder:7b"
OUTPUT_JSON_FILE = "zero_shot_v04.json"
DEV_JSON_PATH = os.path.join(os.path.dirname(__file__), "spider4sparql", "dev.json")
DEFAULT_QUESTION_INDEX = 41

# Cache globale per modello e tokenizer
_MODEL = None
_TOKENIZER = None
_LOADED_MODEL_NAME = None
_BACKEND = None


def load_question_from_dev(
    index: int = DEFAULT_QUESTION_INDEX,
    dev_path: str = DEV_JSON_PATH
) -> Tuple[str, str, str]:
    """
    Carica la domanda e la query gold dal file dev.json all'indice specificato (0-based).
    Restituisce (question, gold_query, kg_name).
    """
    if not os.path.exists(dev_path):
        alt_path = os.path.join("spider4sparql", "dev.json")
        if os.path.exists(alt_path):
            dev_path = alt_path
        else:
            raise FileNotFoundError(f"File dev.json non trovato in '{dev_path}'")
            
    with open(dev_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if index < 0 or index >= len(data):
        raise IndexError(f"Indice {index} non valido per dev.json (totale elementi: {len(data)})")
    entry = data[index]
    return entry["question"], entry.get("query", ""), entry.get("kg_name", "")


SYSTEM_PROMPT = (
    "you're an expert in ontologies, sparql querying and text-to sparql. "
    "given a question in natural language and ontology triples translate the question into a valid sparql query. "
    "Use only the triples and the resources in the ontology subgraph provided. "
    "Answer only with the sparql query enclosed between '''."
    "Use CURIE notation as in the provided ontology."
    "Do not specify prefixes"
)

SYSTEM_PROMPT_WITH_TAXONOMY = (
    "you're an expert in ontologies, sparql querying and text-to sparql. "
    "given a question in natural language, ontology triples, and hierarchical taxonomy translate the question into a valid sparql query. "
    "Use only the triples and the resources in the ontology subgraph and the taxonomy provided. "
    "Answer only with the sparql query enclosed between '''."
    "Use CURIE notation as in the provided ontology and taxonomy."
    "Do not specify prefixes"
)


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


def run_zero_shot(
    nl_query: str,
    subgraph: str,
    call_id: Optional[Union[str, int]] = None,
    question_index: Optional[int] = None,
    model_name: str = MODEL_NAME,
    output_json_path: str = OUTPUT_JSON_FILE,
    temperature: float = 0.0,
    taxonomy: Optional[Dict[str, Any]] = None,
    num_ctx: int = 4096
) -> Dict[str, Any]:
    """
    Esegue una chiamata zero-shot a Transformers/Ollama per tradurre una query in linguaggio naturale
    in SPARQL dato un sottografo di triple ontologiche (stringa o percorso a file .ttl/.txt).
    Salva la chiamata nel file JSON con ID incrementale automatico se non specificato.
    """
    global _MODEL, _TOKENIZER

    if call_id is None:
        call_id = get_next_id(output_json_path)

    # Se subgraph è un percorso file esistente, ne legge il contenuto
    subgraph_content = subgraph
    if isinstance(subgraph, str) and os.path.isfile(subgraph):
        print(f"--> Lettura del file subgraph da: '{subgraph}'")
        with open(subgraph, "r", encoding="utf-8") as f:
            subgraph_content = f.read().strip()

    active_system_prompt = SYSTEM_PROMPT_WITH_TAXONOMY if taxonomy else SYSTEM_PROMPT

    if taxonomy:
        tax_str = json.dumps(taxonomy, indent=2, ensure_ascii=False)
        user_prompt = (
            f"ontology subgraph:\n{subgraph_content}\n\n"
            f"Taxonomy (Hierarchical Format):\n{tax_str}\n\n"
            f"question: {nl_query}, query SPARQL:"
        )
    else:
        user_prompt = f"ontology subgraph: {subgraph_content}, question: {nl_query}, query SPARQL:"

    messages = [
        {"role": "system", "content": active_system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(f"\n--- Esecuzione Chiamata ID: {call_id} ---")
    if question_index is not None:
        print(f"Question Index: {question_index}")
    print(f"Question: {nl_query}")
    if taxonomy:
        print(f"Taxonomy: Inclusa ({len(taxonomy.get('subclasses', []))} sottoclassi di livello 1)")
    print(f"Num Ctx: {num_ctx}")

    result_text = ""
    error_msg = None

    try:
        if _LOADED_MODEL_NAME != model_name or _BACKEND is None:
            ready = load_model(model_name)
            if not ready:
                raise RuntimeError(f"Impossibile inizializzare il modello '{model_name}'.")

        if _BACKEND == "ollama":
            response = ollama.chat(
                model=model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "num_predict": 512
                }
            )
            result_text = response.get("message", {}).get("content", "").strip()
        else:
            # Formattazione del prompt con il chat template se disponibile
            if hasattr(_TOKENIZER, "apply_chat_template") and _TOKENIZER.chat_template:
                try:
                    formatted_input = _TOKENIZER.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    # Fallback se il template non supporta il ruolo 'system'
                    formatted_input = f"{active_system_prompt}\n\n{user_prompt}\n"
            else:
                formatted_input = f"{active_system_prompt}\n\n{user_prompt}\n"

            inputs = _TOKENIZER(formatted_input, return_tensors="pt").to(_MODEL.device)

            with torch.no_grad():
                outputs = _MODEL.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=_TOKENIZER.eos_token_id if _TOKENIZER.eos_token_id is not None else _TOKENIZER.pad_token_id
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            result_text = _TOKENIZER.decode(generated_tokens, skip_special_tokens=True).strip()

    except Exception as e:
        error_msg = str(e)
        print(f"[ERRORE] Chiamata a Transformers fallita: {e}")

    record = {
    "id": call_id,
    "pipeline": "zero_shot_v04",
    "question_index": question_index,
    "timestamp": datetime.now().isoformat(),
    "model": model_name,
    "num_ctx": num_ctx,
    "prompt": {
        "system": active_system_prompt,
        "user": user_prompt
    },
    "result": result_text if not error_msg else f"ERROR: {error_msg}",
    "comment": ""
    }

    append_to_json(record, filepath=output_json_path)
    return record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Esecuzione Zero-Shot Text-to-SPARQL con Transformers/Ollama da dev.json")
    parser.add_argument("-i", "--index", type=int, default=DEFAULT_QUESTION_INDEX, help=f"Indice della domanda in dev.json (default: {DEFAULT_QUESTION_INDEX})")
    parser.add_argument("--subgraph", type=str, default="concert_singer_subgraph.ttl", help="File o percorso del sottografo RDF")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help=f"Modello Transformers/Ollama (default: {MODEL_NAME})")
    args = parser.parse_args()

    print(f"=== Avvio Esecuzione zero_shot_v04.py (Domanda Indice: {args.index}) ===")
    
    # 1. Caricamento domanda da dev.json tramite indice
    test_query, gold_query, kg_name = load_question_from_dev(args.index)
    print(f"Knowledge Graph : {kg_name}")
    print(f"Domanda (NL)    : \"{test_query}\"")
    print(f"Query Gold      : {gold_query}")

    # 2. Caricamento / verifica modello Transformers/Ollama
    ready = load_model(args.model)
    
    if ready:
        result = run_zero_shot(
            nl_query=test_query,
            subgraph=args.subgraph,
            question_index=args.index,
            model_name=args.model
        )
        print("\n--- Risultato Ottenuto dal Modello ---")
        print(result["result"])
    else:
        print("[INFO] Il modello non è pronto. Verifica che transformers e torch siano installati e che il modello sia accessibile.")

