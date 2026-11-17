"""
full_pipeline_transformers.py
=============================
Pipeline end-to-end per Text-to-SPARQL su Knowledge Graph con Hugging Face Transformers.

Struttura della classe `FullPipeline`:
1. Configurazione variabili:
   - target_question_index: Indice della domanda target in dev.json
   - generation_type / generation_method: Tipo/modulo di generazione ('few_shot', 'zero_shot', 'cot')
   - model_name: Modello LLM Hugging Face Transformers (es. 'Qwen/Qwen2.5-Coder-7B-Instruct')
   - preprocessing_method: Modulo di preprocessing 
   - postprocessing_method: Modulo di postprocessing 
   - dev_json_path, output_json_path, custom_full_graph_path

2. Sequenza di esecuzione ordinata:
   - Step 1: Preprocessing (estrazione del sottografo dinamico tramite preprocessing_v12)
   - Step 2: Generazione LLM (Few-shot, Zero-Shot o CoT tramite Transformers)
   - Step 3: Postprocessing & Querying (risoluzione URI ed esecuzione sul Full Graph con salvataggio nel file JSON tramite postprocessing_v03)
"""

import argparse
import datetime
import importlib
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

# Assicura che la root del workspace sia nel sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ==============================================================================
# CLASSE PRINCIPALE: FULL PIPELINE (TRANSFORMERS)
# ==============================================================================

class FullPipeline:
    """
    Classe per orchestrare l'intera pipeline Text-to-SPARQL con Transformers:
    Preprocessing -> Generazione (Few-Shot / Zero-Shot / CoT) -> Postprocessing & Querying su Full Graph.
    """

    def __init__(
        self,
        target_question_index: int = 41,
        generation_type: str = "zero_shot",
        generation_method: Optional[Union[str, Any]] = None,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        temperature: float = 0.0,
        num_ctx: Optional[int] = None,
        preprocessing_method: Union[str, Any] = "preprocessing_v12",
        postprocessing_method: Union[str, Any] = "postprocessing_v03",
        dev_json_path: Union[str, Path] = ROOT_DIR / "spider4sparql" / "dev.json",
        output_json_path: Optional[Union[str, Path]] = "zero_shot_v04.json",
        custom_full_graph_path: Optional[Union[str, Path]] = None,
        top_k_triples: int = 40,
        update_json_file: bool = True,
        verbose: bool = True,
        include_taxonomy: Optional[bool] = None
    ):
        """
        Inizializza la pipeline con tutte le variabili di configurazione.
        """
        self.target_question_index = target_question_index

        gen_clean = generation_type.strip().lower()
        if gen_clean.startswith("few"):
            self.generation_type = "few_shot"
        elif gen_clean.startswith("zero"):
            self.generation_type = "zero_shot"
        elif gen_clean.startswith("cot") or "thought" in gen_clean:
            self.generation_type = "cot"
        else:
            self.generation_type = gen_clean

        self.generation_method = generation_method or (
            "few_shot_v07" if self.generation_type == "few_shot" else (
                "cot_v13" if self.generation_type == "cot" else "zero_shot_v04"
                )
            )
        self.generation_mod = self._resolve_generation_module(self.generation_method)

        self.model_name = model_name
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.preprocessing_method = preprocessing_method
        self.preprocessing_mod = self._resolve_preprocessing_module(preprocessing_method)
        self.postprocessing_method = postprocessing_method
        self.postprocessing_mod = self._resolve_postprocessing_module(postprocessing_method)
        self.include_taxonomy = include_taxonomy

        if hasattr(self.generation_mod, "set_preprocessing_module"):
            try:
                self.generation_mod.set_preprocessing_module(self.preprocessing_mod)
            except Exception:
                pass
        
        self.dev_json_path = Path(dev_json_path)
        if not self.dev_json_path.is_absolute():
            self.dev_json_path = ROOT_DIR / self.dev_json_path

        # File JSON di output predefinito
        if output_json_path:
            self.output_json_path = Path(output_json_path)
        else:
            output_name = output_json_path or f"{self.generation_method}.json"
            self.output_json_path = Path(output_name)

        if not self.output_json_path.is_absolute():
            self.output_json_path = ROOT_DIR / self.output_json_path

        self.custom_full_graph_path = Path(custom_full_graph_path) if custom_full_graph_path else None
        self.top_k_triples = top_k_triples
        self.update_json_file = update_json_file
        self.verbose = verbose

        # Variabili di stato interne
        self.target_question: str = ""
        self.target_gold_query: str = ""
        self.kg_name: str = "concert_singer"
        self.target_subgraph_path: Optional[Path] = None
        self.target_taxonomy: Optional[Dict[str, Any]] = None
        self.target_entities: List[str] = []
        self.generation_record: Optional[Dict[str, Any]] = None
        self.execution_record: Optional[Dict[str, Any]] = None

    def _resolve_preprocessing_module(self, method: Union[str, Any]) -> Any:
        if hasattr(method, "run_pipeline_for_ontology"):
            return method
        if isinstance(method, str):
            return importlib.import_module(method.strip())
        raise ValueError(f"Metodo di preprocessing non valido: {method}")

    def _resolve_generation_module(self, method: Union[str, Any]) -> Any:
        if hasattr(method, "run_few_shot") or hasattr(method, "run_zero_shot") or hasattr(method, "run_cot"):
            return method
        if isinstance(method, str):
            return importlib.import_module(method.strip())
        raise ValueError(f"Metodo di generazione non valido: {method}")

    def _resolve_postprocessing_module(self, method: Union[str, Any]) -> Any:
        if hasattr(method, "run_pipeline"):
            return method
        if isinstance(method, str):
            return importlib.import_module(method.strip())
        raise ValueError(f"Metodo di postprocessing non valido: {method}")

    # --------------------------------------------------------------------------
    # CARICAMENTO DATI DA DEV.JSON
    # --------------------------------------------------------------------------
    def load_questions_info(self):
        """Carica la domanda target da dev.json."""
        if not self.dev_json_path.exists():
            raise FileNotFoundError(f"File dev.json non trovato in: {self.dev_json_path}")

        with open(self.dev_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Domanda Target
        if self.target_question_index < 0 or self.target_question_index >= len(data):
            raise IndexError(f"Indice target {self.target_question_index} non valido per dev.json (lunghezza: {len(data)})")
        target_entry = data[self.target_question_index]
        self.target_question = target_entry.get("question", "")
        self.target_gold_query = target_entry.get("query", "")
        self.kg_name = target_entry.get("kg_name", "concert_singer")

        if self.verbose:
            print("=" * 80)
            print("            FULL PIPELINE TEXT-TO-SPARQL (TRANSFORMERS)")
            print("=" * 80)
            print(f"  - Tipo Generazione       : {self.generation_type.upper()} ({getattr(self.generation_mod, '__name__', str(self.generation_method))})")
            print(f"  - Modello LLM            : {self.model_name}")
            prep_name = getattr(self.preprocessing_mod, "__name__", str(self.preprocessing_method))
            post_name = getattr(self.postprocessing_mod, "__name__", str(self.postprocessing_method))
            print(f"  - Metodo Preprocessing   : {prep_name}")
            print(f"  - Metodo Postprocessing  : {post_name}")
            print(f"  - Knowledge Graph        : {self.kg_name}")
            print(f"  - Indice Domanda Target  : {self.target_question_index} -> \"{self.target_question}\"")
            print(f"  - File JSON Risultati    : {self.output_json_path.name}")
            print("=" * 80 + "\n")

    # --------------------------------------------------------------------------
    # STEP 1: PREPROCESSING (Estrazione Sottografi)
    # --------------------------------------------------------------------------
    def run_preprocessing(self) -> Path:
        """
        Esegue la pipeline del modulo di preprocessing configurato:
        Estrae il sottografo RDF per la domanda target.
        """
        if self.preprocessing_mod is None:
            raise ImportError(f"Modulo di preprocessing '{self.preprocessing_method}' non disponibile.")

        print("\n" + "#" * 80)
        print("  [STEP 1 / 3] PREPROCESSING - ESTRAZIONE SOTTOGRAFO")
        print("#" * 80)

        # Risoluzione percorso ontologia originale
        if self.custom_full_graph_path:
            onto_full_path = str(self.custom_full_graph_path)
        else:
            onto_full_path, _, _ = self.preprocessing_mod.get_concert_singer_dataset(str(self.dev_json_path.parent))

        dense_dir = f"dense_space_{self.kg_name}" if (ROOT_DIR / f"dense_space_{self.kg_name}").exists() else "dense_space_cs"

        # 1. Estrazione sottografo per domanda target (t_subgraph)
        t_ttl_file = ROOT_DIR / f"{self.kg_name.lower()}_t_subgraph.ttl"
        print(f"\n--> Estrazione sottografo Target (t_subgraph) (Indice {self.target_question_index})...")
        res_target = self.preprocessing_mod.run_pipeline_for_ontology(
            onto_name=self.kg_name,
            onto_path=onto_full_path,
            query_test=self.target_question,
            output_dir=dense_dir,
            output_subgraph_path=str(t_ttl_file)
        )
        self.target_subgraph_path = Path(res_target.get("subgraph_ttl_path", str(t_ttl_file)))
        self.target_taxonomy = res_target.get("taxonomy")
        self.target_entities = list(res_target.get("selected_classes", []))
        print(f"[OK] Sottografo Target salvato in: '{self.target_subgraph_path}' (Triple: {res_target.get('triples_count', len(res_target.get('subgraph_triples', [])))})")

        return self.target_subgraph_path

    # --------------------------------------------------------------------------
    # STEP 2: GENERAZIONE LLM (Few-Shot Statico / Zero-Shot / CoT)
    # --------------------------------------------------------------------------
    def run_generation(self) -> Dict[str, Any]:
        """
        Esegue la generazione del prompt e l'inferenza LLM tramite il modulo prescelto.
        """
        print("\n" + "#" * 80)
        print(f"  [STEP 2 / 3] GENERAZIONE LLM TRANSFORMERS ({self.generation_type.upper()})")
        print("#" * 80)

        # Verifica e caricamento modello Transformers
        if hasattr(self.generation_mod, "load_model"):
            self.generation_mod.load_model(self.model_name)

        if self.generation_type == "few_shot":
            if hasattr(self.generation_mod, "run_few_shot"):
                import inspect
                sig = inspect.signature(self.generation_mod.run_few_shot)
                kwargs = {
                    "nl_query": self.target_question,
                    "subgraph": str(self.target_subgraph_path),
                    "question_index": self.target_question_index,
                    "kg_name": self.kg_name,
                    "model_name": self.model_name,
                    "temperature": self.temperature,
                    "output_json_path": str(self.output_json_path),
                    "dev_path": str(self.dev_json_path)
                }
                if "num_ctx" in sig.parameters and self.num_ctx is not None:
                    kwargs["num_ctx"] = self.num_ctx
                if "taxonomy" in sig.parameters and hasattr(self, "target_taxonomy") and self.target_taxonomy is not None:
                    kwargs["taxonomy"] = self.target_taxonomy
                if "entities" in sig.parameters and hasattr(self, "target_entities") and self.target_entities is not None:
                    kwargs["entities"] = self.target_entities
                if "include_taxonomy" in sig.parameters and self.include_taxonomy is not None:
                    kwargs["include_taxonomy"] = self.include_taxonomy
                self.generation_record = self.generation_mod.run_few_shot(**kwargs)
            else:
                raise AttributeError(f"Modulo '{self.generation_mod.__name__}' non possiede 'run_few_shot'")

        elif self.generation_type == "zero_shot":
            if hasattr(self.generation_mod, "run_zero_shot"):
                import inspect
                sig = inspect.signature(self.generation_mod.run_zero_shot)
                kwargs = {
                    "nl_query": self.target_question,
                    "subgraph": str(self.target_subgraph_path),
                    "question_index": self.target_question_index,
                    "model_name": self.model_name,
                    "output_json_path": str(self.output_json_path)
                }
                if "temperature" in sig.parameters:
                    kwargs["temperature"] = self.temperature
                if "num_ctx" in sig.parameters and self.num_ctx is not None:
                    kwargs["num_ctx"] = self.num_ctx
                if "taxonomy" in sig.parameters and hasattr(self, "target_taxonomy") and self.target_taxonomy is not None:
                    kwargs["taxonomy"] = self.target_taxonomy
                if "include_taxonomy" in sig.parameters and self.include_taxonomy is not None:
                    kwargs["include_taxonomy"] = self.include_taxonomy
                self.generation_record = self.generation_mod.run_zero_shot(**kwargs)
            else:
                raise AttributeError(f"Modulo '{self.generation_mod.__name__}' non possiede 'run_zero_shot'")

        elif self.generation_type.startswith("cot") or self.generation_type in ("cot", "chain_of_thought"):
            if hasattr(self.generation_mod, "run_cot"):
                kwargs = {
                    "nl_query": self.target_question,
                    "subgraph": str(self.target_subgraph_path),
                    "question_index": self.target_question_index,
                    "model_name": self.model_name,
                    "output_json_path": str(self.output_json_path)
                }
                import inspect
                sig = inspect.signature(self.generation_mod.run_cot)
                if "temperature" in sig.parameters:
                    kwargs["temperature"] = self.temperature
                if "num_ctx" in sig.parameters and self.num_ctx is not None:
                    kwargs["num_ctx"] = self.num_ctx
                if "taxonomy" in sig.parameters and hasattr(self, "target_taxonomy") and self.target_taxonomy is not None:
                    kwargs["taxonomy"] = self.target_taxonomy
                if "include_taxonomy" in sig.parameters and self.include_taxonomy is not None:
                    kwargs["include_taxonomy"] = self.include_taxonomy
                if "kg_name" in sig.parameters:
                    kwargs["kg_name"] = self.kg_name
                self.generation_record = self.generation_mod.run_cot(**kwargs)
            else:
                raise AttributeError(f"Modulo '{self.generation_mod.__name__}' non possiede 'run_cot'")

        else:
            raise ValueError(f"Tipo di generazione non supportato: '{self.generation_type}'. Usa 'few_shot', 'zero_shot' o 'cot'.")

        print(f"\n--> Query Generata (ID: {self.generation_record.get('id')}):")
        print(self.generation_record.get("result", "(VUOTA)"))
        # Inietta sempre le entità estratte nel record e aggiorna il file JSON
        if self.generation_record is not None:
            self.generation_record["extracted_entities"] = self.target_entities
            
            # Aggiorna il record appena appeso nel file output JSON
            if self.output_json_path.exists():
                try:
                    with open(self.output_json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data:
                        # Aggiorna l'ultimo record aggiunto
                        data[-1]["extracted_entities"] = self.target_entities
                        with open(self.output_json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[WARN] Impossibile aggiornare extracted_entities nel file JSON: {e}")
        return self.generation_record

    # --------------------------------------------------------------------------
    # STEP 3: POSTPROCESSING & QUERYING SU FULL GRAPH
    # --------------------------------------------------------------------------
    def run_postprocessing_and_querying(self) -> Dict[str, Any]:
        """
        Esegue il postprocessing della query generata, la esecuzione sul Full Graph,
        l'esecuzione della query gold e il salvataggio dei risultati nel file JSON.
        """
        if self.postprocessing_mod is None:
            raise ImportError(f"Modulo di postprocessing '{self.postprocessing_method}' non disponibile.")

        print("\n" + "#" * 80)
        print("  [STEP 3 / 3] POSTPROCESSING & QUERYING SU FULL GRAPH")
        print("#" * 80)

        gen_id = self.generation_record.get("id") if self.generation_record else None

        updated_records = self.postprocessing_mod.run_pipeline(
            json_path=self.output_json_path,
            target_id=gen_id,
            dev_path=self.dev_json_path,
            custom_full_graph=self.custom_full_graph_path,
            update_file=self.update_json_file,
            verbose=self.verbose
        )

        if updated_records:
            self.execution_record = updated_records[-1]
        return self.execution_record or {}

    # --------------------------------------------------------------------------
    # ESECUZIONE COMPLETA DELLA PIPELINE
    # --------------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Esegue tutti i 3 step in ordine sequenziale."""
        start_time = datetime.datetime.now()

        # Inizializzazione dati
        self.load_questions_info()

        # Step 1: Preprocessing
        self.run_preprocessing()

        # Step 2: Generazione LLM
        self.run_generation()

        # Step 3: Postprocessing & Querying
        final_record = self.run_postprocessing_and_querying()

        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 80)
        print(f"      PIPELINE COMPLETATA CON SUCCESSO IN {elapsed:.2f} SECONDI!")
        print("=" * 80)
        # ----------------------------------------------------------------------
        # SALVATAGGIO AUTOMATICO DI PROCESSING_TIME NEL RECORD E NEL JSON
        # ----------------------------------------------------------------------
        elapsed_sec = round(elapsed, 2)
        if final_record:
            final_record["processing_time"] = elapsed_sec
        if self.output_json_path.exists():
            try:
                with open(self.output_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data:
                    target_id = self.generation_record.get("id") if self.generation_record else None
                    # Cerca il record corrispondente per ID oppure aggiorna l'ultimo
                    updated = False
                    for item in reversed(data):
                        if target_id is not None and item.get("id") == target_id:
                            item["processing_time"] = elapsed_sec
                            updated = True
                            break
                    if not updated and len(data) > 0:
                        data[-1]["processing_time"] = elapsed_sec
                    with open(self.output_json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[WARN] Impossibile aggiornare processing_time nel file JSON: {e}")
        return final_record

def update_method_comparison_file(
    records: List[Dict[str, Any]],
    method_key: str,
    model_name: str,
    generation_type: str,
    generation_method: str,
    preprocessing_method: str,
    postprocessing_method: str,
    dev_json_path: Path,
    comp_file: Path = ROOT_DIR / "method_comparison.json"
):
    """Aggiorna automaticamente method_comparison.json con le metriche aggregate del batch."""
    total = len(records)
    if total == 0:
        return

    correct_list = []
    mismatch_list = []
    error_list = []
    details = []

    dev_data = []
    if dev_json_path.exists():
        try:
            with open(dev_json_path, "r", encoding="utf-8") as f:
                dev_data = json.load(f)
        except Exception:
            dev_data = []

    for rec in records:
        q_idx = rec.get("question_index")
        exec_res = rec.get("execution_result", {})
        exec_ok = exec_res.get("execution_success", False)
        is_match = exec_res.get("is_correct", False)
        err = exec_res.get("error")

        q_text = dev_data[q_idx].get("question", "") if (q_idx is not None and 0 <= q_idx < len(dev_data)) else ""

        if is_match:
            status = "CORRECT"
            correct_list.append(q_idx)
        elif exec_ok:
            status = "RESULT_MISMATCH"
            mismatch_list.append(q_idx)
        else:
            status = "EXECUTION_ERROR"
            error_list.append(q_idx)

        details.append({
            "id": rec.get("id"),
            "question_index": q_idx,
            "question": q_text,
            "status": status,
            "is_correct": is_match,
            "execution_success": exec_ok,
            "error": err,
            "predicted_rows_count": exec_res.get("predicted_rows_count", 0),
            "gold_rows_count": exec_res.get("gold_rows_count", 0),
            "cleaned_query": exec_res.get("cleaned_query"),
            "resolved_query": exec_res.get("resolved_query"),
            "processing_time": rec.get("processing_time")
        })

    details.sort(key=lambda x: (x.get("question_index") if x.get("question_index") is not None else 9999))
    acc_pct = round((len(correct_list) / total) * 100, 2) if total > 0 else 0.0
    exec_pct = round(((len(correct_list) + len(mismatch_list)) / total) * 100, 2) if total > 0 else 0.0

    method_data = {
        "method_name": method_key,
        "description": f"{method_key.upper()} ({generation_method}) con {model_name}, PP: {preprocessing_method}, POST: {postprocessing_method}",
        "configuration": {
            "model": model_name,
            "generation_strategy": generation_method,
            "generation_type": generation_type,
            "preprocessing_module": preprocessing_method,
            "postprocessing_module": postprocessing_method,
            "dataset": str(dev_json_path.name),
            "total_evaluated_queries": total,
            "evaluated_indices": sorted([r.get("question_index") for r in records if r.get("question_index") is not None])
        },
        "metrics": {
            "total_queries": total,
            "correct_queries": len(correct_list),
            "accuracy_percentage": acc_pct,
            "executed_successfully": len(correct_list) + len(mismatch_list),
            "execution_success_percentage": exec_pct,
            "result_mismatches": len(mismatch_list),
            "result_mismatch_percentage": round((len(mismatch_list) / total) * 100, 2) if total > 0 else 0.0,
            "execution_errors": len(error_list),
            "execution_error_percentage": round((len(error_list) / total) * 100, 2) if total > 0 else 0.0
        },
        "breakdown_by_index": {
            "correct_indices": sorted(correct_list),
            "mismatch_indices": sorted(mismatch_list),
            "error_indices": sorted(error_list)
        },
        "detailed_results": details
    }

    comparison_root = {"benchmark": "Spider4SPARQL Dev Set", "last_updated": datetime.datetime.now().isoformat(), "methods": {}}
    if comp_file.exists():
        try:
            with open(comp_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, dict):
                    comparison_root = content
                    if "methods" not in comparison_root:
                        comparison_root["methods"] = {}
        except Exception as e:
            print(f"[WARN] Errore lettura {comp_file.name}: {e}")

    comparison_root["last_updated"] = datetime.datetime.now().isoformat()
    comparison_root["methods"][method_key] = method_data

    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comparison_root, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f" [OK] Aggiornato {comp_file.name} con il metodo '{method_key}'!")
    print(f"      Accuratezza (Gold Match) : {acc_pct}% ({len(correct_list)}/{total})")
    print(f"      Successo Esecuzione      : {exec_pct}% ({len(correct_list) + len(mismatch_list)}/{total})")
    print("=" * 80)
    
# ==============================================================================
# FUNZIONE DI ESECUZIONE BATCH SU LISTA DI INDICI / KNOWLEDGE GRAPH
# ==============================================================================

def run_batch(
    indices: List[int],
    generation_type: str = "zero_shot",
    generation_method: str = "zero_shot_v04",
    model_name: str = "qwen2.5-coder:7b",
    temperature: float = 0.0,
    num_ctx: Optional[int] = None,
    preprocessing_method: str = "preprocessing_v12",
    postprocessing_method: str = "postprocessing_v03",
    output_json_path: str = "batch_results.json",
    dev_json_path: str = "spider4sparql/dev.json",
    clean_output: bool = True,
    method_key: Optional[str] = None,
    update_comparison: bool = True
) -> List[Dict[str, Any]]:
    out_path = Path(output_json_path)
    if not out_path.is_absolute():
        out_path = ROOT_DIR / out_path

    if clean_output:
        if out_path.exists():
            os.remove(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    total = len(indices)
    print("=" * 80)
    print(f" BATCH RUN: {generation_type.upper()} ({generation_method}) | Modello: {model_name}")
    print(f" Totale domande da fare : {total} indici | Output: {out_path.name}")
    print("=" * 80 + "\n")

    results = []
    start_all = time.time()

    for i, idx in enumerate(indices, 1):
        t0 = time.time()
        print(f"\n>>>>>> [{i}/{total}] ELABORAZIONE INDICE DEV.JSON: {idx} <<<<<<")
        try:
            pipeline = FullPipeline(
                target_question_index=idx,
                generation_type=generation_type,
                generation_method=generation_method,
                model_name=model_name,
                temperature=temperature,
                num_ctx=num_ctx,
                preprocessing_method=preprocessing_method,
                postprocessing_method=postprocessing_method,
                dev_json_path=dev_json_path,
                output_json_path=out_path,
                verbose=False
            )
            rec = pipeline.run()
            results.append(rec)
            dt = time.time() - t0
            print(f"[OK] Completato Indice {idx} in {dt:.2f}s (Progresso: {i}/{total})")

        except Exception as e:
            dt = time.time() - t0
            print(f"[ERRORE GRAVE] Indice {idx} fallito dopo {dt:.2f}s: {e}")
            # Fallback: salva l'errore nel JSON e prosegue al successivo
            err_rec = {"question_index": idx, "error": str(e), "execution_success": False, "is_correct": False}
            results.append(err_rec)
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data.append(err_rec)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
            continue

    elapsed_all = time.time() - start_all
    print("\n" + "=" * 80)
    print(f" BATCH RUN COMPLETATO: {len(results)} record in {elapsed_all:.2f}s ({elapsed_all/60:.2f} minuti)")
    print("=" * 80)

    # Aggiornamento finale automatico di method_comparison.json
    if update_comparison:
        dev_file = Path(dev_json_path)
        if not dev_file.is_absolute():
            dev_file = ROOT_DIR / dev_file
        effective_key = method_key or generation_method
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                final_saved_records = json.load(f)
            update_method_comparison_file(
                records=final_saved_records,
                method_key=effective_key,
                model_name=model_name,
                generation_type=generation_type,
                generation_method=generation_method,
                preprocessing_method=preprocessing_method,
                postprocessing_method=postprocessing_method,
                dev_json_path=dev_file
            )
        except Exception as e:
            print(f"[WARN] Impossibile aggiornare method_comparison.json: {e}")

    return results


# ==============================================================================
# MAIN DI ESEMPIO / ESECUZIONE DIRETTA
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Pipeline Text-to-SPARQL con Preprocessing, Generazione Transformers e Postprocessing")
    parser.add_argument("-i", "--index", type=int, default=None, help="Indice singolo della domanda target in dev.json")
    parser.add_argument("--kg", type=str, default=None, help="Nome del Knowledge Graph da eseguire in batch per tutti i suoi indici (es. concert_singer)")
    parser.add_argument("--gen-type", type=str, default="zero_shot", help="Tipo di generazione ('zero_shot', 'few_shot', 'cot')")
    parser.add_argument("--gen-method", type=str, default="zero_shot_v04", help="Modulo di generazione ('zero_shot_v04', 'few_shot_v07', 'cot_v13', ecc.)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Modello Hugging Face Transformers (default: Qwen/Qwen2.5-Coder-7B-Instruct)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperatura del modello LLM (default: 0.0)")
    parser.add_argument("--num-ctx", type=int, default=None, help="Context length (num_ctx) (es. 14336, 16384)")
    parser.add_argument("--prep-method", type=str, default="preprocessing_v12", help="Modulo di preprocessing (default: preprocessing_v12)")
    parser.add_argument("--post-method", type=str, default="postprocessing_v03", help="Modulo di postprocessing (default: postprocessing_v03)")
    parser.add_argument("--output-json", type=str, default="zero_shot_v04.json", help="File JSON di output (default: zero_shot_v04.json)")
    parser.add_argument("--dev-path", type=str, default="spider4sparql/dev.json", help="Percorso di dev.json")
    parser.add_argument("--no-clean", action="store_true", help="Non ripulire il file output prima del batch")
    args = parser.parse_args()

    # Se richiesto batch per KG (es. concert_singer)
    if args.kg:
        dev_file = Path(args.dev_path)
        if not dev_file.is_absolute():
            dev_file = ROOT_DIR / dev_file
        with open(dev_file, "r", encoding="utf-8") as f:
            dev_data = json.load(f)
        target_indices = [idx for idx, item in enumerate(dev_data) if item.get("kg_name") == args.kg]
        print(f"--> Trovate {len(target_indices)} domande per il KG '{args.kg}' in dev.json: {target_indices}")
        run_batch(
            indices=target_indices,
            generation_type=args.gen_type,
            generation_method=args.gen_method,
            model_name=args.model,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            preprocessing_method=args.prep_method,
            postprocessing_method=args.post_method,
            output_json_path=args.output_json,
            dev_json_path=args.dev_path,
            clean_output=not args.no_clean
        )
    elif args.index is not None:
        pipeline = FullPipeline(
            target_question_index=args.index,
            generation_type=args.gen_type,
            generation_method=args.gen_method,
            model_name=args.model,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            preprocessing_method=args.prep_method,
            postprocessing_method=args.post_method,
            dev_json_path=args.dev_path,
            output_json_path=args.output_json,
            verbose=True
        )
        pipeline.run()
    else:
        # Default: esegui singolo indice 41
        pipeline = FullPipeline(
            target_question_index=41,
            generation_type="zero_shot",
            generation_method="zero_shot_v04",
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            num_ctx=args.num_ctx,
            preprocessing_method="preprocessing_v12",
            postprocessing_method="postprocessing_v03",
            output_json_path="zero_shot_v04.json",
            verbose=True
        )
        pipeline.run()
