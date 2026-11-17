# Text-to-SPARQL

Questo progetto implementa una pipeline per il task **Text-to-SPARQL**: data una domanda in linguaggio naturale e un'ontologia, il sistema mira a generare la query SPARQL corretta. La pipeline combina preprocessing dell’ontologia e della domanda, generazione con LLM (zero-shot, few-shot o chain-of-thought) e valutazione del risultato.

## Struttura della repository

- **Pipeline orchestratrici** — `full_pipeline_ollama.py` e `full_pipeline_transformers.py` coordinano preprocessing, generazione e postprocessing attraverso backend alternativi: il primo su Ollama, il secondo su Hugging Face Transformers.
- **Strategie di generazione** — `zero_shot_v04.py`, `few_shot_v07.py` e `cot_v13.py` implementano rispettivamente i metodi zero-shot, few-shot e Chain-of-Thought.
- **Preprocessing e postprocessing** — `preprocessing_v12.py` prepara il sottografo e la tassonomia che faranno da contesto per la domanda; `postprocessing_v03.py` elabora la query generata, la esegue sull'ontologia target e valuta la corrispondenza dei risultati, infine salva i record in un file .json di monitoraggio del singolo metodo.
- **Esecuzioni batch** — `run_batch_zero_shot.py`, `run_batch_few_shot.py` e `run_batch_cot.py` impostano i parametri della pipeline e lanciano le tre strategie su un batch di domande, infine salvano i metadati dei risultati nel file di monitoraggio di tutti gli esperimenti `method_comparison.json`.
- **Record e monitoraggio** — i JSON `zs4_*.json`, `fs7_*.json` e `cot13_*.json` raccolgono i risultati zero-shot, few-shot e CoT per i diversi modelli. `global_comparison.json` e `method_comparison.json` contengono confronti tra metodi e modelli. Per un confronto tra i metodi selezionati per la misurazione dei risultati si consulti il primo, per una visione complessiva di tutti i metodi sperimentati dall'inizio del progetto si guardi il secondo. 
- **Confronto e valutazione** — `execute_mcnemar.py`, `execute_plot_error_histograms.py` eseguono le pipeline `mcnemar_test.py` - per il calcolo del test di McNemar sulle differenze tra i valori di Execution Accuracy (EA), Execution Success e Mismatch raggiunti da una coppia di metodi sugli stessi ID - e `plot_error_histograms.py` - per l'error analysis tra due metodi -. I prodotti dei due moduli sono conservati rispettivamente nei file `inner_comparison_*.json` / `cross_comparison_*.json` presenti in `metrics/` e nei file PNG raccolti in `metrics/error_analysis`. I valori percentuali di EA, Execution Success e Mismatch sono invece registrati in `metrics/EA_metrics.xlsx`.
- **Dati e contesto del grafo** — i file `.ttl` descrivono l’ontologia e il sottografo RDF; i file `concert_singer_{entities,properties,prefixes}.txt` contengono informazioni ausiliarie usate dal preprocessing.
- **Benchmark Spider4SPARQL** — la cartella `spider4sparql/` contiene `dev.json`, l’ontologia `concert_singer.ttl`, la cache della tassonomia e l’archivio ZIP delle ontologie.

## Utilizzo

Gli esempi seguenti eseguono la pipeline con Ollama e il modello `qwen2.5-coder:7b`. Assicurati che Ollama e il modello siano disponibili e che le dipendenze Python richieste dagli script siano installate.

### Da CLI

Dalla cartella principale del progetto, esegui una domanda del file `spider4sparql/dev.json` (l’indice è a base zero):

```bash
python full_pipeline_ollama.py --index 41 --gen-type zero_shot --gen-method zero_shot_v04 --model qwen2.5-coder:7b --prep-method preprocessing_v12 --post-method postprocessing_v03
```

Per processare tutte le domande associate a un'ontologia, si può usare `--kg`, ad esempio `--kg concert_singer`.

### Da modulo Python

La classe `FullPipeline` è importabile e configurabile direttamente:

```python
from full_pipeline_ollama import FullPipeline

pipeline = FullPipeline(
    target_question_index=41,
    generation_type="zero_shot",
    generation_method="zero_shot_v04",
    model_name="qwen2.5-coder:7b",
    preprocessing_method="preprocessing_v12",
    postprocessing_method="postprocessing_v03",
    output_json_path="risultati.json",
)
risultato = pipeline.run()
print(risultato)
```
Esempi di questo tipo sono implementati nei file `run_batch_{zero_shot,few_shot,cot}.py`.

## Citazione del benchmark Spider4SPARQL

Se utilizzi il benchmark, cita il lavoro secondo il BibTeX riportato nella [repository ufficiale Spider4SPARQL](https://github.com/ckosten/Spider4SPARQL):

```bibtex
@INPROCEEDINGS{10386182,
  author    = {C. Kosten and P. Cudre-Mauroux and K. Stockinger},
  booktitle = {2023 IEEE International Conference on Big Data (BigData)},
  title     = {Spider4SPARQL: A Complex Benchmark for Evaluating Knowledge Graph Question Answering Systems},
  year      = {2023},
  pages     = {5272--5281},
  doi       = {10.1109/BigData59044.2023.10386182},
  publisher = {IEEE Computer Society},
  address   = {Los Alamitos, CA, USA},
  month     = dec
}
```
