import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Set, Iterable
import numpy as np

# Supporto encoding UTF-8 su console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Disabilita warning e barre su Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
os.environ.pop("HF_HUB_ETAG_TIMEOUT", None)

# Import RDFLib per il parsing delle ontologie (.owl/.ttl)
try:
    import rdflib
    from rdflib import Graph, URIRef, BNode, Literal, Namespace, RDF, RDFS, OWL
except ImportError:
    Graph = URIRef = BNode = Literal = Namespace = RDF = RDFS = OWL = None

# Import PyTorch e GLiNER
try:
    import torch
except ImportError:
    torch = None

try:
    from gliner import GLiNER
except ImportError:
    GLiNER = None

try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except Exception:
    pass


# ==============================================================================
# FASE 0: HELPER DATASET (SPIDER4SPARQL)
# ==============================================================================

def get_concert_singer_dataset(base_dir: str = "spider4sparql") -> Tuple[str, str, List[Dict[str, Any]]]:
    """Recupera l'ontologia concert_singer.ttl e le relative domande dal dataset spider4sparql."""
    script_dir = Path(__file__).parent.resolve()
    base_path = script_dir / base_dir if not Path(base_dir).is_absolute() else Path(base_dir)

    ttl_file = base_path / "concert_singer.ttl"
    zip_file = base_path / "materialized_triples_dev.zip"
    json_file = base_path / "dev.json"

    if not ttl_file.exists():
        if zip_file.exists():
            print(f"--> Estrazione 'concert_singer.ttl' da '{zip_file}'...", flush=True)
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extract('materialized_triples/concert_singer.ttl', base_path)
            extracted_path = base_path / "materialized_triples" / "concert_singer.ttl"
            if extracted_path.exists():
                extracted_path.rename(ttl_file)
                try:
                    (base_path / "materialized_triples").rmdir()
                except Exception:
                    pass
        else:
            raise FileNotFoundError(f"Impossibile trovare {zip_file} o {ttl_file}")

    questions = []
    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
        questions = [item for item in dev_data if item.get('kg_name') == 'concert_singer']

    return str(ttl_file), str(json_file), questions


# ==============================================================================
# FASE 1: ESTRAZIONE TERMINI ONTOLOGIA (CLASSI E INDIVIDUI / LETTERALI)
# ==============================================================================

def split_camel_case(text: str) -> str:
    """Separa le parole in PascalCase, camelCase, trattini, delimitatori e cifre."""
    if not text:
        return text

    if text.startswith('"') and text.endswith('"'):
        return text[1:-1].strip()

    text = re.sub(r'[\-_=;]+', ' ', text)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
    text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
    tokens = text.split()
    deduped = []
    for tok in tokens:
        if not deduped or deduped[-1].lower() != tok.lower():
            deduped.append(tok)
    return " ".join(deduped).strip()


def extract_ontology_prefixes(graph: Optional[Graph]) -> Dict[str, str]:
    """
    Estrae prefissi e namespace dal grafo RDF.
    Mantiene il prefisso base vuoto ':' per il namespace principale ed evita la creazione
    di prefissi secondari/artificiali per tabelle/classi con '#' (es. visitor#name),
    in modo tale che vengano serializzati con CURIE del tipo :visitor\\#name.
    """
    prefixes: Dict[str, str] = {}
    if graph is None:
        return prefixes

    for pfx, ns in graph.namespaces():
        ns_str = str(ns)
        if pfx:
            prefixes[str(pfx)] = ns_str
        elif ns_str:
            prefixes[""] = ns_str

    used_uris: Set[str] = set()
    for s, p, o in graph:
        if isinstance(s, URIRef): used_uris.add(str(s))
        if isinstance(p, URIRef): used_uris.add(str(p))
        if isinstance(o, URIRef): used_uris.add(str(o))

    ns_to_pfx = {v: k for k, v in prefixes.items()}

    # Identifica il base namespace per valuenet o namespace principale
    for uri in used_uris:
        if "valuenet/ontop/" in uri and "http://valuenet/ontop/" not in ns_to_pfx:
            prefixes[""] = "http://valuenet/ontop/"
            ns_to_pfx["http://valuenet/ontop/"] = ""
            break

    base_ns = prefixes.get("")

    for uri in used_uris:
        # Se l'URI appartiene al base namespace (es. http://valuenet/ontop/), non creiamo sotto-prefissi artificiali
        if base_ns and uri.startswith(base_ns):
            continue
        if "#" in uri:
            ns_part, _ = uri.rsplit("#", 1)
            ns_full = ns_part + "#"
            if ns_full not in ns_to_pfx:
                ns_clean = ns_part.rstrip("/")
                last_seg = ns_clean.rsplit("/", 1)[-1] if "/" in ns_clean else ns_clean
                cand = re.sub(r'\.(owl|ttl|rdf|xml|obo|nt|n3|jsonld)$', '', last_seg, flags=re.IGNORECASE).lower()
                cand = re.sub(r'[^a-zA-Z0-9_]', '_', cand).strip('_')
                if not cand:
                    cand = f"ns{len(prefixes) + 1}"
                final_pfx = cand
                ctr = 1
                while final_pfx in prefixes and prefixes[final_pfx] != ns_full:
                    final_pfx = f"{cand}_{ctr}"
                    ctr += 1
                prefixes[final_pfx] = ns_full
                ns_to_pfx[ns_full] = final_pfx

    # Se è presente il base namespace vuoto, rimuoviamo eventuali prefissi secondari nidificati sotto di esso
    if "" in prefixes:
        base_val = prefixes[""]
        to_remove = [k for k, v in prefixes.items() if k != "" and v.startswith(base_val)]
        for k in to_remove:
            del prefixes[k]

    return prefixes


def parse_uri_components(uri: str) -> Tuple[str, str]:
    """Estrae identificatore compatto (:visitor\\#name) e testo leggibile da un URI o Literal."""
    term_str = uri.strip()
    if term_str.startswith("<") and term_str.endswith(">"):
        term_str = term_str[1:-1]
    if "^^" in term_str:
        term_str = term_str.split("^^")[0]

    if term_str.startswith('"'):
        val = re.sub(r'^"(.*)"(@[a-zA-Z\-]+)?$', r'\1', term_str).strip()
        return val, val

    if "#" in term_str:
        ns_part, local_part = term_str.rsplit("#", 1)
        ns_clean = ns_part.rstrip("/")
        last_seg = ns_clean.rsplit("/", 1)[-1] if "/" in ns_clean else ns_clean

        is_onto_file = bool(re.search(r'\.(owl|ttl|rdf|xml|obo|nt|n3|jsonld)$', last_seg, re.IGNORECASE))
        is_std_ns = any(std in ns_part.lower() for std in ["w3.org", "xmlns.com", "purl.org"])

        if last_seg and not is_onto_file and not is_std_ns and not last_seg.startswith("http:") and not last_seg.startswith("https:"):
            identifier = f":{last_seg}\\#{local_part}"
            readable_text = f"{last_seg} {local_part}"
            return identifier, readable_text
        else:
            return local_part, local_part
    elif "/" in term_str:
        parts = [p for p in term_str.split("/") if p and not p.startswith("http:") and not p.startswith("https:")]
        if len(parts) >= 2 and any("=" in p for p in parts[-2:]):
            identifier = "/".join(parts[-2:])
            readable_text = " ".join(parts[-2:])
            return identifier, readable_text
        elif parts:
            return parts[-1], parts[-1]
        return term_str, term_str
    elif ":" in term_str and not (term_str.startswith("http:") or term_str.startswith("https:")):
        local_part = term_str.split(":")[-1]
        return local_part, local_part

    return term_str, term_str


def clean_term_to_text(term_str: str) -> str:
    """
    Pulisce un termine RDF (URI/Literal) convertendolo in testo leggibile lowercase.
    Rimuove qualificatori geografici o di disambiguazione dopo la virgola (es. ,_Chennai o , Chennai)
    e qualificatori tra parentesi (es. _(Canada) o (Canada)).
    Risorse come <http://dbpedia.org/resource/Mint_Clock_Tower,_Chennai> vengono codificate come 'mint clock tower'.
    """
    _, readable_text = parse_uri_components(term_str)
    # Rimuove qualificatori di disambiguazione dopo la virgola (es. ,_Chennai o , Chennai)
    readable_text = re.sub(r',.*$', '', readable_text)
    # Rimuove qualificatori di disambiguazione tra parentesi (es. _(Canada) o (Canada))
    readable_text = re.sub(r'_?\([^)]*\)', '', readable_text)
    return split_camel_case(readable_text).lower()


def clean_term_label(uri_or_str: Union[URIRef, str]) -> str:
    """Estrae una label sintetica e leggibile (es. singer, stadium_id, Thing)."""
    s = str(uri_or_str).strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1]
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    elif "/" in s:
        parts = [p for p in s.split("/") if p]
        return parts[-1] if parts else s
    return s


# ==============================================================================
# FASE 1.1: ESTRAZIONE TASSONOMIA IN FORMA GERARCHICA (PREPROCESSING V12 CON CACHE)
# ==============================================================================

_GLOBAL_TAXONOMY_CACHE: Dict[str, Dict[str, Any]] = {}

def get_taxonomy_cache_filepath(key: str) -> Optional[Path]:
    """Determina il percorso del file di cache della tassonomia su disco per una data ontologia."""
    try:
        p = Path(key)
        if p.exists() or p.suffix.lower() in (".ttl", ".owl", ".rdf", ".xml"):
            return p.with_suffix(".taxonomy_cache.json")
        script_dir = Path(__file__).parent.resolve()
        safe_key = re.sub(r'[^a-zA-Z0-9_\-]', '_', key)
        return script_dir / f"{safe_key}_taxonomy_cache.json"
    except Exception:
        return None


def extract_taxonomy_hierarchy(
    source: Union[str, Graph],
    selected_classes: Optional[Union[List[str], Set[str]]] = None,
    output_tree_path: Optional[str] = None,
    use_cache: bool = True,
    cache_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Estrae la tassonomia dell'ontologia in forma gerarchica ad albero ricorsivo (Formato 1, Preprocessing v12).
    Supporta il caching a due livelli (in-memory e persistente su disco):
    - Se la tassonomia base per l'ontologia è già in cache, la carica istantaneamente senza ri-analizzare il grafo.
    - Se non è presente, calcola la tassonomia alla prima inferenza e la memorizza in cache per tutte le chiamate successive.
    """
    # 0. Risoluzione della chiave di cache
    normalized_key = None
    if isinstance(source, (str, Path)):
        try:
            normalized_key = str(Path(source).resolve())
        except Exception:
            normalized_key = str(source)
    elif cache_key:
        try:
            normalized_key = str(Path(cache_key).resolve()) if os.path.exists(str(cache_key)) else str(cache_key)
        except Exception:
            normalized_key = str(cache_key)

    base_tree: Optional[Dict[str, Any]] = None

    # 1. Verifica cache in memoria
    if use_cache and normalized_key and normalized_key in _GLOBAL_TAXONOMY_CACHE:
        print(f"--> [CACHE] Hit tassonomia in memoria per '{clean_term_label(normalized_key)}'", flush=True)
        base_tree = _GLOBAL_TAXONOMY_CACHE[normalized_key]

    # 2. Verifica cache persistente su disco
    if base_tree is None and use_cache and normalized_key:
        disk_cache_file = get_taxonomy_cache_filepath(normalized_key)
        if disk_cache_file and disk_cache_file.exists():
            try:
                with open(disk_cache_file, "r", encoding="utf-8") as f:
                    disk_tree = json.load(f)
                if disk_tree:
                    base_tree = disk_tree
                    _GLOBAL_TAXONOMY_CACHE[normalized_key] = base_tree
                    print(f"--> [CACHE] Tassonomia caricata da file di cache: '{disk_cache_file}'", flush=True)
            except Exception as e:
                print(f"[CACHE] Warning: lettura cache su disco fallita ({e}), calcolo in corso...", flush=True)
                base_tree = None

    # 3. Calcolo ex-novo alla prima inferenza (in assenza di cache)
    if base_tree is None:
        print(f"--> [TAXONOMY] Calcolo tassonomia ex-novo per: '{clean_term_label(normalized_key) if normalized_key else 'Grafo RDF'}'...", flush=True)

        if isinstance(source, Graph):
            g = source
        else:
            g = Graph()
            ext = os.path.splitext(str(source))[1].lower()
            fmt = "xml" if ext in (".owl", ".rdf", ".xml") else "turtle"
            try:
                g.parse(str(source), format=fmt)
            except Exception:
                g.parse(str(source))

        inst_to_class: Dict[URIRef, URIRef] = {}
        class_to_instances: Dict[URIRef, Set[URIRef]] = {}
        explicit_classes: Set[URIRef] = set()
        parents_map: Dict[URIRef, Set[URIRef]] = {}
        children_map: Dict[URIRef, Set[URIRef]] = {}

        for s, p, o in g.triples((None, RDF.type, None)):
            if o in (OWL.Class, RDFS.Class):
                if isinstance(s, URIRef):
                    explicit_classes.add(s)
            elif o not in (OWL.NamedIndividual, OWL.Ontology):
                if isinstance(o, URIRef):
                    inst_to_class[s] = o
                    class_to_instances.setdefault(o, set()).add(s)

        for sub, p, sup in g.triples((None, RDFS.subClassOf, None)):
            if isinstance(sub, URIRef) and isinstance(sup, URIRef):
                explicit_classes.add(sub)
                explicit_classes.add(sup)
                parents_map.setdefault(sub, set()).add(sup)
                children_map.setdefault(sup, set()).add(sub)

        all_classes: Set[URIRef] = set(explicit_classes) | set(class_to_instances.keys())

        # Fallback euristico per ontologie Spider senza rdf:type esplicito (es. wta_1.ttl)
        if not all_classes:
            for p in g.predicates():
                p_str = str(p)
                if "#" in p_str:
                    base_uri = p_str.rsplit("#", 1)[0]
                    cls_ref = URIRef(base_uri)
                    all_classes.add(cls_ref)

            for s in g.subjects():
                s_str = str(s)
                for cls_ref in all_classes:
                    if s_str.startswith(str(cls_ref) + "/"):
                        inst_to_class[s] = cls_ref
                        class_to_instances.setdefault(cls_ref, set()).add(s)
                        break

        root_ref = URIRef("http://www.w3.org/2002/07/owl#Thing")
        for c in sorted(all_classes, key=lambda x: clean_term_label(x)):
            c_parents = parents_map.get(c, set())
            if not c_parents or c_parents == {root_ref}:
                parents_map.setdefault(c, set()).add(root_ref)
                children_map.setdefault(root_ref, set()).add(c)

        # Helper per determinare se una classe è generica (es. owl:Thing, owl:Class, rdfs:Resource)
        def is_generic_class(cls_ref: Any) -> bool:
            if cls_ref is None:
                return True
            if cls_ref in (OWL.Thing, OWL.Class, RDFS.Resource, RDFS.Class):
                return True
            cls_str = str(cls_ref).strip()
            if cls_str in (
                "http://www.w3.org/2002/07/owl#Thing",
                "http://www.w3.org/2002/07/owl#Class",
                "http://www.w3.org/2000/01/rdf-schema#Resource",
                "http://www.w3.org/2000/01/rdf-schema#Class"
            ):
                return True
            label = clean_term_label(cls_ref).lower()
            return label in ("thing", "resource", "class", "owl:thing", "owl:class")

        # Helper per calcolare la minima classe comune (Lowest Common Ancestor - LCA)
        def find_lowest_common_ancestor(classes: List[URIRef]) -> Optional[URIRef]:
            valid_classes = [c for c in classes if isinstance(c, URIRef) and not is_generic_class(c)]
            if not valid_classes:
                return None

            if len(set(valid_classes)) == 1:
                return valid_classes[0]

            def _get_ancestors(c: URIRef) -> Set[URIRef]:
                ancestors = {c}
                queue = [c]
                while queue:
                    curr = queue.pop(0)
                    for p in parents_map.get(curr, set()):
                        if p not in ancestors:
                            ancestors.add(p)
                            queue.append(p)
                return ancestors

            common_ancestors: Optional[Set[URIRef]] = None
            for c in valid_classes:
                anc = _get_ancestors(c)
                if common_ancestors is None:
                    common_ancestors = set(anc)
                else:
                    common_ancestors &= anc

            if not common_ancestors:
                return root_ref

            candidates = []
            for a in common_ancestors:
                is_min = True
                for b in common_ancestors:
                    if b != a and a in _get_ancestors(b):
                        is_min = False
                        break
                if is_min:
                    candidates.append(a)

            if not candidates:
                return root_ref

            depth_cache: Dict[URIRef, int] = {}
            def _get_depth(node: URIRef, seen: Optional[Set[URIRef]] = None) -> int:
                if node == root_ref:
                    return 0
                if node in depth_cache:
                    return depth_cache[node]
                if seen is None:
                    seen = set()
                seen.add(node)
                d = 0
                for p in parents_map.get(node, set()):
                    if p not in seen:
                        d = max(d, _get_depth(p, seen.copy()) + 1)
                depth_cache[node] = d
                return d

            candidates.sort(key=lambda x: (_get_depth(x), clean_term_label(x)), reverse=True)
            return candidates[0]

        # Helper per recuperare la classe di un nodo (istanza o termine URI)
        def get_node_class(node: Any) -> Optional[URIRef]:
            if not isinstance(node, URIRef):
                return None
            if node in inst_to_class and not is_generic_class(inst_to_class[node]):
                return inst_to_class[node]
            for t in g.objects(node, RDF.type):
                if isinstance(t, URIRef) and t in all_classes and not is_generic_class(t):
                    return t
            if node in all_classes and not is_generic_class(node):
                return node
            node_str = str(node)
            for c in all_classes:
                c_str = str(c)
                if node_str.startswith(c_str + "/") or node_str.startswith(c_str + "#"):
                    return c
            return None

        # Helper per estrarre le classi che definiscono una restrizione
        def find_classes_holding_restriction(r_node: Any) -> Set[URIRef]:
            classes = set()
            for s in list(g.subjects(RDFS.subClassOf, r_node)) + list(g.subjects(OWL.equivalentClass, r_node)):
                if isinstance(s, URIRef) and not is_generic_class(s):
                    classes.add(s)
            list_nodes = set(g.subjects(RDF.first, r_node))
            while list_nodes:
                l = list_nodes.pop()
                for parent_list in g.subjects(RDF.rest, l):
                    list_nodes.add(parent_list)
                for expr in list(g.subjects(OWL.intersectionOf, l)) + list(g.subjects(OWL.unionOf, l)):
                    for s in list(g.subjects(RDFS.subClassOf, expr)) + list(g.subjects(OWL.equivalentClass, expr)):
                        if isinstance(s, URIRef) and not is_generic_class(s):
                            classes.add(s)
            return classes

        # Helper per estrarre le classi range da una restrizione
        def find_ranges_from_restriction(r_node: Any) -> Set[URIRef]:
            ranges = set()
            for target in list(g.objects(r_node, OWL.someValuesFrom)) + list(g.objects(r_node, OWL.allValuesFrom)):
                if isinstance(target, URIRef) and not is_generic_class(target):
                    ranges.add(target)
            for val in g.objects(r_node, OWL.hasValue):
                c = get_node_class(val)
                if c and not is_generic_class(c):
                    ranges.add(c)
                elif isinstance(val, URIRef) and not is_generic_class(val):
                    ranges.add(val)
            return ranges

        # Helper per formattare il datatype in modo sintetico e standard (es. xsd:integer, xsd:boolean, xsd:string)
        def format_datatype(dt_uri: Any) -> Optional[str]:
            if not dt_uri:
                return None
            dt_str = str(dt_uri).strip()
            if not dt_str:
                return None
            try:
                qname = g.namespace_manager.qname(URIRef(dt_str))
                if qname:
                    return qname
            except Exception:
                pass
            if dt_str.startswith("http://www.w3.org/2001/XMLSchema#"):
                return f"xsd:{dt_str.split('#')[-1]}"
            if dt_str.startswith("http://www.w3.org/1999/02/22-rdf-syntax-ns#"):
                return f"rdf:{dt_str.split('#')[-1]}"
            if dt_str.startswith("http://www.w3.org/2000/01/rdf-schema#"):
                return f"rdfs:{dt_str.split('#')[-1]}"
            if "#" in dt_str:
                return dt_str.split("#")[-1]
            if "/" in dt_str:
                return dt_str.split("/")[-1]
            return dt_str

        # Helper per estrarre il datatipo di una data property seguendo la strategia per livelli:
        # Livello 1: datatipo espresso nativamente su rdfs:range o restrizioni OWL
        # Livello 2: datatipo dichiarato sul primo oggetto della property
        def resolve_data_property_datatype(p_ref: URIRef) -> Optional[str]:
            # LIVELLO 1: Verifica se il datatipo è espresso nativamente (rdfs:range o restrizioni OWL)
            for r in g.objects(p_ref, RDFS.range):
                if isinstance(r, URIRef):
                    r_str = str(r)
                    if (
                        "XMLSchema#" in r_str
                        or r in (RDFS.Literal, RDF.langString, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#PlainLiteral"))
                        or (r, RDF.type, RDFS.Datatype) in g
                        or (r not in all_classes and not is_generic_class(r))
                    ):
                        return format_datatype(r)

            for r_node in g.subjects(OWL.onProperty, p_ref):
                for target in list(g.objects(r_node, OWL.allValuesFrom)) + \
                              list(g.objects(r_node, OWL.someValuesFrom)) + \
                              list(g.objects(r_node, URIRef("http://www.w3.org/2002/07/owl#onDataRange"))):
                    if isinstance(target, URIRef):
                        t_str = str(target)
                        if (
                            "XMLSchema#" in t_str
                            or target in (RDFS.Literal, RDF.langString)
                            or (target, RDF.type, RDFS.Datatype) in g
                            or (target not in all_classes and not is_generic_class(target))
                        ):
                            return format_datatype(target)

            # LIVELLO 2: Se non espresso, valuta il primo oggetto della property e vedi se ha il datatipo dichiarato
            for s, p, o in g.triples((None, p_ref, None)):
                if isinstance(o, Literal) and o.datatype:
                    return format_datatype(o.datatype)
                # Abbiamo valutato il primo oggetto usato con quella property
                break

            # Se il datatipo non è specificato presumiamo sia una stringa (xsd:string)
            return "xsd:string"

        # Dizionari per data properties e object properties per classe
        class_data_properties: Dict[URIRef, Dict[str, Dict[str, Any]]] = {c: {} for c in all_classes}
        class_object_properties: Dict[URIRef, List[Dict[str, Any]]] = {c: [] for c in all_classes}

        # Raccogli tutti i predicati da analizzare nell'ontologia
        all_predicates: Set[URIRef] = set()
        for p in g.predicates():
            if isinstance(p, URIRef):
                p_str = str(p)
                if not any(p_str.startswith(ns) for ns in [
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                    "http://www.w3.org/2000/01/rdf-schema#",
                    "http://www.w3.org/2002/07/owl#"
                ]):
                    all_predicates.add(p)

        if OWL is not None:
            for p in g.subjects(RDF.type, OWL.ObjectProperty):
                if isinstance(p, URIRef): all_predicates.add(p)
            for p in g.subjects(RDF.type, OWL.DatatypeProperty):
                if isinstance(p, URIRef): all_predicates.add(p)

        # Risoluzione domain e range per ciascuna proprietà (Strategia a 3 Livelli)
        for p_uri in sorted(all_predicates, key=lambda x: str(x)):
            p_str = str(p_uri)
            p_label = clean_term_label(p_uri)
            p_ident, _ = parse_uri_components(p_str)
            p_compound = p_ident if p_ident.startswith(":") else (f":{p_ident}" if not ":" in p_ident else p_ident)

            domain_ref: Optional[URIRef] = None
            range_ref: Optional[URIRef] = None

            # ------------------------------------------------------------------
            # LIVELLO 1: Proprietà native rdfs:domain e rdfs:range (se non generiche)
            # ------------------------------------------------------------------
            native_domains = [d for d in g.objects(p_uri, RDFS.domain) if isinstance(d, URIRef) and not is_generic_class(d)]
            if native_domains:
                domain_ref = find_lowest_common_ancestor(native_domains)

            native_ranges = [r for r in g.objects(p_uri, RDFS.range) if isinstance(r, URIRef) and not is_generic_class(r)]
            if native_ranges:
                range_ref = find_lowest_common_ancestor(native_ranges)

            # ------------------------------------------------------------------
            # LIVELLO 2: Ricerca delle restrizioni OWL (owl:Restriction)
            # ------------------------------------------------------------------
            if domain_ref is None or range_ref is None:
                restr_nodes = set(g.subjects(OWL.onProperty, p_uri))
                restr_domains = []
                restr_ranges = []
                for r_node in restr_nodes:
                    if domain_ref is None:
                        restr_domains.extend(find_classes_holding_restriction(r_node))
                    if range_ref is None:
                        restr_ranges.extend(find_ranges_from_restriction(r_node))

                if domain_ref is None and restr_domains:
                    domain_ref = find_lowest_common_ancestor(restr_domains)
                if range_ref is None and restr_ranges:
                    range_ref = find_lowest_common_ancestor(restr_ranges)

            # ------------------------------------------------------------------
            # LIVELLO 3: Campionamento prime tre triple (soggetti e oggetti) e LCA
            # ------------------------------------------------------------------
            sample_subjects = []
            sample_objects = []
            for s, p, o in g.triples((None, p_uri, None)):
                if s not in sample_subjects and len(sample_subjects) < 3:
                    sample_subjects.append(s)
                if o not in sample_objects and len(sample_objects) < 3:
                    sample_objects.append(o)
                if len(sample_subjects) >= 3 and len(sample_objects) >= 3:
                    break

            has_literals = any(isinstance(o, Literal) for o in sample_objects)

            if domain_ref is None and sample_subjects:
                subj_classes = []
                for s in sample_subjects:
                    sc = get_node_class(s)
                    if sc and not is_generic_class(sc):
                        subj_classes.append(sc)
                if subj_classes:
                    domain_ref = find_lowest_common_ancestor(subj_classes)

            if range_ref is None and sample_objects and not has_literals:
                obj_classes = []
                for o in sample_objects:
                    oc = get_node_class(o)
                    if oc and not is_generic_class(oc):
                        obj_classes.append(oc)
                if obj_classes:
                    range_ref = find_lowest_common_ancestor(obj_classes)

            # ------------------------------------------------------------------
            # Fallback euristiche Spider / URI naming
            # ------------------------------------------------------------------
            if range_ref is None and "ref-" in p_label:
                ref_candidate = p_label.replace("ref-", "").replace("_id", "")
                for c in all_classes:
                    if clean_term_label(c).lower() == ref_candidate.lower():
                        range_ref = c
                        break

            if domain_ref is None and "#" in p_str:
                cand_base = p_str.rsplit("#", 1)[0]
                for c in all_classes:
                    if str(c) == cand_base:
                        domain_ref = c
                        break

            # ------------------------------------------------------------------
            # Determinazione tipo di proprietà (ObjectProperty vs DataProperty)
            # ------------------------------------------------------------------
            is_object_property = False
            if OWL is not None and (p_uri, RDF.type, OWL.ObjectProperty) in g:
                is_object_property = True
            elif OWL is not None and (p_uri, RDF.type, OWL.DatatypeProperty) in g:
                is_object_property = False
            elif "ref-" in p_label:
                is_object_property = True
            elif range_ref is not None and (range_ref in all_classes or range_ref in explicit_classes):
                is_object_property = True
            elif has_literals:
                is_object_property = False
            elif range_ref is not None:
                is_object_property = True

            # Registrazione della proprietà nelle strutture della tassonomia con label composte
            # Mantenuta SOLO nella classe che fa da domain, con range verso la classe di destinazione
            if is_object_property and domain_ref and range_ref:
                range_raw = clean_term_label(range_ref)
                range_name = range_raw if ":" in range_raw else f":{range_raw}"
                obj_prop_dict = {
                    "property_name": p_compound,
                    "range": range_name
                }
                if domain_ref in class_object_properties and obj_prop_dict not in class_object_properties[domain_ref]:
                    class_object_properties[domain_ref].append(obj_prop_dict)
            elif not is_object_property:
                dt_val = resolve_data_property_datatype(p_uri)
                if domain_ref and domain_ref in class_data_properties:
                    if p_str not in class_data_properties[domain_ref]:
                        class_data_properties[domain_ref][p_str] = {
                            "name": p_compound,
                            "uri": p_str,
                            "datatype": dt_val
                        }

        # Registrazione diretta per proprietà dato osservate su triple di istanze di ciascuna classe
        for s, p, o in g:
            if p in all_predicates and isinstance(o, Literal):
                s_cls = get_node_class(s)
                if s_cls and s_cls in class_data_properties:
                    p_str = str(p)
                    p_ident, _ = parse_uri_components(p_str)
                    p_compound = p_ident if p_ident.startswith(":") else (f":{p_ident}" if not ":" in p_ident else p_ident)
                    if p_str not in class_data_properties[s_cls]:
                        dt_val = resolve_data_property_datatype(p)
                        if dt_val is None and o.datatype:
                            dt_val = format_datatype(o.datatype)
                        class_data_properties[s_cls][p_str] = {
                            "name": p_compound,
                            "uri": p_str,
                            "datatype": dt_val
                        }
                    elif class_data_properties[s_cls][p_str].get("datatype") is None and o.datatype:
                        class_data_properties[s_cls][p_str]["datatype"] = format_datatype(o.datatype)

        # Costruzione ricorsiva albero gerarchico base
        def build_tree_dict(node_uri: URIRef, level: int = 0, visited: Optional[Set[URIRef]] = None) -> Dict[str, Any]:
            if visited is None:
                visited = set()

            is_root = (str(node_uri) == str(root_ref))
            raw_name = clean_term_label(node_uri)
            node_name = "owl:Thing" if is_root else (raw_name if ":" in raw_name else f":{raw_name}")

            node_dict: Dict[str, Any] = {
                "class_name": node_name
            }

            if not is_root:
                node_dict["data_properties"] = sorted(
                    [
                        {
                            "property_name": p["name"],
                            "datatype": p.get("datatype") or "xsd:string"
                        }
                        for p in class_data_properties.get(node_uri, {}).values()
                    ],
                    key=lambda x: x["property_name"]
                )
                node_dict["object_properties"] = sorted(
                    class_object_properties.get(node_uri, []),
                    key=lambda x: (x["property_name"], x.get("range", x.get("range_class", "")))
                )

            if node_uri in visited:
                node_dict["cycle_detected"] = True
                node_dict["subclasses"] = []
                return node_dict

            curr_visited = visited.copy()
            curr_visited.add(node_uri)

            direct_children = sorted(children_map.get(node_uri, set()), key=lambda x: clean_term_label(x))
            node_dict["subclasses"] = [
                build_tree_dict(child, level=level + 1, visited=curr_visited)
                for child in direct_children
            ]

            return node_dict

        base_tree = build_tree_dict(root_ref, level=0)

        # Salva in cache in memoria
        if use_cache and normalized_key:
            _GLOBAL_TAXONOMY_CACHE[normalized_key] = base_tree
            # Salva in cache persistente su file JSON
            disk_cache_file = get_taxonomy_cache_filepath(normalized_key)
            if disk_cache_file:
                try:
                    with open(disk_cache_file, "w", encoding="utf-8") as f:
                        json.dump(base_tree, f, indent=2, ensure_ascii=False)
                    print(f"--> [CACHE] Tassonomia salvata su file di cache: '{disk_cache_file}'", flush=True)
                except Exception as e:
                    pass

    final_taxonomy_tree = base_tree
    return final_taxonomy_tree


_GLOBAL_TERMS_CACHE: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Graph, Dict[str, Set[str]], Dict[str, Set[str]]]] = {}


def extract_classified_terms(
    source_path: str,
    output_prefix: Optional[str] = None,
    return_mappings: bool = True
) -> Union[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Graph, Dict[str, Set[str]], Dict[str, Set[str]]]]:
    """
    Estrae dall'ontologia:
    1. ENTITY:
       - Classi (category='class')
       - Individui URI e Valori Letterali/Oggetto (category='individual')
    2. PROPERTY:
       - Predicati/Proprietà (category='property')
    3. Mappature Nodi/Istanze (calcolate 1 sola volta per evitare re-parsing):
       - class_to_instances: mappa da URI classe a set di URI individui
       - instance_to_classes: mappa da URI individuo a set di URI classi
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"File ontologia non trovato: '{source_path}'")

    norm_path = str(Path(source_path).resolve())
    if norm_path in _GLOBAL_TERMS_CACHE:
        cached_entities, cached_props, cached_g, cached_c2i, cached_i2c = _GLOBAL_TERMS_CACHE[norm_path]
        if return_mappings:
            return cached_entities, cached_props, cached_g, cached_c2i, cached_i2c
        return cached_entities, cached_props

    print(f"--> [PREPROCESSING V12] Parsing ontologia da: '{source_path}'...", flush=True)

    g = Graph()
    ext = os.path.splitext(source_path)[1].lower()
    fmt = "xml" if ext == ".owl" else "turtle"

    try:
        g.parse(source_path, format=fmt)
    except Exception:
        g.parse(source_path)

    ontology_prefixes = extract_ontology_prefixes(g)

    class_nodes = set()
    individual_nodes = set()
    property_nodes = set()
    literal_values = set()
    instance_to_classes: Dict[str, Set[str]] = {}
    class_to_instances: Dict[str, Set[str]] = {}

    for s, p, o in g:
        # Properties
        if isinstance(p, URIRef):
            p_str = str(p)
            if not (p_str.startswith("http://www.w3.org/1999/02/22-rdf-syntax-ns#") or
                    p_str.startswith("http://www.w3.org/2000/01/rdf-schema#") or
                    p_str.startswith("http://www.w3.org/2002/07/owl#")):
                property_nodes.add(p)

        # Classes & Individuals
        if RDF is not None and p == RDF.type:
            if o in (OWL.Class, RDFS.Class):
                class_nodes.add(s)
            elif isinstance(o, URIRef) and not str(o).startswith("http://www.w3.org/2002/07/owl#"):
                individual_nodes.add(s)
                class_nodes.add(o)
                s_str, o_str = str(s), str(o)
                if s_str not in instance_to_classes:
                    instance_to_classes[s_str] = set()
                instance_to_classes[s_str].add(o_str)

                if o_str not in class_to_instances:
                    class_to_instances[o_str] = set()
                class_to_instances[o_str].add(s_str)
        elif RDFS is not None and p == RDFS.subClassOf:
            if isinstance(s, URIRef): class_nodes.add(s)
            if isinstance(o, URIRef): class_nodes.add(o)

        # Literal Values (Individui come valori di proprietà, es. "France", "Peterhead", "John")
        if isinstance(o, Literal):
            lit_val = str(o).strip()
            if lit_val and len(lit_val) > 1 and not lit_val.isdigit():
                literal_values.add(lit_val)
        elif isinstance(o, URIRef) and not (RDF is not None and p == RDF.type):
            o_str = str(o)
            if not (o_str.startswith("http://www.w3.org/1999/02/22-rdf-syntax-ns#") or
                    o_str.startswith("http://www.w3.org/2000/01/rdf-schema#") or
                    o_str.startswith("http://www.w3.org/2002/07/owl#")):
                individual_nodes.add(o)

    individual_nodes = individual_nodes - class_nodes - property_nodes

    entities_list = []
    seen_texts = set()

    # 1. Aggiunta Classi
    for c in sorted(class_nodes, key=lambda x: str(x)):
        raw_str = str(c)
        ident, _ = parse_uri_components(raw_str)
        cleaned = clean_term_to_text(raw_str)
        if cleaned and cleaned not in ("class", "named individual", "thing"):
            entities_list.append({
                "id": len(entities_list),
                "cleaned": cleaned,
                "original": raw_str,
                "identifier": ident,
                "category": "class",
                "is_literal": False
            })

    # 2. Aggiunta Individui URI
    for ind in sorted(individual_nodes, key=lambda x: str(x)):
        raw_str = str(ind)
        ident, _ = parse_uri_components(raw_str)
        cleaned = clean_term_to_text(raw_str)
        if cleaned:
            entities_list.append({
                "id": len(entities_list),
                "cleaned": cleaned,
                "original": raw_str,
                "identifier": ident,
                "category": "individual",
                "is_literal": False
            })

    # 3. Aggiunta Valori Letterali / Istanze testuali
    for lit in sorted(literal_values):
        cleaned = split_camel_case(lit).lower()
        if cleaned and cleaned not in seen_texts and len(cleaned) >= 2:
            seen_texts.add(cleaned)
            entities_list.append({
                "id": len(entities_list),
                "cleaned": cleaned,
                "original": lit,
                "identifier": lit,
                "category": "individual",
                "is_literal": True
            })

    properties_list = []
    for pr in sorted(property_nodes, key=lambda x: str(x)):
        raw_str = str(pr)
        ident, _ = parse_uri_components(raw_str)
        cleaned = clean_term_to_text(raw_str)
        if cleaned:
            properties_list.append({
                "id": len(properties_list),
                "cleaned": cleaned,
                "original": raw_str,
                "identifier": ident,
                "category": "property",
                "is_literal": False
            })

    num_classes = sum(1 for e in entities_list if e["category"] == "class")
    num_individuals = sum(1 for e in entities_list if e["category"] == "individual")
    print(f"[OK] Estratte ENTITY: {num_classes:,} Classi, {num_individuals:,} Individui/Letterali; e {len(properties_list):,} PROPERTY.", flush=True)

    if output_prefix:
        ent_txt = f"{output_prefix}_entities.txt"
        prop_txt = f"{output_prefix}_properties.txt"
        pfx_json = f"{output_prefix}_prefixes.json"

        with open(ent_txt, "w", encoding="utf-8") as f:
            for item in entities_list:
                f.write(f"[{item['category'].upper()}]\t{item['cleaned']}\t{item['original']}\n")
        with open(prop_txt, "w", encoding="utf-8") as f:
            for item in properties_list:
                f.write(f"[PROPERTY]\t{item['cleaned']}\t{item['original']}\n")
        with open(pfx_json, "w", encoding="utf-8") as f:
            json.dump(ontology_prefixes, f, indent=2, ensure_ascii=False)

    _GLOBAL_TERMS_CACHE[norm_path] = (entities_list, properties_list, g, class_to_instances, instance_to_classes)

    if return_mappings:
        return entities_list, properties_list, g, class_to_instances, instance_to_classes
    return entities_list, properties_list


# ==============================================================================
# FASE 2: INFORMATION EXTRACTION ZERO-SHOT CON GLINER-RELEX E TASSONOMIA (PREPROCESSING V12)
# ==============================================================================

_GLOBAL_GLINER_MODEL = None

def load_gliner_model(
    model_name: str = "knowledgator/gliner-relex-large-v1.0",
    max_retries: int = 3
) -> Any:
    """Carica il modello GLiNER per l'estrazione zero-shot di entità con cache in memoria."""
    global _GLOBAL_GLINER_MODEL
    if _GLOBAL_GLINER_MODEL is not None:
        return _GLOBAL_GLINER_MODEL

    if GLiNER is None or torch is None:
        raise ImportError("La libreria 'gliner' e 'torch' sono necessarie. Esegui: pip install gliner torch")

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            model = GLiNER.from_pretrained(model_name)
            if torch.cuda.is_available():
                model = model.to("cuda")
            _GLOBAL_GLINER_MODEL = model
            return _GLOBAL_GLINER_MODEL
        except Exception as e:
            last_err = e
            time.sleep(1)

    raise RuntimeError(f"Impossibile caricare il modello GLiNER '{model_name}'. Errore: {last_err}") from last_err


def extract_subclass_hierarchy(g: Graph) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Estrae le relazioni dirette di sottoclasse (rdfs:subClassOf) dal grafo RDF."""
    parents_map: Dict[str, Set[str]] = {}
    children_map: Dict[str, Set[str]] = {}
    if RDFS is not None:
        for sub, p, sup in g.triples((None, RDFS.subClassOf, None)):
            s_sub, s_sup = str(sub), str(sup)
            parents_map.setdefault(s_sub, set()).add(s_sup)
            children_map.setdefault(s_sup, set()).add(s_sub)
    return parents_map, children_map


def get_class_ancestors(class_uri: str, parents_map: Dict[str, Set[str]]) -> Set[str]:
    """Calcola la chiusura transitiva di tutte le superclassi antenate di una classe."""
    ancestors = set()
    queue = [class_uri]
    while queue:
        curr = queue.pop(0)
        for p in parents_map.get(curr, set()):
            if p not in ancestors:
                ancestors.add(p)
                queue.append(p)
    return ancestors


def prune_superclasses(class_uris: Iterable[str], parents_map: Optional[Dict[str, Set[str]]]) -> Set[str]:
    """
    Regola Tassonomica Universale:
    Se nell'insieme è già presente una sottoclasse più specifica, non selezionare la superclasse antenata.
    """
    classes = set(class_uris)
    if len(classes) <= 1 or not parents_map:
        return classes
    pruned = set(classes)
    for c in classes:
        anc = get_class_ancestors(c, parents_map)
        pruned -= anc
    return pruned


def extract_entities_and_classes_gliner(
    query: str,
    entities: List[Dict[str, Any]],
    properties: Optional[List[Dict[str, Any]]] = None,
    subclass_parents: Optional[Dict[str, Set[str]]] = None,
    model_name: str = "knowledgator/gliner-relex-large-v1.0",
    gliner_model: Any = None,
    threshold: float = 0.3
) -> Tuple[Set[str], Set[str], Set[str], List[Dict[str, Any]]]:
    """
    [PREPROCESSING V12 - GLINER-RELEX]:
    Utilizza knowledgator/gliner-relex-large-v1.0 per estrarre entità
    dalla query in linguaggio naturale. Applica la tassonomia dell'ontologia per potare
    le superclassi se sono già presenti sottoclassi più specifiche.
    Restituisce:
    (selected_classes, selected_individuals_uri, selected_literals, extracted_entities)
    """
    if gliner_model is None:
        gliner_model = load_gliner_model(model_name=model_name)

    # 1. Mappatura classi ontologiche (cleaned -> original URI)
    class_map: Dict[str, str] = {}
    for e in entities:
        if e.get("category") == "class":
            class_map[e["cleaned"]] = e["original"]

    # Mappe per individui e letterali
    ind_map: Dict[str, str] = {}
    lit_map: Dict[str, str] = {}
    for e in entities:
        if e.get("category") == "individual":
            if e.get("is_literal"):
                lit_map[e["cleaned"]] = e["original"]
            else:
                ind_map[e["cleaned"]] = e["original"]

    class_labels = sorted(list(set(class_map.keys())))

    # 2.1 Esegui inferenza GLiNER per le classi tramite predict_entities (zero-shot NER pulito)
    extracted_entities = []
    try:
        extracted_entities = gliner_model.predict_entities(
            query,
            labels=class_labels,
            threshold=threshold
        )
    except Exception as e:
        print(f"[GLINER WARNING] predict_entities ha generato un errore ({e})", flush=True)
        extracted_entities = []

    selected_classes: Set[str] = set()

    # 3. Associa le entità estratte da GLiNER all'ontologia
    for ent in extracted_entities:
        span_text = ent.get("text", "").strip().lower()
        predicted_label = ent.get("label", "").strip().lower()

        if not span_text or len(span_text) <= 1 or re.match(r'^[^\w]+$', span_text):
            continue

        if predicted_label in class_map:
            selected_classes.add(class_map[predicted_label])
        elif span_text in class_map:
            selected_classes.add(class_map[span_text])

    # 3.1 Applicazione della Tassonomia: se è presente una sottoclasse, non selezionare la superclasse
    if subclass_parents:
        selected_classes = prune_superclasses(selected_classes, subclass_parents)

    # 3.2 Match individui: longest non-overlapping phrase match nella query
    selected_individuals_uri: Set[str] = set()
    selected_literals: Set[str] = set()
    query_lower = query.lower()

    matched_spans: List[Tuple[int, int]] = []
    for c_text, orig_uri in sorted(ind_map.items(), key=lambda x: len(x[0]), reverse=True):
        if len(c_text) < 3:
            continue
        pat = r'\b' + re.escape(c_text) + r'\b'
        match = re.search(pat, query_lower)
        if match:
            start, end = match.span()
            if not any(sp[0] <= start and end <= sp[1] for sp in matched_spans):
                matched_spans.append((start, end))
                selected_individuals_uri.add(orig_uri)

    # 3.3 Match letterali
    for c_text, orig_val in sorted(lit_map.items(), key=lambda x: len(x[0]), reverse=True):
        if len(c_text) < 3:
            continue
        pat = r'\b' + re.escape(c_text) + r'\b'
        if re.search(pat, query_lower):
            selected_literals.add(orig_val)

    # 4. Fallback se non è stato selezionato nulla
    if not selected_classes and not selected_individuals_uri and not selected_literals:
        for c_text, orig_uri in class_map.items():
            if re.search(r'\b' + re.escape(c_text) + r'\b', query_lower):
                selected_classes.add(orig_uri)
        if subclass_parents:
            selected_classes = prune_superclasses(selected_classes, subclass_parents)
        for c_text, orig_uri in ind_map.items():
            if len(c_text) >= 3 and re.search(r'\b' + re.escape(c_text) + r'\b', query_lower):
                selected_individuals_uri.add(orig_uri)
        for c_text, orig_val in lit_map.items():
            if len(c_text) >= 3 and re.search(r'\b' + re.escape(c_text) + r'\b', query_lower):
                selected_literals.add(orig_val)

    return selected_classes, selected_individuals_uri, selected_literals, extracted_entities


# ==============================================================================
# FASE 3: ESTRAZIONE SUBGRAFO E TASSONOMIA (PREPROCESSING V12)
# ==============================================================================

def extract_subgraph_v12(
    source: Union[str, Graph],
    selected_classes: Set[str],
    selected_individuals: Set[str],
    selected_literals: Set[str],
    class_to_instances: Optional[Dict[str, Set[str]]] = None,
    instance_to_classes: Optional[Dict[str, Set[str]]] = None,
    subclass_parents: Optional[Dict[str, Set[str]]] = None,
    max_samples_per_pattern: int = 3,
    return_taxonomy: bool = False
) -> Union[List[Dict[str, Any]], Tuple[Graph, List[Dict[str, Any]], Dict[str, Any]], Tuple[Graph, List[Dict[str, Any]]]]:
    """
    [REGOLA PREPROCESSING V12]:
    Estrae il sottografo dinamico per la query:
    1. Seleziona le proprietà usate in almeno una tripla assieme ad una delle classi selezionate.
    2. Seleziona massimo tre triple in cui compaia un individuo della classe selezionata e una delle proprietà selezionate nel passaggio 1.
    3. Seleziona tutte le triple contenenti almeno un individuo/letterale di quelli selezionati a inizio preprocessing.
    
    VINCOLO GLOBALE:
    Il subgrafo finale non deve contenere più di tre triple con subject e object appartenenti
    alle stesse rispettive classi e predicate uguale.
    I predicati ed entità del namespace target vengono formattati con CURIE :visitor\\#name.
    """
    if isinstance(source, Graph):
        g = source
    else:
        g = Graph()
        ext = os.path.splitext(str(source))[1].lower()
        fmt = "xml" if ext in (".owl", ".rdf", ".xml") else "turtle"
        try:
            g.parse(str(source), format=fmt)
        except Exception:
            g.parse(str(source))

    # --------------------------------------------------------------------------
    # 0. MAPPATURA NODI -> CLASSI / TIPI E TASSONOMIA SUBCLASSOF
    # --------------------------------------------------------------------------
    if subclass_parents is None:
        subclass_parents, _ = extract_subclass_hierarchy(g)

    if class_to_instances is None or instance_to_classes is None:
        instance_to_classes = {}
        class_to_instances = {}
        known_classes: Set[str] = set()

        for s, p, o in g:
            if RDF is not None and p == RDF.type:
                if o in (OWL.Class, RDFS.Class):
                    known_classes.add(str(s))
                else:
                    s_str, o_str = str(s), str(o)
                    if s_str not in instance_to_classes:
                        instance_to_classes[s_str] = set()
                    instance_to_classes[s_str].add(o_str)

                    if o_str not in class_to_instances:
                        class_to_instances[o_str] = set()
                    class_to_instances[o_str].add(s_str)
                    known_classes.add(o_str)
            elif RDFS is not None and p == RDFS.subClassOf:
                known_classes.add(str(s))
                known_classes.add(str(o))
    else:
        known_classes = set(class_to_instances.keys())

    def get_node_class_signature(node: Any) -> str:
        """Restituisce la firma della classe/tipo per un nodo soggetto o oggetto."""
        if Literal is not None and isinstance(node, Literal):
            if node.datatype:
                return f"Literal:{str(node.datatype)}"
            return "Literal:string"
        node_str = str(node)
        if node_str in instance_to_classes and instance_to_classes[node_str]:
            # Regola Tassonomica: mantieni solo le sottoclassi più specifiche dell'istanza
            specific_classes = prune_superclasses(instance_to_classes[node_str], subclass_parents)
            return "ClassInstance:" + "|".join(sorted(specific_classes))
        if node_str in known_classes:
            return f"ClassMeta:{node_str}"
        # In RDF standard, ogni nodo risorsa/URI non tipizzato appartiene a rdfs:Resource
        return "rdfs:Resource"

    # --------------------------------------------------------------------------
    # 1. SELEZIONA LE PROPRIETÀ USATE IN ALMENO UNA TRIPLA CON UNA DELLE CLASSI
    #    O CON UN INDIVIDUO DI UNA DELLE CLASSI SELEZIONATE A INIZIO PREPROCESSING
    # --------------------------------------------------------------------------
    target_classes: Set[str] = set(selected_classes)
    for ind_uri in selected_individuals:
        ind_classes = instance_to_classes.get(ind_uri, set())
        target_classes.update(prune_superclasses(ind_classes, subclass_parents))

    # Regola Tassonomica: se è già presente una sottoclasse, non selezionare la superclasse
    target_classes = prune_superclasses(target_classes, subclass_parents)

    target_individuals: Set[str] = set(selected_individuals)
    for c_uri in target_classes:
        target_individuals.update(class_to_instances.get(c_uri, set()))

    selected_properties: Set[str] = set()
    selected_class_properties: Dict[str, Set[str]] = {c: set() for c in target_classes}

    for s, p, o in g:
        s_str, p_str, o_str = str(s), str(p), str(o)

        is_relevant_triple = False
        if s_str in target_classes or o_str in target_classes or s_str in target_individuals or o_str in target_individuals:
            is_relevant_triple = True
            selected_properties.add(p_str)

        if is_relevant_triple:
            for c_uri in target_classes:
                c_instances = class_to_instances.get(c_uri, set())
                if s_str in c_instances or o_str in c_instances or s_str == c_uri or o_str == c_uri:
                    selected_class_properties[c_uri].add(p_str)

    print(f"--> [FASE 1] Proprietà identificate per classi/individui a inizio preprocessing:")
    print(f"    - Totale classi target: {len(target_classes)}")
    print(f"    - Totale proprietà uniche selezionate: {len(selected_properties)}")
    for c_uri, props in selected_class_properties.items():
        print(f"    - Classe <{c_uri}>: {len(props)} proprietà associate")

    # --------------------------------------------------------------------------
    # STRUTTURE PER SUBGRAFO E VINCOLO DI MASSIMO 3 TRIPLE PER PATTERN
    # --------------------------------------------------------------------------
    subgraph: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    pattern_counts: Dict[Tuple[str, str, str], int] = {}

    def try_add_triple(s_node: Any, p_node: Any, o_node: Any, match_type: str) -> bool:
        s_str = str(s_node)
        p_str = str(p_node)
        o_str = str(o_node)
        key = (s_str, p_str, o_str)

        if key in seen:
            return False

        s_sig = get_node_class_signature(s_node)
        p_sig = p_str
        o_sig = get_node_class_signature(o_node)
        pattern = (s_sig, p_sig, o_sig)

        if pattern_counts.get(pattern, 0) >= max_samples_per_pattern:
            return False

        seen.add(key)
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        subgraph.append({
            "subject": s_str,
            "predicate": p_str,
            "object": o_str,
            "match_type": match_type,
            "rdf_triple": (s_node, p_node, o_node),
            "pattern": pattern
        })
        return True

    # --------------------------------------------------------------------------
    # 3. SELEZIONA TUTTE LE TRIPLE CONTENENTI ALMENO UN INDIVIDUO / LETTERALE SELEZIONATO
    # --------------------------------------------------------------------------
    for ind_uri in selected_individuals:
        for s, p, o in g:
            if str(s) == ind_uri:
                try_add_triple(s, p, o, "individual_selected")
            elif str(o) == ind_uri:
                try_add_triple(s, p, o, "individual_selected")

    for lit_val in selected_literals:
        lit_lower = lit_val.lower()
        for s, p, o in g:
            if isinstance(o, Literal) and lit_lower in str(o).lower():
                try_add_triple(s, p, o, "literal_selected")

    # --------------------------------------------------------------------------
    # 2. CAMPIONA MASSIMO 3 TRIPLE PER CLASSE E PROPRIETÀ SELEZIONATA
    # --------------------------------------------------------------------------
    for c_uri in sorted(target_classes):
        c_instances = class_to_instances.get(c_uri, set())
        props_for_c = sorted(selected_class_properties.get(c_uri, set()))

        # Triple di tipo (rdf:type) per la classe
        if RDF is not None:
            for s, p, o in g:
                if p == RDF.type and str(o) == c_uri:
                    try_add_triple(s, p, o, "class_type_sample")

        # Triple per ciascuna proprietà associata alla classe
        for p_uri in props_for_c:
            if RDF is not None and p_uri == str(RDF.type):
                continue
            for s, p, o in g:
                if str(p) == p_uri:
                    s_str, o_str = str(s), str(o)
                    if s_str in c_instances or o_str in c_instances or s_str == c_uri or o_str == c_uri:
                        try_add_triple(s, p, o, "class_property_sample")

    print(f"--> [SOTTOGRAFO V12] Subgrafo estratto con successo: {len(subgraph)} triple totali (max {max_samples_per_pattern} per pattern classe-predicato-classe).", flush=True)

    if return_taxonomy:
        taxonomy_tree = extract_taxonomy_hierarchy(g, cache_key=str(source) if isinstance(source, str) else None)
        return g, subgraph, taxonomy_tree
    return subgraph


def is_valid_pname_local(local: str) -> bool:
    """Valida se una stringa è un PN_LOCAL valido per CURIE SPARQL/Turtle, inclusi caratteri con cancelletto (es. visitor#name)."""
    if not local:
        return True
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_\-\.]*(?:#|\\#)?[a-zA-Z0-9_\-\.]*$', local))


def format_term_curie(term_val: Any, prefixes: Dict[str, str]) -> str:
    """Formatta un termine RDF in sintassi CURIE con supporto a :visitor\\#name."""
    if Literal is not None and isinstance(term_val, Literal):
        val_escaped = str(term_val).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        if term_val.datatype:
            dt_curie = format_term_curie(str(term_val.datatype), prefixes)
            return f'"{val_escaped}"^^{dt_curie}'
        elif term_val.language:
            return f'"{val_escaped}"@{term_val.language}'
        return f'"{val_escaped}"'

    if BNode is not None and isinstance(term_val, BNode):
        return f"_:{term_val}"

    term_str = str(term_val).strip()
    if term_str.startswith("<") and term_str.endswith(">"):
        term_str = term_str[1:-1]

    if "^^" in term_str:
        val, dt = term_str.split("^^", 1)
        dt_clean = dt.strip("<> ")
        dt_curie = format_term_curie(dt_clean, prefixes)
        val_clean = val.strip().strip('"').strip("'")
        val_escaped = val_clean.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        return f'"{val_escaped}"^^{dt_curie}'

    if (term_str.startswith('"') and term_str.endswith('"')) or (term_str.startswith("'") and term_str.endswith("'")):
        return term_str

    if term_str.startswith("_:"):
        return term_str

    if not (term_str.startswith("http://") or term_str.startswith("https://") or term_str.startswith("urn:") or term_str.startswith("ftp://")):
        val_escaped = term_str.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        return f'"{val_escaped}"'

    sorted_pfx = sorted(
        [(p.rstrip(":"), ns) for p, ns in prefixes.items() if ns],
        key=lambda x: len(x[1]),
        reverse=True
    )

    for pfx, ns in sorted_pfx:
        if term_str.startswith(ns):
            local = term_str[len(ns):]
            if is_valid_pname_local(local):
                # Esegue l'escape del cancelletto in sintassi CURIE SPARQL/Turtle (es. :visitor\#name)
                local_escaped = local.replace('\\#', '#').replace('#', r'\#')
                if pfx == "":
                    return f":{local_escaped}"
                return f"{pfx}:{local_escaped}"

    return f"<{term_str}>"


def save_subgraph_to_ttl(
    source_g: Graph,
    subgraph_triples: List[Dict[str, Any]],
    output_filepath: str,
    include_prefixes: bool = False
) -> str:
    """
    Salva il subgrafo estratto con triple formattate in CURIE (:visitor\\#name),
    escludendo l'header delle direttive @prefix direttamente dalla formazione del file
    per alleggerire il sottografo e il prompt LLM.
    I prefissi completi dell'ontologia rimangono salvati separatamente nel file '{onto_name}_prefixes.json'.
    """
    prefixes = extract_ontology_prefixes(source_g)

    if "" not in prefixes and ":" not in prefixes:
        all_uris = [t.get("subject", "") for t in subgraph_triples] + [t.get("predicate", "") for t in subgraph_triples] + [t.get("object", "") for t in subgraph_triples]
        for u in all_uris:
            if "valuenet/ontop/" in u:
                prefixes[""] = "http://valuenet/ontop/"
                break

    prefix_lines = []
    if include_prefixes:
        def prefix_sort_key(item):
            pfx, ns_uri = item
            ns_lower = ns_uri.lower()
            is_std = any(std in ns_lower for std in ["w3.org", "schema.org", "purl.org", "xmlns.com"])
            return (1 if is_std else 0, pfx)

        for pfx, ns_uri in sorted(prefixes.items(), key=prefix_sort_key):
            if pfx:
                prefix_lines.append(f"@prefix {pfx}: <{ns_uri}> .")
            elif ns_uri:
                prefix_lines.append(f"@prefix : <{ns_uri}> .")

    prefix_header = "\n".join(prefix_lines) + ("\n\n" if prefix_lines else "")

    triple_lines = []
    for t in subgraph_triples:
        s_val = t["subject"]
        p_val = t["predicate"]
        o_val = t["object"]

        s_curie = format_term_curie(s_val, prefixes)
        p_curie = format_term_curie(p_val, prefixes)
        o_curie = format_term_curie(o_val, prefixes)

        triple_lines.append(f"{s_curie} {p_curie} {o_curie} .")

    full_content = prefix_header + "\n".join(triple_lines) + ("\n" if triple_lines else "")

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"[OK] Subgrafo salvato in: '{output_filepath}' (Totale triple: {len(triple_lines):,}, prefissi header inclusi: {include_prefixes})", flush=True)
    return output_filepath


# ==============================================================================
# PIPELINE COMPLETA PREPROCESSING V12
# ==============================================================================

def run_pipeline_for_ontology(
    onto_name: str,
    onto_path: str,
    query_test: str,
    output_dir: Optional[str] = None,
    output_subgraph_path: Optional[str] = None,
    include_prefixes_in_subgraph: bool = False
) -> Dict[str, Any]:
    """
    Esegue la pipeline di preprocessing v12:
    - 1. Estrazione termini dall'ontologia e indicizzazione class_to_instances / instance_to_classes (1 sola volta)
    - 2. Estrazione e salvataggio della tassonomia gerarchica (con cache a due livelli, eseguita PRIMA del sottografo)
    - 3. Information Extraction con GLiNER (zero-shot su query naturale)
    - 4. Estrazione del sottografo dinamico per la query (zero re-parsing, riutilizzo delle mappe pronte)
    - 5. Serializzazione del sottografo in formato Turtle (.ttl con sintassi CURIE)
    """
    print(f"\n" + "=" * 70, flush=True)
    print(f" PIPELINE PREPROCESSING V12: {onto_name}", flush=True)
    print(f" Ontologia   : {onto_path}", flush=True)
    print(f" Query Test  : '{query_test}'", flush=True)
    print("=" * 70 + "\n", flush=True)

    # 1. Parsing ontologia, estrazione termini e mappatura istanze (FATTO 1 SOLA VOLTA)
    entities, properties, g, class_to_instances, instance_to_classes = extract_classified_terms(
        source_path=onto_path,
        output_prefix=onto_name.lower(),
        return_mappings=True
    )
    subclass_parents, _ = extract_subclass_hierarchy(g)

    # 2. Estrazione Tassonomia (Statica, con cache a due livelli: eseguita PRIMA del sottografo)
    taxonomy_tree = extract_taxonomy_hierarchy(g, cache_key=onto_path)

    # 3. Information Extraction con GLiNER (knowledgator/gliner-relex-large-v1.0)
    selected_classes, selected_individuals_uri, selected_literals, gliner_ents = extract_entities_and_classes_gliner(
        query=query_test,
        entities=entities,
        properties=properties,
        subclass_parents=subclass_parents
    )

    print("\n--- RISULTATO ESTRAZIONE GLINER (PREPROCESSING V12) ---")
    print(f"  Entità estratte GLiNER : {gliner_ents}")
    print(f"  Classi Selezionate     : {selected_classes}")
    print(f"  Individui URI          : {selected_individuals_uri}")
    print(f"  Letterali / Valori     : {selected_literals}")

    # 4. Estrazione Subgrafo Dinamico (riceve g, class_to_instances e instance_to_classes già pronti)
    subgraph_triples = extract_subgraph_v12(
        source=g,
        selected_classes=selected_classes,
        selected_individuals=selected_individuals_uri,
        selected_literals=selected_literals,
        class_to_instances=class_to_instances,
        instance_to_classes=instance_to_classes,
        subclass_parents=subclass_parents,
        max_samples_per_pattern=3,
        return_taxonomy=False
    )

    # 5. Salvataggio del Subgrafo in formato Turtle
    if output_subgraph_path is None:
        output_subgraph_path = f"{onto_name.lower()}_t_subgraph.ttl"

    final_path = save_subgraph_to_ttl(
        g,
        subgraph_triples,
        output_subgraph_path,
        include_prefixes=include_prefixes_in_subgraph
    )

    return {
        "subgraph_ttl_path": final_path,
        "triples_count": len(subgraph_triples),
        "selected_classes": list(selected_classes),
        "selected_individuals": list(selected_individuals_uri),
        "selected_literals": list(selected_literals),
        "gliner_entities": gliner_ents,
        "taxonomy": taxonomy_tree
    }


run_pipeline_for_ontology_v12 = run_pipeline_for_ontology
run_pipeline_for_ontology_v11 = run_pipeline_for_ontology
run_pipeline_for_ontology_v10 = run_pipeline_for_ontology


if __name__ == "__main__":
    # Test su Spider4SPARQL (Concert_Singer)
    cs_ttl, json_path, cs_questions = get_concert_singer_dataset("spider4sparql")
    cs_query = "Show countries where a singer above age 40 and a singer below 30 are from."

    res_cs = run_pipeline_for_ontology(
        onto_name="Concert_Singer",
        onto_path=cs_ttl,
        query_test=cs_query
    )

    print("\n" + "=" * 70)
    print(" TASSONOMIA ESTRATTA IN FORMA GERARCHICA (FORMATO 1 - V12):")
    print("=" * 70)
    print(json.dumps(res_cs["taxonomy"], indent=2, ensure_ascii=False))
