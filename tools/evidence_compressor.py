

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


def _extract_attr_values(value_list: list[dict]) -> list[Any]:

    out = []
    for v in value_list or []:
        if "valString" in v and v["valString"] is not None:
            out.append(v["valString"].get("v"))
        elif "valInt" in v and v["valInt"] is not None:
            out.append(v["valInt"].get("v"))
        elif "valPairStrings" in v and v["valPairStrings"] is not None:
            out.append((v["valPairStrings"].get("f"), v["valPairStrings"].get("s")))
        elif "valAuthor" in v and v["valAuthor"] is not None:
            d = v["valAuthor"].get("v", {})
            out.append(f"{d.get('year')}-{d.get('month')}-{d.get('day')}")
        elif "valDateRange" in v and v["valDateRange"] is not None:
            out.append(v["valDateRange"])  # структура сложная - отдаём как есть
        else:
            # неизвестный/пустой вариант - пропускаем, а не падаем
            continue
    return out


def _parsing_fields_to_dict(items: list[dict]) -> dict[str, Any]:

    result: dict[str, Any] = {}
    for item in items or []:
        for field in item.get("parsingFields", []):
            name = field.get("name")
            if not name:
                continue
            values = _extract_attr_values(field.get("value", []))
            if not values:
                continue
            result[name] = values[0] if len(values) == 1 else values
    return result




def parse_word_info(raw: dict) -> dict:
    props = raw.get("propsData")
    if not props:
        return {}
    return _parsing_fields_to_dict(props.get("items", []))




def parse_frequency(raw: dict) -> dict:
    return raw.get("frequencyData") or {}




def parse_similar(raw: dict, top_n: int = 10) -> list[dict]:
    similar = raw.get("similarData") or []
    out = []
    for entry in similar:
        values = sorted(entry.get("values", []), key=lambda x: x.get("weight", 0), reverse=True)[:top_n]
        out.append({"category": entry.get("category"), "words": values})
    return out




def parse_morpheme(raw: dict) -> list[dict]:
    return (raw.get("morphemeData") or {}).get("morphemes", [])




def parse_wordforms(raw: dict, top_n: int | None = None) -> list[dict]:
    values = (raw.get("wordformsData") or {}).get("values", [])
    out = []
    for v in values:
        wf = v.get("wfValue", {})
        freq = wf.get("freq", {})
        out.append({
            "case": (v.get("rowLabel") or {}).get("v"),
            "number": (v.get("columnLabel") or {}).get("v"),
            "form": wf.get("value"),
            "ipm": freq.get("ipm"),
            "category": freq.get("category"),
        })
    out.sort(key=lambda x: (x["ipm"] is None, -(x["ipm"] or 0)))
    return out[:top_n] if top_n else out



def parse_first_mention(raw: dict) -> dict:
    fm = raw.get("firstMentionData")
    if not fm:
        return {}
    info = _parsing_fields_to_dict((fm.get("info") or {}).get("items", []))
    out = {**info}
    if fm.get("redirectLemma"):
        out["redirect_lemma"] = fm["redirectLemma"]
    if fm.get("redirectCorpus"):
        out["redirect_corpus"] = fm["redirectCorpus"].get("type")
    return out




def parse_stats(raw: dict, top_n: int | None = None) -> list[dict]:
    stats = raw.get("statsData") or {}
    out = []
    for fs in stats.get("fieldStats", []):
        field_name = fs.get("field")
        bins = []
        for v in fs.get("values", []):
            key = v.get("key", {})
            key_str = None
            if "valString" in key and key["valString"]:
                key_str = key["valString"].get("v")
            bins.append({
                "period": key_str,
                "count": int(v.get("count", 0)),
                "total_count_in_corpus": int(v.get("totalCount", 0)),
            })
        bins.sort(key=lambda x: -x["count"])
        out.append({
            "field": field_name,
            "bins": bins[:top_n] if top_n else bins,
        })
    return out



def parse_sketch(raw: dict, top_n_per_relation: int = 5) -> dict:
    sketch = raw.get("sketchData") or {}
    relations = []
    for group in sketch.get("collocates", []):
        collocations = []
        for c in group.get("collocations", []):
            word = (c.get("collocate") or {}).get("valString", {}).get("v")
            metrics = {m["name"]: m["value"] for m in c.get("metrics", [])}
            collocations.append({"word": word, **metrics})
        collocations.sort(key=lambda x: x.get("dice", 0), reverse=True)
        relations.append({
            "relation": group.get("sketchSynRelation"),
            "collocates": collocations[:top_n_per_relation],
        })
    return {"lemma": sketch.get("lex"), "relations": relations}




def _reconstruct_sentence(sequence: dict) -> str:
    parts = []
    for w in sequence.get("words", []):
        text = w.get("text", "")
        if (w.get("displayParams") or {}).get("hit"):
            text = f"**{text.strip()}**"
        parts.append(text)
    return "".join(parts).strip()


def _extract_doc_date(doc_explain_info: dict | None) -> str | None:
    if not doc_explain_info:
        return None
    flat = _parsing_fields_to_dict(doc_explain_info.get("items", []))
    return flat.get("created")


def parse_concordance(raw: dict, top_k: int = 10, snippets_per_doc: int = 2) -> list[dict]:
    concordance = raw.get("concordanceData") or {}
    docs = []
    for group in concordance.get("groups", []):
        docs.extend(group.get("docs", []))

    examples = []
    round_index = 0
    while len(examples) < top_k and any(docs):
        added_this_round = False
        for doc in docs:
            if len(examples) >= top_k:
                break
            info = doc.get("info", {})
            snippet_groups = doc.get("snippetGroups", [])
            flat_snippets = [s for sg in snippet_groups for s in sg.get("snippets", [])]
            if round_index >= min(len(flat_snippets), snippets_per_doc):
                continue
            snippet = flat_snippets[round_index]
            sequences = snippet.get("sequences", [])
            sentence = " / ".join(_reconstruct_sentence(seq) for seq in sequences if seq.get("words"))
            if not sentence:
                continue
            examples.append({
                "text": sentence,
                "doc_title": info.get("title"),
                "doc_id": (info.get("source") or {}).get("docId"),
                "date": _extract_doc_date(info.get("docExplainInfo")),
            })
            added_this_round = True
        if not added_this_round:
            break
        round_index += 1

    return examples


def compress_word_portrait_response(
    raw_response: dict,
    requested_result_types: list[str] | None = None,
    *,
    max_items: int = 25,
    max_examples: int = 10,
) -> dict:

    if not isinstance(raw_response, dict):
        return {}

    compressed: dict[str, Any] = {}
    if raw_response.get("possiblePos"):
        compressed["possible_pos"] = raw_response["possiblePos"]

    field_by_result_type = {
        "PORTRAIT_WORD_INFO": "propsData",
        "PORTRAIT_CONCORDANCE": "concordanceData",
        "PORTRAIT_STATS": "statsData",
        "PORTRAIT_SKETCH": "sketchData",
        "PORTRAIT_FREQUENCY": "frequencyData",
        "PORTRAIT_SIMILAR": "similarData",
        "PORTRAIT_MORPHEME": "morphemeData",
        "PORTRAIT_WORDFORMS": "wordformsData",
        "PORTRAIT_FIRST_MENTION": "firstMentionData",
    }

    parsers_by_result_type = {
        "PORTRAIT_WORD_INFO": lambda raw: parse_word_info(raw),
        "PORTRAIT_FREQUENCY": lambda raw: parse_frequency(raw),
        "PORTRAIT_SIMILAR": lambda raw: parse_similar(raw, top_n=max_items),
        "PORTRAIT_MORPHEME": lambda raw: parse_morpheme(raw)[:max_items],
        "PORTRAIT_WORDFORMS": lambda raw: parse_wordforms(raw, top_n=max_items),
        "PORTRAIT_FIRST_MENTION": lambda raw: parse_first_mention(raw),
        "PORTRAIT_STATS": lambda raw: parse_stats(raw, top_n=max_items),
        "PORTRAIT_SKETCH": lambda raw: parse_sketch(
            raw, top_n_per_relation=max_items
        ),
        "PORTRAIT_CONCORDANCE": lambda raw: parse_concordance(
            raw, top_k=max_examples
        ),
    }

    for result_type, parser in parsers_by_result_type.items():
        if requested_result_types is not None and result_type not in requested_result_types:
            continue
        raw_field = field_by_result_type[result_type]
        if raw_response.get(raw_field):
            compressed[result_type.lower()] = parser(raw_response)

    return compressed


@dataclass(frozen=True, slots=True)
class CompressionLimits:
    """Hard limits applied before evidence enters shared graph state."""

    max_items_per_collection: int = 25
    max_examples: int = 10
    max_string_chars: int = 2_000
    max_payload_chars: int = 50_000

    def __post_init__(self) -> None:
        if min(
            self.max_items_per_collection,
            self.max_examples,
            self.max_string_chars,
            self.max_payload_chars,
        ) < 1:
            raise ValueError("compression limits must be positive")


def raw_response_hash(raw_response: Any) -> str:
    canonical = json.dumps(
        raw_response,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _bounded_value(
    value: Any,
    *,
    item_limit: int,
    string_limit: int,
) -> tuple[Any, bool]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        truncated = False
        items = sorted(value.items(), key=lambda pair: str(pair[0]))
        for key, item in items[:item_limit]:
            bounded, child_truncated = _bounded_value(
                item,
                item_limit=item_limit,
                string_limit=string_limit,
            )
            result[str(key)] = bounded
            truncated = truncated or child_truncated
        return result, truncated or len(items) > item_limit
    if isinstance(value, (list, tuple)):
        result_list = []
        truncated = len(value) > item_limit
        for item in value[:item_limit]:
            bounded, child_truncated = _bounded_value(
                item,
                item_limit=item_limit,
                string_limit=string_limit,
            )
            result_list.append(bounded)
            truncated = truncated or child_truncated
        return result_list, truncated
    if isinstance(value, str) and len(value) > string_limit:
        return value[:string_limit], True
    return value, False


def _fit_payload(data: Any, limits: CompressionLimits) -> tuple[Any, bool]:
    item_limit = limits.max_items_per_collection
    string_limit = limits.max_string_chars
    while True:
        bounded, truncated = _bounded_value(
            data,
            item_limit=item_limit,
            string_limit=string_limit,
        )
        encoded = json.dumps(
            bounded,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if len(encoded) <= limits.max_payload_chars:
            return bounded, truncated
        if item_limit == 1 and string_limit <= 64:
            return {"summary": "payload omitted by size limit"}, True
        item_limit = max(1, item_limit // 2)
        string_limit = max(64, string_limit // 2)


def _word_portrait_data(
    raw_response: dict[str, Any],
    requested_result_types: list[str] | None,
    limits: CompressionLimits,
) -> tuple[dict[str, Any], bool]:
    requested = set(requested_result_types or ())
    data = compress_word_portrait_response(
        raw_response,
        requested_result_types,
        max_items=limits.max_items_per_collection,
        max_examples=limits.max_examples,
    )
    truncated = False
    if not requested or "PORTRAIT_SIMILAR" in requested:
        similar = raw_response.get("similarData") or []
        truncated = truncated or len(similar) > limits.max_items_per_collection
        truncated = truncated or any(
            len(entry.get("values", [])) > limits.max_items_per_collection
            for entry in similar
        )
    if not requested or "PORTRAIT_MORPHEME" in requested:
        morphemes = (raw_response.get("morphemeData") or {}).get("morphemes", [])
        truncated = truncated or len(morphemes) > limits.max_items_per_collection
    if not requested or "PORTRAIT_WORDFORMS" in requested:
        wordforms = (raw_response.get("wordformsData") or {}).get("values", [])
        truncated = truncated or len(wordforms) > limits.max_items_per_collection
    if not requested or "PORTRAIT_STATS" in requested:
        fields = (raw_response.get("statsData") or {}).get("fieldStats", [])
        truncated = truncated or len(fields) > limits.max_items_per_collection
        truncated = truncated or any(
            len(field.get("values", [])) > limits.max_items_per_collection
            for field in fields
        )
    if not requested or "PORTRAIT_SKETCH" in requested:
        relations = (raw_response.get("sketchData") or {}).get("collocates", [])
        truncated = truncated or len(relations) > limits.max_items_per_collection
        truncated = truncated or any(
            len(relation.get("collocations", []))
            > limits.max_items_per_collection
            for relation in relations
        )
    if not requested or "PORTRAIT_CONCORDANCE" in requested:
        truncated = truncated or _concordance_count(raw_response) > limits.max_examples
    return data, truncated


def _concordance_count(raw_response: dict[str, Any]) -> int:
    concordance = raw_response.get("concordanceData") or {}
    return sum(
        len(snippet_group.get("snippets", []))
        for group in concordance.get("groups", [])
        for doc in group.get("docs", [])
        for snippet_group in doc.get("snippetGroups", [])
    )


def _concordance_data(
    raw_response: dict[str, Any], limits: CompressionLimits
) -> tuple[dict[str, Any], bool]:
    if "concordanceData" in raw_response:
        examples = parse_concordance(raw_response, top_k=limits.max_examples)
        truncated = _concordance_count(raw_response) > len(examples)
    elif "groups" in raw_response:
        wrapped = {"concordanceData": raw_response}
        examples = parse_concordance(wrapped, top_k=limits.max_examples)
        truncated = _concordance_count(wrapped) > len(examples)
    else:
        examples = raw_response.get("examples", raw_response)
        truncated = isinstance(examples, list) and len(examples) > limits.max_examples
        if isinstance(examples, list):
            examples = examples[: limits.max_examples]
    return {"concordance": examples}, truncated


def compress_response(
    tool: str,
    raw_response: Any,
    *,
    params: Mapping[str, Any] | None = None,
    limits: CompressionLimits | None = None,
) -> dict[str, Any]:
    """Return one bounded, reproducible schema for every NKRJA response.

    Numeric values and their surrounding keys are never rewritten.  Lists and
    strings may be truncated, while examples retain document metadata and the
    hash always identifies the complete source response.
    """

    current_limits = limits or CompressionLimits()
    safe_params = dict(params or {})
    corpus = safe_params.get("corpus")
    lemma = safe_params.get("lemma")
    parser_truncated = False

    if isinstance(raw_response, dict):
        if tool == "get_word_portrait":
            data, parser_truncated = _word_portrait_data(
                raw_response,
                list(safe_params.get("resultType", [])) or None,
                current_limits,
            )
        elif tool == "get_simple_concordance":
            data, parser_truncated = _concordance_data(
                raw_response, current_limits
            )
        elif tool == "get_corpus_stats":
            data = {"corpus_statistics": raw_response}
        elif tool == "get_corpus_config":
            data = {"corpus_configuration": raw_response}
        elif tool == "get_corpus_attributes":
            data = {"corpus_attributes": raw_response}
        elif tool == "get_attribute_values":
            data = {"attribute_values": raw_response}
        elif tool == "get_sketch_difference":
            data = {"sketch_difference": raw_response}
        else:
            data = {"result": raw_response}
    else:
        data = {"result": raw_response}

    bounded, bounded_truncated = _fit_payload(data, current_limits)
    return {
        "schema_version": 1,
        "tool": tool,
        "corpus": corpus,
        "lemma": lemma,
        "data": bounded,
        "truncated": parser_truncated or bounded_truncated,
        "raw_response_hash": raw_response_hash(raw_response),
    }


__all__ = [
    "CompressionLimits",
    "compress_response",
    "compress_word_portrait_response",
    "parse_concordance",
    "parse_frequency",
    "parse_first_mention",
    "parse_morpheme",
    "parse_similar",
    "parse_sketch",
    "parse_stats",
    "parse_word_info",
    "parse_wordforms",
    "raw_response_hash",
]
