

from typing import Any




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


_PARSERS_BY_RESULT_TYPE = {
    "PORTRAIT_WORD_INFO": lambda raw: parse_word_info(raw),
    "PORTRAIT_FREQUENCY": lambda raw: parse_frequency(raw),
    "PORTRAIT_SIMILAR": lambda raw: parse_similar(raw),
    "PORTRAIT_MORPHEME": lambda raw: parse_morpheme(raw),
    "PORTRAIT_WORDFORMS": lambda raw: parse_wordforms(raw),
    "PORTRAIT_FIRST_MENTION": lambda raw: parse_first_mention(raw),
    "PORTRAIT_STATS": lambda raw: parse_stats(raw),
    "PORTRAIT_SKETCH": lambda raw: parse_sketch(raw),
    "PORTRAIT_CONCORDANCE": lambda raw: parse_concordance(raw),
}


def compress_word_portrait_response(raw_response: dict, requested_result_types: list[str] | None = None) -> dict:

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

    for result_type, parser in _PARSERS_BY_RESULT_TYPE.items():
        if requested_result_types is not None and result_type not in requested_result_types:
            continue
        raw_field = field_by_result_type[result_type]
        if raw_response.get(raw_field):
            compressed[result_type.lower()] = parser(raw_response)

    return compressed
