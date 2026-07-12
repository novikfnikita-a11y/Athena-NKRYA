"""
probe_corpus_resulttype_matrix.py

Эмпирически строит матрицу совместимости (corpus x resultType) для
эндпоинта get_word_portrait НКРЯ API, дергая реальный API строго по
ОДНОМУ resultType за раз для каждого корпуса из CorpusTypeEnum. Так
получилось быстро понять, какой именно resultType падает в каком именно корпусе,
вместо того чтобы гадать по 422-ошибке батчевого запроса. """



import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

from tools.nkrja_client import NKRJAClient

try:
    # Если whitelist уже сгенерирован прошлым прогоном - используем его,
    # чтобы для режима --preset schema не гадать корпус, а сразу брать
    # тот, что уже подтверждён как OK.
    from whitelist_generated import RESULTTYPE_CORPUS_WHITELIST as _KNOWN_WHITELIST
except ImportError:
    _KNOWN_WHITELIST = {}

#
# Справочники (синхронизированы с tools/registry.py)

ALL_CORPORA = [
    "MAIN", "SYNTAX", "PAPER", "REGIONAL", "PARA", "MULTI", "SCHOOL",
    "DIALECT", "POETIC", "SPOKEN", "ACCENT", "MURCO", "MULTIPARC_RUS",
    "MULTIPARC", "OLD_RUS", "BIRCHBARK", "MID_RUS", "ORTHLIB", "PANCHRON",
    "KIDS", "CLASSICS", "BLOGS", "EPIGRAPHICA", "GICR",
]

# Корпуса, которые реально фигурируют в логике выбора PLANNER_SYSTEM_PROMPT
# (см. LLM/prompts.py). Остальные 14 - экзотика, которую planner практически
#
# ФИКС: PANCHRON раньше отсутствовал здесь, хотя он есть в CORPUS_TYPE_ENUM_DESCRIPTION
# как "диахронический корпус для анализа изменений сквозь века" - и LLM реально
# выбирает его для вопросов об этимологии/истории слова (см. planner.py). Из-за
# отсутствия PANCHRON в этом списке whitelist_generated.py генерировался без него
# вообще, и api_orchestrator.py резал ЛЮБОЙ resultType для corpus=PANCHRON, даже
# те, что бэкенд реально поддерживает (см. matrix_raw_results.json).

CORE_CORPORA = [
    "MAIN", "POETIC", "SPOKEN", "OLD_RUS", "MID_RUS",
    "BIRCHBARK", "BLOGS", "GICR", "CLASSICS", "KIDS", "PANCHRON",
]


ALL_RESULT_TYPES = [
    "PORTRAIT_WORD_INFO",
    "PORTRAIT_CONCORDANCE",
    "PORTRAIT_STATS",
    "PORTRAIT_SKETCH",
    "PORTRAIT_FREQUENCY",
    "PORTRAIT_SIMILAR",
    "PORTRAIT_MORPHEME",
    "PORTRAIT_WORDFORMS",
    "PORTRAIT_COGNATES",
    "PORTRAIT_FIRST_MENTION",
    "PORTRAIT_MEANING",
]

# Обязательные доп. параметры для некоторых resultType (см. registry.py)
EXTRA_PARAMS = {
    "PORTRAIT_STATS": {"statFields": ["created"]},
    "PORTRAIT_SIMILAR": {"similarCategories": ["all"]},
}


FIELD_BY_RESULT_TYPE = {
    "PORTRAIT_WORD_INFO": "propsData",
    "PORTRAIT_CONCORDANCE": "concordanceData",
    "PORTRAIT_STATS": "statsData",
    "PORTRAIT_SKETCH": "sketchData",
    "PORTRAIT_FREQUENCY": "frequencyData",
    "PORTRAIT_SIMILAR": "similarData",
    "PORTRAIT_MORPHEME": "morphemeData",
    "PORTRAIT_WORDFORMS": "wordformsData",
    "PORTRAIT_COGNATES": "cognatesData",
    "PORTRAIT_FIRST_MENTION": "firstMentionData",
    "PORTRAIT_MEANING": "meaningData",
}

DEBUG = False  # поставьте True, чтобы печатать сырой JSON-ответ на каждую пару

RAW_RESULTS_FILE = "matrix_raw_results.json"
WHITELIST_OUTPUT_FILE = "whitelist_generated.py"
RAW_DUMPS_DIR = "raw_dumps"

# По данным openapi_NKRYA.py у PortraitResult вообще нет полей cognatesData/meaningData -
# это не "у леммы нет данных", а не реализовано на бэкенде в этой версии API.
# Пробовать их в режиме schema бессмысленно, только жжём rate-limit budget.
DEAD_RESULT_TYPES = ["PORTRAIT_COGNATES", "PORTRAIT_MEANING"]




def probe_pair(
    client: NKRJAClient,
    corpus: str,
    result_type: str,
    lemma: str,
    pos: str | None,
    max_retries_429: int = 4,
    backoff_base_429: float = 20.0,
    retry_5xx: int = 1,
    retry_5xx_wait: float = 4.0,
    dump_raw_dir: Path | None = None,
) -> dict:

    extra = EXTRA_PARAMS.get(result_type, {})
    attempt_429 = 0
    attempt_5xx = 0

    while True:
        try:
            response = client.get_word_portrait(
                lemma=lemma,
                corpus=corpus,
                resultType=[result_type],
                pos=pos,
                **extra,
            )
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            body = (e.response.text or "")[:300] if e.response is not None else str(e)

            if status_code == 429 and attempt_429 < max_retries_429:
                attempt_429 += 1
                wait = backoff_base_429 * (2 ** (attempt_429 - 1))
                print(f"\n      [429 rate limit] бэкофф {wait:.0f}с, попытка {attempt_429}/{max_retries_429}...", end="", flush=True)
                time.sleep(wait)
                continue

            if status_code in (500, 502, 503, 504) and attempt_5xx < retry_5xx:
                attempt_5xx += 1
                print(f"\n      [{status_code}] короткая повторная попытка {attempt_5xx}/{retry_5xx} через {retry_5xx_wait:.0f}с (вероятно неподдерживаемая комбинация, не троттлинг)...", end="", flush=True)
                time.sleep(retry_5xx_wait)
                continue

            return {"status": f"ERROR_{status_code or '?'}", "detail": body}
        except requests.exceptions.RequestException as e:
            return {"status": "NETWORK_ERROR", "detail": str(e)}
        except Exception as e:
            return {"status": "EXCEPTION", "detail": str(e)}

        if DEBUG:
            print(json.dumps(response, ensure_ascii=False, indent=2)[:1000])

        field = FIELD_BY_RESULT_TYPE.get(result_type)
        data = response.get(field) if isinstance(response, dict) else None

        # НОВОЕ: сохраняем ПОЛНОЕ сырое тело ответа на диск, а не только факт
        # наличия/отсутствия поля. Именно этого не хватало в предыдущем прогоне -
        # matrix_raw_results.json содержал только status/detail, реальный JSON
        # нигде не оседал, поэтому по нему нельзя было понять форму propsData/
        # statsData/frequencyData/wordformsData/firstMentionData/sketchData -
        # все они в openapi объявлены как "additionalProperties: true", то есть
        # схема сама по себе не документирует их внутреннюю структуру.
        if dump_raw_dir is not None and isinstance(response, dict):
            dump_raw_dir.mkdir(parents=True, exist_ok=True)
            dump_path = dump_raw_dir / f"{corpus}__{result_type}.json"
            dump_path.write_text(
                json.dumps(response, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if data:
            return {"status": "OK", "detail": None}
        return {"status": "EMPTY", "detail": f"поле '{field}' пустое/отсутствует в ответе"}


def load_existing_results(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[!] Не удалось разобрать {path}, начинаем с чистого листа.")
    return {}


def save_results(results: dict, path: Path) -> None:
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def make_key(corpus: str, result_type: str) -> str:
    return f"{corpus}|{result_type}"


STATUS_SYMBOLS = {
    "OK": "OK",
    "EMPTY": "--",
    "NETWORK_ERROR": "NET",
}


def status_symbol(status: str) -> str:
    if status in STATUS_SYMBOLS:
        return STATUS_SYMBOLS[status]
    if status.startswith("ERROR_"):
        return status.replace("ERROR_", "E")
    return "??"


def print_matrix(results: dict, corpora: list, result_types: list) -> None:
    col_width = 6
    header = "CORPUS".ljust(14) + "".join(rt.replace("PORTRAIT_", "")[:col_width].ljust(col_width + 1) for rt in result_types)
    print("\n" + header)
    print("-" * len(header))
    for corpus in corpora:
        row = corpus.ljust(14)
        for rt in result_types:
            key = make_key(corpus, rt)
            status = results.get(key, {}).get("status", "?")
            row += status_symbol(status).ljust(col_width + 1)
        print(row)
    print("\nЛегенда: OK=данные есть, --=запрос прошёл но данные пустые, "
          "E422/E4xx/E5xx=HTTP-ошибка, NET=сетевая ошибка, ??=не протестировано\n")


def generate_whitelist_file(results: dict, result_types: list, lemma: str, path: Path) -> None:
    whitelist = {rt: [] for rt in result_types}
    empty_map = {rt: [] for rt in result_types}
    error_map = {rt: [] for rt in result_types}

    for key, res in results.items():
        corpus, rt = key.split("|", 1)
        if rt not in whitelist:
            continue
        status = res.get("status")
        if status == "OK":
            whitelist[rt].append(corpus)
        elif status == "EMPTY":
            empty_map[rt].append(corpus)
        elif status not in ("?",):
            error_map[rt].append(corpus)

    lines = [
        "# Автоматически сгенерировано probe_corpus_resulttype_matrix.py",
        f"# Тестовая лемма: {lemma!r}",
        "# ВНИМАНИЕ: 'EMPTY' может означать как 'фича не поддерживается корпусом',",
        "# так и 'этой леммы просто нет в данном корпусе'. Перед тем как класть",
        "# EMPTY-корпуса в чёрный список, имеет смысл перепроверить на 1-2 других леммах.",
        "",
        "RESULTTYPE_CORPUS_WHITELIST = {",
    ]
    for rt, corpora_list in whitelist.items():
        lines.append(f"    {rt!r}: {sorted(corpora_list)!r},")
    lines.append("}")
    lines.append("")
    lines.append("# Корпуса, где запрос прошёл (200 OK), но данные пусты - требуют перепроверки:")
    lines.append("RESULTTYPE_CORPUS_EMPTY = {")
    for rt, corpora_list in empty_map.items():
        lines.append(f"    {rt!r}: {sorted(corpora_list)!r},")
    lines.append("}")
    lines.append("")
    lines.append("# Корпуса, где запрос упал с ошибкой (HTTP 4xx/5xx или сеть):")
    lines.append("RESULTTYPE_CORPUS_ERRORS = {")
    for rt, corpora_list in error_map.items():
        lines.append(f"    {rt!r}: {sorted(corpora_list)!r},")
    lines.append("}")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Построение матрицы совместимости corpus x resultType для get_word_portrait")
    parser.add_argument("--lemma", default="мать", help="Тестовая лемма (по умолчанию распространённое существительное)")
    parser.add_argument("--pos", default=None, help="Часть речи, если нужно зафиксировать (например 'S' для PORTRAIT_WORDFORMS)")
    parser.add_argument("--preset", choices=["core", "all", "schema"], default="core",
                         help="core = только корпуса, реально используемые planner'ом (10 шт, быстрее); "
                              "all = все 24 корпуса из CorpusTypeEnum; "
                              "schema = МИНИМАЛЬНЫЙ прогон для написания парсера: ровно один OK-корпус "
                              "на каждый resultType (берётся из уже сгенерированного whitelist_generated.py, "
                              "фоллбек - MAIN), мёртвые resultType (COGNATES/MEANING) пропускаются")
    parser.add_argument("--corpora", default=None, help="Явный список корпусов через запятую - переопределяет --preset")
    parser.add_argument("--result-types", default=None, help="Список resultType через запятую (по умолчанию - все)")
    parser.add_argument("--delay", type=float, default=2.0, help="Базовая пауза между запросами в секундах")
    parser.add_argument("--jitter", type=float, default=1.0, help="Случайная добавка к паузе 0..jitter секунд (сбивает регулярность запросов)")
    parser.add_argument("--batch-size", type=int, default=20, help="Через сколько запросов делать длинную паузу")
    parser.add_argument("--batch-pause", type=float, default=20.0, help="Длина длинной паузы между батчами, секунды")
    parser.add_argument("--max-retries-429", type=int, default=4, help="Сколько раз повторять запрос при 429 с экспоненциальным backoff'ом (реальный троттлинг)")
    parser.add_argument("--backoff-base", type=float, default=20.0, help="Базовое время ожидания при 429, секунды (растёт экспоненциально: base, base*2, base*4...)")
    parser.add_argument("--retry-5xx", type=int, default=1, help="Сколько раз коротко повторить запрос при 5xx (это обычно НЕ троттлинг, а неподдерживаемая комбинация corpus+resultType, поэтому по умолчанию всего 1 короткая попытка, не backoff)")
    parser.add_argument("--retry-5xx-wait", type=float, default=4.0, help="Фиксированная пауза перед повтором при 5xx, секунды")
    parser.add_argument("--resume", action="store_true", help="Продолжить с места, где сохранён matrix_raw_results.json, пропуская уже протестированные пары")
    parser.add_argument("--dry-run", action="store_true", help="Только показать план (сколько запросов, какие пары), не дёргать API")
    parser.add_argument("--output", default=RAW_RESULTS_FILE, help="Файл для сырых результатов")
    parser.add_argument("--dump-raw", action="store_true",
                         help="Сохранять ПОЛНОЕ тело каждого успешного (не EMPTY/ERROR) ответа в --dump-raw-dir. "
                              "Именно это нужно, чтобы написать парсер под propsData/statsData/frequencyData/"
                              "wordformsData/firstMentionData/sketchData - их форма не описана в openapi (additionalProperties: true).")
    parser.add_argument("--dump-raw-dir", default=RAW_DUMPS_DIR, help="Папка для сырых JSON-дампов (используется вместе с --dump-raw)")
    parser.add_argument("--include-dead", action="store_true",
                         help="Всё равно пробовать PORTRAIT_COGNATES/PORTRAIT_MEANING в режиме --preset schema, "
                             "хотя по openapi у PortraitResult нет соответствующих полей вообще (на случай, "
                             "если бэкенд обновили и фичи наконец реализовали)")
    return parser.parse_args()


def build_schema_pairs(include_dead: bool) -> list[tuple[str, str]]:
    """
    Строит МИНИМАЛЬНЫЙ набор пар для снятия схемы: ровно один заведомо
    OK-корпус на каждый resultType (по данным уже сгенерированного
    whitelist_generated.py), вместо полного перебора corpus x resultType.
    Не имеет смысла снимать схему одного и того же JSON-поля дважды
    из разных корпусов - форма ответа не зависит от корпуса, зависит
    только от resultType.
    """
    result_types = [rt for rt in ALL_RESULT_TYPES if include_dead or rt not in DEAD_RESULT_TYPES]

    pairs = []
    for rt in result_types:
        known_ok = _KNOWN_WHITELIST.get(rt, [])
        corpus = known_ok[0] if known_ok else "MAIN"
        pairs.append((corpus, rt))
    return pairs


def main():
    args = parse_args()

    if args.preset == "schema":
        corpora = None  # не используется в этом режиме, пары строятся отдельно
        result_types = None
        pairs = build_schema_pairs(include_dead=args.include_dead)
        print("[preset=schema] Снимаем схему: по одному OK-корпусу на каждый resultType.")
        if not args.include_dead:
            print(f"[preset=schema] Пропущены заведомо мёртвые resultType (нет полей в PortraitResult): {DEAD_RESULT_TYPES}")
    else:
        if args.corpora:
            corpora = args.corpora.split(",")
        else:
            corpora = CORE_CORPORA if args.preset == "core" else ALL_CORPORA
        result_types = args.result_types.split(",") if args.result_types else ALL_RESULT_TYPES
        corpora = [c.strip().upper() for c in corpora]
        result_types = [r.strip().upper() for r in result_types]
        pairs = [(c, rt) for c in corpora for rt in result_types]

    print(f"Запланировано пар (corpus x resultType): {len(pairs)}")
    print(f"Лемма: {args.lemma!r} | pos: {args.pos!r} | preset: {args.preset}")
    print(f"delay: {args.delay}s (+jitter до {args.jitter}s) | батч-пауза {args.batch_pause}s каждые {args.batch_size} запросов")
    if args.dump_raw:
        print(f"[dump-raw] Сырые тела ответов будут сохранены в: {args.dump_raw_dir}/")
    est_seconds = len(pairs) * (args.delay + args.jitter / 2) + (len(pairs) / args.batch_size) * args.batch_pause
    print(f"Ориентировочное время прогона (без учёта 429-бэкоффов): ~{est_seconds / 60:.1f} мин\n")

    if args.dry_run:
        for c, rt in pairs:
            print(f"  would probe: {c:14s} {rt}")
        return

    output_path = Path(args.output)
    results = load_existing_results(output_path) if args.resume else {}

    if args.resume:
        already = sum(1 for c, rt in pairs if make_key(c, rt) in results)
        print(f"[resume] Уже есть результатов: {already}/{len(pairs)}\n")

    try:
        client = NKRJAClient()
        auth_status = client.check_auth()
        print(f"[auth] check_auth: {auth_status}\n")
    except Exception as e:
        print(f"[!] Не удалось проверить авторизацию перед стартом: {e}")
        print("[!] Продолжаем на свой страх и риск...\n")
        client = NKRJAClient()

    dump_raw_dir = Path(args.dump_raw_dir) if args.dump_raw else None

    done_count = 0
    try:
        for corpus, result_type in pairs:
            key = make_key(corpus, result_type)

            if args.resume and key in results:
                continue

            print(f"  [{done_count + 1}/{len(pairs)}] {corpus:14s} {result_type:26s} ... ", end="", flush=True)
            res = probe_pair(
                client, corpus, result_type, args.lemma, args.pos,
                max_retries_429=args.max_retries_429, backoff_base_429=args.backoff_base,
                retry_5xx=args.retry_5xx, retry_5xx_wait=args.retry_5xx_wait,
                dump_raw_dir=dump_raw_dir,
            )
            results[key] = res
            print(f"{res['status']}" + (f" ({res['detail'][:80]})" if res.get("detail") else ""))

            done_count += 1
            if done_count % 5 == 0:
                save_results(results, output_path)

            if done_count % args.batch_size == 0 and done_count < len(pairs):
                print(f"  -- батч-пауза {args.batch_pause:.0f}с ({done_count}/{len(pairs)} сделано) --")
                time.sleep(args.batch_pause)
            else:
                time.sleep(args.delay + random.uniform(0, args.jitter))

    except KeyboardInterrupt:
        print("\n[!] Прервано пользователем, сохраняю прогресс...")
    finally:
        save_results(results, output_path)
        print(f"\nСырые результаты сохранены в {output_path}")
        if args.dump_raw:
            print(f"Сырые JSON-тела сохранены в {args.dump_raw_dir}/ (по одному файлу на corpus__resultType.json)")

    if args.preset != "schema":
        print_matrix(results, corpora, result_types)

        whitelist_path = Path(WHITELIST_OUTPUT_FILE)
        generate_whitelist_file(results, result_types, args.lemma, whitelist_path)
        print(f"Готовый словарь для tools/registry.py сохранён в {whitelist_path}")


if __name__ == "__main__":
    main()