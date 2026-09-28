# Source adapters retained from the research implementation; see NOTICE.
"""Render the already-audited local union; no network or remote loading scripts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path

from .data import file_hash, json_rows, rows, write_json
from .schema import typed_record




def from_kev(row):
    return typed_record(row, source=row.get("_meta", {}).get("source", "kev"))


def from_typed(row):
    decode = lambda x: json.loads(x) if isinstance(x, str) else x
    data = {"state": decode(row["state"]), "questions": decode(row["questions"])}
    return typed_record(data, source="typed_decisions/" + row["workflow"],
                        record_id=row["id"], gold=decode(row["gold"]))


def from_nimble(row):
    data = row["input"]
    if len(data["questions"]) != 1:
        raise ValueError("Nimble adapter expects one labelled decision per record")
    qid = next(iter(data["questions"]))
    return typed_record(data, source="nimble/" + row["domain"], record_id="nimble/" + row["id"],
                        group_id="nimble/" + row.get("source_family", row["id"]),
                        gold={qid: {"label": row["reference"]["target"]}})


def source_question(repo, item, label_names=None, *, seed=0, option_cap=16):
    label = item.get("label")
    text = item.get("text", "")
    if repo == "google/boolq":
        state = item["passage"]
        q = {"type": "noul", "instructions": item["question"], "label": item["answer"]}
    elif repo == "nyu-mll/multi_nli":
        if label == -1:
            raise ValueError("unlabelled MNLI row")
        names = ["entailment", "neutral", "contradiction"]
        state = {"premise": item["premise"], "hypothesis": item["hypothesis"]}
        q = {"type": "choice", "instructions": "How does the hypothesis relate to the premise?",
             "criteria": {"entailment": "Follows from the premise", "neutral": "Not established either way",
                          "contradiction": "Conflicts with the premise"}, "label": names[int(label)]}
    elif repo in {"SetFit/sst5", "Yelp/yelp_review_full", "SetFit/amazon_reviews_multi_en"}:
        levels = (["very negative", "negative", "neutral", "positive", "very positive"]
                  if repo == "SetFit/sst5" else [f"{i} star rating" for i in range(1, 6)])
        state = text
        q = {"type": "score", "instructions": "Rate the sentiment." if repo == "SetFit/sst5" else "Which rating did the reviewer give?",
             "criteria": levels, "label": int(label)}
    elif repo == "stanfordnlp/imdb":
        state = text
        q = {"type": "choice", "instructions": "What is the sentiment of this movie review?",
             "criteria": {"negative": "Negative review", "positive": "Positive review"},
             "label": ["negative", "positive"][int(label)]}
    else:
        if repo == "fancyzhx/ag_news":
            names = ["world", "sports", "business", "science_and_technology"]
            state, instruction = text, "Which news topic best fits this article?"
        elif repo == "fancyzhx/dbpedia_14":
            names = ["company", "educational_institution", "artist", "athlete", "office_holder", "transport",
                     "building", "natural_place", "village", "animal", "plant", "album", "film", "written_work"]
            state, instruction = {"title": item["title"], "content": item["content"]}, "Classify this entity."
        elif repo == "CogComp/trec":
            names = ["abbreviation", "entity", "description", "human", "location", "numeric"]
            label = item["coarse_label"]
            state, instruction = text, "Which answer type is requested by the question?"
        elif repo == "legacy-datasets/banking77":
            if not label_names:
                raise ValueError("Banking77 class names must come from the pinned Arrow metadata")
            names = label_names
            state, instruction = text, "Which banking intent matches the customer's request?"
        else:
            raise ValueError(f"no source renderer for {repo}")
        gold = names[int(label)]
        # A deterministic shortlist changes the task explicitly; always retain gold.
        if len(names) > option_cap:
            others = sorted((n for n in names if n != gold),
                            key=lambda n: hashlib.sha256(f"{seed}:{text}:{n}".encode()).digest())[:option_cap - 1]
            names = sorted([gold, *others])
        q = {"type": "choice", "instructions": instruction,
             "criteria": {n: n.replace("_", " ") for n in names}, "label": gold}
    return {"state": state, "questions": {"decision": q}}
