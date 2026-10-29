from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}


def qn(tag: str) -> str:
    prefix, local = tag.split(":")
    if prefix != "w":
        raise ValueError(f"Unsupported prefix: {prefix}")
    return f"{{{W_NS}}}{local}"


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(node.text or "" for node in paragraph.findall(".//w:t", NS))


def clear_runs(paragraph: ET.Element) -> None:
    for child in list(paragraph):
        if child.tag != qn("w:pPr"):
            paragraph.remove(child)


def append_run(paragraph: ET.Element, text: str) -> None:
    run = ET.Element(qn("w:r"))
    t = ET.SubElement(run, qn("w:t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text
    paragraph.append(run)


def set_paragraph_text(paragraph: ET.Element, text: str) -> None:
    clear_runs(paragraph)
    if text:
        append_run(paragraph, text)


def rewrite_section(paragraphs: list[ET.Element]) -> None:
    replacements = {
        179: "The retrieval logic is mainly implemented in `09_candidate_fusion.py`. The script reads the prepared user, business, review, cluster, profile, and semantic assets, then builds a multi-route candidate pool for each bucket instead of relying on a single recommender output.",
        180: "`get_bucket_recall_limits()` and the bucket policy dictionaries control how many ALS, cluster, profile, popular, and pretrim candidates are allowed for each user bucket.",
        181: "`build_profile_candidates()` adds profile-vector, shared-tag, bridge-user, and optional multivector profile routes so that semantic preference signals can become actual recall candidates.",
        182: "The main loop in `main()` builds ALS candidates, cluster-popular candidates, global-popular fallback candidates, profile candidates, and context candidates, then unions them into one candidate surface.",
        183: "After unioning sources, the code aggregates by `user_idx` and `item_idx`, keeps route flags such as `has_als`, `has_cluster`, `has_profile`, and `has_popular`, and computes a fused `pre_score`.",
        184: "The final handoff is `candidates_pretrim.parquet`. This file is produced after layered pretrim logic protects strong head candidates, rescues useful challenger candidates, and limits the candidate list to a manageable size for Stage10.",
        186: "Stage10 is implemented mainly by `10_1_rank_train.py` for training and `10_2_rank_infer_eval.py` for inference and evaluation. The training script resolves the Stage09 candidate directory, attaches optional Stage10 feature builds, samples positives and negatives, trains the selected backend, and writes `rank_model.json`.",
        187: "Important code-level pieces include:",
        188: "`add_feature_columns()` constructs model features such as inverse ranks, source indicators, route priors, user-history flags, semantic scores, and cross features.",
        189: "`attach_stage10_v2_match_channels()`, `attach_stage10_v2_text_match()`, `attach_stage10_v2_relative_cross()`, and `attach_stage10_v2_group_gap()` join the additional Stage09/Stage10 feature-chain parquets into the candidate table.",
        190: "`normalize_model_backend()` chooses the backend, including `sklearn_lr`, `xgboost_cls`, `xgboost_ranker`, `wide_deep`, and `wide_deep_v2`; our AZ reruns used the `xgboost_cls` path without changing the model family.",
        191: "The training code then tunes `blend_alpha` with `tune_blend_alpha()`, combining the original pre-rank prior with the learned model score. During inference, `10_2_rank_infer_eval.py` rebuilds the same feature table, applies the saved model, creates `learned_blend_score`, ranks candidates per user, and writes the final metrics rows for ALS, PreScore, and LearnedBlend.",
        193: "Stage11 is implemented by a separate reward-model chain: `11_1_qlora_build_dataset.py` builds local comparison examples, `11_2_rm_train.py` trains the reward model, and `11_3_qlora_sidecar_eval.py` applies the trained adapter as a bounded rescue layer.",
        194: "The code is not asking an LLM to rerank every restaurant. Instead, it constructs local comparison or listwise prompts around candidates that Stage10 already placed in a plausible but imperfect rank window.",
        195: "`11_1_qlora_build_dataset.py` gathers candidate evidence, user evidence, business text, structured features, and local rival information into training rows.",
        196: "`11_2_rm_train.py` supports pairwise and listwise reward objectives; functions such as `build_listwise_rm_examples()` group a chosen candidate with nearby negative rivals and compute sample weights.",
        197: "`PairwiseRewardTrainer` and `ListwiseRewardTrainer` convert model logits into scalar reward scores and optimize whether the chosen candidate should beat rejected local rivals.",
        198: "`11_3_qlora_sidecar_eval.py` builds inference prompts, scores bounded candidate windows with the adapter, and fuses reward scores back with the Stage10 baseline.",
        199: "The rescue logic is segmented by rank bands such as `11-30`, `31-60`, and `61-100`, because a near-boundary candidate and a deep-rank candidate need different promotion strength.",
        200: "This design keeps Stage11 controlled: it acts as a sidecar correction layer rather than replacing the structured Stage10 ranking.",
        201: "In code terms, the pipeline is therefore layered: Stage09 writes the candidate parquet, Stage10 writes the learned ranking artifacts and metrics, and Stage11 reads those outputs to run targeted reward-model rescue on difficult local comparisons."
    }

    for idx, text in replacements.items():
        set_paragraph_text(paragraphs[idx], text)


def process_docx(src: Path, dst: Path) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with zipfile.ZipFile(src, "r") as zin:
            zin.extractall(tmpdir_path)

        document_xml = tmpdir_path / "word" / "document.xml"
        tree = ET.parse(document_xml)
        root = tree.getroot()
        body = root.find("w:body", NS)
        if body is None:
            raise RuntimeError("word/document.xml missing w:body")

        paragraphs = body.findall(".//w:p", NS)
        if len(paragraphs) <= 202:
            raise RuntimeError(f"Unexpected paragraph count: {len(paragraphs)}")
        if paragraph_text(paragraphs[177]) != "7. Model Explanation":
            raise RuntimeError("Section 7 anchor paragraph did not match expected text")
        if paragraph_text(paragraphs[202]) != "8. Implementation Experience and Development Journal":
            raise RuntimeError("Section 8 anchor paragraph did not match expected text")

        rewrite_section(paragraphs)
        tree.write(document_xml, encoding="UTF-8", xml_declaration=True)

        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for path in tmpdir_path.rglob("*"):
                if path.is_file():
                    zout.write(path, path.relative_to(tmpdir_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("--dst", type=Path)
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve() if args.dst else src.with_name(f"{src.stem}_section7_code_expanded{src.suffix}")
    process_docx(src, dst)
    print(dst)


if __name__ == "__main__":
    main()
