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
        119: "Stage10 is the global ranking backbone of the system. After Stage09 produces a bounded but still noisy candidate set, Stage10 learns a stronger ordering over those candidates using structured features and learned reranking models.",
        120: "Its input is not raw Yelp data, but the candidate pool from Stage09 enriched with route information, prior scores, user history, business attributes, semantic evidence, and relative-position signals.",
        121: "The Stage10 design combines multiple feature families:",
        122: "route and source features, which describe how each candidate entered the fused pool",
        123: "user-business match features, which measure behavioral fit between the user and the merchant",
        124: "text-match and semantic similarity features, which connect review language and intent signals to item content",
        125: "relative cross features and group-gap features, which compare each candidate against nearby alternatives",
        126: "rank priors and calibration features, which stabilize the final ordered list",
        127: "This stage matters because ranking is not only about scoring good restaurants highly. It is also about deciding which plausible candidate should win against other plausible candidates in the same shortlist.",
        128: "Why this stage is important:",
        129: "it creates a stable mainline ranking before any LLM-style intervention",
        130: "it reduces noise inherited from multi-route retrieval",
        131: "it provides a strong structured baseline that can be evaluated on its own",
        132: "it defines the bounded candidate window that Stage11 later uses for targeted rescue reranking",
        133: "Stage10 Result",
        134: "Current fixed evaluation results show consistent gains over a prescore baseline:",
        135: "bucket2: Recall@10 improved from `0.1098` to `0.1127`; NDCG@10 improved from `0.0513` to `0.0522`, showing a modest but stable uplift",
        136: "bucket5: Recall@10 improved from `0.0935` to `0.1261`; NDCG@10 improved from `0.0440` to `0.0581`, which is the strongest mainline result in the report",
        137: "bucket10: Recall@10 improved from `0.0569` to `0.0772`; NDCG@10 improved from `0.0265` to `0.0341`, showing that the reranker also helps in a harder segment",
        138: "Overall, Stage10 is the mainline decision layer: Stage09 decides what is worth considering, Stage10 decides the strongest structured order, and Stage11 later focuses only on difficult boundary cases."
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
        if len(paragraphs) <= 140:
            raise RuntimeError(f"Unexpected paragraph count: {len(paragraphs)}")
        if paragraph_text(paragraphs[118]) != "6.3 Stage10: Structured Reranking":
            raise RuntimeError("Section 6.3 anchor paragraph did not match expected text")

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
    dst = args.dst.resolve() if args.dst else src.with_name(f"{src.stem}_section63_updated{src.suffix}")
    process_docx(src, dst)
    print(dst)


if __name__ == "__main__":
    main()
