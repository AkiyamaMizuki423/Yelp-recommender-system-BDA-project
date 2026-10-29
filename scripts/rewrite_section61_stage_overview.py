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
        93: "Although the reproducible release path begins at Stage09, Stage01 to Stage08 still matter because they explain how the recommendation pipeline was built up step by step and why later design choices were made.",
        94: "Stage01 focuses on data preparation. It converts raw Yelp-style business, review, and user records into analysis-ready tables so that later stages can operate on consistent identifiers, usable schemas, and filtered state-specific subsets.",
        95: "Stage02 is exploratory analysis. It profiles user activity, review volume, category structure, and basic distribution patterns, helping the project identify what kinds of sparsity, popularity skew, and behavioral segments the recommender must handle.",
        96: "Stage03 and Stage04 are early collaborative-filtering experiments and validation steps. They test how far ALS-style recommendation can go on its own, establish early offline baselines, and reveal both the usefulness and the limitations of pure interaction-based ranking.",
        97: "Stage05 and Stage06 deepen the descriptive layer by adding more business understanding and text-oriented evidence. These steps help connect raw reviews and item attributes to interpretable preference signals instead of relying only on co-occurrence patterns.",
        98: "Stage07 introduces embedding and clustering workflows. Businesses and user behaviors are grouped in a semantic space, which makes it easier to capture similarity, latent intent, and neighborhood structure that do not appear clearly in standard matrix-factorization outputs.",
        99: "Stage08 continues with relabeling, refinement, and consolidation. Its role is to clean noisy structure discovered in earlier experiments and turn exploratory findings into more stable representations that can support downstream retrieval and ranking features.",
        100: "Taken together, these stages form the historical foundation of the system: they move the project from raw data ingestion, to behavioral diagnosis, to baseline recommenders, and then to richer semantic and structural representations.",
        101: "In the LA-oriented repository snapshot, these early files also preserve the original local framing of the project through explicit LA-targeted filtering and validation logic, which makes them useful for understanding how the pipeline was first assembled before the later release cleanup.",
        102: "The main lesson from Stage01 to Stage08 is that collaborative filtering alone was not enough. The later success of Stage09 to Stage11 depends on combining interaction signals with semantic evidence, cluster structure, route-aware recall logic, and more informative user-item features."
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
        if len(paragraphs) <= 102:
            raise RuntimeError(f"Unexpected paragraph count: {len(paragraphs)}")
        if paragraph_text(paragraphs[92]) != "6.1 Stage01 to Stage08: Historical Foundation":
            raise RuntimeError("Section 6.1 anchor paragraph did not match expected text")

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
    dst = args.dst.resolve() if args.dst else src.with_name(f"{src.stem}_section61_updated{src.suffix}")
    process_docx(src, dst)
    print(dst)


if __name__ == "__main__":
    main()
