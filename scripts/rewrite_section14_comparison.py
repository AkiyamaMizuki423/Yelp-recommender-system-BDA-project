from __future__ import annotations

import argparse
import copy
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
    texts = []
    for node in paragraph.findall(".//w:t", NS):
        texts.append(node.text or "")
    return "".join(texts)


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
        351: "This section compares two different result sources: the local AZ reruns produced in this project workspace and the frozen LA reference metrics stored under current_release.",
        352: "Therefore, the purpose of the board is to show the result gap between AZ local reruns and LA reference outputs, rather than to claim that the two sides share the same published numbers.",
        353: "All numbers below should be read as cross-run comparisons, because the evaluation cohorts and candidate volumes are not fully identical between the local AZ reruns and the frozen LA reference.",
        354: "",
        355: "",
        356: "",
        357: "14.2 AZ Local Rerun vs LA Frozen Metrics",
        358: "Table 14-1 reports the directly observable gap between the local AZ pipeline outputs and the LA metrics snapshot already published in current_release.",
        359: "Table 14-1. AZ local rerun versus LA reference results",
        360: "Result item",
        361: "AZ local rerun",
        362: "LA frozen reference",
        363: "Difference and reading",
        364: "Stage09 bucket5 truth_in_pretrim",
        365: "0.465430",
        366: "0.745110",
        367: "AZ is lower by 0.279680, so the local AZ candidate chain keeps far fewer true items inside the pretrim set.",
        368: "Stage09 bucket5 hard_miss",
        369: "0.399101",
        370: "0.118996",
        371: "AZ is higher by 0.280105, showing a substantially larger miss rate before Stage10 reranking.",
        372: "Stage10 bucket5 LearnedBlendXGBCls recall@10",
        373: "0.072505",
        374: "0.126098",
        375: "AZ is lower by 0.053593 on the final learned blend recall.",
        376: "Stage10 bucket5 LearnedBlendXGBCls ndcg@10",
        377: "0.035069",
        378: "0.058108",
        379: "AZ is lower by 0.023039 on the final ranking quality metric.",
        380: "",
        381: "Stage10 bucket5 evaluation users",
        382: "1062",
        383: "1935",
        384: "The AZ local rerun is evaluated on a smaller user cohort, so absolute metrics are not perfectly apples-to-apples.",
        385: "Stage10 bucket5 evaluation items",
        386: "1092",
        387: "1798",
        388: "The AZ rerun also covers fewer items than the LA frozen reference.",
        389: "Stage10 bucket5 candidate volume",
        390: "159300",
        391: "935160",
        392: "The candidate pool is much smaller in the AZ rerun, which changes both recall headroom and ranking difficulty.",
        393: "Stage11 bucket5 direct comparison",
        394: "No AZ local rerun available yet",
        395: "LA reference available only",
        396: "A final AZ-versus-LA Stage11 number should not be claimed until the AZ side is rerun under a matching evaluation setup.",
        397: "14.3 Interpretation",
        398: "Using the currently available artifacts, the local AZ reruns underperform the frozen LA reference on both Stage09 recall retention and Stage10 final ranking metrics for bucket5.",
        399: "The largest visible gap already appears in Stage09, where the AZ candidate chain loses more true items before reranking begins.",
        400: "",
        401: "At the same time, the comparison is not a pure state-only effect, because the AZ reruns and the LA current_release metrics use different cohorts, item universes, and candidate counts."
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
        if len(paragraphs) <= 401:
            raise RuntimeError(f"Unexpected paragraph count: {len(paragraphs)}")

        if paragraph_text(paragraphs[350]) != "14. AZ and LA Comparison Board":
            raise RuntimeError("Section 14 anchor paragraph did not match expected text")

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
    dst = args.dst.resolve() if args.dst else src.with_name(f"{src.stem}_AZLA_comparison_updated{src.suffix}")
    process_docx(src, dst)
    print(dst)


if __name__ == "__main__":
    main()
