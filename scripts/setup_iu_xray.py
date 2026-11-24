"""Fetches the Indiana University Chest X-ray Collection (IU X-ray / Open-I)
and builds the manifest CSV consumed by src/data/iu_xray.py.

This script does NOT run automatically as part of any pipeline — it downloads
~2GB of data from the NLM Open-i public archive. Run it yourself:

    python scripts/setup_iu_xray.py --data-dir data/iu_xray

What it does:
  1. Downloads and extracts the PNG image archive and the XML reports archive
     from the NLM Open-i IU X-ray release into <data-dir>/images and
     <data-dir>/reports.
  2. Parses each report XML (MeSH major/minor tags + Findings/Impression text)
     and pairs it with its associated frontal/lateral image(s).
  3. Writes <data-dir>/manifest.csv with columns:
     image_path, patient_id, report_text, mesh_tags (semicolon-separated).

The NLM Open-i download URLs occasionally move; if the URLs below 404, get the
current links from https://openi.nlm.nih.gov/faq (Chest X-ray collection) and
pass them via --images-url / --reports-url.
"""

import argparse
import csv
import os
import tarfile
import urllib.request
import xml.etree.ElementTree as ET

DEFAULT_IMAGES_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
DEFAULT_REPORTS_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"already downloaded: {dest}")
        return
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def extract(archive_path: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    print(f"extracting {archive_path} -> {dest_dir}")
    with tarfile.open(archive_path) as tar:
        tar.extractall(dest_dir)


def parse_report(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    findings, impression = "", ""
    for elem in root.iter("AbstractText"):
        label = (elem.get("Label") or "").upper()
        text = elem.text or ""
        if label == "FINDINGS":
            findings = text
        elif label == "IMPRESSION":
            impression = text
    report_text = " ".join(t for t in (findings, impression) if t).strip()

    mesh_tags = [m.text for m in root.iter("major") if m.text] + \
                [m.text for m in root.iter("minor") if m.text]

    image_ids = [img.get("id") for img in root.iter("parentImage") if img.get("id")]
    patient_id = os.path.splitext(os.path.basename(xml_path))[0]
    return patient_id, report_text, mesh_tags, image_ids


def build_manifest(reports_dir: str, images_dir: str, out_csv: str) -> None:
    rows = []
    xml_files = [f for f in os.listdir(reports_dir) if f.endswith(".xml")]
    for fname in sorted(xml_files):
        patient_id, report_text, mesh_tags, image_ids = parse_report(
            os.path.join(reports_dir, fname))
        for image_id in image_ids:
            image_path = f"{image_id}.png"
            if not os.path.exists(os.path.join(images_dir, image_path)):
                continue
            rows.append({
                "image_path": image_path,
                "patient_id": patient_id,
                "report_text": report_text,
                "mesh_tags": ";".join(mesh_tags),
            })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "patient_id", "report_text", "mesh_tags"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/iu_xray")
    parser.add_argument("--images-url", default=DEFAULT_IMAGES_URL)
    parser.add_argument("--reports-url", default=DEFAULT_REPORTS_URL)
    parser.add_argument("--skip-download", action="store_true",
                         help="use if you already have images/ and reports/ populated")
    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, "images")
    reports_dir = os.path.join(args.data_dir, "reports")

    if not args.skip_download:
        images_archive = os.path.join(args.data_dir, "NLMCXR_png.tgz")
        reports_archive = os.path.join(args.data_dir, "NLMCXR_reports.tgz")
        os.makedirs(args.data_dir, exist_ok=True)
        download(args.images_url, images_archive)
        download(args.reports_url, reports_archive)
        extract(images_archive, images_dir)
        extract(reports_archive, reports_dir)

    build_manifest(reports_dir, images_dir, os.path.join(args.data_dir, "manifest.csv"))


if __name__ == "__main__":
    main()
