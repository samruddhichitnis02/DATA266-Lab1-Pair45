"""Create the fixed 30-sample two-rater human-audit worksheet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from train_cyclegan import image_paths


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    photos = image_paths(data_root / "photo_jpg")[: args.count]
    generated = image_paths(ROOT / "outputs" / "pred_B2A")[: len(photos)]
    if len(photos) != len(generated):
        raise FileNotFoundError("Generate pred_B2A images before creating the audit sheet")
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    thumb_size = 180
    label_height = 28
    sheet = Image.new("RGB", (thumb_size * 2, (thumb_size + label_height) * len(photos)), "white")
    draw = ImageDraw.Draw(sheet)
    rows = []
    for index, (photo, translation) in enumerate(zip(photos, generated)):
        left = Image.open(photo).convert("RGB").resize((thumb_size, thumb_size))
        right = Image.open(translation).convert("RGB").resize((thumb_size, thumb_size))
        y = index * (thumb_size + label_height)
        sheet.paste(left, (0, y)); sheet.paste(right, (thumb_size, y))
        draw.text((4, y + thumb_size + 5), f"{index:02d} input photo", fill="black")
        draw.text((thumb_size + 4, y + thumb_size + 5), f"{index:02d} generated Monet", fill="black")
        rows.append({
            "sample_id": index,
            "input_file": photo.name,
            "translated_file": translation.name,
            "rater_1_style": "", "rater_1_content": "", "rater_1_artifacts": "",
            "rater_2_style": "", "rater_2_content": "", "rater_2_artifacts": "",
            "notes": "",
        })
    sheet.save(output_dir / "human_audit_sheet.png")
    with (ROOT / "human_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    (ROOT / "human_audit_instructions.md").write_text(
        "# Blinded human audit\n\n"
        "Two raters should independently score the same 30 fixed input/translation pairs. "
        "Use a 1–5 scale for style transfer quality, content preservation, and artifacts "
        "(5 is best for style/content; 5 means no visible artifacts). Do not let either rater "
        "see the other rater's scores before both worksheets are complete. Fill the two rater "
        "column groups in `human_audit.csv`, then rerun `evaluate_task3.py` to calculate "
        "Cohen's kappa and percent agreement.\n",
        encoding="utf-8",
    )
    print(f"Created {ROOT / 'human_audit.csv'} and {output_dir / 'human_audit_sheet.png'}")


if __name__ == "__main__":
    main()
