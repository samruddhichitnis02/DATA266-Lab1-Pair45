# Blinded human audit

Two raters should independently score the same 30 fixed input/translation pairs. Use a 1–5 scale for style transfer quality, content preservation, and artifacts (5 is best for style/content; 5 means no visible artifacts). Do not let either rater see the other rater's scores before both worksheets are complete. Fill the two rater column groups in `human_audit.csv`, then rerun `evaluate_task3.py` to calculate Cohen's kappa and percent agreement.
