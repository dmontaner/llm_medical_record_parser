# tests.py
# 2023-12-15 david.montaner@gmail.com
# to be converted to pytest

from colonoscopy_reporter import ColonoscopyRecord

for i in [1, 2, 3]:
    file = f"data/medical-record-{i}.pdf"
    print('=' * 100, flush=True)
    print(file)

    record = ColonoscopyRecord(pdf_filename=file)
    record.main_pipeline()
    print(record.format_final_decision())
    record.print_report_to_json()
