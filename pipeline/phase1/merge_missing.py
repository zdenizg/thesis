import csv
import sys

MAIN_CSV = '/Users/denizguvenol/Desktop/thesis/cleaning/phase 1 /JFK Pages Rows.csv'
MISSING_CSV = '/Users/denizguvenol/Desktop/thesis/cleaning/phase 1 /jfk_categorization_55missing.csv'
OUTPUT_CSV = '/Users/denizguvenol/Desktop/thesis/cleaning/phase 1 /JFK_Pages_Merged.csv'

print("Loading missing CSV into memory...", flush=True)

# Build lookup: (file_id, page_number) -> row dict
missing_lookup = {}
with open(MISSING_CSV, 'r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    missing_cols = reader.fieldnames
    for row in reader:
        key = (row['file_id'].strip(), row['page_number'].strip())
        missing_lookup[key] = row

print(f"Loaded {len(missing_lookup)} rows from missing CSV.", flush=True)

# Columns to fill from missing if empty in main (excluding 'id' which missing doesn't have)
fill_cols = [
    'document_type', 'ocr_difficulty', 'includes_handwriting', 'has_shadowy_background',
    'document_quality', 'text_density', 'has_stamps', 'has_redactions', 'has_forms',
    'has_tables', 'is_typewritten', 'paper_condition', 'primary_characteristics', 'content'
]

print("Processing main CSV...", flush=True)
filled = 0
total = 0
not_found = 0
written_keys = set()
max_id = 0  # track highest id seen, to assign fresh ids to inserted rows

with open(MAIN_CSV, 'r', newline='', encoding='utf-8') as fin, \
     open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as fout:

    reader = csv.DictReader(fin)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        total += 1
        if total % 500000 == 0:
            print(f"  Processed {total} rows, filled {filled}...", flush=True)

        # Track max id so inserted rows can receive a unique id
        try:
            row_id = int(float(row.get('id', 0) or 0))
            if row_id > max_id:
                max_id = row_id
        except (ValueError, TypeError):
            pass

        # Check if content is empty
        if not row.get('content', '').strip():
            key = (row['file_id'].strip(), row['page_number'].strip())
            if key in missing_lookup:
                src = missing_lookup[key]
                for col in fill_cols:
                    if col in src and not row.get(col, '').strip():
                        row[col] = src[col]
                    elif col == 'content' and not row.get(col, '').strip():
                        row[col] = src.get(col, '')
                # Always fill content from missing if available
                if not row['content'].strip() and src.get('content', '').strip():
                    row['content'] = src['content']
                filled += 1
            else:
                not_found += 1

        writer.writerow(row)
        written_keys.add((row['file_id'].strip(), row['page_number'].strip()))

    # Insert rows from missing CSV that were not present in main CSV at all
    inserted = 0
    for key, src in missing_lookup.items():
        if key not in written_keys:
            max_id += 1
            new_row = {col: '' for col in fieldnames}
            new_row['id'] = max_id
            new_row['file_id'] = src['file_id']
            new_row['page_number'] = src['page_number']
            new_row['number_of_pages'] = src.get('number_of_pages', '')
            new_row['filename'] = src.get('filename', '')
            for col in fill_cols:
                if col in src:
                    new_row[col] = src[col]
            writer.writerow(new_row)
            inserted += 1
            print(f"  Inserted missing row: file_id={key[0]}, page_number={key[1]}, assigned id={max_id}", flush=True)

print(f"\nDone!")
print(f"  Total rows processed: {total}")
print(f"  Rows filled from missing CSV: {filled}")
print(f"  Empty rows not found in missing CSV: {not_found}")
print(f"  New rows inserted from missing CSV: {inserted}")
print(f"  Output written to: {OUTPUT_CSV}")
