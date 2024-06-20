import json

# JSONファイルのリスト
json_files = [
    '/gs/bs/tge-gc24sp03/datasets/arxiv_tikz/arxiv_tikz_org_train.json',
    '/gs/bs/tge-gc24sp03/datasets/arxiv_tikz/arxiv_tikz_revise_train.json',
    '/gs/bs/tge-gc24sp03/datasets/arxiv_tikz/arxiv_tikz_aug_train.json',
    '/gs/bs/tge-gc24sp03/datasets/datikz-v2/datikzv2_train_filter_train.json',
    "/gs/bs/tge-gc24sp03/datasets/arxiv_tikz/arxiv_tikz_img2description_train.json",
    "/gs/bs/tge-gc24sp03/datasets/im2latex_handwritten/im2latex_handwritten_train.json",
    "/gs/bs/tge-gc24sp03/datasets/latex_formulas/latex_formulas_train.json",
]

# 全てのJSONデータを格納するリスト
merged_data = []

# 各ファイルを読み込んでデータをリストに追加
for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"size of {file}: {len(data)}")
        for d in data:
            merged_data.append(d)

# 出力ファイルの名前
output_file = '/gs/bs/tge-gc24sp03/datasets/tikz/step2-1-2-3-5-7-10-11-merge_train.json'
print(f"size of merged_data: {len(merged_data)}")

# save jsonl
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"JSON files have been merged into {output_file}")
