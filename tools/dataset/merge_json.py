import json

# JSONファイルのリスト
json_files = [
    '/gs/bs/tge-gc24sp03/datasets/arxiv_tikz/arxiv_tikz_org_train.json',
    '/gs/bs/tge-gc24sp03/datasets/arxiv_tikz/arxiv_tikz_revise_train.json',
    '/gs/bs/tge-gc24sp03/datasets/datikz-v2/datikzv2_train_filter_train.json',
    "/gs/bs/tge-gc24sp03/datasets/sketikz/sketikz_train.json"
]

# 全てのJSONデータを格納するリスト
merged_data = []

# 各ファイルを読み込んでデータをリストに追加
for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            merged_data.append(d)

# 出力ファイルの名前
output_file = '/gs/bs/tge-gc24sp03/datasets/tikz/merge_train.json'

# save jsonl
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"JSON files have been merged into {output_file}")
