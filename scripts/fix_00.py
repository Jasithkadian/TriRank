import json
with open('00_setup.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if 'def load_msmarco_dev' in line:
                # We found the data_loader cell. Overwrite it.
                new_source = [
                    "%%writefile /workspace/gemcol_evaluation/utils/data_loader.py\n",
                    "import json\n",
                    "from .config import MSMARCO, PATHS\n",
                    "from tqdm import tqdm\n",
                    "from datasets import load_dataset\n",
                    "import urllib.request\n",
                    "\n",
                    "def load_msmarco_dev(data_dir=None):\n",
                    "    print(\"Loading queries via HF Hub (Tevatron/msmarco-passage)...\")\n",
                    "    dev_ds = load_dataset(\"Tevatron/msmarco-passage\", split=\"validation\")\n",
                    "    queries = {str(row[\"query_id\"]): str(row[\"query\"]) for row in dev_ds}\n",
                    "    \n",
                    "    print(\"Downloading qrels from BeIR (to fix missing Azure blob)...\")\n",
                    "    req = urllib.request.urlopen(\"https://huggingface.co/datasets/BeIR/msmarco-qrels/resolve/main/dev.tsv\")\n",
                    "    qrels = {}\n",
                    "    for line in req.read().decode(\"utf-8\").splitlines()[1:]:\n",
                    "        qid, docid, score = line.split(\"\\t\")\n",
                    "        if qid not in qrels: qrels[qid] = {}\n",
                    "        qrels[qid][docid] = int(score)\n",
                    "            \n",
                    "    print(\"Loading 8.8M corpus via HF Hub (Tevatron/msmarco-passage-corpus)...\")\n",
                    "    corpus_ds = load_dataset(\"Tevatron/msmarco-passage-corpus\", split=\"train\")\n",
                    "    \n",
                    "    corpus = {}\n",
                    "    for row in tqdm(corpus_ds, desc=\"Building corpus dictionary\"):\n",
                    "        corpus[str(row[\"docid\"])] = str(row[\"text\"])\n",
                    "        \n",
                    "    return queries, qrels, corpus\n",
                    "\n",
                    "def load_beir_dataset(name, beir_dir):\n",
                    "    pass\n",
                    "\n",
                    "def beir_corpus_to_texts(corpus):\n",
                    "    return {docid: doc.get(\"title\", \"\") + \" \" + doc.get(\"text\", \"\") for docid, doc in corpus.items()}\n",
                    "\n",
                    "def save_run_json(run, path):\n",
                    "    with open(path, 'w') as f: json.dump(run, f)\n",
                    "\n",
                    "def load_run_json(path):\n",
                    "    with open(path, 'r') as f: return json.load(f)\n",
                    "\n",
                    "def save_run(run, path): save_run_json(run, path)\n",
                    "def load_run(path): return load_run_json(path)\n"
                ]
                cell['source'] = new_source
                break

with open('00_setup.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Fixed 00_setup.ipynb with BeIR qrels')
