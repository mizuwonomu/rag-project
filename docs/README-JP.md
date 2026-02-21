# 🤖 HUST 履修規定チャットボット 
<p align="center">
  [<a href="../README.md">🇺🇸 English</a>]
</p>

## 概要
本プロジェクトは、HUST(ハノイ工科大学)の学生が自然言語で履修規定について質問をできるよう、RAG(検索拡張生成)チャットボットを提供することを目的としています。

仕組みは **親ドキュメントリトリーバー** とハイブリッドリトリーバー(密ベクトル + BM25)です。なお、ルーターがユーザーの入力を雑談か文書に基づく回答かに自動判別し、処理を振り分けます。"

## 前提条件
| 要件 | バージョン / 備考 |
|---|---|
| Python | `3.13以上` (具体的なバージョン: `3.13.5`)
| Groq API Key | 必須 - LLM(`qwen/qwen3-32b`)を駆動するために必要
| LangSmith API Key | 任意 - チェーンのトレース用
| GPU (CUDA) | 任意 - 自動検出(最小 2GB VRAM 使用) ; GPUがない場合はCPUで埋め込み計算を行います|
---

## 設定

### 1. リポジトリのクローン
```bash
git clone https://github.com/mizuwonomu/rag-project.git
cd rag-project
```


### 2. 環境変数の設定
プロジェクトルートに `.env`ファイルを作成してください:
```bash
GROQ_API_KEY=your_groq_api_key
LANGSMITH_API_KEY=your_langsmith_api_key   # 任意
```

### 3. 依存関係のインストール
`uv`の使用を推奨します:
```bash
uv sync
# または pip を使用する場合:
pip install -r requirements.txt
```

### 4.ベクトルストアを構築
* プロジェクトルートに `data_quyche/`フォルダを作成してください。
* `data_quyche/`に参照元のPDF [QCDT_2025_DHBK.pdf](https://ctt.hust.edu.vn/Upload/Nguy%E1%BB%85n%20Qu%E1%BB%91c%20%C4%90%E1%BA%A1t/files/DTDH_QDQC/Hoctap/QCDT_2025_5445_QD-DHBK.pdf) を置き、コードを実行:
```bash
python -m src.ingestion.ingest_regulations
```
このスクリプトを実行することで、以下の処理が行われます:
- *親チャンク* (一つずつ "Điều" 条)にPDFを分割
- *子チャンク* を作成し (`chunk_size=400`, `chunk_overlap=50`) 、ChromaDB (`chroma_db/` フォルダ)に埋め込む
- 親ドキュメント**pickled 化されたバイトデータ**として保存(自動的に生成されます)

### 5. アプリケーションの実行
```bash
streamlit run frontend/app.py
```
---

## 仕組み

![システム構成図](../assets/DIAGRAM.png)

```
ユーザーの質問
      │
      ▼
 ルーターチェーン ──► "chat（雑談）" ──► 通常の会話応答
                                            │
                                            ▼
                              会話履歴の保存 + ユーザーフィードバック
      │
      └──────► "RAG（検索拡張生成）"
                    │
                    ▼
         アンサンブルリトリーバー（子チャンクのみ対象）
         ├── 密ベクトル検索：ChromaDB（重み 0.5）
         └── 疎ベクトル検索：BM25 キーワード検索（重み 0.5）
                    │
                    ▼
         上位 k 件の子チャンク doc_id → doc_store_pdr/ から親ドキュメントを取得
                    │
                    ▼
         QA チェーン（出典引用必須）→ ストリーミング回答 + 参照元
                    │
                    ▼
         会話履歴へ保存 + 回答に対するユーザーフィードバック
```

---

## プロジェクト構成

```
rag-project/
├── frontend/
│   └── app.py                    # Streamlit フロントエンドのエントリーポイント
├── src/
│   ├── __init__.py
│   ├── qa_chain.py               # RAG・会話チェーンのコアロジック
│   ├── utils.py                  # 共通ユーティリティ（埋め込みモデルのローダー）
│   └── ingestion/
│       ├── __init__.py
│       └── ingest_regulations.py # PDF 取り込み・ベクトルストア構築
├── data_quyche/                  # 参照元 PDF 文書（バージョン管理対象外）
├── chroma_db/                    # ChromaDB ベクトルストア（自動生成・バージョン管理対象外）
├── doc_store_pdr/                # 親ドキュメントストア（自動生成・バージョン管理対象外）
├── assets/                       # 静的アセット
├── docs/                         # 付加的なドキュメント
├── legacy/                       # アーカイブ済みコード - 使用不可
├── feedback_log.csv              # ユーザーフィードバックログ（自動生成・バージョン管理対象外）
├── .env                          # API キー - コミット禁止
├── pyproject.toml                # プロジェクトメタデータ・依存ライブラリの固定バージョン
├── requirements.txt              # pip インストール用マニフェスト
└── uv.lock                       # uv ロックファイル - 手動編集禁止
```

---

## 主要設計方針

| コンポーネント | 選択 | 理由 |
|---|---|---|
| 埋め込みモデル | `BAAI/bge-m3`（HuggingFace） | 多言語（ベトナム語）での高い性能 |
| ベクトルストア | ChromaDB | 永続化可能・ローカル動作・サーバー不要 |
| 親ドキュメントストア | `LocalFileStore` + pickle | 各「Điều（条）」単位で全文コンテキストを保持 |
| 検索方式 | EnsembleRetriever（密ベクトル + BM25）| 意味検索とキーワード検索のバランスを確保 |
| LLM | `ChatGroq` - `qwen/qwen3-32b` | Groq API による高速推論 |
| 記憶管理 | インプロセス `ChatMessageHistory` | 軽量なシングルセッション対応 |

---

## 注意事項

- `chroma_db/` および `doc_store_pdr/` はバージョン管理対象外です。クローン後は `python -m src.ingestion.ingest_regulations` を実行して再生成してください。
- 埋め込みモデルを変更する場合は、必ず取り込み処理（インジェスト）を再実行してください。ベクトルストアと埋め込みモデルは一致している必要があります。
- `.env` ファイルは厳重に管理し、リポジトリへのコミットは絶対に行わないでください。
