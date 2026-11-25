Synthetic demo documents for the search_docs tool. No real company data.

Each *.md file is one document. Its front matter lists the groups allowed to
read it; ingestion rejects a document without groups. Load them with:

    make ingest        # or: uv run python -m chatbot.rag.ingest db/seed/documents

Groups used here:

- employees   everyone at the company
- engineering delivery and platform teams
- finance     pricing and budgets
- hr          people team
- leadership  managers and directors

This file is .txt on purpose so ingestion skips it.
