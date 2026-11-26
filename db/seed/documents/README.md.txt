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

Adversarial fixtures (C21) contain prompt injections on purpose, for the
injection tests. Their front matter starts with a comment saying so:

- facilities/office-faq.md            formatting hijack (reply in French, sign off)
- it/vpn-setup.md                     link exfiltration of the user's question, a
                                      fake closing tag and an image beacon
- handbook/travel-policy-addendum.md  fake system message, prompt and data
                                      extraction, and a hidden instruction in
                                      invisible Unicode tag characters
