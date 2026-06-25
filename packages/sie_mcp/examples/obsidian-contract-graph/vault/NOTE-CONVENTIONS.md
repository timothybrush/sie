# Vault note conventions (read this before writing notes)

When turning an offloaded document into a vault note, follow this structure exactly so the graph and properties stay consistent.

## Where notes go
- Contract/document notes: `Contracts/<Short Title>.md`
- Entity notes (companies, people): `Entities/<Name>.md`
- Never write the raw source document into the vault. Only the SIE markdown artifact and notes derived from it.

## Document note format
Frontmatter (YAML), then a short body. Every party, person, and organization is an Obsidian wikilink `[[Name]]` so the graph connects.

```markdown
---
title: Acme-Globex MSA
source: acme-globex-msa.html
type: contract            # contract | nda | invoice
parties: ["[[Acme Corp]]", "[[Globex Inc]]"]
people: ["[[Jane Doe]]", "[[Robert Lang]]"]
effective_date: 2026-03-03
amount: $2,500,000
status: active
redacted: false           # true if PII was redacted before the model saw it
pii_redacted: {}          # e.g. {person: 1, email: 1, ssn: 1} when redacted
tags: [contract, msa, renewal]
---

## Summary
Two or three sentences. Mention [[parties]] and [[people]] as links.

## Key terms
- Term: ...
- Renewal: ...
- Amount: [[amount]] ...

## Linked entities
[[Acme Corp]] | [[Globex Inc]] | [[Jane Doe]]
```

## Entity note format
```markdown
---
type: organization        # organization | person
role: Client              # for people: title; for orgs: their role
tags: [entity]
---

# Acme Corp
Brief one-line description. Backlinks below show every document this entity appears in.
```

## Redaction rule (privacy beat)
- For any document that contains personal data (SSN, home address, date of birth, personal email/phone), run SIE `redact_pii` FIRST, and build the note from the redacted text.
- In the note, set `redacted: true` and fill `pii_redacted` with the counts SIE returns.
- The redacted note keeps placeholders like `[PERSON_1]`, `[EMAIL_1]`, `[SOCIAL_SECURITY_NUMBER_1]`. The original values stay on the SIE cluster and never enter the vault or the model context.

## Index
Maintain `Contracts.md` as a Map of Content: a bullet list linking every contract note, grouped by status.
