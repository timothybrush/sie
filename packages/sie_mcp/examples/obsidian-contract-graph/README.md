# Example: a private contract knowledge graph (SIE MCP edge + Obsidian)

Turn a folder of messy legal documents into a private, linked Obsidian knowledge graph.
Parsing, entity extraction, and PII redaction all run on open models on your own cluster
through the SIE MCP edge, so the raw documents and the personal data never reach the model
API. The agent only ever sees small markdown.

An Obsidian vault is just a folder of markdown. Claude Code's file tools write into it; the SIE
MCP edge does the document processing. Nothing in this example modifies the `sie_mcp` package.

## What is in here

| Path | What it is |
|---|---|
| `inbox/` | Sample legal documents with synthetic PII: a master services agreement, an NDA with a contractor SSN, an invoice, and more. |
| `vault/` | An Obsidian vault that already holds a standing web of agreements across ten organizations, so the graph is large before you start. Contains `Home.md`, `NOTE-CONVENTIONS.md`, `Contracts/`, and `Entities/`. |
| `processed-fallback/` | A finished example note, so you can see the output shape without running anything. |
| `token-savings.html` | An interactive visual of the token reduction. Open it in a browser. |

## Prerequisites

- [Obsidian](https://obsidian.md) (free).
- Claude Code with the SIE MCP edge connected and the document skills installed. See the
  package README (`packages/sie_mcp/README.md`) and the `INSTALL.md` from a generated plugin
  pack for the exact `claude mcp add` command and the `cp -R claude-code/* ~/.claude/skills/`
  step.
- Launch Claude Code in this folder so it can read `inbox/` and write `vault/`.

## Setup

1. **Open the vault in Obsidian.** Open folder as vault, point it at this example's `vault/`,
   and open Graph View in a side pane. It is already a dense graph (about 33 cross-linked
   notes), so it looks substantial before you do anything.
2. **Connect the edge** with the `claude mcp add` command from your `INSTALL.md`, copy the
   skills into `~/.claude/skills/`, and restart Claude Code.

## Step 1: parse and link two documents

Ask the agent:

> Read `vault/NOTE-CONVENTIONS.md`. Then for `inbox/acme-globex-msa.html` and
> `inbox/globex-invoice.html`: use the **parse-document** skill to convert each to markdown
> without reading the raw file into your context, then use **extract-entities** to pull the
> parties, people, dates, and amounts. Create a note under `Contracts/` for each following the
> conventions, with every party and person as a `[[wikilink]]`, and create `Entities/` notes
> for the organizations and people. Do not open the source HTML directly.

What to notice:

- The parse-document receipt shows the token reduction (around 85 percent versus uploading the
  document).
- The raw HTML never entered the agent's context; it worked from the markdown the edge returned.
- Two contract notes plus entity notes appear in the vault.

## Step 2: redact PII before the model sees it

> `inbox/stardust-nda.html` contains a contractor's SSN, date of birth, and home address. Run
> the **redact-pii** skill on it first, so the personal data is replaced with placeholders on
> the cluster. Then build the `Contracts/` note from the redacted text, set `redacted: true`,
> and record the PII counts.

Open the new note. The agent's context only ever contained `[SOCIAL_SECURITY_NUMBER_1]`, never
the real number. The personal data stayed on the cluster.

## Step 3: synthesize from the small notes

> Update `Contracts.md` as a Map of Content linking all the notes grouped by status, then give
> me a one-paragraph briefing on the Acme-Globex relationship.

The agent writes the index and briefing from the small markdown notes, not the originals.

To keep the summary step on your own models too:

> Use the **summarize-document** skill on the redacted NDA note and add the summary to it.

This runs generation on an open model on your cluster.

## Step 4: explore the graph

Open Graph View. The documents are now one connected graph: `Acme Corp` at the center, linked
to `Globex Inc`, `Jane Doe`, `Stardust Analytics LLC`, and the redacted contractor, joined to
the standing vault of agreements that was already there. Open `Entities/Acme Corp.md` to see
its backlinks; open the NDA note to see `redacted: true` and the placeholders.

## How it works

- `docs_to_markdown` converts the document on the cluster (docling, with VL-OCR for scans).
- `extract_entities` and `redact_pii` run GLiNER on the cluster.
- The agent reasons over the small markdown artifacts: cheaper tokens, more context-window
  headroom, and raw bytes and PII that never leave your cloud.

## Extend it

- Drop your own PDF into `inbox/` and point the agent at it.
- Add a new entity label to the extract step.
- Wire an Obsidian Dataview query over the note frontmatter.
- Run the summary step on a different model through `SIE_MCP_GENERATE_MODEL`.

## Notes

- The vault is fully synthetic: made-up companies, people, and PII.
- The first model-backed request can take longer while the cluster loads weights; later calls
  are fast. `processed-fallback/` shows the finished output shape if you want to read ahead.
