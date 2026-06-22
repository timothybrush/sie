# Install Superlinked document offload

`superlinked.md` is a router to each surface's **native** install path. There is no
drop-a-file auto-install — pick your surface below.

## Cowork (lead surface)

Install the **Superlinked plugin**, which provides this skill together with the remote MCP
connector. Configure the connector with your **connector secret** in the connector settings
(not in the chat).

Replace `<host>` below with your Superlinked deployment host (the address the MCP edge is
served at — e.g. the value of `SIE_MCP_HOST` for your cluster):

- MCP endpoint: `https://<host>/mcp`
- Auth header: `Authorization: Bearer <connector-secret>`
- Health check: `GET https://<host>/healthz`

## claude.ai (#1312)

claude.ai is a **two-piece** install: a custom **connector** (the remote MCP) plus a
**skill ZIP** (the agent skill). Unlike Cowork, claude.ai connectors are OAuth-only — you
cannot paste a connector secret into a header — so the connector secret is entered during
an OAuth sign-in instead. Requires a plan that allows custom connectors and skills
(Pro / Max / Team / Enterprise) with code execution enabled.

Replace `<host>` with your Superlinked deployment host (the externally reachable origin of
the MCP edge — set `SIE_MCP_PUBLIC_URL` to pin it for OAuth metadata).

### 1. Connector (remote MCP + OAuth)

1. In claude.ai go to **Customize → Connectors → `+` → Add custom connector**.
2. Enter the remote MCP server URL: `https://<host>/mcp`. Click **Add**.
3. Click **Connect** and complete the OAuth sign-in. On the Superlinked authorize page,
   enter your **connector secret** (provided by your cluster operator). The edge maps the
   secret to your user identity — credentials are never pasted into the chat.

The OAuth metadata, registration, authorize, and token endpoints are served by the same
edge (`/.well-known/oauth-protected-resource`, `/.well-known/oauth-authorization-server`,
`/register`, `/authorize`, `/token`); the connector-secret check at `/mcp` is unchanged.

### 2. Skill ZIP

1. Build the ZIP: `mise run mcp-skill-zip` (writes `dist/superlinked-docs-skill.zip`).
2. In claude.ai go to **Customize → Skills → `+` → Create skill → Upload a skill** and
   upload the ZIP. The archive contains a single `superlinked-docs/SKILL.md` folder, the
   format claude.ai expects.

### File-landing

Generated markdown is written **in the sandbox**; export it by **Download** or **save to
Google Drive**, and optionally add the downloaded file to a **Project**'s knowledge to
reuse it across chats. The skill drives this flow (there is no persistent connected folder
as on Cowork).

### Operator notes

- Pin `SIE_MCP_PUBLIC_URL` to the externally reachable origin so the OAuth metadata URLs
  are stable (otherwise they are derived per-request from forwarded host/proto headers).
- Run the edge as a **single worker** (the default `mise run mcp-serve`): the OAuth
  authorization-code store is in-process, so a code issued on one worker cannot be
  redeemed on another. A shared store lands with per-user key issuance in Req 10 (#1313).
- The OAuth redirect allowlist defaults to claude.ai's callback; override with
  `SIE_MCP_OAUTH_REDIRECT_URIS` (comma-separated). Set `SIE_MCP_OAUTH_ENABLED=0` to
  disable the bridge for Cowork-only deployments.
