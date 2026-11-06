# kb/work — work experience entries (private)

Freelance and client work lives here, one YAML file per entry (e.g. `design_stencil.yaml`, kit §5A).
**These files are git-ignored**: they hold details of private client codebases, so they stay on the
owner's machine. The facts themselves are kept in `MASTER_RESUME_KIT.md` (private).

A work entry is a `Project` with `kind: work`. It is written, checked and repaired like a project, but never
scored as one; it renders first under Experience, with no GitHub link. Minimal shape:

```yaml
id: acme_freelance
kind: work
name: "Acme"
tier: 1
section: "5A"
github: ""                     # private: no link
org: "Acme"
location: "Remote"
dates: "October 2026 -- Present"
stack: [TypeScript, Node.js]
one_liner: "What the product is, in one line."
tags: [llm, backend]
rules: ["Title is 'Freelance AI Engineer'."]
banned_terms: ["intern", "lead"]
facts:
  - {id: acme_prs, claim: "12 merged PRs", values: ["12"], status: verified}
titles:
  - {subtitle: "Freelance AI Engineer", stack: []}
bullets:
  - {id: G1a, group: G1, text: "Shipped ... across 12 merged PRs.", fact_ids: [acme_prs]}
```

`group` matches the `work_groups` of a JD type in `kb/archetypes.yaml` (the kit §9 "G" column): an entry is
included when the JD type lists groups, and those groups lead. Without any file here the agent simply
uses no work entry (4 projects, Research Intern only), and the tests that need one are skipped.
