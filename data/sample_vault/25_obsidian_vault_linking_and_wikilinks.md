---
title: "Obsidian Vault Linking Conventions and WikiLink Normalization"
tags: [obsidian, wikilinks, markdown, second-brain]
created: 2026-11-06T15:30:00Z
source_type: markdown
---

# Obsidian Vault Linking Conventions

## WikiLink Syntax Variations

### Standard Formats
Personal notes created in Obsidian or Logseq frequently employ bidirectional WikiLinks:
1. Basic link: `[[Note Title]]`
2. Piped display alias: `[[Note Title|Custom Label]]`
3. Anchor link with heading: `[[Note Title#Subheading]]`

### Heading Target References
Links often reference specific section anchors within complex notes.

## Normalization Pipeline

### Slugification & Parsing
1. Regex matches `\[\[([^\]]+)\]\]`.
2. Normalizes target to standard slug format (`adr-001-langgraph-state-machine`).
3. Preserves anchor fragments for citation resolution.
See [[knowledge-graphs-vs-vector-retrieval]] and [[adr-002-two-stage-chunking-chromadb]].
