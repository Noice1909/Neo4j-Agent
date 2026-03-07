# Architecture Diagrams

## Files

### Entity Resolution Flow
- **Source**: `entity_resolution_flow.mmd` (Mermaid diagram)
- **Image**: `entity_resolution_flow.png` (304 KB)

**Shows**: Complete 5-layer entity resolution pipeline with Layer 2.5 entity hints system.

### System Design
- **Source**: `system_design.mmd` (Mermaid diagram)
- **Image**: `system_design.png` (264 KB)

**Shows**: Complete system architecture from user request to response, including all 10 agents, data layer, and LLM integration.

## Viewing

- **PNG files**: Use any image viewer or insert into presentations
- **Mermaid files**: View on GitHub (auto-renders) or edit and regenerate PNG

## Regenerating Images

If you update the `.mmd` files:

```bash
cd docs/images

# Regenerate entity resolution flow
mmdc -i entity_resolution_flow.mmd -o entity_resolution_flow.png -w 2400 -H 2000 -b transparent

# Regenerate system design
mmdc -i system_design.mmd -o system_design.png -w 2400 -H 2000 -b transparent
```

## Editing

Edit `.mmd` files in any text editor or use [Mermaid Live Editor](https://mermaid.live).
