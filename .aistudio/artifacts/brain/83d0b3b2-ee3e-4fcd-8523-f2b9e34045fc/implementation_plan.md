# Multi-Attribute Resident Queries & Composite Spotlight

Simultaneous extraction and spotlighting of multiple resident attributes (e.g., care level, primary caregiver, room, admission date, status) in a single zero-token deterministic query lifecycle.

## User Review & Critical Decisions

> [!IMPORTANT]
> The user confirmed the following design and architectural choices for compound resident queries:
> - **Multi-Attribute Processing**: Extract all requested fields simultaneously and present a unified Multi-Attribute Spotlight Card combining both target values.
> - **Composite Headline Formatting**: Headline will synthesize both values into an immediate, direct answer: `"Assisted Living · Nurse David Kim, LPN — Care Level & Primary Caregiver"`.

- **Confirmed Decision 1**: Parse compound conjunctions (`and`, `&`, `as well as`, comma-separated attribute inquiries) to extract multiple target attributes into `targetAttributes: ResidentAttribute[]`.
- **Confirmed Decision 2**: When multiple attributes are detected (e.g. `care level and care taker`), render a dual-attribute spotlight card with paired attribute badges and clinical context.

---

## 1. Overview & Core Concept

- **What It Does**: Care staff often query multiple resident attributes in one sentence (e.g. *"Donald Draper belongs to which care level and care taker ?"* or *"Which room is Eleanor Vance in and who is her primary nurse?"*). Instead of picking only the first matched attribute and dropping the second, the system detects all requested attributes, retrieves them via MCP, and displays both prominently.
- **Target Audience**: Assisted living, memory care, and nursing clinical staff who need rapid, zero-token answers during shift handover and rounds.
- **Key Value**: Eliminates the need to ask two separate questions or sift through general record dumps to find the primary caregiver and care level.

---

## 2. User Experience & Visual Design

### Key User Flows
1. **User asks a compound query**: e.g., *"Donald Draper belongs to which care level and care taker ?"*.
2. **Immediate Zero-Token Recognition**: Pipeline detects `Donald Draper` (resident entity) and two target attributes: `careType` (care level) and `primaryCaregiver` (caretaker).
3. **Composite Headline Spotlight**:
   - Headline: `Assisted Living · Nurse David Kim, LPN`
   - Kicker/Subtitle: `Care Level & Primary Caregiver for Donald Draper`
4. **Dual Multi-Attribute Spotlight Card**:
   - Renders a 2-column high-contrast card:
     - Left column: **CARE LEVEL**: `Assisted Living` (with care category detail).
     - Right column: **PRIMARY CAREGIVER**: `Nurse David Kim, LPN` (with clinical role).
5. **Full Record Below**: Complete resident profile and room details remain accessible beneath the spotlight.

### Visual Styling & Theme
- **Color System (60-30-10)**:
  - 60% Canvas: Dark Slate (`#020617` / `bg-slate-950`).
  - 30% Structural Surfaces: Slate 900 (`#0f172a`), hairline borders (`border-slate-800`).
  - 10% Accent: Clinical Teal (`#14b8a6` / `text-teal-400`).
- **Zero-Pill Typography**:
  - Metadata separated by clean dots (`·`).
  - Bold, prominent values without garish candy capsules.
  - Numbers set with `tabular-nums font-mono`.

---

## 3. Key Product Decisions & Trade-Offs

- **Decision 1: Multi-Attribute Array vs Single Target Field**:
  - *Chosen Approach*: Expand `targetAttribute?: ResidentAttribute` to also support `targetAttributes?: ResidentAttribute[]` across the type definition, ontology extractor, parameter validator, and synthesizer.
  - *Why*: Allows backwards compatibility with single-attribute queries while natively handling compound queries (`care level + caregiver`, `room + wing`, `status + admission date`).
  - *Alternatives Considered*: Forcing users to ask one question at a time (rejected: poor UX for medical staff).

- **Decision 2: Compound Headline Synthesis**:
  - *Chosen Approach*: Combine values into a clear, unified headline (`Assisted Living · Nurse David Kim, LPN`), followed by a paired spotlight card.
  - *Why*: Gives staff the immediate answer within the first 5 words they read.

---

## 4. Technical Architecture & Data Strategy

### Request Flow & Component Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Query: "Donald Draper belongs to which care level and care taker ?"          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Semantic Slot & Multi-Attribute Extractor                          │
│ • Entity: Donald Draper (RES-1006)                                          │
│ • Multi-Attribute Slots: ['careType', 'primaryCaregiver']                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: MCP Dispatch (`care_app.get_resident_by_name_or_id`)               │
│ • Returns: Resident record for Donald Draper                                │
│ • requestedFields: ['careType', 'primaryCaregiver']                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 4: Composite Response Synthesizer & Multi-Spotlight Card              │
│ • Headline: "Assisted Living · Nurse David Kim, LPN"                        │
│ • Spotlight: [ { label: 'Care Level', value: 'Assisted Living' },           │
│                { label: 'Primary Caregiver', value: 'Nurse David Kim, LPN' }]│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed Changes Across Codebase
1. **`src/types/resident.ts`**:
   - Add `targetAttributes?: ResidentAttribute[]` to `ExtractedParameters`.
   - Add `multiSpotlight?: AttributeSpotlight[]` to `QueryExecutionResult['response']`.
2. **`src/pipeline/semanticOntology.ts`**:
   - Implement `detectTargetAttributes(query: string): AttributeSpotlightDefinition[]` to detect all mentioned attributes in compound sentences.
3. **`src/pipeline/deterministicParser.ts`**:
   - Extract all matching attributes and populate both `targetAttributes` and `targetAttribute` (for backwards compatibility).
4. **`src/pipeline/queryOrchestrator.ts`**:
   - Synthesize composite headline and `multiSpotlight` array when multiple attributes are present.
5. **`src/components/ChatMessageItem.tsx`**:
   - Render multi-column spotlight card when `multiSpotlight` is present.
6. **`src/components/QuickQueryChips.tsx`**:
   - Add test chip for: `"Donald Draper belongs to which care level and care taker ?"`
