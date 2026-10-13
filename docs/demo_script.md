# GeoRescue AI — Three-Minute Demo Script

**Multimodal AI Hackathon 2026 — Track B: Mapping Flood Damage from Space**

Total duration: 3 minutes (180 seconds). Timings are approximate; a
small buffer is built in.

Before you start:

- Dashboard is running: `streamlit run app.py`.
- Browser is open to `http://localhost:8501`.
- Terminal window is visible on a second screen or beside the browser.
- Do not read from a script; this is a guide, not a teleprompter.
- Speak clearly. Pause briefly between sections.

---

## Scene 1 — User inputs (0:00 – 0:20)

**Show:** the sidebar with the AOI inputs and the event date.

**Say:**

> "This is GeoRescue AI, a system that answers three questions from
> satellite data for any area and date. On the left, the user picks
> the area of interest and the event date. The defaults are the
> Trishuli corridor in Nepal for the August 2026 flood, but every
> input here is user-supplied. Nothing is hardcoded."

**Do:** click the event date and hover over one of the AOI fields.

---

## Scene 2 — Satellite acquisition (0:20 – 0:50)

**Show:** the terminal with the pipeline logs.

**Say:**

> "The pipeline starts by searching the Copernicus Data Space
> Ecosystem for Sentinel-1 radar scenes that cover this area before
> and after the event. Radar is chosen because it sees through
> monsoon clouds. The system selects a compatible pre/post pair —
> same relative orbit, same direction, roughly twelve days apart —
> so the images can be compared pixel by pixel."

**Do:** point to the two product names in the log.

> "For the precomputed Trishuli case study, the pair is 16 August and
> 28 August 2026, both ASCENDING. When the pipeline is re-run from
> scratch with a wider search window, it selects a different but
> equally valid pair — 24 August and 5 September, both DESCENDING,
> same relative orbit. Both pairs are pixel-comparable. The pipeline
> does not hardcode the pair; it chooses the best available
> same-orbit pair within the requested window."

---

## Scene 3 — Flood mapping (0:50 – 1:20)

**Show:** the Flood mapping section in the dashboard.

**Say:**

> "After terrain correction, we compute the ratio of post-event to
> pre-event backscatter. Water scatters radar energy away from the
> sensor, so a strong drop in backscatter is the signature of
> flooding. We apply a slope mask, a minimum mapping unit, and a
> median speckle filter. The result is a candidate flood extent —
> the words 'candidate' and 'extent' matter, because this is not a
> confirmed damage map."

**Do:** point to the blue polygons on the map.

> "For Trishuli, the system maps 3.05 square kilometres across 181
> polygons. For the Syapru Besi AOI we ran later for evaluation,
> 0.28 square kilometres across 32 merged polygons."

---

## Scene 4 — Infrastructure exposure (1:20 – 1:45)

**Show:** the Infrastructure exposure section.

**Say:**

> "The flood extent is then intersected with pre-event OpenStreetMap
> buildings, roads, and bridges. OSM data is a historical snapshot
> dated 27 July 2026, before the event — not current OSM."

**Do:** point to the four KPI cards.

> "For the Trishuli corridor: 514 buildings intersect the candidate
> extent, with an exposed footprint of 3.86 hectares. 1,256 road
> segments intersect, totaling 64 kilometres. 11 bridges intersect."

**Pause, then say:**

> "These are potentially exposed features. They are not confirmed
> damaged. A building whose footprint touches a flood polygon may or
> may not have sustained damage. The system is explicit about that
> distinction."

---

## Scene 5 — Settlement connectivity (1:45 – 2:10)

**Show:** the Access disruption section.

**Say:**

> "This is the part rescuers care about most. We build a road-network
> graph from the pre-event OpenStreetMap data using NetworkX. Nodes
> are intersections and dead-ends; edges are road segments. Then we
> compute baseline connectivity: which settlements can reach a
> hospital or a town."

**Do:** point to the four KPI cards.

> "249 settlements in the graph. 87 were connected to a destination
> at baseline. 85 remain connected after we remove road segments
> whose flooded length exceeds 20% of their segment length."

**Show:** the two OSM IDs below.

> "Two settlements lose their connection. OSM IDs 2313807996 and 8187696685. These are the settlements a rescue team should
> investigate first."

**Pause, then say:**

> "This is a modelled scenario. A flooded road is not automatically
> impassable in reality. The system labels the output as
> potentially cut off, not confirmed cut off."

---

## Scene 6 — AI situation report (2:10 – 2:40)

**Show:** the Situation report section. Click Generate report.

**Say:**

> "Finally, the situation-report copilot. Every number in this report
> is computed by Python and stored in a structured evidence object.
> The language model is given that evidence and is instructed never
> to invent a number. Every number it produces is validated against
> the evidence. If it writes anything unsupported, the response is
> rejected and we fall back to a deterministic report."

**Do:** point to the source label.

> "This report was generated by — see the source field — either the
> LLM or the deterministic fallback. Both paths produce the same
> numbers and the same structure. The difference is style."

**Show:** the Copilot tab. Click one suggested question.

> "Rescuers can also ask free-form questions. The copilot answers
> from the same evidence."

---

## Scene 7 — Data provenance and limitations (2:40 – 3:00)

**Show:** the Data provenance and limitations section.

**Say:**

> "Every result traces to a source file and a processing step. The
> system lists all twelve metrics with their source layer and method,
> plus the required attributions — Copernicus Sentinel data,
> Copernicus WorldDEM-30, and OpenStreetMap contributors."

**Do:** scroll to the attribution block.

> "We have also evaluated the Syapru Besi flood extent against the
> official Copernicus EMSR927 reference for AOI01. The IoU is 0.007
> against the mass-movement footprint and 0.009 against buffered
> damaged buildings. Those numbers are low because the two layers
> represent different phenomena — a flood extent and a landslide
> footprint. We report them honestly. We did not tune the algorithm
> against the reference, because that would violate the competition
> rules."

**Close:**

> "GeoRescue AI is a research prototype for the hackathon. It is
> not an operational emergency-response system. Thank you."

---

## Short version (if time is tight)

If the demo slot is shorter than three minutes, collapse scenes 3, 4,
and 5 into a single walkthrough of the Map, then jump straight to
scene 6. The Map already shows all the layers; a single sentence about
each KPI card is enough.

## Q&A talking points

Anticipated questions and short answers:

**"Why is the IoU so low?"**

> Because the reference is a landslide-damage map and our candidate
> is a flood-water extent. They are different phenomena. We report
> the honest comparison and explain why the numbers are low.

**"Did you train a model?"**

> No. We implement a transparent threshold-based baseline, not a
> trained model. We did not use Kuro Siwo or Sen1Floods11 because we
> did not have time to train and validate a model in the hackathon
> window. Using them would be a next step.

**"What if the LLM hallucinates a number?"**

> Our validator extracts every number from the response and checks
> it against the evidence object. If any number is unsupported, we
> reject the response and fall back to a deterministic answer.

**"Is this hardcoded to the Trishuli flood?"**

> No. The pipeline accepts any AOI and any event date. We
> demonstrated this by re-running the entire pipeline for the Syapru
> Besi AOI, which is a different location and uses different
> Sentinel-1 slices.

**"What happens if Sentinel-1 doesn't cover an area?"**

> The system reports that no suitable scene was found and stops.
> It does not fabricate data.

**"What are the limitations?"**

> Sentinel-1 revisit is 6–12 days; sudden events cannot be warned
> about in advance. The detector is threshold-based, not trained. OSM
> coverage in remote areas is incomplete. The system is a prototype,
> not an operational tool.

**"Why do some tiles fail to download from OSM?"**

> The ohsome API has rate limits. Our tiled fetch uses exponential
> backoff and continues with the tiles that succeed. The summary
> reports how many tiles were successfully queried.

**"How did you pick the thresholds?"**

> The Trishuli thresholds (ratio 0.4, slope 5°) came from the flood
> literature and produced a physically plausible extent. The Syapru
> Besi thresholds (ratio 0.5, slope 25°) were adjusted for the
> steeper terrain and higher elevation of that AOI. Neither was
> tuned against the EMSR927 reference.
