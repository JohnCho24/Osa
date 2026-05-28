# MVP scope — what's IN, what's OUT

*Discipline document. If a future feature request doesn't move one of the
"validation questions" below, it doesn't get built until M0 has happened.*

---

## The one persona

**The mid-week analyst at a tier-2 European or tier-1 American football club.**

- 28–40 years old, sports-science background, dashboard-comfortable, R/Python-curious.
- Works for a head coach who does not personally use software during a match.
- Their job is to prepare a 30-minute Monday-morning film session on the weekend's match. They currently spend ~12 hours doing it across video, Excel, and a tactics-board app.
- They have institutional access to the club's tracking-data feed (Stats Perform / Hawk-Eye / Sportec / SkillCorner / Metrica).
- They have authority to evaluate new tools but NOT to sign a six-figure contract.
- They are the **person who would champion us internally** if the product solves their problem.

We are NOT building for: head coaches (use software during matches at most ~3% of the time), broadcasters (different sales motion, deferred to M5+), fans (B2C, deferred to M8+), academic researchers (don't pay).

---

## The one moment

**Monday morning, 09:15, the analyst has a still frame from Saturday's match on screen.**

They want to answer: *"if our left back had pressed instead of dropping, would we have prevented the goal?"*

Today they: pause the video, drag pieces on a static tactics board, describe the alternative in prose, attach a screenshot to a Slack message for the head coach.

With us they: draw an arrow on the left back, hit Generate, watch the alternative play out for 4 seconds, screenshot the 2D side-by-side, paste it into the same Slack message.

**That's the entire MVP.** Everything else is feature creep until the first ten analysts have done this and either kept using us or churned.

---

## IN the MVP

- [x] Renderer with a single match clip loaded
- [x] Player arrow drawing (the killer mechanic)
- [x] Inference server returning a per-frame trajectory JSON
- [x] Side-by-side compare (actual vs alternative)
- [x] Physics post-processor so the output is legible
- [x] Hard-projection guarantee that the player ends at the arrow target
- [x] Schema-versioned checkpoint loading (so the model can be swapped without breaking clients)
- [x] Eval harness so we can quantify whether a new model is actually better
- [x] HTTPS-deployable server (FastAPI + API key + request bounds)
- [ ] **A real trained model** (M7) — gates the value of every other item
- [ ] **A signed evaluation agreement with one analyst** (M0) — gates whether the MVP matters

## OUT of the MVP (defer or kill)

| Feature | Status | Why deferred |
|---|---|---|
| Photorealistic 3D video generation | Out | Downstream vendor concern; not our brain. Demo with 2D until M0 conversations confirm 3D is the actual unblock. |
| Multi-arrow joint plausibility | Out | The single-arrow case isn't validated yet. Don't optimize a workflow we haven't seen anyone use. |
| Team / league / objective style conditioning | Out | Paper does this; we don't. Adds 6+ weeks of training work for unverified user value. Revisit post-M0. |
| Event head / tactical event labels | Out | Code exists, parked. Labels aren't aligned to our training data. Adds complexity for unclear payoff. |
| Touch / iPad support | Shipped, demote | Built it; cool. No analyst has asked. Don't prioritize polish until one does. |
| Overlay diff in compare mode | Shipped, demote | Same. |
| K=20 grid view of alternatives | Out (M3.2 deprioritized) | Original roadmap idea, supplanted by arrow-driven generation. |
| Authentication beyond API key | Out | One-key-per-customer is enough for design partners. OAuth / SSO is post-revenue. |
| Multi-tenant infra | Out | Single-tenant deployment per customer is fine until customer #3. |
| Compliance posture (SOC 2, GDPR DPA, FIFPro) | Out, scoped only | Don't start the cost until the first paying customer requires it. Have the templates ready, don't run the audit. |
| Ball-pass arrows | Shipped, demote | Cool feature; no analyst has asked. Demote until validated. |

---

## Validation questions the MVP exists to answer

The MVP is a research instrument as much as a product. Each question has a yes/no answer that should change the roadmap:

1. **Can an analyst learn to use it unattended?** Test: send the URL to 5 analysts cold, watch what they do without a demo. Pass = ≥3 generate an alternative in <5 minutes.
2. **Is the alternative compelling enough to share with the head coach?** Test: of those 5, do any of them voluntarily screenshot the result and forward it? Pass = ≥1.
3. **Would they pay for it?** Test: explicit ask after the trial. Pass = anyone says "yes, my budget would cover that, here's the procurement process."
4. **Does the alternative look real enough to a head coach?** Test: show the alternative to a head coach with the analyst. Pass = the coach engages with the tactical content, not the rendering quality.

If we get 4/4 yeses across 5 analysts, we know we have a product. If we get 0/4 yeses, we don't, and the next sprint pivots, not iterates.

---

## Anti-scope: things explicitly not the product

- **A general-purpose ML platform.** We solve one tactical-counterfactual question; we don't sell "the model" as an API for arbitrary downstream use.
- **A broadcaster tool** until M5 (data partnership) is signed.
- **A consumer fan-app** ever, at this stage of the company.
- **An open-source project.** The repo is internal-facing; we don't accept external PRs and we don't market the code.

---

## Success metric

**Number of distinct analyst-organization accounts that generated ≥3 alternatives in the past 7 days.** Weekly active analyst orgs. Nothing else.

Not API requests. Not signups. Not page views. Not stars. **Distinct organizations that came back twice in a week.**

The first time this number hits 5, we have the world's smallest sports-tech traction story. Until then, every metric in the dashboard is vanity.
