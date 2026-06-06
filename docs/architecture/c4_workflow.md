# B Tactical What-If Engine - C4 Workflow Diagrams

This document maps the project using C4-style architecture views. It is grounded
in the current repository structure, especially `README.md`, `index.html`,
`src/render/pitch.js`, `src/server/main.py`, `src/model/*`, `scripts/*`, and
`docs/handoff_api.md`.

## Scope And Assumptions

- The core product workflow is: load a match moment, draw tactical arrows,
  generate counterfactual trajectories, and compare actual vs alternative on a
  2D pitch.
- Runtime data and checkpoints are generated artifacts. The clean checkout
  inspected for this diagram does not include `data/` or `checkpoints/`, but the
  application and docs expect `data/processed/*.json`,
  `data/processed/samples.json`, and `checkpoints/*/*.ckpt`.
- The README references `src/data/metrica_to_gentac.py` and
  `src/data/extract_clip.py`, but `src/data/` is not present in this checkout.
  The diagrams therefore show tracking conversion as an expected offline ingest
  stage rather than an implemented runtime container.
- The photorealistic 3D/video layer is intentionally out of scope for this
  repo. This project owns trajectory generation and exposes a handoff contract
  for downstream rendering.

## Legend

| C4 concept | Meaning in these diagrams |
| --- | --- |
| Person | Coach, analyst, buyer, or operator interacting with the system |
| Software system | The product as a whole or an external system outside repo ownership |
| Container | Deployable or independently runnable part: browser UI, API server, scripts, stores |
| Component | Major code module inside a container |
| Data store | File or object-store backed artifact used by runtime or training |
| Dynamic view | Ordered workflow across containers and components |

## C4 Level 1 - System Context

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, ui-sans-serif, system-ui", "primaryTextColor": "#0f172a", "lineColor": "#475569"}}}%%
flowchart LR
  coach["Coach / Analyst / Player<br/>Reviews match moments, draws tactical intent, compares outcomes"]
  buyer["League / Club Buyer<br/>Provides data access, validates workflow, purchases deployment"]
  operator["Founder / ML Operator<br/>Runs training, deploys API, monitors design-partner usage"]

  bSystem["B Tactical What-If Engine<br/>Resimulates football moments from tracking history plus tactical arrows.<br/>Owns 2D renderer, GenTac diffusion model, physics post-processing, and trajectory API."]

  trackingProvider["Tracking Data Providers<br/>Metrica sample data today; Hawk-Eye, Sportec DFL, Stats Perform, SkillCorner later"]
  cloudGpu["GPU Training Providers<br/>Local Mac MPS for smoke tests; Modal, Lambda Labs, RunPod, or vast.ai for A100/H100 runs"]
  videoGen["Downstream 3D / Video Generator<br/>Future consumer of trajectory keypoints through handoff contract"]
  mailClient["Mail Client<br/>Landing-page demo inquiry via mailto link"]

  coach -->|"Selects decision frame, draws player run or ball-pass arrows, clicks Generate"| bSystem
  bSystem -->|"Returns actual vs alternative 2D playback with 23-entity trajectories"| coach

  buyer -->|"Supplies league constraints, data partnership, validation feedback"| bSystem
  bSystem -->|"Provides design-partner demo, handoff API, deployment plan"| buyer

  operator -->|"Runs smoke/full training, evaluates checkpoints, starts server"| bSystem
  bSystem -->|"Logs request IDs, health status, model schema, eval metrics"| operator

  trackingProvider -->|"Tracking JSON or provider feed: frames, ball, team/player coordinates, FPS, pitch metadata"| bSystem
  bSystem -->|"Full-training jobs and checkpoint artifacts"| cloudGpu
  cloudGpu -->|"Trained checkpoint candidates"| bSystem
  bSystem -->|"Trajectory JSON: history, actual_future, samples, physical guarantees"| videoGen
  bSystem -->|"mailto:leagues@bstartup.dev"| mailClient

  classDef person fill:#fff7ed,stroke:#ea580c,color:#111827,stroke-width:2px;
  classDef system fill:#e0f2fe,stroke:#0369a1,color:#0f172a,stroke-width:3px;
  classDef external fill:#f8fafc,stroke:#64748b,color:#0f172a,stroke-width:2px;

  class coach,buyer,operator person;
  class bSystem system;
  class trackingProvider,cloudGpu,videoGen,mailClient external;
```

## C4 Level 2 - Container View

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, ui-sans-serif, system-ui", "primaryTextColor": "#0f172a", "lineColor": "#475569"}}}%%
flowchart TB
  user["Coach / Analyst Browser"]
  buyer["League Buyer Browser"]
  rawTracking["Raw Tracking Data<br/>Provider files or feed"]
  cloudGpu["Cloud GPU Runner<br/>Modal / Lambda / RunPod / vast.ai"]
  videoGen["Downstream 3D / Video Generator"]

  subgraph bBoundary["B Tactical What-If Engine"]
    direction TB

    subgraph browserBoundary["Browser-Facing Containers"]
      direction LR
      landing["Landing Website<br/>site/index.html, site/app.js, site/style.css<br/>Static product site and mailto contact flow"]
      renderer["2D Tactics Board<br/>index.html, src/render/pitch.js, src/render/style.css<br/>Vanilla JS + Canvas, draw mode, compare mode, samples mode"]
    end

    subgraph apiBoundary["Inference Runtime"]
      direction LR
      fastapi["FastAPI Inference Server<br/>src/server/main.py<br/>/api/health, /api/generate, /api/generate/stream"]
      modelPkg["GenTac Model Package<br/>src/model/*<br/>Config, dataset, tokenizer, backbone, diffusion, sampler, physics, Lightning wrapper"]
    end

    subgraph offlineBoundary["Offline ML And QA"]
      direction LR
      trainingScripts["Training Scripts<br/>scripts/smoke_train.py, scripts/train_full.py, scripts/modal_train.py<br/>Smoke, dry-run, full paper-config training"]
      evalScripts["Sampling And Evaluation Scripts<br/>scripts/sample.py, scripts/eval.py<br/>Generate samples.json, ADE/FDE/diversity/speed/off-pitch/arrow-honor metrics"]
      tests["Automated Tests<br/>tests/*<br/>Request validation, physics invariants, checkpoint schema"]
    end

    subgraph dataBoundary["Artifact Stores"]
      direction LR
      processedData[("Processed Match JSON<br/>data/processed/*.json<br/>Runtime match frames, masks, metadata")]
      samplesData[("Sample Output JSON<br/>data/processed/samples.json<br/>history + actual_future + generated samples")]
      checkpoints[("Model Checkpoints<br/>checkpoints/smoke/*.ckpt, checkpoints/full/*.ckpt<br/>Schema-versioned Lightning checkpoints")]
      handoffDoc[("Handoff Contract<br/>docs/handoff_api.md<br/>Trajectory API for future 3D/video layer")]
    end
  end

  buyer -->|"Reads product story and sends inquiry"| landing
  landing -->|"Opens mailto inquiry"| buyer

  user -->|"GET static page over local http.server"| renderer
  renderer -->|"GET Sample_Game_1_clip.json or samples.json"| processedData
  renderer -->|"POST /api/generate<br/>decision_frame, match, horizon, k, arrows, mode, guidance_scale"| fastapi
  fastapi -->|"samples.json-shaped response"| renderer

  fastapi -->|"Loads checkpoint at startup; calls model + sampler + physics"| modelPkg
  fastapi -->|"Safe path resolution + match-array cache"| processedData
  fastapi -->|"Reads smoke/full checkpoint through GenTacTrajectoryModule"| checkpoints

  rawTracking -->|"Expected offline conversion to GenTac JSON"| processedData
  trainingScripts -->|"Reads processed matches through GenTacDataModule"| processedData
  trainingScripts -->|"Writes checkpoints and schema metadata"| checkpoints
  cloudGpu <-->|"Runs full paper-config training path"| trainingScripts

  evalScripts -->|"Reads match data and checkpoint"| processedData
  evalScripts -->|"Loads model checkpoint"| checkpoints
  evalScripts -->|"Writes sample alternatives for renderer samples mode"| samplesData
  renderer -->|"GET samples.json in ?samples=1 mode"| samplesData
  tests -->|"Validate bounds, paths, physics, schema contracts"| fastapi
  tests -->|"Validate post-processor and config roundtrip"| modelPkg
  handoffDoc -->|"Defines contract consumed later"| videoGen
  fastapi -->|"Production shape: HTTPS trajectory response"| videoGen

  classDef person fill:#fff7ed,stroke:#ea580c,color:#111827,stroke-width:2px;
  classDef container fill:#dcfce7,stroke:#15803d,color:#052e16,stroke-width:2px;
  classDef api fill:#e0f2fe,stroke:#0369a1,color:#0f172a,stroke-width:2px;
  classDef script fill:#fef9c3,stroke:#a16207,color:#422006,stroke-width:2px;
  classDef store fill:#fee2e2,stroke:#b91c1c,color:#450a0a,stroke-width:2px;
  classDef external fill:#f8fafc,stroke:#64748b,color:#0f172a,stroke-width:2px;

  class user,buyer person;
  class landing,renderer container;
  class fastapi,modelPkg api;
  class trainingScripts,evalScripts,tests script;
  class processedData,samplesData,checkpoints,handoffDoc store;
  class rawTracking,cloudGpu,videoGen external;

  style bBoundary fill:#ffffff,stroke:#334155,stroke-width:2px,color:#0f172a
  style browserBoundary fill:#f8fafc,stroke:#94a3b8,color:#0f172a
  style apiBoundary fill:#f8fafc,stroke:#94a3b8,color:#0f172a
  style offlineBoundary fill:#f8fafc,stroke:#94a3b8,color:#0f172a
  style dataBoundary fill:#f8fafc,stroke:#94a3b8,color:#0f172a
```

## C4 Level 3 - Inference And Model Components

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, ui-sans-serif, system-ui", "primaryTextColor": "#0f172a", "lineColor": "#475569"}}}%%
flowchart TB
  renderer["2D Tactics Board<br/>POSTs arrow-conditioned generation requests"]
  processedData[("data/processed/*.json<br/>frames, masks, period, frame_ids, team slots")]
  checkpoints[("checkpoints/*/*.ckpt<br/>schema-versioned Lightning checkpoint")]

  subgraph server["FastAPI Container: src/server/main.py"]
    direction TB
    lifespan["Lifespan Loader<br/>Chooses mps or cpu, loads checkpoint, restores cfg, model, schedule"]
    middleware["HTTP Middleware<br/>Content-Length guard, CORS allowlist, JSON request logging, X-Request-Id"]
    authSchema["API Key + Pydantic Schemas<br/>GenerateRequest, PlayerArrow, BallPassArrow, request bounds"]
    queueGuard["Concurrency And Queue Guard<br/>GENTAC_MAX_CONCURRENCY semaphore, GENTAC_MAX_QUEUE fast fail"]
    matchResolver["Match Resolver<br/>_safe_match_path, _get_match, in-memory match_cache"]
    arrowMapper["Arrow To Waypoint Mapper<br/>player pin, ball_pass pin, recipient receiving-stance offset"]
    plausibility["Joint Plausibility Check<br/>max reachable distance, r_min spacing, pitch-normalized target clamp"]
    orchestrator["_generate_sync Orchestrator<br/>Builds history, valid_static, target_mask, mode-specific opponent future"]
    stream["SSE Stream Wrapper<br/>/api/generate/stream emits metadata, window chunks, done"]
    responseBuilder["Response Builder<br/>Denormalizes meters, rounds frame dicts, returns history/actual_future/samples"]
  end

  subgraph model["GenTac Model Package: src/model/*"]
    direction TB
    cfg["GenTacConfig<br/>105 x 68 pitch, 23 entities, 25 FPS, H=100, w=5, DDPM steps, waypoint CFG"]
    dataset["_load_match_arrays / TrajectoryDataset<br/>Stable 11-player slots per team, normalized coords, validity masks"]
    lightning["GenTacTrajectoryModule<br/>Checkpoint cfg roundtrip, schema validation, EMA buffers, Lightning training shell"]
    schedule["LinearBetaSchedule<br/>beta, alpha, alpha_bar tables and q_sample"]
    diffusion["GenTacDiffusion<br/>epsilon-prediction network"]
    tokenizer["TrajectoryTokenizer<br/>coordinate, temporal, group, entity, waypoint/null embeddings"]
    backbone["SpatioTemporalBackbone<br/>factorized temporal and entity attention blocks"]
    sampler["causal_rollout + sample_window<br/>autoregressive w-frame reverse diffusion with hard pins and learned waypoints"]
    physics["apply_physics<br/>seam anchor, speed caps, EMA smoothing, pitch clamp, player repulsion, final re-clip"]
  end

  renderer -->|"JSON body with arrows and mode"| middleware
  middleware --> authSchema
  authSchema --> queueGuard
  queueGuard --> orchestrator
  lifespan -->|"loads cfg/model/schedule"| lightning
  checkpoints --> lifespan

  orchestrator --> matchResolver
  matchResolver -->|"safe read + cached arrays"| processedData
  matchResolver --> dataset
  orchestrator --> arrowMapper
  arrowMapper --> plausibility
  plausibility --> sampler

  cfg --> dataset
  cfg --> tokenizer
  cfg --> backbone
  cfg --> sampler
  cfg --> physics
  lightning --> diffusion
  lightning --> schedule
  diffusion --> tokenizer
  diffusion --> backbone
  sampler -->|"per diffusion step predicts epsilon"| diffusion
  sampler --> schedule
  sampler -->|"raw normalized samples"| physics
  physics -->|"physically plausible meter-space samples"| responseBuilder
  responseBuilder -->|"samples.json-shaped payload"| renderer
  stream -->|"wraps _generate_sync output by rollout windows"| responseBuilder

  classDef external fill:#f8fafc,stroke:#64748b,color:#0f172a,stroke-width:2px;
  classDef server fill:#e0f2fe,stroke:#0369a1,color:#0f172a,stroke-width:2px;
  classDef model fill:#f3e8ff,stroke:#7e22ce,color:#1f102e,stroke-width:2px;
  classDef store fill:#fee2e2,stroke:#b91c1c,color:#450a0a,stroke-width:2px;

  class renderer external;
  class processedData,checkpoints store;
  class lifespan,middleware,authSchema,queueGuard,matchResolver,arrowMapper,plausibility,orchestrator,stream,responseBuilder server;
  class cfg,dataset,lightning,schedule,diffusion,tokenizer,backbone,sampler,physics model;

  style server fill:#f8fbff,stroke:#0ea5e9,stroke-width:2px,color:#0f172a
  style model fill:#fbf7ff,stroke:#a855f7,stroke-width:2px,color:#0f172a
```

## C4 Level 3 - Browser Renderer Components

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, ui-sans-serif, system-ui", "primaryTextColor": "#0f172a", "lineColor": "#475569"}}}%%
flowchart TB
  coach["Coach / Analyst"]
  clipData[("data/processed/Sample_Game_1_clip.json")]
  samplesData[("data/processed/samples.json")]
  api["FastAPI Inference Server<br/>http://127.0.0.1:8001"]

  subgraph browser["2D Tactics Board: index.html + src/render/pitch.js"]
    direction TB
    boot["main Bootstrapping<br/>Binds DOM controls, URL params, panel labels, status text"]
    loader["JSON Loader<br/>loadJson, samples-mode switch, error status if data missing"]
    panelFactory["Panel Factory<br/>createPanel, drawPitch, drawFrame, trails, currentFrame/currentFrameId"]
    playback["Playback Controls<br/>play/pause, scrubber, speed, trails, compare mode"]
    samplePicker["Alternative Picker<br/>buildVirtualClip, sample prev/next, actual vs generated panels"]
    drawMode["Draw Mode State Machine<br/>hover, drag, right-click clear, Esc cancel, touch-to-mouse"]
    arrowOverlay["Arrow Overlay<br/>player arrows, ball-pass dashed shaft, recipient ring, hover pulse"]
    payloadBuilder["Payload Builder<br/>decision_frame, match path, dynamic horizon, fade_frames, guidance_scale, mode, arrows"]
    generateClient["Generate API Client<br/>fetch POST /api/generate, swaps panels to actual vs alternative"]
    overlayDiff["Overlay Diff<br/>Ghosts actual future on alternative panel for visual divergence"]
    onboarding["First-Run Onboarding<br/>localStorage gentac.onboarded, start/skip/dismiss wiring"]
  end

  coach -->|"Loads localhost page and chooses frame"| boot
  boot --> loader
  loader -->|"normal mode GET"| clipData
  loader -->|"samples=1 mode GET"| samplesData
  loader --> panelFactory
  panelFactory --> playback
  panelFactory --> samplePicker

  coach -->|"Clicks player, drags arrow, drags ball then clicks recipient"| drawMode
  drawMode --> arrowOverlay
  arrowOverlay --> panelFactory
  drawMode --> payloadBuilder
  payloadBuilder -->|"JSON body"| generateClient
  generateClient -->|"POST /api/generate"| api
  api -->|"history + actual_future + samples"| generateClient
  generateClient --> samplePicker
  samplePicker --> playback
  overlayDiff --> panelFactory
  onboarding --> drawMode

  classDef person fill:#fff7ed,stroke:#ea580c,color:#111827,stroke-width:2px;
  classDef ui fill:#dcfce7,stroke:#15803d,color:#052e16,stroke-width:2px;
  classDef store fill:#fee2e2,stroke:#b91c1c,color:#450a0a,stroke-width:2px;
  classDef external fill:#f8fafc,stroke:#64748b,color:#0f172a,stroke-width:2px;

  class coach person;
  class boot,loader,panelFactory,playback,samplePicker,drawMode,arrowOverlay,payloadBuilder,generateClient,overlayDiff,onboarding ui;
  class clipData,samplesData store;
  class api external;

  style browser fill:#f7fff9,stroke:#22c55e,stroke-width:2px,color:#0f172a
```

## C4 Dynamic View - Arrow-Driven Generation Workflow

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, ui-sans-serif, system-ui", "primaryTextColor": "#0f172a", "lineColor": "#475569"}}}%%
flowchart LR
  coach["1. Coach / Analyst"]
  uiLoad["2. Browser loads tactics board<br/>normal clip mode or samples mode"]
  frameSelect["3. Decision frame selected<br/>scrubber or URL ?frame=N"]
  drawIntent["4. Tactical intent captured<br/>player run arrows and optional ball_pass recipient"]
  payload["5. Payload built<br/>dynamic feasible horizon, fade_frames=horizon-1, k=1, mode=unconditioned"]
  apiIngress["6. FastAPI ingress<br/>CORS, size guard, request ID, optional X-API-Key, schema bounds"]
  matchLoad["7. Match loaded safely<br/>path constrained to data/processed, arrays cached"]
  waypointMap["8. Arrows mapped to waypoints<br/>player pins, ball pin, recipient stance offset"]
  plausible["9. Plausibility gate<br/>speed reachability and player spacing checks"]
  rollout["10. Causal rollout<br/>H=100 history frames, w=5 windows, DDPM reverse steps, learned waypoints"]
  physics["11. Physics post-pass<br/>seam anchor, speed caps, smoothing, pitch clamp, repulsion, final pin safety net"]
  response["12. Response assembled<br/>metadata, history, actual_future, samples in meter-space JSON"]
  compare["13. UI comparison<br/>left actual, right alternative, synced playback, optional overlay diff"]
  handoff["14. Later handoff<br/>same trajectory JSON can drive downstream 3D/video service"]

  coach --> uiLoad --> frameSelect --> drawIntent --> payload --> apiIngress --> matchLoad --> waypointMap --> plausible --> rollout --> physics --> response --> compare --> coach
  response -.-> handoff

  classDef person fill:#fff7ed,stroke:#ea580c,color:#111827,stroke-width:2px;
  classDef ui fill:#dcfce7,stroke:#15803d,color:#052e16,stroke-width:2px;
  classDef api fill:#e0f2fe,stroke:#0369a1,color:#0f172a,stroke-width:2px;
  classDef model fill:#f3e8ff,stroke:#7e22ce,color:#1f102e,stroke-width:2px;
  classDef contract fill:#f8fafc,stroke:#64748b,color:#0f172a,stroke-width:2px;

  class coach person;
  class uiLoad,frameSelect,drawIntent,payload,compare ui;
  class apiIngress,matchLoad,waypointMap,plausible,response api;
  class rollout,physics model;
  class handoff contract;
```

## C4 Dynamic View - Training, Sampling, And Evaluation Workflow

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, ui-sans-serif, system-ui", "primaryTextColor": "#0f172a", "lineColor": "#475569"}}}%%
flowchart TB
  rawTracking["1. Raw Tracking Data<br/>Metrica sample data today; provider data later"]
  ingest["2. Expected Ingest / Conversion<br/>Tracking feed to GenTac match JSON and dev clips<br/>Documented in README, not present as src/data in this checkout"]
  processed[("3. Processed JSON Store<br/>data/processed/*.json<br/>frames, ball, team0, team1, pitch metadata")]

  subgraph training["Training Path"]
    direction TB
    dm["4. GenTacDataModule<br/>discovers matches, match-level val split"]
    dataset["5. TrajectoryDataset<br/>H=100 history + w=5 future windows, normalized coords, validity masks"]
    lightning["6. GenTacTrajectoryModule<br/>random pretrain mode, optimizer, cosine schedule, EMA, schema hparams"]
    loss["7. DDPM Training Step<br/>synthetic waypoints, CFG dropout, q_sample, epsilon loss"]
    smoke["8a. Smoke Training<br/>scripts/smoke_train.py<br/>Mac MPS/CPU, tiny config, 3 epochs"]
    full["8b. Full Training<br/>scripts/train_full.py / scripts/modal_train.py<br/>A100/H100, paper config, dry-run preflight"]
    ckpt[("9. Checkpoint Store<br/>checkpoints/smoke or checkpoints/full<br/>cfg_dict + schema_version")]
  end

  subgraph eval["Sampling And Evaluation Path"]
    direction TB
    sampleScript["10. scripts/sample.py<br/>loads checkpoint, samples K alternatives, applies physics"]
    samplesJson[("11. data/processed/samples.json<br/>renderer-ready actual vs alternatives")]
    evalScript["12. scripts/eval.py<br/>ADE/FDE, diversity, max speed, off-pitch, role ADE, realism discriminator, arrow honor sweep"]
    evalJson[("13. checkpoints/run/eval.json<br/>machine-readable metrics")]
  end

  subgraph validation["Automated Guardrails"]
    direction LR
    serverTests["tests/test_server_validation.py<br/>Pydantic bounds, path traversal, body size"]
    physicsTests["tests/test_physics.py<br/>speed caps, seam anchor, pitch clamp, no NaN"]
    schemaTests["tests/test_config_schema.py<br/>checkpoint cfg roundtrip and schema mismatch"]
  end

  renderer["2D Renderer<br/>?samples=1 consumes samples.json"]
  cloudGpu["Cloud GPU Provider<br/>Modal / Lambda / RunPod / vast.ai"]

  rawTracking --> ingest --> processed
  processed --> dm --> dataset --> lightning --> loss
  loss --> smoke --> ckpt
  loss --> full --> ckpt
  cloudGpu --> full
  ckpt --> sampleScript
  processed --> sampleScript --> samplesJson --> renderer
  ckpt --> evalScript
  processed --> evalScript --> evalJson
  serverTests --> validation
  physicsTests --> validation
  schemaTests --> validation
  validation -->|"keeps runtime contracts from drifting"| ckpt

  classDef source fill:#f8fafc,stroke:#64748b,color:#0f172a,stroke-width:2px;
  classDef process fill:#fef9c3,stroke:#a16207,color:#422006,stroke-width:2px;
  classDef model fill:#f3e8ff,stroke:#7e22ce,color:#1f102e,stroke-width:2px;
  classDef store fill:#fee2e2,stroke:#b91c1c,color:#450a0a,stroke-width:2px;
  classDef ui fill:#dcfce7,stroke:#15803d,color:#052e16,stroke-width:2px;
  classDef guard fill:#e0f2fe,stroke:#0369a1,color:#0f172a,stroke-width:2px;

  class rawTracking,cloudGpu source;
  class ingest,dm,dataset,smoke,full,sampleScript,evalScript process;
  class lightning,loss model;
  class processed,ckpt,samplesJson,evalJson store;
  class renderer ui;
  class serverTests,physicsTests,schemaTests guard;

  style training fill:#fffbeb,stroke:#d97706,stroke-width:2px,color:#0f172a
  style eval fill:#fff7ed,stroke:#ea580c,stroke-width:2px,color:#0f172a
  style validation fill:#f0f9ff,stroke:#0284c7,stroke-width:2px,color:#0f172a
```

## C4 Deployment View - Current Local State And First-Customer Target

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, ui-sans-serif, system-ui", "primaryTextColor": "#0f172a", "lineColor": "#475569"}}}%%
flowchart LR
  subgraph current["Current Local Development Deployment"]
    direction TB
    localBrowser["Developer Browser<br/>http://localhost:8000"]
    httpServer["Python http.server<br/>serves index.html and src/render/*"]
    localApi["uvicorn src.server.main:app<br/>127.0.0.1:8001"]
    localFs[("Local Filesystem<br/>data/processed, checkpoints, docs")]
    localModel["Smoke Model In Memory<br/>Mac MPS if available, otherwise CPU"]

    localBrowser --> httpServer
    localBrowser -->|"POST /api/generate"| localApi
    httpServer --> localFs
    localApi --> localFs
    localApi --> localModel
  end

  subgraph target["First Paid Customer Target Deployment"]
    direction TB
    customerBrowser["Customer Browser / API Client"]
    cloudflare["Cloudflare<br/>TLS, WAF, rate limiting"]
    caddy["Caddy On Single GPU VM<br/>TLS termination / reverse proxy"]
    uvicorn["Uvicorn + FastAPI<br/>systemd process, model loaded in RAM"]
    gpuModel["A10G/A100 Model Runtime<br/>one active generation by default"]
    s3[("S3 Bucket<br/>checkpoints, request artifacts, access logs")]
    vercel["Vercel / Static Host<br/>landing site and renderer assets"]
    monitoring["Missing/Planned Observability<br/>metrics endpoint, alerts, artifact capture, key rotation"]

    customerBrowser --> vercel
    customerBrowser -->|"HTTPS generate request"| cloudflare --> caddy --> uvicorn --> gpuModel
    uvicorn --> s3
    uvicorn --> monitoring
  end

  current -.->|"docs/operations.md maps gaps to this target"| target

  classDef currentClass fill:#dcfce7,stroke:#15803d,color:#052e16,stroke-width:2px;
  classDef targetClass fill:#e0f2fe,stroke:#0369a1,color:#0f172a,stroke-width:2px;
  classDef store fill:#fee2e2,stroke:#b91c1c,color:#450a0a,stroke-width:2px;
  classDef risk fill:#fff7ed,stroke:#ea580c,color:#431407,stroke-width:2px;

  class localBrowser,httpServer,localApi,localModel currentClass;
  class customerBrowser,cloudflare,caddy,uvicorn,gpuModel,vercel targetClass;
  class localFs,s3 store;
  class monitoring risk;

  style current fill:#f7fff9,stroke:#22c55e,stroke-width:2px,color:#0f172a
  style target fill:#f8fbff,stroke:#0ea5e9,stroke-width:2px,color:#0f172a
```

## Architectural Notes

- The browser renderer is deliberately static and framework-free. It expects the
  processed JSON schema directly, so generated samples and ground-truth clips
  share the same rendering path.
- The API response intentionally matches `scripts/sample.py` output. That lets
  the same UI render offline samples and live arrow-conditioned inference.
- The model is held in memory at FastAPI startup. Request-time work is mostly
  match slicing, waypoint construction, diffusion sampling, physics
  post-processing, and response serialization.
- Safety and operability are already represented in code: path traversal guard,
  request body limit, bounded `k`, bounded horizon, bounded arrow count, optional
  API key, request IDs, queue cap, and checkpoint schema validation.
- Production readiness gaps remain operational rather than architectural:
  generated artifacts are local, there is no Dockerfile, metrics endpoint, alert
  routing, artifact capture, key rotation, HTTPS setup, or deployed GPU host yet.

## Source Grounding

| Area | Repository source |
| --- | --- |
| Product workflow and roadmap | `README.md`, `docs/product_vision.md`, `docs/mvp.md` |
| Browser UI workflow | `index.html`, `src/render/pitch.js`, `src/render/style.css` |
| Inference API and validation | `src/server/main.py`, `tests/test_server_validation.py` |
| Model internals | `src/model/config.py`, `dataset.py`, `tokenizer.py`, `backbone.py`, `diffusion.py`, `samplers.py`, `physics.py`, `lightning_module.py` |
| Training and evaluation | `scripts/smoke_train.py`, `scripts/train_full.py`, `scripts/modal_train.py`, `scripts/sample.py`, `scripts/eval.py`, `docs/cloud_training.md` |
| Handoff and deployment | `docs/handoff_api.md`, `docs/operations.md` |
