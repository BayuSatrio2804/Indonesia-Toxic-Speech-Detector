# Vite Frontend Implementation Plan — Indonesia Hatespeech Detector

## Context

The repo already has a working FastAPI backend (`HateSpeech-BE/`) exposing three endpoints
(`/health`, `/model-info`, `/predict`) for an Indonesian binary toxic-speech classifier
(IndoBERTweet, threshold 0.49, max 128 tokens, no auth, CORS pre-configured for
`http://localhost:5173`). A fresh React 19 + Vite 7 + Tailwind v4 + shadcn (base-lyra/mist)
scaffold lives in `HateSpeech-FE/` with only the `Button` component installed and a working
`ThemeProvider` (system default, `d` key toggle).

The goal is a **clean, minimal, dashboard-style single-page UI** that lets a user:
1. Paste Indonesian text and see a toxicity verdict + confidence scores.
2. Browse their own past predictions (kept in this browser via localStorage).
3. Inspect model metadata and backend health.
4. Switch UI language between English and Bahasa Indonesia, and switch dark/light theme.

Deployment target is **Cloudflare Pages** (static hosting). No tests in scope for this
iteration. No router, no TanStack Query, no React Hook Form — those would be overkill for
three endpoints and one form field.

## Working Directory

All work lives in `HateSpeech-FE/`.

## Dependencies to Add

```
npm i react-i18next i18next i18next-browser-languagedetector sonner
```

shadcn components to install via the shadcn CLI:
```
npx shadcn@latest add sidebar card badge progress textarea label separator sonner tooltip scroll-area dropdown-menu
```

Already present (do not re-add): `button`, `theme-provider`, `lucide-react`, Tailwind v4,
path alias `@/`, ESLint + Prettier.

## API Contract (verbatim from backend)

```ts
// GET /health
type Health = { status: "ok"; model_loaded: boolean };

// GET /model-info
type ModelInfo = {
  model_key: string;
  model_name: string;
  labels: ["non_toxic", "toxic"];
  threshold: number;
  max_length: number;
};

// POST /predict
type PredictRequest = { text: string }; // 1..5000 chars, trimmed
type PredictResponse = {
  label: "non_toxic" | "toxic";
  is_toxic: boolean;
  scores: { non_toxic: number; toxic: number };
  threshold: number;
};
```

Errors: 422 (validation), 503 (model unavailable). Wrap `fetch` in `lib/api.ts` and throw
typed `ApiError` instances; surface them as toasts and inline messages.

## Folder Layout (inside `HateSpeech-FE/src/`)

```
src/
  components/ui/                  # shadcn primitives (auto-generated)
  components/
    AppShell.tsx                  # header + sidebar + main content slot
    StatusDot.tsx                 # green/red dot bound to /health result
    LanguageSwitcher.tsx          # dropdown EN / ID
    ThemeToggle.tsx               # sun/moon button using existing ThemeProvider
  features/
    detector/
      DetectorPanel.tsx
      ResultCard.tsx              # verdict badge + two Progress bars + threshold marker
      SampleChips.tsx             # 3–4 curated Indonesian examples
    history/
      HistoryPanel.tsx
      useHistory.ts               # localStorage-backed hook
    model-info/
      ModelInfoPanel.tsx
  lib/
    api.ts                        # predict(), getHealth(), getModelInfo(), ApiError
    types.ts                      # Health, ModelInfo, Predict* (mirroring backend)
  i18n/
    index.ts                      # i18next.init({...}) with browser language detector
    en.json
    id.json
  App.tsx                         # ThemeProvider > I18nextProvider > AppShell + <Toaster />
  main.tsx                        # unchanged except importing ./i18n
```

Reuse `src/lib/utils.ts` (`cn` helper) — already present from the shadcn scaffold.

## UI Specification

**Layout.** AppShell has a fixed header and a `Sidebar` (shadcn) on the left. Header shows
app title, `StatusDot`, `LanguageSwitcher`, `ThemeToggle`. Sidebar lists three items with
`lucide-react` icons: Detector (default), History, Model Info. Selection is local
`useState` in `AppShell` — no router. Sidebar collapses to a `Sheet` on mobile (native to
shadcn Sidebar).

**Detector panel.**
- 3–4 sample chips above the textarea (clickable; populate the textarea).
- `Textarea` with character counter (`<current>/5000`); button disabled when empty or > 5000.
- Primary `Button` "Analyze" with loading spinner; also bound to Ctrl/Cmd+Enter.
- `ResultCard` below input shows: large badge (`TOXIC` red / `NON-TOXIC` green), two
  `Progress` bars labeled "non_toxic %" and "toxic %", and a one-line note about the
  threshold (`Threshold: 0.49`).
- On success, append to history.

**History panel.**
- `ScrollArea` with rows: truncated text, label `Badge`, relative timestamp.
- Click row → loads text back into Detector and switches active panel.
- "Clear all" button (with `Tooltip` + confirmation via `sonner` undo toast).
- Cap at 50 entries (FIFO) so localStorage doesn't grow unbounded.

**Model Info panel.**
- Static `Card` rendering fields from `/model-info` plus the last `/health` result.
- "Refresh" button to re-fetch both.

**Header status dot.** One-shot `getHealth()` on mount (re-runs on Refresh in Model Info or
when a `/predict` call fails). Green = `model_loaded: true`, red = error or
`model_loaded: false`. Hover `Tooltip` reads "Backend healthy" / "Backend unreachable".

**Errors.** All API errors raised via `sonner` toast. Detector additionally renders an
inline error in place of the result card.

## State Strategy

- Theme: existing `ThemeProvider` (no change).
- Language: `I18nextProvider` from `react-i18next` (built-in).
- App-wide light state in `AppShell` only:
  - `activePanel: "detector" | "history" | "model-info"`
  - `health: Health | { error: string }`
  - `modelInfo: ModelInfo | null`
- Detector state: local `useState` for `text`, `loading`, `result`, `error`.
- History state: `useHistory()` hook reading/writing `localStorage["hatespeech.history"]`.

No Context beyond ThemeProvider + I18nextProvider. No global store.

## i18n Setup

`src/i18n/index.ts`:
```ts
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import LanguageDetector from "i18next-browser-languagedetector";
import en from "./en.json";
import id from "./id.json";

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    fallbackLng: "en",
    supportedLngs: ["en", "id"],
    resources: { en: { translation: en }, id: { translation: id } },
    interpolation: { escapeValue: false },
    detection: { order: ["localStorage", "navigator"], caches: ["localStorage"] },
  });

export default i18n;
```

Imported once in `main.tsx` before `<App />`. Sample-chip *texts* stay in Indonesian in
both locales (the model only handles Indonesian); only labels around them translate.

## Cloudflare Pages Configuration

- Build command: `npm run build`
- Build output directory: `dist`
- Node version: pin via `.nvmrc` (e.g. `20`) — Pages reads it automatically.
- Env var (Pages dashboard, both Preview + Production): `VITE_API_BASE_URL` pointing at the
  deployed backend URL.
- Commit `.env.example` with `VITE_API_BASE_URL=http://localhost:8000`.
- Add `public/_headers` for caching:
  ```
  /assets/*
    Cache-Control: public, max-age=31536000, immutable

  /*
    Cache-Control: public, max-age=0, must-revalidate
  ```
- No `_redirects` needed (no client-side router).

## Files to Create or Modify

**Modify:**
- `HateSpeech-FE/package.json` — add the four runtime deps listed above.
- `HateSpeech-FE/src/main.tsx` — import `./i18n` and wrap with `I18nextProvider` (or rely
  on the `initReactI18next` global instance — choose the latter for simplicity).
- `HateSpeech-FE/src/App.tsx` — replace placeholder content with `<ThemeProvider>` >
  `<AppShell />` + `<Toaster />` (sonner).
- `HateSpeech-FE/src/index.css` — keep existing Tailwind + shadcn vars; verify sidebar
  CSS variables are present after `npx shadcn add sidebar`.

**Create:**
- All files under `components/`, `features/`, `lib/`, `i18n/` listed in the folder layout.
- `HateSpeech-FE/.env.example`
- `HateSpeech-FE/.nvmrc`
- `HateSpeech-FE/public/_headers`

## Critical Files to Reference During Implementation

- `HateSpeech-BE/app/main.py`, `app/routers/*.py`, `app/schemas/*.py` — confirm endpoint
  shapes match `lib/types.ts`. Source of truth for the API contract.
- `HateSpeech-FE/components.json` — shadcn config (style: base-lyra, base color: mist).
- `HateSpeech-FE/src/components/theme-provider.tsx` — existing pattern; keep, do not
  rewrite.
- `HateSpeech-FE/vite.config.ts` and `tsconfig*.json` — confirm `@/` path alias still
  resolves after adding folders.

## Verification (end-to-end)

1. **Backend up:** in one terminal,
   `cd HateSpeech-BE && uvicorn app.main:app --reload` — confirm `GET /health` returns
   `{"status":"ok","model_loaded":true}`.
2. **Frontend dev:** in another terminal,
   `cd HateSpeech-FE && npm install && npm run dev` — open `http://localhost:5173`.
3. **Smoke test:**
   - Header shows green status dot.
   - Click a sample chip → text fills textarea → click "Analyze" → verdict + probability
     bars render within ~1s.
   - Toggle EN ↔ ID: all UI labels swap, persists across reload.
   - Toggle theme via header button and via `d` key — both work.
   - Submit empty / over-5000-char input — button disabled / clear inline message.
   - Stop backend → click Analyze → red toast appears, status dot turns red.
   - Reload page → History panel still lists previous predictions; clicking one loads it
     back into the Detector.
   - Resize to mobile width → sidebar becomes a Sheet trigger.
4. **Static checks:**
   `npm run typecheck && npm run lint && npm run build` — all pass, `dist/` populated.
5. **Cloudflare preview (manual, after first PR):** push branch, let Pages build a preview,
   set `VITE_API_BASE_URL` to the deployed backend, repeat the smoke test on the preview URL.

## Out of Scope (explicit non-goals)

- Authentication, accounts, server-side history.
- Batch upload / CSV processing.
- Multi-page routing.
- Test suite (Vitest/RTL) — deferred.
- Backend changes of any kind.
- Telemetry / analytics.
