# Webpage / obclip clipping for llm-wiki

Use this reference when a non-paper webpage, WeChat public-account article, X/article post, or dynamic blog needs browser-backed capture into the user's layered `llm-wiki`. This is a supplemental mechanics reference; `llm-wiki` remains the primary router and policy owner.

## Route owner boundary

- `llm-wiki` owns note policy, raw filename conventions, raw/body constraints, `_meta`/log integration, pending wiki integration, and native/zvec freshness reporting.
- This reference owns only webpage capture mechanics: obclip invocation, browser/profile choice, dynamic waits, WeChat/X failure recognition, image recovery, and cleanup of capture artifacts.
- Do not create a second clipping router. For paper-like links, use raw-fast automation first. For ordinary webpages, use this only when browser-backed capture or obclip-specific repair is needed.

## Method fidelity

If the user explicitly says `use obclip`, treat that as a method requirement.

- Prefer the user's Windows-native obclip workflow when the page is hostile, login-gated, or known to work better in Windows.
- For native Windows obclip, use a dedicated browser executable and profile configured locally in the project `.env` when needed; resolve their paths at runtime, not in this tracked skill.
- Keep the obclip output as the base capture when it succeeds; post-process, normalize, and verify it instead of rewriting the article from scratch.
- Fall back to rendered-DOM or manual reconstruction only after the obclip/browser route genuinely fails or saves a semantic placeholder rather than the article.

## Install / invoke check

Before a real capture, verify one of these works in the shell you will use:

```powershell
obclip --help
npx @harris7/obclip --help
node .\dist\cli.cjs --help
```

Use cases:

- `obclip ...` when the global command resolves.
- `npx -y @harris7/obclip ...` when avoiding a global install.
- `node .\dist\cli.cjs ...` only when intentionally working inside the source repo.
- If Playwright says Chromium or `chrome-headless-shell` is missing, install the browser runtime and retry the same command rather than changing the clipping plan.

## Browser/profile selection

- Public static page: start with plain obclip and a small settle delay.
- Dynamic SPA/blog: add `--settle-ms 3000` or `--settle-ms 5000`; add `--wait-selector` only when the selector is stable and content-bearing.
- Logged-in or anti-bot page: use `--browser-executable`, `--browser-profile`, and usually `--headful`.
- Use a dedicated profile, not the user's everyday Chrome profile.
- Do not assume WSL Playwright can automate a Windows Chromium binary. If Windows browsing works but WSL automation fails, run obclip natively in Windows.

## Command templates

After the project-local `.env` path bootstrap in `SKILL.md`, use the configured wiki root and optional browser paths; do not paste host-specific paths into this tracked reference.

Public page:

```bash
npx -y @harris7/obclip "https://example.com/article" --output "$LLM_WIKI_ROOT/raw/clip" --settle-ms 5000
```

Dedicated WSL Chromium/profile (only when both optional obclip keys are configured):

```bash
obclip "https://example.com/article" \
  --browser-executable "$LLM_WIKI_OBCLIP_WRAPPER" \
  --browser-profile "$LLM_WIKI_OBCLIP_PROFILE" \
  --headful --settle-ms 5000 \
  --output "$LLM_WIKI_ROOT/raw/clip"
```

Dynamic page with an explicit selector:

```bash
obclip "https://example.com/app" --wait-selector "article" --settle-ms 3000 --output "$LLM_WIKI_ROOT/raw/clip"
```

When native Windows browsing is required, convert the wiki output path resolved from `.env` to a Windows path (for example with `wslpath -w`) and supply dedicated Windows browser/profile paths from local configuration at runtime. Keep those machine-specific values out of this tracked skill. For long Windows commands launched from WSL/Hermes, use a temporary PowerShell script outside the wiki root rather than fragile nested quoting.

## Success detection

- `Saved note: <path>` may be printed to stderr.
- On Windows npm shims, `obclip.ps1` can print `Saved note:` while PowerShell reports `NativeCommandError` / nonzero exit. If `Saved note:` is present, verify by checking the saved file before declaring failure.
- A command exit code of `0` is not enough. Inspect the saved markdown for the real article title/body, not just login walls, CAPTCHA text, navigation chrome, or placeholder boilerplate.

## Failure interpretation and fallback

- If a wait selector times out but the page is reachable, retry once without the over-strict selector and keep only a settle delay.
- If the saved note is a login/signup shell, fix browser state with a dedicated profile; delay alone does not solve authentication.
- If WeChat saves `微信公众平台` or `环境异常，完成验证后即可继续访问。`, treat it as a failed semantic capture, delete the placeholder note, and rerun with a verified logged-in/profiled browser.
- If Windows Chromium closes early with Playwright/ICU/file-descriptor errors but the article is reachable in a browser, use rendered-DOM/static extraction as fallback and record the obclip failure in evidence/log, not in the raw-note body.
- For X long-form article posts, inspect the saved markdown before switching surfaces. The browser-accessible status page can expose a complete article while the `/article/<id>` surface shows a login wall.

## WSL dedicated browser fallback

When intentionally staying in WSL for a profile-gated page:

- Browser wrapper: `${LLM_WIKI_OBCLIP_WRAPPER}` from the repository `.env` (optional; configure before this route).
- Persistent profile: `${LLM_WIKI_OBCLIP_PROFILE}` from the repository `.env` (optional; configure before this route).
- Helper scripts when present: `obclip-wechat-open` and `obclip-wechat-clip`.
- Open the target page headfully with the same profile, complete verification/login, close the browser completely, then run obclip with the same profile.
- If Chromium reports ProcessSingleton / profile-in-use, close or kill the stale browser using that profile before retrying.

## WeChat image recovery

WeChat can save article text while replacing real images with `data:image/svg+xml` 1x1 placeholders. Repair this before closeout.

1. Reopen the original article with the same verified browser/profile.
2. Wait for `#js_content`, scroll to lazy-load images, and extract `#js_content img` metadata.
3. Prefer `img.dataset.src` / `data-src` over `src` or `currentSrc`; WeChat often keeps `src/currentSrc` as placeholders while `data-src` holds the real `mmbiz.qpic.cn` image.
4. Filter out avatar, QR code, reward/follow, and footer decoration images.
5. Download article images with browser-like headers and `Referer: <WeChat URL>` into `raw/images/<doc-folder>/`.
6. Rewrite markdown image lines to relative local paths such as `../images/<doc-folder>/img01.png`.
7. Replace whole placeholder image lines rather than using a naive `![](...)` URL-only regex. Data URI SVGs contain `)` and can leave broken XML tail fragments if only the URL substring is replaced.
8. After footer/promo cleanup, remove any orphan localized footer image whose surrounding footer text was deleted.

Verify after WeChat repair:

- No `data:image/svg+xml` remains.
- No remote `mmbiz.qpic.cn` markdown image URL remains.
- No stray SVG/XML fragments such as `fill='%23FFFFFF'` remain.
- Local image references resolve from the raw note path.
- Dedicated Chromium/profile processes are closed.

## llm-wiki post-processing

When the capture is destined for `llm-wiki/raw/clip`, do not stop at `Saved note:`.

- Rename to `raw/clip/<YYMM>/<YYMMDDNN>_<readable-title>.md` using the clipping/raw-note creation date and same-day sequence.
- Keep source URLs, route facts, failed probes, browser errors, image-recovery details, and link-health facts in frontmatter/evidence/log/closeout, not in the raw-note body.
- Remove promotional/footer/navigation noise such as subscribe prompts, course ads, account footer navigation, and “continue sliding” blocks.
- Localize retained images into `raw/images/<doc-folder>/` and rewrite note links to relative local paths.
- If the article points to a paper, keep the paper route/resource/probe details out of the raw body unless the current llm-wiki note contract explicitly allows that branch; prefer frontmatter/evidence/log/closeout for process details.
- Reopen the clipped markdown or source page immediately before final write if you spent time reading neighboring wiki files or `_meta` pages first.

## Final checks

- Canonical raw note exists and duplicate search by title/source resolves to it.
- Body contains the article content, not a placeholder/login/challenge page.
- Source/provenance is preserved in allowed metadata/evidence surfaces.
- No remote markdown images, `data:image`, or broken local image references remain.
- Raw body follows the current `llm-wiki` body constraints: no bare URLs/Markdown links/process evidence sections.
- Temp scripts, extracted JSON, scratch browser outputs, and placeholder captures are removed.
- Report raw saved / wiki pending / graph pending / graph fresh separately.
