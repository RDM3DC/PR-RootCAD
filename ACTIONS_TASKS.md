# GITHUB ACTIONS CI/CD ROADMAP — 200 TASKS

## Foundations & Repo Hygiene
1. Add a top-level CI badge in `README.md` for build status.
2. Create a central `.github/workflows/ci.yml` orchestrator document.
3. Define `default` permissions to `read-all` in all workflows.
4. Set `concurrency` groups to cancel superseded runs per-branch.
5. Add a repo-wide `CODEOWNERS` for CI-related file paths.
6. Add required status checks for `main` branch protection.
7. Enforce PR reviews minimum of 1 for CI-impacting changes.
8. Limit workflow runs to pull_request from trusted forks only where needed.
9. Document CI conventions in `CONTRIBUTING.md` (naming, labels, skip rules).
10. Add a `ci/` label and auto-apply it for workflow file changes.
11. Create a changelog policy section for CI modifications.
12. Add `paths` filters to limit runs to relevant directories.
13. Add `paths-ignore` to skip runs on docs-only edits, where appropriate.
14. Ensure `.gitattributes` sets `eol` norms to avoid platform diffs.
15. Pin actions by major version (e.g., `actions/checkout@v4`) and track updates.

## Python Environment & Caching
16. Use `actions/setup-python@v5` with `cache: pip` enabled.
17. Cache pip based on `requirements` hashes and `pyproject.toml`.
18. Add Windows-specific wheel caching for `numpy`/`PySide6` if used in CI.
19. Install minimal test dependencies (pytest, numpy) for unit-only jobs.
20. Add optional install for `PySide6` in GUI matrix with conditional steps.
21. Add `pip compile` job to lock deps (if using pip-tools) and cache results.
22. Validate `python --version` and `pip list` logs at start of each job.
23. Add `pip install -e .` job for package layout checks.
24. Cache `~/.cache/pip` and `~/.local` directories on Linux.
25. Use `uv` (optional) to speed installs in a future task.
26. Avoid `sudo apt-get` in Ubuntu unless necessary; document when used.
27. Add `PIP_NO_CACHE_DIR=off` to leverage caches on large deps.
28. Validate that `adaptivecad` imports with a one-liner smoke script.
29. Create a reusable composite action to set `PYTHONPATH` to repo root.
30. Add a pre-job to fail fast if Python version doesn’t match matrix.

## Linting, Formatting, Typing
31. Add a `lint.yml` workflow to run flake8/ruff over the repo.
32. Configure `ruff` with a stable ruleset and a baseline if needed.
33. Add `black` format check with `--check` (no reformat in CI).
34. Add `isort` check for import ordering.
35. Add `mypy` type-checking workflow with a baseline config.
36. Add `pyproject.toml` sections for ruff/black/isort/mypy consistency.
37. Fail the CI on lint/type errors; publish annotations to PRs.
38. Use `python -m pip install` exact versions of linters for reproducibility.
39. Add `pre-commit` job to run hooks in CI with `pre-commit.ci` style.
40. Cache `pre-commit` to speed subsequent runs.
41. Enable `pylint` selectively for critical modules with a strict config.
42. Add `bandit` static security checks for Python.
43. Lint YAML of workflows using `actionshub/yamllint`.
44. Validate Markdown links in docs using `lycheeverse/lychee-action`.
45. Add spelling checks for docs using `codespell`.
46. Enforce no-tabs or consistent indentation via editorconfig checks.
47. Add `pip check` to verify dependency conflicts after install.
48. Fail CI on unused imports (ruff rule) to keep code clean.
49. Add a job comment summarizing lint results in PR via `peter-evans/create-or-update-comment`.
50. Export SARIF from linters (where supported) for code scanning upload.

## Unit Tests
51. Run `pytest -q` for core modules (fast unit subset) on every push.
52. Split unit tests to a `unit` marker and run in parallel (`-n auto`).
53. Enable `pytest-randomly` to uncover order-dependent tests.
54. Add `--maxfail=1` to fail fast on catastrophic failures.
55. Collect coverage for unit subset (`coverage.py`) and upload artifact.
56. Report coverage to PR via a summarized comment.
57. Gate PRs on minimum coverage delta (e.g., no decrease for touched files).
58. Parametrize unit tests across Python versions 3.10/3.11/3.12.
59. Test on Windows and Ubuntu for unit path issues.
60. Add a specialized job for `adaptivecad/sketch` tests (geometry, units).
61. Add a retry step for flaky tests with `pytest-rerunfailures`.
62. Store `pytest` JUnit XML and upload as an artifact.
63. Publish test timings to identify slow tests.
64. Track slowest 20 tests and output to logs for triage.
65. Add a `--durations=10` report to each test job.
66. Ensure deterministic seeds with `PYTHONHASHSEED` and test-level seeds.
67. Run tests with `-W error` to fail on unexpected warnings.
68. Introduce `pytest-cov` branch coverage for core math modules.
69. Add coverage for `adaptivecad.linalg` and `adaptivecad.geom` modules.
70. Split tests into shards to keep jobs under 10 mins.
71. Add `pytest-timeout` to abort hanging tests.
72. Verify `pytest.ini` markers are enforced and documented.
73. Add `coverage combine` for multi-job coverage aggregation (future).
74. Cache `.pytest_cache` between runs to speed discovery (optional).
75. Integrate `pytest-xdist` to parallelize heavy modules.

## Integration Tests
76. Create an `integration.yml` workflow for heavier, scenario tests.
77. Run CLI import/export paths (e.g., STL, AMA) headless.
78. Test `export_slices.py` with a small fixture and validate files exist.
79. Test conversion commands guard paths (analytic ↔ mesh) conditionally.
80. Add end-to-end script to import `.stl` and export `.stl` round-trip.
81. Validate `run_playground.py` imports but do not open GUI (headless safe).
82. Mock OpenGL/OCC where necessary to avoid GPU dependencies.
83. Smoke-test `adaptivecad.gui.playground` module import.
84. Run `check_qt.py` in a headless-friendly mode or skip on CI.
85. Ensure `environment.yml` solves on Ubuntu (conda job optional later).
86. Validate examples execute lightweight (e.g., `example_script.py`).
87. Test `tools/torus_knot_cascade_cli.py` with a small param set.
88. Validate JSON schema compatibility for sketch save/load.
89. Ensure molecule loader reads sample `examples/molecules/*.xyz`.
90. Save artifacts from integration runs (exports, logs) for triage.
91. Add a failure triage step to upload logs on error.
92. Test `tests/test_import_system.py` core cases headless.
93. Add `pytest -m "not gui"` marker to exclude GUI-only tests here.
94. Validate OCC presence conditionally, skip if not available.
95. Upload integration test evidence (zip of outputs) as artifacts.

## GUI/Qt Smoke Tests (Careful)
96. Add a minimal Qt smoke test that only imports PySide6 classes.
97. Guard GUI tests behind `if: runner.os == 'windows-latest'` initially.
98. Set `QT_QPA_PLATFORM=offscreen` for import-only tests.
99. Skip any test that calls `app.exec()` in CI.
100. Add a dedicated `qt_smoke_test.py` job that exits quickly.
101. Cache PySide6 wheels on Windows to reduce install time.
102. Measure import time to keep under 10s (soft goal).
103. Add optional Ubuntu GUI matrix later if stable.
104. Gate merges on smoke test pass once stable for Windows.
105. Provide a CI-safe `--no-gui` flag to relevant entry points.
106. Add log capture of Qt plugin paths for diagnostics.
107. Document GUI test policy in `PLAYGROUND_GUIDE.md` (CI notes).
108. Auto-skip GUI smoke on forks without permissions to install large deps.
109. Add `pytest --maxfail=1` for GUI smoke to fail fast.
110. Upload Qt plugin diagnostic logs on failure.

## Cross-Platform Matrix
111. Run the unit suite on `ubuntu-latest` and `windows-latest`.
112. Add macOS job later (opt-in due to runner minutes cost).
113. Use `shell: bash` vs `shell: pwsh` as appropriate per OS.
114. Normalize path separators in tests for cross-platform compatibility.
115. Add `dos2unix` conversions for text fixtures where necessary.
116. Use `actions/setup-python` `cache: pip` across all OS.
117. Verify file permissions for scripts on Linux (chmod +x as needed).
118. Add Windows CRLF normalization step for generated files.
119. Ensure test temp directories resolve on all OS (use `tempfile`).
120. Upload OS-specific artifacts with OS suffix in names.

## Build Artifacts & Packaging
121. Add a `build.yml` that builds any Python package wheels/sdist.
122. Upload built wheels as artifacts for PRs.
123. Add `twine check` to validate package metadata.
124. Add optional PyPI test release on tagged pre-releases.
125. Sign artifacts (optional future) or checksum generation.
126. Add caching for `build/` and `dist/` to speed rebuilds.
127. Build docs artifacts in a separate job and upload HTML.
128. Produce a combined test log artifact per run.
129. Publish `playground_run.log` on failures for GUI smoke.
130. Build OpenGL shader cache (if any) and upload (optional).
131. Generate coverage HTML and upload as artifact.
132. Publish `pip freeze` artifact for reproducibility.
133. Store benchmark CSV outputs as artifacts (later task).
134. Add `release.yml` to attach artifacts to GitHub Releases.
135. Document artifact retention policy (days) and sizes.

## Docs & Site
136. Add `docs.yml` to build docs (mkdocs or Sphinx if added later).
137. Deploy docs to GitHub Pages on `main` if build passes.
138. Add link checker to docs pipeline.
139. Add `doctest` stage if using Sphinx notebooks.
140. Build API reference stubs with `sphinx-apidoc` (future).
141. Cache `~/.cache/pip` during docs build.
142. Fail docs build on warnings (`-W`) for strictness.
143. Upload docs HTML as artifact on PRs for preview.
144. Comment PRs with a temporary docs preview URL (pages preview action).
145. Add `README` examples validation script in docs CI.

## Security & Compliance
146. Enable GitHub CodeQL workflow for Python code scanning.
147. Configure CodeQL to run weekly and on PRs touching Python files.
148. Upload linter SARIF (ruff/bandit) to the Security tab when supported.
149. Add `pip-audit` to scan Python dependencies for CVEs.
150. Enable Dependabot for `pip` and GitHub Actions updates.
151. Auto-label and auto-merge safe Dependabot updates after tests pass.
152. Add `secretlint` or `gitleaks` to scan for secrets in commits.
153. Restrict `pull_request_target` usage; document security implications.
154. Add license scanning for dependencies (pip-licenses).
155. Validate license headers if policy requires (optional).
156. Enforce SLSA provenance for releases (future task).
157. Rotate repository secrets schedule documented.
158. Mask secrets in workflow logs and redact on failure.
159. Add policy to prevent artifacts from containing secrets.
160. Enable branch protection rules for workflow files.

## PR Automation & Hygiene
161. Add `labeler.yml` to auto-label PRs based on file paths.
162. Add `stale.yml` to auto-mark inactive issues/PRs.
163. Add auto-assign reviewers (e.g., geometry, GUI, CI owners).
164. Add changelog bot or PR title linter for conventional commits.
165. Add size labels (XS, S, M, L, XL) based on diff lines.
166. Add a check that tests were added/updated for code changes.
167. Comment with summary of affected tests and coverage.
168. Auto-request re-run when flaky tests fail (one retry).
169. Post coverage trend chart comment (last N runs).
170. Validate that `pytest.ini` markers are not abused in PRs.
171. Enforce that workflows use pinned action versions (policy check).
172. Notify in PR if new large dependencies are introduced.
173. Run a license compliance check on each PR.
174. Auto-close PRs that modify workflow secrets (if attempted).
175. Block merging when workflow files fail lint.

## Performance & Benchmarks
176. Add a `benchmarks.yml` workflow to run micro-benchmarks (if added).
177. Cache and compare CSV results across runs; post delta in PR.
178. Guard benchmarks to run nightly due to runtime cost.
179. Add a timeout budget per benchmark to avoid timeouts.
180. Tag performance regressions as `perf-regression` automatically.
181. Add profiling script for hot functions and upload flamegraphs.
182. Track test runtime per module over time in artifact.
183. Fail PRs if runtime exceeds set thresholds (soft gate initially).
184. Create a perf dashboard issue updated by a bot (optional).
185. Add basic memory usage checks for key algorithms.

## Monitoring & Reporting
186. Enable build summaries using GitHub Job Summaries (markdown output).
187. Post flaky test report to PR with failure signatures.
188. Add workflow run badges for unit, integration, lint, docs.
189. Create a `CI_STATUS.md` updated nightly with recent runs.
190. Notify Slack/Discord webhook (if configured) on `main` failures.
191. Track average queue and run times; add to summary.
192. Publish a weekly CI health report as an Issue comment.

## Workflow Maintenance & Scaling
193. Split mega workflows into smaller, focused jobs to reduce coupling.
194. Convert common steps into composite actions in `.github/actions/`.
195. Add `workflow_call` reusable workflows for matrix reuse.
196. Add `if: contains(github.event.head_commit.message, '[skip ci]')` (policy).
197. Implement self-hosted runners for GPU/Qt heavy jobs (future).
198. Introduce job artifacts pruning policies for cost control.
199. Review and bump action versions quarterly; log changes.
200. Document a CI incident response playbook in `CONTRIBUTING.md`.

---

IMPLEMENTATION NOTE: START BY HARDENING FOUNDATIONS (1–15), THEN ENABLE UNIT TEST MATRIX (51–75) AND LINT/TYPING (31–50). ADD SECURITY SCANS (146–160), INTEGRATION TESTS (76–95), AND MINIMAL GUI SMOKE (96–110). SCALE OUT PACKAGING, DOCS, AND REPORTING AFTER CORE CI IS STABLE.
