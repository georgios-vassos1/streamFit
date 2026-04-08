# streamFit — Claude Code project context

## Skills

| Skill | Purpose |
|---|---|
| `/test-developer` | Write or revise tests in `tests/testthat/` |

Skills are defined in `.claude/skills/` and provide expert workflows, reference sources, and quality checklists.

## Package overview

R package for online/streaming statistical estimation.

| Module | File | Class |
|---|---|---|
| Recursive Least Squares (Gaussian) | `R/rls.R` | `stream_fit` |
| RLS-GLM (natural-gradient online IRLS) | `R/rls_glm.R` | `stream_fit` |
| Implicit SGD | `R/isgd.R` | `stream_fit` |
| Piecewise-exponential Cox | `R/pe_cox.R` | `pe_cox_fit` |
| Batch Super Learner | `R/super_learner.R` | `super_learner_fit` |
| Online Super Learner (SGD weights) | `R/online_super_learner.R` | `online_sl` |
| Online ATE / ATT estimator | `R/online_ate.R` | `online_ate` |

## Key conventions

- **Intercept**: callers supply the intercept column in `W`/`X`; functions never add it.
- **Families**: standard R `family` objects throughout (`gaussian()`, `binomial()`, `poisson()`).
- **Online update pattern**: always predict with the current model *before* updating it (predict-then-update).
- **Internal helpers**: prefix with `.` (e.g. `.g_clamp`, `.softmax`); document with `@keywords internal`.
- `make_learner()` constructs `sl_learner` objects for Super Learner libraries.

## Workflow requirements

### Before committing
CI enforces that `man/` and `NAMESPACE` are up to date:

```r
devtools::document()   # regenerates man/*.Rd from @-tags
```

Stage the generated `man/*.Rd` files together with any `R/*.R` changes.
**NAMESPACE is hand-maintained** (not roxygen2-managed); edit it directly.

### Running tests
```r
devtools::test()                        # full suite
devtools::test(filter = "online_ate")   # single module
```

### Commit style
Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.
Keep the subject line under 72 characters.

## CI

GitHub Actions runs `R CMD CHECK` plus a documentation staleness check
(`roxygen2::roxygenise()` followed by `git diff --name-only`).
The check fails if any `man/` file is missing or out of date.
