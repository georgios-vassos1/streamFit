## ============================================================================
## Benchmark: iSGD vs RLS-GLM computational scaling
##
## SCOPE: This benchmark measures per-step cost and convergence quality under
## well-calibrated conditions.  It does NOT demonstrate global convergence from
## arbitrary initial states: both algorithms require hyperparameter tuning, and
## convergence is not guaranteed without it.
##
## INITIAL STATE: beta = 0 for both (principled default; no prior information).
##
## HYPERPARAMETER REQUIREMENT:
##   Both algorithms use hp_scale = 1/(w*p) = 4/p (balanced binomial, w=0.25).
##   - RLS-GLM: S0 = hp_scale * I.  The package's documented default
##     S0_scale = 100 diverges at p = 200 via an eta-clipping feedback loop;
##     calibrated S0 is required.
##   - iSGD:    gamma_n = hp_scale * n^{-0.7}.  Wrong gamma1 gives slow or
##     no convergence, but iSGD's implicit update is self-bounding and will
##     not diverge — only converge slowly.  RLS-GLM with wrong S0 can diverge.
##
## Key computational findings (verified on this machine):
##
##   - For small p (< ~50), RLS-GLM is *faster* per step.  The Sherman-Morrison
##     update is simple matrix algebra; iSGD's scalar root-find via uniroot()
##     carries a fixed overhead that dominates when p is small.
##
##   - The crossover occurs around p ~ 75-100.  Beyond that, RLS-GLM's O(p^2)
##     tcrossprod() and S %*% x become the bottleneck; iSGD stays O(p).
##
##   - At p = 200, iSGD is ~5x faster per step.
##   - At p = 500, iSGD is ~30x faster per step.
##
##   - Memory: RLS-GLM stores an S matrix — p^2 doubles.  iSGD stores only
##     beta — p doubles.  At p = 500, that is 501x more storage for RLS-GLM.
##
## Run from the package root:
##   source("inst/benchmarks/benchmark_scaling.R")
## or from the shell:
##   Rscript inst/benchmarks/benchmark_scaling.R
## ============================================================================

library(streamFit)

set.seed(42)

N_REPS <- 2000L   ## update steps per timing cell
P_GRID <- c(5L, 10L, 25L, 50L, 100L, 200L, 500L)

## ---------------------------------------------------------------------------
## Helper: time n_reps bare update calls; return microseconds per step.
## Low-level *_update() functions are used to isolate algorithmic cost.
## ---------------------------------------------------------------------------
time_rls_glm <- function(p, n_reps) {
  x <- rnorm(p); y <- 1L; beta <- numeric(p); S <- diag(p); fam <- binomial()
  elapsed <- system.time(
    for (i in seq_len(n_reps)) {
      out  <- rls_glm_update(x, y, beta, S, fam)
      beta <- out$beta
      S    <- out$S
    }
  )["elapsed"]
  elapsed / n_reps * 1e6
}

time_isgd <- function(p, n_reps) {
  x <- rnorm(p); y <- 1L; beta <- numeric(p); fam <- binomial()
  elapsed <- system.time(
    for (i in seq_len(n_reps)) {
      beta <- isgd_update(x, y, beta, gamma = 0.5, fam)
    }
  )["elapsed"]
  elapsed / n_reps * 1e6
}

## ---------------------------------------------------------------------------
## 1. Timing and memory table
## ---------------------------------------------------------------------------
cat("Computing per-step timings ...\n")
res <- data.frame(
  p        = P_GRID,
  rls_us   = sapply(P_GRID, time_rls_glm, n_reps = N_REPS),
  isgd_us  = sapply(P_GRID, time_isgd,    n_reps = N_REPS)
)
res$time_ratio <- res$rls_us / res$isgd_us
res$mem_rls    <- P_GRID^2L + P_GRID   ## S (p^2) + beta (p)
res$mem_isgd   <- P_GRID               ## beta only
res$mem_ratio  <- res$mem_rls / res$mem_isgd

cat("\n=== Per-step timing and storage ===\n")
cat("(time in µs/step; storage in 8-byte doubles)\n\n")
cat(sprintf("%-6s  %11s  %9s  %11s  %12s  %10s\n",
            "p", "RLS-GLM µs", "iSGD µs", "Speed-up",
            "RLS-GLM els", "Mem ratio"))
cat(strrep("-", 68), "\n")
for (i in seq_len(nrow(res))) {
  r <- res[i, ]
  speed <- if (r$time_ratio >= 1)
    sprintf("%8.1fx  (iSGD faster)", r$time_ratio)
  else
    sprintf("%8.1fx  (RLS-GLM faster)", 1 / r$time_ratio)
  cat(sprintf("%-6d  %11.2f  %9.2f  %s\n",
              r$p, r$rls_us, r$isgd_us, speed))
}
cat(sprintf("\nMemory at p = 500: RLS-GLM needs %dx more storage than iSGD.\n",
            res$mem_ratio[res$p == 500]))

## ---------------------------------------------------------------------------
## 2. Quality comparison at p = 200, n = 5000
##
## Timing is kept separate from quality measurement to avoid contamination:
## per-step costs come from the timing table above; the quality loop records
## only RMSE at each step without being timed itself.
##
## COEFFICIENT SCALING: b0 = rnorm(p) / sqrt(p) ensures eta_i = X_i'*b0 ~ N(0,1).
## With X_i ~ N(0, I_p), Var(eta_i) = ||b0||^2.  Using b0 / sqrt(p) gives
## ||b0|| ~ 1, keeping linear predictors well within eta_clip = 10 and
## Fisher weights bounded away from zero.  Without this scaling,
## ||rnorm(p)|| ~ sqrt(p) ~ 14 for p=200, every predictor is clipped.
##
## S0 CALIBRATION FOR RLS-GLM:
##   A diffuse prior S0_scale = 100 works for small p but diverges for large p.
##   Mechanism: with S0_scale=100 and p=200, beta grows via random walk until
##   |eta| hits eta_clip=10 (~25 steps).  At clipping, Fisher weight w -> 0,
##   so S stops deflating.  With S still large and nonzero score, beta steps
##   are huge (||S x score|| ~ 100 * 14 * 1 = 1400) and beta diverges.
##   Calibrated choice: S0 = 1/(w*p) = 4/p (balanced logistic, w ~ 0.25).
##   This matches prior precision to Fisher information per observation.
##
## SAME HYPERPARAMETER for both algorithms:
##   hp_scale = 4/p  =>  S0 = hp_scale * I  for RLS-GLM
##                        gamma_n = hp_scale * n^{-0.7}  for iSGD
##
## CRAMER-RAO FLOOR: sqrt(tr(F^{-1}) / p) = sqrt(1 / (w * n/p)) ~ sqrt(4/(n*0.25))
##   For n=5000, p=200: floor ~ 0.028.  RLS-GLM (second-order) reaches it;
##   iSGD (first-order) converges more slowly and lands ~2x above floor.
##
## Expected behaviour:
##   - RMSE vs n:    RLS-GLM drops steeply in the first ~p observations
##                   (second-order convergence); iSGD converges more slowly.
##   - RMSE vs time: iSGD's cheaper steps let it process more observations
##                   per second.  Whether it catches RLS-GLM depends on n/p:
##                   for n >> p (streaming regime) iSGD wins; for n ~ p
##                   RLS-GLM's observation-efficiency matters more.
## ---------------------------------------------------------------------------
cat("\nRunning quality comparison (p = 200, n = 5000) ...\n")

p_q  <- 200L
N_Q  <- 5000L
b0   <- rnorm(p_q) / sqrt(p_q)
Xq   <- cbind(1, matrix(rnorm(N_Q * (p_q - 1L)), N_Q, p_q - 1L))
yq   <- rbinom(N_Q, 1L, 1 / (1 + exp(-Xq %*% b0)))

## Shared hyperparameter: 1 / (w * p) where w ~ 0.25 (balanced binomial)
hp_scale <- 4 / p_q

## Cramér-Rao RMSE floor: sqrt(1 / (n*w)) for a balanced logistic model.
## Derivation: Fisher information per observation = w * E[x x'] ~ w * I_p
## (for standardised features).  After n observations, F_n ~ n*w*I_p, so
## each coordinate has variance >= 1/(n*w) and the average RMSE is
## sqrt(mean_j Var(beta_j)) = sqrt(1/(n*w)).
cr_floor <- sqrt(1 / (N_Q * 0.25))

## --- RLS-GLM quality path (calibrated S0 = hp_scale * I) -------------------
br     <- numeric(p_q)
Sr     <- hp_scale * diag(p_q)
rmse_r <- numeric(N_Q)
for (i in seq_len(N_Q)) {
  out    <- rls_glm_update(Xq[i, ], yq[i], br, Sr, binomial())
  br     <- out$beta
  Sr     <- out$S
  rmse_r[i] <- sqrt(mean((br - b0)^2))
}

## --- iSGD quality path (gamma_n = hp_scale * n^{-0.7}) ---------------------
bi     <- numeric(p_q)
rmse_i <- numeric(N_Q)
for (i in seq_len(N_Q)) {
  bi        <- isgd_update(Xq[i, ], yq[i], bi,
                            gamma = hp_scale * i^(-0.7), binomial())
  rmse_i[i] <- sqrt(mean((bi - b0)^2))
}

## Per-step times for p = 200 from the timing table (timing-loop measured,
## not contaminated by RMSE overhead)
us_r <- res$rls_us[res$p  == 200L]
us_i <- res$isgd_us[res$p == 200L]

## Cumulative wall-clock time axis (seconds)
t_axis_r <- seq_len(N_Q) * us_r / 1e6
t_axis_i <- seq_len(N_Q) * us_i / 1e6

cat(sprintf("  Cramér-Rao RMSE floor: %.4f\n", cr_floor))
cat(sprintf("  Final RMSE  — RLS-GLM: %.4f   iSGD: %.4f\n",
            rmse_r[N_Q], rmse_i[N_Q]))
cat(sprintf("  Per-step    — RLS-GLM: %.1f µs   iSGD: %.1f µs   (%.1fx)\n",
            us_r, us_i, us_r / us_i))
cat(sprintf("  Time for %d steps — RLS-GLM: %.2f s   iSGD: %.2f s\n",
            N_Q, t_axis_r[N_Q], t_axis_i[N_Q]))

## ---------------------------------------------------------------------------
## 3. Plots (2x2 layout)
## ---------------------------------------------------------------------------
op <- par(mfrow = c(2, 2), mar = c(4.5, 4.8, 3.8, 1.5),
          oma = c(0, 0, 2.5, 0))

## --- Panel 1: per-step time vs p (log-log) ---------------------------------
ylim1 <- range(c(res$rls_us, res$isgd_us)) * c(0.7, 1.5)
plot(res$p, res$rls_us, log = "xy", type = "b", pch = 16, col = "steelblue",
     xlab = "p (number of parameters)",
     ylab = expression("Time per step (" * mu * "s)"),
     main = "Update cost vs p  [log-log]",
     ylim = ylim1)
lines(res$p, res$isgd_us, type = "b", pch = 16, col = "tomato")
idx0  <- which(res$p == 100L)
p_ref <- res$p[idx0]
lines(res$p, res$rls_us[idx0]  * (res$p / p_ref)^2, lty = 2, col = "steelblue")
lines(res$p, res$isgd_us[idx0] * (res$p / p_ref),   lty = 2, col = "tomato")
abline(v = 100, lty = 3, col = "grey50")
mtext("crossover\n~p=100", side = 3, at = 100, cex = 0.65, col = "grey40",
      line = -1.2)
legend("topleft",
       legend = c("RLS-GLM (observed)", "iSGD (observed)",
                  expression(O(p^2)), expression(O(p))),
       col = c("steelblue", "tomato", "steelblue", "tomato"),
       lty = c(1, 1, 2, 2), pch = c(16, 16, NA, NA),
       bty = "n", cex = 0.78)

## --- Panel 2: observed speed-up ratio vs p ---------------------------------
plot(res$p, res$time_ratio, type = "b", pch = 16, col = "darkgreen",
     xlab = "p (number of parameters)",
     ylab = "RLS-GLM time / iSGD time",
     main = "Speed-up of iSGD over RLS-GLM",
     ylim = c(0, max(res$time_ratio) * 1.1))
abline(h = 1, lty = 2, col = "grey50")
text(max(res$p) * 0.82, 1.5, "break-even", cex = 0.72, col = "grey40")
legend("topleft", legend = "Observed speed-up",
       col = "darkgreen", lty = 1, pch = 16, bty = "n", cex = 0.85)

## --- Panel 3: RMSE vs number of observations --------------------------------
## Shows second-order vs first-order convergence rate.
## RLS-GLM reaches the Cramér-Rao floor in ~p steps; iSGD takes many more.
ylim_q <- range(c(rmse_r[rmse_r > 0], rmse_i[rmse_i > 0], cr_floor))
plot(seq_len(N_Q), rmse_r, type = "l", col = "steelblue", log = "y",
     xlab = "Observations processed",
     ylab = "RMSE to true beta",
     main = sprintf("Convergence rate  (p = %d)", p_q),
     ylim = ylim_q)
lines(seq_len(N_Q), rmse_i, col = "tomato")
abline(h = cr_floor, lty = 2, col = "grey40")
mtext(sprintf("C-R floor %.3f", cr_floor), side = 4, at = cr_floor,
      cex = 0.6, col = "grey40", las = 1)
## Mark ~p and ~5p for orientation
abline(v = p_q,     lty = 3, col = "grey60"); mtext("p",   side=3, at=p_q,     cex=0.65, col="grey40", line=-1.2)
abline(v = 5 * p_q, lty = 3, col = "grey60"); mtext("5p",  side=3, at=5*p_q,   cex=0.65, col="grey40", line=-1.2)
legend("topright",
       legend = c(sprintf("RLS-GLM  (final RMSE %.3f)", rmse_r[N_Q]),
                  sprintf("iSGD     (final RMSE %.3f)", rmse_i[N_Q])),
       col = c("steelblue", "tomato"), lty = 1, bty = "n", cex = 0.82)

## --- Panel 4: RMSE vs cumulative wall-clock time ---------------------------
## Shows the actual computational trade-off: iSGD processes more observations
## per second, partially compensating for its slower convergence rate.
## The x-axis uses clean per-step times from the timing table (not timed
## inside the quality loop) so RMSE computation does not contaminate timing.
plot(t_axis_r, rmse_r, type = "l", col = "steelblue", log = "y",
     xlab = "Cumulative wall-clock time (s)",
     ylab = "RMSE to true beta",
     main = sprintf("Accuracy vs wall-clock time  (p = %d)", p_q),
     ylim = ylim_q)
lines(t_axis_i, rmse_i, col = "tomato")
abline(h = cr_floor, lty = 2, col = "grey40")
legend("topright",
       legend = c(sprintf("RLS-GLM  (%.2f s for %d steps)", t_axis_r[N_Q], N_Q),
                  sprintf("iSGD     (%.2f s for %d steps)", t_axis_i[N_Q], N_Q)),
       col = c("steelblue", "tomato"), lty = 1, bty = "n", cex = 0.82)

mtext(
  sprintf(paste0("streamFit benchmark  —  binomial, S0=hp_scale*I, %d timing reps/cell;",
                 "  iSGD per-step advantage grows beyond p ~ 100"), N_REPS),
  outer = TRUE, cex = 0.78
)

par(op)
