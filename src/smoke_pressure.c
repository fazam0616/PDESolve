#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../include/smoke_pressure.h"
#include "../include/literal.h"

// Helper to access scalar value at indices without allocating
static double lit_get_val(const Literal *lit, const uint32_t *indices) {
    return literal_get(lit, (uint32_t*)indices);
}

// RB Gauss-Seidel smoother: in-place update of `p` solving A p = -rhs where A = -L
static int rbgs_smooth(GridField *p, const GridField *rhs, int n_iters, double omega) {
    if (!p || !rhs) return 1;
    GridMetadata *grid = p->grid;
    uint32_t nx = grid->dims[0];
    uint32_t ny = (grid->n_dims > 1) ? grid->dims[1] : 1;
    double dx = grid->spacing[0];
    double dy = (grid->n_dims > 1) ? grid->spacing[1] : 1.0;
    double idx2 = 1.0 / (dx * dx);
    double idy2 = 1.0 / (dy * dy);
    double denom = 2.0 * (idx2 + idy2);
    uint32_t idxv[3];
    for (int sweep = 0; sweep < n_iters; ++sweep) {
        for (int color = 0; color < 2; ++color) {
            for (uint32_t j = 0; j < ny; ++j) {
                for (uint32_t i = 0; i < nx; ++i) {
                    if (((i + j) & 1) != color) continue;
                    idxv[0] = i; idxv[1] = j; idxv[2] = 0;
                    if (grid_is_boundary(grid, idxv)) continue;
                    uint32_t ip[3] = { i+1, j, 0 };
                    uint32_t im[3] = { i-1, j, 0 };
                    uint32_t jp[3] = { i, j+1, 0 };
                    uint32_t jm[3] = { i, j-1, 0 };
                    double b = literal_get(&rhs->data, idxv); /* b = rhs for A p = b */
                    double sum = idx2 * (literal_get(&p->data, ip) + literal_get(&p->data, im))
                               + idy2 * (literal_get(&p->data, jp) + literal_get(&p->data, jm));
                    double p_new = (b + sum) / denom;
                    double p_old = literal_get(&p->data, idxv);
                    double p_upd = p_old + omega * (p_new - p_old);
                    literal_set(&p->data, idxv, p_upd);
                }
            }
        }
    }
    return 0;
}

int smoke_solve_poisson(const GridField *rhs, GridField *p, double tol, int max_iter, int *out_iters) {
    if (!rhs || !p) return 1;
    GridMetadata *grid = rhs->grid;
    if (grid != p->grid) return 1;

    uint32_t nx = grid->dims[0];
    uint32_t ny = (grid->n_dims > 1) ? grid->dims[1] : 1;
    double dx = grid->spacing[0];
    double dy = (grid->n_dims > 1) ? grid->spacing[1] : 1.0;
    double idx2 = 1.0 / (dx * dx);
    double idy2 = 1.0 / (dy * dy);
    double denom = 2.0 * (idx2 + idy2);
    int debug_every = (max_iter > 0) ? ((max_iter / 10) > 0 ? (max_iter / 10) : 1) : 1000;
    if (debug_every < 1) debug_every = 1;

    /* Fixed configuration: use RBGS smoother, disable env-based overrides and restarts */
    fprintf(stderr, "[poisson] Grid: nx=%u ny=%u dx=%g dy=%g\n", nx, ny, dx, dy);
    fprintf(stderr, "[poisson] Coefficients: idx2=%g idy2=%g denom=%g\n", idx2, idy2, denom);
    int cg_debug = 0; /* disabled runtime debug via env */

    const char *precond_type = "rbgs"; /* force RBGS preconditioner/smoother */
    int restart_k = 0; /* disable automatic restart recomputations */
    int pre_smooth = 2; /* small fixed amount of pre-smoothing */
    double omega = 0.8; /* low relaxation (not zero) */

    /* Always initialize p to zero (no warm start) */
    {
        Literal zero = { {nx, ny, 1}, NULL };
        grid_field_fill(p, &zero);
    }

    // Diagnostic: if grid is 2D and small, compute manufactured p_true and
    // compare manual Laplacian vs grid_field_laplacian to detect mismatches.
    if (grid->n_dims > 1 && nx <= 128 && ny <= 128) {
        GridField *p_true = grid_field_create(grid);
        // fill p_true with sin(pi x) sin(pi y)
        uint32_t indices[N_DIM]; double coords[N_DIM];
        for (uint32_t linear = 0; linear < grid->total_points; linear++) {
            grid_linear_to_index(grid, linear, indices);
            grid_index_to_coord(grid, indices, coords);
            double x = coords[0]; double y = coords[1];
            double val = sin(M_PI * x) * sin(M_PI * y);
            literal_set(&p_true->data, indices, val);
        }

        GridField *lap_helper = grid_field_laplacian_compact(p_true);

        // manual laplacian via neighbor accesses
        /* Compute manual Laplacian using grid shift helpers to avoid per-point loops.
           lap_manual = (shift_x_plus + shift_x_minus - 2*p_true)/dx^2
                      + (shift_y_plus + shift_y_minus - 2*p_true)/dy^2
        */
        GridField *lap_manual = NULL;
        double dx2 = dx*dx; double dy2 = dy*dy;
        /* create neighbor-shifted fields */
        GridField *fxp = grid_field_shift(p_true, 0, +1);
        GridField *fxm = grid_field_shift(p_true, 0, -1);
        GridField *fyp = grid_field_shift(p_true, 1, +1);
        GridField *fym = grid_field_shift(p_true, 1, -1);

        if (!fxp || !fxm || !fyp || !fym) {
            /* fallback to zeroed manual field on allocation failure */
            lap_manual = grid_field_create(grid);
            if (lap_manual) {
                Literal zero = { {nx, ny, 1}, NULL };
                grid_field_fill(lap_manual, &zero);
            }
        } else {
            GridField *sumx = grid_field_add(fxp, fxm);           // fxp + fxm
            GridField *sumy = grid_field_add(fyp, fym);           // fyp + fym
            GridField *center2 = grid_field_scale(p_true, -2.0);  // -2 * p_true

            GridField *sumx2 = grid_field_add(sumx, center2);    // fxp + fxm - 2*p
            GridField *sumy2 = grid_field_add(sumy, center2);    // fyp + fym - 2*p

            GridField *sumx2_s = grid_field_scale(sumx2, 1.0 / dx2);
            GridField *sumy2_s = grid_field_scale(sumy2, 1.0 / dy2);

            lap_manual = grid_field_add(sumx2_s, sumy2_s);

            /* free intermediates */
            grid_field_free(sumx); grid_field_free(sumy);
            grid_field_free(center2);
            grid_field_free(sumx2); grid_field_free(sumy2);
            grid_field_free(sumx2_s); grid_field_free(sumy2_s);
        }

        grid_field_free(fxp); grid_field_free(fxm); grid_field_free(fyp); grid_field_free(fym);

        // compute statistics of difference
        double max_abs = 0.0; double max_rel = 0.0; double max_lap = 0.0;
        for (uint32_t linear = 0; linear < grid->total_points; linear++) {
            grid_linear_to_index(grid, linear, indices);
            double a = literal_get(&lap_helper->data, indices);
            double b = literal_get(&lap_manual->data, indices);
            double diff = fabs(a - b);
            double rel = (fabs(b) > 1e-15) ? diff / fabs(b) : diff;
            if (diff > max_abs) max_abs = diff;
            if (rel > max_rel) max_rel = rel;
            if (fabs(b) > max_lap) max_lap = fabs(b);
        }

        fprintf(stderr, "[poisson-debug] lap_manual_max_abs=%g lap_manual_max_rel=%g max_lap=%g\n",
                max_abs, max_rel, max_lap);

        grid_field_free(p_true);
        grid_field_free(lap_helper);
        grid_field_free(lap_manual);
    }

    uint32_t idx[3];

    // Conjugate Gradient solver using grid_field_laplacian as A operator
    // Solve A p = rhs, where A is the discrete Laplacian with Dirichlet BCs
    // r = rhs - A*p (p is zero initially) -> since p is zero, r = rhs
    GridField *r = grid_field_copy(rhs);
    if (!r) { fprintf(stderr, "[poisson] Failed to allocate residual\n"); return 1; }
    /* For CG we use A = -L (positive-definite); negate RHS so system is A p = -rhs */
    grid_field_scale_inplace(r, -1.0);

    /* Zero-mean subtraction disabled: keep RHS as-is for Dirichlet problems */

    /* Setup preconditioner: force RBGS-only, compute z_buf ≈ A^{-1} r via a few RBGS sweeps */
    GridField *z_buf = grid_field_create(grid);
    if (!z_buf) { grid_field_free(r); return 1; }
    /* initialize z_buf to zero */
    {
        Literal zero = { {nx, ny, 1}, NULL };
        grid_field_fill(z_buf, &zero);
        int rb_sweeps = pre_smooth > 0 ? pre_smooth : 2;
        if (rbgs_smooth(z_buf, r, rb_sweeps, omega) != 0) {
            /* non-fatal: continue without preconditioner */
            fprintf(stderr, "[poisson] Warning: RBGS preconditioner failed during setup\n");
            grid_field_free(z_buf);
            z_buf = NULL;
        }
    }

     /* Optional pre-smoothing: perform a few Richardson/Jacobi steps before CG.
         POISSON_PRE_SMOOTH and POISSON_OMEGA were read earlier. */
    if (pre_smooth > 0) {
        /* RBGS pre-smoothing: apply approximate correction to p */
        {
            /* ensure p exists (warm-start) */
            /* Use residual `r` (already negated) as the RHS for the RBGS smoother */
            /* For RBGS pre-smoothing we solve approximately for the correction e
               from A e = res where res = r - A*p (i.e., residual). Build res_tmp,
               solve for e (on a temporary field) and add e into p. Then recompute
               r and z_buf so CG starts with consistent state. */
            {
                GridField *res_tmp = grid_field_copy(r);
                if (res_tmp) {
                    GridField *lap = grid_field_laplacian(p);
                    if (lap) {
                        if (grid_field_axpy(res_tmp, 1.0, lap) != 0) {
                            fprintf(stderr, "[poisson] Warning: failed to form residual for RBGS pre-smooth\n");
                        }
                        grid_field_free(lap);
                    } else {
                        fprintf(stderr, "[poisson] Warning: failed to compute laplacian for RBGS pre-smooth\n");
                    }

                    GridField *e = grid_field_create(p->grid);
                    if (e) {
                        if (rbgs_smooth(e, res_tmp, pre_smooth, omega) == 0) {
                            if (grid_field_axpy(p, 1.0, e) != 0) {
                                fprintf(stderr, "[poisson] Warning: failed to apply RBGS correction to p\n");
                            }
                        } else {
                            fprintf(stderr, "[poisson] Warning: RBGS pre-smooth failed on correction field\n");
                        }
                        grid_field_free(e);
                    } else {
                        fprintf(stderr, "[poisson] Warning: failed to allocate RBGS correction field\n");
                    }

                    /* Recompute residual r = -rhs + L p so CG has consistent r */
                    GridField *lap2 = grid_field_laplacian(p);
                    if (lap2) {
                        if (grid_field_copy_into(rhs, r) == 0) {
                            grid_field_scale_inplace(r, -1.0);
                            if (grid_field_axpy(r, 1.0, lap2) != 0) {
                                fprintf(stderr, "[poisson] Warning: failed to update residual after RBGS pre-smooth\n");
                            }
                        } else {
                            fprintf(stderr, "[poisson] Warning: failed to copy rhs into r after RBGS pre-smooth\n");
                        }
                        grid_field_free(lap2);
                    } else {
                        fprintf(stderr, "[poisson] Warning: failed to compute laplacian after RBGS pre-smooth\n");
                    }

                    /* Update z_buf by recomputing RBGS-based approximation (if present) */
                    if (z_buf) {
                        Literal zero_lit = { {nx, ny, 1}, NULL };
                        grid_field_fill(z_buf, &zero_lit);
                        int rb_sweeps = pre_smooth > 0 ? pre_smooth : 2;
                        if (rbgs_smooth(z_buf, r, rb_sweeps, omega) != 0) {
                            fprintf(stderr, "[poisson] Warning: failed to update z_buf after RBGS pre-smooth\n");
                        }
                    }

                    grid_field_free(res_tmp);
                    fprintf(stderr, "[poisson] Performed %d RBGS pre-smooth steps (omega=%g)\n", pre_smooth, omega);
                } else {
                    fprintf(stderr, "[poisson] Warning: failed to allocate residual copy for RBGS pre-smooth\n");
                }
            }
        }
    }

    GridField *d = NULL;
    if (z_buf) {
        d = grid_field_copy(z_buf);
        if (!d) { grid_field_free(r); grid_field_free(z_buf); return 1; }
    } else {
        d = grid_field_copy(r);
        if (!d) { grid_field_free(r); return 1; }
    }

    double delta_new = 0.0;
    if (z_buf) {
        Literal *dl = literal_dot(&r->data, &z_buf->data);
        delta_new = dl ? dl->field[0] : 0.0;
        if (dl) literal_free(dl);
    } else {
        delta_new = grid_field_norm(r);
        delta_new = delta_new * delta_new;
    }
    double denom_count = (double)((nx > 2 && ny > 2) ? ((nx-2)*(ny-2)) : 1);

    if (delta_new == 0.0) {
        grid_field_free(r); grid_field_free(d);
        fprintf(stderr, "[poisson] RHS is zero, trivial solution\n");
        return 0;
    }

    // allocate reusable buffers
    GridField *q_buf = grid_field_create(grid);
    if (!q_buf) { grid_field_free(r); grid_field_free(d); if (z_buf) grid_field_free(z_buf); return 1; }

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        if (iter == 0 && cg_debug) {
            fprintf(stderr, "[poisson-debug] ptrs: p=%p r=%p d=%p q=%p z=%p\n",
                (void*)(p? p->data.field:NULL), (void*)(r? r->data.field:NULL), (void*)(d? d->data.field:NULL),
                (void*)(q_buf? q_buf->data.field:NULL), (void*)(z_buf? z_buf->data.field:NULL));
        }
        GridField *q = grid_field_laplacian(d);
        if (!q) { grid_field_free(r); grid_field_free(d); grid_field_free(q_buf); if (z_buf) grid_field_free(z_buf); return 1; }
        /* Use negative Laplacian as operator A = -∇² (make A SPD) */
        grid_field_scale_inplace(q, -1.0);

        Literal *dq_lit = literal_dot(&d->data, &q->data);
        double dq = dq_lit ? dq_lit->field[0] : 0.0;
        if (dq_lit) literal_free(dq_lit);

        /* Additional diagnostics: recompute dot in long double and print samples/addresses.
           This helps detect reduction/precision/race/aliasing issues. */
        size_t nelems = literal_total_elements(&d->data);
        long double dq_ld = 0.0L;
        if (d->data.field && q->data.field) {
            for (size_t i = 0; i < nelems; i++) dq_ld += (long double)d->data.field[i] * (long double)q->data.field[i];
        }

        if (!isfinite(dq) || dq <= 0.0) {
            fprintf(stderr, "[poisson-debug] BAD dq at iter=%d dq=%g dq_ld=%Lg\n", iter, dq, dq_ld);
            if (cg_debug) {
                double nd = grid_field_norm(d);
                double nq_q = grid_field_norm(q);
                double nq_buf = grid_field_norm(q_buf);
                fprintf(stderr, "[poisson-debug] norms: ||d||=%g ||q||=%g ||q_buf||=%g\n", nd, nq_q, nq_buf);
                Literal *dd = literal_dot(&d->data, &d->data);
                Literal *qq_q = literal_dot(&q->data, &q->data);
                Literal *qq_buf = literal_dot(&q_buf->data, &q_buf->data);
                if (dd) { fprintf(stderr, "[poisson-debug] d.d = %g\n", dd->field[0]); literal_free(dd); }
                if (qq_q) { fprintf(stderr, "[poisson-debug] q.q = %g\n", qq_q->field[0]); literal_free(qq_q); }
                if (qq_buf) { fprintf(stderr, "[poisson-debug] q_buf.q = %g\n", qq_buf->field[0]); literal_free(qq_buf); }

                size_t sample = nelems;
                size_t nprint = sample > 8 ? 8 : sample;
                fprintf(stderr, "[poisson-debug] addrs: d=%p q=%p q_buf=%p\n", (void*)(d? d->data.field:NULL), (void*)(q? q->data.field:NULL), (void*)(q_buf? q_buf->data.field:NULL));
                fprintf(stderr, "[poisson-debug] sample d[0..%zu]:", nprint);
                for (size_t si = 0; si < nprint; si++) fprintf(stderr, " %g", d->data.field[si]);
                fprintf(stderr, "\n[poisson-debug] sample q[0..%zu]:", nprint);
                for (size_t si = 0; si < nprint; si++) fprintf(stderr, " %g", q->data.field[si]);
                fprintf(stderr, "\n[poisson-debug] sample q_buf[0..%zu]:", nprint);
                for (size_t si = 0; si < nprint; si++) fprintf(stderr, " %g", q_buf->data.field[si]);
                fprintf(stderr, "\n");
            }
            grid_field_free(q_buf);
            if (z_buf) grid_field_free(z_buf);
            grid_field_free(r);
            grid_field_free(d);
            return 1;
        }
        if (cg_debug) {
            double norm_d = grid_field_norm(d);
            double norm_q = grid_field_norm(q);
            double norm_qbuf = grid_field_norm(q_buf);
            fprintf(stderr, "[poisson-debug] iter=%d dq=%g dq_ld=%Lg ||d||=%g ||q||=%g ||q_buf||=%g\n",
                    iter, dq, dq_ld, norm_d, norm_q, norm_qbuf);
        }
        if (fabs(dq) < 1e-30) {
            break;
        }

        double alpha = delta_new / dq;
        if (cg_debug) fprintf(stderr, "[poisson-debug] iter=%d alpha=%g delta_new=%g dq=%g\n", iter, alpha, delta_new, dq);

        // p = p + alpha * d  (in-place)
        if (grid_field_axpy(p, alpha, d) != 0) { grid_field_free(q); grid_field_free(q_buf); grid_field_free(r); grid_field_free(d); if (z_buf) grid_field_free(z_buf); return 1; }

        

        // r = r - alpha * q  (in-place)
        if (grid_field_axpy(r, -alpha, q) != 0) { grid_field_free(q); grid_field_free(q_buf); grid_field_free(r); grid_field_free(d); if (z_buf) grid_field_free(z_buf); return 1; }

        if (cg_debug && iter == 0) {
            double np = grid_field_norm(p);
            double nr = grid_field_norm(r);
            double nd = grid_field_norm(d);
            double nq = grid_field_norm(q);
            fprintf(stderr, "[poisson-debug] post-update norms: ||p||=%g ||r||=%g ||d||=%g ||q||=%g\n", np, nr, nd, nq);
            size_t sample = literal_total_elements(&p->data);
            size_t nprint = sample > 8 ? 8 : sample;
            fprintf(stderr, "[poisson-debug] sample p[0..%zu]:", nprint);
            for (size_t si = 0; si < nprint; si++) fprintf(stderr, " %g", p->data.field[si]);
            fprintf(stderr, "\n[poisson-debug] sample r[0..%zu]:", nprint);
            for (size_t si = 0; si < nprint; si++) fprintf(stderr, " %g", r->data.field[si]);
            fprintf(stderr, "\n");
        }

        grid_field_free(q);

        

        double norm_r = grid_field_norm(r);
        double res = (denom_count > 0) ? norm_r / sqrt(denom_count) : norm_r;

        /* Apply preconditioner into z_buf (Jacobi) or compute z_new via RBGS each iteration */
        GridField *z_new = NULL;
        /* compute z_new by solving approximately A z = r via RBGS sweeps */
        {
            Literal zero_lit = { {nx, ny, 1}, NULL };
            if (!z_buf) {
                z_buf = grid_field_create(grid);
                if (!z_buf) { grid_field_free(q_buf); grid_field_free(r); grid_field_free(d); return 1; }
            }
            grid_field_fill(z_buf, &zero_lit);
            int rb_apply_sweeps = pre_smooth > 0 ? pre_smooth : 2;
            if (rbgs_smooth(z_buf, r, rb_apply_sweeps, omega) != 0) {
                fprintf(stderr, "[poisson] Warning: RBGS preconditioner failed during iteration\n");
            }
            z_new = z_buf;
        }

        if (iter % debug_every == 0 || iter == 0) {
            fprintf(stderr, "[poisson] iter=%d res=%g\n", iter, res);
        }

        if (res < tol) {
            grid_field_free(q_buf);
            grid_field_free(r);
            grid_field_free(d);
            if (z_buf) grid_field_free(z_buf);
            fprintf(stderr, "[poisson] Converged at iter=%d res=%g\n", iter, res);
            if (out_iters) *out_iters = iter;
            return 0;
        }

        double delta_old = delta_new;
        if (z_new) {
            Literal *dl = literal_dot(&r->data, &z_new->data);
            delta_new = dl ? dl->field[0] : 0.0;
            if (dl) literal_free(dl);
        } else {
            delta_new = norm_r * norm_r;
        }

        double beta = (delta_old > 0.0) ? (delta_new / delta_old) : 0.0;
        if (cg_debug) fprintf(stderr, "[poisson-debug] iter=%d delta_old=%g delta_new=%g beta=%g\n", iter, delta_old, delta_new, beta);

        /* Restart logic: occasionally recompute residual and restart search direction */
        if (restart_k > 0 && ((iter+1) % restart_k) == 0) {
            // recompute residual r = rhs - A*p using fresh q
            GridField *q2 = grid_field_laplacian(p);
            if (!q2) { grid_field_free(q_buf); grid_field_free(r); grid_field_free(d); if (z_buf) grid_field_free(z_buf); return 1; }
            /* keep q2 as computed by grid_field_laplacian (original convention) */
                if (grid_field_copy_into(rhs, r) != 0) { grid_field_free(q2); grid_field_free(q_buf); grid_field_free(r); grid_field_free(d); if (z_buf) grid_field_free(z_buf); return 1; }
                /* keep q2 as computed by grid_field_laplacian (Lp).
                    Maintain A = -L sign convention: r = -rhs + L p */
                grid_field_scale_inplace(r, -1.0);
                if (grid_field_axpy(r, 1.0, q2) != 0) { grid_field_free(q2); grid_field_free(q_buf); grid_field_free(r); grid_field_free(d); if (z_buf) grid_field_free(z_buf); return 1; }
            grid_field_free(q2);
            if (z_new) {
                /* recompute z_buf via RBGS approximation */
                if (z_buf) {
                    Literal zero_lit = { {nx, ny, 1}, NULL };
                    grid_field_fill(z_buf, &zero_lit);
                    int rb_sweeps = pre_smooth > 0 ? pre_smooth : 2;
                    if (rbgs_smooth(z_buf, r, rb_sweeps, omega) != 0) {
                        fprintf(stderr, "[poisson] Warning: failed to update z_buf during restart\n");
                    }
                }
                z_new = z_buf;
            }
            /* reset search direction to z (or r for unprecond) */
            if (z_new) {
                if (grid_field_copy_into(z_buf, d) != 0) { grid_field_free(q_buf); grid_field_free(r); grid_field_free(d); if (z_buf) grid_field_free(z_buf); return 1; }
            } else {
                if (grid_field_copy_into(r, d) != 0) { grid_field_free(q_buf); grid_field_free(r); grid_field_free(d); if (z_buf) grid_field_free(z_buf); return 1; }
            }
            delta_old = delta_new;
            beta = 0.0;
        }

        /* d = z_new + beta * d   (or r + beta*d when unpreconditioned) */
        GridField *base = z_new ? z_new : r;
        grid_field_scale_inplace(d, beta);
        if (grid_field_axpy(d, 1.0, base) != 0) { grid_field_free(q_buf); grid_field_free(r); grid_field_free(d); if (z_buf) grid_field_free(z_buf); return 1; }

        
    }

    // cleanup on failure to converge
    grid_field_free(q_buf);
    if (z_buf) grid_field_free(z_buf);
    grid_field_free(r);
    grid_field_free(d);

    fprintf(stderr, "[poisson] Failed to converge after %d iterations\n", max_iter);
    if (out_iters) *out_iters = max_iter;
    return 1;
}

// Compute divergence, solve for pressure, subtract grad(p) from velocities
int smoke_pressure_project(GridField *vx, GridField *vy, GridField *pressure, double dt, double tol, int max_iter) {
    if (!vx || !vy || !pressure) return 1;
    GridMetadata *grid = vx->grid;
    if (grid != vy->grid || grid != pressure->grid) return 1;

    uint32_t idx[3];

    // compute divergence field: div = d vx/dx + d vy/dy
    // Vectorized using grid_field_shift + arithmetic ops to avoid per-point allocations
    GridField *div = NULL;
    double dx = grid->spacing[0];
    double dy = (grid->n_dims>1)? grid->spacing[1] : 1.0;

    // Shifted neighbors for vx along x
    GridField *vx_p = grid_field_shift(vx, 0, +1);
    GridField *vx_m = grid_field_shift(vx, 0, -1);
    GridField *dvx = grid_field_subtract(vx_p, vx_m);
    GridField *dvx_scaled = grid_field_scale(dvx, 1.0 / (2.0 * dx));
    grid_field_free(vx_p); grid_field_free(vx_m); grid_field_free(dvx);

    GridField *dvy_scaled = NULL;
    if (grid->n_dims > 1) {
        GridField *vy_p = grid_field_shift(vy, 1, +1);
        GridField *vy_m = grid_field_shift(vy, 1, -1);
        GridField *dvy = grid_field_subtract(vy_p, vy_m);
        dvy_scaled = grid_field_scale(dvy, 1.0 / (2.0 * dy));
        grid_field_free(vy_p); grid_field_free(vy_m); grid_field_free(dvy);
    }

    // Sum components
    if (dvy_scaled) {
        GridField *sum = grid_field_add(dvx_scaled, dvy_scaled);
        grid_field_free(dvx_scaled); grid_field_free(dvy_scaled);
        div = grid_field_scale(sum, 1.0 / dt);
        grid_field_free(sum);
    } else {
        div = grid_field_scale(dvx_scaled, 1.0 / dt);
        grid_field_free(dvx_scaled);
    }

    // Solve Poisson: lap p = div/dt
    int rc = smoke_solve_poisson(div, pressure, tol, max_iter, NULL);
    if (rc != 0) {
        grid_field_free(div);
        return rc;
    }

    // Subtract dt * grad p from velocity
    for (uint32_t linear = 0; linear < grid->total_points; linear++) {
        grid_linear_to_index(grid, linear, idx);
        uint32_t ip[3] = { (idx[0]+1 < grid->dims[0])? idx[0]+1:idx[0], idx[1], 0 };
        uint32_t im[3] = { (idx[0]>0)? idx[0]-1:idx[0], idx[1], 0 };
        uint32_t jp[3] = { idx[0], (idx[1]+1 < grid->dims[1])? idx[1]+1:idx[1], 0 };
        uint32_t jm[3] = { idx[0], (idx[1]>0)? idx[1]-1:idx[1], 0 };
        double dpdx = 0.0, dpdy = 0.0;
        if (idx[0] == 0) dpdx = (literal_get(&pressure->data, ip) - literal_get(&pressure->data, idx)) / dx;
        else if (idx[0] == grid->dims[0]-1) dpdx = (literal_get(&pressure->data, idx) - literal_get(&pressure->data, im)) / dx;
        else dpdx = (literal_get(&pressure->data, ip) - literal_get(&pressure->data, im)) / (2.0*dx);

        if (grid->n_dims > 1) {
            if (idx[1] == 0) dpdy = (literal_get(&pressure->data, jp) - literal_get(&pressure->data, idx)) / dy;
            else if (idx[1] == grid->dims[1]-1) dpdy = (literal_get(&pressure->data, idx) - literal_get(&pressure->data, jm)) / dy;
            else dpdy = (literal_get(&pressure->data, jp) - literal_get(&pressure->data, jm)) / (2.0*dy);
        }

        double vx_new = literal_get(&vx->data, idx) - dt * dpdx;
        double vy_new = literal_get(&vy->data, idx) - dt * dpdy;
        Literal *lvx = literal_create_scalar(vx_new);
        Literal *lvy = literal_create_scalar(vy_new);
        grid_field_set(vx, idx, lvx); literal_free(lvx);
        grid_field_set(vy, idx, lvy); literal_free(lvy);
    }

    grid_field_free(div);
    return 0;
}

/* Test wrapper to expose RBGS for diagnostics (not used by main API). */
int rbgs_smooth_public(GridField *p, const GridField *rhs, int n_iters, double omega) {
    return rbgs_smooth(p, rhs, n_iters, omega);
}
