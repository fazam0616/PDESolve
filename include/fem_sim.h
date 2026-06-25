#ifndef FEM_SIM_H
#define FEM_SIM_H

#include "fem_mesh.h"
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>

/* ============================================================
 * FemSimGPU — GPU-resident state for one FEM simulation step.
 *
 * All physics buffers are float SSBOs (GL_SHADER_STORAGE_BUFFER).
 * The per-substep compute pipeline is:
 *   gather → tet_force → scatter → ext_force → integrate
 *   (+ surf_normals before each render frame)
 *
 * SpMV scatter: one thread per node sums forces from all incident
 * tetrahedra using the CSR adjacency.  No graph coloring needed;
 * no write conflicts between threads.
 * ============================================================ */

typedef struct {
    /* ---- simulation SSBOs (float) ---- */
    GLuint pos_ssbo;          /* float[n*3]   — current node positions     */
    GLuint vel_ssbo;          /* float[n*3]   — node velocities             */
    GLuint force_ssbo;        /* float[n*3]   — elastic + ext force sum     */
    GLuint mass_ssbo;         /* float[n]     — lumped mass per node        */

    /* ---- rest-state SSBOs ---- */
    GLuint D0inv_ssbo;        /* float[m*9]   — per-tet rest inverse mat3   */
    GLuint V0_ssbo;           /* float[m]     — per-tet rest volume         */

    /* ---- structure matrix SSBOs (forward gather, uint) ---- */
    GLuint ci_ssbo[4];        /* uint[m] each — corner node indices         */

    /* ---- backward adjacency SSBOs (SpMV scatter, uint) ---- */
    GLuint adj_start_ssbo;    /* uint[n+1]    — CSR row starts              */
    GLuint adj_data_ssbo;     /* uint[total]  — packed (k<<28)|j entries    */

    /* ---- anchor / external force SSBOs ---- */
    GLuint anchor_ssbo;       /* uint[n]      — 0=free, 1=anchored          */
    GLuint ext_force_ssbo;    /* float[n*3]   — user-painted node forces    */

    /* ---- intermediate per-tet SSBOs ---- */
    GLuint x_ssbo[4];         /* float[m*3]   — gathered corner positions   */
    GLuint f_ssbo[4];         /* float[m*3]   — per-corner elastic forces   */

    /* ---- plasticity per-tet SSBOs ---- */
    GLuint eps_p_ssbo;        /* float[m*6]   — Voigt plastic strain tensor  */
    GLuint acc_p_ssbo;        /* float[m]     — accumulated plastic strain   */
    GLuint broken_ssbo;       /* uint[m]      — 1=fractured (zero force)     */
    GLuint surf_node_ssbo;    /* uint[n]      — 1=node on fracture surface   */

    /* ---- surface SSBOs ---- */
    GLuint surf_tri_ssbo;     /* uint[ns*3]   — surface triangle indices    */
    GLuint surf_tri_tet_ssbo; /* uint[ns]     — parent tet index per tri    */
    GLuint surf_norm_ssbo;    /* float[ns*3]  — per-triangle normals        */

    /* ---- interior face SSBOs (fracture crack surface) ---- */
    GLuint int_tri_ssbo;       /* uint[ni*3]   — interior face node indices  */
    GLuint int_tri_tet_ssbo;   /* uint[ni*2]   — (ta,tb) tet pair per face   */
    GLuint int_tri_tet4th_ssbo;/* uint[ni*2]   — 4th corner of ta / tb       */

    /* ---- compute programs ---- */
    GLuint prog_gather;       /* gathers all 4 corners in one dispatch      */
    GLuint prog_tet_force;    /* per-tet FEM physics → fi[0..3]             */
    GLuint prog_scatter;      /* SpMV scatter → force_ssbo                  */
    GLuint prog_ext_force;    /* add ext force, gravity, enforce anchors    */
    GLuint prog_integrate;    /* symplectic Euler                           */
    GLuint prog_surf_normals; /* per-triangle normals from current pos      */
    GLuint prog_fracture_surf;/* marks corner nodes of broken tets surface  */
    GLuint prog_render_int;   /* interior fracture-face render              */

    /* ---- render programs ---- */
    GLuint prog_render;       /* vertex+fragment for surface shading        */

    /* ---- VAO for surface draw call ---- */
    GLuint surface_vao;
    GLuint int_vao;           /* for interior fracture-face draw            */

    /* ---- sizes and material ---- */
    int   n_nodes;
    int   m_tets;
    int   n_surf_tri;
    int   n_int_tri;          /* number of interior shared faces            */
    int   total_adj;
    float lame_mu;            /* μ Lamé parameter   (Pa)                   */
    float lame_lambda;        /* λ Lamé parameter   (Pa)                   */

    /* ---- simulation params (updated from UI) ---- */
    float dt;
    float gravity;            /* m/s²  (default 9.81)                      */

    /* ---- scalar visualization SSBOs ---- */
    GLuint rest_pos_ssbo;       /* float[n*3] — initial rest positions (static) */
    GLuint node_scalar_ssbo;    /* float[n]   — per-node scalar (disp or F-norm)*/
    GLuint tet_scalar_ssbo;     /* float[m]   — per-tet ||F-I||_F scratch       */

    /* ---- scalar visualization programs ---- */
    GLuint prog_disp_scalar;    /* CS:    global displacement magnitude         */
    GLuint prog_f_norm_tet;     /* CS:    per-tet ||F-I||_F                     */
    GLuint prog_f_norm_node;    /* CS:    scatter-average tet→node              */
    GLuint prog_vol_strain_tet; /* CS:    per-tet signed volumetric strain tr(E) */
    GLuint prog_slice;          /* VS+FS: slice plane point cloud rendering     */
    GLuint slice_vao;

    /* ---- visualization parameters ---- */
    int   color_mode;           /* 0=solid color, 1=colormap by scalar         */
    int   show_slice;           /* 0=off, 1=show slice plane                   */
    int   scalar_mode;          /* 0=global displacement, 1=local F-norm       */
    float slice_pt[3];          /* a world-space point on the slice plane       */
    float slice_nrm[3];         /* unit normal to the slice plane               */
    float slice_h;              /* half-thickness of the visible slab (m)       */
    float scalar_max;           /* colormap normalization maximum               */

    /* ---- plasticity material params (updated from UI) ---- */
    float yield_stress;         /* Pa — von Mises yield threshold (steel ~250e6)*/
    float hardening_mod;        /* Pa — isotropic hardening modulus H (~1e9)    */
    float fracture_strain;      /* acc. plastic strain at tet fracture (~0.3)   */
    int   plasticity_enabled;   /* 0=elastic-only, 1=elastic-plastic            */

    /* ---- uniform locations cache ---- */
    GLint uloc_gather_m;
    GLint uloc_tet_mu, uloc_tet_lambda, uloc_tet_m;
    GLint uloc_scatter_n;
    GLint uloc_ext_gravity, uloc_ext_n;
    GLint uloc_int_dt, uloc_int_n;
    GLint uloc_sn_n;
    GLint uloc_render_MVP, uloc_render_MV;
    GLint uloc_render_light;
    GLint uloc_render_color;
    GLint uloc_render_color_mode;
    GLint uloc_render_scalar_max;
    GLint uloc_disp_n;
    GLint uloc_fnorm_tet_m;
    GLint uloc_fnorm_node_n;
    GLint uloc_vol_strain_tet_m;
    GLint uloc_render_scalar_signed;
    GLint uloc_slice_MVP;
    GLint uloc_slice_pt;
    GLint uloc_slice_nrm;
    GLint uloc_slice_h;
    GLint uloc_slice_scalar_max;
    GLint uloc_slice_scalar_signed;

    /* ---- plasticity uniform locations ---- */
    GLint uloc_tet_yield_stress;
    GLint uloc_tet_hardening_mod;
    GLint uloc_tet_fracture_strain;
    GLint uloc_tet_plasticity;
    GLint uloc_fsurf_m;           /* fracture_surf: uniform uint m */

    /* ---- interior fracture-face render uniform locations ---- */
    GLint uloc_int_MVP, uloc_int_MV, uloc_int_light, uloc_int_color;
    GLint uloc_int_color_mode, uloc_int_scalar_max, uloc_int_scalar_signed;
} FemSimGPU;

/* Allocate GPU buffers from mesh data and compile all compute + render shaders.
 * Requires an active GL 4.3 context.
 * Returns NULL on failure.                                                  */
FemSimGPU* fem_sim_create(const FemMesh *mesh, float lame_mu, float lame_lambda);

/* Free all GPU resources.  GL context must still be current.               */
void fem_sim_free(FemSimGPU *sim);

/* Upload current anchor/ext_force arrays to GPU.
 * anchor[n]: 0=free, 1=anchored.   ext_force[n*3]: force per node (N).    */
void fem_sim_upload_constraints(FemSimGPU *sim,
                                const unsigned int *anchor,
                                const float        *ext_force);

/* Reset all node velocities to zero and positions to rest state.
 * rest_pos: initial positions from FemMesh (may be NULL to keep current).  */
void fem_sim_reset(FemSimGPU *sim, const float *rest_pos);

/* Execute one simulation substep (gather→force→scatter→ext→integrate).    */
void fem_sim_substep(FemSimGPU *sim);

/* Compute per-triangle normals from current pos_ssbo.  Call once per frame
 * before fem_sim_render().                                                  */
void fem_sim_update_normals(FemSimGPU *sim);

/* Draw the deformed surface with Blinn-Phong shading.
 * MVP: column-major 4x4; MV: column-major 4x4; light_dir: unit vec3.       */
void fem_sim_render(FemSimGPU *sim,
                    const float *MVP,
                    const float *MV,
                    const float *light_dir,
                    const float *material_color);

/* Compute per-node displacement magnitude and store in node_scalar_ssbo.
 * Call once per frame after fem_sim_update_normals().                       */
void fem_sim_update_scalar(FemSimGPU *sim);

/* Render nodes near the slice plane as coloured GL_POINTS.
 * Uses sim->slice_pt, slice_nrm, slice_h, scalar_max.
 * MVP: column-major 4x4.                                                    */
void fem_sim_render_slice(FemSimGPU *sim, const float *MVP);

#endif /* FEM_SIM_H */
