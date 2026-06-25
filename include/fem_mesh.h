#ifndef FEM_MESH_H
#define FEM_MESH_H

#include <stdint.h>
#include <stdlib.h>

/* ============================================================
 * CSG Solid — union of signed axis-aligned rectangular prisms.
 * is_inside(p) = true if p is in at least one +prism and in
 * no -prism (CSG subtraction).
 * ============================================================ */

typedef struct {
    float min[3];  /* axis-aligned bounding box */
    float max[3];
    int   sign;    /* +1 = additive material, -1 = subtractive void */
} RectPrism;

typedef struct {
    RectPrism *prisms;
    int        n_prisms;
    int        cap_prisms;
} CsgSolid;

CsgSolid* csg_solid_create(void);
void      csg_solid_free(CsgSolid *s);
void      csg_solid_add_prism(CsgSolid *s, float mn[3], float mx[3], int sign);
int       csg_is_inside(const CsgSolid *s, float p[3]);

/* ============================================================
 * FemMesh — all CPU-side data needed to drive the GPU sim.
 * ============================================================ */
typedef struct {
    /* ---- nodes ---- */
    int     n_nodes;
    float  *pos;       /* [n_nodes * 3] initial positions */
    float  *mass;      /* [n_nodes]     lumped mass */

    /* ---- tetrahedra ---- */
    int     m_tets;
    /* Structure matrices (forward): ci_k[j] = node index of corner k in tet j */
    unsigned int *ci[4];    /* [m_tets] each — 4 selection-matrix index arrays */

    float  *D0_inv;    /* [m_tets * 9] per-tet rest inverse (column-major mat3) */
    float  *V0;        /* [m_tets]     per-tet rest volume (> 0) */

    /* ---- backward adjacency (for SpMV scatter, no coloring needed) ----
     * adj_data[t] = (corner_k << 28) | tet_j
     * For node i: adj_data[adj_start[i] .. adj_start[i+1]) lists all
     * (corner_k, tet_j) pairs where node i is corner k of tet j.
     * One thread per node, no write conflicts.                          */
    int           total_adj;
    unsigned int *adj_start;  /* [n_nodes + 1] CSR row starts */
    unsigned int *adj_data;   /* [total_adj]   packed (k<<28)|j entries */

    /* ---- exterior surface ---- */
    int           n_surf_tri;
    unsigned int *surf_tri;     /* [n_surf_tri * 3] node indices per triangle  */
    unsigned int *surf_tri_tet; /* [n_surf_tri]     parent tet index per tri   */

    /* ---- interior tet-shared faces (exposed when a tet fractures) ---- */
    int           n_int_tri;
    unsigned int *int_tri;          /* [n_int_tri * 3] node indices per face       */
    unsigned int *int_tri_tets;     /* [n_int_tri * 2] (ta, tb) tet pair per face  */
    unsigned int *int_tri_tet4th;   /* [n_int_tri * 2] 4th corner of ta / tb       */

    /* ---- CSG reference (kept for edit mode) ---- */
    CsgSolid     *solid;      /* owned; freed by fem_mesh_free */
    float         lattice_h;  /* lattice spacing used to generate this mesh */
} FemMesh;

/* Generate the full FEM mesh from a CSG solid at lattice spacing h.
 * density: material density (kg/m³) for lumped mass computation.
 * Returns newly allocated FemMesh (caller owns; free with fem_mesh_free).  */
FemMesh* fem_mesh_generate(CsgSolid *solid, float h, float density);

/* Free all resources owned by the mesh (including the embedded CsgSolid). */
void     fem_mesh_free(FemMesh *m);

/* Convenience: ray–triangle intersection (Möller–Trumbore).
 * Returns t >= 0 if hit, -1 if miss.  out_u/out_v are barycentric coords. */
float fem_ray_tri_intersect(const float *ray_orig, const float *ray_dir,
                             const float *va, const float *vb, const float *vc,
                             float *out_u, float *out_v);

#endif /* FEM_MESH_H */
