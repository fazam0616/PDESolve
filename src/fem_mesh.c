/*
 * fem_mesh.c — CPU-side FEM mesh generation.
 *
 * Steps performed by fem_mesh_generate():
 *  1. Enumerate lattice nodes inside the CSG solid.
 *  2. Build a 6-tet (Freudenthal) subdivision of each lattice cube cell.
 *  3. Compute rest-state quantities (D0_inv, V0) per tet.
 *  4. Build structure-matrix index arrays ci[4] (forward gather).
 *  5. Build backward SpMV adjacency (adj_start / adj_data).
 *  6. Extract exterior surface triangles.
 *  7. Compute lumped node masses.
 */

#include "../include/fem_mesh.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

/* ===========================================================
 * Small vec3 helpers (float)
 * =========================================================== */
static inline void v3_sub(float *r, const float *a, const float *b) {
    r[0]=a[0]-b[0]; r[1]=a[1]-b[1]; r[2]=a[2]-b[2];
}
static inline void v3_cross(float *r, const float *a, const float *b) {
    r[0]=a[1]*b[2]-a[2]*b[1];
    r[1]=a[2]*b[0]-a[0]*b[2];
    r[2]=a[0]*b[1]-a[1]*b[0];
}
static inline float v3_dot(const float *a, const float *b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
static inline float v3_len(const float *a) {
    return sqrtf(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
}

/* ===========================================================
 * CsgSolid
 * =========================================================== */
CsgSolid* csg_solid_create(void) {
    CsgSolid *s = calloc(1, sizeof(CsgSolid));
    s->cap_prisms = 8;
    s->prisms = malloc(s->cap_prisms * sizeof(RectPrism));
    return s;
}
void csg_solid_free(CsgSolid *s) {
    if (!s) return;
    free(s->prisms);
    free(s);
}
void csg_solid_add_prism(CsgSolid *s, float mn[3], float mx[3], int sign) {
    if (s->n_prisms == s->cap_prisms) {
        s->cap_prisms *= 2;
        s->prisms = realloc(s->prisms, s->cap_prisms * sizeof(RectPrism));
    }
    RectPrism *p = &s->prisms[s->n_prisms++];
    memcpy(p->min, mn, 12); memcpy(p->max, mx, 12);
    p->sign = sign;
}
int csg_is_inside(const CsgSolid *s, float p[3]) {
    int in_positive = 0;
    for (int k = 0; k < s->n_prisms; k++) {
        const RectPrism *r = &s->prisms[k];
        int in = (p[0]>=r->min[0]&&p[0]<=r->max[0] &&
                  p[1]>=r->min[1]&&p[1]<=r->max[1] &&
                  p[2]>=r->min[2]&&p[2]<=r->max[2]);
        if (in) {
            if (r->sign < 0) return 0;   /* subtracted out */
            in_positive = 1;
        }
    }
    return in_positive;
}

/* ===========================================================
 * Open-addressing hash map: int3 key → node index
 * =========================================================== */
#define NODE_HASH_EMPTY   (-1)
#define NODE_HASH_DELETED (-2)
typedef struct { int ix, iy, iz, node_idx; } NodeEntry;

typedef struct {
    NodeEntry *table;
    int cap; /* must be power of 2 */
    int size;
} NodeHashMap;

static void nhm_init(NodeHashMap *m, int cap) {
    m->cap = cap; m->size = 0;
    m->table = malloc(cap * sizeof(NodeEntry));
    for(int i=0;i<cap;i++) m->table[i].ix = NODE_HASH_EMPTY;
}
static void nhm_free(NodeHashMap *m) { free(m->table); }

static unsigned int nhm_hash(int ix, int iy, int iz) {
    unsigned int h = (unsigned int)(ix*73856093 ^ iy*19349663 ^ iz*83492791);
    return h;
}
static void nhm_insert(NodeHashMap *m, int ix, int iy, int iz, int nidx) {
    if (m->size * 2 > m->cap) {
        /* grow */
        int old_cap = m->cap;
        NodeEntry *old = m->table;
        m->cap *= 2; m->size = 0;
        m->table = malloc(m->cap * sizeof(NodeEntry));
        for(int i=0;i<m->cap;i++) m->table[i].ix = NODE_HASH_EMPTY;
        for(int i=0;i<old_cap;i++)
            if(old[i].ix != NODE_HASH_EMPTY && old[i].ix != NODE_HASH_DELETED)
                nhm_insert(m, old[i].ix, old[i].iy, old[i].iz, old[i].node_idx);
        free(old);
    }
    unsigned int h = nhm_hash(ix,iy,iz) & (unsigned int)(m->cap-1);
    while(m->table[h].ix != NODE_HASH_EMPTY && m->table[h].ix != NODE_HASH_DELETED) {
        if(m->table[h].ix==ix && m->table[h].iy==iy && m->table[h].iz==iz) return;
        h = (h+1) & (unsigned int)(m->cap-1);
    }
    m->table[h].ix=ix; m->table[h].iy=iy; m->table[h].iz=iz; m->table[h].node_idx=nidx;
    m->size++;
}
static int nhm_lookup(const NodeHashMap *m, int ix, int iy, int iz) {
    unsigned int h = nhm_hash(ix,iy,iz) & (unsigned int)(m->cap-1);
    for(;;) {
        if(m->table[h].ix == NODE_HASH_EMPTY) return -1;
        if(m->table[h].ix==ix && m->table[h].iy==iy && m->table[h].iz==iz)
            return (m->table[h].ix==NODE_HASH_EMPTY||m->table[h].ix==NODE_HASH_DELETED)
                   ? -1 : m->table[h].node_idx;
        h = (h+1) & (unsigned int)(m->cap-1);
    }
}

/* ===========================================================
 * Open-addressing hash map: sorted uint3 → face data
 * =========================================================== */
typedef struct {
    unsigned int a, b, c;   /* sorted node triple (a<b<c) */
    int          tet_count;
    unsigned int orig[3];   /* original (unsorted) corners of 1st tet seeing this face */
    unsigned int fourth;    /* 4th corner of that tet (for normal orientation) */
    unsigned int tet_idx;   /* index of the first tet owning this face */
    unsigned int tet_idx2;  /* index of the 2nd tet (interior faces only) */
    unsigned int fourth2;   /* 4th corner of the 2nd tet */
} FaceEntry;

typedef struct {
    FaceEntry *table;
    int cap, size;
} FaceHashMap;

static void fhm_init(FaceHashMap *m, int cap) {
    m->cap=cap; m->size=0;
    m->table = calloc(cap, sizeof(FaceEntry));
    for(int i=0;i<cap;i++) m->table[i].tet_count = -1; /* -1 = empty */
}
static void fhm_free(FaceHashMap *m) { free(m->table); }
static unsigned int fhm_hash(unsigned int a, unsigned int b, unsigned int c) {
    return a*1000003u ^ b*1300033u ^ c*1700003u;
}

static void fhm_add(FaceHashMap *m, unsigned int pa, unsigned int pb, unsigned int pc,
                    unsigned int oa, unsigned int ob, unsigned int oc,
                    unsigned int fourth_corner, unsigned int tet_j)
{
    if (m->size * 2 > m->cap) {
        /* grow and rehash */
        int old_cap = m->cap;
        FaceEntry *old = m->table;
        m->cap *= 2; m->size = 0;
        m->table = calloc(m->cap, sizeof(FaceEntry));
        for(int i=0;i<m->cap;i++) m->table[i].tet_count = -1;
        for(int i=0;i<old_cap;i++) {
            if(old[i].tet_count < 0) continue;
            /* re-insert (we don't need orig/fourth after initial build so just bump count) */
            unsigned int h = fhm_hash(old[i].a,old[i].b,old[i].c) & (unsigned int)(m->cap-1);
            while(m->table[h].tet_count >= 0) h=(h+1)&(unsigned int)(m->cap-1);
            m->table[h] = old[i]; m->size++;
        }
        free(old);
    }

    /* sort pa,pb,pc so a<b<c */
    unsigned int sa=pa,sb=pb,sc=pc;
    if(sa>sb){unsigned int t=sa;sa=sb;sb=t;}
    if(sb>sc){unsigned int t=sb;sb=sc;sc=t;}
    if(sa>sb){unsigned int t=sa;sa=sb;sb=t;}

    unsigned int h = fhm_hash(sa,sb,sc) & (unsigned int)(m->cap-1);
    while(m->table[h].tet_count >= 0) {
        if(m->table[h].a==sa && m->table[h].b==sb && m->table[h].c==sc) {
            if(m->table[h].tet_count == 1) {
                m->table[h].tet_idx2 = tet_j;
                m->table[h].fourth2  = fourth_corner;
            }
            m->table[h].tet_count++;
            return;
        }
        h=(h+1)&(unsigned int)(m->cap-1);
    }
    m->table[h].a=sa; m->table[h].b=sb; m->table[h].c=sc;
    m->table[h].tet_count = 1;
    m->table[h].orig[0]=oa; m->table[h].orig[1]=ob; m->table[h].orig[2]=oc;
    m->table[h].fourth = fourth_corner;
    m->table[h].tet_idx = tet_j;
    m->size++;
}

/* ===========================================================
 * 3x3 matrix inverse via Cramer's rule (float, column-major).
 * Columns: c0=[0..2], c1=[3..5], c2=[6..8].
 * Returns det; writes inverse into inv[9].
 * =========================================================== */
static float mat3_inv_cramer(const float *m, float *inv) {
    /* m is column-major: m[0,1,2]=col0, m[3,4,5]=col1, m[6,7,8]=col2 */
    float a00=m[0],a10=m[1],a20=m[2];
    float a01=m[3],a11=m[4],a21=m[5];
    float a02=m[6],a12=m[7],a22=m[8];

    float c00 =  (a11*a22 - a21*a12);
    float c10 = -(a10*a22 - a20*a12);
    float c20 =  (a10*a21 - a20*a11);
    float c01 = -(a01*a22 - a21*a02);
    float c11 =  (a00*a22 - a20*a02);
    float c21 = -(a00*a21 - a20*a01);
    float c02 =  (a01*a12 - a11*a02);
    float c12 = -(a00*a12 - a10*a02);
    float c22 =  (a00*a11 - a10*a01);

    float det = a00*c00 + a01*c10 + a02*c20;
    if (fabsf(det) < 1e-30f) det = 1e-30f; /* degenerate protection */
    float inv_det = 1.0f / det;

    /* column-major inverse = cofactor transposed / det */
    inv[0]=c00*inv_det; inv[1]=c10*inv_det; inv[2]=c20*inv_det;
    inv[3]=c01*inv_det; inv[4]=c11*inv_det; inv[5]=c21*inv_det;
    inv[6]=c02*inv_det; inv[7]=c12*inv_det; inv[8]=c22*inv_det;
    return det;
}

/* ===========================================================
 * Freudenthal / Kuhn 6-tet subdivision of a unit cube.
 *
 * Local vertex index = dx + 2*dy + 4*dz, (dx,dy,dz) ∈ {0,1}³.
 * v0=(0,0,0)=0, v1=(1,0,0)=1, v2=(0,1,0)=2, v3=(1,1,0)=3,
 * v4=(0,0,1)=4, v5=(1,0,1)=5, v6=(0,1,1)=6, v7=(1,1,1)=7.
 *
 * All 6 tets share the main diagonal of the cube (v0–v7).
 * Orientation verified: det(D0) > 0 for each tet.
 * =========================================================== */
static const int CUBE_TETS[6][4] = {
    {0, 1, 3, 7},  /* σ=(x,y,z) even  → det=1 ✓ */
    {0, 5, 1, 7},  /* σ=(x,z,y) odd, swapped  → det=1 ✓ */
    {0, 3, 2, 7},  /* σ=(y,x,z) odd, swapped  → det=1 ✓ */
    {0, 2, 6, 7},  /* σ=(y,z,x) even  → det=1 ✓ */
    {0, 4, 5, 7},  /* σ=(z,x,y) even  → det=1 ✓ */
    {0, 6, 4, 7},  /* σ=(z,y,x) odd, swapped  → det=1 ✓ */
};

/* The 4 faces of a tet, each described as 3 local corner indices
 * (the 4th corner is the one opposite the face).                 */
static const int TET_FACES[4][3] = {
    {1,2,3}, /* face opposite corner 0 */
    {0,3,2}, /* face opposite corner 1 */
    {0,1,3}, /* face opposite corner 2 */
    {0,2,1}, /* face opposite corner 3 */
};

/* ===========================================================
 * Dynamic arrays for building the mesh
 * =========================================================== */
#define DA_PUSH(arr,cnt,cap,val) do { \
    if ((cnt)==(cap)) { (cap)*=2; (arr)=realloc((arr),(size_t)(cap)*sizeof(*(arr))); } \
    (arr)[(cnt)++]=(val); } while(0)

/* ===========================================================
 * Terminal progress bar (overwrites the current line via \r).
 * Call with cur==total to finalize and print a summary line.
 * =========================================================== */
static void print_progress(const char *label, int cur, int total, const char *suffix) {
    int pct    = (total > 0) ? (int)(100LL * cur / total) : 100;
    int filled = pct / 5; /* 20-char wide bar */
    fprintf(stdout, "\r%-14s [", label);
    for (int i = 0; i < 20; i++) fputc(i < filled ? '#' : ' ', stdout);
    fprintf(stdout, "] %3d%%", pct);
    if (suffix) fprintf(stdout, "  %s", suffix);
    /* Pad to fixed width so shorter lines don't leave stale chars */
    fprintf(stdout, "          "); /* trailing blanks */
    fflush(stdout);
    if (cur == total) fputc('\n', stdout);
}

/* ===========================================================
 * Helper: qsort comparator for (adj_node, corner, tet_j) triples
 * =========================================================== */
typedef struct { unsigned int node_i; unsigned int packed; } AdjEntry;
static int adj_cmp(const void *a, const void *b) {
    const AdjEntry *ea = (const AdjEntry*)a;
    const AdjEntry *eb = (const AdjEntry*)b;
    if (ea->node_i < eb->node_i) return -1;
    if (ea->node_i > eb->node_i) return  1;
    return 0;
}

/* ===========================================================
 * fem_mesh_generate — main entry point
 * =========================================================== */
FemMesh* fem_mesh_generate(CsgSolid *solid, float h, float density) {
    assert(solid && solid->n_prisms > 0 && h > 0.0f);

    /* ---- compute AABB of all prisms ---- */
    float aabb_min[3] = { solid->prisms[0].min[0], solid->prisms[0].min[1], solid->prisms[0].min[2] };
    float aabb_max[3] = { solid->prisms[0].max[0], solid->prisms[0].max[1], solid->prisms[0].max[2] };
    for (int k = 1; k < solid->n_prisms; k++) {
        for (int d = 0; d < 3; d++) {
            if (solid->prisms[k].min[d] < aabb_min[d]) aabb_min[d] = solid->prisms[k].min[d];
            if (solid->prisms[k].max[d] > aabb_max[d]) aabb_max[d] = solid->prisms[k].max[d];
        }
    }

    /* ---- enumerate lattice nodes ---- */
    int nx = (int)ceilf((aabb_max[0]-aabb_min[0])/h) + 1;
    int ny = (int)ceilf((aabb_max[1]-aabb_min[1])/h) + 1;
    int nz = (int)ceilf((aabb_max[2]-aabb_min[2])/h) + 1;

    int pos_cap = nx*ny*nz/2+64;
    float *pos_arr = malloc((size_t)pos_cap*3*sizeof(float));
    int n_nodes = 0;

    NodeHashMap nhm; nhm_init(&nhm, 64);

    for (int iz = 0; iz < nz; iz++) {
        print_progress("Lattice nodes", iz, nz, NULL);
        for (int iy = 0; iy < ny; iy++)
        for (int ix = 0; ix < nx; ix++) {
            float p[3] = {
                aabb_min[0] + ix*h,
                aabb_min[1] + iy*h,
                aabb_min[2] + iz*h
            };
            if (!csg_is_inside(solid, p)) continue;
            if (n_nodes*3 >= pos_cap*3) {
                pos_cap *= 2;
                pos_arr = realloc(pos_arr, (size_t)pos_cap*3*sizeof(float));
            }
            pos_arr[n_nodes*3+0] = p[0];
            pos_arr[n_nodes*3+1] = p[1];
            pos_arr[n_nodes*3+2] = p[2];
            nhm_insert(&nhm, ix, iy, iz, n_nodes);
            n_nodes++;
        }
    }
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d x %d x %d, %d inside", nx, ny, nz, n_nodes);
        print_progress("Lattice nodes", nz, nz, buf);
    }

    if (n_nodes == 0) {
        nhm_free(&nhm); free(pos_arr);
        fprintf(stderr, "[fem_mesh] No nodes inside CSG solid\n");
        return NULL;
    }

    /* ---- build tetrahedra ---- */
    int tet_cap = 64;
    /* ci_arr[k][j] = node index of corner k in tet j */
    unsigned int *ci_arr[4];
    for (int k = 0; k < 4; k++) ci_arr[k] = malloc((size_t)tet_cap*sizeof(unsigned int));
    float *D0inv_arr = malloc((size_t)tet_cap*9*sizeof(float));
    float *V0_arr    = malloc((size_t)tet_cap*sizeof(float));
    int m_tets = 0;

    /* Iterate over all cube cells (ix, iy, iz) where ix/iy/iz are the
     * lower-left corner indices. */
    for (int iz = 0; iz < nz-1; iz++) {
        print_progress("Building tets", iz, nz-1, NULL);
    for (int iy = 0; iy < ny-1; iy++)
    for (int ix = 0; ix < nx-1; ix++) {
        /* Compute local-to-global node mapping for this cube's 8 corners */
        int gnode[8];
        int valid = 1;
        const int dx[8] = {0,1,0,1,0,1,0,1}; /* index = dx+2*dy+4*dz */
        const int dy[8] = {0,0,1,1,0,0,1,1};
        const int dz_[8]= {0,0,0,0,1,1,1,1};
        for (int v = 0; v < 8; v++) {
            int gn = nhm_lookup(&nhm, ix+dx[v], iy+dy[v], iz+dz_[v]);
            if (gn < 0) { valid = 0; break; }
            gnode[v] = gn;
        }
        if (!valid) continue;

        /* Add 6 tets for this cube */
        for (int t = 0; t < 6; t++) {
            unsigned int g0 = (unsigned int)gnode[CUBE_TETS[t][0]];
            unsigned int g1 = (unsigned int)gnode[CUBE_TETS[t][1]];
            unsigned int g2 = (unsigned int)gnode[CUBE_TETS[t][2]];
            unsigned int g3 = (unsigned int)gnode[CUBE_TETS[t][3]];

            /* Build D0 = [p1-p0, p2-p0, p3-p0] (column-major) */
            const float *p0 = pos_arr + g0*3;
            const float *p1 = pos_arr + g1*3;
            const float *p2 = pos_arr + g2*3;
            const float *p3 = pos_arr + g3*3;
            float d0[9];
            /* col 0 = p1-p0 */
            d0[0]=p1[0]-p0[0]; d0[1]=p1[1]-p0[1]; d0[2]=p1[2]-p0[2];
            /* col 1 = p2-p0 */
            d0[3]=p2[0]-p0[0]; d0[4]=p2[1]-p0[1]; d0[5]=p2[2]-p0[2];
            /* col 2 = p3-p0 */
            d0[6]=p3[0]-p0[0]; d0[7]=p3[1]-p0[1]; d0[8]=p3[2]-p0[2];

            float inv[9];
            float det = mat3_inv_cramer(d0, inv);
            float vol = det / 6.0f;

            /* All 6 Freudenthal tets should have det>0; skip if degenerate */
            if (vol < 1e-20f) continue;

            if (m_tets >= tet_cap) {
                tet_cap *= 2;
                for (int k=0;k<4;k++) ci_arr[k]=realloc(ci_arr[k],(size_t)tet_cap*sizeof(unsigned int));
                D0inv_arr = realloc(D0inv_arr, (size_t)tet_cap*9*sizeof(float));
                V0_arr    = realloc(V0_arr,    (size_t)tet_cap*sizeof(float));
            }

            ci_arr[0][m_tets] = g0;
            ci_arr[1][m_tets] = g1;
            ci_arr[2][m_tets] = g2;
            ci_arr[3][m_tets] = g3;
            memcpy(D0inv_arr + m_tets*9, inv, 9*sizeof(float));
            V0_arr[m_tets] = vol;
            m_tets++;
        }
    }

    } /* end iz */
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d tets", m_tets);
        print_progress("Building tets", nz-1, nz-1, buf);
    }

    nhm_free(&nhm);

    if (m_tets == 0) {
        for(int k=0;k<4;k++) free(ci_arr[k]);
        free(D0inv_arr); free(V0_arr); free(pos_arr);
        fprintf(stderr, "[fem_mesh] No tetrahedra generated\n");
        return NULL;
    }
    fprintf(stdout, "[fem_mesh] %d nodes, %d tets\n", n_nodes, m_tets);

    /* ---- build backward SpMV adjacency ---- */
    /* Generate all (node_i, packed=(k<<28|j)) entries */
    int total_adj = m_tets * 4;
    AdjEntry *adj_all = malloc((size_t)total_adj * sizeof(AdjEntry));
    int adj_cnt = 0;
    for (int k = 0; k < 4; k++) {
        print_progress("Adjacency", k, 4, NULL);
        for (int j = 0; j < m_tets; j++) {
            unsigned int ni = ci_arr[k][j];
            adj_all[adj_cnt].node_i = ni;
            adj_all[adj_cnt].packed = ((unsigned int)k << 28) | (unsigned int)j;
            adj_cnt++;
        }
    }
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d entries, sorting...", total_adj);
        print_progress("Adjacency", 4, 4, buf);
    }
    qsort(adj_all, (size_t)total_adj, sizeof(AdjEntry), adj_cmp);

    unsigned int *adj_start = calloc((size_t)(n_nodes+1), sizeof(unsigned int));
    unsigned int *adj_data  = malloc((size_t)total_adj * sizeof(unsigned int));
    /* Build CSR */
    for (int t = 0; t < total_adj; t++) adj_data[t] = adj_all[t].packed;
    /* Count per-node */
    for (int t = 0; t < total_adj; t++) adj_start[adj_all[t].node_i + 1]++;
    /* Prefix sum */
    for (int i = 0; i < n_nodes; i++) adj_start[i+1] += adj_start[i];
    /* Fill adj_data in CSR order */
    {
        unsigned int *cur = calloc((size_t)n_nodes, sizeof(unsigned int));
        for (int t = 0; t < total_adj; t++) {
            unsigned int ni = adj_all[t].node_i;
            adj_data[adj_start[ni] + cur[ni]] = adj_all[t].packed;
            cur[ni]++;
        }
        free(cur);
    }
    free(adj_all);

    /* ---- extract exterior surface triangles ---- */
    /* Cap must be power of 2 and large enough (load < 50%) */
    int fhm_cap = 16;
    while (fhm_cap < m_tets * 8 + 16) fhm_cap *= 2;
    FaceHashMap fhm; fhm_init(&fhm, fhm_cap);

    int last_pct = -1;
    for (int j = 0; j < m_tets; j++) {
        int pct = j * 100 / m_tets;
        if (pct != last_pct) { print_progress("Surf faces", j, m_tets, NULL); last_pct = pct; }
        unsigned int c[4] = {ci_arr[0][j],ci_arr[1][j],ci_arr[2][j],ci_arr[3][j]};
        for (int f = 0; f < 4; f++) {
            unsigned int fa=c[TET_FACES[f][0]], fb=c[TET_FACES[f][1]], fc=c[TET_FACES[f][2]];
            /* The 4th corner is the one opposite this face */
            int opp = (f==0)?0:(f==1)?1:(f==2)?2:3;
            /* find the remaining corner index */
            int used[4]={0}; used[TET_FACES[f][0]]=1; used[TET_FACES[f][1]]=1; used[TET_FACES[f][2]]=1;
            unsigned int fourth_local = 0;
            for(int i=0;i<4;i++) if(!used[i]) { fourth_local=(unsigned int)i; break; }
            (void)opp;
            fhm_add(&fhm, fa, fb, fc, fa, fb, fc, c[fourth_local], (unsigned int)j);
        }
    }
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d unique faces", fhm.size);
        print_progress("Surf faces", m_tets, m_tets, buf);
    }

    /* Collect boundary faces (appear exactly once) */
    int surf_cap = 64;
    unsigned int *surf_tri     = malloc((size_t)surf_cap*3*sizeof(unsigned int));
    unsigned int *surf_tri_tet = malloc((size_t)surf_cap*sizeof(unsigned int));
    int n_surf = 0;

    for (int i = 0; i < fhm.cap; i++) {
        FaceEntry *fe = &fhm.table[i];
        if (fe->tet_count != 1) continue;

        /* Orient outward: normal should point away from fourth_corner.
         * n = cross(b-a, c-a); if dot(n, fourth-a) > 0 → inward → swap b,c */
        unsigned int oa=fe->orig[0], ob=fe->orig[1], oc=fe->orig[2], od=fe->fourth;
        const float *pa=pos_arr+oa*3, *pb=pos_arr+ob*3;
        const float *pc=pos_arr+oc*3, *pd=pos_arr+od*3;
        float ba[3], ca[3], da[3];
        v3_sub(ba,pb,pa); v3_sub(ca,pc,pa); v3_sub(da,pd,pa);
        float n[3]; v3_cross(n,ba,ca);
        if (v3_dot(n,da) > 0.0f) {
            /* flip b and c to make normal point away from d */
            unsigned int tmp=ob; ob=oc; oc=tmp;
        }

        if (n_surf*3+2 >= surf_cap*3) {
            surf_cap*=2;
            surf_tri    =realloc(surf_tri,    (size_t)surf_cap*3*sizeof(unsigned int));
            surf_tri_tet=realloc(surf_tri_tet,(size_t)surf_cap  *sizeof(unsigned int));
        }
        surf_tri[n_surf*3+0]=oa;
        surf_tri[n_surf*3+1]=ob;
        surf_tri[n_surf*3+2]=oc;
        surf_tri_tet[n_surf] = fe->tet_idx;
        n_surf++;
    }
    /* Collect interior faces (appear exactly twice) — exposed by fracture */
    int int_cap = 64;
    unsigned int *int_tri      = malloc((size_t)int_cap*3*sizeof(unsigned int));
    unsigned int *int_tri_tets = malloc((size_t)int_cap*2*sizeof(unsigned int));
    unsigned int *int_tri_tet4th = malloc((size_t)int_cap*2*sizeof(unsigned int));
    int n_int = 0;
    for (int i = 0; i < fhm.cap; i++) {
        FaceEntry *fe = &fhm.table[i];
        if (fe->tet_count != 2) continue;
        if (n_int >= int_cap) {
            int_cap *= 2;
            int_tri      = realloc(int_tri,      (size_t)int_cap*3*sizeof(unsigned int));
            int_tri_tets = realloc(int_tri_tets, (size_t)int_cap*2*sizeof(unsigned int));
            int_tri_tet4th = realloc(int_tri_tet4th, (size_t)int_cap*2*sizeof(unsigned int));
        }
        /* Store original (unsorted) vertices from the first tet's perspective */
        int_tri[n_int*3+0] = fe->orig[0];
        int_tri[n_int*3+1] = fe->orig[1];
        int_tri[n_int*3+2] = fe->orig[2];
        int_tri_tets[n_int*2+0] = fe->tet_idx;
        int_tri_tets[n_int*2+1] = fe->tet_idx2;
        int_tri_tet4th[n_int*2+0] = fe->fourth;
        int_tri_tet4th[n_int*2+1] = fe->fourth2;
        n_int++;
    }

    fhm_free(&fhm);
    fprintf(stdout, "[fem_mesh] %d surface triangles, %d interior faces\n", n_surf, n_int);

    /* ---- compute lumped node masses ---- */
    float *mass_arr = calloc((size_t)n_nodes, sizeof(float));
    for (int j = 0; j < m_tets; j++) {
        float contrib = density * V0_arr[j] / 4.0f;
        for (int k = 0; k < 4; k++)
            mass_arr[ci_arr[k][j]] += contrib;
    }

    /* ---- assemble FemMesh ---- */
    FemMesh *mesh = calloc(1, sizeof(FemMesh));
    mesh->n_nodes   = n_nodes;
    mesh->pos       = pos_arr;
    mesh->mass      = mass_arr;
    mesh->m_tets    = m_tets;
    for (int k=0;k<4;k++) mesh->ci[k] = ci_arr[k];
    mesh->D0_inv    = D0inv_arr;
    mesh->V0        = V0_arr;
    mesh->total_adj = total_adj;
    mesh->adj_start = adj_start;
    mesh->adj_data  = adj_data;
    mesh->n_surf_tri      = n_surf;
    mesh->surf_tri        = surf_tri;
    mesh->surf_tri_tet    = surf_tri_tet;
    mesh->n_int_tri       = n_int;
    mesh->int_tri         = int_tri;
    mesh->int_tri_tets    = int_tri_tets;
    mesh->int_tri_tet4th  = int_tri_tet4th;
    mesh->solid     = solid;
    mesh->lattice_h = h;
    return mesh;
}

void fem_mesh_free(FemMesh *m) {
    if (!m) return;
    free(m->pos); free(m->mass);
    for (int k=0;k<4;k++) free(m->ci[k]);
    free(m->D0_inv); free(m->V0);
    free(m->adj_start); free(m->adj_data);
    free(m->surf_tri);
    free(m->surf_tri_tet);
    free(m->int_tri);
    free(m->int_tri_tets);
    free(m->int_tri_tet4th);
    csg_solid_free(m->solid);
    free(m);
}

/* ===========================================================
 * Möller–Trumbore ray–triangle intersection
 * =========================================================== */
float fem_ray_tri_intersect(const float *orig, const float *dir,
                             const float *va, const float *vb, const float *vc,
                             float *out_u, float *out_v)
{
    float edge1[3], edge2[3], h[3], s[3], q[3];
    v3_sub(edge1,vb,va); v3_sub(edge2,vc,va);
    v3_cross(h,dir,edge2);
    float a = v3_dot(edge1,h);
    if (fabsf(a) < 1e-8f) return -1.0f;
    float f=1.0f/a;
    v3_sub(s,orig,va);
    float u=f*v3_dot(s,h);
    if (u<0.0f||u>1.0f) return -1.0f;
    v3_cross(q,s,edge1);
    float v=f*v3_dot(dir,q);
    if (v<0.0f||u+v>1.0f) return -1.0f;
    float t=f*v3_dot(edge2,q);
    if (t<0.001f) return -1.0f;
    if (out_u) *out_u=u;
    if (out_v) *out_v=v;
    return t;
}
