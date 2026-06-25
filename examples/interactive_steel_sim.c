/*
 * interactive_steel_sim.c — Real-time FEM elastic solid with first-person camera.
 *
 * Controls (SIMULATE mode):
 *   WASD + Mouse    − first-person camera navigation
 *   Tab             − switch to EDIT mode
 *   Hold K          − anchor nodes under cursor (ray-cast paint)
 *   Hold F          − apply external force in camera-forward direction
 *   Hold C          − clear paint under cursor
 *   R               − reset simulation to rest state
 *   E               − export CSG + constraints to .fes file (OS save dialog)
 *   O               − open .fes scene file via OS dialog and load it
 *   Escape          − quit
 *
 * Controls (EDIT mode):
 *   Mouse hover     − highlight face of a CSG prism under cursor
 *   Left drag       − extend / contract highlighted face along its normal
 *   Tab / Enter     − confirm edit → rebuild mesh and re-enter SIMULATE mode
 *   Escape          − cancel edit and revert to previous solid
 *
 * Material defaults (adjustable via menu):
 *   E*  = 2×10^5 Pa  (soft rubber-like, numerically stable with explicit Euler)
 *   ν   = 0.40
 *   ρ   = 1000 kg/m³
 *   h   = 0.10 m     lattice spacing
 *   dt  = 1×10^-4 s  per substep
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#else
#define MAX_PATH 1024
#endif

#include "../include/fem_mesh.h"
#include "../include/fem_sim.h"
#include "../include/Menu.h"

/* ─── window ─────────────────────────────────────────────── */
#define WIN_W 1280
#define WIN_H 720

/* ─── camera ─────────────────────────────────────────────── */
#define CAM_SPEED       2.0f
#define CAM_SENSITIVITY 0.0015f
#define NEAR_Z          0.01f
#define FAR_Z           200.0f
#define FOV_Y           ((float)(M_PI * 0.5))   /* 90° */

/* ─── default material ───────────────────────────────────── */
/* Calibrated for structural steel at ~2 m scale, h = 0.08 m.
 * E = 200 GPa, ν = 0.30, ρ = 7850 kg/m³.
 * P-wave speed ≈ 5860 m/s → CFL dt_crit ≈ 1.36e-5 s at h=0.08 m.
 * Use dt = 4e-6 s (safety factor ~3.4). dt scales with h; nsteps
 * scales inversely so physical time per frame stays constant. */
#define DEF_E_STAR   200e9
#define DEF_NU       0.30
#define DEF_DENSITY  7850.0
#define DEF_H        0.08f
#define DEF_DT       4e-6
#define DEF_NSTEPS   250
#define DEF_GRAV     9.81
#define DEF_PRADIUS  0.15

/* ============================================================
 * Minimal mat4 library (column-major, matches GLSL / OpenGL)
 * ========================================================== */

/* column-major C = A * B */
static void m4_mul(const float *A, const float *B, float *C) {
    for (int c=0; c<4; c++)
        for (int r=0; r<4; r++) {
            float s=0;
            for (int k=0; k<4; k++) s+=A[k*4+r]*B[c*4+k];
            C[c*4+r]=s;
        }
}

/* Perspective projection (column-major) */
static void m4_perspective(float fovy, float asp, float zn, float zf, float *P) {
    memset(P,0,64);
    float f=1.0f/tanf(fovy*0.5f);
    P[0]  = f/asp;
    P[5]  = f;
    P[10] = (zf+zn)/(zn-zf);
    P[11] = -1.0f;
    P[14] = 2.0f*zf*zn/(zn-zf);
}

/* View matrix from camera angles (FPS, column-major) */
static void m4_view(float yaw, float pitch,
                    float px, float py, float pz, float *V)
{
    /* forward direction camera looks (into scene) */
    float cy=cosf(yaw), sy=sinf(yaw);
    float cp=cosf(pitch), sp=sinf(pitch);
    float fx= sy*cp, fy=-sp, fz=-cy*cp;
    /* right = yaw-only rotation of world +X */
    float rx=cy, ry=0.0f, rz=sy;
    /* up = cross(right, forward) */
    float ux=ry*fz-rz*fy, uy=rz*fx-rx*fz, uz=rx*fy-ry*fx;
    /* view matrix rows: right, up, -forward; then translation */
    V[0]=rx; V[4]=ry; V[8] =rz; V[12]=-(rx*px+ry*py+rz*pz);
    V[1]=ux; V[5]=uy; V[9] =uz; V[13]=-(ux*px+uy*py+uz*pz);
    V[2]=-fx;V[6]=-fy;V[10]=-fz;V[14]= (fx*px+fy*py+fz*pz);
    V[3]=0;  V[7]=0;  V[11]=0;  V[15]=1;
}

/* Retrieve camera forward/right vectors from yaw/pitch */
static void cam_dirs(float yaw, float pitch,
                     float *fwd, float *right, float *up)
{
    float cy=cosf(yaw),sy=sinf(yaw),cp=cosf(pitch),sp=sinf(pitch);
    if (fwd)  { fwd[0]=sy*cp; fwd[1]=-sp; fwd[2]=-cy*cp; }
    if (right){ right[0]=cy;  right[1]=0;  right[2]=sy;   }
    if (up)   {
        float rx=cy,ry=0,rz=sy;
        float fx=sy*cp,fy=-sp,fz=-cy*cp;
        up[0]=ry*fz-rz*fy; up[1]=rz*fx-rx*fz; up[2]=rx*fy-ry*fx;
    }
}

/* ============================================================
 * CSG solid clone helper
 * ========================================================== */
static CsgSolid* csg_clone(const CsgSolid *src) {
    CsgSolid *c = csg_solid_create();
    for (int i=0; i<src->n_prisms; i++)
        csg_solid_add_prism(c,
            (float*)src->prisms[i].min,
            (float*)src->prisms[i].max,
            src->prisms[i].sign);
    return c;
}

/* ============================================================
 * Ray–AABB face intersection
 * Returns 1 on hit; writes t, which face axis (0-2), side (-1/+1 for min/max).
 * ========================================================== */
static int ray_prism_face(const float *ro, const float *rd,
                          const float *mn, const float *mx,
                          float *t_out, int *ax_out, int *si_out)
{
    float bt=1e30f; int bax=-1, bsi=0;
    for (int ax=0; ax<3; ax++) {
        for (int si=0; si<2; si++) {
            float plane = si ? mx[ax] : mn[ax];
            if (fabsf(rd[ax])<1e-9f) continue;
            float t=(plane-ro[ax])/rd[ax];
            if (t<1e-3f || t>=bt) continue;
            int a1=(ax+1)%3, a2=(ax+2)%3;
            float p1=ro[a1]+t*rd[a1], p2=ro[a2]+t*rd[a2];
            if (p1<mn[a1]-1e-4f || p1>mx[a1]+1e-4f) continue;
            if (p2<mn[a2]-1e-4f || p2>mx[a2]+1e-4f) continue;
            bt=t; bax=ax; bsi=si?+1:-1;
        }
    }
    if (bax<0) return 0;
    *t_out=bt; *ax_out=bax; *si_out=bsi; return 1;
}

/* ============================================================
 * Ray from camera through screen point (sx,sy in [-1,+1])
 * ========================================================== */
static void screen_ray(float yaw, float pitch,
                       float px, float py, float pz,
                       float sx, float sy, float asp,
                       float *orig, float *dir)
{
    orig[0]=px; orig[1]=py; orig[2]=pz;
    float fwd[3], right[3], up[3];
    cam_dirs(yaw, pitch, fwd, right, up);
    float half_h = tanf(FOV_Y*0.5f);
    float half_w = half_h * asp;
    dir[0] = fwd[0] + sx*half_w*right[0] + sy*half_h*up[0];
    dir[1] = fwd[1] + sx*half_w*right[1] + sy*half_h*up[1];
    dir[2] = fwd[2] + sx*half_w*right[2] + sy*half_h*up[2];
    float len=sqrtf(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
    if (len>0){dir[0]/=len;dir[1]/=len;dir[2]/=len;}
}

/* ============================================================
 * Constraint arrays
 * ========================================================== */
#define MAX_NODES (64 * 1024)   /* plenty for h=0.1 on unit cube */
static unsigned int g_anchor[MAX_NODES];
static float        g_ext_force[MAX_NODES*3];

static void constraints_clear(int n) {
    memset(g_anchor,    0, (size_t)n*sizeof(unsigned int));
    memset(g_ext_force, 0, (size_t)n*3*sizeof(float));
}

/* Default: anchor all nodes with y ≈ 0 */
static void constraints_default(const FemMesh *mesh) {
    constraints_clear(mesh->n_nodes);
    for (int i=0; i<mesh->n_nodes; i++)
        if (mesh->pos[i*3+1] < mesh->lattice_h * 0.5f)
            g_anchor[i] = 1;
}

/* forward-declare so paint_near can reference the slider variable */
static double d_force_mag = 1e4;   /* N per node per paint stroke */

/* ============================================================
 * Painting: ray-cast against rest-pos surface triangles on CPU
 * ========================================================== */
static void paint_near(const FemMesh *mesh, const float *ro, const float *rd,
                       float radius, int mode /* 0=anchor,1=force,2=clear */)
{
    /* Find nearest hit triangle */
    float best_t = 1e30f;
    int   hit_tri = -1;
    for (int t=0; t<mesh->n_surf_tri; t++) {
        unsigned int a=mesh->surf_tri[t*3+0],
                     b=mesh->surf_tri[t*3+1],
                     c=mesh->surf_tri[t*3+2];
        float u,v;
        float ht = fem_ray_tri_intersect(ro,rd,
            mesh->pos+a*3, mesh->pos+b*3, mesh->pos+c*3, &u,&v);
        if (ht>=0 && ht<best_t){best_t=ht;hit_tri=t;}
    }
    if (hit_tri<0) return;

    /* Hit point in world space (rest position) */
    float hp[3] = {
        ro[0]+best_t*rd[0],
        ro[1]+best_t*rd[1],
        ro[2]+best_t*rd[2]
    };

    /* Paint all nodes within radius */
    float r2 = radius*radius;
    for (int i=0; i<mesh->n_nodes; i++) {
        float dx=mesh->pos[i*3+0]-hp[0];
        float dy=mesh->pos[i*3+1]-hp[1];
        float dz=mesh->pos[i*3+2]-hp[2];
        if (dx*dx+dy*dy+dz*dz > r2) continue;
        if (mode==2){
            g_anchor[i]=0;
            g_ext_force[i*3+0]=g_ext_force[i*3+1]=g_ext_force[i*3+2]=0;
        } else if (mode==0){
            g_anchor[i]=1;
        } else {
            /* Force in camera-forward direction (stored as impulse Newton) */
            /* magnitude = 10 N for demonstration */
            float *ef = g_ext_force+i*3;
            ef[0]+=rd[0]*(float)d_force_mag;
            ef[1]+=rd[1]*(float)d_force_mag;
            ef[2]+=rd[2]*(float)d_force_mag;
        }
    }
}

/* ============================================================
 * Menu creation
 * ========================================================== */
static double d_E_star   = DEF_E_STAR;
static double d_nu       = DEF_NU;
static double d_density  = DEF_DENSITY;
static double d_dt       = DEF_DT;
static double d_nsteps   = DEF_NSTEPS;
static double d_gravity  = DEF_GRAV;
static double d_pradius  = DEF_PRADIUS;
static int    d_colorize = 1;
static int    d_show_slice = 0;
static double d_scalar_max = 0.15;
static double d_slice_x    = 0.5;
static double d_slice_y    = 0.5;
static double d_slice_z    = 0.5;
static double d_slice_yaw  = 0.0;
static double d_slice_pitch= 0.0;
static double d_slice_h    = DEF_H * 0.75;

/* ---- plasticity parameters ---- */
static double d_yield_stress   = 250e6;   /* Pa, mild steel ~250 MPa */
static double d_hardening_mod  = 1e9;     /* Pa, strain hardening    */
static double d_fracture_strain = 0.30;   /* acc. plastic strain at fracture */
static int    d_plasticity_on  = 1;

static int build_face_extrusion(const RectPrism *base, int ax, int si,
                                float length, RectPrism *out)
{
    /* length > 0: extend outward from face (new prism added beyond face)
     * length < 0: inset inward; if |inset| >= base thickness, drill-through:
     *   returns a sign=-1 subtractive prism spanning the full drag depth. */
    if (!base || !out || ax < 0 || ax > 2 || si == 0) return 0;
    float base_len = base->max[ax] - base->min[ax];
    if (length > 0.0f) {
        /* outward: new prism shares the selected face, extends beyond it */
        *out = *base;
        out->sign = +1;
        if (si > 0) {
            out->min[ax] = base->max[ax];
            out->max[ax] = base->max[ax] + length;
        } else {
            out->max[ax] = base->min[ax];
            out->min[ax] = base->min[ax] - length;
        }
        return 1;
    } else {
        float inset = -length;  /* positive magnitude */
        if (inset >= base_len - 1e-5f) {
            /* Drill-through: subtractive prism spanning the full drag depth.
             * Same cross-section as base; starts at the selected face and
             * extends inward by inset (potentially past the opposite face). */
            *out = *base;
            out->sign = -1;
            if (si > 0) {
                out->min[ax] = base->max[ax] - inset;
                out->max[ax] = base->max[ax];
            } else {
                out->min[ax] = base->min[ax];
                out->max[ax] = base->min[ax] + inset;
            }
            return 1;
        }
        /* Normal inward shrink */
        *out = *base;
        out->sign = +1;
        if (si > 0) {
            out->max[ax] = base->max[ax] - inset;
        } else {
            out->min[ax] = base->min[ax] + inset;
        }
        return 1;
    }
}

static Menu* build_menu(void) {
    Color tc = {230,230,230,255};
    Color bc = {30,30,30,200};
    Menu *m = menu_create(10, 10, 320, 520, 0, "FEM Steel Sim", tc, bc);

#define ADD_SLIDER(var, label, lo, hi) do { \
    MenuRow *row = menurow_create(); \
    menurow_add_interaction(row, \
        variableinteraction_create(&var, label, lo, hi, VAR_SLIDER, NULL, NULL)); \
    menu_add_row(m, row); } while(0)
#define ADD_SLIDER_STEP(var, label, lo, hi, stepv) do { \
    MenuRow *row = menurow_create(); \
    VariableInteraction *vi = variableinteraction_create(&var, label, lo, hi, VAR_SLIDER, NULL, NULL); \
    variableinteraction_set_step(vi, stepv); \
    menurow_add_interaction(row, vi); \
    menu_add_row(m, row); } while(0)
#define ADD_BOOL(var, label) do { \
    MenuRow *row = menurow_create(); \
    menurow_add_interaction(row, \
        variableinteraction_create(&var, label, 0, 1, VAR_BOOL, NULL, NULL)); \
    menu_add_row(m, row); } while(0)

    ADD_SLIDER(d_E_star,  "E* (Pa)",          1e9,  4e11);
    ADD_SLIDER(d_nu,      "nu",                0.01, 0.48);
    ADD_SLIDER(d_density, "density (kg/m3)",   500,  12000);
    ADD_SLIDER(d_dt,      "dt (s)",            1e-7, 5e-5);
    ADD_SLIDER_STEP(d_nsteps,  "substeps",     1,    300,  1.0);
    ADD_SLIDER(d_gravity, "gravity (m/s2)",    0,    20);
    ADD_SLIDER(d_pradius, "paint radius (m)",  0.01, 0.5);
    ADD_SLIDER(d_force_mag, "force mag (N)",    1e1,  1e4);
    ADD_BOOL(d_colorize,  "Color shell by disp");
    ADD_BOOL(d_show_slice,"Show slice");
    ADD_SLIDER(d_scalar_max, "scalar max",     1e-4, 1e-2);
    ADD_SLIDER(d_slice_x,    "slice x",       -2.0,  3.0);
    ADD_SLIDER(d_slice_y,    "slice y",       -2.0,  3.0);
    ADD_SLIDER(d_slice_z,    "slice z",       -2.0,  3.0);
    ADD_SLIDER_STEP(d_slice_yaw,   "slice yaw",   -3.14159, 3.14159, 0.05);
    ADD_SLIDER_STEP(d_slice_pitch, "slice pitch", -1.50,    1.50,    0.05);
    ADD_SLIDER(d_slice_h,    "slice half-thick", 0.005, 0.30);
    ADD_BOOL(d_plasticity_on,      "Plasticity on");
    ADD_SLIDER(d_yield_stress,     "yield stress (Pa)", 1e6,  2e9);
    ADD_SLIDER(d_hardening_mod,    "hardening H (Pa)",  0,    5e9);
    ADD_SLIDER(d_fracture_strain,  "fracture strain",   0.01, 2.0);
#undef ADD_BOOL
#undef ADD_SLIDER_STEP
#undef ADD_SLIDER

    menu_open_font(12);
    return m;
}

/* ============================================================
 * Constraint visualization: colored GL_POINTS at rest positions
 *   blue   = anchored
 *   red    = force applied
 *   magenta= both
 * ========================================================== */
static void draw_constraints(const FemMesh *mesh)
{
    glPointSize(7.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < mesh->n_nodes; i++) {
        int   anch = g_anchor[i];
        float fx   = g_ext_force[i*3+0];
        float fy   = g_ext_force[i*3+1];
        float fz   = g_ext_force[i*3+2];
        int   hf   = (fx*fx + fy*fy + fz*fz) > 0.0f;
        if (anch && hf)    glColor4f(0.85f, 0.20f, 0.85f, 1.0f); /* magenta */
        else if (anch)     glColor4f(0.20f, 0.45f, 1.00f, 1.0f); /* blue    */
        else if (hf)       glColor4f(1.00f, 0.30f, 0.15f, 1.0f); /* red     */
        else continue;
        glVertex3fv(mesh->pos + i*3);
    }
    glEnd();
    glPointSize(1.0f);
}

/* ============================================================
 * Custom 2-D cursor drawn in screen space (fixed-function GL)
 * tool: 0=none 1=anchor 2=force 3=clear 4=extrude
 * r,g,b: tool colour.  ring_r: screen-pixel radius for paint ring.
 * ========================================================== */
static void draw_cursor_2d(int x, int y, float r, float g, float b,
                           int tool, int ring_r, int win_w, int win_h)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix(); glLoadIdentity();
    glOrtho(0, win_w, win_h, 0, -1, 1);   /* top-left origin, Y down */
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix(); glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(2.0f);

    if (tool >= 1 && tool <= 3) {
        /* Paint tools: coloured circle + small cross */
        glColor4f(r, g, b, 0.90f);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < 32; i++) {
            float a = (float)i / 32.0f * 2.0f * (float)M_PI;
            glVertex2f((float)x + cosf(a)*(float)ring_r,
                       (float)y + sinf(a)*(float)ring_r);
        }
        glEnd();
        /* centre cross */
        glBegin(GL_LINES);
        glVertex2f((float)x-5, (float)y); glVertex2f((float)x+5, (float)y);
        glVertex2f((float)x, (float)y-5); glVertex2f((float)x, (float)y+5);
        glEnd();
    } else if (tool == 4) {
        /* Extrude: 4-arrow crosshair */
        int sz = 14, hs = 5;
        glColor4f(r, g, b, 1.0f);
        glBegin(GL_LINES);
        glVertex2i(x-sz, y); glVertex2i(x+sz, y);
        glVertex2i(x, y-sz); glVertex2i(x, y+sz);
        glEnd();
        glBegin(GL_TRIANGLES);
        /* right */ glVertex2i(x+sz,y); glVertex2i(x+sz-hs,y-hs+2); glVertex2i(x+sz-hs,y+hs-2);
        /* left  */ glVertex2i(x-sz,y); glVertex2i(x-sz+hs,y-hs+2); glVertex2i(x-sz+hs,y+hs-2);
        /* up    */ glVertex2i(x,y-sz); glVertex2i(x-hs+2,y-sz+hs); glVertex2i(x+hs-2,y-sz+hs);
        /* down  */ glVertex2i(x,y+sz); glVertex2i(x-hs+2,y+sz-hs); glVertex2i(x+hs-2,y+sz-hs);
        glEnd();
    } else {
        /* None: small dot */
        glColor4f(r, g, b, 1.0f);
        glPointSize(6.0f);
        glBegin(GL_POINTS); glVertex2i(x, y); glEnd();
        glPointSize(1.0f);
    }

    glLineWidth(1.0f);
    glMatrixMode(GL_PROJECTION); glPopMatrix();
    glMatrixMode(GL_MODELVIEW);  glPopMatrix();
}

/* ============================================================
 * Wireframe helper for edit-mode AABB overlay (fixed pipeline)
 * ========================================================== */
static void draw_aabb_wire(const float *mn, const float *mx,
                           float r, float g, float b)
{
    /* 12 edges of a box */
    static const int E[12][2][3] = {
        {{0,0,0},{1,0,0}}, {{0,1,0},{1,1,0}},
        {{0,0,1},{1,0,1}}, {{0,1,1},{1,1,1}},
        {{0,0,0},{0,1,0}}, {{1,0,0},{1,1,0}},
        {{0,0,1},{0,1,1}}, {{1,0,1},{1,1,1}},
        {{0,0,0},{0,0,1}}, {{1,0,0},{1,0,1}},
        {{0,1,0},{0,1,1}}, {{1,1,0},{1,1,1}},
    };
    glColor3f(r,g,b);
    glBegin(GL_LINES);
    for (int e=0; e<12; e++) {
        for (int v=0; v<2; v++)
            glVertex3f(E[e][v][0]?mx[0]:mn[0],
                       E[e][v][1]?mx[1]:mn[1],
                       E[e][v][2]?mx[2]:mn[2]);
    }
    glEnd();
}

/* Draw a filled quad at face (ax, si=+/-1) of AABB with custom colour */
static void draw_face_highlight_color(const float *mn, const float *mx, int ax, int si,
                                      float r, float g, float b)
{
    float v[4][3];
    float plane = si>0 ? mx[ax] : mn[ax];
    int a1=(ax+1)%3, a2=(ax+2)%3;
    float corners[2][2] = {{mn[a1],mn[a2]},{mx[a1],mx[a2]}};
    for (int i=0; i<4; i++) {
        v[i][ax]=plane;
        v[i][a1]=corners[(i>>0)&1][0];
        v[i][a2]=corners[(i>>1)&1][1];
    }
    glColor4f(r, g, b, 0.35f);
    glBegin(GL_QUADS);
    for (int i=0; i<4; i++) glVertex3fv(v[i]);
    glEnd();
}

/* Yellow face highlight (hover) */
static void draw_face_highlight(const float *mn, const float *mx, int ax, int si)
{
    draw_face_highlight_color(mn, mx, ax, si, 1.0f, 1.0f, 0.0f);
}

/* ============================================================
 * Multi-face selection: one entry per selected face
 * ========================================================== */
#define MAX_SEL_FACES 64
typedef struct {
    int   prism_idx;   /* index into edit_solid->prisms */
    int   ax;          /* axis 0-2 */
    int   si;          /* side -1 or +1 */
    float join_accum;  /* edit_drag_accum when this face joined the group */
} SelectedFace;

/* ============================================================
 * Scene file I/O
 * Format: FEMSCENE_V1 header, LATTICE_H, PRISMS n (one per line),
 *         NODES n (anchor + force[3] per node).
 * ========================================================== */
static int save_scene(const char *path, const CsgSolid *solid,
                      float lattice_h, int n_nodes,
                      const unsigned int *anchor, const float *force)
{
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fprintf(f, "FEMSCENE_V1\n");
    fprintf(f, "LATTICE_H %.9f\n", (double)lattice_h);
    fprintf(f, "PRISMS %d\n", solid->n_prisms);
    for (int i = 0; i < solid->n_prisms; i++) {
        const RectPrism *pr = &solid->prisms[i];
        fprintf(f, "%.9f %.9f %.9f %.9f %.9f %.9f %d\n",
                (double)pr->min[0], (double)pr->min[1], (double)pr->min[2],
                (double)pr->max[0], (double)pr->max[1], (double)pr->max[2],
                pr->sign);
    }
    fprintf(f, "NODES %d\n", n_nodes);
    for (int i = 0; i < n_nodes; i++)
        fprintf(f, "%u %.9f %.9f %.9f\n", anchor[i],
                (double)force[i*3+0], (double)force[i*3+1], (double)force[i*3+2]);
    fclose(f);
    return 1;
}

/* Returns 1 on success; caller must free *out_anchor and *out_force. */
static int load_scene(const char *path, CsgSolid **out_solid,
                      float *out_lattice_h, int *out_n_nodes,
                      unsigned int **out_anchor, float **out_force)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char tag[64];
    if (fscanf(f, "%63s", tag) != 1 || strcmp(tag, "FEMSCENE_V1") != 0)
        { fclose(f); return 0; }
    float lh;
    if (fscanf(f, " LATTICE_H %f", &lh) != 1 || lh <= 0.0f)
        { fclose(f); return 0; }
    int np;
    if (fscanf(f, " PRISMS %d", &np) != 1 || np <= 0 || np > 4096)
        { fclose(f); return 0; }
    CsgSolid *s = csg_solid_create();
    for (int i = 0; i < np; i++) {
        float mn[3], mx[3]; int sign;
        if (fscanf(f, " %f %f %f %f %f %f %d",
                   &mn[0], &mn[1], &mn[2], &mx[0], &mx[1], &mx[2], &sign) != 7)
            { csg_solid_free(s); fclose(f); return 0; }
        csg_solid_add_prism(s, mn, mx, sign);
    }
    int nn;
    if (fscanf(f, " NODES %d", &nn) != 1 || nn <= 0 || nn > MAX_NODES)
        { csg_solid_free(s); fclose(f); return 0; }
    unsigned int *anch = calloc((size_t)nn, sizeof(unsigned int));
    float        *forc = calloc((size_t)nn * 3, sizeof(float));
    if (!anch || !forc)
        { free(anch); free(forc); csg_solid_free(s); fclose(f); return 0; }
    for (int i = 0; i < nn; i++) {
        if (fscanf(f, " %u %f %f %f",
                   &anch[i], &forc[i*3+0], &forc[i*3+1], &forc[i*3+2]) != 4)
            { free(anch); free(forc); csg_solid_free(s); fclose(f); return 0; }
    }
    fclose(f);
    *out_solid     = s;
    *out_lattice_h = lh;
    *out_n_nodes   = nn;
    *out_anchor    = anch;
    *out_force     = forc;
    return 1;
}

/* OS-native file dialogs ------------------------------------ */
#ifdef _WIN32
static int open_file_dialog(char *out_path, int max_len) {
    OPENFILENAMEA ofn;
    memset(&ofn, 0, sizeof(ofn));
    out_path[0] = '\0';
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter = "FEM Scene (*.fes)\0*.fes\0All Files (*.*)\0*.*\0\0";
    ofn.lpstrFile   = out_path;
    ofn.nMaxFile    = (DWORD)max_len;
    ofn.lpstrTitle  = "Open FEM Scene";
    ofn.Flags       = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    return GetOpenFileNameA(&ofn) ? 1 : 0;
}
static int save_file_dialog(char *out_path, int max_len) {
    OPENFILENAMEA ofn;
    memset(&ofn, 0, sizeof(ofn));
    out_path[0] = '\0';
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFilter  = "FEM Scene (*.fes)\0*.fes\0All Files (*.*)\0*.*\0\0";
    ofn.lpstrFile    = out_path;
    ofn.nMaxFile     = (DWORD)max_len;
    ofn.lpstrTitle   = "Save FEM Scene";
    ofn.lpstrDefExt  = "fes";
    ofn.Flags        = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;
    return GetSaveFileNameA(&ofn) ? 1 : 0;
}
#else
static int open_file_dialog(char *p, int n) { (void)p; (void)n; return 0; }
static int save_file_dialog(char *p, int n) { (void)p; (void)n; return 0; }
#endif

/* ============================================================
 * MAIN
 * ========================================================== */
static void print_help(void) {
    printf(
"FEM Steel Sim — Real-time finite-element elastic solid simulator\n"
"\n"
"Usage: interactive_steel_sim [-h | --help]\n"
"\n"
"  -h, --help    Print this help message and exit.\n"
"\n"
"=== MODES ====================================================================\n"
"  The program has two input modes, toggled with Tab:\n"
"\n"
"  CURSOR MODE (default on startup)\n"
"    The mouse cursor is free.  Interact with solid geometry and paint\n"
"    constraints.  Switch tools with the scroll wheel.\n"
"\n"
"  CAMERA MODE\n"
"    Mouse is captured for first-person look.  WASD moves the camera.\n"
"    Tab returns to CURSOR MODE.\n"
"\n"
"=== CAMERA CONTROLS (CAMERA MODE) ===========================================\n"
"  W / S           Forward / backward\n"
"  A / D           Strafe left / right\n"
"  Space           Move up\n"
"  Left Ctrl       Move down\n"
"  Mouse           Look (yaw / pitch)\n"
"  Tab             Switch to CURSOR MODE\n"
"\n"
"=== TOOLS (cycle with the scroll wheel) =====================================\n"
"  0 - None\n"
"  1 - Anchor Paint   Hold LMB: pin nodes so they cannot move (shown blue).\n"
"  2 - Force Paint    Hold LMB: apply a constant force in the camera-forward\n"
"                     direction to nodes under the cursor (shown red).\n"
"                     Adjust magnitude with the 'force mag' menu slider.\n"
"  3 - Clear Paint    Hold LMB: erase anchor or force paint from nodes.\n"
"  4 - Extrude Face   (default) Hover a face of a prism (highlighted yellow),\n"
"                     then drag LMB to push or pull that face:\n"
"                       Drag outward  → grow the solid (blue preview box)\n"
"                       Drag inward   → shrink the solid\n"
"                       Drag all the way through → drill a hole (red preview)\n"
"                     Shift+LMB adds more faces to the group before or during\n"
"                     a drag so all move together in one operation.\n"
"\n"
"=== KEYBOARD SHORTCUTS =======================================================\n"
"  Tab          Toggle CURSOR / CAMERA mode\n"
"  R            Reset simulation to rest shape (painted constraints kept)\n"
"  Shift+R      Rebuild mesh from current CSG (constraints reset)\n"
"  P            Pause / unpause physics\n"
"  E            Export scene (CSG + constraints) to .fes via save dialog\n"
"  O            Open / load a .fes scene file via open dialog\n"
"  1            Scalar colour: global displacement magnitude\n"
"  2            Scalar colour: Green-Lagrange strain ||E||_F\n"
"  3            Scalar colour: volumetric strain (blue/white/red)\n"
"  4            Scalar colour: accumulated plastic strain\n"
"  Escape       Quit\n"
"\n"
"=== CSG SOLID SYSTEM =========================================================\n"
"  The simulated object is described by a list of axis-aligned rectangular\n"
"  prisms (boxes) combined with Constructive Solid Geometry rules:\n"
"\n"
"    Additive prism   (sign +1)  Contributes material wherever it occupies\n"
"                                space in the scene.\n"
"    Subtractive prism (sign -1) Carves a void out of any additive prism\n"
"                                it overlaps.  Created automatically when you\n"
"                                drag the Extrude tool all the way through\n"
"                                an existing prism face.\n"
"\n"
"  The FEM tetrahedral mesh is rebuilt each time you press Shift+R or finish\n"
"  an extrude operation.  Every lattice cell of width h (set in the menu)\n"
"  that lies inside the CSG solid becomes two tetrahedra.\n"
"  Smaller h = finer mesh = higher accuracy but slower simulation.\n"
"\n"
"=== SCENE FILES (.fes) =======================================================\n"
"  Press E to open a save dialog and write a .fes file that stores the full\n"
"  CSG solid geometry plus every painted anchor and force constraint.\n"
"  Press O to reload a previously saved .fes scene at any time.\n"
"\n"
"=== MENU (left panel) ========================================================\n"
"  Drag sliders to change material and simulation parameters in real time:\n"
"    E* (Pa)           Young's modulus (stiffness)\n"
"    nu                Poisson's ratio\n"
"    density (kg/m3)   Material density\n"
"    dt (s)            Physics timestep per substep\n"
"    substeps          Physics substeps per rendered frame\n"
"    gravity (m/s2)    Gravitational acceleration\n"
"    paint radius      Radius of the constraint-paint brush\n"
"    force mag         Force per node applied by Force Paint\n"
"    Plasticity on     Enable / disable elastic-plastic material model\n"
"    yield stress      Stress at which plastic flow begins (Pa)\n"
"    hardening H       Isotropic strain-hardening modulus (Pa)\n"
"    fracture strain   Accumulated plastic strain at fracture\n"
);
}

int main(int argc, char **argv) {
    /* ---- command-line arguments --------------------------- */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help();
            return 0;
        }
    }

    /* ---- SDL + GL context --------------------------------- */
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError()); return 1;
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

    SDL_Window *win = SDL_CreateWindow("FEM Steel Sim",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIN_W, WIN_H, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (!win) { fprintf(stderr,"SDL_CreateWindow: %s\n",SDL_GetError()); return 1; }

    SDL_GLContext gl_ctx = SDL_GL_CreateContext(win);
    SDL_GL_SetSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLenum glerr = glewInit();
    if (glerr != GLEW_OK) {
        fprintf(stderr,"GLEW: %s\n",glewGetErrorString(glerr)); return 1;
    }
    if (!GLEW_VERSION_4_3) {
        fprintf(stderr,"OpenGL 4.3 required (compute shaders).\n"); return 1;
    }

    glEnable(GL_MULTISAMPLE);

    printf("Starting FEM Steel Sim\n");
    /* ---- initial CSG solid: unit cube --------------------- */
    CsgSolid *edit_solid = csg_solid_create();
    float mn[3]={0,0,0}, mx[3]={1,1,1};
    csg_solid_add_prism(edit_solid, mn, mx, +1);
    printf("Initial solid: %d prisms\n", edit_solid->n_prisms);

    float lattice_h = DEF_H;

    /* ---- generate mesh ------------------------------------ */
    FemMesh *mesh = fem_mesh_generate(csg_clone(edit_solid), lattice_h, (float)d_density);
    if (!mesh || mesh->m_tets==0) {
        fprintf(stderr,"Empty mesh — reduce lattice_h.\n"); return 1;
    }
    printf("Mesh: %d nodes, %d tets, %d surf_tri\n",
           mesh->n_nodes, mesh->m_tets, mesh->n_surf_tri);

    /* ---- Lamé parameters ---------------------------------- */
    float mu     = (float)(d_E_star / (2.0*(1.0+d_nu)));
    float lambda = (float)(d_E_star * d_nu / ((1.0+d_nu)*(1.0-2.0*d_nu)));

    /* ---- create GPU sim ----------------------------------- */
    FemSimGPU *sim = fem_sim_create(mesh, mu, lambda);
    if (!sim) { fprintf(stderr,"fem_sim_create failed.\n"); return 1; }

    /* ---- default constraints (anchor bottom face) --------- */
    constraints_default(mesh);
    fem_sim_upload_constraints(sim, g_anchor, g_ext_force);

    /* ---- initial surface normals -------------------------- */
    fem_sim_update_normals(sim);

    /* ---- menu --------------------------------------------- */
    Menu *menu = build_menu();

    /* ---- notification HUD --------------------------------- */
    char   hud_msg[256] = {0};
    Uint32 hud_msg_expire = 0;

    /* ---- camera state ------------------------------------- */
    float cam_pos[3] = {0.5f, 1.5f, 3.0f};
    float cam_yaw    = 0.0f;
    float cam_pitch  = -0.3f;
    SDL_SetRelativeMouseMode(SDL_FALSE);

    /* ---- state: separate pause from mouse lock ----------- */
    int paused      = 0;     /* physics paused               */
    int mouse_locked= 0;     /* start in cursor mode         */

    /* ---- edit state: hover + multi-face selection + drag - */
    int hover_prism = -1;    /* prism index under cursor     */
    int hover_ax    = -1;    /* face axis (0-2) under cursor */
    int hover_si    = 0;     /* face side (-1/+1) under cursor */
    SelectedFace sel_faces[MAX_SEL_FACES];
    int   n_sel_faces  = 0;  /* number of faces in selection */
    int   sel_ax       = -1; /* axis shared by all selected faces */
    int   sel_si       = 0;  /* side shared by all selected faces */
    int   edit_drag      = 0;
    float edit_drag_accum = 0.0f;  /* master accumulated drag metres */

    /* ---- paint mode --------------------------------------- */
    /* tool: 0=none,1=anchor,2=force,3=clear,4=extrude */
    int active_tool = 4;
    int lmb_down    = 0;
    static const char *tool_names[] = {
        "None", "Anchor Paint", "Force Paint", "Clear Paint", "Extrude Face"
    };
    (void)tool_names; /* used in HUD below */

    /* ---- GL state ----------------------------------------- */
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE); glCullFace(GL_BACK);

    /* ---- projection & view matrices ----------------------- */
    float P[16], V[16], MVP[16];
    float aspect = (float)WIN_W / (float)WIN_H;
    m4_perspective(FOV_Y, aspect, NEAR_Z, FAR_Z, P);

    /* ---- timing ------------------------------------------- */
    Uint32 last_time = SDL_GetTicks();
    int    frame_count = 0;
    float  fps = 0.0f;
    char   fps_buf[32];

    /* ---- main loop ---------------------------------------- */
    int running = 1;
    while (running) {
        Uint32 now = SDL_GetTicks();
        float dt_frame = (float)(now - last_time) * 1e-3f;
        if (dt_frame > 0.1f) dt_frame = 0.1f;  /* cap */
        last_time = now;
        frame_count++;
        if (frame_count % 30 == 0) {
            fps = 1.0f / (dt_frame > 0 ? dt_frame : 0.016f);
            snprintf(fps_buf, sizeof(fps_buf), "FPS: %.0f", fps);
        }

        /* ---- update Lamé from menu sliders; always update sim params - */
        sim->lame_mu     = (float)(d_E_star / (2.0*(1.0+d_nu)));
        sim->lame_lambda = (float)(d_E_star * d_nu / ((1.0+d_nu)*(1.0-2.0*d_nu)));
        sim->dt          = paused ? 0.0f : (float)d_dt;  /* zero dt when paused */
        sim->gravity     = (float)d_gravity;
        sim->color_mode  = d_colorize ? 1 : 0;
        sim->show_slice  = d_show_slice ? 1 : 0;
        sim->scalar_max         = (float)d_scalar_max;
        sim->yield_stress        = (float)d_yield_stress;
        sim->hardening_mod       = (float)d_hardening_mod;
        sim->fracture_strain     = (float)d_fracture_strain;
        sim->plasticity_enabled  = d_plasticity_on ? 1 : 0;
        sim->slice_pt[0] = (float)d_slice_x;
        sim->slice_pt[1] = (float)d_slice_y;
        sim->slice_pt[2] = (float)d_slice_z;
        sim->slice_h     = (float)d_slice_h;
        cam_dirs((float)d_slice_yaw, (float)d_slice_pitch, sim->slice_nrm, NULL, NULL);

        /* ---- event handling ------------------------------- */
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            /* Pass to menu first */
            int menu_consumed = 0;
            if (ev.type == SDL_MOUSEBUTTONDOWN)
                menu_consumed = menu_handle_mouse_button(menu,
                    ev.button.button, SDL_PRESSED,
                    ev.button.x, ev.button.y);
            else if (ev.type == SDL_MOUSEBUTTONUP)
                menu_consumed = menu_handle_mouse_button(menu,
                    ev.button.button, SDL_RELEASED,
                    ev.button.x, ev.button.y);
            else if (ev.type == SDL_MOUSEMOTION)
                menu_consumed = menu_handle_mouse_motion(menu,
                    ev.motion.x, ev.motion.y);

            if (menu_consumed) continue;

            if (ev.type == SDL_QUIT) { running=0; break; }

            if (ev.type == SDL_KEYDOWN) {
                const Uint8 *ks_state = SDL_GetKeyboardState(NULL);
                int shift_down = ks_state[SDL_SCANCODE_LSHIFT] || ks_state[SDL_SCANCODE_RSHIFT];
                
                switch (ev.key.keysym.sym) {
                case SDLK_ESCAPE: running=0; break;
                
                case SDLK_r:
                    if (shift_down) {
                        /* Shift+R: regenerate mesh and pause */
                        /* Rebuild mesh from edit_solid */
                        fem_sim_free(sim); sim=NULL;
                        fem_mesh_free(mesh); mesh=NULL;
                        mesh = fem_mesh_generate(csg_clone(edit_solid),
                                                 lattice_h, (float)d_density);
                        if (!mesh||mesh->m_tets==0) {
                            /* fallback to unit cube */
                            CsgSolid *fb=csg_solid_create();
                            float mn[3]={0,0,0},mx[3]={1,1,1};
                            csg_solid_add_prism(fb,mn,mx,+1);
                            mesh=fem_mesh_generate(fb,lattice_h,(float)d_density);
                        }
                        printf("Rebuilt: %d nodes, %d tets\n",
                               mesh->n_nodes, mesh->m_tets);
                        mu     =(float)(d_E_star/(2.0*(1.0+d_nu)));
                        lambda =(float)(d_E_star*d_nu/((1.0+d_nu)*(1.0-2.0*d_nu)));
                        sim = fem_sim_create(mesh, mu, lambda);
                        constraints_default(mesh);
                        fem_sim_upload_constraints(sim, g_anchor, g_ext_force);
                        fem_sim_update_normals(sim);
                        fem_sim_update_scalar(sim);
                        n_sel_faces=0; sel_ax=-1; sel_si=0;
                        hover_prism=-1; hover_ax=-1;
                        paused=1;
                    } else {
                        /* R alone: reset simulated shape to rest state,
                         * but keep painted anchors/forces intact */
                        fem_sim_reset(sim, mesh->pos);
                        fem_sim_upload_constraints(sim, g_anchor, g_ext_force);
                    }
                    break;
                
                case SDLK_p:
                    paused = !paused;
                    break;

                case SDLK_1:
                    if (sim) sim->scalar_mode = 0;
                    break;

                case SDLK_2:
                    if (sim) sim->scalar_mode = 1;
                    break;

                case SDLK_3:
                    if (sim) sim->scalar_mode = 2;
                    break;

                case SDLK_4:
                    if (sim) sim->scalar_mode = 3;
                    break;

                case SDLK_e: {
                    /* ---- Export scene (CSG + painted constraints) ---- */
                    int was_locked = mouse_locked;
                    if (was_locked) SDL_SetRelativeMouseMode(SDL_FALSE);
                    char epath[MAX_PATH];
                    if (save_file_dialog(epath, MAX_PATH)) {
                        if (save_scene(epath, edit_solid, lattice_h,
                                       mesh->n_nodes, g_anchor, g_ext_force))
                            snprintf(hud_msg, sizeof(hud_msg), "Saved: %s", epath);
                        else
                            snprintf(hud_msg, sizeof(hud_msg), "Save failed!");
                        hud_msg_expire = SDL_GetTicks() + 3000;
                    }
                    if (was_locked) SDL_SetRelativeMouseMode(SDL_TRUE);
                    break;
                }

                case SDLK_o: {
                    /* ---- Open / load scene ---- */
                    int was_locked = mouse_locked;
                    if (was_locked) SDL_SetRelativeMouseMode(SDL_FALSE);
                    char opath[MAX_PATH];
                    if (open_file_dialog(opath, MAX_PATH)) {
                        CsgSolid     *ls  = NULL;
                        float         llh = 0.0f;
                        int           ln  = 0;
                        unsigned int *la  = NULL;
                        float        *lf  = NULL;
                        if (load_scene(opath, &ls, &llh, &ln, &la, &lf)) {
                            fem_sim_free(sim);   sim  = NULL;
                            fem_mesh_free(mesh); mesh = NULL;
                            csg_solid_free(edit_solid);
                            edit_solid = ls;
                            lattice_h  = llh;
                            mesh = fem_mesh_generate(csg_clone(edit_solid),
                                                     lattice_h, (float)d_density);
                            if (!mesh || mesh->m_tets == 0) {
                                /* fallback — loaded CSG produced empty mesh */
                                csg_solid_free(edit_solid);
                                CsgSolid *fb = csg_solid_create();
                                float fmn[3]={0,0,0}, fmx[3]={1,1,1};
                                csg_solid_add_prism(fb, fmn, fmx, +1);
                                edit_solid = fb;
                                mesh = fem_mesh_generate(csg_clone(edit_solid),
                                                         lattice_h, (float)d_density);
                                snprintf(hud_msg, sizeof(hud_msg),
                                         "Loaded CSG produced empty mesh!");
                            } else {
                                constraints_clear(mesh->n_nodes);
                                int cn = (ln < mesh->n_nodes) ? ln : mesh->n_nodes;
                                memcpy(g_anchor,    la, (size_t)cn * sizeof(unsigned int));
                                memcpy(g_ext_force, lf, (size_t)cn * 3 * sizeof(float));
                                snprintf(hud_msg, sizeof(hud_msg), "Loaded: %s", opath);
                            }
                            free(la); free(lf);
                            mu     = (float)(d_E_star / (2.0*(1.0+d_nu)));
                            lambda = (float)(d_E_star * d_nu / ((1.0+d_nu)*(1.0-2.0*d_nu)));
                            sim = fem_sim_create(mesh, mu, lambda);
                            fem_sim_upload_constraints(sim, g_anchor, g_ext_force);
                            fem_sim_update_normals(sim);
                            fem_sim_update_scalar(sim);
                            n_sel_faces = 0; sel_ax = -1; sel_si = 0;
                            hover_prism = -1; hover_ax = -1;
                            paused = 1;
                        } else {
                            snprintf(hud_msg, sizeof(hud_msg),
                                     "Failed to parse scene file!");
                        }
                        hud_msg_expire = SDL_GetTicks() + 3000;
                    }
                    if (was_locked) SDL_SetRelativeMouseMode(SDL_TRUE);
                    break;
                }

                case SDLK_TAB:
                    mouse_locked = !mouse_locked;
                    SDL_SetRelativeMouseMode(mouse_locked ? SDL_TRUE : SDL_FALSE);
                    edit_drag=0;
                    n_sel_faces=0; sel_ax=-1; sel_si=0;
                    break;
                
                default: break;
                }
            }

            /* ---- camera mouse look (when mouse locked) ------- */
            if (mouse_locked && ev.type == SDL_MOUSEMOTION) {
                cam_yaw   += ev.motion.xrel * CAM_SENSITIVITY;
                cam_pitch += ev.motion.yrel * CAM_SENSITIVITY;
                float lim = (float)(M_PI*0.5-0.05);
                if (cam_pitch > lim)  cam_pitch= lim;
                if (cam_pitch <-lim)  cam_pitch=-lim;
            }

            /* ---- scroll wheel: cycle active tool ---------- */
            if (ev.type == SDL_MOUSEWHEEL) {
                int dir = (ev.wheel.y > 0) ? 1 : -1;
                active_tool = (active_tool + 5 + dir) % 5;
                lmb_down = 0; /* cancel any active paint on tool switch */
                n_sel_faces = 0; sel_ax = -1; sel_si = 0;
                edit_drag = 0;
            }

            /* ---- LMB tracking for paint ------------------- */
            if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_LEFT)
                lmb_down = 1;
            if (ev.type == SDL_MOUSEBUTTONUP   && ev.button.button == SDL_BUTTON_LEFT)
                lmb_down = 0;

            /* ---- face selection + drag (active_tool==4, cursor mode) ----------
             *
             * Pre-drag:
             *   Shift+LMB  — toggle hovered face in/out of selection group.
             *                 All faces in a group share the same (ax, si) normal.
             *   LMB        — start drag on current selection (or implicitly select
             *                 the hovered face if nothing is pre-selected yet).
             *
             * During drag:
             *   Shift+LMB  — add the currently hovered face to the group at its
             *                 current join point; it moves from 0 relative to now
             *                 so no discontinuous jump occurs.
             *
             * Each face tracks its own join_accum.  Displacement for face i =
             *   edit_drag_accum - face.join_accum
             * This lets faces added mid-drag lag behind by exactly their offset,
             * enabling the pyramid-flattening "chain de-extension" use case.
             *
             * On release: each face is committed independently (inward = replace
             * + cascade, outward = add new prism).  Selection is cleared.
             * ------------------------------------------------------------------ */
            if (active_tool == 4 && !mouse_locked) {
                int shift_held = (SDL_GetModState() & KMOD_SHIFT) != 0;

                /* --- Shift+LMB: toggle face selection (works pre-drag AND mid-drag) */
                if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button==SDL_BUTTON_LEFT
                    && shift_held && hover_prism >= 0) {
                    int same_nrm = (n_sel_faces == 0)
                                || (sel_ax == hover_ax && sel_si == hover_si);
                    if (same_nrm) {
                        if (n_sel_faces == 0) { sel_ax = hover_ax; sel_si = hover_si; }
                        /* Toggle: remove if already present */
                        int found = -1;
                        for (int fi = 0; fi < n_sel_faces; fi++)
                            if (sel_faces[fi].prism_idx == hover_prism) { found=fi; break; }
                        if (found >= 0) {
                            sel_faces[found] = sel_faces[--n_sel_faces];
                            if (n_sel_faces == 0) { sel_ax=-1; sel_si=0; }
                        } else if (n_sel_faces < MAX_SEL_FACES) {
                            sel_faces[n_sel_faces].prism_idx = hover_prism;
                            sel_faces[n_sel_faces].ax        = hover_ax;
                            sel_faces[n_sel_faces].si        = hover_si;
                            /* join_accum captures current drag position so face
                             * starts with zero relative displacement (no jump) */
                            sel_faces[n_sel_faces].join_accum = edit_drag_accum;
                            n_sel_faces++;
                        }
                    }
                    lmb_down = 0;
                }

                /* --- LMB (no shift): start drag on current selection */
                if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button==SDL_BUTTON_LEFT
                    && !shift_held && !edit_drag) {
                    /* If nothing pre-selected, implicitly select hovered face */
                    if (n_sel_faces == 0 && hover_prism >= 0) {
                        sel_faces[0].prism_idx  = hover_prism;
                        sel_faces[0].ax         = hover_ax;
                        sel_faces[0].si         = hover_si;
                        sel_faces[0].join_accum = 0.0f;
                        n_sel_faces = 1;
                        sel_ax = hover_ax; sel_si = hover_si;
                    }
                    if (n_sel_faces > 0) {
                        edit_drag = 1;
                        edit_drag_accum = 0.0f;
                        /* All pre-selected faces start at join_accum 0 */
                        for (int fi = 0; fi < n_sel_faces; fi++)
                            sel_faces[fi].join_accum = 0.0f;
                        lmb_down = 0;
                    }
                }

                /* --- Mouse-up: commit every selected face then clear selection */
                if (ev.type == SDL_MOUSEBUTTONUP && ev.button.button==SDL_BUTTON_LEFT
                    && edit_drag) {
                    /* Collect drill-through base prism indices for deletion after loop */
                    int drill_del[MAX_SEL_FACES];
                    int n_drill_del = 0;
                    for (int fi = 0; fi < n_sel_faces; fi++) {
                        SelectedFace *sf = &sel_faces[fi];
                        float delta   = edit_drag_accum - sf->join_accum;
                        float snapped = roundf(delta / lattice_h) * lattice_h;
                        if (fabsf(snapped) < lattice_h * 0.5f) continue;
                        RectPrism preview;
                        if (!build_face_extrusion(&edit_solid->prisms[sf->prism_idx],
                                                  sf->ax, sf->si, snapped, &preview))
                            continue;
                        if (snapped < 0.0f && preview.sign == -1) {
                            /* Drill-through: schedule base prism for deletion,
                             * add subtractive prism */
                            drill_del[n_drill_del++] = sf->prism_idx;
                            csg_solid_add_prism(edit_solid,
                                                preview.min, preview.max, -1);
                        } else if (snapped < 0.0f) {
                            /* Normal inward shrink: replace prism in-place + cascade */
                            int ax = sf->ax, si = sf->si;
                            float old_plane = si > 0
                                ? edit_solid->prisms[sf->prism_idx].max[ax]
                                : edit_solid->prisms[sf->prism_idx].min[ax];
                            float new_plane = si > 0
                                ? preview.max[ax] : preview.min[ax];
                            edit_solid->prisms[sf->prism_idx] = preview;
                            for (int pi = 0; pi < edit_solid->n_prisms; pi++) {
                                if (pi == sf->prism_idx) continue;
                                RectPrism *pr = &edit_solid->prisms[pi];
                                float *adj = (si > 0) ? &pr->min[ax] : &pr->max[ax];
                                if (fabsf(*adj - old_plane) < 1e-4f)
                                    *adj = new_plane;
                            }
                        } else {
                            /* Outward: append new prism */
                            csg_solid_add_prism(edit_solid,
                                                preview.min, preview.max, preview.sign);
                        }
                    }
                    /* Remove drilled-through base prisms (descending order preserves indices) */
                    for (int di = n_drill_del - 1; di >= 0; di--) {
                        int idx = drill_del[di];
                        /* Swap with last and decrement (order-irrelevant for CSG) */
                        edit_solid->prisms[idx] =
                            edit_solid->prisms[--edit_solid->n_prisms];
                    }
                    edit_drag = 0;
                    n_sel_faces = 0; sel_ax = -1; sel_si = 0;
                }

                /* --- Motion: accumulate drag using group normal direction */
                if (edit_drag && ev.type == SDL_MOUSEMOTION && n_sel_faces > 0) {
                    float fwd_c[3], right_c[3], up_c[3];
                    cam_dirs(cam_yaw, cam_pitch, fwd_c, right_c, up_c);
                    float nrm[3] = {0.0f, 0.0f, 0.0f};
                    nrm[sel_ax] = (float)sel_si;
                    float cam_nx = right_c[0]*nrm[0] + right_c[1]*nrm[1] + right_c[2]*nrm[2];
                    float cam_ny = up_c[0]*nrm[0]    + up_c[1]*nrm[1]    + up_c[2]*nrm[2];
                    float mag = sqrtf(cam_nx*cam_nx + cam_ny*cam_ny);
                    float nx_s, ny_s;
                    if (mag > 0.05f) { nx_s = cam_nx/mag; ny_s = cam_ny/mag; }
                    else             { nx_s = 1.0f;       ny_s = 0.0f;      }
                    float delta_px = (float)ev.motion.xrel * nx_s
                                   - (float)ev.motion.yrel * ny_s;
                    edit_drag_accum += delta_px * 0.005f;
                }
            } else {
                /* Not extrude tool — cancel any stale drag */
                if (!mouse_locked) edit_drag = 0;
            }
        } /* SDL_PollEvent */

        /* ---- per-frame paint (LMB held, paint tool active) */
        if (lmb_down && active_tool >= 1 && active_tool <= 3) {
            float ro[3], rd[3];
            float sx_p = 0.0f, sy_p = 0.0f;
            if (!mouse_locked) {
                int mx_px, my_px;
                SDL_GetMouseState(&mx_px, &my_px);
                sx_p = 2.0f*(float)mx_px/(float)WIN_W - 1.0f;
                sy_p = 1.0f - 2.0f*(float)my_px/(float)WIN_H;
            }
            screen_ray(cam_yaw, cam_pitch,
                       cam_pos[0], cam_pos[1], cam_pos[2],
                       sx_p, sy_p, aspect, ro, rd);
            paint_near(mesh, ro, rd, (float)d_pradius, active_tool - 1);
            fem_sim_upload_constraints(sim, g_anchor, g_ext_force);
        }

        /* ---- keyboard movement (always allowed) --------- */
        {
            const Uint8 *ks = SDL_GetKeyboardState(NULL);
            float fwd[3], right[3];
            cam_dirs(cam_yaw, cam_pitch, fwd, right, NULL);
            float spd = CAM_SPEED * dt_frame;
            if (ks[SDL_SCANCODE_W]){ cam_pos[0]+=fwd[0]*spd; cam_pos[1]+=fwd[1]*spd; cam_pos[2]+=fwd[2]*spd; }
            if (ks[SDL_SCANCODE_S]){ cam_pos[0]-=fwd[0]*spd; cam_pos[1]-=fwd[1]*spd; cam_pos[2]-=fwd[2]*spd; }
            if (ks[SDL_SCANCODE_A]){ cam_pos[0]-=right[0]*spd; cam_pos[1]-=right[1]*spd; cam_pos[2]-=right[2]*spd; }
            if (ks[SDL_SCANCODE_D]){ cam_pos[0]+=right[0]*spd; cam_pos[1]+=right[1]*spd; cam_pos[2]+=right[2]*spd; }
            if (ks[SDL_SCANCODE_SPACE])     cam_pos[1]+=spd;
            if (ks[SDL_SCANCODE_LCTRL])     cam_pos[1]-=spd;
        }

        /* ---- physics substeps (when not paused) --------- */
        if (!paused) {
            int ns = (int)(d_nsteps + 0.5);
            if (ns < 1)  ns = 1;
            if (ns > 300) ns = 300;
            for (int s=0; s<ns; s++)
                fem_sim_substep(sim);
            fem_sim_update_normals(sim);
        }
        fem_sim_update_scalar(sim);

        /* ---- face picking: update hover face under cursor --- */
        /* Runs every frame in cursor mode — even during drag so Shift+click
         * mid-drag can add the currently hovered face to the selection. */
        if (!mouse_locked) {
            int mx_px, my_px;
            SDL_GetMouseState(&mx_px, &my_px);
            float sx = 2.0f*(float)mx_px/(float)WIN_W - 1.0f;
            float sy = 1.0f - 2.0f*(float)my_px/(float)WIN_H;
            float ro[3], rd[3];
            screen_ray(cam_yaw, cam_pitch,
                       cam_pos[0], cam_pos[1], cam_pos[2],
                       sx, sy, aspect, ro, rd);

            hover_prism=-1; hover_ax=-1;
            float best_t=1e30f;
            for (int pi=0; pi<edit_solid->n_prisms; pi++) {
                RectPrism *pr=&edit_solid->prisms[pi];
                float t; int ax, si;
                if (ray_prism_face(ro,rd,pr->min,pr->max,&t,&ax,&si) && t<best_t) {
                    best_t=t; hover_prism=pi; hover_ax=ax; hover_si=si;
                }
            }
        }

        /* ---- build view / MVP ----------------------------- */
        m4_view(cam_yaw, cam_pitch,
                cam_pos[0], cam_pos[1], cam_pos[2], V);
        m4_mul(P, V, MVP);

        /* ---- clear ---------------------------------------- */
        glClearColor(0.07f, 0.07f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        /* ---- render surface mesh (simulate or edit) ------- */
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        static const float ld[3] = {0.577f, 0.816f, 0.577f};  /* normalised */
        static const float mc[3] = {0.70f, 0.72f, 0.75f};
        fem_sim_render(sim, MVP, V, ld, mc);
        if (sim->show_slice) {
            glDisable(GL_CULL_FACE);
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            fem_sim_render_slice(sim, MVP);
            glDisable(GL_BLEND);
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_CULL_FACE);
        }

        /* ---- constraint visualization (always visible) --- */
        {
            glDisable(GL_CULL_FACE);
            glDisable(GL_DEPTH_TEST);
            glUseProgram(0);
            glMatrixMode(GL_PROJECTION); glLoadMatrixf(P);
            glMatrixMode(GL_MODELVIEW);  glLoadMatrixf(V);
            draw_constraints(mesh);
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_CULL_FACE);
        }

        /* ---- geometry edit overlays (when mouse unlocked) - */
        if (!mouse_locked) {
            glDisable(GL_CULL_FACE);
            glUseProgram(0);
            glMatrixMode(GL_PROJECTION); glLoadMatrixf(P);
            glMatrixMode(GL_MODELVIEW);  glLoadMatrixf(V);

            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glLineWidth(2.0f);
            glPolygonOffset(-1,-1); glEnable(GL_POLYGON_OFFSET_LINE);

            /* Wireframe for each prism; brighten if selected or hovered */
            for (int pi=0; pi<edit_solid->n_prisms; pi++) {
                RectPrism *pr=&edit_solid->prisms[pi];
                int in_sel = 0;
                for (int fi=0; fi<n_sel_faces; fi++)
                    if (sel_faces[fi].prism_idx == pi) { in_sel=1; break; }
                int is_hov = (pi == hover_prism);
                float wb = (in_sel || is_hov) ? 1.0f : 0.3f;
                float wg = in_sel ? 1.0f : is_hov ? 1.0f : 0.6f;
                draw_aabb_wire(pr->min, pr->max, wb, wg, 0.2f);
            }

            /* Cyan face highlight for every selected face */
            for (int fi=0; fi<n_sel_faces; fi++) {
                SelectedFace *sf=&sel_faces[fi];
                RectPrism *pr=&edit_solid->prisms[sf->prism_idx];
                draw_face_highlight_color(pr->min, pr->max, sf->ax, sf->si,
                                          0.0f, 0.9f, 1.0f); /* cyan */
            }

            /* Yellow hover highlight (only if face not already in selection) */
            if (hover_prism >= 0 && hover_ax >= 0) {
                int hov_sel = 0;
                for (int fi=0; fi<n_sel_faces; fi++)
                    if (sel_faces[fi].prism_idx == hover_prism) { hov_sel=1; break; }
                if (!hov_sel)
                    draw_face_highlight(edit_solid->prisms[hover_prism].min,
                                        edit_solid->prisms[hover_prism].max,
                                        hover_ax, hover_si);
            }

            /* Per-face preview wireframes during drag */
            if (edit_drag) {
                for (int fi=0; fi<n_sel_faces; fi++) {
                    SelectedFace *sf=&sel_faces[fi];
                    float delta   = edit_drag_accum - sf->join_accum;
                    float snapped = roundf(delta / lattice_h) * lattice_h;
                    if (fabsf(snapped) < lattice_h * 0.5f) continue;
                    RectPrism prev;
                    if (build_face_extrusion(&edit_solid->prisms[sf->prism_idx],
                                             sf->ax, sf->si, snapped, &prev)) {
                        if (prev.sign == -1)
                            draw_aabb_wire(prev.min, prev.max, 1.0f, 0.15f, 0.15f); /* red = subtraction */
                        else
                            draw_aabb_wire(prev.min, prev.max, 0.2f, 0.8f, 1.0f);  /* blue = addition/shrink */
                    }
                }
            }

            glDisable(GL_BLEND);
            glDisable(GL_POLYGON_OFFSET_LINE);
            glEnable(GL_CULL_FACE);
        }

        /* ---- 2D HUD --------------------------------------- */
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);

        /* menu (handles its own ortho projection) */
        menu_render(menu, WIN_W, WIN_H);

        /* mode label + FPS */
        {
            Color wh={255,255,255,255};
            Color yw={255,220,0,255};
            Color cy_={0,200,255,255};
            Color rd_={255,100,80,255};
            menu_draw_text_at(fps_buf, WIN_W-90, 8, wh);

            /* scalar mode indicator (top-left) */
            {
                const char *smode_name =
                    (sim && sim->scalar_mode == 3) ? "Scalar[1/2/3/4]: Plastic Strain" :
                    (sim && sim->scalar_mode == 2) ? "Scalar[1/2/3/4]: Vol.Strain (B/W/R)" :
                    (sim && sim->scalar_mode == 1) ? "Scalar[1/2/3/4]: ||E||_F (GL strain)" :
                                                     "Scalar[1/2/3/4]: Global Disp";
                menu_draw_text_at(smode_name, 10, 8, wh);
            }

            /* tool indicator bottom-left */
            char tool_buf[80];
            if (active_tool == 4 && n_sel_faces > 0)
                snprintf(tool_buf, sizeof(tool_buf),
                         "Tool [Scroll]: %s  [%d face%s — Shift+click adds]",
                         tool_names[active_tool], n_sel_faces,
                         n_sel_faces == 1 ? "" : "s");
            else
                snprintf(tool_buf, sizeof(tool_buf),
                         "Tool [Scroll]: %s", tool_names[active_tool]);
            Color tool_col = (active_tool==1) ? cy_ :
                             (active_tool==2) ? rd_ :
                             (active_tool==3) ? yw  : wh;
            menu_draw_text_at(tool_buf, 10, WIN_H-30, tool_col);

            /* status line */
            if (paused) {
                menu_draw_text_at(
                    "PAUSED  P=unpause  Shift+R=rebuild  Tab=cursor  R=reset  E=export  O=open",
                    10, WIN_H-54, yw);
            } else if (!mouse_locked) {
                menu_draw_text_at(
                    "CURSOR MODE  Tab=camera  Scroll=tool  Shift+click=multi-select  LMB=drag  E=export  O=open",
                    10, WIN_H-54, wh);
            } else {
                menu_draw_text_at(
                    "CAMERA  Tab=cursor  Scroll=tool  LMB=paint  P=pause  R=reset  E=export  O=open",
                    10, WIN_H-54, wh);
            }

            /* legend bottom-right */
            Color blu={100,130,255,255};
            Color dim={160,160,160,200};
            menu_draw_text_at("E=export", WIN_W-110, WIN_H-70, dim);
            menu_draw_text_at("O=open",   WIN_W-110, WIN_H-54, dim);
            menu_draw_text_at("Anchor",   WIN_W-90,  WIN_H-38, blu);
            menu_draw_text_at("Force",    WIN_W-90,  WIN_H-22, rd_);

            /* crosshair (camera mode only) */
            if (mouse_locked) {
                int cxp=WIN_W/2, cyp=WIN_H/2;
                Color gr={200,200,200,200};
                menu_draw_text_at("+", cxp-4, cyp-6, gr);
            }

            /* timed notification message (E=save / O=load feedback) */
            if (hud_msg[0] && SDL_GetTicks() < hud_msg_expire) {
                Color ng = {160, 240, 160, 255};
                menu_draw_text_at(hud_msg, 10, 26, ng);
            }
        }

        /* ---- custom cursor (cursor mode only) ------------- */
        if (!mouse_locked) {
            SDL_ShowCursor(SDL_DISABLE);  /* hide system cursor */
            int mx_px, my_px;
            SDL_GetMouseState(&mx_px, &my_px);
            /* per-tool colour */
            float tc[3];
            if      (active_tool==1){ tc[0]=0.20f; tc[1]=0.45f; tc[2]=1.00f; } /* blue   anchor */
            else if (active_tool==2){ tc[0]=1.00f; tc[1]=0.30f; tc[2]=0.15f; } /* red    force  */
            else if (active_tool==3){ tc[0]=1.00f; tc[1]=0.87f; tc[2]=0.00f; } /* yellow clear  */
            else if (active_tool==4){ tc[0]=0.20f; tc[1]=1.00f; tc[2]=0.30f; } /* green  extrude*/
            else                    { tc[0]=0.80f; tc[1]=0.80f; tc[2]=0.80f; } /* grey   none   */
            /* paint-ring radius: world-space pradius * rough pixels-per-metre */
            int ring_r = (int)(d_pradius * 120.0);
            if (ring_r < 8)  ring_r = 8;
            if (ring_r > 80) ring_r = 80;
            draw_cursor_2d(mx_px, my_px, tc[0], tc[1], tc[2],
                           active_tool, ring_r, WIN_W, WIN_H);
        } else {
            SDL_ShowCursor(SDL_ENABLE);   /* restore when back in camera mode */
        }

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);

        SDL_GL_SwapWindow(win);
    }

    /* ---- cleanup ------------------------------------------ */
    /* ---- cleanup ------------------------------------------ */
    menu_clear_font();
    menu_free(menu);
    if (sim)  fem_sim_free(sim);
    if (mesh) fem_mesh_free(mesh);
    csg_solid_free(edit_solid);
    SDL_GL_DeleteContext(gl_ctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
