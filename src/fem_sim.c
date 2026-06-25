/*
 * fem_sim.c — GPU FEM simulation using GL 4.3 compute shaders.
 *
 * Physics pipeline per substep:
 *   [gather]     pos_ssbo → x_ssbo[0..3]   (forward structure matrices)
 *   [tet_force]  x_ssbo + D0inv + V0 → f_ssbo[0..3]
 *   [scatter]    f_ssbo + adj CSR → force_ssbo   (SpMV, 1 thread/node)
 *   [ext_force]  force_ssbo += ext_force + gravity; zero anchors
 *   [integrate]  vel += F/m*dt; pos += vel*dt
 *
 * All physics SSBOs hold 32-bit floats (not double).
 */

#include "../include/fem_sim.h"
#include "../include/fem_mesh.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===========================================================
 * Internal: compile / link helpers
 * =========================================================== */
static GLuint compile_cs(const char *src) {
    GLuint s = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096]; glGetShaderInfoLog(s, sizeof(log), NULL, log);
        fprintf(stderr, "[fem_sim] CS compile error:\n%s\n---\n%s\n", log, src);
        glDeleteShader(s); return 0;
    }
    return s;
}
static GLuint compile_vs(const char *src) {
    GLuint s = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; glGetShaderInfoLog(s,sizeof(log),NULL,log);
        fprintf(stderr,"[fem_sim] VS error:\n%s\n",log); glDeleteShader(s); return 0; }
    return s;
}
static GLuint compile_fs(const char *src) {
    GLuint s = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; glGetShaderInfoLog(s,sizeof(log),NULL,log);
        fprintf(stderr,"[fem_sim] FS error:\n%s\n",log); glDeleteShader(s); return 0; }
    return s;
}
static GLuint link_compute(GLuint cs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, cs); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p,sizeof(log),NULL,log);
        fprintf(stderr,"[fem_sim] CS link error:\n%s\n",log); glDeleteProgram(p); return 0; }
    glDeleteShader(cs); return p;
}
static GLuint link_render(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p,vs); glAttachShader(p,fs); glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p,sizeof(log),NULL,log);
        fprintf(stderr,"[fem_sim] render link error:\n%s\n",log); glDeleteProgram(p); return 0; }
    glDeleteShader(vs); glDeleteShader(fs); return p;
}

static GLuint make_ssbo(const void *data, size_t bytes, GLenum usage) {
    GLuint buf; glGenBuffers(1, &buf);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)bytes, data, usage);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return buf;
}

/* ===========================================================
 * GLSL source strings
 * =========================================================== */

/* ---- gather: forward structure matrices ---- */
static const char *SRC_GATHER =
"#version 430 core\n"
"layout(local_size_x=64,local_size_y=1,local_size_z=1) in;\n"
"layout(std430,binding=0) readonly  buffer BufPos { float pos[]; };\n"
"layout(std430,binding=1) readonly  buffer BufC0  { uint  ci0[]; };\n"
"layout(std430,binding=2) readonly  buffer BufC1  { uint  ci1[]; };\n"
"layout(std430,binding=3) readonly  buffer BufC2  { uint  ci2[]; };\n"
"layout(std430,binding=4) readonly  buffer BufC3  { uint  ci3[]; };\n"
"layout(std430,binding=5) writeonly buffer BufX0  { float x0[];  };\n"
"layout(std430,binding=6) writeonly buffer BufX1  { float x1[];  };\n"
"layout(std430,binding=7) writeonly buffer BufX2  { float x2[];  };\n"
"layout(std430,binding=8) writeonly buffer BufX3  { float x3[];  };\n"
"uniform uint m;\n"
"void main(){\n"
"  uint j=gl_GlobalInvocationID.x;\n"
"  if(j>=m)return;\n"
"  uint n0=ci0[j],n1=ci1[j],n2=ci2[j],n3=ci3[j];\n"
"  x0[j*3u]=pos[n0*3u]; x0[j*3u+1u]=pos[n0*3u+1u]; x0[j*3u+2u]=pos[n0*3u+2u];\n"
"  x1[j*3u]=pos[n1*3u]; x1[j*3u+1u]=pos[n1*3u+1u]; x1[j*3u+2u]=pos[n1*3u+2u];\n"
"  x2[j*3u]=pos[n2*3u]; x2[j*3u+1u]=pos[n2*3u+1u]; x2[j*3u+2u]=pos[n2*3u+2u];\n"
"  x3[j*3u]=pos[n3*3u]; x3[j*3u+1u]=pos[n3*3u+1u]; x3[j*3u+2u]=pos[n3*3u+2u];\n"
"}\n";

/* ---- tet_force: per-tet FEM physics with additive plasticity + fracture ---- *
 * Additive split: E_total = E_e + E_p  (Green-Lagrange)                        *
 * Radial return mapping onto von Mises yield surface per substep.               *
 * Fracture: when acc_p[j] >= fracture_strain, tet goes zero-force (broken).    *
 *                                                                                *
 * Plastic strain Voigt layout (6 floats per tet):                               *
 *   [0]=E_00  [1]=E_11  [2]=E_22  [3]=E_01  [4]=E_02  [5]=E_12                *
 * (symmetric, stored in GLSL column-major mat3 as cols (e00,e01,e02),          *
 *  (e01,e11,e12), (e02,e12,e22))                                                *
 * ============================================================================ */
static const char *SRC_TET_FORCE =
"#version 430 core\n"
"layout(local_size_x=64,local_size_y=1,local_size_z=1) in;\n"
"layout(std430,binding=0)  readonly buffer BufX0    { float x0[];    };\n"
"layout(std430,binding=1)  readonly buffer BufX1    { float x1[];    };\n"
"layout(std430,binding=2)  readonly buffer BufX2    { float x2[];    };\n"
"layout(std430,binding=3)  readonly buffer BufX3    { float x3[];    };\n"
"layout(std430,binding=4)  readonly buffer BufD0inv { float d0inv[]; };\n"
"layout(std430,binding=5)  readonly buffer BufV0    { float v0[];    };\n"
"layout(std430,binding=6) writeonly buffer BufF0    { float f0[];    };\n"
"layout(std430,binding=7) writeonly buffer BufF1    { float f1[];    };\n"
"layout(std430,binding=8) writeonly buffer BufF2    { float f2[];    };\n"
"layout(std430,binding=9) writeonly buffer BufF3    { float f3[];    };\n"
"layout(std430,binding=10) buffer   BufEpsP  { float eps_p[]; };\n"
"layout(std430,binding=11) buffer   BufAccP  { float acc_p[]; };\n"
"layout(std430,binding=12) buffer   BufBroken{ uint  broken[]; };\n"
"uniform float mu;\n"
"uniform float lambda;\n"
"uniform uint  m;\n"
"uniform float yield_stress;\n"
"uniform float hardening_mod;\n"
"uniform float fracture_strain;\n"
"uniform int   plasticity_on;\n"
"void main(){\n"
"  uint j=gl_GlobalInvocationID.x;\n"
"  if(j>=m)return;\n"
"  // Fractured tet: zero forces, no update needed\n"
"  if(broken[j]>0u){\n"
"    f0[j*3u]=0.0; f0[j*3u+1u]=0.0; f0[j*3u+2u]=0.0;\n"
"    f1[j*3u]=0.0; f1[j*3u+1u]=0.0; f1[j*3u+2u]=0.0;\n"
"    f2[j*3u]=0.0; f2[j*3u+1u]=0.0; f2[j*3u+2u]=0.0;\n"
"    f3[j*3u]=0.0; f3[j*3u+1u]=0.0; f3[j*3u+2u]=0.0;\n"
"    return;\n"
"  }\n"
"  vec3 p0=vec3(x0[j*3u],x0[j*3u+1u],x0[j*3u+2u]);\n"
"  vec3 p1=vec3(x1[j*3u],x1[j*3u+1u],x1[j*3u+2u]);\n"
"  vec3 p2=vec3(x2[j*3u],x2[j*3u+1u],x2[j*3u+2u]);\n"
"  vec3 p3=vec3(x3[j*3u],x3[j*3u+1u],x3[j*3u+2u]);\n"
"  mat3 Dt=mat3(p1-p0, p2-p0, p3-p0);\n"
"  uint b=j*9u;\n"
"  mat3 D0inv=mat3(d0inv[b],d0inv[b+1u],d0inv[b+2u],\n"
"                  d0inv[b+3u],d0inv[b+4u],d0inv[b+5u],\n"
"                  d0inv[b+6u],d0inv[b+7u],d0inv[b+8u]);\n"
"  mat3 F=Dt*D0inv;\n"
"  mat3 FtF=transpose(F)*F;\n"
"  mat3 I3=mat3(1.0);\n"
"  mat3 E_total=0.5*(FtF-I3);\n"
"  // Load plastic strain (Voigt, 6 floats per tet)\n"
"  mat3 E_p=mat3(0.0);\n"
"  float acc=0.0;\n"
"  if(plasticity_on!=0){\n"
"    uint pb=j*6u;\n"
"    float e00=eps_p[pb],e11=eps_p[pb+1u],e22=eps_p[pb+2u];\n"
"    float e01=eps_p[pb+3u],e02=eps_p[pb+4u],e12=eps_p[pb+5u];\n"
"    // Reconstruct symmetric mat3 (GLSL column-major)\n"
"    E_p=mat3(e00,e01,e02, e01,e11,e12, e02,e12,e22);\n"
"    acc=acc_p[j];\n"
"  }\n"
"  // Elastic strain\n"
"  mat3 E_e=E_total-E_p;\n"
"  float trEe=E_e[0][0]+E_e[1][1]+E_e[2][2];\n"
"  // 2nd Piola-Kirchhoff trial stress\n"
"  mat3 S=2.0*mu*E_e+lambda*trEe*I3;\n"
"  // Radial return mapping (von Mises)\n"
"  if(plasticity_on!=0){\n"
"    float trS=S[0][0]+S[1][1]+S[2][2];\n"
"    mat3 devS=S-(trS/3.0)*I3;\n"
"    float nDevSq=dot(devS[0],devS[0])+dot(devS[1],devS[1])+dot(devS[2],devS[2]);\n"
"    float vonMises=sqrt(1.5*nDevSq);\n"
"    float sy=yield_stress+hardening_mod*acc;\n"
"    if(vonMises>sy){\n"
"      float scale=sy/vonMises;\n"
"      float inv2mu=1.0/(2.0*max(mu,1.0));\n"
"      mat3 dEp=devS*((1.0-scale)*inv2mu);\n"
"      mat3 Epn=E_p+dEp;\n"
"      float nDep=sqrt(dot(dEp[0],dEp[0])+dot(dEp[1],dEp[1])+dot(dEp[2],dEp[2]));\n"
"      acc+=sqrt(2.0/3.0)*nDep;\n"
"      // Write back Voigt\n"
"      uint pb=j*6u;\n"
"      eps_p[pb+0u]=Epn[0][0]; eps_p[pb+1u]=Epn[1][1]; eps_p[pb+2u]=Epn[2][2];\n"
"      eps_p[pb+3u]=Epn[0][1]; eps_p[pb+4u]=Epn[0][2]; eps_p[pb+5u]=Epn[1][2];\n"
"      acc_p[j]=acc;\n"
"      if(acc>=fracture_strain) broken[j]=1u;\n"
"      // Return-mapped stress\n"
"      S=devS*scale+(trS/3.0)*I3;\n"
"    }\n"
"  }\n"
"  // PK1 = F * S,  H = -V * PK1 * D0inv^T\n"
"  float V=v0[j];\n"
"  mat3 P=F*S;\n"
"  mat3 H=-V*P*transpose(D0inv);\n"
"  vec3 fv1=H[0],fv2=H[1],fv3=H[2];\n"
"  vec3 fv0=-(fv1+fv2+fv3);\n"
"  f0[j*3u]=fv0.x; f0[j*3u+1u]=fv0.y; f0[j*3u+2u]=fv0.z;\n"
"  f1[j*3u]=fv1.x; f1[j*3u+1u]=fv1.y; f1[j*3u+2u]=fv1.z;\n"
"  f2[j*3u]=fv2.x; f2[j*3u+1u]=fv2.y; f2[j*3u+2u]=fv2.z;\n"
"  f3[j*3u]=fv3.x; f3[j*3u+1u]=fv3.y; f3[j*3u+2u]=fv3.z;\n"
"}\n";

/* ---- scatter: SpMV force accumulation (1 thread/node, no atomics) ---- */
static const char *SRC_SCATTER =
"#version 430 core\n"
"layout(local_size_x=64,local_size_y=1,local_size_z=1) in;\n"
"layout(std430,binding=0) readonly  buffer BufF0       { float f0[];       };\n"
"layout(std430,binding=1) readonly  buffer BufF1       { float f1[];       };\n"
"layout(std430,binding=2) readonly  buffer BufF2       { float f2[];       };\n"
"layout(std430,binding=3) readonly  buffer BufF3       { float f3[];       };\n"
"layout(std430,binding=4) readonly  buffer BufAdjStart { uint  adj_start[];};\n"
"layout(std430,binding=5) readonly  buffer BufAdjData  { uint  adj_data[]; };\n"
"layout(std430,binding=6) writeonly buffer BufForce    { float force[];    };\n"
"uniform uint n;\n"
"void main(){\n"
"  uint i=gl_GlobalInvocationID.x;\n"
"  if(i>=n)return;\n"
"  vec3 acc=vec3(0.0);\n"
"  uint start=adj_start[i];\n"
"  uint end=adj_start[i+1u];\n"
"  uint pk=0u;\n"
"  uint ci=0u;\n"
"  uint ti=0u;\n"
"  vec3 fv=vec3(0.0);\n"
"  for(uint t=start;t<end;t++){\n"
"    pk=adj_data[t];\n"
"    ci=bitfieldExtract(pk,28,4);\n"
"    ti=bitfieldExtract(pk,0,28);\n"
"    if(ci==0u)      fv=vec3(f0[ti*3u],f0[ti*3u+1u],f0[ti*3u+2u]);\n"
"    else if(ci==1u) fv=vec3(f1[ti*3u],f1[ti*3u+1u],f1[ti*3u+2u]);\n"
"    else if(ci==2u) fv=vec3(f2[ti*3u],f2[ti*3u+1u],f2[ti*3u+2u]);\n"
"    else            fv=vec3(f3[ti*3u],f3[ti*3u+1u],f3[ti*3u+2u]);\n"
"    acc+=fv;\n"
"  }\n"
"  force[i*3u]=acc.x; force[i*3u+1u]=acc.y; force[i*3u+2u]=acc.z;\n"
"}\n";

/* ---- ext_force: gravity + painted forces + anchor enforcement ---- */
static const char *SRC_EXT_FORCE =
"#version 430 core\n"
"layout(local_size_x=64,local_size_y=1,local_size_z=1) in;\n"
"layout(std430,binding=0) buffer    BufForce    { float force[];    };\n"
"layout(std430,binding=1) readonly  buffer BufExtF  { float ext_force[];};\n"
"layout(std430,binding=2) readonly  buffer BufMass  { float mass[];     };\n"
"layout(std430,binding=3) readonly  buffer BufAnchor{ uint  anchor[];   };\n"
"uniform float gravity;\n"
"uniform uint n;\n"
"void main(){\n"
"  uint i=gl_GlobalInvocationID.x;\n"
"  if(i>=n)return;\n"
"  float m=max(mass[i],1e-20);\n"
"  vec3 f=vec3(force[i*3u],force[i*3u+1u],force[i*3u+2u]);\n"
"  f+=vec3(ext_force[i*3u],ext_force[i*3u+1u],ext_force[i*3u+2u]);\n"
"  f.y-=gravity*m;\n"
"  if(anchor[i]>0u) f=vec3(0.0);\n"
"  force[i*3u]=f.x; force[i*3u+1u]=f.y; force[i*3u+2u]=f.z;\n"
"}\n";

/* ---- integrate: symplectic Euler ---- */
static const char *SRC_INTEGRATE =
"#version 430 core\n"
"layout(local_size_x=64,local_size_y=1,local_size_z=1) in;\n"
"layout(std430,binding=0) buffer    BufPos    { float pos[];    };\n"
"layout(std430,binding=1) buffer    BufVel    { float vel[];    };\n"
"layout(std430,binding=2) readonly  buffer BufForce  { float force[];  };\n"
"layout(std430,binding=3) readonly  buffer BufMass   { float mass[];   };\n"
"layout(std430,binding=4) readonly  buffer BufAnchor { uint  anchor[]; };\n"
"uniform float dt;\n"
"uniform uint n;\n"
"void main(){\n"
"  uint i=gl_GlobalInvocationID.x;\n"
"  if(i>=n)return;\n"
"  if(anchor[i]>0u)return;\n"
"  float m=max(mass[i],1e-20);\n"
"  vec3 v=vec3(vel[i*3u],vel[i*3u+1u],vel[i*3u+2u]);\n"
"  vec3 f=vec3(force[i*3u],force[i*3u+1u],force[i*3u+2u]);\n"
"  v+=(f/m)*dt;\n"
"  vec3 p=vec3(pos[i*3u],pos[i*3u+1u],pos[i*3u+2u]);\n"
"  p+=v*dt;\n"
"  vel[i*3u]=v.x; vel[i*3u+1u]=v.y; vel[i*3u+2u]=v.z;\n"
"  pos[i*3u]=p.x; pos[i*3u+1u]=p.y; pos[i*3u+2u]=p.z;\n"
"}\n";

/* ---- surface normals: per-triangle cross product ---- */
static const char *SRC_SURF_NORMALS =
"#version 430 core\n"
"layout(local_size_x=64,local_size_y=1,local_size_z=1) in;\n"
"layout(std430,binding=0) readonly  buffer BufPos     { float pos[];      };\n"
"layout(std430,binding=1) readonly  buffer BufSurfTri { uint  surf_tri[]; };\n"
"layout(std430,binding=2) writeonly buffer BufNorm    { float surf_norm[];};\n"
"uniform uint n_surf;\n"
"void main(){\n"
"  uint t=gl_GlobalInvocationID.x;\n"
"  if(t>=n_surf)return;\n"
"  uint a=surf_tri[t*3u],b=surf_tri[t*3u+1u],c=surf_tri[t*3u+2u];\n"
"  vec3 pa=vec3(pos[a*3u],pos[a*3u+1u],pos[a*3u+2u]);\n"
"  vec3 pb=vec3(pos[b*3u],pos[b*3u+1u],pos[b*3u+2u]);\n"
"  vec3 pc=vec3(pos[c*3u],pos[c*3u+1u],pos[c*3u+2u]);\n"
"  vec3 n=normalize(cross(pb-pa,pc-pa));\n"
"  surf_norm[t*3u]=n.x; surf_norm[t*3u+1u]=n.y; surf_norm[t*3u+2u]=n.z;\n"
"}\n";

/* ---- displacement scalar compute shader ---- */
static const char *SRC_DISP_SCALAR =
"#version 430 core\n"
"layout(local_size_x=64,local_size_y=1,local_size_z=1) in;\n"
"layout(std430,binding=0) readonly  buffer BufPos  { float pos[];      };\n"
"layout(std430,binding=1) readonly  buffer BufRest { float rest_pos[]; };\n"
"layout(std430,binding=2) writeonly buffer BufScal { float scal[];     };\n"
"uniform uint n;\n"
"void main(){\n"
"  uint i=gl_GlobalInvocationID.x;\n"
"  if(i>=n)return;\n"
"  float dx=pos[i*3u]-rest_pos[i*3u];\n"
"  float dy=pos[i*3u+1u]-rest_pos[i*3u+1u];\n"
"  float dz=pos[i*3u+2u]-rest_pos[i*3u+2u];\n"
"  scal[i]=sqrt(dx*dx+dy*dy+dz*dz);\n"
"}\n";

/* ---- local deformation: ||F - I||_F per tet ---- */
static const char *SRC_F_NORM_TET =
"#version 430 core\n"
"layout(local_size_x=64,local_size_y=1,local_size_z=1) in;\n"
"layout(std430,binding=0) readonly  buffer BufPos   { float pos[];   };\n"
"layout(std430,binding=1) readonly  buffer BufC0    { uint  ci0[];   };\n"
"layout(std430,binding=2) readonly  buffer BufC1    { uint  ci1[];   };\n"
"layout(std430,binding=3) readonly  buffer BufC2    { uint  ci2[];   };\n"
"layout(std430,binding=4) readonly  buffer BufC3    { uint  ci3[];   };\n"
"layout(std430,binding=5) readonly  buffer BufD0inv { float d0inv[]; };\n"
"layout(std430,binding=6) writeonly buffer BufTetS  { float tet_s[]; };\n"
"uniform uint m;\n"
"void main(){\n"
"  uint j=gl_GlobalInvocationID.x;\n"
"  if(j>=m)return;\n"
"  uint n0=ci0[j],n1=ci1[j],n2=ci2[j],n3=ci3[j];\n"
"  vec3 p0=vec3(pos[n0*3u],pos[n0*3u+1u],pos[n0*3u+2u]);\n"
"  vec3 p1=vec3(pos[n1*3u],pos[n1*3u+1u],pos[n1*3u+2u]);\n"
"  vec3 p2=vec3(pos[n2*3u],pos[n2*3u+1u],pos[n2*3u+2u]);\n"
"  vec3 p3=vec3(pos[n3*3u],pos[n3*3u+1u],pos[n3*3u+2u]);\n"
"  mat3 Dt=mat3(p1-p0,p2-p0,p3-p0);\n"
"  uint b=j*9u;\n"
"  mat3 D0inv=mat3(d0inv[b],d0inv[b+1u],d0inv[b+2u],\n"
"                  d0inv[b+3u],d0inv[b+4u],d0inv[b+5u],\n"
"                  d0inv[b+6u],d0inv[b+7u],d0inv[b+8u]);\n"
"  mat3 F=Dt*D0inv;\n"
"  // Green-Lagrange strain E=(F^T*F-I)/2 -- rotation-invariant\n"
"  // ||E||_F: zero for rigid body motion, non-zero only for true deformation\n"
"  mat3 FtF=transpose(F)*F;\n"
"  mat3 E=0.5*(FtF-mat3(1.0));\n"
"  float s=0.0;\n"
"  for(int c=0;c<3;c++) s+=dot(E[c],E[c]);\n"
"  tet_s[j]=sqrt(s);\n"
"}\n";

/* ---- local deformation: scatter-average tet scalar to nodes ---- */
static const char *SRC_F_NORM_NODE =
"#version 430 core\n"
"layout(local_size_x=64,local_size_y=1,local_size_z=1) in;\n"
"layout(std430,binding=0) readonly  buffer BufTetS    { float tet_s[];    };\n"
"layout(std430,binding=1) readonly  buffer BufAdjStart{ uint  adj_start[];};\n"
"layout(std430,binding=2) readonly  buffer BufAdjData { uint  adj_data[]; };\n"
"layout(std430,binding=3) writeonly buffer BufNodeS   { float node_s[];   };\n"
"uniform uint n;\n"
"void main(){\n"
"  uint i=gl_GlobalInvocationID.x;\n"
"  if(i>=n)return;\n"
"  uint start=adj_start[i];\n"
"  uint end=adj_start[i+1u];\n"
"  float acc=0.0;\n"
"  for(uint t=start;t<end;t++){\n"
"    uint ti=bitfieldExtract(adj_data[t],0,28);\n"
"    acc+=tet_s[ti];\n"
"  }\n"
"  uint cnt=end-start;\n"
"  node_s[i]=(cnt>0u)?acc/float(cnt):0.0;\n"
"}\n";

/* ---- volumetric strain: tr(E) per tet (signed: >0 tension, <0 compression) ---- */
static const char *SRC_VOL_STRAIN_TET =
"#version 430 core\n"
"layout(local_size_x=64,local_size_y=1,local_size_z=1) in;\n"
"layout(std430,binding=0) readonly  buffer BufPos   { float pos[];   };\n"
"layout(std430,binding=1) readonly  buffer BufC0    { uint  ci0[];   };\n"
"layout(std430,binding=2) readonly  buffer BufC1    { uint  ci1[];   };\n"
"layout(std430,binding=3) readonly  buffer BufC2    { uint  ci2[];   };\n"
"layout(std430,binding=4) readonly  buffer BufC3    { uint  ci3[];   };\n"
"layout(std430,binding=5) readonly  buffer BufD0inv { float d0inv[]; };\n"
"layout(std430,binding=6) writeonly buffer BufTetS  { float tet_s[]; };\n"
"uniform uint m;\n"
"void main(){\n"
"  uint j=gl_GlobalInvocationID.x;\n"
"  if(j>=m)return;\n"
"  uint n0=ci0[j],n1=ci1[j],n2=ci2[j],n3=ci3[j];\n"
"  vec3 p0=vec3(pos[n0*3u],pos[n0*3u+1u],pos[n0*3u+2u]);\n"
"  vec3 p1=vec3(pos[n1*3u],pos[n1*3u+1u],pos[n1*3u+2u]);\n"
"  vec3 p2=vec3(pos[n2*3u],pos[n2*3u+1u],pos[n2*3u+2u]);\n"
"  vec3 p3=vec3(pos[n3*3u],pos[n3*3u+1u],pos[n3*3u+2u]);\n"
"  mat3 Dt=mat3(p1-p0,p2-p0,p3-p0);\n"
"  uint b=j*9u;\n"
"  mat3 D0inv=mat3(d0inv[b],d0inv[b+1u],d0inv[b+2u],\n"
"                  d0inv[b+3u],d0inv[b+4u],d0inv[b+5u],\n"
"                  d0inv[b+6u],d0inv[b+7u],d0inv[b+8u]);\n"
"  mat3 F=Dt*D0inv;\n"
"  // Green-Lagrange strain E=(F^T*F-I)/2, take trace for volumetric strain\n"
"  mat3 FtF=transpose(F)*F;\n"
"  float trE=0.5*((FtF[0][0]-1.0)+(FtF[1][1]-1.0)+(FtF[2][2]-1.0));\n"
"  tet_s[j]=trE;\n"
"}\n";

/* ---- fracture_surf: mark corner nodes of every broken tet as surface ----
 * One thread per tet.  If broken[j]==1 we atomicOr each of the 4 corner      *
 * nodes' surf_node entry with 1u.  Multiple threads writing the same node      *
 * is safe via atomicOr (all write 1u; value is idempotent).                   *
 * surf_node is never cleared mid-simulation, only on full reset.              */
static const char *SRC_FRACTURE_SURF =
"#version 430 core\n"
"layout(local_size_x=64,local_size_y=1,local_size_z=1) in;\n"
"layout(std430,binding=0) readonly buffer BufBroken  { uint broken[];   };\n"
"layout(std430,binding=1) readonly buffer BufC0      { uint ci0[];      };\n"
"layout(std430,binding=2) readonly buffer BufC1      { uint ci1[];      };\n"
"layout(std430,binding=3) readonly buffer BufC2      { uint ci2[];      };\n"
"layout(std430,binding=4) readonly buffer BufC3      { uint ci3[];      };\n"
"layout(std430,binding=5)          buffer BufSurfNode{ uint surf_node[];};\n"
"uniform uint m;\n"
"void main(){\n"
"  uint j=gl_GlobalInvocationID.x;\n"
"  if(j>=m)return;\n"
"  if(broken[j]==0u)return;\n"
"  atomicOr(surf_node[ci0[j]],1u);\n"
"  atomicOr(surf_node[ci1[j]],1u);\n"
"  atomicOr(surf_node[ci2[j]],1u);\n"
"  atomicOr(surf_node[ci3[j]],1u);\n"
"}\n";

/* ---- interior fracture-face vertex shader ----------------------------------------
 * One draw vertex per interior face vertex (3 per face, gl_VertexID/3 = face index).
 * Each face has two adjacent tets (ta, tb).  The face is rendered when exactly one
 * is broken.  Normal is computed from current positions and oriented to face INTO
 * the fractured void.  Clipped to behind near-plane when neither/both broken.       */
static const char *SRC_INT_RENDER_VERT =
"#version 430 compatibility\n"
"layout(std430,binding=0) readonly buffer BufPos    { float pos[];       };\n"
"layout(std430,binding=1) readonly buffer BufIntTri { uint  int_tri[];   };\n"
"layout(std430,binding=2) readonly buffer BufIntTets{ uint  int_tets[];  };\n"
"layout(std430,binding=3) readonly buffer BufInt4th { uint  int_tet4th[];};\n"
"layout(std430,binding=4) readonly buffer BufBroken { uint  broken[];    };\n"
"layout(std430,binding=5) readonly buffer BufScalar { float scalar[];    };\n"
"uniform mat4 MVP;\n"
"uniform mat4 MV;\n"
"out vec3  v_normal;\n"
"out vec3  v_world;\n"
"out float v_scalar;\n"
"out float v_broken;\n"
"void main(){\n"
"  uint vid=uint(gl_VertexID);\n"
"  uint fi=vid/3u;\n"
"  uint li=vid%3u;\n"
"  uint ta=int_tets[fi*2u];\n"
"  uint tb=int_tets[fi*2u+1u];\n"
"  uint ba=broken[ta];\n"
"  uint bb=broken[tb];\n"
"  // Clip if neither or both tets are broken\n"
"  if(ba+bb!=1u){\n"
"    gl_Position=vec4(0.0,0.0,-2.0,1.0);\n"
"    v_normal=vec3(0.0); v_world=vec3(0.0); v_scalar=0.0; v_broken=0.0;\n"
"    return;\n"
"  }\n"
"  uint na=int_tri[fi*3u];\n"
"  uint nb=int_tri[fi*3u+1u];\n"
"  uint nc=int_tri[fi*3u+2u];\n"
"  uint ni=(li==0u)?na:(li==1u)?nb:nc;\n"
"  vec3 p=vec3(pos[ni*3u],pos[ni*3u+1u],pos[ni*3u+2u]);\n"
"  vec3 pa=vec3(pos[na*3u],pos[na*3u+1u],pos[na*3u+2u]);\n"
"  vec3 pb=vec3(pos[nb*3u],pos[nb*3u+1u],pos[nb*3u+2u]);\n"
"  vec3 pc=vec3(pos[nc*3u],pos[nc*3u+1u],pos[nc*3u+2u]);\n"
"  vec3 nr=cross(pb-pa,pc-pa);\n"
"  float nl=length(nr); if(nl>0.0) nr/=nl;\n"
"  // Orient nr to point AWAY from ta (outside ta, toward tb's region)\n"
"  uint fa_idx=int_tet4th[fi*2u];\n"
"  vec3 pfA=vec3(pos[fa_idx*3u],pos[fa_idx*3u+1u],pos[fa_idx*3u+2u]);\n"
"  if(dot(nr,pfA-pa)>0.0) nr=-nr;\n"
"  // If ta is broken: flip normal so it faces into ta's void\n"
"  if(ba==1u) nr=-nr;\n"
"  gl_Position=MVP*vec4(p,1.0);\n"
"  v_normal=mat3(MV)*nr;\n"
"  v_world=p;\n"
"  v_scalar=scalar[ni];\n"
"  v_broken=0.0;\n"
"}\n";

/* ---- surface render vertex shader (reads SSBOs via gl_VertexID) ---- */
static const char *SRC_RENDER_VERT =
"#version 430 compatibility\n"
"layout(std430,binding=0) readonly buffer BufPos     { float pos[];       };\n"
"layout(std430,binding=1) readonly buffer BufSurfTri { uint  surf_tri[];  };\n"
"layout(std430,binding=2) readonly buffer BufNorm    { float surf_norm[]; };\n"
"layout(std430,binding=3) readonly buffer BufScalar  { float scalar[];    };\n"
"layout(std430,binding=4) readonly buffer BufSurfTet { uint  surf_tet[];  };\n"
"layout(std430,binding=5) readonly buffer BufBroken  { uint  broken[];    };\n"
"uniform mat4 MVP;\n"
"uniform mat4 MV;\n"
"out vec3 v_normal;\n"
"out vec3 v_world;\n"
"out float v_scalar;\n"
"out float v_broken;\n"
"void main(){\n"
"  uint vi=surf_tri[uint(gl_VertexID)];\n"
"  vec3 p=vec3(pos[vi*3u],pos[vi*3u+1u],pos[vi*3u+2u]);\n"
"  uint ti=uint(gl_VertexID)/3u;\n"
"  vec3 n=vec3(surf_norm[ti*3u],surf_norm[ti*3u+1u],surf_norm[ti*3u+2u]);\n"
"  gl_Position=MVP*vec4(p,1.0);\n"
"  v_normal=mat3(MV)*n;\n"
"  v_world=p;\n"
"  v_scalar=scalar[vi];\n"
"  v_broken=float(broken[surf_tet[ti]]);\n"
"}\n";

/* ---- surface render fragment shader (Blinn-Phong + optional colormap) ---- */
static const char *SRC_RENDER_FRAG =
"#version 430 compatibility\n"
"in vec3 v_normal;\n"
"in vec3 v_world;\n"
"in float v_scalar;\n"
"in float v_broken;\n"
"out vec4 frag_color;\n"
"uniform vec3  light_dir;\n"
"uniform vec3  mat_color;\n"
"uniform int   color_mode;\n"
"uniform int   scalar_signed;\n"
"uniform float scalar_max;\n"
"vec3 heat_color(float t){\n"
"  t=clamp(t,0.0,1.0);\n"
"  vec3 cl;\n"
"  if(t<0.25)      cl=mix(vec3(0.0,0.0,1.0),vec3(0.0,1.0,1.0),t*4.0);\n"
"  else if(t<0.5)  cl=mix(vec3(0.0,1.0,1.0),vec3(0.0,1.0,0.0),(t-0.25)*4.0);\n"
"  else if(t<0.75) cl=mix(vec3(0.0,1.0,0.0),vec3(1.0,1.0,0.0),(t-0.5)*4.0);\n"
"  else             cl=mix(vec3(1.0,1.0,0.0),vec3(1.0,0.0,0.0),(t-0.75)*4.0);\n"
"  return cl;\n"
"}\n"
"vec3 diverge_color(float t){\n"
"  t=clamp(t,0.0,1.0);\n"
"  if(t<0.5) return mix(vec3(0.1,0.3,1.0),vec3(1.0,1.0,1.0),t*2.0);\n"
"  else      return mix(vec3(1.0,1.0,1.0),vec3(1.0,0.15,0.1),(t-0.5)*2.0);\n"
"}\n"
"void main(){\n"
"  if(v_broken>0.5) discard;\n"
"  vec3 n=normalize(v_normal);\n"
"  float diff=max(dot(n,light_dir),0.0);\n"
"  vec3 base;\n"
"  if(color_mode==0){\n"
"    base=mat_color;\n"
"  } else if(scalar_signed==1){\n"
"    float sv=(scalar_max>0.0)?clamp(v_scalar/scalar_max*3.0,-1.0,1.0):0.0;\n"
"    base=diverge_color((sv+1.0)*0.5);\n"
"  } else {\n"
"    float sv=(scalar_max>0.0)?clamp(v_scalar/scalar_max*3.0,0.0,1.0):0.0;\n"
"    base=heat_color(sv);\n"
"  }\n"
"  frag_color=vec4(base*(0.18+0.82*diff),1.0);\n"
"}\n";

/* ---- slice plane vertex shader (point cloud, one vertex per node) ---- */
static const char *SRC_SLICE_VERT =
"#version 430 compatibility\n"
"layout(std430,binding=0) readonly buffer BufPos    { float pos[];    };\n"
"layout(std430,binding=1) readonly buffer BufScalar { float scalar[]; };\n"
"uniform mat4  MVP;\n"
"uniform vec3  slice_pt;\n"
"uniform vec3  slice_nrm;\n"
"uniform float slice_h;\n"
"out float v_scalar;\n"
"out float v_dist;\n"
"void main(){\n"
"  uint i=uint(gl_VertexID);\n"
"  vec3 p=vec3(pos[i*3u],pos[i*3u+1u],pos[i*3u+2u]);\n"
"  float ds=dot(p-slice_pt,slice_nrm);\n"
"  v_dist=ds;\n"
"  v_scalar=scalar[i];\n"
"  if(abs(ds)<=slice_h){\n"
"    gl_Position=MVP*vec4(p,1.0);\n"
"    gl_PointSize=6.0;\n"
"  } else {\n"
"    gl_Position=vec4(2.0,2.0,2.0,1.0);\n"
"    gl_PointSize=0.0;\n"
"  }\n"
"}\n";

/* ---- slice plane fragment shader ---- */
static const char *SRC_SLICE_FRAG =
"#version 430 compatibility\n"
"in float v_scalar;\n"
"in float v_dist;\n"
"out vec4 frag_color;\n"
"uniform float slice_h;\n"
"uniform float scalar_max;\n"
"uniform int   scalar_signed;\n"
"vec3 heat_color(float t){\n"
"  t=clamp(t,0.0,1.0);\n"
"  vec3 cl;\n"
"  if(t<0.25)      cl=mix(vec3(0.0,0.0,1.0),vec3(0.0,1.0,1.0),t*4.0);\n"
"  else if(t<0.5)  cl=mix(vec3(0.0,1.0,1.0),vec3(0.0,1.0,0.0),(t-0.25)*4.0);\n"
"  else if(t<0.75) cl=mix(vec3(0.0,1.0,0.0),vec3(1.0,1.0,0.0),(t-0.5)*4.0);\n"
"  else             cl=mix(vec3(1.0,1.0,0.0),vec3(1.0,0.0,0.0),(t-0.75)*4.0);\n"
"  return cl;\n"
"}\n"
"vec3 diverge_color(float t){\n"
"  t=clamp(t,0.0,1.0);\n"
"  if(t<0.5) return mix(vec3(0.1,0.3,1.0),vec3(1.0,1.0,1.0),t*2.0);\n"
"  else      return mix(vec3(1.0,1.0,1.0),vec3(1.0,0.15,0.1),(t-0.5)*2.0);\n"
"}\n"
"void main(){\n"
"  if(abs(v_dist)>slice_h) discard;\n"
"  vec3 col;\n"
"  if(scalar_signed==1){\n"
"    float sv=(scalar_max>0.0)?clamp(v_scalar/scalar_max*3.0,-1.0,1.0):0.0;\n"
"    col=diverge_color((sv+1.0)*0.5);\n"
"  } else {\n"
"    float sv=(scalar_max>0.0)?clamp(v_scalar/scalar_max*3.0,0.0,1.0):0.0;\n"
"    col=heat_color(sv);\n"
"  }\n"
"  frag_color=vec4(col,1.0);\n"
"}\n";

/* ===========================================================
 * Helper macro for GL memory barriers between shader passes
 * =========================================================== */
#define SSBO_BARRIER() glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

static GLuint dispatch_size(int n) {
    return (GLuint)((n + 63) / 64);
}

/* ===========================================================
 * fem_sim_create
 * =========================================================== */
FemSimGPU* fem_sim_create(const FemMesh *mesh, float lame_mu, float lame_lambda) {
    FemSimGPU *s = calloc(1, sizeof(FemSimGPU));
    if (!s) return NULL;

    s->n_nodes   = mesh->n_nodes;
    s->m_tets    = mesh->m_tets;
    s->n_surf_tri= mesh->n_surf_tri;
    s->total_adj = mesh->total_adj;
    s->lame_mu   = lame_mu;
    s->lame_lambda = lame_lambda;
    s->dt        = 1e-4f;
    s->gravity   = 9.81f;
    s->color_mode = 1;
    s->show_slice = 0;

    int n = mesh->n_nodes;
    int m = mesh->m_tets;
    int ns= mesh->n_surf_tri;
    float bb_min[3] = { mesh->pos[0], mesh->pos[1], mesh->pos[2] };
    float bb_max[3] = { mesh->pos[0], mesh->pos[1], mesh->pos[2] };
    for (int i = 1; i < n; i++) {
        const float *p = mesh->pos + i * 3;
        if (p[0] < bb_min[0]) bb_min[0] = p[0];
        if (p[1] < bb_min[1]) bb_min[1] = p[1];
        if (p[2] < bb_min[2]) bb_min[2] = p[2];
        if (p[0] > bb_max[0]) bb_max[0] = p[0];
        if (p[1] > bb_max[1]) bb_max[1] = p[1];
        if (p[2] > bb_max[2]) bb_max[2] = p[2];
    }
    s->slice_pt[0] = 0.5f * (bb_min[0] + bb_max[0]);
    s->slice_pt[1] = 0.5f * (bb_min[1] + bb_max[1]);
    s->slice_pt[2] = 0.5f * (bb_min[2] + bb_max[2]);
    s->slice_nrm[0] = 0.0f;
    s->slice_nrm[1] = 1.0f;
    s->slice_nrm[2] = 0.0f;
    {
        float ext_x = bb_max[0] - bb_min[0];
        float ext_y = bb_max[1] - bb_min[1];
        float ext_z = bb_max[2] - bb_min[2];
        float max_extent = ext_x;
        if (ext_y > max_extent) max_extent = ext_y;
        if (ext_z > max_extent) max_extent = ext_z;
        s->slice_h = fmaxf(mesh->lattice_h * 0.6f, max_extent * 0.01f);
        s->scalar_max = fmaxf(mesh->lattice_h, max_extent * 0.1f);
    }

    /* ---- upload initial position + zero vel/force/mass ---- */
    s->pos_ssbo    = make_ssbo(mesh->pos,   (size_t)n*3*sizeof(float), GL_DYNAMIC_DRAW);
    {
        float *zeros = calloc((size_t)n*3, sizeof(float));
        s->vel_ssbo   = make_ssbo(zeros, (size_t)n*3*sizeof(float), GL_DYNAMIC_DRAW);
        s->force_ssbo = make_ssbo(zeros, (size_t)n*3*sizeof(float), GL_DYNAMIC_DRAW);
        free(zeros);
    }
    s->mass_ssbo   = make_ssbo(mesh->mass,   (size_t)n*sizeof(float), GL_STATIC_DRAW);

    /* ---- rest state ---- */
    s->D0inv_ssbo  = make_ssbo(mesh->D0_inv, (size_t)m*9*sizeof(float), GL_STATIC_DRAW);
    s->V0_ssbo     = make_ssbo(mesh->V0,     (size_t)m*sizeof(float),   GL_STATIC_DRAW);
    s->rest_pos_ssbo = make_ssbo(mesh->pos,  (size_t)n*3*sizeof(float), GL_STATIC_DRAW);
    {
        float *zeros = calloc((size_t)n, sizeof(float));
        s->node_scalar_ssbo = make_ssbo(zeros, (size_t)n*sizeof(float), GL_DYNAMIC_DRAW);
        free(zeros);
    }
    s->tet_scalar_ssbo = make_ssbo(NULL, (size_t)m*sizeof(float), GL_DYNAMIC_COPY);

    /* ---- structure matrices (forward gather, uint) ---- */
    for (int k = 0; k < 4; k++)
        s->ci_ssbo[k] = make_ssbo(mesh->ci[k], (size_t)m*sizeof(unsigned int), GL_STATIC_DRAW);

    /* ---- backward adjacency (uint) ---- */
    s->adj_start_ssbo = make_ssbo(mesh->adj_start,
                                  (size_t)(n+1)*sizeof(unsigned int), GL_STATIC_DRAW);
    s->adj_data_ssbo  = make_ssbo(mesh->adj_data,
                                  (size_t)mesh->total_adj*sizeof(unsigned int), GL_STATIC_DRAW);

    /* ---- anchor / ext_force (zeroed initially) ---- */
    {
        unsigned int *zeros_u = calloc((size_t)n, sizeof(unsigned int));
        s->anchor_ssbo = make_ssbo(zeros_u, (size_t)n*sizeof(unsigned int), GL_DYNAMIC_DRAW);
        free(zeros_u);
        float *zeros_f = calloc((size_t)n*3, sizeof(float));
        s->ext_force_ssbo = make_ssbo(zeros_f, (size_t)n*3*sizeof(float), GL_DYNAMIC_DRAW);
        free(zeros_f);
    }

    /* ---- intermediate per-tet SSBOs (uninitialised, overwritten each step) ---- */
    for (int k = 0; k < 4; k++) {
        s->x_ssbo[k] = make_ssbo(NULL, (size_t)m*3*sizeof(float), GL_DYNAMIC_COPY);
        s->f_ssbo[k] = make_ssbo(NULL, (size_t)m*3*sizeof(float), GL_DYNAMIC_COPY);
    }

    /* ---- plasticity per-tet SSBOs (all zeroed initially = pure elastic start) ---- */
    {
        float *zeros_f = calloc((size_t)m*6, sizeof(float));
        s->eps_p_ssbo = make_ssbo(zeros_f, (size_t)m*6*sizeof(float), GL_DYNAMIC_DRAW);
        free(zeros_f);
    }
    {
        float *zeros_f = calloc((size_t)m, sizeof(float));
        s->acc_p_ssbo = make_ssbo(zeros_f, (size_t)m*sizeof(float), GL_DYNAMIC_DRAW);
        free(zeros_f);
    }
    {
        unsigned int *zeros_u = calloc((size_t)m, sizeof(unsigned int));
        s->broken_ssbo = make_ssbo(zeros_u, (size_t)m*sizeof(unsigned int), GL_DYNAMIC_DRAW);
        free(zeros_u);
    }
    {
        unsigned int *zeros_u = calloc((size_t)n, sizeof(unsigned int));
        s->surf_node_ssbo = make_ssbo(zeros_u, (size_t)n*sizeof(unsigned int), GL_DYNAMIC_COPY);
        free(zeros_u);
    }

    /* ---- plasticity defaults (mild steel) ---- */
    s->yield_stress       = 250e6f;
    s->hardening_mod      = 1e9f;
    s->fracture_strain    = 0.30f;
    s->plasticity_enabled = 1;

    /* ---- surface SSBOs ---- */
    s->surf_tri_ssbo     = make_ssbo(mesh->surf_tri,     (size_t)ns*3*sizeof(unsigned int), GL_STATIC_DRAW);
    s->surf_tri_tet_ssbo = make_ssbo(mesh->surf_tri_tet, (size_t)ns*sizeof(unsigned int),   GL_STATIC_DRAW);
    s->surf_norm_ssbo    = make_ssbo(NULL, (size_t)ns*3*sizeof(float), GL_DYNAMIC_COPY);

    /* ---- interior face SSBOs (fracture crack surface, static topology) ---- */
    {
        int ni = mesh->n_int_tri;
        s->n_int_tri = ni;
        s->int_tri_ssbo      = make_ssbo(mesh->int_tri,        (size_t)ni*3*sizeof(unsigned int), GL_STATIC_DRAW);
        s->int_tri_tet_ssbo  = make_ssbo(mesh->int_tri_tets,   (size_t)ni*2*sizeof(unsigned int), GL_STATIC_DRAW);
        s->int_tri_tet4th_ssbo = make_ssbo(mesh->int_tri_tet4th,(size_t)ni*2*sizeof(unsigned int), GL_STATIC_DRAW);
    }

    /* ---- compile compute programs ---- */
    GLuint cs;
#define COMPILE_CS(prog, src) \
    cs = compile_cs(src); \
    if (!cs) { fem_sim_free(s); return NULL; } \
    s->prog = link_compute(cs); \
    if (!s->prog) { fem_sim_free(s); return NULL; }

    COMPILE_CS(prog_gather,       SRC_GATHER)
    COMPILE_CS(prog_tet_force,    SRC_TET_FORCE)
    COMPILE_CS(prog_scatter,      SRC_SCATTER)
    COMPILE_CS(prog_ext_force,    SRC_EXT_FORCE)
    COMPILE_CS(prog_integrate,    SRC_INTEGRATE)
    COMPILE_CS(prog_surf_normals, SRC_SURF_NORMALS)
    COMPILE_CS(prog_disp_scalar,  SRC_DISP_SCALAR)
    COMPILE_CS(prog_f_norm_tet,    SRC_F_NORM_TET)
    COMPILE_CS(prog_f_norm_node,   SRC_F_NORM_NODE)
    COMPILE_CS(prog_vol_strain_tet, SRC_VOL_STRAIN_TET)
    COMPILE_CS(prog_fracture_surf,  SRC_FRACTURE_SURF)
#undef COMPILE_CS

    /* ---- compile render program ---- */
    GLuint vs = compile_vs(SRC_RENDER_VERT);
    GLuint fs = compile_fs(SRC_RENDER_FRAG);
    if (!vs || !fs) { fem_sim_free(s); return NULL; }
    s->prog_render = link_render(vs, fs);
    if (!s->prog_render) { fem_sim_free(s); return NULL; }

    vs = compile_vs(SRC_SLICE_VERT);
    fs = compile_fs(SRC_SLICE_FRAG);
    if (!vs || !fs) { fem_sim_free(s); return NULL; }
    s->prog_slice = link_render(vs, fs);
    if (!s->prog_slice) { fem_sim_free(s); return NULL; }

    /* ---- compile interior face render program (shares fragment shader) ---- */
    vs = compile_vs(SRC_INT_RENDER_VERT);
    fs = compile_fs(SRC_RENDER_FRAG);
    if (!vs || !fs) { fem_sim_free(s); return NULL; }
    s->prog_render_int = link_render(vs, fs);
    if (!s->prog_render_int) { fem_sim_free(s); return NULL; }

    /* ---- cache uniform locations ---- */
    s->uloc_gather_m    = glGetUniformLocation(s->prog_gather,       "m");
    s->uloc_tet_mu      = glGetUniformLocation(s->prog_tet_force,    "mu");
    s->uloc_tet_lambda  = glGetUniformLocation(s->prog_tet_force,    "lambda");
    s->uloc_tet_m       = glGetUniformLocation(s->prog_tet_force,    "m");
    s->uloc_tet_yield_stress   = glGetUniformLocation(s->prog_tet_force, "yield_stress");
    s->uloc_tet_hardening_mod  = glGetUniformLocation(s->prog_tet_force, "hardening_mod");
    s->uloc_tet_fracture_strain= glGetUniformLocation(s->prog_tet_force, "fracture_strain");
    s->uloc_tet_plasticity     = glGetUniformLocation(s->prog_tet_force, "plasticity_on");
    s->uloc_fsurf_m            = glGetUniformLocation(s->prog_fracture_surf, "m");
    s->uloc_scatter_n   = glGetUniformLocation(s->prog_scatter,      "n");
    s->uloc_ext_gravity = glGetUniformLocation(s->prog_ext_force,    "gravity");
    s->uloc_ext_n       = glGetUniformLocation(s->prog_ext_force,    "n");
    s->uloc_int_dt      = glGetUniformLocation(s->prog_integrate,    "dt");
    s->uloc_int_n       = glGetUniformLocation(s->prog_integrate,    "n");
    s->uloc_sn_n        = glGetUniformLocation(s->prog_surf_normals, "n_surf");
    s->uloc_render_MVP  = glGetUniformLocation(s->prog_render, "MVP");
    s->uloc_render_MV   = glGetUniformLocation(s->prog_render, "MV");
    s->uloc_render_light= glGetUniformLocation(s->prog_render, "light_dir");
    s->uloc_render_color= glGetUniformLocation(s->prog_render, "mat_color");
    s->uloc_render_color_mode   = glGetUniformLocation(s->prog_render, "color_mode");
    s->uloc_render_scalar_max   = glGetUniformLocation(s->prog_render, "scalar_max");
    s->uloc_render_scalar_signed= glGetUniformLocation(s->prog_render, "scalar_signed");
    s->uloc_disp_n              = glGetUniformLocation(s->prog_disp_scalar, "n");
    s->uloc_fnorm_tet_m         = glGetUniformLocation(s->prog_f_norm_tet,  "m");
    s->uloc_fnorm_node_n        = glGetUniformLocation(s->prog_f_norm_node, "n");
    s->uloc_vol_strain_tet_m    = glGetUniformLocation(s->prog_vol_strain_tet, "m");
    s->uloc_slice_MVP           = glGetUniformLocation(s->prog_slice, "MVP");
    s->uloc_slice_pt            = glGetUniformLocation(s->prog_slice, "slice_pt");
    s->uloc_slice_nrm           = glGetUniformLocation(s->prog_slice, "slice_nrm");
    s->uloc_slice_h             = glGetUniformLocation(s->prog_slice, "slice_h");
    s->uloc_slice_scalar_max    = glGetUniformLocation(s->prog_slice, "scalar_max");
    s->uloc_slice_scalar_signed = glGetUniformLocation(s->prog_slice, "scalar_signed");

    /* ---- interior face render uniform locations ---- */
    s->uloc_int_MVP          = glGetUniformLocation(s->prog_render_int, "MVP");
    s->uloc_int_MV           = glGetUniformLocation(s->prog_render_int, "MV");
    s->uloc_int_light        = glGetUniformLocation(s->prog_render_int, "light_dir");
    s->uloc_int_color        = glGetUniformLocation(s->prog_render_int, "mat_color");
    s->uloc_int_color_mode   = glGetUniformLocation(s->prog_render_int, "color_mode");
    s->uloc_int_scalar_max   = glGetUniformLocation(s->prog_render_int, "scalar_max");
    s->uloc_int_scalar_signed= glGetUniformLocation(s->prog_render_int, "scalar_signed");

    /* ---- VAO for surface draw call (no vertex attributes; reads SSBOs) ---- */
    glGenVertexArrays(1, &s->surface_vao);
    glGenVertexArrays(1, &s->int_vao);
    glGenVertexArrays(1, &s->slice_vao);

    return s;
}

/* ===========================================================
 * fem_sim_free
 * =========================================================== */
void fem_sim_free(FemSimGPU *s) {
    if (!s) return;
#define DEL_BUF(b) if(s->b){glDeleteBuffers(1,&s->b);s->b=0;}
    DEL_BUF(pos_ssbo); DEL_BUF(vel_ssbo); DEL_BUF(force_ssbo); DEL_BUF(mass_ssbo);
    DEL_BUF(D0inv_ssbo); DEL_BUF(V0_ssbo);
    DEL_BUF(rest_pos_ssbo); DEL_BUF(node_scalar_ssbo); DEL_BUF(tet_scalar_ssbo);
    for(int k=0;k<4;k++){DEL_BUF(ci_ssbo[k]); DEL_BUF(x_ssbo[k]); DEL_BUF(f_ssbo[k]);}
    DEL_BUF(eps_p_ssbo); DEL_BUF(acc_p_ssbo); DEL_BUF(broken_ssbo); DEL_BUF(surf_node_ssbo);
    DEL_BUF(adj_start_ssbo); DEL_BUF(adj_data_ssbo);
    DEL_BUF(anchor_ssbo); DEL_BUF(ext_force_ssbo);
    DEL_BUF(surf_tri_ssbo); DEL_BUF(surf_tri_tet_ssbo); DEL_BUF(surf_norm_ssbo);
    DEL_BUF(int_tri_ssbo); DEL_BUF(int_tri_tet_ssbo); DEL_BUF(int_tri_tet4th_ssbo);
#undef DEL_BUF
#define DEL_PROG(p) if(s->p){glDeleteProgram(s->p);s->p=0;}
    DEL_PROG(prog_gather); DEL_PROG(prog_tet_force); DEL_PROG(prog_scatter);
    DEL_PROG(prog_ext_force); DEL_PROG(prog_integrate);
    DEL_PROG(prog_surf_normals); DEL_PROG(prog_disp_scalar);
    DEL_PROG(prog_f_norm_tet); DEL_PROG(prog_f_norm_node);
    DEL_PROG(prog_vol_strain_tet); DEL_PROG(prog_fracture_surf);
    DEL_PROG(prog_render); DEL_PROG(prog_render_int); DEL_PROG(prog_slice);
#undef DEL_PROG
    if (s->surface_vao) { glDeleteVertexArrays(1, &s->surface_vao); s->surface_vao=0; }
    if (s->int_vao) { glDeleteVertexArrays(1, &s->int_vao); s->int_vao=0; }
    if (s->slice_vao) { glDeleteVertexArrays(1, &s->slice_vao); s->slice_vao=0; }
    free(s);
}

/* ===========================================================
 * fem_sim_upload_constraints
 * =========================================================== */
void fem_sim_upload_constraints(FemSimGPU *s,
                                const unsigned int *anchor,
                                const float *ext_force)
{
    int n = s->n_nodes;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, s->anchor_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)n*sizeof(unsigned int), anchor);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, s->ext_force_ssbo);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)n*3*sizeof(float), ext_force);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

/* ===========================================================
 * fem_sim_reset
 * =========================================================== */
void fem_sim_reset(FemSimGPU *s, const float *rest_pos) {
    int n = s->n_nodes;
    if (rest_pos) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, s->pos_ssbo);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)n*3*sizeof(float), rest_pos);
    }
    /* zero velocities */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, s->vel_ssbo);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, NULL);
    /* zero plasticity state */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, s->eps_p_ssbo);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, NULL);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, s->acc_p_ssbo);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, NULL);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, s->broken_ssbo);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, s->surf_node_ssbo);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

/* ===========================================================
 * fem_sim_substep
 * =========================================================== */
void fem_sim_substep(FemSimGPU *s) {
    GLuint n = (GLuint)s->n_nodes;
    GLuint m = (GLuint)s->m_tets;

    /* ---- gather: pos → x_ssbo[0..3] ---- */
    glUseProgram(s->prog_gather);
    glUniform1ui(s->uloc_gather_m, m);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->pos_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->ci_ssbo[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->ci_ssbo[1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s->ci_ssbo[2]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, s->ci_ssbo[3]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, s->x_ssbo[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, s->x_ssbo[1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, s->x_ssbo[2]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, s->x_ssbo[3]);
    glDispatchCompute(dispatch_size(m), 1, 1);
    SSBO_BARRIER();

    /* ---- tet_force: x_ssbo → f_ssbo[0..3], updates plasticity SSBOs ---- */
    glUseProgram(s->prog_tet_force);
    glUniform1f(s->uloc_tet_mu,     s->lame_mu);
    glUniform1f(s->uloc_tet_lambda, s->lame_lambda);
    glUniform1ui(s->uloc_tet_m, m);
    glUniform1f(s->uloc_tet_yield_stress,    s->yield_stress);
    glUniform1f(s->uloc_tet_hardening_mod,   s->hardening_mod);
    glUniform1f(s->uloc_tet_fracture_strain, s->fracture_strain);
    glUniform1i(s->uloc_tet_plasticity,      s->plasticity_enabled);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->x_ssbo[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->x_ssbo[1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->x_ssbo[2]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s->x_ssbo[3]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, s->D0inv_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, s->V0_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, s->f_ssbo[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, s->f_ssbo[1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, s->f_ssbo[2]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, s->f_ssbo[3]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, s->eps_p_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, s->acc_p_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, s->broken_ssbo);
    glDispatchCompute(dispatch_size(m), 1, 1);
    SSBO_BARRIER();

    /* ---- fracture_surf: mark corner nodes of newly broken tets as surface ---- */
    glUseProgram(s->prog_fracture_surf);
    glUniform1ui(s->uloc_fsurf_m, m);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->broken_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->ci_ssbo[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->ci_ssbo[1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s->ci_ssbo[2]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, s->ci_ssbo[3]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, s->surf_node_ssbo);
    glDispatchCompute(dispatch_size(m), 1, 1);
    SSBO_BARRIER();

    /* ---- scatter: SpMV f_ssbo → force_ssbo ---- */
    glUseProgram(s->prog_scatter);
    glUniform1ui(s->uloc_scatter_n, n);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->f_ssbo[0]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->f_ssbo[1]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->f_ssbo[2]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s->f_ssbo[3]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, s->adj_start_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, s->adj_data_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, s->force_ssbo);
    glDispatchCompute(dispatch_size(n), 1, 1);
    SSBO_BARRIER();

    /* ---- ext_force: add gravity + painted forces, zero anchors ---- */
    glUseProgram(s->prog_ext_force);
    glUniform1f(s->uloc_ext_gravity, s->gravity);
    glUniform1ui(s->uloc_ext_n, n);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->force_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->ext_force_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->mass_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s->anchor_ssbo);
    glDispatchCompute(dispatch_size(n), 1, 1);
    SSBO_BARRIER();

    /* ---- integrate: symplectic Euler ---- */
    glUseProgram(s->prog_integrate);
    glUniform1f(s->uloc_int_dt, s->dt);
    glUniform1ui(s->uloc_int_n, n);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->pos_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->vel_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->force_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s->mass_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, s->anchor_ssbo);
    glDispatchCompute(dispatch_size(n), 1, 1);
    SSBO_BARRIER();
}

/* ===========================================================
 * fem_sim_update_normals
 * =========================================================== */
void fem_sim_update_normals(FemSimGPU *s) {
    GLuint ns = (GLuint)s->n_surf_tri;
    glUseProgram(s->prog_surf_normals);
    glUniform1ui(s->uloc_sn_n, ns);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->pos_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->surf_tri_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->surf_norm_ssbo);
    glDispatchCompute(dispatch_size(ns), 1, 1);
    SSBO_BARRIER();
}

/* ===========================================================
 * fem_sim_render
 * =========================================================== */
void fem_sim_render(FemSimGPU *s,
                    const float *MVP, const float *MV,
                    const float *light_dir, const float *material_color)
{
    glUseProgram(s->prog_render);
    glUniformMatrix4fv(s->uloc_render_MVP,   1, GL_FALSE, MVP);
    glUniformMatrix4fv(s->uloc_render_MV,    1, GL_FALSE, MV);
    glUniform3fv(s->uloc_render_light, 1, light_dir);
    glUniform3fv(s->uloc_render_color, 1, material_color);
    glUniform1i(s->uloc_render_color_mode,    s->color_mode);
    glUniform1f(s->uloc_render_scalar_max,    s->scalar_max);
    glUniform1i(s->uloc_render_scalar_signed, (s->scalar_mode == 2) ? 1 : 0); /* only vol-strain signed */

    /* Bind SSBOs at their shader binding points */
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->pos_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->surf_tri_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->surf_norm_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s->node_scalar_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, s->surf_tri_tet_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, s->broken_ssbo);

    glBindVertexArray(s->surface_vao);
    glDrawArrays(GL_TRIANGLES, 0, s->n_surf_tri * 3);
    glBindVertexArray(0);

    /* ---- second pass: interior fracture faces ---- */
    if (s->n_int_tri > 0) {
        glUseProgram(s->prog_render_int);
        glUniformMatrix4fv(s->uloc_int_MVP,          1, GL_FALSE, MVP);
        glUniformMatrix4fv(s->uloc_int_MV,           1, GL_FALSE, MV);
        glUniform3fv(s->uloc_int_light, 1, light_dir);
        glUniform3fv(s->uloc_int_color, 1, material_color);
        glUniform1i(s->uloc_int_color_mode,    s->color_mode);
        glUniform1f(s->uloc_int_scalar_max,    s->scalar_max);
        glUniform1i(s->uloc_int_scalar_signed, (s->scalar_mode == 2) ? 1 : 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->pos_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->int_tri_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->int_tri_tet_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s->int_tri_tet4th_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, s->broken_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, s->node_scalar_ssbo);
        glBindVertexArray(s->int_vao);
        glDrawArrays(GL_TRIANGLES, 0, s->n_int_tri * 3);
        glBindVertexArray(0);
    }
}

/* ===========================================================
 * fem_sim_update_scalar
 * =========================================================== */
void fem_sim_update_scalar(FemSimGPU *s) {
    GLuint n = (GLuint)s->n_nodes;
    GLuint m = (GLuint)s->m_tets;

    if (s->scalar_mode == 1) {
        /* ---- mode 1: local F-norm per tet, then scatter-average to nodes ---- */
        glUseProgram(s->prog_f_norm_tet);
        glUniform1ui(s->uloc_fnorm_tet_m, m);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->pos_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->ci_ssbo[0]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->ci_ssbo[1]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s->ci_ssbo[2]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, s->ci_ssbo[3]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, s->D0inv_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, s->tet_scalar_ssbo);
        glDispatchCompute(dispatch_size(m), 1, 1);
        SSBO_BARRIER();

        glUseProgram(s->prog_f_norm_node);
        glUniform1ui(s->uloc_fnorm_node_n, n);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->tet_scalar_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->adj_start_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->adj_data_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s->node_scalar_ssbo);
        glDispatchCompute(dispatch_size(n), 1, 1);
        SSBO_BARRIER();
    } else if (s->scalar_mode == 2) {
        /* ---- mode 2: signed volumetric strain tr(E) per tet, scatter to nodes ---- */
        glUseProgram(s->prog_vol_strain_tet);
        glUniform1ui(s->uloc_vol_strain_tet_m, m);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->pos_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->ci_ssbo[0]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->ci_ssbo[1]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s->ci_ssbo[2]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, s->ci_ssbo[3]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, s->D0inv_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, s->tet_scalar_ssbo);
        glDispatchCompute(dispatch_size(m), 1, 1);
        SSBO_BARRIER();

        /* Reuse prog_f_norm_node — identical scatter-average logic */
        glUseProgram(s->prog_f_norm_node);
        glUniform1ui(s->uloc_fnorm_node_n, n);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->tet_scalar_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->adj_start_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->adj_data_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s->node_scalar_ssbo);
        glDispatchCompute(dispatch_size(n), 1, 1);
        SSBO_BARRIER();
    } else if (s->scalar_mode == 3) {
        /* ---- mode 3: accumulated plastic strain (acc_p per tet, scatter to nodes) ---- */
        /* Reuse SRC_F_NORM_NODE: takes tet_s at binding 0, outputs node_s at binding 3 */
        glUseProgram(s->prog_f_norm_node);
        glUniform1ui(s->uloc_fnorm_node_n, n);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->acc_p_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->adj_start_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->adj_data_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, s->node_scalar_ssbo);
        glDispatchCompute(dispatch_size(n), 1, 1);
        SSBO_BARRIER();
    } else {
        /* ---- mode 0: global displacement magnitude per node ---- */
        glUseProgram(s->prog_disp_scalar);
        glUniform1ui(s->uloc_disp_n, n);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->pos_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->rest_pos_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, s->node_scalar_ssbo);
        glDispatchCompute(dispatch_size(n), 1, 1);
        SSBO_BARRIER();
    }
}

/* ===========================================================
 * fem_sim_render_slice
 * =========================================================== */
void fem_sim_render_slice(FemSimGPU *s, const float *MVP) {
    glUseProgram(s->prog_slice);
    glUniformMatrix4fv(s->uloc_slice_MVP, 1, GL_FALSE, MVP);
    glUniform3fv(s->uloc_slice_pt, 1, s->slice_pt);
    glUniform3fv(s->uloc_slice_nrm, 1, s->slice_nrm);
    glUniform1f(s->uloc_slice_h,             s->slice_h);
    glUniform1f(s->uloc_slice_scalar_max,    s->scalar_max);
    glUniform1i(s->uloc_slice_scalar_signed, (s->scalar_mode == 2) ? 1 : 0); /* only vol-strain signed */

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, s->pos_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, s->node_scalar_ssbo);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glBindVertexArray(s->slice_vao);
    glDrawArrays(GL_POINTS, 0, s->n_nodes);
    glBindVertexArray(0);
    glDisable(GL_PROGRAM_POINT_SIZE);
}
