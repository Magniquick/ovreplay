// ovreplay recorder: an LD_PRELOAD OpenCL interposer.
//
// Loaded into an OpenVINO (or any OpenCL) process, it tracks every
// allocation, program, kernel and kernel argument. Between shim_start() and
// shim_stop() it logs every enqueue of one inference: kernel launches with
// their arguments, and host writes with their source pointers so inputs can
// be matched by name. Allocations are snapshotted at start; any whose
// contents change during the inference are marked DIRTY (scratch), the rest
// hold constants (weights). record.py drives it; pack.py turns the log into
// the format the ovreplay crate loads.
//
// Anything the replayer does not support is logged as UNSUPPORTED so the
// recording fails instead of replaying something wrong.
#define _GNU_SOURCE
#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static pthread_mutex_t mu = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
static void *libcl(void){ static void *h; if(!h) h = dlopen("libOpenCL.so.1", RTLD_NOW|RTLD_GLOBAL); return h; }
typedef void *(*dlsym_t)(void *, const char *);
static dlsym_t real_dlsym_fn(void){ static dlsym_t f; if(!f) f = (dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34"); if(!f) f = (dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"); return f; }
#define REAL(name) static __typeof__(&name) real_##name; if (!real_##name) real_##name = real_dlsym_fn()(libcl(), #name)

static const char *outdir = NULL; static FILE *lg = NULL; static int recording = 0;
static cl_context gctx; static cl_device_id gdev;

// ---- allocation table
typedef struct { int id; int kind; /*0 buf 1 dev 2 host 3 shared 4 subbuf*/ void *key; /*cl_mem or ptr*/ size_t size; cl_mem_flags flags; int parent; size_t origin; int live; } Alloc;
static Alloc *al; static int nal, cal;
static int add_alloc(int kind, void *key, size_t size, cl_mem_flags flags, int parent, size_t origin) {
  if (nal == cal) { cal = cal ? 2*cal : 256; al = realloc(al, cal*sizeof *al); }
  al[nal] = (Alloc){nal, kind, key, size, flags, parent, origin, 1};
  if (recording) fprintf(lg, "ALLOC %d %d %zu %lu %d %zu\n", nal, kind, size, (unsigned long)flags, parent, origin);
  return nal++;
}
static int find_mem(cl_mem m) { for (int i = nal-1; i >= 0; i--) if (al[i].live && al[i].kind != 1 && al[i].kind != 2 && al[i].kind != 3 && al[i].key == m) return i; return -1; }
static int find_ptr(const void *p, size_t *off) {
  for (int i = nal-1; i >= 0; i--) if (al[i].live && al[i].kind >= 1 && al[i].kind <= 3) {
    uintptr_t b = (uintptr_t)al[i].key; if ((uintptr_t)p >= b && (uintptr_t)p < b + al[i].size) { *off = (uintptr_t)p - b; return i; } }
  return -1;
}
// ---- programs / kernels
typedef struct { cl_program p; int id; int dumped; } Prog; static Prog *pr; static int npr, cpr;
typedef struct { int kind; /*0 unset 1 mem 2 usm 3 val 4 local 5 nullmem 6 hostptr-val*/ int alloc; size_t off; size_t size; unsigned char val[64]; } Arg;
typedef struct { cl_kernel k; int id; int prog; char name[256]; Arg args[64]; int nargs; int logged; } Kern; static Kern *kn; static int nkn, ckn;
static int prog_idx(cl_program p) { for (int i = npr-1; i >= 0; i--) if (pr[i].p == p) return i; return -1; }
static int kern_idx(cl_kernel k) { for (int i = nkn-1; i >= 0; i--) if (kn[i].k == k) return i; return -1; }
static void add_prog(cl_program p) { if (npr == cpr) { cpr = cpr ? 2*cpr : 256; pr = realloc(pr, cpr*sizeof *pr); } pr[npr] = (Prog){p, npr, 0}; npr++; }
static void add_kern(cl_kernel k, cl_program p, const char *name) {
  if (nkn == ckn) { ckn = ckn ? 2*ckn : 256; kn = realloc(kn, ckn*sizeof *kn); }
  Kern *K = &kn[nkn]; memset(K, 0, sizeof *K); K->k = k; K->id = nkn; K->prog = prog_idx(p); snprintf(K->name, sizeof K->name, "%s", name); nkn++;
}
static void dump_prog(int pi) {
  if (pr[pi].dumped) return; pr[pi].dumped = 1;
  REAL(clGetProgramInfo);
  size_t sz = 0; real_clGetProgramInfo(pr[pi].p, CL_PROGRAM_BINARY_SIZES, sizeof sz, &sz, NULL);
  unsigned char *b = malloc(sz); unsigned char *bp[1] = {b};
  real_clGetProgramInfo(pr[pi].p, CL_PROGRAM_BINARIES, sizeof bp, bp, NULL);
  char fn[512]; snprintf(fn, sizeof fn, "%s/prog_%d.bin", outdir, pi); FILE *f = fopen(fn, "wb"); fwrite(b, 1, sz, f); fclose(f); free(b);
  fprintf(lg, "PROG %d %zu\n", pi, sz);
}
static int blobn = 0;
static int write_blob(const void *p, size_t n) { char fn[512]; snprintf(fn, sizeof fn, "%s/blob_%d.bin", outdir, blobn); FILE *f = fopen(fn, "wb"); fwrite(p, 1, n, f); fclose(f); return blobn++; }

// ---- ext function pointers
static clDeviceMemAllocINTEL_fn r_dev; static clHostMemAllocINTEL_fn r_host; static clSharedMemAllocINTEL_fn r_shared;
static clMemFreeINTEL_fn r_free; static clMemBlockingFreeINTEL_fn r_bfree; static clSetKernelArgMemPointerINTEL_fn r_setptr;
static clEnqueueMemcpyINTEL_fn r_memcpy; static clEnqueueMemFillINTEL_fn r_fill; static clEnqueueMemsetINTEL_fn r_memset;

static void *w_dev(cl_context c, cl_device_id d, const cl_mem_properties_intel *pp, size_t s, cl_uint a, cl_int *e) { void *p = r_dev(c, d, pp, s, a, e); pthread_mutex_lock(&mu); if (p) add_alloc(1, p, s, 0, -1, 0); pthread_mutex_unlock(&mu); return p; }
static void *w_host(cl_context c, const cl_mem_properties_intel *pp, size_t s, cl_uint a, cl_int *e) { void *p = r_host(c, pp, s, a, e); pthread_mutex_lock(&mu); if (p) add_alloc(2, p, s, 0, -1, 0); pthread_mutex_unlock(&mu); return p; }
static void *w_shared(cl_context c, cl_device_id d, const cl_mem_properties_intel *pp, size_t s, cl_uint a, cl_int *e) { void *p = r_shared(c, d, pp, s, a, e); pthread_mutex_lock(&mu); if (p) add_alloc(3, p, s, 0, -1, 0); pthread_mutex_unlock(&mu); return p; }
static void free_ptr(void *p) { pthread_mutex_lock(&mu); size_t off; int i = find_ptr(p, &off); if (i >= 0 && off == 0) { al[i].live = 0; if (recording) fprintf(lg, "FREE %d\n", i); } pthread_mutex_unlock(&mu); }
static cl_int w_free(cl_context c, void *p) { free_ptr(p); return r_free(c, p); }
static cl_int w_bfree(cl_context c, void *p) { free_ptr(p); return r_bfree(c, p); }
static cl_int w_setptr(cl_kernel k, cl_uint idx, const void *p) {
  pthread_mutex_lock(&mu); int ki = kern_idx(k);
  if (ki >= 0 && idx < 64) { Arg *A = &kn[ki].args[idx]; size_t off = 0; int a = p ? find_ptr(p, &off) : -1;
    if (!p) { A->kind = 5; } else if (a >= 0) { A->kind = 2; A->alloc = a; A->off = off; } else { A->kind = 6; memcpy(A->val, &p, 8); A->size = 8; fprintf(stderr, "shim: unknown usm ptr arg %s[%u]\n", kn[ki].name, idx); }
    if ((int)idx >= kn[ki].nargs) kn[ki].nargs = idx+1; }
  pthread_mutex_unlock(&mu); return r_setptr(k, idx, p);
}
static void log_host_write(int dst, size_t off, const void *src, size_t n) { int b = write_blob(src, n); fprintf(lg, "WRITE %d %zu %zu %d %p\n", dst, off, n, b, src); if (dst < 0) fprintf(lg, "UNSUPPORTED write to unknown allocation\n"); }
static cl_int w_memcpy(cl_command_queue q, cl_bool bl, void *dst, const void *src, size_t n, cl_uint ne, const cl_event *ev, cl_event *oe) {
  pthread_mutex_lock(&mu);
  if (recording) { size_t od, os; int d = find_ptr(dst, &od), s = find_ptr(src, &os);
    if (d >= 0 && s >= 0) fprintf(lg, "COPY %d %zu %d %zu %zu\n", d, od, s, os, n);
    else if (d >= 0) { REAL(clFinish); real_clFinish(q); log_host_write(d, od, src, n); }
    else if (s >= 0) fprintf(lg, "READ %d %zu %zu\n", s, os, n);
    else fprintf(lg, "UNSUPPORTED host-host memcpy %zu\n", n); }
  pthread_mutex_unlock(&mu); return r_memcpy(q, bl, dst, src, n, ne, ev, oe);
}
static cl_int w_fill(cl_command_queue q, void *dst, const void *pat, size_t ps, size_t n, cl_uint ne, const cl_event *ev, cl_event *oe) {
  pthread_mutex_lock(&mu);
  if (recording) { size_t od; int d = find_ptr(dst, &od); fprintf(lg, "FILL %d %zu %zu %zu", d, od, n, ps); for (size_t i = 0; i < ps; i++) fprintf(lg, " %02x", ((unsigned char*)pat)[i]); fprintf(lg, "\n"); }
  pthread_mutex_unlock(&mu); return r_fill(q, dst, pat, ps, n, ne, ev, oe);
}
static cl_int w_memset(cl_command_queue q, void *dst, cl_int v, size_t n, cl_uint ne, const cl_event *ev, cl_event *oe) {
  pthread_mutex_lock(&mu);
  if (recording) { size_t od; int d = find_ptr(dst, &od); fprintf(lg, "FILL %d %zu %zu 1 %02x\n", d, od, n, v & 0xff); }
  pthread_mutex_unlock(&mu); return r_memset(q, dst, v, n, ne, ev, oe);
}
#define HOOKEXT(nm, var, w) if (!strcmp(name, #nm)) { var = (void*)r; return (void*)w; }
static void *wrap_ext(const char *name, void *r) {
  if (!r) return r;
  HOOKEXT(clDeviceMemAllocINTEL, r_dev, w_dev) HOOKEXT(clHostMemAllocINTEL, r_host, w_host) HOOKEXT(clSharedMemAllocINTEL, r_shared, w_shared)
  HOOKEXT(clMemFreeINTEL, r_free, w_free) HOOKEXT(clMemBlockingFreeINTEL, r_bfree, w_bfree) HOOKEXT(clSetKernelArgMemPointerINTEL, r_setptr, w_setptr)
  HOOKEXT(clEnqueueMemcpyINTEL, r_memcpy, w_memcpy) HOOKEXT(clEnqueueMemFillINTEL, r_fill, w_fill) HOOKEXT(clEnqueueMemsetINTEL, r_memset, w_memset)
  if (getenv("SHIM_VERBOSE")) fprintf(stderr, "shim: ext %s passthrough\n", name);
  return r;
}
CL_API_ENTRY void *CL_API_CALL clGetExtensionFunctionAddressForPlatform(cl_platform_id p, const char *name) { REAL(clGetExtensionFunctionAddressForPlatform); return wrap_ext(name, real_clGetExtensionFunctionAddressForPlatform(p, name)); }
CL_API_ENTRY void *CL_API_CALL clGetExtensionFunctionAddress(const char *name) { REAL(clGetExtensionFunctionAddress); return wrap_ext(name, real_clGetExtensionFunctionAddress(name)); }

// ---- core hooks
CL_API_ENTRY cl_context CL_API_CALL clCreateContext(const cl_context_properties *pp, cl_uint n, const cl_device_id *d, void (CL_CALLBACK *cb)(const char *, const void *, size_t, void *), void *ud, cl_int *e) {
  REAL(clCreateContext); cl_context c = real_clCreateContext(pp, n, d, cb, ud, e); gctx = c; gdev = d[0]; return c; }
CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(cl_context c, cl_mem_flags fl, size_t s, void *hp, cl_int *e) {
  REAL(clCreateBuffer); cl_mem m = real_clCreateBuffer(c, fl, s, hp, e);
  pthread_mutex_lock(&mu); if (m) { int i = add_alloc(0, m, s, fl, -1, 0); if (hp && (fl & CL_MEM_USE_HOST_PTR) && getenv("SHIM_VERBOSE")) fprintf(stderr, "shim: USE_HOST_PTR buffer %d\n", i); } pthread_mutex_unlock(&mu); return m; }
CL_API_ENTRY cl_mem CL_API_CALL clCreateSubBuffer(cl_mem b, cl_mem_flags fl, cl_buffer_create_type t, const void *info, cl_int *e) {
  REAL(clCreateSubBuffer); cl_mem m = real_clCreateSubBuffer(b, fl, t, info, e);
  pthread_mutex_lock(&mu); if (m) { const cl_buffer_region *r = info; add_alloc(4, m, r->size, fl, find_mem(b), r->origin); } pthread_mutex_unlock(&mu); return m; }
CL_API_ENTRY cl_mem CL_API_CALL clCreateImage(cl_context c, cl_mem_flags fl, const cl_image_format *f, const cl_image_desc *d, void *hp, cl_int *e) {
  REAL(clCreateImage); pthread_mutex_lock(&mu); if (lg) fprintf(lg, "UNSUPPORTED clCreateImage\n"); pthread_mutex_unlock(&mu); return real_clCreateImage(c, fl, f, d, hp, e); }
CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(cl_mem m) {
  REAL(clReleaseMemObject); REAL(clGetMemObjectInfo); cl_uint rc = 0; real_clGetMemObjectInfo(m, CL_MEM_REFERENCE_COUNT, sizeof rc, &rc, NULL);
  if (rc == 1) { pthread_mutex_lock(&mu); int i = find_mem(m); if (i >= 0) { al[i].live = 0; if (recording) fprintf(lg, "FREE %d\n", i); } pthread_mutex_unlock(&mu); }
  return real_clReleaseMemObject(m); }
CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource(cl_context c, cl_uint n, const char **s, const size_t *l, cl_int *e) {
  REAL(clCreateProgramWithSource); cl_program p = real_clCreateProgramWithSource(c, n, s, l, e); pthread_mutex_lock(&mu); if (p) add_prog(p); pthread_mutex_unlock(&mu); return p; }
CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithBinary(cl_context c, cl_uint n, const cl_device_id *d, const size_t *l, const unsigned char **b, cl_int *st, cl_int *e) {
  REAL(clCreateProgramWithBinary); cl_program p = real_clCreateProgramWithBinary(c, n, d, l, b, st, e); pthread_mutex_lock(&mu); if (p) add_prog(p); pthread_mutex_unlock(&mu); return p; }
CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(cl_program p, const char *name, cl_int *e) {
  REAL(clCreateKernel); cl_kernel k = real_clCreateKernel(p, name, e); pthread_mutex_lock(&mu); if (k) add_kern(k, p, name); pthread_mutex_unlock(&mu); return k; }
CL_API_ENTRY cl_int CL_API_CALL clCreateKernelsInProgram(cl_program p, cl_uint n, cl_kernel *ks, cl_uint *nr) {
  REAL(clCreateKernelsInProgram); REAL(clGetKernelInfo); cl_int r = real_clCreateKernelsInProgram(p, n, ks, nr);
  if (r == CL_SUCCESS && ks) { cl_uint cnt = nr ? *nr : n; pthread_mutex_lock(&mu); for (cl_uint i = 0; i < cnt; i++) { char nm[256] = {0}; real_clGetKernelInfo(ks[i], CL_KERNEL_FUNCTION_NAME, sizeof nm, nm, NULL); add_kern(ks[i], p, nm); } pthread_mutex_unlock(&mu); }
  return r; }
CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(cl_kernel k, cl_uint idx, size_t sz, const void *v) {
  REAL(clSetKernelArg); pthread_mutex_lock(&mu); int ki = kern_idx(k);
  if (ki >= 0 && idx < 64) { Arg *A = &kn[ki].args[idx]; int mi;
    if (!v) { A->kind = 4; A->size = sz; }
    else if (sz == sizeof(cl_mem) && (mi = find_mem(*(cl_mem*)v)) >= 0) { A->kind = 1; A->alloc = mi; A->off = 0; }
    else if (sz == sizeof(cl_mem) && *(void**)v == NULL) { A->kind = 5; }
    else { A->kind = 3; A->size = sz; if (sz > 64) fprintf(stderr, "shim: big scalar arg %zu\n", sz); memcpy(A->val, v, sz > 64 ? 64 : sz);
      if (sz == 8) { size_t off; void *pv = *(void**)v; if (find_ptr(pv, &off) >= 0) fprintf(stderr, "shim: scalar looks like usm ptr in %s[%u]\n", kn[ki].name, idx); } }
    if ((int)idx >= kn[ki].nargs) kn[ki].nargs = idx+1; }
  pthread_mutex_unlock(&mu); return real_clSetKernelArg(k, idx, sz, v); }
CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(cl_command_queue q, cl_kernel k, cl_uint dim, const size_t *go, const size_t *gs, const size_t *ls, cl_uint ne, const cl_event *ev, cl_event *oe) {
  REAL(clEnqueueNDRangeKernel); pthread_mutex_lock(&mu);
  if (recording) { int ki = kern_idx(k);
    if (ki < 0) fprintf(lg, "UNSUPPORTED unknown kernel\n"); else { Kern *K = &kn[ki];
      if (K->prog >= 0) dump_prog(K->prog);
      if (!K->logged) { fprintf(lg, "KERN %d %d %s\n", ki, K->prog, K->name); K->logged = 1; }
      fprintf(lg, "NDR %d %u", ki, dim); for (cl_uint i = 0; i < 3; i++) fprintf(lg, " %zu", i < dim ? gs[i] : 1);
      for (cl_uint i = 0; i < 3; i++) fprintf(lg, " %zu", ls && i < dim ? ls[i] : 0);
      for (cl_uint i = 0; i < 3; i++) fprintf(lg, " %zu", go && i < dim ? go[i] : 0);
      fprintf(lg, " %d\n", K->nargs);
      for (int i = 0; i < K->nargs; i++) { Arg *A = &K->args[i];
        switch (A->kind) { case 1: fprintf(lg, " M %d\n", A->alloc); break; case 2: fprintf(lg, " U %d %zu\n", A->alloc, A->off); break;
          case 4: fprintf(lg, " L %zu\n", A->size); break; case 5: fprintf(lg, " N\n"); break;
          case 3: case 6: fprintf(lg, " V %zu", A->size); for (size_t j = 0; j < A->size && j < 64; j++) fprintf(lg, " %02x", A->val[j]); fprintf(lg, "\n"); break;
          default: fprintf(lg, " X\n"); fprintf(lg, "UNSUPPORTED unset arg %d of %s\n", i, K->name); } } } }
  pthread_mutex_unlock(&mu); return real_clEnqueueNDRangeKernel(q, k, dim, go, gs, ls, ne, ev, oe); }
CL_API_ENTRY cl_int CL_API_CALL clEnqueueWriteBuffer(cl_command_queue q, cl_mem b, cl_bool bl, size_t off, size_t n, const void *p, cl_uint ne, const cl_event *ev, cl_event *oe) {
  REAL(clEnqueueWriteBuffer); pthread_mutex_lock(&mu); if (recording) log_host_write(find_mem(b), off, p, n); pthread_mutex_unlock(&mu); return real_clEnqueueWriteBuffer(q, b, bl, off, n, p, ne, ev, oe); }
CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue q, cl_mem b, cl_bool bl, size_t off, size_t n, void *p, cl_uint ne, const cl_event *ev, cl_event *oe) {
  REAL(clEnqueueReadBuffer); pthread_mutex_lock(&mu); if (recording) fprintf(lg, "READ %d %zu %zu\n", find_mem(b), off, n); pthread_mutex_unlock(&mu); return real_clEnqueueReadBuffer(q, b, bl, off, n, p, ne, ev, oe); }
CL_API_ENTRY cl_int CL_API_CALL clEnqueueCopyBuffer(cl_command_queue q, cl_mem s, cl_mem d, size_t so, size_t dof, size_t n, cl_uint ne, const cl_event *ev, cl_event *oe) {
  REAL(clEnqueueCopyBuffer); pthread_mutex_lock(&mu); if (recording) fprintf(lg, "COPY %d %zu %d %zu %zu\n", find_mem(d), dof, find_mem(s), so, n); pthread_mutex_unlock(&mu); return real_clEnqueueCopyBuffer(q, s, d, so, dof, n, ne, ev, oe); }
CL_API_ENTRY cl_int CL_API_CALL clEnqueueFillBuffer(cl_command_queue q, cl_mem b, const void *pat, size_t ps, size_t off, size_t n, cl_uint ne, const cl_event *ev, cl_event *oe) {
  REAL(clEnqueueFillBuffer); pthread_mutex_lock(&mu); if (recording) { fprintf(lg, "FILL %d %zu %zu %zu", find_mem(b), off, n, ps); for (size_t i = 0; i < ps; i++) fprintf(lg, " %02x", ((unsigned char*)pat)[i]); fprintf(lg, "\n"); } pthread_mutex_unlock(&mu); return real_clEnqueueFillBuffer(q, b, pat, ps, off, n, ne, ev, oe); }
CL_API_ENTRY void *CL_API_CALL clEnqueueMapBuffer(cl_command_queue q, cl_mem b, cl_bool bl, cl_map_flags f, size_t off, size_t n, cl_uint ne, const cl_event *ev, cl_event *oe, cl_int *e) {
  REAL(clEnqueueMapBuffer); pthread_mutex_lock(&mu); if (recording) fprintf(lg, "UNSUPPORTED MAP %d %zu %zu flags %lu\n", find_mem(b), off, n, (unsigned long)f); pthread_mutex_unlock(&mu); return real_clEnqueueMapBuffer(q, b, bl, f, off, n, ne, ev, oe, e); }

// ---- control, called from the host program via dlsym
static cl_command_queue myq; static int snapb[16384]; static int snapn;
void shim_start(const char *dir) {
  pthread_mutex_lock(&mu); outdir = strdup(dir); char fn[512]; snprintf(fn, sizeof fn, "%s/rec.txt", dir); lg = fopen(fn, "w");
  REAL(clCreateCommandQueueWithProperties); REAL(clEnqueueReadBuffer); REAL(clFinish);
  REAL(clGetKernelInfo); REAL(clGetContextInfo);
  if (nkn) { real_clGetKernelInfo(kn[nkn-1].k, CL_KERNEL_CONTEXT, sizeof gctx, &gctx, NULL); real_clGetContextInfo(gctx, CL_CONTEXT_DEVICES, sizeof gdev, &gdev, NULL); }
  { REAL(clGetDeviceInfo); char v[512];
    real_clGetDeviceInfo(gdev, CL_DEVICE_NAME, sizeof v, v, NULL); fprintf(lg, "DEVICE_NAME %s\n", v);
    real_clGetDeviceInfo(gdev, CL_DRIVER_VERSION, sizeof v, v, NULL); fprintf(lg, "DRIVER_VERSION %s\n", v);
    cl_uint id = 0; if (real_clGetDeviceInfo(gdev, 0x4251 /* CL_DEVICE_ID_INTEL */, sizeof id, &id, NULL) == CL_SUCCESS) fprintf(lg, "DEVICE_ID %u\n", id); }
  cl_int qe = 0; myq = real_clCreateCommandQueueWithProperties(gctx, gdev, NULL, &qe);
  for (int i = 0; i < 16384; i++) snapb[i] = -1; snapn = nal;
  // snapshot every live allocation
  for (int i = 0; i < nal; i++) if (al[i].live) {
    fprintf(lg, "ALLOC %d %d %zu %lu %d %zu\n", i, al[i].kind, al[i].size, (unsigned long)al[i].flags, al[i].parent, al[i].origin);
    if (al[i].kind == 4) continue;
    void *h = malloc(al[i].size);
    if (al[i].kind == 0) real_clEnqueueReadBuffer(myq, al[i].key, CL_TRUE, 0, al[i].size, h, 0, NULL, NULL);
    else r_memcpy(myq, CL_TRUE, h, al[i].key, al[i].size, 0, NULL, NULL);
    int b = write_blob(h, al[i].size); free(h); fprintf(lg, "SNAP %d %d\n", i, b); snapb[i] = b; }
  real_clFinish(myq); recording = 1; fprintf(lg, "START\n"); pthread_mutex_unlock(&mu);
}
void shim_stop(void) { pthread_mutex_lock(&mu); recording = 0; fprintf(lg, "STOP\n");
  REAL(clEnqueueReadBuffer); REAL(clFinish); real_clFinish(myq);
  // allocations whose contents changed during the recorded inference are scratch
  for (int i = 0; i < nal && i < snapn; i++) if (al[i].live && snapb[i] >= 0) {
    void *h = malloc(al[i].size), *o = malloc(al[i].size); char fn[512]; snprintf(fn, sizeof fn, "%s/blob_%d.bin", outdir, snapb[i]);
    FILE *f = fopen(fn, "rb"); size_t got = fread(o, 1, al[i].size, f); fclose(f); (void)got;
    if (al[i].kind == 0) real_clEnqueueReadBuffer(myq, al[i].key, CL_TRUE, 0, al[i].size, h, 0, NULL, NULL);
    else r_memcpy(myq, CL_TRUE, h, al[i].key, al[i].size, 0, NULL, NULL);
    if (memcmp(h, o, al[i].size)) fprintf(lg, "DIRTY %d\n", i);
    free(h); free(o); } fclose(lg); lg = NULL; pthread_mutex_unlock(&mu); }

// The allocation holding host pointer `p` (an OpenVINO output tensor), or -1.
int shim_locate(const void *p, size_t *off) { pthread_mutex_lock(&mu); int i = find_ptr(p, off); pthread_mutex_unlock(&mu); return i; }

// oneDNN resolves OpenCL through dlopen+dlsym; route those names to our hooks.
static const char *hooked[] = {"clGetExtensionFunctionAddressForPlatform","clGetExtensionFunctionAddress","clCreateContext","clCreateBuffer","clCreateSubBuffer","clCreateImage","clReleaseMemObject","clCreateProgramWithSource","clCreateProgramWithBinary","clCreateKernel","clCreateKernelsInProgram","clSetKernelArg","clEnqueueNDRangeKernel","clEnqueueWriteBuffer","clEnqueueReadBuffer","clEnqueueCopyBuffer","clEnqueueFillBuffer","clEnqueueMapBuffer","clCloneKernel","clCreateProgramWithIL","clLinkProgram","clCreateContextFromType",NULL};
static void *self_handle(void){ static void *h; if(!h){ Dl_info di; dladdr((void*)&self_handle, &di); h = dlopen(di.dli_fname, RTLD_NOW|RTLD_NOLOAD); } return h; }
void *dlsym(void *h, const char *name) {
  void *r = real_dlsym_fn()(h, name);
  if (name && name[0]==0x63 && name[1]==0x6c && (h == libcl() || h == RTLD_DEFAULT || h == RTLD_NEXT)) { for (int i = 0; hooked[i]; i++) if (!strcmp(hooked[i], name)) { void *mine = real_dlsym_fn()(self_handle(), name); if (mine && r) return mine; } 
    if (getenv("SHIM_VERBOSE") && r) fprintf(stderr, "shim: dlsym %s\n", name); }
  return r;
}
