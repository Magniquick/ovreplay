//! The slice of the `OpenCL` C API the replayer uses, as owning handles.
//!
//! Only what a static, image-free, USM-only trace needs: a context and queue,
//! programs built from recorded binaries, kernels, USM allocations, and the
//! Intel USM extension entry points resolved once at startup.

use std::ffi::{CStr, CString, c_void};
use std::ptr;

use opencl_sys as sys;

#[derive(Debug, thiserror::Error)]
#[error("{call} failed: {code}")]
pub struct ClError {
    pub call: &'static str,
    pub code: i32,
}

const fn check(call: &'static str, code: sys::cl_int) -> Result<(), ClError> {
    if code == sys::CL_SUCCESS {
        Ok(())
    } else {
        Err(ClError { call, code })
    }
}

/// The Intel USM entry points, resolved through the platform.
struct Usm {
    device_alloc: sys::clDeviceMemAllocINTEL_t,
    host_alloc: sys::clHostMemAllocINTEL_t,
    memcpy: sys::clEnqueueMemcpyINTEL_t,
    set_arg_ptr: sys::clSetKernelArgMemPointerINTEL_t,
    free: sys::clMemBlockingFreeINTEL_t,
}

impl Usm {
    fn resolve(platform: sys::cl_platform_id) -> Result<Self, ClError> {
        // SAFETY: valid platform; each name is a NUL-terminated literal. A
        // missing entry point returns null, caught below.
        unsafe {
            let get = |name: &CStr| sys::clGetExtensionFunctionAddressForPlatform(platform, name.as_ptr());
            let device_alloc = get(c"clDeviceMemAllocINTEL");
            let host_alloc = get(c"clHostMemAllocINTEL");
            let memcpy = get(c"clEnqueueMemcpyINTEL");
            let set_arg_ptr = get(c"clSetKernelArgMemPointerINTEL");
            let free = get(c"clMemBlockingFreeINTEL");
            if device_alloc.is_null() || host_alloc.is_null() || memcpy.is_null() || set_arg_ptr.is_null()
                || free.is_null()
            {
                return Err(ClError { call: "clGetExtensionFunctionAddressForPlatform", code: 0 });
            }
            Ok(Self {
                device_alloc: std::mem::transmute::<*mut c_void, sys::clDeviceMemAllocINTEL_t>(device_alloc),
                host_alloc: std::mem::transmute::<*mut c_void, sys::clHostMemAllocINTEL_t>(host_alloc),
                memcpy: std::mem::transmute::<*mut c_void, sys::clEnqueueMemcpyINTEL_t>(memcpy),
                set_arg_ptr: std::mem::transmute::<*mut c_void, sys::clSetKernelArgMemPointerINTEL_t>(set_arg_ptr),
                free: std::mem::transmute::<*mut c_void, sys::clMemBlockingFreeINTEL_t>(free),
            })
        }
    }
}

/// A USM allocation, freed on drop.
pub struct Alloc {
    ptr: *mut c_void,
    ctx: sys::cl_context,
    free: sys::clMemBlockingFreeINTEL_t,
}

// SAFETY: exclusively owned USM; freed once, on drop, by its single owner.
unsafe impl Send for Alloc {}

impl Alloc {
    #[must_use]
    pub const fn ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for Alloc {
    fn drop(&mut self) {
        if let Some(free) = self.free {
            // SAFETY: `ptr` came from the matching alloc on `ctx`.
            unsafe { free(self.ctx, self.ptr) };
        }
    }
}

pub struct Context {
    device: sys::cl_device_id,
    ctx: sys::cl_context,
    queue: sys::cl_command_queue,
    usm: Usm,
}

// SAFETY: the handles are used only through `&mut`/`&` on a single owner; the
// OpenCL runtime allows their use from any one thread at a time.
unsafe impl Send for Context {}

impl Context {
    /// The first GPU device on any platform, with a context and in-order queue.
    pub fn first_gpu() -> Result<Self, ClError> {
        // SAFETY: standard enumeration; out-params sized correctly.
        unsafe {
            let mut platforms: [sys::cl_platform_id; 16] = [ptr::null_mut(); 16];
            let mut n: sys::cl_uint = 0;
            check("clGetPlatformIDs", sys::clGetPlatformIDs(16, platforms.as_mut_ptr(), &raw mut n))?;
            for &platform in platforms.iter().take(usize::try_from(n).unwrap_or(0)) {
                let mut device = ptr::null_mut();
                let mut nd = 0;
                let e = sys::clGetDeviceIDs(platform, sys::CL_DEVICE_TYPE_GPU, 1, &raw mut device, &raw mut nd);
                if e != sys::CL_SUCCESS || nd == 0 {
                    continue;
                }
                let mut err = 0;
                let ctx = sys::clCreateContext(ptr::null(), 1, &raw mut device, None, ptr::null_mut(), &raw mut err);
                check("clCreateContext", err)?;
                let queue = sys::clCreateCommandQueueWithProperties(ctx, device, ptr::null(), &raw mut err);
                check("clCreateCommandQueueWithProperties", err)?;
                let usm = Usm::resolve(platform)?;
                return Ok(Self { device, ctx, queue, usm });
            }
            Err(ClError { call: "clGetDeviceIDs", code: sys::CL_DEVICE_NOT_FOUND })
        }
    }

    /// A device-info string, e.g. `CL_DRIVER_VERSION`.
    pub fn device_string(&self, param: sys::cl_device_info) -> Result<String, ClError> {
        // SAFETY: valid device; buffer sized to the reported length.
        unsafe {
            let mut len = 0;
            check("clGetDeviceInfo", sys::clGetDeviceInfo(self.device, param, 0, ptr::null_mut(), &raw mut len))?;
            let mut buf = vec![0u8; len];
            check("clGetDeviceInfo", sys::clGetDeviceInfo(self.device, param, len, buf.as_mut_ptr().cast(), &raw mut len))?;
            Ok(String::from_utf8_lossy(&buf).trim_end_matches('\0').to_owned())
        }
    }

    /// A `cl_uint` device-info value, or `None` when the device does not
    /// report it (e.g. `CL_DEVICE_ID_INTEL` on a non-Intel device).
    pub fn device_uint(&self, param: sys::cl_device_info) -> Option<u32> {
        let mut value: sys::cl_uint = 0;
        // SAFETY: valid device; the out-param is a cl_uint of the size passed.
        let e = unsafe {
            sys::clGetDeviceInfo(self.device, param, size_of::<sys::cl_uint>(), (&raw mut value).cast(), ptr::null_mut())
        };
        (e == sys::CL_SUCCESS).then_some(value)
    }

    pub fn device_alloc(&self, size: usize) -> Result<Alloc, ClError> {
        let alloc = self.usm.device_alloc.ok_or(ClError { call: "clDeviceMemAllocINTEL", code: 0 })?;
        let mut err = 0;
        // SAFETY: valid context and device; 0 flags, default alignment.
        let ptr = unsafe { alloc(self.ctx, self.device, ptr::null(), size.max(1), 0, &raw mut err) };
        check("clDeviceMemAllocINTEL", err)?;
        Ok(Alloc { ptr, ctx: self.ctx, free: self.usm.free })
    }

    pub fn host_alloc(&self, size: usize) -> Result<Alloc, ClError> {
        let alloc = self.usm.host_alloc.ok_or(ClError { call: "clHostMemAllocINTEL", code: 0 })?;
        let mut err = 0;
        // SAFETY: valid context; 0 flags, default alignment.
        let ptr = unsafe { alloc(self.ctx, ptr::null(), size.max(1), 0, &raw mut err) };
        check("clHostMemAllocINTEL", err)?;
        Ok(Alloc { ptr, ctx: self.ctx, free: self.usm.free })
    }

    /// Blocking USM copy `src` -> `dst` of `n` bytes.
    ///
    /// # Safety
    ///
    /// `dst` and `src` must be valid USM or host pointers for `n` bytes.
    pub unsafe fn copy(&self, dst: *mut c_void, src: *const c_void, n: usize) -> Result<(), ClError> {
        let memcpy = self.usm.memcpy.ok_or(ClError { call: "clEnqueueMemcpyINTEL", code: 0 })?;
        // SAFETY: caller guarantees the pointers; blocking, no wait list.
        check("clEnqueueMemcpyINTEL", unsafe {
            memcpy(self.queue, sys::CL_TRUE, dst, src, n, 0, ptr::null(), ptr::null_mut())
        })
    }

    pub fn finish(&self) -> Result<(), ClError> {
        // SAFETY: valid queue.
        check("clFinish", unsafe { sys::clFinish(self.queue) })
    }

    /// Build a program from a recorded device binary.
    pub fn program_from_binary(&self, binary: &[u8]) -> Result<Program, ClError> {
        let mut len = binary.len();
        let mut ptr_bin = binary.as_ptr();
        let (mut err, mut status) = (0, 0);
        // SAFETY: one device, one binary of `len` bytes.
        let program = unsafe {
            sys::clCreateProgramWithBinary(self.ctx, 1, &raw const self.device, &raw mut len, &raw mut ptr_bin, &raw mut status, &raw mut err)
        };
        check("clCreateProgramWithBinary", err)?;
        check("clCreateProgramWithBinary(status)", status)?;
        // SAFETY: valid program and device; no options, no callback.
        check("clBuildProgram", unsafe {
            sys::clBuildProgram(program, 1, &raw const self.device, c"".as_ptr(), None, ptr::null_mut())
        })?;
        Ok(Program(program))
    }

    pub fn kernel(program: &Program, name: &str) -> Result<Kernel, ClError> {
        let cname = CString::new(name).map_err(|_| ClError { call: "clCreateKernel", code: 0 })?;
        let mut err = 0;
        // SAFETY: valid program; NUL-terminated name.
        let kernel = unsafe { sys::clCreateKernel(program.0, cname.as_ptr(), &raw mut err) };
        check("clCreateKernel", err)?;
        Ok(Kernel(kernel))
    }

    /// Bind a USM pointer argument.
    pub fn set_arg_ptr(&self, kernel: &Kernel, index: u32, ptr: *const c_void) -> Result<(), ClError> {
        let set = self.usm.set_arg_ptr.ok_or(ClError { call: "clSetKernelArgMemPointerINTEL", code: 0 })?;
        // SAFETY: valid kernel; `ptr` is a live USM pointer for this context.
        check("clSetKernelArgMemPointerINTEL", unsafe { set(kernel.0, index, ptr) })
    }

    /// Bind a by-value scalar argument.
    pub fn set_arg_value(kernel: &Kernel, index: u32, bytes: &[u8]) -> Result<(), ClError> {
        // SAFETY: valid kernel; `bytes` is read for `len` bytes.
        check("clSetKernelArg", unsafe {
            sys::clSetKernelArg(kernel.0, index, bytes.len(), bytes.as_ptr().cast())
        })
    }

    /// Bind `size` bytes of local memory.
    pub fn set_arg_local(kernel: &Kernel, index: u32, size: usize) -> Result<(), ClError> {
        // SAFETY: valid kernel; a null value with a size requests local memory.
        check("clSetKernelArg", unsafe { sys::clSetKernelArg(kernel.0, index, size, ptr::null()) })
    }

    /// Bind a null global pointer.
    pub fn set_arg_null(kernel: &Kernel, index: u32) -> Result<(), ClError> {
        let null: sys::cl_mem = ptr::null_mut();
        // SAFETY: valid kernel; a pointer to a null `cl_mem` binds a null buffer.
        check("clSetKernelArg", unsafe {
            sys::clSetKernelArg(kernel.0, index, size_of::<sys::cl_mem>(), (&raw const null).cast())
        })
    }

    /// Enqueue `kernel` over `global`, with `local` when non-zero and `offset`.
    pub fn launch(&self, kernel: &Kernel, dim: u32, global: &[usize; 3], local: &[usize; 3], offset: &[usize; 3])
        -> Result<(), ClError> {
        let local_ptr = if local[0] == 0 { ptr::null() } else { local.as_ptr() };
        // SAFETY: valid kernel and queue; the arrays hold `dim` values.
        check("clEnqueueNDRangeKernel", unsafe {
            sys::clEnqueueNDRangeKernel(self.queue, kernel.0, dim, offset.as_ptr(), global.as_ptr(), local_ptr, 0,
                ptr::null(), ptr::null_mut())
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: owned handles created above.
        unsafe {
            sys::clReleaseCommandQueue(self.queue);
            sys::clReleaseContext(self.ctx);
        }
    }
}

pub struct Program(sys::cl_program);

// SAFETY: exclusively owned program object.
unsafe impl Send for Program {}

impl Drop for Program {
    fn drop(&mut self) {
        // SAFETY: owned program.
        unsafe { sys::clReleaseProgram(self.0) };
    }
}

pub struct Kernel(sys::cl_kernel);

// SAFETY: exclusively owned kernel object, used by one owner at a time.
unsafe impl Send for Kernel {}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: owned kernel.
        unsafe { sys::clReleaseKernel(self.0) };
    }
}
