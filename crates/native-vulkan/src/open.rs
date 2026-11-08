//! Finding and opening a GPU: the instance, the choice among its devices and
//! the logical device with half floats enabled.

use std::ffi::{CStr, c_void};
use std::ptr::{null, null_mut};

use crate::check;
use crate::ffi::{self, Handle};

/// Invocations of the largest workgroup the models' shaders declare, all
/// along x. Vulkan guarantees only 128.
const INVOCATIONS: u32 = 256;

/// An instance, destroyed when dropped.
pub(crate) struct Instance {
    pub(crate) handle: Handle,
    pub(crate) i: ffi::instance::Functions,
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: the device made from it, if any, is already destroyed: it
        // is owned by the `Device` holding this instance, dropped first.
        unsafe { (self.i.destroy_instance)(self.handle, null()) };
    }
}

/// The chosen physical device.
pub(crate) struct Choice {
    pub(crate) handle: Handle,
    pub(crate) family: u32,
    pub(crate) props: Box<ffi::PhysicalDeviceProperties>,
}

/// A device's name as its driver gives it.
pub(crate) fn device_name(props: &ffi::PhysicalDeviceProperties) -> String {
    let bytes: Vec<u8> = props.device_name.iter().map(|&c| c as u8).collect();
    CStr::from_bytes_until_nul(&bytes).map_or_else(
        |_| String::from_utf8_lossy(&bytes).into_owned(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// The instance, made through the system's Vulkan loader.
pub(crate) fn create_instance() -> Result<Instance, String> {
    let get = ffi::loader()?;
    // SAFETY: vkGetInstanceProcAddr with a null instance returns global
    // commands, vkCreateInstance among them, or null.
    let create =
        unsafe { get(null_mut(), c"vkCreateInstance".as_ptr()) }.ok_or("no vkCreateInstance")?;
    // SAFETY: that is vkCreateInstance's signature.
    let create: unsafe extern "system" fn(
        *const ffi::InstanceCreateInfo,
        *const c_void,
        *mut Handle,
    ) -> i32 = unsafe { std::mem::transmute(create) };
    let app = ffi::ApplicationInfo {
        s_type: ffi::TYPE_APPLICATION_INFO,
        p_next: null(),
        application_name: c"image-engine-native".as_ptr(),
        application_version: 0,
        engine_name: c"image-engine-native".as_ptr(),
        engine_version: 0,
        api_version: ffi::API_1_2,
    };
    let instance_info = ffi::InstanceCreateInfo {
        s_type: ffi::TYPE_INSTANCE_CREATE_INFO,
        p_next: null(),
        flags: 0,
        application_info: &app,
        layer_count: 0,
        layers: null(),
        extension_count: 0,
        extensions: null(),
    };
    let mut handle = null_mut();
    // SAFETY: `instance_info` and what it points to outlive the call.
    check("creating an instance", unsafe {
        create(&instance_info, null(), &mut handle)
    })?;
    // SAFETY: the instance is valid, and vkGetInstanceProcAddr returns its
    // commands or null.
    let i = unsafe { ffi::instance::Functions::load(|name| get(handle, name.as_ptr())) };
    match i {
        Ok(i) => Ok(Instance { handle, i }),
        Err(e) => {
            // SAFETY: as above; vkDestroyInstance has this signature.
            unsafe {
                if let Some(f) = get(handle, c"vkDestroyInstance".as_ptr()) {
                    let destroy: unsafe extern "system" fn(Handle, *const c_void) =
                        std::mem::transmute(f);
                    destroy(handle, null());
                }
            }
            Err(e)
        }
    }
}

/// The best GPU of `instance` for the engine, and a queue family on it with
/// compute, or why there is none.
pub(crate) fn choose(instance: &Instance) -> Result<Choice, String> {
    let (i, h) = (&instance.i, instance.handle);
    let mut count = 0;
    // SAFETY: the two-call enumeration, the second with room for the count
    // the first gave.
    let devices = unsafe {
        check(
            "listing GPUs",
            (i.enumerate_physical_devices)(h, &mut count, null_mut()),
        )?;
        let mut devices = vec![null_mut(); count as usize];
        check(
            "listing GPUs",
            (i.enumerate_physical_devices)(h, &mut count, devices.as_mut_ptr()),
        )?;
        devices.truncate(count as usize);
        devices
    };
    let mut best: Option<(u32, Choice)> = None;
    let mut rejected = Vec::new();
    for pd in devices {
        // SAFETY: `pd` came from this instance, and the output is a zeroed
        // struct of the type the command writes.
        let props = unsafe {
            let mut props = Box::<ffi::PhysicalDeviceProperties>::new_zeroed().assume_init();
            (i.get_properties)(pd, &mut *props);
            props
        };
        let name = device_name(&props);
        let rank = match props.device_type {
            ffi::DEVICE_TYPE_DISCRETE => 3,
            ffi::DEVICE_TYPE_INTEGRATED => 2,
            ffi::DEVICE_TYPE_VIRTUAL => 1,
            _ => {
                rejected.push(format!("{name}: not a GPU"));
                continue;
            }
        };
        if props.api_version < ffi::API_1_2 {
            rejected.push(format!("{name}: Vulkan older than 1.2"));
            continue;
        }
        let limits = &props.limits;
        if limits.max_compute_work_group_invocations < INVOCATIONS
            || limits.max_compute_work_group_size[0] < INVOCATIONS
        {
            rejected.push(format!(
                "{name}: workgroups under {INVOCATIONS} invocations"
            ));
            continue;
        }
        if !has_half_floats(i, pd) {
            rejected.push(format!("{name}: no half-float arithmetic or storage"));
            continue;
        }
        let Some(family) = compute_family(i, pd) else {
            rejected.push(format!("{name}: no compute queue"));
            continue;
        };
        if best.as_ref().is_none_or(|b| rank > b.0) {
            best = Some((
                rank,
                Choice {
                    handle: pd,
                    family,
                    props,
                },
            ));
        }
    }
    match best {
        Some((_, choice)) => Ok(choice),
        None if rejected.is_empty() => Err("no Vulkan device".into()),
        None => Err(rejected.join("; ")),
    }
}

/// Whether `pd` has half-float arithmetic and 16-bit storage buffers.
fn has_half_floats(i: &ffi::instance::Functions, pd: Handle) -> bool {
    let mut f12 = ffi::Vulkan12Features {
        s_type: ffi::TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        p_next: null_mut(),
        flags: [0; 47],
    };
    let mut f11 = ffi::Vulkan11Features {
        s_type: ffi::TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        p_next: (&raw mut f12).cast(),
        flags: [0; 12],
    };
    let mut f2 = ffi::PhysicalDeviceFeatures2 {
        s_type: ffi::TYPE_PHYSICAL_DEVICE_FEATURES_2,
        p_next: (&raw mut f11).cast(),
        features: [0; 55],
    };
    // SAFETY: `pd` came from the instance of `i`, and the chain links live
    // structs of the types their s_type names.
    unsafe { (i.get_features2)(pd, &mut f2) };
    f11.flags[ffi::STORAGE_BUFFER_16BIT_ACCESS] != 0 && f12.flags[ffi::SHADER_FLOAT16] != 0
}

/// A queue family of `pd` with compute: a compute-only one first, which runs
/// on the asynchronous compute rings, apart from the desktop's graphics work.
fn compute_family(i: &ffi::instance::Functions, pd: Handle) -> Option<u32> {
    // SAFETY: `pd` came from the instance of `i`; the two-call enumeration,
    // the second with room for the count the first gave.
    let families = unsafe {
        let mut n = 0;
        (i.get_queue_families)(pd, &mut n, null_mut());
        let mut families = vec![ffi::QueueFamilyProperties::default(); n as usize];
        (i.get_queue_families)(pd, &mut n, families.as_mut_ptr());
        families.truncate(n as usize);
        families
    };
    let compute = |q: &ffi::QueueFamilyProperties| q.queue_flags & ffi::QUEUE_COMPUTE != 0;
    families
        .iter()
        .position(|q| compute(q) && q.queue_flags & ffi::QUEUE_GRAPHICS == 0)
        .or_else(|| families.iter().position(compute))
        .map(|f| f as u32)
}

/// The logical device on `choice`, with half floats enabled, and its
/// commands.
pub(crate) fn create_device(
    instance: &Instance,
    choice: &Choice,
) -> Result<(Handle, ffi::device::Functions), String> {
    let i = &instance.i;
    let priority = 1.0f32;
    let queue_info = ffi::DeviceQueueCreateInfo {
        s_type: ffi::TYPE_DEVICE_QUEUE_CREATE_INFO,
        p_next: null(),
        flags: 0,
        queue_family_index: choice.family,
        queue_count: 1,
        priorities: &priority,
    };
    let mut f12 = ffi::Vulkan12Features {
        s_type: ffi::TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        p_next: null_mut(),
        flags: [0; 47],
    };
    f12.flags[ffi::SHADER_FLOAT16] = 1;
    let mut f11 = ffi::Vulkan11Features {
        s_type: ffi::TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        p_next: (&raw mut f12).cast(),
        flags: [0; 12],
    };
    f11.flags[ffi::STORAGE_BUFFER_16BIT_ACCESS] = 1;
    let device_info = ffi::DeviceCreateInfo {
        s_type: ffi::TYPE_DEVICE_CREATE_INFO,
        p_next: (&raw const f11).cast(),
        flags: 0,
        queue_info_count: 1,
        queue_infos: &queue_info,
        layer_count: 0,
        layers: null(),
        extension_count: 0,
        extensions: null(),
        features: null(),
    };
    let mut handle = null_mut();
    // SAFETY: `device_info` and its chain outlive the call, and `choose` checked
    // that the device has the features enabled.
    check("creating the device", unsafe {
        (i.create_device)(choice.handle, &device_info, null(), &mut handle)
    })?;
    // SAFETY: the device is valid, and vkGetDeviceProcAddr returns its
    // commands or null.
    let d = unsafe {
        ffi::device::Functions::load(|name| (i.get_device_proc_addr)(handle, name.as_ptr()))
    };
    let d = match d {
        Ok(d) => d,
        Err(e) => {
            // SAFETY: as above; vkDestroyDevice has this signature.
            unsafe {
                if let Some(f) = (i.get_device_proc_addr)(handle, c"vkDestroyDevice".as_ptr()) {
                    let destroy: unsafe extern "system" fn(Handle, *const c_void) =
                        std::mem::transmute(f);
                    destroy(handle, null());
                }
            }
            return Err(e);
        }
    };
    Ok((handle, d))
}
