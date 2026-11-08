//! A fixed set of worker threads that split one layer's work between them.
//!
//! [`Pool::run`] hands out task numbers from a shared counter, so a fast core
//! takes more tasks than a slow one, and returns only when every task
//! is done. Each thread owns a [`Scratch`] that its tasks reuse, so a run
//! allocates only while a buffer is still growing to its working size.
//!
//! Tasks never split a reduction, so the result does not depend on the number
//! of threads or on which thread ran which task.

use std::cell::UnsafeCell;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

use crate::gemm::{GemmScratch, Matrix};

/// Per-thread working memory.
#[derive(Default)]
pub struct Scratch {
    pub floats: Vec<f32>,
    pub gemm: GemmScratch,
}

impl Scratch {
    /// The first `len` floats of [`Scratch::floats`], grown if needed. The
    /// contents are whatever an earlier task left.
    pub fn floats(&mut self, len: usize) -> (&mut [f32], &mut GemmScratch) {
        if self.floats.len() < len {
            self.floats.resize(len, 0.0);
        }
        (&mut self.floats[..len], &mut self.gemm)
    }
}

struct Slot(UnsafeCell<Scratch>);

// SAFETY: a slot is only reached through `Pool::run`, which gives slot `i` to
// thread `i` alone while a run lasts, and runs are serialised by `&mut Pool`.
unsafe impl Sync for Slot {}

type Job = *const (dyn Fn(usize) + Sync);

struct JobPtr(Job);

// SAFETY: the pointer is dereferenced only while `Pool::run` keeps the closure
// alive (it waits for every worker before returning), and the closure is Sync.
unsafe impl Send for JobPtr {}

struct State {
    generation: u64,
    job: Option<JobPtr>,
    busy: usize,
    panicked: bool,
    stop: bool,
}

struct Control {
    state: Mutex<State>,
    start: Condvar,
    finish: Condvar,
}

pub struct Pool {
    control: Arc<Control>,
    workers: Vec<JoinHandle<()>>,
    slots: Vec<Slot>,
    /// The Winograd convolutions' transformed weights, kept from one
    /// convolution to the next so that their memory is not filled with zeros
    /// again every time.
    pub(crate) winograd: Vec<Matrix>,
}

impl Pool {
    /// A pool running tasks on `threads` threads: the caller's and
    /// `threads - 1` workers, named `name-1`, `name-2` and so on.
    pub fn new(threads: usize, name: &str) -> std::io::Result<Self> {
        let threads = threads.max(1);
        let control = Arc::new(Control {
            state: Mutex::new(State {
                generation: 0,
                job: None,
                busy: 0,
                panicked: false,
                stop: false,
            }),
            start: Condvar::new(),
            finish: Condvar::new(),
        });
        let slots = (0..threads)
            .map(|_| Slot(UnsafeCell::new(Scratch::default())))
            .collect();
        // Workers join the pool one by one, so that if one cannot start, the
        // pool's drop stops those already running.
        let mut pool = Self {
            control,
            workers: Vec::with_capacity(threads - 1),
            slots,
            winograd: Vec::new(),
        };
        for index in 1..threads {
            let control = Arc::clone(&pool.control);
            let worker = std::thread::Builder::new()
                .name(format!("{name}-{index}"))
                .spawn(move || work(&control, index))?;
            pool.workers.push(worker);
        }
        Ok(pool)
    }

    pub fn threads(&self) -> usize {
        self.slots.len()
    }

    /// [`Pool::split_by`] in whole row groups of the matrix kernel.
    pub fn split(
        &mut self,
        rows: usize,
        max: usize,
        task: impl Fn(Range<usize>, &mut Scratch) + Sync,
    ) {
        self.split_by(rows, crate::gemm::MR, max, task);
    }

    /// Run `task(items, scratch)` over consecutive ranges that cover
    /// `0..items`, in multiples of `grain` and at most `max` long. Each thread
    /// takes the next range as it finishes one, sized to share what is left
    /// (guided scheduling): long ranges while there is much to do, short ones
    /// at the end, so threads finish together instead of waiting on the last
    /// long task.
    pub fn split_by(
        &mut self,
        items: usize,
        grain: usize,
        max: usize,
        task: impl Fn(Range<usize>, &mut Scratch) + Sync,
    ) {
        let threads = self.threads();
        let max = (max / grain).max(1) * grain;
        let next = AtomicUsize::new(0);
        self.run(threads.min(items.div_ceil(grain)), |_, scratch| {
            let mut start = next.load(Ordering::Relaxed);
            loop {
                if start >= items {
                    return;
                }
                let left = items - start;
                let len = (left.div_ceil(2 * threads).div_ceil(grain) * grain)
                    .clamp(grain, max)
                    .min(left);
                match next.compare_exchange_weak(
                    start,
                    start + len,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        task(start..start + len, scratch);
                        start = next.load(Ordering::Relaxed);
                    }
                    Err(current) => start = current,
                }
            }
        });
    }

    /// Run `task(i, scratch)` for every `i` in `0..tasks` and wait for all of
    /// them. A panic in a task is raised here once every thread has stopped.
    pub fn run(&mut self, tasks: usize, task: impl Fn(usize, &mut Scratch) + Sync) {
        if tasks == 0 {
            return;
        }
        let next = AtomicUsize::new(0);
        let slots = &self.slots;
        let body = |thread: usize| {
            // SAFETY: thread indices are unique among the threads of this run
            // (see `Slot`), so this is the only reference to the slot.
            let scratch = unsafe { &mut *slots[thread].0.get() };
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= tasks {
                    break;
                }
                task(i, scratch);
            }
        };
        if self.workers.is_empty() || tasks == 1 {
            body(0);
            return;
        }
        let job: &(dyn Fn(usize) + Sync) = &body;
        // SAFETY: only the lifetime is erased. The workers call the job before
        // `busy` reaches zero, and this function waits for that below, also
        // when `body` panics on this thread, so the closure outlives every call.
        let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(job) };
        {
            let mut state = self
                .control
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            state.job = Some(JobPtr(job));
            state.busy = self.workers.len();
            state.panicked = false;
            state.generation += 1;
        }
        self.control.start.notify_all();
        let mine = catch_unwind(AssertUnwindSafe(|| body(0)));
        let mut state = self
            .control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        while state.busy > 0 {
            state = self
                .control
                .finish
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.job = None;
        let theirs = state.panicked;
        drop(state);
        if let Err(payload) = mine {
            resume_unwind(payload);
        }
        assert!(!theirs, "a worker task panicked");
    }
}

fn work(control: &Control, index: usize) {
    let mut seen = 0;
    loop {
        let job = {
            let mut state = control.state.lock().unwrap_or_else(PoisonError::into_inner);
            while state.generation == seen && !state.stop {
                state = control
                    .start
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if state.stop {
                return;
            }
            seen = state.generation;
            state.job.as_ref().map(|j| j.0)
        };
        let failed = job.is_some_and(|job| {
            // SAFETY: `Pool::run` keeps the job alive until `busy` is zero.
            catch_unwind(AssertUnwindSafe(|| unsafe { (*job)(index) })).is_err()
        });
        let mut state = control.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.panicked |= failed;
        state.busy -= 1;
        if state.busy == 0 {
            control.finish.notify_one();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.control
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .stop = true;
        self.control.start.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// A buffer that the tasks of one [`Pool::run`] write in disjoint parts.
pub struct Shared<'a, T = f32> {
    ptr: *mut T,
    len: usize,
    _buffer: std::marker::PhantomData<&'a mut [T]>,
}

impl<T> Clone for Shared<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Shared<'_, T> {}

// SAFETY: tasks only write through `part`, whose callers promise disjoint
// ranges, so sharing the pointer between threads creates no data race.
unsafe impl<T: Send> Sync for Shared<'_, T> {}
// SAFETY: as for `Sync`.
unsafe impl<T: Send> Send for Shared<'_, T> {}

impl<'a, T> Shared<'a, T> {
    pub fn new(buffer: &'a mut [T]) -> Self {
        Self {
            ptr: buffer.as_mut_ptr(),
            len: buffer.len(),
            _buffer: std::marker::PhantomData,
        }
    }

    /// Rows `rows` of the buffer read as rows `stride` apart, each `width`
    /// floats used: from the start of the first to the end of the last.
    ///
    /// # Safety
    /// As [`Shared::part`].
    #[expect(clippy::mut_from_ref, reason = "callers promise disjoint parts")]
    pub unsafe fn rows(&self, rows: Range<usize>, stride: usize, width: usize) -> &mut [T] {
        let len = if rows.is_empty() {
            0
        } else {
            (rows.len() - 1) * stride + width
        };
        // SAFETY: forwarded to the caller.
        unsafe { self.part(rows.start * stride, len) }
    }

    /// Elements `start..start + len` of the buffer.
    ///
    /// # Safety
    /// No two live slices from this buffer may overlap.
    #[expect(clippy::mut_from_ref, reason = "callers promise disjoint parts")]
    pub unsafe fn part(&self, start: usize, len: usize) -> &mut [T] {
        assert!(
            start <= self.len && len <= self.len - start,
            "part out of bounds"
        );
        // SAFETY: in bounds by the assert, and disjoint by the caller.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_runs_once() {
        let mut pool = Pool::new(4, "test").expect("worker threads");
        for tasks in [0, 1, 3, 100] {
            let mut hits = vec![0.0f32; tasks];
            let out = Shared::new(&mut hits);
            pool.run(tasks, |i, _| {
                // SAFETY: task `i` writes element `i` alone.
                unsafe { out.part(i, 1)[0] += 1.0 };
            });
            assert!(hits.iter().all(|&h| h == 1.0));
        }
    }

    #[test]
    fn worker_panic_reaches_the_caller() {
        let mut pool = Pool::new(3, "test").expect("worker threads");
        let result = catch_unwind(AssertUnwindSafe(|| {
            pool.run(64, |i, _| assert!(i != 40, "task 40"));
        }));
        assert!(result.is_err());
        // The pool still works afterwards.
        pool.run(8, |_, _| {});
    }
}
