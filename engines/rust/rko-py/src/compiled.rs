//! Foreign-function evaluator for objectives compiled outside the interpreter.
//!
//! Objective:
//! - Let the engine score random keys by calling a native function pointer, such as a
//!   Numba `cfunc`, from every worker thread without touching the interpreter lock.
//!
//! Technical approach:
//! - A `CompiledProblem` stores the function address and a reference that keeps the
//!   compiled object alive; `evaluate` reinterprets the address as
//!   `extern "C" fn(*const f64, isize) -> f64` and passes the key slice and its length.
//! - This is the only module of the workspace that uses `unsafe`; the evaluator is a
//!   thin call with no state of its own, so thread safety is the function's contract.
//!
//! Flow (pseudocode):
//! 1. Python hands over the address, the key dimension and the owning object.
//! 2. Each evaluation calls the pointer with the key slice; non-finite results stop the run.
//!
//! Important criteria:
//! - The function must be reentrant, must not retain the pointer, must not read beyond
//!   `n` doubles and must not raise or unwind; a violation is undefined behavior.
//! - The owner reference outlives every call because the solve holds the problem.
//!
//! Inputs/Outputs:
//! - Input: function address, dimension and owner.
//! - Output: a `Problem` evaluated entirely in native code.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rko_core::problem::{Incremental, Problem};

/// Native objective called through a function pointer.
#[pyclass(frozen, module = "rko_solver._native")]
pub struct CompiledProblem {
    address: usize,
    dimension: usize,
    swap: Option<usize>,
    #[allow(dead_code)]
    owner: Py<PyAny>,
}

#[pymethods]
impl CompiledProblem {
    #[new]
    #[pyo3(signature = (address, dimension, owner, swap_address=None))]
    fn new(
        address: usize,
        dimension: usize,
        owner: Py<PyAny>,
        swap_address: Option<usize>,
    ) -> PyResult<Self> {
        if address == 0 || dimension == 0 || swap_address == Some(0) {
            return Err(PyValueError::new_err(
                "address and dimension must be nonzero",
            ));
        }
        Ok(Self {
            address,
            dimension,
            swap: swap_address,
            owner,
        })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Borrowed view that implements `Problem` over a `CompiledProblem`.
pub struct CompiledAdapter<'a>(pub &'a CompiledProblem);

impl Problem for CompiledAdapter<'_> {
    fn dimension(&self) -> usize {
        self.0.dimension
    }

    #[allow(unsafe_code)]
    fn evaluate(&self, keys: &[f64]) -> f64 {
        debug_assert_eq!(keys.len(), self.0.dimension);
        let function: extern "C" fn(*const f64, isize) -> f64 = unsafe {
            std::mem::transmute::<usize, extern "C" fn(*const f64, isize) -> f64>(self.0.address)
        };
        function(keys.as_ptr(), keys.len() as isize)
    }

    fn incremental<'a>(&'a self, keys: &[f64]) -> Option<Box<dyn Incremental + 'a>> {
        self.0.swap?;
        let mut state = NativeSwapState {
            problem: self.0,
            score: 0.0,
        };
        state.reset(keys);
        Some(Box::new(state))
    }
}

/// Two-change swap of key values `(i, keys[j])` and `(j, keys[i])`, if the changes have that shape.
fn swap_pair(keys: &[f64], changes: &[(usize, f64)]) -> Option<(usize, usize)> {
    match changes {
        [(i, vi), (j, vj)] if i != j && *vi == keys[*j] && *vj == keys[*i] => Some((*i, *j)),
        _ => None,
    }
}

/// Incremental state over a native swap delta that receives the current score.
struct NativeSwapState<'a> {
    problem: &'a CompiledProblem,
    score: f64,
}

impl NativeSwapState<'_> {
    #[allow(unsafe_code)]
    fn native_swap(&self, keys: &[f64], score: f64, i: usize, j: usize) -> f64 {
        let address = self.problem.swap.unwrap_or(0);
        let function: extern "C" fn(*const f64, isize, f64, isize, isize) -> f64 = unsafe {
            std::mem::transmute::<usize, extern "C" fn(*const f64, isize, f64, isize, isize) -> f64>(
                address,
            )
        };
        function(
            keys.as_ptr(),
            keys.len() as isize,
            score,
            i as isize,
            j as isize,
        )
    }

    fn full(&self, keys: &[f64], changes: &[(usize, f64)]) -> f64 {
        let mut moved = keys.to_vec();
        for &(index, value) in changes {
            moved[index] = value;
        }
        CompiledAdapter(self.problem).evaluate(&moved)
    }
}

impl Incremental for NativeSwapState<'_> {
    fn reset(&mut self, keys: &[f64]) {
        self.score = CompiledAdapter(self.problem).evaluate(keys);
    }

    fn probe(&self, keys: &[f64], changes: &[(usize, f64)]) -> f64 {
        match swap_pair(keys, changes) {
            Some((i, j)) if keys[i] == keys[j] => self.score,
            Some((i, j)) => self.native_swap(keys, self.score, i, j),
            None => self.full(keys, changes),
        }
    }

    fn commit(&mut self, keys: &[f64], changes: &[(usize, f64)]) {
        self.score = self.probe(keys, changes);
    }

    fn is_delta(&self, keys: &[f64], changes: &[(usize, f64)]) -> bool {
        swap_pair(keys, changes).is_some()
    }
}
