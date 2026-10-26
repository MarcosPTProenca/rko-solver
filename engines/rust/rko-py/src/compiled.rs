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
use rko_core::problem::Problem;

/// Native objective called through a function pointer.
#[pyclass(frozen, module = "rko_solver._native")]
pub struct CompiledProblem {
    address: usize,
    dimension: usize,
    #[allow(dead_code)]
    owner: Py<PyAny>,
}

#[pymethods]
impl CompiledProblem {
    #[new]
    fn new(address: usize, dimension: usize, owner: Py<PyAny>) -> PyResult<Self> {
        if address == 0 || dimension == 0 {
            return Err(PyValueError::new_err(
                "address and dimension must be nonzero",
            ));
        }
        Ok(Self {
            address,
            dimension,
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
}
