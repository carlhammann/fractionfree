//! Fraction-free linear algebra for [ndarray], with minimal trait bounds.
//!
//! This library provides linear algebra primitives that will work without inexact divisions.
//! Sometimes, this is called "integer-preserving" linear algebra, because all intermediate
//! results are kept as integers. The trick is to represent a matrix `A` with rational entries as a
//! product `D^(-1)B`, where `D` is diagonal, and both `D` and `B` have only integral entries.
//! Currently,
//!
//! - [LU decomposititon](lu) of full-rank, square-or-wider matrices,
//! - [unique solutions](LU::solve_square) to linear systems,
//! - matrix [inverses](LU::inverse), and
//! - [determinant](LU::determinant)s
//!
//! are implemented.
//!
//! Trait bounds placed on entry types of matrices are as loose as reasonably possible, hence this
//! library should in principle be applicable for every integral domain. However, the various
//! primitive integer types are the motivating example and main intended use case. In particular,
//! all functions require matrix entries to be [Copy].
//!
//! Everything here is based on an algorithm that computes a fraction-free LU decomposition taken
//! from Dureisseix' paper "[Generalized fraction-free LU factorization for singular systems with
//! kernel extraction][thepaper]". However, the "singular" and "kernel extraction" parts of the
//! paper are not yet implemented: For now, this library can solve only systems that have exactly
//! one solution. That is enough to compute the inverse of a square matrix, if it exists, which was
//! the original motivation.
//!
//! Written in pure Rust, with few dependencies and even fewer optimisations, this library is for
//! you if your matrices aren't extremely big, and you want to work without specialised hardware or
//! platform restrictions.
//!
//! [thepaper]: https://hal.science/hal-00678543/document

use ndarray::{s, Array1, Array2, ArrayViewMut1, ArrayViewMut2, Zip};
use num::{integer::gcd, Integer, One, Signed, Zero};
use std::error::Error;
use std::fmt;
use std::ops::{AddAssign, Div, DivAssign, Mul, Neg, Sub};

/// Errors related to linear algebra.
#[derive(Debug, PartialEq)]
pub enum LinalgErr {
    /// A matrix was expected to be square, but wasn't. Contains the name of the function that
    /// complained, and the numbers of rows and columns of the offending matrix.
    NotSquare(&'static str, usize, usize),

    /// A function of two arguments whose dimensions have to match received arguments of
    /// incompatible shapes. Contains the name of the function that complained, and the relevant
    /// dimensions of the first and second argument.
    DimensionMismatch(&'static str, usize, usize),

    /// A matrix of less-than-full rank was passed to [lu]. If you want to use LU decomposition to
    /// compute the determinant, this means that the determinant is zero. For the (not necessarily
    /// square) coefficient matrix of a system of linear equations, this means the system is
    /// indeterminate.
    LURankDeficient,

    /// A matrix that is taller than wide was passed to [lu]. Contains the numbers of rows and
    /// columns.
    LUMatrixTall(usize, usize),
}

impl fmt::Display for LinalgErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinalgErr::NotSquare(fname, n, m) =>
                write!(f, "argument for {} is not square: {} rows and {} columns", fname, n, m),
            LinalgErr::DimensionMismatch(fname, n, m) =>
                write!(f, "argument dimensions for {} don't match: first argument is {}-dimensional, second is {}-dimensional", fname, n, m),
            LinalgErr::LURankDeficient => write!(f, "rank-deficient argument in LU decomposition"),
            LinalgErr::LUMatrixTall(n, m) => write!(
                f,
                "input for LU decomposition must be at least as wide as tall, but has {} rows, {} columns",
                n, m
            ),
        }
    }
}

impl Error for LinalgErr {}

/// Pre-computed fraction-free LU decomposition. Obtained as the output of the function [lu].
///
/// The fraction-free LU decomposition of `A` is given by four integer matrices
///
/// * a permutation matrix `P`,
/// * a lower diagonal matrix `L`,
/// * a diagonal matrix `D`, and
/// * an upper diagonal matrix `U`,
///
/// such that `P A = L D^(-1) U`. Crucially, such a representation can be obtained without any
/// inexact divisions: The type variable `T` can in principle be any integral domain.
///
/// Instead of directly computing the four matrices and keeping them in memory, this type is a more
/// compact representation, which can be used to cheaply compute [determinant](LU::determinant)s,
/// and [inverse](LU::inverse)s, or to [extract](LU::extract) the concrete factors of the
/// decomposition, or [solve](LU::solve_square) linear systems of equations.
#[derive(Debug, PartialEq)]
pub struct LU<T> {
    /// A representation of the permutation matrix `P` in terms of transpositions. That is, the
    /// actual permutation `P` applied to the rows of `A` is the following product of
    /// transpositions:
    ///
    /// (0, p[0]) (1, p[1]) ... (n-1, p[n-1])
    permutation: Array1<usize>,

    /// The matrix that has `U` in the upper half and `L` in the lower half.
    ///
    /// The diagonal elements stored here are those of `U`, but the diagonals of `L` and of `D` are
    /// cheaply obtained, as they are closely related. (See the definition of [LU::extract].)
    ///
    /// Furthermore, if `A` is square, the diagonal elements are the leading principal minors of
    /// `A` (thus the element in the lower right corner is the determinant), up to a sign flip,
    /// determined by the permutation. TODO: (How) is this correct? Research more precisely.
    data: Array2<T>,
}

impl<T> LU<T> {
    /// Construct the four matrices of the fraction-free [LU] decomposition.
    ///
    /// Called on the result of `lu(A)`, this function returns `(P, L, d, U)` such that `P A = L
    /// diag(d)^(-1) U`, where `P` is a permutation matrix, `L` is lower triangular, and `U` is
    /// upper triangular. See also the documentation of [LU].
    ///
    /// It might be useful to call [normalise] on `(d, U)` before proceeding.
    pub fn extract(&self) -> (Array2<T>, Array2<T>, Array1<T>, Array2<T>)
    where
        T: Zero + One + Copy,
    {
        let a = &self.data;
        let n = a.shape()[0]; //rows
        let m = a.shape()[1]; //columns
        let table = tabulate_permutation(&self.permutation);
        let p = Array2::from_shape_fn(
            (n, n),
            |(i, j)| {
                if table[i] == j {
                    T::one()
                } else {
                    T::zero()
                }
            },
        );
        let mut l = Array2::zeros((n, n));
        let mut d = Array1::zeros(n);
        let mut u = Array2::zeros(a.raw_dim());

        for i in 0..n {
            for j in 0..(i + 1) {
                l[[i, j]] = a[[i, j]];
            }
        }
        l[[n - 1, n - 1]] = T::one();

        for i in 0..n {
            for j in i..m {
                u[[i, j]] = a[[i, j]];
            }
        }

        d[0] = a[[0, 0]];
        for i in 1..n {
            d[i] = a[[i - 1, i - 1]] * a[[i, i]];
        }
        d[n - 1] = a[[n - 2, n - 2]];

        (p, l, d, u)
    }

    /// Calculate the determinant of a square matrix.
    ///
    /// Called on the result of `lu(A)`, this returns the determinant of `A`.
    pub fn determinant(&self) -> Result<T, LinalgErr>
    where
        T: Neg<Output = T> + Copy,
    {
        let a = &self.data;
        let d = a.raw_dim();
        let n = d[0];
        if n != d[1] {
            return Err(LinalgErr::NotSquare("determinant", n, d[1]));
        }

        let x = a[[n - 1, n - 1]];

        Ok(if permutation_parity(&self.permutation) {
            x
        } else {
            -x
        })
    }

    /// Solve a system of equations with square coefficient matrix.
    ///
    /// Called on the result of `lu(A)`, this can be used to solve systems of the form `Ax=b`, where
    /// `A` is a `n x n`-matrix. Note that, since [lu] will only succeed when `A` has full rank,
    /// this function can only be used to find the unique solution to such systems.
    ///
    /// This function returns `(d, x)`, where `1/d * x` is the solution. In particular, `d` or `-d`
    /// will be the determinant of `A`.
    ///
    /// This function takes ownership of its argument, because the `X` will be overwritten on
    /// `rhs`.
    ///
    /// It might be useful to [normalise] the result before proceeding.
    pub fn solve_square(&self, mut rhs: Array1<T>) -> Result<(T, Array1<T>), LinalgErr>
    where
        T: One + Zero + Copy + Sub<Output = T> + Div<Output = T> + AddAssign,
    {
        let mut d = T::one();
        self.solve_square_inplace(&mut rhs.view_mut(), &mut d)?;
        Ok((d, rhs))
    }

    /// Like [LU::solve_square], but mutating its arguments in place.
    ///
    /// See the documentation at [LU::solve_square].
    pub fn solve_square_inplace(
        &self,
        rhs: &mut ArrayViewMut1<T>,
        d: &mut T,
    ) -> Result<(), LinalgErr>
    where
        T: One + Zero + Copy + Sub<Output = T> + Div<Output = T> + AddAssign,
    {
        let a = &self.data;
        if a.raw_dim()[0] != a.raw_dim()[1] {
            return Err(LinalgErr::NotSquare(
                "solve_square_inplace",
                a.raw_dim()[0],
                a.raw_dim()[1],
            ));
        }
        if a.raw_dim()[1] != rhs.raw_dim()[0] {
            return Err(LinalgErr::DimensionMismatch(
                "solve_square_inplace",
                a.raw_dim()[1],
                rhs.raw_dim()[0],
            ));
        }
        permute_vector(&self.permutation, rhs);
        self.forward(rhs);
        self.backward(rhs, d);
        Ok(())
    }

    /// Calculate the inverse of a square matrix.
    ///
    /// If the input was obtained as `lu(A)`, this returns `(d, B)` such that `1/d * B` is
    /// the inverse of `A`. In particular, `d` or `-d` will be the determinant of `A`.
    ///
    /// It might be useful to [normalise] the result before proceeding.
    pub fn inverse(&self) -> Result<(T, Array2<T>), LinalgErr>
    where
        T: One + Zero + Copy + Sub<Output = T> + Div<Output = T> + AddAssign,
    {
        let a = &self.data;
        let d = a.raw_dim();
        let n = d[0];
        if n != d[1] {
            return Err(LinalgErr::NotSquare("inverse", n, d[1]));
        }
        let mut det = T::one();
        let mut inv = Array2::eye(n);
        for i in 0..n {
            let mut ei = inv.slice_mut(s![.., i]);
            self.solve_square_inplace(&mut ei, &mut det)?;
        }
        Ok((det, inv))
    }
}

/// Compute the fraction-free [LU] decomposition of an integer matrix.
///
/// The matrix must be at least as wide as tall, and of full rank. Otherwise, this function fails,
/// returning [LinalgErr::LUMatrixTall] or [LinalgErr::LURankDeficient], respectively.
///
/// [Div]isions performed by this function are exact (i.e. leave no remainder).
///
/// This function takes ownership of its argument, which it overwrites, and which becomes part of
/// the returned value.
pub fn lu<T>(mut a: Array2<T>) -> Result<LU<T>, LinalgErr>
where
    T: Zero + One + Sub<Output = T> + Div<Output = T> + Copy,
{
    let n = a.shape()[0]; //rows
    let m = a.shape()[1]; //columns
    if m < n {
        return Err(LinalgErr::LUMatrixTall(n, m));
    }

    let mut p = Array1::zeros(n);
    let mut oldpivot = T::one();
    for k in 0..n {
        p[k] = k;
        if a[[k, k]].is_zero() {
            // search for the first non-null pivot in the k-th column, below the diagonal.
            let mut kpivot = k + 1;
            while kpivot < n && a[[kpivot, k]].is_zero() {
                kpivot += 1;
            }
            if kpivot == n {
                return Err(LinalgErr::LURankDeficient);
            } else {
                // row interchange
                let (mut v, mut w) = a.multi_slice_mut((s![k, ..], s![kpivot, ..]));
                Zip::from(&mut v).and(&mut w).for_each(std::mem::swap);
                p[k] = kpivot;
            }
        }
        let pivot = a[[k, k]];
        for i in (k + 1)..n {
            let aik = a[[i, k]];
            for j in (k + 1)..m {
                a[[i, j]] = (pivot * a[[i, j]] - aik * a[[k, j]]) / oldpivot;
            }
        }
        oldpivot = pivot;
    }
    Ok(LU {
        data: a,
        permutation: p,
    })
}

/// Reduce a fraction-free representation of a rational matrix.
///
/// * `numerators` must be an `n x m`-matrix of `Integer`s
/// * `denominators` must be an `n`-vector of non-zero `Integer`s
///
/// We interpret these two data together as the rational matrix `diag(denominators)^(-1)
/// numerators`. That is: the `i´-th row of `numerators` is divided by the `i`-entry of of
/// `denominators`.
///
/// This function divides each row of `numerators` and its corresponding entry in `denominators` by
/// their common [gcd]. This makes it so that each row is "in lowest terms". It additionally
/// ensures that the `denominators` are positive.
pub fn normalise<T>(
    denominators: &mut ArrayViewMut1<T>,
    numerators: &mut ArrayViewMut2<T>,
) -> Result<(), LinalgErr>
where
    T: Integer + DivAssign + Signed + Copy,
{
    let d = numerators.raw_dim();
    let n = d[0];
    let m = d[1];
    if n != denominators.len() {
        return Err(LinalgErr::DimensionMismatch(
            "normalise",
            denominators.len(),
            n,
        ));
    }

    for i in 0..n {
        let mut g = denominators[i];
        for j in 0..m {
            g = gcd(numerators[[i, j]], g);
        }
        if denominators[i].is_negative() {
            g = -g;
        }
        denominators[i] /= g;
        for j in 0..m {
            numerators[[i, j]] /= g;
        }
    }
    Ok(())
}

// private helper functions

/// Obtain from [LU::permutation] a table of values.
fn tabulate_permutation(p: &Array1<usize>) -> Array1<usize> {
    let mut res = Array1::from_shape_fn(p.raw_dim(), |i| i);
    for i in 0..p.len() {
        let tmp = res[i];
        res[i] = res[p[i]];
        res[p[i]] = tmp;
    }
    res
}

/// Compute the parity of [LU::permutation]. Returns `true` for even permutations.
fn permutation_parity(p: &Array1<usize>) -> bool {
    let mut res = true;
    for i in 0..p.len() {
        if p[i] != i {
            res = !res;
        }
    }
    res
}

/// Apply [LU::permutation] to the entries of a vector. It is assumed that the dimensions match.
fn permute_vector<T>(p: &Array1<usize>, a: &mut ArrayViewMut1<T>)
where
    T: Copy,
{
    for i in 0..p.len() {
        if p[i] == i {
            continue;
        }
        let tmp = a[i];
        a[i] = a[p[i]];
        a[p[i]] = tmp;
    }
}

impl<T> LU<T> {
    /// Forward substitution, solving `L D^(-1) y = b`. It assumed that dimensions match.
    fn forward(&self, b: &mut ArrayViewMut1<T>)
    where
        T: One + Copy + Sub<Output = T> + Div<Output = T>,
    {
        let a = &self.data;
        let n = a.raw_dim()[0];

        let mut oldpivot = T::one();

        for k in 0..(n - 1) {
            let pivot = a[[k, k]];
            for i in (k + 1)..n {
                b[i] = (pivot * b[i] - a[[i, k]] * b[k]) / oldpivot;
            }
            oldpivot = pivot;
        }
    }

    /// Backward substitution, solving `U x = det(A) y`. It assumed that dimensions match.
    fn backward(&self, y: &mut ArrayViewMut1<T>, d: &mut T)
    where
        T: Zero + Mul<Output = T> + Sub<Output = T> + Div<Output = T> + AddAssign + Copy,
    {
        let a = &self.data;
        let n = a.raw_dim()[0];
        *d = a[[n - 1, n - 1]];
        for i in (0..n).rev() {
            let mut x = T::zero();
            for k in (i + 1)..n {
                x += a[[i, k]] * y[[k]];
            }
            y[[i]] = (*d * y[[i]] - x) / a[[i, i]];
        }
    }
}

// tests

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array2};

    #[test]
    fn lu_errors() {
        let tall = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
        let e = lu(tall);
        assert_eq!(e, Err(LinalgErr::LUMatrixTall(4, 3)));

        let deficient = arr2(&[[1, 2, 3], [4, 5, 6], [2, 4, 6]]);
        let e = lu(deficient);
        assert_eq!(e, Err(LinalgErr::LURankDeficient));
    }

    fn extract_recombine(a: Array2<i32>) {
        let x = lu(a.clone()).unwrap();
        let (p, l, mut d, u) = x.extract();
        let m = d.fold(1, |acc, new| acc * new);
        d = m / d; // this is an exact division
        let dd = Array2::from_diag(&d);
        assert_eq!(m * p.dot(&a), l.dot(&dd).dot(&u));
    }

    #[test]
    fn extract_roundtrip() {
        extract_recombine(arr2(&[[1, -1], [0, 5]]));
        extract_recombine(arr2(&[[-1, -1, -1], [3, 3, 1], [3, 0, 3]]));
        extract_recombine(arr2(&[[1, 1, 0, 0], [5, 6, 7, 8], [9, 10, 11, 11]]));
        extract_recombine(arr2(&[
            [1, 1, 0, 0],
            [5, 6, 7, 8],
            [9, 10, 11, 11],
            [-3, 5, 7, 3],
        ]));
    }

    #[test]
    fn determinant() {
        assert_eq!(
            -1,
            lu(arr2(&[[0, 1], [1, 0]])).unwrap().determinant().unwrap()
        );
        assert_eq!(
            6,
            lu(arr2(&[[-1, -1, -1], [3, 3, 1], [3, 0, 3]]))
                .unwrap()
                .determinant()
                .unwrap()
        );
        assert_eq!(
            1638,
            lu(arr2(&[
                [-1, 5, -1, 7],
                [9, 3, 3, 1],
                [3, 0, -33, 1],
                [0, 0, 0, 1]
            ]))
            .unwrap()
            .determinant()
            .unwrap()
        );
    }

    #[test]
    fn normalise() {
        let mut k = arr1(&[6, 4, -15]);
        let mut a = arr2(&[[81, -9, 24], [56, 4, 8], [-10, 30, 45]]);
        let _ = super::normalise(&mut k.view_mut(), &mut a.view_mut());
        assert_eq!(k, arr1(&[2, 1, 3]));
        assert_eq!(a, arr2(&[[27, -3, 8], [14, 1, 2], [2, -6, -9]]));
    }

    fn solve_unsolve(a: Array2<i32>, b: Array1<i32>) {
        let h = lu(a.clone()).unwrap();
        let (d, x) = h.solve_square(b.clone()).unwrap();
        println!("d = {}\nx = {}", d, x);
        assert_eq!(a.dot(&x), d * b);
    }

    #[test]
    fn solve_determinate_roundtrip() {
        solve_unsolve(arr2(&[[1, -1], [0, 5]]), arr1(&[4, 5]));
        solve_unsolve(
            arr2(&[[-1, -1, -1], [3, 3, 1], [3, 0, 3]]),
            arr1(&[1, 5, 9]),
        );
        solve_unsolve(
            arr2(&[[1, 1, 0, 0], [5, 6, 7, 8], [9, 10, 11, 11], [-3, 5, 7, 3]]),
            arr1(&[1, 7, 9, -1]),
        );
    }

    #[test]
    fn inverse_dot_eye() {
        let a = arr2(&[
            [8, -9, 10, 11],
            [0, 0, 2, 12],
            [-10, 30, 5, 45],
            [1, 0, 0, 0],
        ]);
        let (d, b) = lu(a.clone()).unwrap().inverse().unwrap();
        assert_eq!(b.dot(&a), Array2::from_diag_elem(4, d));

        let a = arr2(&[[81, -9, 24], [56, 4, 8], [-10, 30, 45]]);
        let (d, b) = lu(a.clone()).unwrap().inverse().unwrap();
        assert_eq!(a.dot(&b), Array2::from_diag_elem(3, d));
    }
}
