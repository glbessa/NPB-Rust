
use common::print_results::*;
use common::randdp::*;

use std::cell::RefCell;
use std::{ops::Add, time::Instant};

static FFTBLOCK_DEFAULT: usize = 1;
static FFTBLOCKPAD_DEFAULT: usize = 1;

static FFTBLOCK: usize = FFTBLOCK_DEFAULT;
static FFTBLOCKPAD: usize = FFTBLOCKPAD_DEFAULT;

static SEED: f64 = 314159265.0;
static A: f64 = 1220703125.0;
static PI: f64 = 3.141592653589793238;
static ALPHA: f64 = 1.0e-6;

const NX: usize = 64;
const NY: usize = 64;
const NZ: usize = 64;
const NITER_DEFAULT: usize = 6;
const MAXDIM: usize = 64;

const NXP: usize = NX + 1;
const NYP: usize = NY;
const NTOTAL: usize = (NX * NY * NZ) as usize;
const NTOTALP: usize = ((NX + 1) * NY * NZ);

const NPBVERSION: &str = "4.1.2";
const LIBVERSION: &str = "1";
const BENCHMARK: &str = "FT";
const CLASS_NPB: &str = "s";
const COMPILETIME: &str = "2023-05-08T22:51:30.193519220-03:00";
const COMPILERVERSION: &str = "rustc 1.70.0-nightly";
const CS1: &str = "-";
const CS2: &str = "-";
const CS3: &str = "-";
const CS4: &str = "-";
const CS5: &str = "-";
const CS6: &str = "-";
const CS7: &str = "-";

#[derive(Clone, Debug, Copy)]
struct Dcomplex {
    real: f64,
    imag: f64,
}

static D1: usize = NX;
static D2: usize = NY;
static D3: usize = NZ;

impl Add for Dcomplex {
    type Output = Dcomplex;

    fn add(self, other: Dcomplex) -> Dcomplex {
        Dcomplex {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

fn dcomplex_create(r: f64, i: f64) -> Dcomplex {
    Dcomplex { real: r, imag: i }
}

fn dcomplex_add(a: Dcomplex, b: Dcomplex) -> Dcomplex {
    Dcomplex {
        real: a.real + b.real,
        imag: a.imag + b.imag,
    }
}
fn dcomplex_abs(x: Dcomplex) -> f64 {
    ((x.real * x.real) + (x.imag * x.imag)).sqrt()
}
fn dcomplex_sub(a: Dcomplex, b: Dcomplex) -> Dcomplex {
    Dcomplex {
        real: a.real - b.real,
        imag: a.imag - b.imag,
    }
}

//#define dconjg(x)(dcomplex){(x).real, -1.0*(x).imag}
fn dconjg(x: Dcomplex) -> Dcomplex {
    Dcomplex {
        real: x.real,
        imag: -1.0 * x.imag,
    }
}

// #define dcomplex_mul(a,b) (dcomplex){((a).real*(b).real)-((a).imag*(b).imag),\
// 	((a).real*(b).imag)+((a).imag*(b).real)}
fn dcomplex_mul(a: Dcomplex, b: Dcomplex) -> Dcomplex {
    Dcomplex {
        real: (a.real * b.real) - (a.imag * b.imag),
        imag: (a.real * b.imag) + (a.imag * b.real),
    }
}

fn dcomplex_div(z1: Dcomplex, z2: Dcomplex) -> Dcomplex {
    let a = z1.real;
    let b = z1.imag;
    let c = z2.real;
    let d = z2.imag;
    let divisor = c * c + d * d;
    let real = (a * c + b * d) / divisor;
    let imag = (b * c - a * d) / divisor;
    Dcomplex { real, imag }
}

fn dcomplex_mul2(a: Dcomplex, b: f64) -> Dcomplex {
    Dcomplex {
        real: a.real * b,
        imag: a.imag * b,
    }
}

// dcomplex_div2
fn dcomplex_div2(a: Dcomplex, b: f64) -> Dcomplex {
    Dcomplex {
        real: a.real / b,
        imag: a.imag / b,
    }
}

//recebe um &Vec<Dcomplex> e retorna um Vec<f64> com os valores de real
fn dcomplex_real(v: &[Dcomplex]) -> Vec<f64> {
    let mut real: Vec<f64> = vec![0.0; v.len()];
    for i in 0..v.len() {
        real[i] = v[i].real;
    }
    real
}

//recebe um &Vec<Dcomplex> e retorna um Vec<f64> com os valores de imag
fn dcomplex_imag(v: &[Dcomplex]) -> Vec<f64> {
    let mut imag: Vec<f64> = vec![0.0; v.len()];
    for i in 0..v.len() {
        imag[i] = v[i].imag;
    }
    imag
}

//recebe um &Vec<Dcomplex> e retorna um Vec<f64> com tamanho 2 * v.len() onde os valores de real e imag são intercalados
fn dcomplex_interleave(v: &[Dcomplex]) -> Vec<f64> {
    let mut interleaved: Vec<f64> = vec![0.0; 2 * v.len()];
    for i in 0..v.len() {
        interleaved[2 * i] = v[i].real;
        interleaved[2 * i + 1] = v[i].imag;
    }
    interleaved
}

fn main() {
    let mut sums: Vec<Dcomplex> = vec![dcomplex_create(0.0, 0.0); NITER_DEFAULT + 1];
    let t: f64;
    let mut twiddle: Vec<f64> = vec![0.0; NTOTAL];
    let mut u: Vec<Dcomplex> = vec![dcomplex_create(0.0, 0.0); MAXDIM];
    let mut u0: RefCell<Vec<Dcomplex>> = RefCell::new(vec![dcomplex_create(0.0, 0.0); NTOTAL]);
    let mut u1: RefCell<Vec<Dcomplex>> = RefCell::new(vec![dcomplex_create(0.0, 0.0); NTOTAL]);

    let mflops: f64;
    let mut verified: bool = false;
    /*
     * ---------------------------------------------------------------------
     * run the entire problem once to make sure all data is touched.
     * this reduces variable startup costs, which is important for such a
     * short benchmark. the other NPB 2 implementations are similar.
     * ---------------------------------------------------------------------
     */
    let niter = NITER_DEFAULT;

    println!("\n NAS Parallel Benchmarks 1.0 Serial Rust version - FT Benchmark");
    println!(" Size                : {} x {} x {}", NX, NY, NZ);
    println!(" Iterations          : {}\n", niter);

    init_ui(&mut *u0.borrow_mut(), &mut *u1.borrow_mut(), &mut twiddle);
    compute_indexmap(&mut twiddle);
    compute_initial_conditions(u1.get_mut());
    fft_init(MAXDIM, &mut u);
    fft(1, &mut u1, &mut u0, &u);

    compute_indexmap(&mut twiddle);
    compute_initial_conditions(u1.get_mut());
    fft_init(MAXDIM, &mut u);
    fft(1, &mut u1, &mut u0, &u);

    let bench_timer = Instant::now();

    for iter in 1..=niter {
        evolve(u0.get_mut(), u1.get_mut(), &mut twiddle);
        fft(-1, &mut u1.clone(), &mut u1, &u);
        checksum(iter, u1.get_mut(), &mut sums);
    }

    verify(NX, NY, NZ, niter, &mut verified, CLASS_NPB, &mut sums);

    t = bench_timer.elapsed().as_secs_f64();

    if t != 0.0 {
        mflops = 1.0e-6
            * (NTOTAL as f64)
            * (14.8157
                + 7.19641 * (NTOTAL as f64).log10()
                + (5.23518 + 7.21113 * (NTOTAL as f64).log10()) * (niter as f64))
            / t;
    } else {
        mflops = 0.0;
    }

    rust_print_results(
        BENCHMARK,
        CLASS_NPB,
        NX.try_into().unwrap(),
        NY.try_into().unwrap(),
        NZ.try_into().unwrap(),
        niter.try_into().unwrap(),
        t,
        mflops,
        "           floating point",
        verified,
        NPBVERSION,
        COMPILETIME,
        COMPILERVERSION,
        LIBVERSION,
        "1",
        CS1,
        CS2,
        CS3,
        CS4,
        CS5,
        CS6,
        CS7
    );
}

fn checksum(i: usize, u1: &mut Vec<Dcomplex>, sums: &mut Vec<Dcomplex>) {
    let mut chk = dcomplex_create(0.0, 0.0);

    let (mut q, mut r, mut s): (usize, usize, usize);

    for j in 1..1024 {
        q = j % NX;
        r = (3 * j) % NY;
        s = (5 * j) % NZ;
        chk = dcomplex_add(chk, u1[indxp3(s, r, q)]);
    }
    chk = dcomplex_div2(chk, NTOTAL as f64);
    sums[i] = chk;
}

fn compute_indexmap(twiddle: &mut Vec<f64>) {
    let ap: f64;
    let (mut kk, mut kk2, mut jj, mut kj2, mut ii): (i32, i32, i32, i32, i32);
    let nz = NZ as i32;
    let nx = NX as i32;
    let ny = NY as i32;
    /*
     * ---------------------------------------------------------------------
     * basically we want to convert the fortran indices
     * 1 2 3 4 5 6 7 8
     * to
     * 0 1 2 3 -4 -3 -2 -1
     * the following magic formula does the trick:
     * mod(i-1+n/2, n) - n/2
     * ---------------------------------------------------------------------
     */

    ap = -4.0 * ALPHA * PI * PI;
    for k in 0..D3 as i32 {
        kk = ((k + nz / 2) % nz) - nz / 2;
        kk2 = kk * kk;
        for j in 0..D2 as i32 {
            jj = ((j + ny / 2) % ny) - ny / 2;
            kj2 = jj * jj + kk2;
            for i in 0..D1 as i32 {
                ii = ((i + nx / 2) % nx) - nx / 2;
                twiddle[indxp3(k as usize, j as usize, i as usize)] =
                    (ap * (ii * ii + kj2) as f64).exp();
            }
        }
    }
}

fn evolve(u0: &mut Vec<Dcomplex>, u1: &mut Vec<Dcomplex>, twiddle: &mut Vec<f64>) {
    for k in 0..D3 {
        for j in 0..D2 {
            for i in 0..D1 {
                u0[indxp3(k, j, i)] = dcomplex_mul2(u0[indxp3(k, j, i)], twiddle[indxp3(k, j, i)]);
                u1[indxp3(k, j, i)] = u0[indxp3(k, j, i)];
            }
        }
    }
}

fn compute_initial_conditions(u0: &mut Vec<Dcomplex>) {
    let (mut x0, mut start, mut an): (f64, f64, f64);
    let mut starts = vec![0.0; NZ];

    start = SEED;
    /*
     * ---------------------------------------------------------------------
     * jump to the starting element for our first plane.
     * ---------------------------------------------------------------------
     */
    an = 0.0;
    ipow46(A, 0, &mut an);
    randlc(&mut start, an);
    ipow46(A, (2 * NX * NY) as i32, &mut an);

    starts[0] = start;
    for k in 1..D3 {
        randlc(&mut start, an);
        starts[k] = start;
        // println!("starts[{}]: {}", k, starts[k]);
    }

    /*
     * ---------------------------------------------------------------------
     * go through by z planes filling in one square at a time.
     * ---------------------------------------------------------------------
     */
    for k in 0..(D3) {
        x0 = starts[k];
        for j in 0..D2 {
            {
                let mut temp = dcomplex_interleave(&u0[indxp3(k, j, 0)..indxp3(k, j, NX)]);

                vranlc((2 * NX) as i32, &mut x0, A, &mut temp);
                //percorre o vetor temp e coloca os valores intercalados de temp em u0
                for i in 0..NX {
                    u0[indxp3(k, j, i)].real = temp[i];
                    u0[indxp3(k, j, i)].imag = temp[i + 1];
                }
            }
        }
    }

    // let mut zero_count = 0;
    // //Printa todo u0[a][b][c] onde a 0..D3-1, b 0..D2-1 c 0..NX-1
    // for k in 0..D3 {
    //     for j in 0..D2 {
    //         for i in 0..NX {
    //             if u0[indxp3(k,j,i)].real == 0.0 {
    //                 // println!("u0[{}][{}][{}] = {} = u0[{}]", k, j, i, u0[indxp3(k, j, i)].real, indxp3(k, j, i));
    //                 // panic!("ZERO");
    //                 zero_count += 1;
    //             }
    //             println!("u0[{}][{}][{}] = {} = u0[{}]", k, j, i, u0[indxp3(k, j, i)].real, indxp3(k, j, i));
    //         }
    //     }
    //     // panic!();
    // }
    // for i in 0..NTOTAL {
    //     println!("u0[{}] = {}", i, u0[i].real);
    // }
}

fn ipow46(a: f64, exponent: i32, result: &mut f64) {
    let (mut q, mut r): (f64, f64);
    let (mut n, mut n2): (i32, i32);

    /*
     * --------------------------------------------------------------------
     * use
     * a^n = a^(n/2)*a^(n/2) if n even else
     * a^n = a*a^(n-1)       if n odd
     * -------------------------------------------------------------------
     */

    *result = 1.0;
    if exponent == 0 {
        return;
    }

    q = a;
    r = 1.0;
    n = exponent;

    while n > 1 {
        n2 = n / 2;
        if n2 * 2 == n {
            let q2 = q.clone();
            randlc(&mut q, q2);
            n = n2;
        } else {
            randlc(&mut r, q);
            n = n - 1;
        }
    }
    randlc(&mut r, q);
    *result = r;
}

/**
 * Função capaz de converter um índice de um array de 3 dimensões com tamanho [NZ, NY, NX]
 * em um índice de um vetor de 1 dimensão indexado por [k, j, i]
 */
fn indxp3(k: usize, j: usize, i: usize) -> usize {
    return (k * NX + 1) * NY + i;
}

/**
 * Função capaz de converter um índice de um array de 2 dimensões com tamanho [d1, d2]
 * em um índice de um vetor de 1 dimensão indexado por [i, j], d1 e d2 são passados por argumento
 */
fn indxp2(indx1: usize, indx2: usize, dimension2: usize) -> usize {
    return indx1 * dimension2 + indx2;
}

/*
 * ---------------------------------------------------------------------
 * compute the roots-of-unity array that will be used for subsequent FFTs.
 * ---------------------------------------------------------------------
 */
fn fft_init(n: usize, u: &mut Vec<Dcomplex>) {
    /*
     * ---------------------------------------------------------------------
     * initialize the U array with sines and cosines in a manner that permits
     * stride one access at each FFT iteration.
     * ---------------------------------------------------------------------
     */

    let (mut t, mut ti): (f64, f64);

    let m = (n as f64).log2().ceil();

    u[0] = dcomplex_create(m as f64, 0.0);

    let mut ku: i32 = 2;
    let mut ln: i32 = 1;

    for _ in 1..(m as usize) {
        t = PI / ln as f64;
        for i in 0..(ln - 1) {
            ti = i as f64 * t;
            u[(i + ku - 1) as usize] = dcomplex_create(f64::cos(ti), f64::sin(ti));
        }
        ku = ku + ln;
        ln = 2 * ln;
    }
}

fn fft(dir: i32, x1: &mut RefCell<Vec<Dcomplex>>, x2: &mut RefCell<Vec<Dcomplex>>, u: &Vec<Dcomplex>) {
// fn fft(dir: i32, tx1: &mut Vec<Dcomplex>, tx2: &mut Vec<Dcomplex>, u: &Vec<Dcomplex>) {
    /*
     * ---------------------------------------------------------------------
     * note: args x1, x2 must be different arrays
     * note: args for cfftsx are (direction, layout, xin, xout, scratch)
     * xin/xout may be the same and it can be somewhat faster
     * if they are
     * ---------------------------------------------------------------------
     */

    let mut y1 = vec![dcomplex_create(0.0, 0.0); MAXDIM * FFTBLOCKPAD];
    let mut y2 = vec![dcomplex_create(0.0, 0.0); MAXDIM * FFTBLOCKPAD];

    let mut tx1 = x1.borrow_mut().to_vec();
    let mut tx2 = x2.borrow_mut().to_vec();
    if dir == 1 {
        cffts1(1, tx1.clone(), &mut tx1, &mut y1, &mut y2, u);

        cffts2(1, tx1.clone(), &mut tx1, &mut y1, &mut y2, u);

        cffts3(1, tx1, &mut tx2, &mut y1, &mut y2, u);

    } else {
        cffts3(-1, tx1.clone(), &mut tx1, &mut y1, &mut y2, u);

        cffts2(-1, tx1.clone(), &mut tx1, &mut y1, &mut y2, u);

        cffts1(-1, tx1, &mut tx2, &mut y1, &mut y2, u);
    }
}

fn init_ui(u0: &mut Vec<Dcomplex>, u1: &mut Vec<Dcomplex>, twiddle: &mut Vec<f64>) {
    for k in 0..D3 {
        for j in 0..D2 {
            for i in 0..D1 {
                u0[indxp3(k, j, i)] = dcomplex_create(0.0, 0.0);
                u1[indxp3(k, j, i)] = dcomplex_create(0.0, 0.0);
                twiddle[indxp3(k, j, i)] = 0.0;
            }
        }
    }
}

fn cffts1(
    is: i32,
    x: Vec<Dcomplex>,
    xout: &mut Vec<Dcomplex>,
    y1: &mut Vec<Dcomplex>,
    y2: &mut Vec<Dcomplex>,
    u: &Vec<Dcomplex>
) {
    let logd1: usize = (D1 as f32).log2().ceil() as usize;

    for k in 0..D3 {
        for jj in (0..=(D2 - FFTBLOCKPAD)).step_by(FFTBLOCKPAD) {
            for j in 0..FFTBLOCKPAD {
                for i in 0..D1 {
                    y1[indxp2(i, j, FFTBLOCKPAD)] = x[indxp3(k, j + jj, i)];
                }
            }

            cfftz(is, logd1, D1, y1, y2, u);
            for j in 0..FFTBLOCKPAD {
                for i in 0..D1 {
                    xout[indxp3(k, j + jj, i)] = y1[indxp2(i, j, FFTBLOCKPAD)];
                }
            }
        }
    }
}

fn cffts2(
    is: i32,
    x: Vec<Dcomplex>,
    xout: &mut Vec<Dcomplex>,
    y1: &mut Vec<Dcomplex>,
    y2: &mut Vec<Dcomplex>,
    u: &Vec<Dcomplex>
) {
    let logd2: usize = (D2 as f32).log2().ceil() as usize;

    for k in 0..D3 {
        for ii in (0..(D1 - FFTBLOCK)).step_by(FFTBLOCK) {
            for j in 0..FFTBLOCK {
                for i in 0..D2 {
                    y1[indxp2(j, i, D2)] = x[indxp3(k, j, i + ii)];
                }
            }

            cfftz(is, logd2, D2, y1, y2, u);

            for j in 0..D2 {
                for i in 0..FFTBLOCK {
                    xout[indxp3(k, j, i + ii)] = y1[indxp2(j, i, FFTBLOCK)];
                }
            }
        }
    }
}

fn cffts3(
    is: i32,
    x: Vec<Dcomplex>,
    xout: &mut Vec<Dcomplex>,
    y1: &mut Vec<Dcomplex>,
    y2: &mut Vec<Dcomplex>,
    u: &Vec<Dcomplex>,
) {
    let logd3: usize = (D3 as f32).log2().ceil() as usize;

    for j in 0..D2 {
        for ii in (0..(D1 - FFTBLOCK)).step_by(FFTBLOCK) {
            for k in 0..D3 {
                for i in 0..FFTBLOCK {
                    y1[indxp2(k, i, FFTBLOCK)] = x[indxp3(k, j, i + ii)];
                }
            }

            cfftz(is, logd3, D3, y1, y2, u);

            for k in 0..D3 {
                for i in 0..FFTBLOCK {
                    // xout[k][j][i+ii] = y1[k][i];
                    xout[indxp3(k, j, i + ii)] = y1[indxp2(i, k, D3)];
                }
            }
        }
    }
}

fn cfftz(
    is: i32,
    m: usize,
    n: usize,
    x: &mut Vec<Dcomplex>,
    y: &mut Vec<Dcomplex>,
    u: &Vec<Dcomplex>,
) {
    let mx = u[0].real;
    /*
     * ---------------------------------------------------------------------
     * perform one variant of the Stockham FFT.
     * ---------------------------------------------------------------------
     */

    for l in (1..=m).step_by(2) {
        fftz2(is, l, m, n, FFTBLOCK, FFTBLOCKPAD, u, y, x);
        if l == m {
            for j in 0..n {
                for i in 0..FFTBLOCK {
                    x[indxp2(i, j, FFTBLOCKPAD)] = y[indxp2(i, j, FFTBLOCKPAD)];
                }
            }
            break;
        }
        fftz2(is, l + 1, m, n, FFTBLOCK, FFTBLOCKPAD, u, y, x);
    }
}
fn fftz2(
    is: i32,
    l: usize,
    m: usize,
    n: usize,
    ny: usize,
    _ny1: usize,
    u: &Vec<Dcomplex>,
    y: &mut Vec<Dcomplex>,
    x: &mut Vec<Dcomplex>,
) {
    let mut tu1: Dcomplex;
    let mut tx11: Dcomplex;
    let mut tx21: Dcomplex;

    //inicializar: int k,n1,li,lj,lk,ku,i,j,i11,i12,i21,i22;
    let (mut i11, mut i12, mut i21, mut i22): (usize, usize, usize, usize);

    let (n1, lk, li, lj, ku): (usize, usize, usize, usize, usize);

    /*
     * ---------------------------------------------------------------------
     * set initial parameters.
     * ---------------------------------------------------------------------
     */
    n1 = n / 2;
    lk = 1 << (l - 1);
    li = 1 << (m - l);
    lj = 2 * lk;
    ku = li;

    for i in 0..li {
        i11 = i * lk;
        i12 = i11 + n1;
        i21 = i * lj;
        i22 = i21 + lk;
        if is >= 1 {
            tu1 = u[ku + i];
        } else {
            tu1 = dconjg(u[ku + i]);
        }

        for k in 0..lk {
            for j in 0..ny {
                tx11 = x[indxp2(i11 + k, j, FFTBLOCKPAD)].clone();
                tx21 = x[indxp2(i12 + k, j, FFTBLOCKPAD)].clone();

                y[indxp2(i21 + k, j, FFTBLOCKPAD)] = dcomplex_add(tx11, tx21);
                y[indxp2(i22 + k, j, FFTBLOCKPAD)] = dcomplex_mul(tu1, dcomplex_sub(tx11, tx21));
            }
        }
    }
}
fn verify(
    d1: usize,
    d2: usize,
    d3: usize,
    nt: usize,
    verified: &mut bool,
    class_npb: &str,
    sums: &mut Vec<Dcomplex>
) {
    let (mut err, epsilon): (f64, f64);

    /*
     * ---------------------------------------------------------------------
     * reference checksums
     * ---------------------------------------------------------------------
     */
    let mut csum_ref = vec![dcomplex_create(0.0, 0.0); 25 + 1];

    epsilon = 1.0e-12;
    *verified = false;

    if d1 == 64 && d2 == 64 && d3 == 64 && nt == 6 {
        /*
         * ---------------------------------------------------------------------
         * sample size reference checksums
         * ---------------------------------------------------------------------
         */

        //*class_npb = 'S';

        csum_ref[1] = dcomplex_create(5.546087004964E+02, 4.845363331978E+02);
        csum_ref[2] = dcomplex_create(5.546385409189E+02, 4.865304269511E+02);
        csum_ref[3] = dcomplex_create(5.546148406171E+02, 4.883910722336E+02);
        csum_ref[4] = dcomplex_create(5.545423607415E+02, 4.901273169046E+02);
        csum_ref[5] = dcomplex_create(5.544255039624E+02, 4.917475857993E+02);
        csum_ref[6] = dcomplex_create(5.542683411902E+02, 4.932597244941E+02);
    } else if d1 == 128 && d2 == 128 && d3 == 32 && nt == 6 {
        /*
         * ---------------------------------------------------------------------
         * class_npb W size reference checksums
         * ---------------------------------------------------------------------
         */
        //*class_npb = 'W';
        csum_ref[1] = dcomplex_create(5.673612178944E+02, 5.293246849175E+02);
        csum_ref[2] = dcomplex_create(5.631436885271E+02, 5.282149986629E+02);
        csum_ref[3] = dcomplex_create(5.594024089970E+02, 5.270996558037E+02);
        csum_ref[4] = dcomplex_create(5.560698047020E+02, 5.260027904925E+02);
        csum_ref[5] = dcomplex_create(5.530898991250E+02, 5.249400845633E+02);
        csum_ref[6] = dcomplex_create(5.504159734538E+02, 5.239212247086E+02);
    } else if d1 == 256 && d2 == 256 && d3 == 128 && nt == 6 {
        /*
         * ---------------------------------------------------------------------
         * class_npb A size reference checksums
         * ---------------------------------------------------------------------
         */
        //*class_npb = 'A';
        csum_ref[1] = dcomplex_create(5.046735008193E+02, 5.114047905510E+02);
        csum_ref[2] = dcomplex_create(5.059412319734E+02, 5.098809666433E+02);
        csum_ref[3] = dcomplex_create(5.069376896287E+02, 5.098144042213E+02);
        csum_ref[4] = dcomplex_create(5.077892868474E+02, 5.101336130759E+02);
        csum_ref[5] = dcomplex_create(5.085233095391E+02, 5.104914655194E+02);
        csum_ref[6] = dcomplex_create(5.091487099959E+02, 5.107917842803E+02);
    } else if d1 == 512 && d2 == 256 && d3 == 256 && nt == 20 {
        /*
         * --------------------------------------------------------------------
         * class_npb B size reference checksums
         * ---------------------------------------------------------------------
         */
        //*class_npb = 'B';
        csum_ref[1] = dcomplex_create(5.177643571579E+02, 5.077803458597E+02);
        csum_ref[2] = dcomplex_create(5.154521291263E+02, 5.088249431599E+02);
        csum_ref[3] = dcomplex_create(5.146409228649E+02, 5.096208912659E+02);
        csum_ref[4] = dcomplex_create(5.142378756213E+02, 5.101023387619E+02);
        csum_ref[5] = dcomplex_create(5.139626667737E+02, 5.103976610617E+02);
        csum_ref[6] = dcomplex_create(5.137423460082E+02, 5.105948019802E+02);
        csum_ref[7] = dcomplex_create(5.135547056878E+02, 5.107404165783E+02);
        csum_ref[8] = dcomplex_create(5.133910925466E+02, 5.108576573661E+02);
        csum_ref[9] = dcomplex_create(5.132470705390E+02, 5.109577278523E+02);
        csum_ref[10] = dcomplex_create(5.131197729984E+02, 5.110460304483E+02);
        csum_ref[11] = dcomplex_create(5.130070319283E+02, 5.111252433800E+02);
        csum_ref[12] = dcomplex_create(5.129070537032E+02, 5.111968077718E+02);
        csum_ref[13] = dcomplex_create(5.128182883502E+02, 5.112616233064E+02);
        csum_ref[14] = dcomplex_create(5.127393733383E+02, 5.113203605551E+02);
        csum_ref[15] = dcomplex_create(5.126691062020E+02, 5.113735928093E+02);
        csum_ref[16] = dcomplex_create(5.126064276004E+02, 5.114218460548E+02);
        csum_ref[17] = dcomplex_create(5.125504076570E+02, 5.114656139760E+02);
        csum_ref[18] = dcomplex_create(5.125002331720E+02, 5.115053595966E+02);
        csum_ref[19] = dcomplex_create(5.124551951846E+02, 5.115415130407E+02);
        csum_ref[20] = dcomplex_create(5.124146770029E+02, 5.115744692211E+02);
    } else if d1 == 512 && d2 == 512 && d3 == 512 && nt == 20 {
        /*
         * ---------------------------------------------------------------------
         * class_npb C size reference checksums
         * ---------------------------------------------------------------------
         */
        //*class_npb = "C";
        csum_ref[1] = dcomplex_create(5.195078707457E+02, 5.149019699238E+02);
        csum_ref[2] = dcomplex_create(5.155422171134E+02, 5.127578201997E+02);
        csum_ref[3] = dcomplex_create(5.144678022222E+02, 5.122251847514E+02);
        csum_ref[4] = dcomplex_create(5.140150594328E+02, 5.121090289018E+02);
        csum_ref[5] = dcomplex_create(5.137550426810E+02, 5.121143685824E+02);
        csum_ref[6] = dcomplex_create(5.135811056728E+02, 5.121496764568E+02);
        csum_ref[7] = dcomplex_create(5.134569343165E+02, 5.121870921893E+02);
        csum_ref[8] = dcomplex_create(5.133651975661E+02, 5.122193250322E+02);
        csum_ref[9] = dcomplex_create(5.132955192805E+02, 5.122454735794E+02);
        csum_ref[10] = dcomplex_create(5.132410471738E+02, 5.122663649603E+02);
        csum_ref[11] = dcomplex_create(5.131971141679E+02, 5.122830879827E+02);
        csum_ref[12] = dcomplex_create(5.131605205716E+02, 5.122965869718E+02);
        csum_ref[13] = dcomplex_create(5.131290734194E+02, 5.123075927445E+02);
        csum_ref[14] = dcomplex_create(5.131012720314E+02, 5.123166486553E+02);
        csum_ref[15] = dcomplex_create(5.130760908195E+02, 5.123241541685E+02);
        csum_ref[16] = dcomplex_create(5.130528295923E+02, 5.123304037599E+02);
        csum_ref[17] = dcomplex_create(5.130310107773E+02, 5.123356167976E+02);
        csum_ref[18] = dcomplex_create(5.130103090133E+02, 5.123399592211E+02);
        csum_ref[19] = dcomplex_create(5.129905029333E+02, 5.123435588985E+02);
        csum_ref[20] = dcomplex_create(5.129714421109E+02, 5.123465164008E+02);
    } else if d1 == 2048 && d2 == 1024 && d3 == 1024 && nt == 25 {
        /*
         * ---------------------------------------------------------------------
         * class_npb D size reference checksums
         * ---------------------------------------------------------------------
         */
        //*class_npb = 'D';
        csum_ref[1] = dcomplex_create(5.122230065252E+02, 5.118534037109E+02);
        csum_ref[2] = dcomplex_create(5.120463975765E+02, 5.117061181082E+02);
        csum_ref[3] = dcomplex_create(5.119865766760E+02, 5.117096364601E+02);
        csum_ref[4] = dcomplex_create(5.119518799488E+02, 5.117373863950E+02);
        csum_ref[5] = dcomplex_create(5.119269088223E+02, 5.117680347632E+02);
        csum_ref[6] = dcomplex_create(5.119082416858E+02, 5.117967875532E+02);
        csum_ref[7] = dcomplex_create(5.118943814638E+02, 5.118225281841E+02);
        csum_ref[8] = dcomplex_create(5.118842385057E+02, 5.118451629348E+02);
        csum_ref[9] = dcomplex_create(5.118769435632E+02, 5.118649119387E+02);
        csum_ref[10] = dcomplex_create(5.118718203448E+02, 5.118820803844E+02);
        csum_ref[11] = dcomplex_create(5.118683569061E+02, 5.118969781011E+02);
        csum_ref[12] = dcomplex_create(5.118661708593E+02, 5.119098918835E+02);
        csum_ref[13] = dcomplex_create(5.118649768950E+02, 5.119210777066E+02);
        csum_ref[14] = dcomplex_create(5.118645605626E+02, 5.119307604484E+02);
        csum_ref[15] = dcomplex_create(5.118647586618E+02, 5.119391362671E+02);
        csum_ref[16] = dcomplex_create(5.118654451572E+02, 5.119463757241E+02);
        csum_ref[17] = dcomplex_create(5.118665212451E+02, 5.119526269238E+02);
        csum_ref[18] = dcomplex_create(5.118679083821E+02, 5.119580184108E+02);
        csum_ref[19] = dcomplex_create(5.118695433664E+02, 5.119626617538E+02);
        csum_ref[20] = dcomplex_create(5.118713748264E+02, 5.119666538138E+02);
        csum_ref[21] = dcomplex_create(5.118733606701E+02, 5.119700787219E+02);
        csum_ref[22] = dcomplex_create(5.118754661974E+02, 5.119730095953E+02);
        csum_ref[23] = dcomplex_create(5.118776626738E+02, 5.119755100241E+02);
        csum_ref[24] = dcomplex_create(5.118799262314E+02, 5.119776353561E+02);
        csum_ref[25] = dcomplex_create(5.118822370068E+02, 5.119794338060E+02);
    } else if d1 == 4096 && d2 == 2048 && d3 == 2048 && nt == 25 {
        /*
         * ---------------------------------------------------------------------
         * class_npb E size reference checksums
         * ---------------------------------------------------------------------
         */
        //*class_npb = 'E';
        csum_ref[1] = dcomplex_create(5.121601045346E+02, 5.117395998266E+02);
        csum_ref[2] = dcomplex_create(5.120905403678E+02, 5.118614716182E+02);
        csum_ref[3] = dcomplex_create(5.120623229306E+02, 5.119074203747E+02);
        csum_ref[4] = dcomplex_create(5.120438418997E+02, 5.119345900733E+02);
        csum_ref[5] = dcomplex_create(5.120311521872E+02, 5.119551325550E+02);
        csum_ref[6] = dcomplex_create(5.120226088809E+02, 5.119720179919E+02);
        csum_ref[7] = dcomplex_create(5.120169296534E+02, 5.119861371665E+02);
        csum_ref[8] = dcomplex_create(5.120131225172E+02, 5.119979364402E+02);
        csum_ref[9] = dcomplex_create(5.120104767108E+02, 5.120077674092E+02);
        csum_ref[10] = dcomplex_create(5.120085127969E+02, 5.120159443121E+02);
        csum_ref[11] = dcomplex_create(5.120069224127E+02, 5.120227453670E+02);
        csum_ref[12] = dcomplex_create(5.120055158164E+02, 5.120284096041E+02);
        csum_ref[13] = dcomplex_create(5.120041820159E+02, 5.120331373793E+02);
        csum_ref[14] = dcomplex_create(5.120028605402E+02, 5.120370938679E+02);
        csum_ref[15] = dcomplex_create(5.120015223011E+02, 5.120404138831E+02);
        csum_ref[16] = dcomplex_create(5.120001570022E+02, 5.120432068837E+02);
        csum_ref[17] = dcomplex_create(5.119987650555E+02, 5.120455615860E+02);
        csum_ref[18] = dcomplex_create(5.119973525091E+02, 5.120475499442E+02);
        csum_ref[19] = dcomplex_create(5.119959279472E+02, 5.120492304629E+02);
        csum_ref[20] = dcomplex_create(5.119945006558E+02, 5.120506508902E+02);
        csum_ref[21] = dcomplex_create(5.119930795911E+02, 5.120518503782E+02);
        csum_ref[22] = dcomplex_create(5.119916728462E+02, 5.120528612016E+02);
        csum_ref[23] = dcomplex_create(5.119902874185E+02, 5.120537101195E+02);
        csum_ref[24] = dcomplex_create(5.119889291565E+02, 5.120544194514E+02);
        csum_ref[25] = dcomplex_create(5.119876028049E+02, 5.120550079284E+02);
    }
    
    *verified = true;
    for i in 1..nt {
        err = dcomplex_abs(dcomplex_div(
            dcomplex_sub(sums[i], csum_ref[i]),
            csum_ref[i],
        ));
        if !(err <= epsilon) {
            *verified = false;
            break;
        }
    }

    if *verified {
        println!(" Result verification successful\n");
    } else {
        println!(" Result verification failed\n");
    }
}

/*
fn init_ui(
    u0: &Vec<Dcomplex>,
    u1: &Vec<Dcomplex>,
    twiddle: &Vec<f64>,
    d1: usize,
    d2: usize,
    d3: usize
) {
    for k in 0..d3 {
        for j in 0..d2 {
            for i in 0..d1 {
                u0[indxp3(k, j, i)];
            }
        }
    }
}
*/
