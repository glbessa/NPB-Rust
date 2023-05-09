//NPB SET_PARAMS GLOBAL VARIABLES
const CLASS: &str= %% CLASS_NPB %%;
const M: u32 = %% M %%;
const MM: u32 = M - MK;
const NN: u32 = 1 << MM;
const COMPILETIME: &str = %% COMPILE_TIME %%;
const NPBVERSION: &str = "4.1";
const COMPILERVERSION: &str = "rustc 1.70.0-nightly";
const LIBVERSION: &str = "1.0";

//EP GLOBAL VARIABLES
const MK: u32 = 16;
const NK: usize = 1 << MK; // 2**MK
const EPSILON: f64 = 1.0e-8;
const A: f64 = 1220703125.0;
const S: f64 = 271828183.0;
const NQ: u32 = 10;
const NK_PLUS: usize = (2 * NK) + 1;

//IMPORTS
use common::randdp;
use common::print_results;
use std::time::{Duration, Instant};
use std::mem::MaybeUninit;
use std::ptr;
use std::env;
use chrono::{Local, DateTime};

//BEGGINING OF EP
fn main() {
    // Integer Variables
    let (mut ik, mut l, nit): (i32, usize, i32);
    let mut kk: i32;
    let k_offset: i32 = -1;
    let np: i32 = NN as i32;
    let mut k:i32;
    // Double Variables
    let mut aux: f64;
    let (mut t1, mut t2, mut t3, mut t4): (f64, f64, f64, f64);
    let (mut x1, mut x2): (f64, f64); // don't forget "x" (it's a contiguous block of memory)
    let (mut sx, mut sy,tm, an, mut gc): (f64, f64, f64, f64, f64);
    let (mut sx_verify_value,mut sy_verify_value): (f64, f64);
    sx_verify_value = -1.0e99; //adicionado por causa do erro: used binding `sx_verify_value` is possibly-uninitialized
    sy_verify_value = -1.0e99; //adicionado por causa do erro: used binding `sx_verify_value` is possibly-uninitialized
    let (sx_err, sy_err): (f64, f64);

    // Boolean Variables
    let mut verified: bool;
    let timers_enabled: bool = false;
    
    let mut x: Vec<f64> = Vec::with_capacity(NK_PLUS);
    let mut q: [f64;NQ as usize] = [0.0;NQ as usize];
    let mut dum0 = 1.0;
    let mut dum1 = 1.0;
    let mut dum2: Vec<f64> = Vec::with_capacity(1);
    
    dum2.push(1.0);
    randdp::vranlc(0, &mut dum0, dum1, &mut dum2);
    let dum3 = 1.0;
    let _dum0:f64 = randdp::randlc(&mut dum1, dum3);
    
    unsafe {
        let ptr = x.as_mut_ptr();
        ptr::write_bytes(ptr, 0xFF, NK_PLUS); // initializes the vector to all 1s
        let default_value = MaybeUninit::new(-1.0e99);
        for i in 0..NK_PLUS {
            ptr::write(ptr.offset(i as isize), default_value.assume_init());
        }
        x.set_len(NK_PLUS);
    }

    let start = Instant::now();
    t1 = A;
    randdp::vranlc(0, &mut t1, A, &mut x);

    t1 = A;

    for _ in 0..(MK + 1) {
        aux = t1;
        let _t2 = randdp::randlc(&mut t1, aux);
    }

    an = t1;
    gc = 0.0;
    sx = 0.0;
    sy = 0.0;

    for k in 1..np+1 {
        kk = k_offset + k;
        t1 = S;
        t2 = an;

        // fin starting seed t1 for this kk
        for _i in 1..=100 {
            ik = kk / 2;
            if (2 * ik) != kk {
                t3 = randdp::randlc(&mut t1, t2);
            }
            if ik == 0 {
                break;
            }
            aux = t2;
            t3 = randdp::randlc(&mut t2, aux);
            kk = ik;
        }

        // compute uniform pseudorandom numbers
        if timers_enabled {
            //timer_start(2);
        }
        randdp::vranlc((2 * NK) as i32, &mut t1, A, &mut x);
        if timers_enabled {
            //timer_stop(2);
        }

        /*
         * compute gaussian deviates by acceptance-rejection method and
         * tally counts in concentric square annuli. this loop is not
         * vectorizable.
         */

        for i in 0..NK {
            let ptr1 = unsafe {x.as_ptr().offset((2 * i) as isize)};
            x1 = 2.0 * unsafe { *ptr1 } - 1.0;
            let ptr = unsafe {x.as_ptr().offset((2 * i + 1) as isize)};
            x2 = 2.0 * unsafe { *ptr } - 1.0;
            t1 = f64::powi(x1, 2) + f64::powi(x2, 2);
            if t1 <= 1.0 {
                t2 = (-2.0 * t1.ln() / t1).sqrt();
                t3 = x1 * t2;
                t4 = x2 * t2;
                l = t3.abs().max(t4.abs()) as usize;
                q[l] += 1.0;
                sx += t3;
                sy += t4;
            }
        }
    }
    for item in q.iter().take((NQ-1) as usize + 1){
        gc += item;
    }

    tm = start.elapsed().as_secs_f64();
    nit = 0;
    verified = true;
    if M == 24 {
		sx_verify_value = -3.247_834_652_034_74e3;
		sy_verify_value = -6.958_407_078_382_297e3;
	}else if M == 25 {
		sx_verify_value = -2.863_319_731_645_753e3;
		sy_verify_value = -6.320_053_679_109_499e3;
	}else if M == 28 {
		sx_verify_value = -4.295_875_165_629_892e3;
		sy_verify_value = -1.580_732_573_678_431e4;
	}else if M == 30 {
		sx_verify_value =  4.033_815_542_441_498e4;
		sy_verify_value = -2.660_669_192_809_235e4;
	}else if M == 32 {
		sx_verify_value =  4.764_367_927_995_374e4;
		sy_verify_value = -8.084_072_988_043_731e4;
	}else if M == 36 {
		sx_verify_value =  1.982_481_200_946_593e5;
		sy_verify_value = -1.020_596_636_361_769e5;
	}else if  M == 40 {
		sx_verify_value = -5.319_717_441_530e5;
		sy_verify_value = -3.688_834_557_731e5;
	}else {
		verified = false;
	}
    if verified {
        sx_err = ((sx - sx_verify_value) / sx_verify_value).abs();
        sy_err = ((sy - sy_verify_value) / sy_verify_value).abs();
        verified = (sx_err <= EPSILON) && (sy_err <= EPSILON);
    }
    else{
        println!("Alguma coisa de errado não etá certo!");
    }
    let mops: f64 = (((1 as i64) << ((M as i64) + 1)) as f64) / tm / 1000000.0;
    println!(" EP Benchmark Results:\n");
    println!(" CPU Time = NOT YET IMPLEMENTED");
    println!(" N = 2^{}",M);
    println!(" No. Gaussian Pairs = {:>15}",gc);
    println!(" Sums: sx = {:25.15e} sy = {:25.15e}",sx,sy); // %25.15e
    println!(" Counts: ");
    for i in 0..(NQ-1) as usize{
        println!("{}     {}",i,q[i]);
    }
    print_results::rust_print_results("EP",
                        CLASS,
                        M+1,
                        0,
                        0,
                        nit,
                        tm,
                        mops,
                        "Random numbers generated",
                        verified,
                        NPBVERSION,
                        COMPILETIME,
                        COMPILERVERSION,
                        LIBVERSION,
                        "OMP_NUM_THREADS",
                        "cs1",
                        "cs2",
                        "cs3",
                        "cs4",
                        "cs5",
                        "cs6",
                        "cs7");
}
