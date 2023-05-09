// if(class_npb == 'S'){problem_size = 32; nit = 4;}
// 	else if(class_npb == 'W'){problem_size = 128; nit = 4;}
// 	else if(class_npb == 'A'){problem_size = 256; nit = 4;}
// 	else if(class_npb == 'B'){problem_size = 256; nit = 20;}
// 	else if(class_npb == 'C'){problem_size = 512; nit = 20;}
// 	else if(class_npb == 'D'){problem_size = 1024; nit = 50;}
// 	else if(class_npb == 'E'){problem_size = 2048; nit = 50;}
//----------PARAMS PASSED TO PROBLEM

const NIT: usize = 50; // for class S
const NX_DEFAULT: usize = 2048;
const NY_DEFAULT: usize = 2048;
const NZ_DEFAULT: usize = 2048;
const NIT_DEFAULT: usize = NIT;
const LM: usize = ilog2(2048) as usize;
const LT_DEFAULT: usize = LM;
const DEBUG_DEFAULT:usize = 0;
const NDIM1: usize = LM;
const NDIM2: usize = ilog2(2048) as usize;
const NDIM3: usize = ilog2(2048) as usize;
const ONE: usize = 1;

const NPBVERSION: &str = "4.1.2";
const COMPILETIME: &str = "2023-04-30T16:35:50.551413853-03:00";
const COMPILERVERSION: &str = "rustc 1.70.0-nightly";
const LIBVERSION: &str = "1";
const CS1: &str = "";
const CS2: &str = "";
const CS3: &str = "";
const CS4: &str = "";
const CS5: &str = "";
const CS6: &str = "";
const CS7: &str = "";

//----------------------------------
//----------------------------------
//----TAMANHOS & COMPRIMENTOS-------
//----------------------------------
//----------------------------------

//actual dimension including ghost cells for communications
const NM: usize = 2+(1<<LM); 
//size of rhs array
const NV: usize = ONE*(2+(1<<NDIM1))*(2+(1<<NDIM2))*(2+(1<<NDIM3));
//size of residual array
const NR: usize = ((NV+NM*NM+5*NM+7*LM+6)/7)*8;
//maximum number of levels
const MAXLEVEL: usize = LT_DEFAULT+1;
// set at m=1024, can handle cases up to 1024ˆ3 case
const M: usize = NM+1;
const MM: usize = 10;
const A: f64 = 1220703125.0;
const X: f64 = 314159265.0;

//----------------------------------
//----------------------------------
//----TIMERS POSITIONS FLAGS--------
//------NOT IMPLEMENTED IET---------
//----------------------------------
// const T_INIT   = 0;
// const T_BENCH  = 1;
// const T_MG3P   = 2;
// const T_PSINV  = 3;
// const T_RESID  = 4;
// const T_RESID2 = 5;
// const T_RPRJ3  = 6;
// const T_INTERP = 7;
// const T_NORM2  = 8;
// const T_COMM3  = 9;
// const T_LAST   = 10;

//----------------------------------
//----------------------------------
//------STATIC VARIABLES------------
//----------------------------------
//----------------------------------

static lt: usize = LT_DEFAULT;
static nit: usize = NIT_DEFAULT;
static lb: usize = 1;
static mut is1: usize = 0;
static mut is2: usize = 0;
static mut is3: usize = 0;
static mut ie1: usize = 0;
static mut ie2: usize = 0;
static mut ie3: usize = 0;

//IMPORTS
use common::randdp::{self, vranlc};
use common::print_results;
use std::env;
use libm::{self, fabs};
use ndarray::prelude::*;
use std::time::Instant;

fn main() {
    let mut nx: Vec<usize> = vec![0;MAXLEVEL+1];
    let mut ny: Vec<usize> = vec![0;MAXLEVEL+1];
    let mut nz: Vec<usize> = vec![0;MAXLEVEL+1];
    let mut m1: Vec<usize> = vec![0;MAXLEVEL+1];
    let mut m2: Vec<usize> = vec![0;MAXLEVEL+1];
    let mut m3: Vec<usize> = vec![0;MAXLEVEL+1];
    let mut ir: Vec<usize> = vec![0;MAXLEVEL+1];
    let mut debug_vec: Vec<usize> = Vec::with_capacity(8);

	 // -----------------------------------------------------------------
     // k is the current level. it is passed down through subroutine args
	 // and is not global. it is the current iteration
	 //--------------------------------------------------------------
    
    let (mut k, mut it):(usize,usize);
    let (mut t, tinit, mflops):(f64,f64,f64);
    let mut a = vec![0.0;4];
    let mut c = vec![0.0;4];
    let (mut rnm2, mut rnmu): (f64,f64) = (0.0,0.0);
    let mut epsilon:f64 = 0.0;
    let (mut n1,mut n2,mut n3): (usize,usize,usize) = (0,0,0);
    let (mut nn, mut veriy_value, mut err): (f64,f64,f64);
    let mut verified: bool = false;

    t = 0.0;
    
    let init_timer = Instant::now();

    nx[lt] = NX_DEFAULT ;
    ny[lt] = NY_DEFAULT ;
    nz[lt] = NZ_DEFAULT ;

    let mut tmax:f64;
    let class_npb: &str;

    if nx[lt] != ny[lt] || nx[lt] != nz[lt] {
        class_npb = "U";
    }
    else if nx[lt] == 32 && nit ==4 {
        class_npb = "S";
    }
    else if nx[lt] == 128 && nit ==4 {
        class_npb = "W";
    }
    else if nx[lt] == 256 && nit ==4 {
        class_npb = "A";
    }
    else if nx[lt] == 256 && nit ==20 {
        class_npb = "B";
    }
    else if nx[lt] == 512 && nit ==20 {
        class_npb = "C";
    }
    else if nx[lt] == 1024 && nit ==50 {
        class_npb = "D";
    }
    else if nx[lt] == 2048 && nit ==50 {
        class_npb = "E";
    }
    else {
        class_npb = "U";
    }

    a[0] = -8.0/3.0;
    a[2] = 1.0/6.0;
    a[3] = 1.0/12.0;

	if class_npb == "A" || class_npb == "S" || class_npb =="W" {
		/* coefficients for the s(a) smoother */
		c[0] =  -3.0/8.0;
		c[1] =  1.0/32.0;
		c[2] =  -1.0/64.0;
	} else {
		/* coefficients for the s(b) smoother */
		c[0] =  -3.0/17.0;
		c[1] =  1.0/33.0;
		c[2] =  -1.0/61.0;
	}

    //lb = 1;
    k = lt;
    
    //----------------------------------
    //-----IN-SCOPE FUNCTION SETUP------
    //----------------------------------
    {
        let mut j:usize;
        let mut ax: usize;
        let mut mi: Vec<[usize; 3]> = vec![[0;3];MAXLEVEL+1];
        let mut ng: Vec<[usize; 3]> = vec![[0;3];MAXLEVEL+1];
        let mut k = lt;
        ng[lt][0] = nx[lt];
        ng[lt][1] = ny[lt];
        ng[lt][2] = nz[lt];
        for ax in 0..3 {
            for k in (1..lt).rev() {
                ng[k][ax] = ng[k+1][ax] / 2;
            }
        }
        for k in (1..=lt).rev() {
            nx[k] = ng[k][0];
            ny[k] = ng[k][1];
            nz[k] = ng[k][2];
        }

        for k in (1..=lt).rev() {
            for ax in 0..3 {
                mi[k][ax] = 2 + ng[k][ax];
            }

            m1[k] = mi[k][0];
            m2[k] = mi[k][1];
            m3[k] = mi[k][2];
        }

        unsafe {
            is1 = 2 + ng[k][0] - ng[lt][0]; //is1 = 2 + ng[k][0] - ng[lt][0];
            ie1 = 1 + ng[k][0];             // ie1 = 1 + ng[k][0];
            n1  = 3 + ie1 - is1;            // *n1 = 3 + ie1 - is1;
            is2 = 2 + ng[k][1] - ng[lt][1]; // is2 = 2 + ng[k][1] - ng[lt][1];
            ie2 = 1 + ng[k][1];             // ie2 = 1 + ng[k][1];
            n2  = 3 + ie2 - is2;            // *n2 = 3 + ie2 - is2;
            is3 = 2 + ng[k][2] - ng[lt][2]; //is3 = 2 + ng[k][2] - ng[lt][2];
            ie3 = 1 + ng[k][2];             //ie3 = 1 + ng[k][2];
            n3  = 3 + ie3 - is3;            //*n3 = 3 + ie3 - is3;
        };

        ir[lt] = 0;

        for j in (1..lt).rev() {
            ir[j] = ir[j+1] + 1 * m1[j+1] * m2[j+1] * m3[j+1];
        }
    }

    // zero3 FUNCTION
    let mut u  = Array1::<f64>::zeros(NR);
    let mut v = Array1::<f64>::zeros(NV);
    let mut r = Array1::<f64>::zeros(NR);

    zran3(&mut v, n1, n2, n3, nx[lt], ny[lt], k);

    norm2u3(&mut v,n1,n2, n3, &mut rnm2,&mut rnmu, nx[lt],ny[lt],nz[lt]);
    
    println!("\n NAS Parallel Benchmarks 4.1 Serial Rust version - MG Benchmark\n");
	println!(" Size: {} x {} x {} (class_npb {})", nx[lt], ny[lt], nz[lt], class_npb);
	println!(" Iterations: {}\n", nit);

    let mut ptr_u = u.slice_mut(s![0..NR]);
    let mut ptr_v = v.slice_mut(s![0..NV]);
    let mut ptr_r = r.slice_mut(s![0..NR]);
    
    resid(&mut ptr_u, &mut ptr_v, &mut ptr_r, n1, n2, n3,&mut a,k);
    
    norm2u3(&mut r,n1,n2,n3, &mut rnm2,&mut rnmu, nx[lt],ny[lt],nz[lt]);
    
    mg3P(&mut u, &mut v, &mut r, &mut a, &mut c, n1, n2, n3, k ,&mut ir,&mut m1,&mut m2,&mut m3);
    
    let mut ptr_u = u.slice_mut(s![0..NR]);
    let mut ptr_v = v.slice_mut(s![0..NV]);
    let mut ptr_r = r.slice_mut(s![0..NR]);
    
    resid(&mut ptr_u, &mut ptr_v, &mut ptr_r, n1, n2, n3, &mut a, k);
    
    //----------------------------------
    //-----IN-SCOPE FUNCTION SETUP------
    //----------------------------------
    {
        let mut j:usize;
        let mut ax: usize;
        let mut mi: Vec<[usize; 3]> = vec![[0;3];MAXLEVEL+1];
        let mut ng: Vec<[usize; 3]> = vec![[0;3];MAXLEVEL+1];
        let mut k = lt;
        ng[lt][0] = nx[lt];
        ng[lt][1] = ny[lt];
        ng[lt][2] = nz[lt];
        for ax in 0..3 {
            for k in (1..lt).rev() {
                ng[k][ax] = ng[k+1][ax] / 2;
            }
        }
        for k in (1..=lt).rev() {
            nx[k] = ng[k][0];
            ny[k] = ng[k][1];
            nz[k] = ng[k][2];
        }
        
        for k in (1..=lt).rev() {
            for ax in 0..3 {
                mi[k][ax] = 2 + ng[k][ax];
            }

            m1[k] = mi[k][0];
            m2[k] = mi[k][1];
            m3[k] = mi[k][2];
        }

        unsafe {
            is1 = 2 + ng[k][0] - ng[lt][0]; //is1 = 2 + ng[k][0] - ng[lt][0];
            ie1 = 1 + ng[k][0];             // ie1 = 1 + ng[k][0];
            n1  = 3 + ie1 - is1;            // *n1 = 3 + ie1 - is1;
            is2 = 2 + ng[k][1] - ng[lt][1]; // is2 = 2 + ng[k][1] - ng[lt][1];
            ie2 = 1 + ng[k][1];             // ie2 = 1 + ng[k][1];
            n2  = 3 + ie2 - is2;            // *n2 = 3 + ie2 - is2;
            is3 = 2 + ng[k][2] - ng[lt][2]; //is3 = 2 + ng[k][2] - ng[lt][2];
            ie3 = 1 + ng[k][2];             //ie3 = 1 + ng[k][2];
            n3  = 3 + ie3 - is3;            //*n3 = 3 + ie3 - is3;
        
        };

        ir[lt] = 0;

        for j in (1..lt).rev() {
            ir[j] = ir[j+1] + 1 * m1[j+1] * m2[j+1] * m3[j+1];
        }
    }

    //zero function
    {
        let mut u_fill = u.slice_mut(s![0..((n1 * n2 * n3))]);
        u_fill.fill(0.0);
    }

    zran3(&mut v, n1 , n2 , n3 , nx[lt], ny[lt], k);
    
    let mut ptr_u = u.slice_mut(s![0..NR]);
    let mut ptr_v = v.slice_mut(s![0..NV]);
    let mut ptr_r = r.slice_mut(s![0..NR]);

    t = init_timer.elapsed().as_secs_f64();
	println!(" Initialization time = {} seconds", &t);
    let bench_timer = Instant::now();

    resid(&mut ptr_u, &mut ptr_v, &mut ptr_r, n1 , n2 , n3 , &mut a, k );

    norm2u3(&mut r, n1 , n2 , n3 , &mut rnm2, &mut rnmu, nx[lt], ny[lt], nz[lt]);

    for it in 1 ..= nit {
        mg3P(&mut u, &mut v, &mut r, &mut a, &mut c, n1 , n2 , n3 , k , &mut ir, &mut m1, &mut m2, &mut m3);

        let mut ptr_u = u.slice_mut(s![0..NR]);
        let mut ptr_v = v.slice_mut(s![0..NV]);
        let mut ptr_r = r.slice_mut(s![0..NR]);
        
        resid(&mut ptr_u, &mut ptr_v, &mut ptr_r, n1, n2, n3, &mut a, k);
    }

    norm2u3(&mut r, n1, n2, n3, &mut rnm2, &mut rnmu, nx[lt], ny[lt], nz[lt]);
    
    t = bench_timer.elapsed().as_secs_f64();

    let mut verify_value = 0.0;
    
    println!(" Benchmark Completed!");
    
    epsilon = 1.0e-8;
    
    if class_npb == "S" {
        verify_value = 0.5307707005734e-04;
    } else if class_npb == "W" {
        verify_value = 0.6467329375339e-05;
    } else if class_npb == "A" {
        verify_value = 0.2433365309069e-05;
    } else if class_npb == "B" {
        verify_value = 0.1800564401355e-05;
    } else if class_npb == "C" {
        verify_value = 0.5706732285740e-06;
    } else if class_npb == "D" {
        verify_value = 0.1583275060440e-09;
    } else if class_npb == "E" {
        verify_value = 0.8157592357404e-10; 
    }

    err = fabs(rnm2 - verify_value) / verify_value;
    
    if err <= epsilon{
        verified = true;
        println!(" VERIFICATION SUCCESSFUL");
        println!(" L2 Norm is {}", &rnm2);
		println!(" Error is   {}", &err);
    }
    else{
        verified = false;
        println!("VERIFICATION FAILED");
        println!(" L2 Norm is             {}", &rnm2);
		println!(" The correct L2 Norm is {}", &verify_value);
    }

    nn = 1.0 * nx[lt] as f64 * ny[lt] as f64 * nz[lt] as f64;

	if t != 0.0 {
		mflops = 58.0 * nit as f64 * nn * 1.0e-6 / t;
	} else {
		mflops = 0.0;
	}

	print_results::rust_print_results("MG",
			class_npb,
			nx[lt].try_into().unwrap(),
			ny[lt].try_into().unwrap(),
			nz[lt].try_into().unwrap(),
			nit.try_into().unwrap(),
			t,
			mflops,
			"          floating point",
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

fn rprj3(r_ptr: Array1<f64>, m1k:usize, m2k:usize,m3k:usize,s_ptr: &mut ArrayViewMut<'_, f64, Dim<[usize; 1]>>, m1j: usize, m2j: usize,m3j: usize, k:usize){
    let r = r_ptr.slice(s![0..(m1k * m2k * m3k)]).into_shape((m1k, m2k, m3k)).unwrap();
    let mut s = s_ptr.slice_mut(s![0..(m1j * m2j * m3j)]).into_shape((m1j, m2j, m3j)).unwrap();
    
    let (mut i3,mut i2, mut i1): (usize, usize, usize);
    let (mut d1,mut d2, mut d3): (usize, usize, usize);
    let mut x1:[f64; M] = [0.0; M];
    let mut y1:[f64; M] = [0.0; M];
    let (mut x2, mut y2): (f64, f64);
    let mut j: usize;
    
    if m1k == 3 {
        d1 = 2;
    } else {
        d1 = 1;
    }

    if m2k == 3 {
        d2 = 2;
    } else {
        d2 = 1;
    }

    if m3k == 3 {
        d3 = 2;
    } else {
        d3 = 1;
    }

    for j3 in 1..(m3j - 1) {
        i3 = 2 * j3 - d3;

        for j2 in 1..(m2j - 1) {
            i2 = 2 * j2 - d2;
            for j1 in 1..m1j {
                i1 = 2 * j1 - d1;

                x1[i1] = r[ [i3 + 1, i2, i1] ] + r[ [i3 + 1, i2 + 2, i1] ]
                       + r[ [i3, i2 + 1, i1] ] + r[ [i3 + 2, i2 + 1, i1] ];
                
                y1[i1] = r[ [i3, i2, i1] ] + r[ [i3 + 2, i2, i1] ]
                       + r[ [i3, i2 + 2, i1]] + r[ [i3 + 2, i2 + 2, i1] ];
            }

            for j1 in 1..(m1j - 1) {
                i1 = 2 * j1 - d1;
            
                y2 = r[ [i3, i2, i1 + 1] ] + r[ [i3 + 2, i2, i1 + 1] ]
                   + r[ [i3, i2 + 2, i1 + 1] ] + r[ [i3 + 2, i2 + 2, i1 + 1] ];
            
                x2 = r[ [i3 + 1, i2, i1 + 1] ] + r[ [i3 + 1, i2 + 2, i1 + 1] ]
                   + r[ [i3, i2 + 1, i1 + 1] ] + r[ [i3 + 2, i2 + 1, i1 + 1] ];

                s[ [j3, j2, j1] ] = 0.5 * r[ [i3 + 1, i2 + 1, i1 + 1] ]
                                  + 0.25 * (r[ [i3 + 1, i2 + 1, i1] ] + r[ [i3 + 1, i2 + 1, i1 + 2] ] + x2)
                                  + 0.125 * (x1[i1] + x1[i1 + 2] + y2)
                                  + 0.0625 * (y1[i1] + y1[i1 + 2]);
            }
        }
    }

    j = k - 1;

    viewd_comm3(s_ptr, m1j, m2j, m3j, j);
    
    let mut count = 0.0;
    let mut ss = s_ptr.slice_mut(s![0..(m1j * m2j * m3j)]).into_shape((m1j, m2j, m3j)).unwrap();
    
    for i3 in 0..(m3j - 1) {
        for i2 in 0..(m2j - 1) {
            for i1 in 0..(m1j - 1) {
                count += ss[[i3, i2, i1]];
            }
        }
    }
}

fn psinv(r_ptr: &mut ArrayViewMut<'_, f64, Dim<[usize; 1]>>, u_ptr: &mut ArrayViewMut<'_, f64, Dim<[usize; 1]>>,n1:usize,n2:usize,n3:usize,c: &mut Vec<f64>,k:usize){
    let mut r = r_ptr.slice_mut(s![0..(n1*n2*n3)]).into_shape((n1,n2,n3)).unwrap();
    let mut u = u_ptr.slice_mut(s![0..(n1*n2*n3)]).into_shape((n1,n2,n3)).unwrap();
    let mut r1: [f64;M] = [0.0;M];
    let mut r2: [f64;M] = [0.0;M];

    for i3 in 1..(n3 - 1) {
        for i2 in 1..(n2 - 1) {
            for i1 in 0..n1 {
                r1[i1] = r[ [i3, i2 - 1, i1] ] + r[ [i3, i2 + 1, i1] ]
                       + r[ [i3 - 1, i2, i1] ] + r[ [i3 + 1, i2, i1] ];
                r2[i1] = r[ [i3 - 1, i2 - 1, i1]] + r[ [i3 - 1, i2 + 1, i1] ]
                       + r[ [i3 + 1, i2 - 1, i1]] + r[ [i3 + 1, i2 + 1, i1] ];     
            }
            for i1 in 1..(n1 - 1) {
                u[[ i3, i2, i1]] = u[ [i3, i2, i1] ]
                               + c[0] * r[[i3, i2, i1]]
                               + c[1] * (r[ [i3, i2, i1 - 1]] + r[ [i3, i2, i1 + 1] ] + r1[i1])
                               + c[2] * (r2[i1] + r1[i1 - 1] + r1[i1 + 1]);
            }
        }
    }

    viewd_comm3(u_ptr, n1, n2, n3, k);
}

fn mg3P(u: &mut Array1<f64>,v: &mut Array1<f64>, r: &mut Array1<f64>, a: &mut Vec<f64>, c: &mut Vec<f64>, n1: usize, n2:usize, n3: usize, kk: usize,ir: &mut Vec<usize>,m1: &mut Vec<usize>, m2: &mut Vec<usize>, m3: &mut Vec<usize>){
    let mut j;
    let mut k = kk;
    
    for k in (lb + 1..=lt).rev() {
        j = k-1;
        
        let r_ptr = r.slice(s![(ir[k])..((ir[k] + (m1[k] * m2[k] * m3[k])))]).to_owned();
        let mut s_ptr = r.slice_mut(s![(ir[j])..((ir[j] + (m1[j] * m2[j] * m3[j])))]);
        rprj3(r_ptr, m1[k] , m2[k] , m3[k] , &mut s_ptr, m1[j] , m2[j] , m3[j] , k);
    }
    
    k = lb ;

    //ZERO3 function
    {
        let len = m1[k] * m2[k] * m3[k];
        let mut u_slice = u.slice_mut(s![ir[k]..(ir[k] + len)]);
        u_slice.fill(0.0);
    }
    
    //psinv
    let mut r_ptr = r.slice_mut(s![ir[k]..(ir[k] + (m1[k] * m2[k] * m3[k]))]);
    let mut u_ptr = u.slice_mut(s![ir[k]..(ir[k] + (m1[k] * m2[k] * m3[k]))]);
    psinv(&mut r_ptr, &mut u_ptr, m1[k] , m2[k] , m3[k] , c, k );
    {
        let mut count = 0.0;
        for i in 0..NR {
            count += u[i];
        }
    }
    
    for k in (lb + 1)..=(lt - 1) {
        j = k - 1;

        //ZERO3 FUNCTION
        {
            let len = m1[k] * m2[k] * m3[k];
            let mut u_slice = u.slice_mut(s![ir[k]..(ir[k] + len)]);
            u_slice.fill(0.0);
        }

        //interp
        let z_ptr = u.slice(s![(ir[j] )..((m1[j] * m2[j] * m3[j]) + ir[j])]).to_owned();
        let mut u_ptr = u.slice_mut(s![ir[k]..(ir[k] + (m1[k] * m2[k] * m3[k]))]);
        interp(z_ptr, m1[j], m2[j], m3[j], &mut u_ptr, m1[k], m2[k], m3[k], k);

        {
            let mut count = 0.0;

            for i in 0..NR {
                count += u[i];
            }
        }

        {
            let mut v_ptr = r.slice_mut(s![ir[k]..(ir[k] + (m1[k] * m2[k] * m3[k]))]).to_owned();
            let mut u_ptr = u.slice_mut(s![ir[k]..(ir[k] + (m1[k] * m2[k] * m3[k]))]);
            let mut r_ptr = r.slice_mut(s![ir[k]..(ir[k] + (m1[k] * m2[k] * m3[k]))]);
        
            resid_two(&mut u_ptr,&mut v_ptr,&mut r_ptr, m1[k] , m2[k] , m3[k] , a, k );
        }

        {
            let mut count = 0.0;
            let rr = r.clone();

            for i in 0..NR {
                count += rr[i];
            }
        }

        {
            let mut v_ptr = r.slice_mut(s![ir[k]..(ir[k] + (m1[k] * m2[k] * m3[k]))]).to_owned();
            let mut u_ptr = u.slice_mut(s![ir[k]..(ir[k] + (m1[k] * m2[k] * m3[k]))]);
            let mut r_ptr = r.slice_mut(s![ir[k]..(ir[k] + (m1[k] * m2[k] * m3[k]))]);
        
            psinv(&mut r_ptr, &mut u_ptr, m1[k] , m2[k] , m3[k] , c, k );
        }

        {
            let mut count = 0.0;

            for i in 0..NR {
                count += u[i];
            }
        }
    }
    
    j = lt - 1;
    k = lt;

    {
        let z_ptr = u.slice(s![(ir[j] )..((ir[j] + m1[j] * m2[j] * m3[j]))]).to_owned();
        let mut u_ptr = u.slice_mut(s![0..(n1 * n2 * n3)]);
    
        interp(z_ptr, m1[j] , m2[j] , m3[j] , &mut u_ptr, n1, n2, n3,k );
    }

    {
        let mut count = 0.0;

        for i in 0..NR {
            count += u[i];
        }
    }

    {
        let mut u_ptr = u.slice_mut(s![0..((n1 * n2 * n3))]);
        let mut v_ptr = v.slice_mut(s![0..((n1 * n2 * n3))]);
        let mut r_ptr = r.slice_mut(s![0..((n1 * n2 * n3))]);

        resid(&mut u_ptr, &mut v_ptr, &mut r_ptr, n1, n2, n3, a, k );
    }

    {
        let mut count = 0.0;
        
        for i in 0..NR {
            count += r[i];
        }
    }

    {
        let mut u_ptr = u.slice_mut(s![0..((n1 * n2 * n3))]);
        let mut r_ptr = r.slice_mut(s![0..((n1 * n2 * n3))]);

        psinv(&mut r_ptr, &mut u_ptr, n1, n2, n3, c, k );
    }

    {
        let mut count = 0.0;

        for i in 0..NR {
            count += u[i];
        }
    }
}

fn interp(z_ptr: Array1<f64>, mm1:usize, mm2:usize, mm3:usize,u_ptr: &mut ArrayViewMut<'_, f64, Dim<[usize; 1]>>, n1:usize, n2:usize, n3:usize, k:usize) {
    let z = z_ptr.slice(s![0..(mm1 * mm2 * mm3)]).into_shape((mm1, mm2, mm3)).unwrap();
    let mut u = u_ptr.slice_mut(s![0..(n1 * n2 * n3)]).into_shape((n1, n2, n3)).unwrap();
    
    let (mut d1,mut d2,mut d3): (usize,usize,usize);
    let (mut t1,mut t2,mut t3): (usize, usize,usize);
    let mut z1: [f64; M] = [0.0; M];
    let mut z2: [f64; M] = [0.0; M];
    let mut z3: [f64; M] = [0.0; M];

    if n1 != 3 && n2 != 3 && n3 != 3 {
        for i3 in 0..(mm3 - 1) {
            for i2  in 0..(mm2 - 1) {
                for i1 in 0..mm1 {
                    z1[i1] = z[ [i3, i2 + 1, i1] ] + z[ [i3, i2, i1] ];
                    z2[i1] = z[ [i3 + 1, i2, i1] ] + z[ [i3, i2, i1] ];
                    z3[i1] = z[ [i3 + 1, i2 + 1, i1] ] + z[ [i3 + 1, i2, i1] ] + z1[i1];
                }

                for i1 in 0..(mm1 - 1) {
                    u[ [2 * i3, 2 * i2, 2 * i1] ] = u[ [2 * i3, 2 * i2, 2 * i1] ]
                                          + z[ [i3, i2, i1] ];
                    u[ [2 * i3, 2 * i2, 2 * i1 + 1]] = u[ [2 * i3, 2 * i2, 2 * i1 + 1] ]
                                           + 0.5 * (z[ [i3, i2, i1 + 1] ] + z[ [i3, i2, i1] ]);
                }

                for i1 in 0..(mm1 - 1) {
                    u[ [2 * i3, 2 * i2 + 1, 2 * i1] ] = u[ [2 * i3, 2 * i2 + 1, 2 * i1] ]
                                            + 0.5 * z1[i1];
                    u[ [2 * i3, 2 * i2 + 1, 2 * i1 + 1] ] = u[ [2 * i3, 2 * i2 + 1, 2 * i1 + 1] ]
                                              + 0.25 * ( z1[i1] + z1[i1 + 1]);
                }

                for i1 in 0..(mm1 - 1) {
                    u[ [2 * i3 + 1, 2 * i2, 2 * i1] ] = u[ [2 * i3 + 1, 2 * i2, 2 * i1] ]
                                            + 0.5 * z2[i1];
                    u[ [2 * i3 + 1, 2 * i2, 2 * i1 + 1] ] = u[ [2 * i3 + 1, 2 * i2, 2 * i1 + 1]]
                                              + 0.25 * (z2[i1] + z2[i1 + 1]);
                }

                for i1 in 0..(mm1-1) {
                    u[ [2*i3+1,2*i2+1,2*i1] ] = u[ [2*i3+1,2*i2+1,2*i1] ]
                                              + 0.25 * z3[i1];
                    u[ [2*i3+1,2*i2+1,2*i1+1] ] = u[ [2*i3+1,2*i2+1,2*i1+1] ]
                                                + 0.125*(z3[i1] + z3[i1+1]);
                }
            }
        }
    } else{
        if n1 == 3{
            d1 = 2;
            t1 = 1;
        }else{
            d1 = 1;
            t1 = 0;
        }
        if n2 == 3{
            d2 = 2;
            t2 = 1;
        }else{
            d2 = 1;
            t2 = 0;
        }
        if n3 == 3{
            d3 = 2;
            t3 = 1;
        }else{
            d3 = 1;
            t3 = 0;
        }
        for i3 in d3..=mm3-1{
            for i2 in d2..=mm2-1{
                for i1 in d1..=mm1-1{
                    u[ [2*i3-d3-1,2*i2-d2-1,2*i1-d1-1] ] = 
                        u[ [2*i3-d3-1,2*i2-d2-1,2*i1-d1-1] ]
                       +z[[i3-1,i2-1,i1-1]];
                }
                for i1 in 1..=mm1-1{
                    u[ [2*i3-d3-1,2*i2-d2-1,2*i1-t1-1] ] =
                        u[ [2*i3-d3-1,2*i2-d2-1,2*i1-t1-1] ]
                       +0.5*(z[ [i3-1,i2-1,i1] ] + z[ [i3-1,i2-1,i1-1] ]); 
                }
            }
            for i2 in 1..=mm2-1{
                for i1 in d1..=mm1-1{
                    u[ [2*i3-d3-1,2*i2-t2-1,2*i1-d1-1] ] =
                        u[ [2*i3-d3-1,2*i2-t2-1,2*i1-d1-1] ]
                        +0.5*(z[ [i3-1,i2,i1-1] ] + z[ [i3-1,i2-1,i1-1] ]);
                }
                for i1 in 1..=mm1-1{
                    u[ [2*i3-d3-1,2*i2-t2-1,2*i1-t1-1] ] =
                        u[ [2*i3-d3-1,2*i2-t2-1,2*i1-t1-1] ]
                        +0.25*(z[ [i3-1,i2,i1] ] + z[ [i3-1,i2-1,i1] ]
                              +z[ [i3-1,i2,i1-1] ] + z[ [i3-1,i2-1,i1-1] ]);
                }
            }
        }
        for i3 in 1..=mm3-1{
            for i2 in d2..=mm2-1{
                for i1 in d1..mm1-1{
                    u[ [2*i3-t3-1,2*i2-d2-1,2*i1-d1-1] ] = 
                        u[ [2*i3-t3-1,2*i2-d2-1,2*i1-d1-1] ]
                        +0.5*(z[[i3,i2-1,i1-1]] + z[ [i3-1,i2-1,i1-1] ]);
                }
                for i1 in 1..mm1-1{
                    u[ [2*i3-t3-1,2*i2-d2-1,2*i1-t1-1] ] =
                        u[ [2*i3-t3-1,2*i2-d2-1,2*i1-t1-1] ]
                        +0.25*(z[[i3,i2-1,i1]]+z[[i3,i2-1,i1-1]] 
                               + z[ [i3-1,i2-1,i1] ] + z[ [i3-1,i2-1,i1] ]
                               + z[ [i3-1,i2-1,i1-1]]);
                }
            }
            for i2 in 1..=mm2-1{
                for i1 in d1..mm1-1{
                    u[ [2*i2-t3-1,2*i2-t2-1,2*i1-d1-1] ] = 
                        u[ [2*i2-t3-1,2*i2-t2-1,2*i1-d1-1] ]
                        +0.25*(z[[ i3,i2,i1-1]] + z[ [i3,i2-1,i1-1] ]
                               + z[ [i3-1,i2,i1-1] ] + z[ [i3-1,i2-1,i1-1] ]);
                }
                for i1 in 1..mm1-1{
                    u[ [2*i3-t3-1,2*i2-t2-1,2*i1-t1-1] ] = 
                        u[ [2*i3-t3-1,2*i2-t2-1,2*i1-t1-1] ]
                        +0.125*(z[ [i3,i2,i1] ] + z[ [i3,i2-1,i1] ]
                               +z[ [i3,i2,i1-1 ] ] + z[ [i3,i2-1,i1-1] ]
                               +z[ [i3-1,i2,i1 ] ] + z[ [i3-1,i2-1,i1] ]
                               +z[ [i3-1,i2,i1-1] ] + z[ [i3-1,i2-1,i1-1]]);
                }
            }
        }
    }
}

const fn ilog2(i:usize) -> i32 {
    let mut log2: i32 = 0;
    let mut exp2 = 1;

    if i <= 0 {
        return -1;
    }

    while log2 < 20{
        if exp2 == i {
            return log2;
        }
        
        exp2 = exp2 *2;
        log2 +=1;
    }

    return -1
}

fn power(a: f64,n:usize) -> f64 {
    let mut aj: f64;
    let mut nj: usize;
    let mut rdummy:f64;
    let mut power: f64;

    power = 1.0;
    nj = n;
    aj = a;
    
    while nj != 0 {
        if (nj % 2) == 1 {
            rdummy = randdp::randlc(&mut power, aj);
        }
        
        let x = aj;
        
        rdummy = randdp::randlc(&mut aj,x);
        
        nj = nj/2;
    }

    return power
}

fn comm3(u_ptr: &mut Array1<f64>,n1: usize,n2: usize,n3: usize, kk:usize) {
    //casting to indexing as 3D matrix
    let mut u = u_ptr.slice_mut(s![0..n1*n2*n3]).into_shape((n1,n2,n3)).unwrap();

    for i3 in 1..n3-1 {
        for i2 in 1..n2-1 {
            u[ [i3, i2, 0] ] = u[ [i3, i2, n1-2] ];
            u[ [i3, i2, n1-1] ] = u[ [i3 , i2, 1] ];
        }

        for i1 in 0..n1{
            u[ [i3, 0, i1] ] = u[ [i3, n2-2, i1]];
            u[ [i3, n2-1, i1]] = u[ [i3, 1, i1] ];
        }
    }

    for i2 in 0..n2 {
        for i1 in 0..n1 {
            u[ [0, i2, i1] ] = u[ [n3-2, i2, i1] ];
            u[ [n3-1, i2, i1] ] = u[ [1, i2, i1] ];
        }
    }
}

fn viewd_comm3(u_ptr: &mut ArrayViewMut<'_, f64, Dim<[usize; 1]>>,n1: usize,n2: usize,n3: usize, kk:usize) {
    //casting to indexing as 3D matrix
    let mut u = u_ptr.slice_mut(s![0..n1*n2*n3]).into_shape((n1,n2,n3)).unwrap();
    for i3 in 1..n3-1{
        for i2 in 1..n2-1{
            u[ [i3,i2,0] ] = u[ [i3,i2,n1-2] ];
            u[ [i3,i2,n1-1] ] = u[ [i3,i2,1] ];
        }
    }
    for i3 in 1..n3-1{
        for i1 in 0..n1{
            u[ [i3,0,i1] ] = u[ [i3,n2-2,i1] ];
            u[ [i3,n2-1,i1] ] = u[ [i3,1,i1] ];
        }
    }
    for i2 in 0..n2{
        for i1 in 0..n1{
            u[ [0,i2,i1] ] = u[ [n3-2,i2,i1] ];
            u[ [n3-1,i2,i1] ] = u[ [1,i2,i1] ];
        }
    }
    let mut sum = 0.0;
    for i3 in 0..n3-1{
        for i2 in 0..n2-1{
            for i1 in 0..n1-1{
                sum += u[[i3,i2,i1]];
            }
        }
    }
}

fn bubble(ten: &mut Array2<f64>, j1: &mut Array2<usize>, j2: &mut Array2<usize>, j3: &mut Array2<usize>, m: usize,ind: usize){

    let mut temp: f64;
    let mut j_temp: usize;
    if ind == 1 {
        for i in 0..m-1{
            if ten[ [ind,i] ] > ten[ [ind,i+1] ]{
                temp = ten[ [ind,i+1] ];
                ten[ [ind, i+1] ] = ten[ [ind, i] ];
                ten[ [ind, i] ] = temp;
                
                j_temp = j1[ [ind,i+1] ];
                j1[ [ind, i+1]] = j1[ [ind,i] ];
                j1[ [ind, i] ] = j_temp;
                
                j_temp = j2[ [ind,i+1] ];
                j2[ [ind, i+1]] = j2[ [ind,i] ];
                j2[ [ind, i] ] = j_temp;

                j_temp = j3[ [ind,i+1] ];
                j3[ [ind, i+1]] = j3[ [ind,i] ];
                j3[ [ind, i] ] = j_temp;
            }
            else {
                return;
            }
        }
    }
    else {
        for i in 0..m-1{
            if ten[ [ind,i] ] < ten[ [ind,i+1] ]{
                temp = ten[ [ind,i+1] ];
                ten[ [ind, i+1] ] = ten[ [ind, i] ];
                ten[ [ind, i] ] = temp;
                
                j_temp = j1[ [ind,i+1] ];
                j1[ [ind, i+1]] = j1[ [ind,i] ];
                j1[ [ind, i] ] = j_temp;
                
                j_temp = j2[ [ind,i+1] ];
                j2[ [ind, i+1]] = j2[ [ind,i] ];
                j2[ [ind, i] ] = j_temp;

                j_temp = j3[ [ind,i+1] ];
                j3[ [ind, i+1]] = j3[ [ind,i] ];
                j3[ [ind, i] ] = j_temp;
            }
            else {
                return;
            }
        }
    }
}

fn zran3(z_ptr: &mut Array1<f64>,n1: usize,n2: usize, n3:usize, nx: usize, ny: usize, k: usize) {
    //casting to indexing as 3D matrix
    let mut z = z_ptr.slice_mut(s![0..n1*n2*n3]).into_shape((n1,n2,n3)).unwrap();

    let (mut i0, m0, m1): (usize,usize,usize);
    let (mut i1, i2, i3,d1, e1, e2, e3): (usize,usize,usize,usize,usize,usize,usize);
    let ( mut xx, mut x0, mut x1, a1, a2, ai):(f64,f64,f64,f64,f64,f64);
    let mut ten = Array2::<f64>::zeros((2,MM));
    let mut best: f64;
    let mut i: usize;
    let mut j1 = Array2::<usize>::zeros((2,MM));
    let mut j2 = Array2::<usize>::zeros((2,MM));
    let mut j3 = Array2::<usize>::zeros((2,MM));
    let mut jg = Array3::<usize>::zeros((2,MM,4));
    
    a1 = power(A, nx);
    a2 = power(A, nx*ny);
    
    unsafe{
        i = is1-2+nx*(is2-2+ny*(is3-2));
    }
    ai = power(A, i);
    unsafe{
        d1 = ie1 - is1 + 1;
        e1 = ie1 - is1 + 2;
        e2 = ie2 - is2 + 2;
        e3 = ie3 - is3 + 2;
    };
    x0 = X;
    randdp::randlc(&mut x0, ai);
    for i3 in 1..e3{
        x1 = x0;
        for i2 in 1..e2{
            xx = x1;
            let mut axis_slice = z.slice_mut(s![i3,i2,..]);
            let mut place_holder = vec![0.0;n3 ];
            vranlc(d1 as i32, &mut xx, A, &mut place_holder[1..]);
            let mut new_array = Array::from_vec(place_holder);
            axis_slice.assign(&new_array);
            randdp::randlc(&mut x1, a1);
        }
        randdp::randlc(&mut x0,a2);
    }
    
    for i in 0..MM {
        ten[ [1,i] ] = 0.0;
        j1 [ [1,i] ] = 0;
        j2 [ [1,i] ] = 0;
        j3 [ [1,i] ] = 0;
        ten [ [0, i]] = 1.0;
        j1 [ [0, i] ] = 0;
        j2 [ [0, i] ] = 0;
        j3 [ [0, i] ] = 0;
    }
    
    for i3 in 1..n3-1{
        for i2 in 1..n2-1{
            for i1 in 1..n1-1{
                if z[ [i3, i2, i1] ] > ten[[1,0]] {
                    ten[ [1, 0] ] = z[ [i3, i2, i1] ];
                    j1[ [1, 0] ] = i1 ;
                    j2[ [1, 0] ] = i2 ;
                    j3[ [1, 0] ] = i3 ;
                    bubble(&mut ten, &mut j1, &mut j2, &mut j3, MM, 1);
                }
                if z[ [i3, i2, i1] ] < ten[ [0, 0]] {
                    ten[ [0, 0] ] = z[ [i3,i2, i1]];
                    j1[ [0, 0] ] = i1 ;
                    j2[ [0, 0] ] = i2 ;
                    j3[ [0, 0] ] = i3 ;
                    bubble(&mut ten, &mut j1, &mut j2, &mut j3, MM, 0);
                }
            }
        }
    }

    i1 = MM;
    i0 = MM;

    for i in (0..=MM-1).rev() {
        best = 0.0;

        if best < ten[ [1, i1-1]]{
            jg[ [1, i, 0] ] = 0;
            unsafe {
                jg[ [1, i, 1] ] = is1 - 2 + j1[ [1,i1-1] ];
                jg[ [1, i, 2] ] = is2 - 2 + j2[ [1,i1-1] ];
                jg[ [1, i, 3] ] = is3 - 2 + j3[ [1,i1-1] ]; 
            };
            i1 = i1-1;
        }else {
            jg[ [1, i, 0] ] = 0;
            jg[ [1, i, 1] ] = 0;
            jg[ [1, i, 2] ] = 0;
            jg[ [1, i, 3] ] = 0;
        }
        best = 1.0;
        if best > ten[ [0, i0-1]] {
            jg[ [0, i, 0] ] = 0;
            unsafe {
                jg[ [0, i, 1] ] = is1 - 2 + j1[ [0,i0-1] ];
                jg[ [0, i, 2] ] = is2 - 2 + j2[ [0,i0-1] ];
                jg[ [0, i, 3] ] = is3 - 2 + j3[ [0,i0-1] ];
            };
             
            i0 = i0-1;

        } else {
            jg[ [0, i, 0] ] = 0;
            jg[ [0, i, 1] ] = 0;
            jg[ [0, i, 2] ] = 0;
            jg[ [0, i, 3] ] = 0;
        }
    }

    m1 = 0;
    m0 = 0;

    for i3 in 0..n3 {
        for i2 in 0..n2 {
            for i1 in 0..n1 {
                z[ [i3, i2, i1]] = 0.0;
            }
        }
    }

    for i in (m0 ..= MM - 1).rev() {
        z[ [ jg[ [0, i, 3] ] , jg[ [0, i, 2] ]  , jg[ [0,i,1] ]  ] ] = -1.0;
    }

    for i in (m1 ..= MM - 1).rev() {
        z[ [jg[[1, i, 3]] , jg[[1, i, 2]] , jg[[1,i,1]] ]] = 1.0;
    }

    comm3(z_ptr, n1, n2, n3, k);
}

fn norm2u3(r_ptr: &mut Array1<f64>,n1:usize,n2:usize,n3:usize,rnm2: &mut f64,rnmu: &mut f64,nx:usize,ny:usize,nz:usize) {
    let mut r = r_ptr.slice_mut(s![0..n1 * n2 * n3]).into_shape((n1, n2, n3)).unwrap();
    let (mut s, mut a, rnmu_local): (f64, f64, f64);
    let mut dn: f64;
    dn = 1.0 * (nx as f64) * (ny as f64) * (nz as f64);
    s = 0.0;
    *rnmu = 0.0;
    for i3 in 1..(n3 - 1) {
        for i2 in 1..(n2 - 1) {
            for i1 in 1..(n1 - 1) {
                s = s + r[ [i3, i2, i1] ] * r[ [i3, i2, i1] ];
                a = libm::fabs(r[[i3, i2, i1]]);
                //a = fabs(r[[i3, i2, i1]]);

                if a > *rnmu {
                    *rnmu = a;
                }
            }
        }
    }

    *rnm2 = libm::sqrt(s / dn);
    //*rnm2 = sqrt(s / dn);
}

fn resid(u_ptr: &mut ArrayViewMut<'_, f64, Dim<[usize; 1]>>, v_ptr: &mut ArrayViewMut<'_, f64, Dim<[usize; 1]>>, r_ptr: &mut ArrayViewMut<'_, f64, Dim<[usize; 1]>>, n1: usize, n2: usize,n3: usize, a: &[f64],k:usize){
    let mut u = u_ptr.slice_mut(s![0..n1*n2*n3]).into_shape((n1,n2,n3)).unwrap();
    let mut v = v_ptr.slice_mut(s![0..n1*n2*n3]).into_shape((n1,n2,n3)).unwrap();
    let mut r = r_ptr.slice_mut(s![0..n1*n2*n3]).into_shape((n1,n2,n3)).unwrap();
    let mut u1: [f64;M] = [0.0;M];
    let mut u2: [f64;M] = [0.0;M];

    for i3 in 1..n3-1 {
        for i2 in 1..n2-1 {
            for i1 in 0..n1 {
                u1[i1] = u[ [i3, i2-1, i1] ] + u[ [i3, i2+1, i1] ]
                       + u[ [i3-1, i2, i1] ] + u[ [i3+1, i2, i1] ];
                u2[i1] = u[ [i3-1, i2-1, i1] ] + u[ [i3-1, i2+1, i1] ]
                       + u[ [i3+1, i2-1, i1] ] + u[ [i3+1, i2+1, i1] ];
            }
            for i1 in 1..n1-1 {
                r[ [i3, i2, i1] ] = v[ [i3, i2, i1] ]
                                  - a[0] * u[ [i3, i2, i1] ]
                                  - a[2] * (u2[i1] + u1[i1-1] + u1[i1+1])
                                  - a[3] * (u2[i1-1] + u2[i1+1]);
            }
        }
    }
	// --------------------------------------------------------------------
	// exchange boundary data
	// --------------------------------------------------------------------
    viewd_comm3(r_ptr,n1,n2,n3,k);
}

fn resid_two(u_ptr: &mut ArrayViewMut<'_, f64, Dim<[usize; 1]>>, v_ptr: &mut Array1<f64>, r_ptr: &mut ArrayViewMut<'_, f64, Dim<[usize; 1]>>, n1: usize, n2: usize,n3: usize, a: &[f64],k:usize){
    let mut u = u_ptr.slice_mut(s![0..n1*n2*n3]).into_shape((n1,n2,n3)).unwrap();
    let mut v = v_ptr.slice_mut(s![0..n1*n2*n3]).into_shape((n1,n2,n3)).unwrap();
    let mut r = r_ptr.slice_mut(s![0..n1*n2*n3]).into_shape((n1,n2,n3)).unwrap();
    let mut u1: [f64;M] = [0.0;M];
    let mut u2: [f64;M] = [0.0;M];

    for i3 in 1..n3-1 {
        for i2 in 1..n2-1 {
            for i1 in 0..n1 {
                u1[i1] = u[ [i3, i2-1, i1] ] + u[ [i3, i2+1, i1] ]
                       + u[ [i3-1, i2, i1] ] + u[ [i3+1, i2, i1] ];
                u2[i1] = u[ [i3-1, i2-1, i1] ] + u[ [i3-1, i2+1, i1] ]
                       + u[ [i3+1, i2-1, i1] ] + u[ [i3+1, i2+1, i1] ];
            }
            for i1 in 1..n1-1 {
                r[ [i3, i2, i1] ] = v[ [i3, i2, i1] ]
                                  - a[0] * u[ [i3, i2, i1] ]
                                  - a[2] * (u2[i1] + u1[i1-1] + u1[i1+1])
                                  - a[3] * (u2[i1-1] + u2[i1+1]);
            }
        }
    }
	// --------------------------------------------------------------------
	// exchange boundary data
	// --------------------------------------------------------------------
    viewd_comm3(r_ptr,n1,n2,n3,k);
}