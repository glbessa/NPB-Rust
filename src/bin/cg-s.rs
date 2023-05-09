use std::env;
use std::time::Instant;
use chrono::Local;

//use common::rust_timers;
use common::print_results;
use common::randdp;

const CLASS: &str = "s";
const NA: i32 = 1400;
const NONZER: i32 = 7;
const NITER: i32 = 15;
const SHIFT: f64 = 10.0;
const NZ: i32 = NA * (NONZER + 1) * (NONZER + 1);
const NAZ: i32 = NA * (NONZER + 1);

const NPBVERSION: &str = "4.1.2";
const COMPILETIME: &str = "2023-04-30T16:36:23.948999684-03:00"; // date
const COMPILERVERSION: &str = "rustc 1.70.0-nightly";
const LIBVERSION: &str = "1";
const CS1: &str = "";
const CS2: &str = "";
const CS3: &str = "";
const CS4: &str = "";
const CS5: &str = "";
const CS6: &str = "";
const CS7: &str = "";

const RCOND: f64 = 0.1;

fn main() {
	let init_timer = Instant::now();
	
	let mut colidx: Vec<i32> = vec![0; NZ.try_into().unwrap()];
	let mut rowstr: Vec<i32> = vec![0; (NA + 1).try_into().unwrap()];
	let mut iv: Vec<i32> = vec![0; NA.try_into().unwrap()];
	let mut arow: Vec<i32> = vec![0; NA.try_into().unwrap()];
	let mut acol: Vec<i32> = vec![0; NAZ.try_into().unwrap()];
	let mut aelt: Vec<f64> = vec![0.0; NAZ.try_into().unwrap()];
	let mut a: Vec<f64> = vec![0.0; NZ.try_into().unwrap()];
	//let mut x: Vec<f64> = vec![0.0; (NA+2).try_into().unwrap()];
	let mut x: Vec<f64> = vec![1.0; (NA+2).try_into().unwrap()];
	let mut z: Vec<f64> = vec![0.0; (NA+2).try_into().unwrap()];
	let mut p: Vec<f64> = vec![0.0; (NA+2).try_into().unwrap()];
	let mut q: Vec<f64> = vec![0.0; (NA+2).try_into().unwrap()];
	let mut r: Vec<f64> = vec![0.0; (NA+2).try_into().unwrap()];

	let mut naa: i32 = 0;
	let mut nzz: i32 = 0;
	let mut firstrow: i32 = 0;
	let mut lastrow: i32 = 0;
	let mut firstcol: i32 = 0;
	let mut lastcol: i32 = 0;
	let mut amult: f64 = 0.0;
	let mut tran: f64 = 0.0;
	
	let (mut zeta, mut norm_temp1, mut norm_temp2, mut t, mflops, zeta_verify_value, epsilon, err): (f64,f64,f64,f64,f64,f64,f64,f64);
	let mut rnorm: f64 = 0.0;
	let verified: bool;

	firstrow = 0;
	lastrow = NA - 1;
	firstcol = 0;
	lastcol = NA - 1;

	zeta_verify_value = match CLASS {
		"s"=>8.5971775078648,
		"w"=>10.362595087124,
		"a"=>17.130235054029,
		"b"=>22.712745482631,
		"c"=>28.973605592845,
		"e"=>52.514532105794,
		"e"=>77.522164599383,
		_=>8.5971775078648
	};

	println!("\n\n NAS Parallel Benchmarks 4.1 Serial Rust version - CG Benchmark\n");
	println!(" Size: {}", &NA);
	println!(" Iterations: {}", &NITER);

	naa = NA;
	nzz = NZ;

	tran  = 314159265.0;
	amult = 1220703125.0;
	zeta  = randdp::randlc(&mut tran, amult);
	
	makea(&mut naa, &mut nzz, &mut a, &mut colidx, &mut rowstr, &firstrow, &lastrow, &firstcol, &lastcol, &mut arow, &mut acol, &mut aelt, &mut iv, &mut tran, &amult);

	for j in 0..=(lastcol - firstrow) {
		for k in rowstr[j as usize]..rowstr[(j + 1) as usize] {
			colidx[k as usize] = colidx[k as usize] - firstcol;
		}
	}

	/*
	for i in 0..=NA {
		x[i as usize] = 1.0;
	}
	for j in 0..=(lastcol - firstcol) {
		q[j as usize] = 0.0;
		z[j as usize] = 0.0;
		r[j as usize] = 0.0;
		p[j as usize] = 0.0;
	}
	*/
	zeta = 0.0;

	conj_grad(&mut colidx, &mut rowstr, &mut x, &mut z, &mut a, &mut p, &mut q, &mut r, &mut rnorm, &naa, &lastcol, &firstcol, &lastrow, &firstrow);

	norm_temp1 = 0.0;
	norm_temp2 = 0.0;

	for j in 0..=(lastcol - firstcol) {
		//let j = j as usize;
		norm_temp1 = norm_temp1 + x[j as usize] * z[j as usize];
		norm_temp2 = norm_temp2 + z[j as usize] * z[j as usize];
	}
	norm_temp2 = 1.0 / norm_temp2.sqrt();

	for j in 0..=(lastcol - firstcol) {
		x[j as usize] = norm_temp2 * z[j as usize];
	}

	for i in 0..=NA {
		x[i as usize] = 1.0;
	}
	zeta = 0.0;

	t = init_timer.elapsed().as_secs_f64();
	println!(" Initialization time = {} seconds", &t);
	let bench_timer = Instant::now();

	// main iteration
	for it in 1..=NITER {
		conj_grad(&mut colidx, &mut rowstr, &mut x, &mut z, &mut a, &mut p, &mut q, &mut r, &mut rnorm, &naa, &lastcol, &firstcol, &lastrow, &firstrow);
		norm_temp1 = 0.0;
		norm_temp2 = 0.0;

		for j in 0..=(lastcol - firstcol) {
			norm_temp1 = norm_temp1 + x[j as usize] * z[j as usize];
			norm_temp2 = norm_temp2 + z[j as usize] * z[j as usize];
		}

		norm_temp2 = 1.0 / norm_temp2.sqrt();

		zeta = SHIFT + 1.0 / norm_temp1;

		if it == 1 {
			println!("\n   iteration           ||r||                 zeta");
		}
		println!("    {}       {}   {}", &it, &rnorm, &zeta);
		
		for j in 0..=(lastcol - firstcol) {
			x[j as usize] = norm_temp2 * z[j as usize];
		}
	}

	t = bench_timer.elapsed().as_secs_f64();
	println!(" Benchmark completed");

	epsilon = 0.0000000001;

	err = (zeta - zeta_verify_value).abs() / zeta_verify_value;

	if err <= epsilon {
		verified = true;
		println!(" VERIFICATION SUCCESSFUL");
		println!(" Zeta is    {}", zeta);
		println!(" Error is   {}", err);
	}
	else {
		verified = false;
		println!(" VERIFICATION FAILED");
		println!(" Zeta is    {}", zeta);
		println!(" Error is   {}", err);
	}

	if t != 0.0 {
		mflops = (2.0 * NITER as f64 * NA as f64) * (3.0 + (NONZER as f64 * (NONZER as f64 + 1.0)) + 25.0 * (5.0 + (NONZER as f64 * (NONZER as f64 + 1.0))) + 3.0) / t / 1000000.0;
	}
	else {
		mflops = 0.0;
	}
	
	print_results::rust_print_results("CG", CLASS, NA.try_into().unwrap(), 0, 0, NITER, t, mflops, "          floating point", verified, NPBVERSION, COMPILETIME, COMPILERVERSION, LIBVERSION, "1", CS1, CS2, CS3, CS4, CS5, CS6, CS7);
}

fn conj_grad(colidx: &mut Vec<i32>, rowstr: &mut Vec<i32>, x: &mut Vec<f64>, z: &mut Vec<f64>, a: &mut Vec<f64>, p: &mut Vec<f64>, q: &mut Vec<f64>, r: &mut Vec<f64>, rnorm: &mut f64, naa: &i32, lastcol: &i32, firstcol: &i32, lastrow: &i32, firstrow: &i32) {
	let cgitmax: i32;
	let (mut d, mut sum, mut rho, mut rho0, mut alpha, mut beta): (f64, f64, f64, f64, f64, f64);

	cgitmax = 25;

	rho = 0.0;

	for j in 0..=(*naa) {
		q[j as usize] = 0.0;
		z[j as usize] = 0.0;
		r[j as usize] = x[j as usize];
		p[j as usize] = r[j as usize];
	}

	for j in 0..=(*lastcol - *firstcol) {
		rho = rho + r[j as usize] * r[j as usize];
	}

	for _cgit in 1..=cgitmax {
		for j in 0..=(*lastrow - *firstrow) {
			sum = 0.0;

			for k in rowstr[j as usize]..rowstr[(j + 1) as usize] {
				sum += a[k as usize] * p[colidx[k as usize] as usize];
			}
			q[j as usize] = sum;
		}

		d = 0.0;
		for j in 0..=(*lastcol - *firstcol) {
			d = d + p[j as usize] * q[j as usize];
		}

		alpha = rho / d;

		rho0 = rho;

		rho = 0.0;

		for j in 0..=(*lastcol - *firstcol) {
			z[j as usize] = z[j as usize] + alpha * p[j as usize];
			r[j as usize] = r[j as usize] - alpha * q[j as usize];
		}

		for j in 0..=(*lastcol - *firstcol) {
			let j = j as usize;
			rho = rho + r[j] * r[j];
		}

		beta = rho / rho0;

		for j in 0..=(*lastcol - *firstcol) {
			p[j as usize] = r[j as usize] + beta * p[j as usize];
		}
	}

	sum = 0.0;
	for j in 1..=(*lastrow - *firstrow) {
		d = 0.0;
		
		//for k in rowstr[j as usize]..rowstr[(j + 1) as usize] - 1 {
		for k in rowstr[j as usize]..rowstr[(j + 1) as usize] {
			d = d + a[k as usize] * z[colidx[k as usize] as usize];
		}
		r[j as usize] = d;
	}

	for j in 0..=(*lastcol - *firstcol) {
		d = x[j as usize] - r[j as usize];
		sum = sum + d * d;
	}

	*rnorm = sum.sqrt();
}
	
fn icnvrt(x: &f64, ipwr2: &i32) -> i32 {
	return ((*ipwr2 as f64) * (*x)).trunc() as i32;
}

fn makea(n: &mut i32, nz: &mut i32, a: &mut Vec<f64>, colidx: &mut Vec<i32>, rowstr: &mut Vec<i32>, firstrow: &i32, lastrow: &i32, firstcol: &i32, lastcol: &i32, arow: &mut Vec<i32>, acol: &mut Vec<i32>, aelt: &mut Vec<f64>, iv: &mut Vec<i32>, tran: &mut f64, amult: &f64 ) {
	let (mut nzv, mut nn1): (i32, i32);
	let mut ivc: Vec<i32> = vec![0; (NONZER + 1).try_into().unwrap()];
	let mut vc: Vec<f64> = vec![0.0; (NONZER + 1).try_into().unwrap()];

	nn1 = 1;

	loop {
		nn1 = 2 * nn1;

		if nn1 >= *n {
			break;
		}
	}

	for iouter in 0..*n {
		nzv = NONZER;
		sprnvc(n, &mut nzv, &nn1, &mut vc, &mut ivc, tran, &amult);
		vecset(&mut vc, &mut ivc, &mut nzv, iouter + 1, 0.5);
		arow[iouter as usize] = nzv;
		for ivelt in 0..nzv {
			acol[(iouter * (NONZER + 1) + ivelt) as usize] = ivc[ivelt as usize] - 1;
			aelt[(iouter * (NONZER + 1) + ivelt) as usize] = vc[ivelt as usize];
		}
	}

	sparse(a, colidx, rowstr, n, nz, arow, acol, aelt, firstrow, lastrow, iv);
}

fn sparse(a: &mut Vec<f64>, colidx: &mut Vec<i32>, rowstr: &mut Vec<i32>, n: &mut i32, nz: &mut i32, arow: &mut Vec<i32>, acol: &mut Vec<i32>, aelt: &mut Vec<f64>, firstrow: &i32, lastrow: &i32, nzloc: &mut Vec<i32>) {
	let nrows: i32;

	let (mut local_j, mut j1, mut j2, mut kk, mut nza, mut jcol): (i32, i32, i32, i32, i32, i32);
	let mut last_k: i32 = 0;
	let (mut size, mut scale, ratio, mut va): (f64, f64, f64, f64);
	let mut goto_40: bool;

	nrows = *lastrow - *firstrow + 1;
	
	for j in 0..=nrows {
		rowstr[j as usize] = 0;
	}
	for i in 0..(*n) {
		for nza in 0..arow[i as usize] {
			//j = acol[i as usize][nza as usize] + 1;
			local_j = acol[(i * (NONZER + 1) + nza) as usize] + 1;
			//j = acol[(i * (NONZER + 1) + nza) as usize];
			rowstr[local_j as usize] = rowstr[local_j as usize] + arow[i as usize];
		}
	}
	
	rowstr[0] = 0;
	for j in 1..=nrows {
		rowstr[j as usize] = rowstr[j as usize] + rowstr[(j - 1) as usize];
	}
	nza = rowstr[nrows as usize] - 1;
	
	if nza > *nz {
		println!("Space for matrix elements exceeded in sparse");
		println!("nza, nzmax = {}, {}", &nza, &nz);
		std::process::exit(-1);
	}
	
	for j in 0..nrows {
		for k in rowstr[j as usize]..rowstr[(j + 1) as usize] {
			a[k as usize] = 0.0;
			colidx[k as usize] = -1;
		}
		nzloc[j as usize] = 0;
	}

	size = 1.0;
	ratio = f64::powf(RCOND, 1.0 / (*n as f64));
	for i in 0..(*n) {
		for nza in 0..arow[i as usize] {
			local_j = acol[(i * (NONZER + 1) + nza) as usize];
			//dbg!(local_j);
			
			scale = size * aelt[(i * (NONZER + 1) + nza) as usize];

			for nzrow in 0..arow[i as usize] {
				jcol = acol[(i * (NONZER + 1) + nzrow) as usize];
				va = aelt[(i * (NONZER + 1) + nzrow) as usize] * scale;

				//if (jcol == j) & (j == i) {
				if (jcol == local_j) && (local_j == i) {
					va = va + RCOND - SHIFT;
				}

				goto_40 = false;
				for k in rowstr[local_j as usize]..rowstr[(local_j + 1) as usize] {
					last_k = k;
					//dbg!(local_j);
					if colidx[k as usize] > jcol {
						kk = rowstr[(local_j + 1) as usize] - 2;
						loop {
							if kk < k {
								break;
							}
							if colidx[kk as usize] > -1 {
								a[(kk + 1) as usize] = a[kk as usize];
								colidx[(kk + 1) as usize] = colidx[kk as usize];
							}
							kk -= 1;
						}
						
						colidx[k as usize] = jcol;
						a[k as usize] = 0.0;
						goto_40 = true;
						break;
					}
					else if colidx[k as usize] == -1 {
						colidx[k as usize] = jcol;
						goto_40 = true;
						break;
					}
					else if colidx[k as usize] == jcol {
						nzloc[local_j as usize] = nzloc[local_j as usize] + 1;
						goto_40 = true;
						break;
					}
					//dbg!(local_j);
				}
				if goto_40 == false {
					println!("internal error in sparse: i = {}", &i);
					std::process::exit(-1);
				}
				
				a[last_k as usize] = a[last_k as usize] + va;
			}
		}
		size = size * ratio;
	}
	
	for j in 1..nrows {
		nzloc[j as usize] = nzloc[j as usize] + nzloc[(j - 1) as usize];
	}

	for j in 0..nrows {
		if j > 0 {
			j1 = rowstr[j as usize] - nzloc[(j - 1) as usize];
		}
		else {
			j1 = 0;
		}
		j2 = rowstr[(j + 1) as usize] - nzloc[j as usize];
		nza = rowstr[j as usize];

		for k in j1..j2 {
			a[k as usize] = a[nza as usize];
			colidx[k as usize] = colidx[nza as usize];
			nza = nza + 1;
		}
	}
	for j in 1..=nrows {
		rowstr[j as usize] = rowstr[j as usize] - nzloc[(j - 1) as usize];
	}
	nza = rowstr[nrows as usize] - 1;
}

fn sprnvc(n: &mut i32, nz: &mut i32, nn1: &i32, v: &mut Vec<f64>, iv: &mut Vec<i32>, tran: &mut f64, amult: &f64) {
	let mut i: i32;
	let (mut vecelt, mut vecloc): (f64, f64);
	let mut was_gen: bool = false;
	let mut nzv: i32 = 0;

	// Não pode ser um for loop
	// Verificar se da pra paralelizar
	while nzv < *nz {
		vecelt = randdp::randlc(tran, *amult);
		vecloc = randdp::randlc(tran, *amult);
		i = icnvrt(&vecloc, nn1) + 1;

		if i > *n {
			continue;
		}

		was_gen = false;
		for ii in 0..nzv {
			if iv[ii as usize] == i {
				was_gen = true;
				break;
			}
		}
		if was_gen == true {
			continue;
		}
		v[nzv as usize] = vecelt;
		iv[nzv as usize] = i;
		nzv = nzv + 1;
	}
}

fn vecset(v: &mut Vec<f64>, iv: &mut Vec<i32>, nzv: &mut i32, i: i32, val: f64) {
	let mut set: bool = false;

	for k in 0..*nzv {
		if iv[k as usize] == i {
			v[k as usize] = val;
			set = true;
		}
	}
	if set == false {
		v[*nzv as usize] = val;
		iv[*nzv as usize] = i;
		*nzv = *nzv + 1;
	}
}