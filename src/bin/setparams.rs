use std::env;
use std::fs;
use std::fs::File;
use std::io::Write as _;
use chrono::Local;

const BIN_PATH: &str = "./src/bin";
const TEMPLATE_PATH: &str = "./src/templates";
const CG_TEMPLATEPATH: &str = "./src/templates/cg.rs";
const EP_TEMPLATEPATH: &str = "./src/templates/ep.rs";

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut kernel = &args[1];
    let mut class_npb = &args[2];
    let binding = kernel.to_lowercase();
    kernel = &binding;
    let binding2 = class_npb.to_lowercase();
    class_npb = &binding2;

    if kernel == "ep" {
        write_ep_info(class_npb.as_str());
    } else if kernel == "ep-pp" {
        write_ep_pp_info(class_npb.as_str());
    } else if kernel == "cg" {
        write_cg_info(class_npb.as_str());
    } else if kernel == "mg" {
        write_mg_info(class_npb.as_str());
    } else if kernel == "ft" {
        write_ft_info(class_npb.as_str());
    }
}

fn write_ep_info(class_npb: &str) {
    let mut binding = fs::read_to_string(&EP_TEMPLATEPATH).expect("File");
    let mut contents: &str = binding.as_mut_str();

    let m: u32 = match class_npb {
        "s"=>24,
        "w"=>25,
        "a"=>28,
        "b"=>30,
        "c"=>32,
        "d"=>36,
        "e"=>40,
        _=>24
    };

    let compile_time = Local::now().to_rfc3339();

    binding = contents.replace("%% CLASS_NPB %%", format!("\"{}\"", class_npb).as_str());
    contents = binding.as_mut_str();
    binding = contents.replace("%% M %%", format!("{}", m).as_str());
    contents = binding.as_mut_str();
    binding = contents.replace("%% COMPILE_TIME %%", format!("\"{}\"", compile_time).as_str());
    contents = binding.as_mut_str();

    let mut bin_file = File::create(format!("{}/ep-{}.rs", &BIN_PATH, class_npb)).unwrap();
    let _ = bin_file.write_all(&contents.as_bytes());
}

fn write_ep_pp_info(class_npb: &str) {
    let mut binding = fs::read_to_string(&format!("{}/ep-pp.rs", TEMPLATE_PATH)).expect("File");
    let mut contents: &str = binding.as_mut_str();

    let m: u32 = match class_npb {
        "s"=>24,
        "w"=>25,
        "a"=>28,
        "b"=>30,
        "c"=>32,
        "d"=>36,
        "e"=>40,
        _=>24
    };

    let compile_time = Local::now().to_rfc3339();

    binding = contents.replace("%% CLASS_NPB %%", format!("\"{}\"", class_npb).as_str());
    contents = binding.as_mut_str();
    binding = contents.replace("%% M %%", format!("{}", m).as_str());
    contents = binding.as_mut_str();
    binding = contents.replace("%% COMPILE_TIME %%", format!("\"{}\"", compile_time).as_str());
    contents = binding.as_mut_str();

    let mut bin_file = File::create(format!("{}/ep-pp-{}.rs", &BIN_PATH, class_npb)).unwrap();
    let _ = bin_file.write_all(&contents.as_bytes());
}

fn write_ft_info(class_npb: &str) {
    let mut binding = fs::read_to_string(&format!("{}/ft.rs", TEMPLATE_PATH)).expect("File");
    let mut contents: &str = binding.as_mut_str();

    let nx = match class_npb {
		"s" => 64,
		"w" => 128,
		"a" => 256,
		"b" => 512,
		"c" => 512,
		"d" => 2048,
		"e" => 4096,
		_   => 64
	};
	let ny = match class_npb {
		"s" => 64,
		"w" => 128,
		"a" => 256,
		"b" => 256,
		"c" => 512,
		"d" => 1024,
		"e" => 2048,
		_   => 64
	};
	let nz = match class_npb {
		"s" => 64,
		"w" => 32,
		"a" => 128,
		"b" => 256,
		"c" => 512,
		"d" => 1024,
		"e" => 2048,
		_   => 64
	};
	let niter = match class_npb {
		"s" => 6,
		"w" => 6,
		"a" => 6,
		"b" => 20,
		"c" => 20,
		"d" => 25,
		"e" => 25,
		_   => 6
	};
    
    let mut maxdim = nx;
    if ny > maxdim {
        maxdim = ny;
    }
    if nz > maxdim {
        maxdim = nz;
    }

    let compile_time = Local::now().to_rfc3339();

    binding = contents.replace("%% CLASS_NPB %%", format!("\"{}\"", class_npb).as_str());
    contents = binding.as_mut_str();
    binding = contents.replace("%% NX %%", format!("{}", nx).as_str());
    contents = binding.as_mut_str();
    binding = contents.replace("%% NY %%", format!("{}", ny).as_str());
    contents = binding.as_mut_str();
    binding = contents.replace("%% NZ %%", format!("{}", nz).as_str());
    contents = binding.as_mut_str();
    binding = contents.replace("%% NITER %%", format!("{}", niter).as_str());
    contents = binding.as_mut_str();
    binding = contents.replace("%% MAXDIM %%", format!("{}", maxdim).as_str());
    contents = binding.as_mut_str();
    binding = contents.replace("%% COMPILE_TIME %%", format!("\"{}\"", compile_time).as_str());
    contents = binding.as_mut_str();

    let mut bin_file = File::create(format!("{}/ft-{}.rs", &BIN_PATH, class_npb)).unwrap();
    let _ = bin_file.write_all(&contents.as_bytes());
}

fn write_cg_info(class_npb: &str) {
    let mut binding = fs::read_to_string(&format!("{}/cg.rs", TEMPLATE_PATH)).expect("File");
    let mut contents: &str = binding.as_mut_str();

    let na = match class_npb {
		"s" => 1400,
		"w" => 7000,
		"a" => 14000,
		"b" => 75000,
		"c" => 150000,
		"d" => 1500000,
		"e" => 9000000,
		_   => 1400
	};
	let nonzer = match class_npb {
		"s" => 7,
		"w" => 8,
		"a" => 11,
		"b" => 13,
		"c" => 15,
		"d" => 21,
		"e" => 26,
		_   => 7
	};
	let niter = match class_npb {
		"s" => "15",
		"w" => "15",
		"a" => "15",
		"b" => "75",
		"c" => "75",
		"d" => "100",
		"e" => "100",
		_   => "15"
	};
	let shift = match class_npb {
		"s" => "10.0",
		"w" => "12.0",
		"a" => "20.0",
		"b" => "60.0",
		"c" => "110.0",
		"d" => "500.0",
		"e" => "1500.0",
		_   => "10.0"
	};

    let nz: i32 = na * (nonzer + 1) * (nonzer + 1);
    let naz: i32 = na * (nonzer + 1);

    let compile_time = Local::now().to_rfc3339();

    binding = contents.replace("%% CLASS_NPB %%", format!("\"{}\"", class_npb).as_str());
    contents = binding.as_mut_str();
    binding = contents.replace("%% NA %%", format!("{}", na).as_str());
    contents = binding.as_mut_str();
    binding = contents.replace("%% NONZER %%", format!("{}", nonzer).as_str());
    contents = binding.as_mut_str();
    binding = contents.replace("%% NITER %%", niter);
    contents = binding.as_mut_str();
    binding = contents.replace("%% SHIFT %%", shift);
    contents = binding.as_mut_str();
    binding = contents.replace("%% COMPILE_TIME %%", format!("\"{}\"", compile_time).as_str());
    contents = binding.as_mut_str();

    let mut bin_file = File::create(format!("{}/cg-{}.rs", &BIN_PATH, class_npb)).unwrap();
    let _ = bin_file.write_all(&contents.as_bytes());
}

fn write_mg_info(class_npb: &str) {
    let mut binding = fs::read_to_string(&format!("{}/mg.rs", TEMPLATE_PATH)).expect("File");
    let mut contents: &str = binding.as_mut_str();

    let problem_size = match class_npb {
        "s" => "32",
        "w" => "128",
        "a" => "256",
        "b" => "256",
        "c" => "512",
        "d" => "1024",
        "e" => "2048",
        _   => "32"
    };

    let nit = match class_npb {
        "s" => "4",
        "w" => "4",
        "a" => "4",
        "b" => "20",
        "c" => "20",
        "d" => "50",
        "e" => "50",
        _   => "4"
    };

    let compile_time = Local::now().to_rfc3339();

    binding = contents.replace("%% PROBLEM_SIZE %%", problem_size);
    contents = binding.as_mut_str();
    binding = contents.replace("%% NIT %%", nit);
    contents = binding.as_mut_str();
    binding = contents.replace("%% COMPILE_TIME %%", format!("\"{}\"", compile_time).as_str());
    contents = binding.as_mut_str();

    let mut bin_file = File::create(format!("{}/mg-{}.rs", &BIN_PATH, class_npb)).unwrap();
    //write!(bin_file, "{}", &contents);
    let _ = bin_file.write_all(&contents.as_bytes());
}