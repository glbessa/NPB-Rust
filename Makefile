# REMOVE RUST WARNINGS
#export RUSTFLAGS="-Awarnings"

# CALLING THIS MAKE FILE
# make <target> KERNEL=<kernel> CLASS=<class> NUM_THREADS=<num_threads>

# DEFAULT KERNEL
KERNEL=
# DEFAULT CLASS
CLASS=
# DEFAULT NUM_THREADS
NUM_THREADS=1
# DONT OVERWRITE THIS
PARAMS_COMMAND=cargo +nightly run --bin setparams --release -- $(KERNEL) $(CLASS) $(NUM_THREADS)

COMP_COMMAND=cargo +nightly rustc --bin $(KERNEL)-$(CLASS) --release

EXEC_COMMAND=cargo +nightly run --bin $(KERNEL)-$(CLASS) --release -- $(NUM_THREADS)

compile:
	$(PARAMS_COMMAND)
	$(COMP_COMMAND)

run:
	$(EXEC_COMMAND)