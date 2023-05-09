#! /bin/bash

export RUSTFLAGS="-Awarnings"

SAVE_PATH=./results2/rust
KERNEL=$1
CLASS=$2
NUM_THREADS=$3
EXEC_COMMAND="cargo +nightly run --bin $KERNEL-$CLASS --release -- $NUM_THREADS"

mkdir $SAVE_PATH

$EXEC_COMMAND

for i in {1..30}
do
	echo $i
	$EXEC_COMMAND > $SAVE_PATH/$KERNEL.$CLASS.$NUM_THREADS.$i.txt
done
