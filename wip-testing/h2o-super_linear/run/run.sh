#!/bin/bash

for SIZE in 0.1 10; do
  ln -sf /train-${SIZE}m.csv train.csv 
  for CORES in $(cat cores.conf); do
    NCORES=$(echo $CORES | cut -d: -f1)
    LCORES=$(echo $CORES | cut -d: -f2)
    TCORES=$(echo $CORES | cut -d: -f3)
    for TOOL in h2o; do
      for i in {1..2}; do
        RUNTIME=$(taskset -c $LCORES R --slave < $TOOL.R $NCORES | tail -1)
        echo $SIZE:$TOOL:$NCORES:$LCORES:$TCORES:$RUNTIME
      done
    done
  done
done

