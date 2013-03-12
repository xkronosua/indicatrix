#!/bin/bash
#
#gnuplot -e "set term png; set output \"$1.png\"; set logscale y; set grid; plot \"$1\" lc rgb '#3A5FCD' lw 1 pt 3 ps 1"
#for distrib in "uniform" "correlated" "clustered"
#do
gnuplot << EOF
    # gnuplot preamble omitted for brevity
set terminal png
set output "$1.png"
#    set title "$distrib distribution, eps = $eps, #points = $points"

plot "$1" #using 1:(\$2/$points) title "exact NN"
EOF

$done