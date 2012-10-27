#!/bin/bash
#
gnuplot -e "set term png; set output \"$1.png\"; set logscale y; set grid; plot \"$1\" lc rgb '#3A5FCD' lw 1 pt 3 ps 1"
