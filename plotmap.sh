#!/bin/bash
#
#gnuplot -e "set term png; set output \"$1.png\"; set logscale y; set grid; plot \"$1\" lc rgb '#3A5FCD' lw 1 pt 3 ps 1"
#for distrib in "uniform" "correlated" "clustered"
#do
#gnuplot << EOF
#unset key
#unset sur
#set view map
#set pm3d at b
#set palette positive nops_allcF maxcolors 0 gamma 1.5 gray
#set terminal png  
#set ountput '297_0_20120523_2119_59/297_0_0001.dat.png'
#splot '297_0_20120523_2119_59/297_0_0001.dat' matrix
#EOF

#!/bin/sh
filename=$(basename "$1")
name="${filename%%.*}"
gnuplot -persist << EOF
set view map
unset sur
unset key
unset tics
set cbtics
set pm3d at b
set palette positive nops_allcF maxcolors 0 gamma 0.8 gray
set title "$name"
#set terminal postscript
#set ountput "$1.ps"
set terminal gif 
#color enhanced
set output "$1.jpg"
splot "$1" matrix
EOF
