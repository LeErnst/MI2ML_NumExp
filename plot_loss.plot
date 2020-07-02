set terminal pngcairo size 1024,1024 enhanced font 'Verdana,28'
set output "loss_plot.png"
set title ''
set border linewidth 1.5
set xlabel 'number of epochs'
set ylabel 'loss'
set key inside left top font "1.0" noopaque
set grid ytics lc -1 lw 0.8 lt 0
set grid xtics lc -1 lw 0.8 lt 0
plot "fitnet1_xavier_uniform.data"  using 6:4 with linespoints lt 1 lw 5 pt 2 ps 2 title "Xavier uniform", \
     "fitnet1_kaiming_normal.data"  using 6:4 with linespoints lt 2 lw 5 pt 2 ps 2 title "Kaiming normal", \
