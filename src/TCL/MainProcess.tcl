logfile /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/visualisation.log
set imag_P_col "0.8 0.2 0.8"
set imag_N_col "0.2 0.8 0"
set real_P_col "1 0 0"
set real_N_col "0 1 0"
set dens_col   "0.3 0.32 0.3"
proc load_file { filename mol_id neg_col_id pos_col_id } {
	set mol_name $filename
	mol new $mol_name
	
	mol addrep $mol_id
   
	mol modstyle 0 $mol_id Isosurface -1e-06 0 0 0 1 1
	mol modcolor 0 $mol_id ColorID $neg_col_id
	mol modstyle 1 $mol_id Isosurface 1e-06 0 0 0 1 1
	mol modcolor 1 $mol_id ColorID $pos_col_id
	mol addrep $mol_id
	mol modstyle 2 $mol_id CPK 1.000000 0.600000 20.000000 20.000000              
   
  mol modselect 2 $mol_id y < 100000 and y > -100000 and x < 100000 and x > -100000 and z < 100000 and z > -100000
	axes location Off
}
proc load_xyz { mol_id } {
	
	mol modstyle 0 $mol_id CPK 0.500000 0.100000 20.000000 20.000000
	mol modmaterial 0 $mol_id Transparent
   mol modcolor 0 $mol_id ColorID 6
}
proc render_pic { mol_id } {
    color Name H tan
    color Name C green
    color Name N yellow
    mol modmaterial 1 $mol_id BrushedMetal
    mol modmaterial 0 $mol_id BrushedMetal
    mol modmaterial 2 $mol_id Edgy
    
    color change rgb 1  0.0 0.0 0.0
    draw text {0 0 0 } " "
    color change rgb gray 0.9 0.87 0.87
    color Display Background gray
    display projection Orthographic
}
proc delete_file { } {
	mol delete all
}
proc rotate_and_scale {  } {
   
  rotate z by t
	rotate y by u
	rotate x by a
  scale by 1.7601203435105475
  translate by 0.03 -0.08 0.0
}
proc set_cols { RGB1 RGB2 RGB3 RGB4 RGB5 } {
   display depthcue off
   set R1 [lindex [split $RGB1] 0 ]
   set G1 [lindex [split $RGB1] 1 ]
   set B1 [lindex [split $RGB1] 2 ]
   set R2 [lindex [split $RGB2] 0 ]
   set G2 [lindex [split $RGB2] 1 ]
   set B2 [lindex [split $RGB2] 2 ]
   set R3 [lindex [split $RGB3] 0 ]
   set G3 [lindex [split $RGB3] 1 ]
   set B3 [lindex [split $RGB3] 2 ]
   set R4 [lindex [split $RGB4] 0 ]
   set G4 [lindex [split $RGB4] 1 ]
   set B4 [lindex [split $RGB4] 2 ]
   set R5 [lindex [split $RGB5] 0 ]
   set G5 [lindex [split $RGB5] 1 ]
   set B5 [lindex [split $RGB5] 2 ]
   color change rgb 18 $R1 $G1 $B1
   color change rgb 19 $R2 $G2 $B2
   color change rgb 20 $R3 $G3 $B3
   color change rgb 21 $R4 $G4 $B4
   color change rgb 22 $R5 $G5 $B5
}
set Negcols {0 21 1 19 2 18 3 18 4 18 5 18 6 21 7 18 8 21 9 19 10 19 11 18 12 18 13 19 14 18 15 21 16 21 17 18 18 19 19 18 20 18 21 18 22 18 23 18 24 18 25 19 26 21 27 21 28 21 29 19 30 21}
set Poscols {0 20 1 18 2 19 3 19 4 19 5 19 6 20 7 19 8 20 9 18 10 18 11 19 12 19 13 18 14 19 15 20 16 20 17 19 18 18 19 19 20 19 21 19 22 19 23 19 24 19 25 18 26 20 27 20 28 20 29 18 30 20}
delete_file
set mol_id 0
foreach i {/home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-200.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-166.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-165.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-133.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-132.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-131.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-130.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-107.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-105.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-83.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-81.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-61.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-59.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-58.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-44.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-42.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-41.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-30.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-29.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-28.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-27.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-18.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-16.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-9.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-8.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-6.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-4.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-3.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-2.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-1.cube /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data/0-0.cube} {
   
   set neg_col_id [dict get $Negcols $mol_id]
   set pos_col_id [dict get $Poscols $mol_id]
   
   load_file $i $mol_id $neg_col_id $pos_col_id
   render_pic $mol_id
   set mol_id [expr $mol_id + 1]
}
#load_xyz $mol_id
set_cols $imag_P_col $imag_N_col $real_P_col $real_N_col $dens_col
rotate_and_scale
source /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/Templates/include.vmd
render Tachyon vmdscene.dat ./bin/tachyon_LINUXAMD64 -aasamples 12 vmdscene.dat -format TARGA -o /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/img/Calibration/0,00_img.tga -trans_max_surfaces 1 -res 950 950
rotate x by 360.000000
rotate x by -360.000000
scale by 1.000000
scale by 1.000000