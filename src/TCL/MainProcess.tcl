logfile /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/visualisation.log
set imag_P_col "0 0.2 0.8"
set imag_N_col "0.8 0.2 0"
set real_P_col "0 1 0"
set real_N_col "0 0 1"
set dens_col   "0.3 0.32 0.3"
proc load_file { filename mol_id neg_col_id pos_col_id } {
	set mol_name $filename
	mol new $mol_name
	
	mol addrep $mol_id
   
	mol modstyle 0 $mol_id Isosurface -5e-05 0 0 0 1 1
	mol modcolor 0 $mol_id ColorID $neg_col_id
	mol modstyle 1 $mol_id Isosurface 5e-05 0 0 0 1 1
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
    color Name H orange
    color Name C green
    color Name N yellow
    mol modmaterial 1 $mol_id BrushedMetal
    mol modmaterial 0 $mol_id BrushedMetal
    mol modmaterial 2 $mol_id Edgy
    
    color change rgb 1  0.0 0.0 0.0
    draw text {232.99349479031207 -6.751603889980686 17.22649179022259 } " "
    color Display Background gray
    color change rgb gray 1 1 1
    display projection Orthographic
}
proc delete_file { } {
	mol delete all
}
proc rotate_and_scale {  } {
   
  rotate z by 0
	rotate y by 0
	rotate x by 0
  scale by 0.20091507414872045
  translate by 0.0 0.0 0.0
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
set Negcols {0 21}
set Poscols {0 20}
delete_file
set mol_id 0
foreach i {/home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/data//tmp0-MainProcess.cube} {
   
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
render Tachyon vmdscene.dat /usr/local/lib/vmd/tachyon_LINUXAMD64 -aasamples 12 vmdscene.dat -format TARGA -o /home/Sangeya/Documents/PhD/Code/Orb_Mov_Mak/img/Calibration/img0.tga -trans_max_surfaces 1 -res 1600 1600
rotate x by 360.000000
rotate x by -360.000000
scale by 1.000000
scale by 1.000000