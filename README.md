This is based on the [evo](https://github.com/MichaelGrupp/evo) package. Use the source code without installation.

### To use the `align_odom` option:
  
  ```
  python test_ape.py tum rtk_pva_gps_preprocessed_latest.txt lego_loam_traj_to_gps_frame.txt --align_odom
  ```
### To use the segment-then-error function :
  
  ```
  python test_seg_ape.py tum rtk_pva_gps_preprocessed_latest.txt lego_loam_traj_to_gps_frame.txt
  ```

- By default, the segment length is 50 meters. To change that, use `--seg <length>`
### Three different options for alignment can be used for both `test_ape.py` and `test_seg_ape.py` : `--align`, `--align_origin`, `--align_odom`

### To compute the error in z direction and xy plane respectively, use `-r z` and `-r xy` 
  
