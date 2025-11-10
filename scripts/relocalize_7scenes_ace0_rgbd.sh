#!/usr/bin/env bash

# Get the absolute path of the directory where this script is located
current_dir="$PWD"

ace_zero_path="../acezero/"
mapping_exe="train_ace.py"
register_exe="register_mapping.py"

# folder with image files
datasets_folder="${current_dir}/datasets/7scenes_ace"
# output directory for the reconstruction
out_dir="${current_dir}/relocalize/7scenes/ace0_rgbd"

# render visualization of the reconstruction, slows down reconstruction considerably
render_visualization=false

scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

for scene in ${scenes[*]}; do

  # find color images for mapping
  mapping_rgb_files="${datasets_folder}/pgt_7scenes_${scene}/train/rgb/*.color.png"
  # find poses for mapping
  mapping_pose_files="${datasets_folder}/pgt_7scenes_${scene}/train/poses/*.pose.txt"
  # find depth for mapping
  mapping_depth_files="${datasets_folder}/pgt_7scenes_${scene}/train/depth/*.depth.png"

  # find color images for query
  query_rgb_files="${datasets_folder}/pgt_7scenes_${scene}/test/rgb/*.color.png"
  # find poses for query
  query_pose_files="${datasets_folder}/pgt_7scenes_${scene}/test/poses/*.pose.txt"

  scene_out_dir="${out_dir}/${scene}"

  if $render_visualization; then
    visualization_cmd="--render_visualization True --render_target_path ${scene_out_dir}/renderings --render_marker_size 0.02 --render_depth_hist True"
  else
    visualization_cmd="--render_visualization False"
  fi

  mkdir -p ${scene_out_dir}

  # network name set to a particular pattern to make sure the visualization works
  network_name="iteration0"

  # run ACE mapping
  (cd "${ace_zero_path}" && python ${mapping_exe} "${mapping_rgb_files}" \
                                                  "${scene_out_dir}/${network_name}.pt" \
                                                  --pose_files "${mapping_pose_files}" \
                                                  --depth_files "${mapping_depth_files}" \
                                                  ${visualization_cmd} \
                                                  --use_external_focal_length 525)

  # estimate query poses
  (cd "${ace_zero_path}" && python ${register_exe} \
                                   "${query_rgb_files}" \
                                   "${scene_out_dir}/${network_name}.pt" \
                                   ${visualization_cmd} \
                                   --use_external_focal_length 525 \
                                   --session ${network_name})

  if $render_visualization; then
    # render video
    /usr/bin/ffmpeg -y -framerate 30 -pattern_type glob -i "${scene_out_dir}/renderings/*.png" -c:v libx264 -pix_fmt yuv420p ${scene_out_dir}/relocalize.mp4
  fi

  # make a copy of the final pose file
  cp ${scene_out_dir}/poses_iteration0.txt ${scene_out_dir}/poses_query.txt

  # evaluate poses
  (cd "${ace_zero_path}" && python eval_poses.py "${scene_out_dir}/poses_query.txt" \
                                                 "${query_pose_files}" \
                                                 --estimate_alignment 'none' \
                                                 --results_file "${scene_out_dir}/pose_eval_results.txt")

done

# Print all evaluation results after all scenes are processed
echo "===================================================="
echo "Scene Reg.Rate 5cm5deg MedianRot MedianTrans ATE RPE"
echo "===================================================="

for scene in ${scenes[*]}; do
  scene_out_dir="${out_dir}/${scene}"
  if [ -f "${scene_out_dir}/pose_eval_results.txt" ]; then
    echo -n "${scene}: "
    sed -n '2p' "${scene_out_dir}/pose_eval_results.txt"
  else
    echo "${scene}: No results found"
  fi
done

echo ""
echo "===================================================="