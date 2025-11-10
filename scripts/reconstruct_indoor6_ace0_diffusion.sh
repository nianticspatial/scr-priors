#!/usr/bin/env bash

# Get the absolute path of the directory where this script is located
current_dir="$PWD"

ace_zero_path="../acezero/"
reconstruction_exe="ace_zero.py"

# folder with image files
datasets_folder="${current_dir}/datasets/indoor6/"
# output directory for the reconstruction
out_dir="${current_dir}/reconstructions/indoor6/ace0_diffusion"
# target directory for benchmarking results
benchmarking_out_dir="${current_dir}/benchmark/indoor6/ace0_diffusion"

scenes=("scene1" "scene2a" "scene3" "scene4a" "scene5" "scene6")

# render visualization of the reconstruction, slows down reconstruction considerably
render_visualization=false

# run view synthesis benchmarking after reconstruction
run_benchmark=true
# benchmarking needs to happen in the nerfstudio environment
benchmarking_environment="nerfstudio"
# benchmarking method, splatfacto or nerfacto
benchmarking_method="splatfacto"
# when using splatfacto, we need a point cloud initialization, either sparse or dense
# dense is recommended if you have very dense coverage of the scene, eg. > 2000 images
benchmarking_dense_pcinit=true

for scene in ${scenes[*]}; do

  # find color images for the reconstruction
  input_rgb_files="${datasets_folder}/${scene}/train/rgb/*.jpg"

  # get focal length (first line, third item) of first mapping image
  input_calibration_files=$(find "${datasets_folder}/${scene}/train/calibration/" -path "*.txt" -print -quit)
  input_calibration_files=$(ls $input_calibration_files | head -1)
  focal_length=$(head -n 1 ${input_calibration_files})
  focal_length=$(echo $focal_length | cut -d ' ' -f 3)
  # print focal length
  echo "Using focal length from Indoor6 for Mapping: ${focal_length}"

  scene_out_dir="${out_dir}/${scene}"

  if $render_visualization; then
    visualization_cmd="--render_visualization True --render_marker_size 0.02 --render_depth_hist True"
  else
    visualization_cmd="--render_visualization False"
  fi

  if ${run_benchmark} && [ "${benchmarking_method}" = "splatfacto" ]; then
    export_pc_cmd="--export_point_cloud True --dense_point_cloud ${benchmarking_dense_pcinit}"
  else
    export_pc_cmd="--export_point_cloud False --dense_point_cloud False"
  fi

  mkdir -p ${scene_out_dir}

  # run ACE0 reconstruction
  (cd "${ace_zero_path}" && python $reconstruction_exe "${input_rgb_files}" \
                                                       "${scene_out_dir}" \
                                                       --try_seeds 5 \
                                                       ${visualization_cmd} \
                                                       --use_external_focal_length "${focal_length}" \
                                                       ${export_pc_cmd} \
                                                       --loss_structure "dsac*" \
                                                       --prior_loss_type diffusion \
                                                       --prior_diffusion_model_path "${current_dir}/models/diffusion_prior.pt" \
                                                       --prior_loss_weight 200 \
                                                       ) 2>&1 | tee "${scene_out_dir}/log_${scene}.txt"

  # run pose evaluation
  (cd "${ace_zero_path}" && python eval_poses.py "${scene_out_dir}/poses_final.txt" \
                                                 "${datasets_folder}/${scene}/train/poses/*.txt" \
                                                 --estimate_alignment least_squares \
                                                 --estimate_alignment_conf_threshold 0 \
                                                 --results_file pose_eval_results.txt)

  # run benchmarking if requested
  if $run_benchmark; then

    # run the standard PSNR benchmark split with 1 query frames and 7 mapping frames
    benchmarking_scene_dir="${benchmarking_out_dir}/1_7/${scene}"
    mkdir -p ${benchmarking_scene_dir}
    (cd "${ace_zero_path}" && conda run --no-capture-output -n ${benchmarking_environment} python -m benchmarks.benchmark_poses --pose_file "${scene_out_dir}/poses_final.txt" \
                                                                                                                                --output_dir "${benchmarking_scene_dir}" \
                                                                                                                                --images_glob_pattern "${input_rgb_files}" \
                                                                                                                                --method ${benchmarking_method} \
                                                                                                                                ) 2>&1 | tee "${benchmarking_out_dir}/log_${scene}.txt"

  fi
done

# Print all evaluation results after all scenes are processed
echo "Pose Benchmark Results"
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

# Also print results of PSNR benchmark
if $run_benchmark; then
echo ""
echo "PSNR Benchmark Results"
echo "===================================================="
(cd "${ace_zero_path}" && python scripts/show_benchmark_results.py "${benchmarking_out_dir}/1_7" --method ${benchmarking_method})
echo "===================================================="
fi