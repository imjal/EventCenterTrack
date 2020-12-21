python main.py ctdet --exp_id all_200_dat_viz_test_last_redo --dataset prophesee --test \
--ann_data_dir /data2/jl5/prophese --load_model /home/jl5/EventCenterTrack/exp/tracking/all_200_dat/model_70.pth \
--data_stream_file /data2/jl5/prophese/detection_dataset_duration_60s_ratio_1.0/last_5_dat.txt \
--num_iters 50000 --debug 4 --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0,1,2,3
# --custom_dataset_ann_path /data2/jl5/mmdetect_results/driving1000/coco_images.json
# --custom_dataset_img_path /data2/jl5/driving1/driving1000
