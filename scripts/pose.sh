cd mmpose
python demo/save_pose_with_det_multiscene.py \
demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
https://download.openxlab.org.cn/models/mmdetection/FasterR-CNN/weight/faster-rcnn_r50_fpn_1x_coco \
configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
../ckpt_weight/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth \
--input examples/88.jpg \
--output-root vis_results/ \
--draw-bbox \
--show-kpt-idx \
--show \
--start 0 \
--end 1 &