export CUDA_VISIBLE_DEVICES=0

python main.py \
    --backbone "ResNet50" \
    --data_path "./data/VOC2012" \
    --input_size 608 \
    --batch_size 16 \
    --eval_batch_size 64 \
    --stage1_total_steps 8000 \
    --stage2_total_steps 4000 \
    --eval_steps 500 \
    --stage1_learning_rate 1e-4 \
    --stage2_learning_rate 1e-5 \
    --output_path "./output"
