# Generate sft data for different settings
# cab dataset
python src/data_process/sft_data_generate.py \
    --dataset cab --mode train --think w_t --personas w_p  # with think and personas

python src/data_process/sft_data_generate.py \
    --dataset cab --mode train --think wo_t --personas w_p 

python src/data_process/sft_data_generate.py \
    --dataset cab --mode train --think w_t --personas wo_p  

python src/data_process/sft_data_generate.py \
    --dataset cab --mode train --think wo_t --personas wo_p  


# # cab_lite dataset
# python src/data_process/sft_data_generate.py \
#     --dataset cab_lite --mode train --think w_t --personas w_p  # with think and personas

# python src/data_process/sft_data_generate.py \
#     --dataset cab_lite --mode train --think wo_t --personas w_p 

# python src/data_process/sft_data_generate.py \
#     --dataset cab_lite --mode train --think w_t --personas wo_p  

# python src/data_process/sft_data_generate.py \
#     --dataset cab_lite --mode train --think wo_t --personas wo_p  


# # cab_ood dataset
# python src/data_process/sft_data_generate.py \
#     --dataset cab_ood --mode train --think w_t --personas w_p  # with think and personas

# python src/data_process/sft_data_generate.py \
#     --dataset cab_ood --mode train --think wo_t --personas w_p 

# python src/data_process/sft_data_generate.py \
#     --dataset cab_ood --mode train --think w_t --personas wo_p  

# python src/data_process/sft_data_generate.py \
#     --dataset cab_ood --mode train --think wo_t --personas wo_p  