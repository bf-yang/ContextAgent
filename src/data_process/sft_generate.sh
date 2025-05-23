# Generate sft data for different settings

python src/data_process/sft_data_generate.py \
    --dataset cab --mode train --think w_t --personas w_p

python src/data_process/sft_data_generate.py \
    --dataset cab --mode train --think wo_t --personas w_p 

python src/data_process/sft_data_generate.py \
    --dataset cab --mode train --think w_t --personas wo_p  

python src/data_process/sft_data_generate.py \
    --dataset cab --mode train --think wo_t --personas wo_p  
