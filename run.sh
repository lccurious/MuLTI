# MuLTI View=2, L=2, p=1, D_c = 4
CUDA_VISIBLE_DEVICES=1 python train_multi.py --batch-size 2000 \
--space-type unbounded \
--num-view 2 \
--n-shared 4 \
--causal-con-order 1 \
--time-lag 2 \
--length 4 \
--m-p 2 \
--c-p 1 \
--c-param 0.05 \
--p 1 \
--n-steps 800001 \
--lr 0.003 \
--seed 42 \
--beta1 0.01 \
--beta2 0.01 \
--save-dir outputs/dry/VAR


# MuLTI View=2, L=4, p=1, D_c = 4
CUDA_VISIBLE_DEVICES=1 python train_multi.py --batch-size 2000 \
--space-type unbounded \
--num-view 2 \
--n-shared 4 \
--causal-con-order 1 \
--time-lag 4 \
--length 6 \
--m-p 2 \
--c-p 1 \
--c-param 0.05 \
--p 1 \
--n-steps 800001 \
--lr 0.003 \
--seed 42 \
--beta1 0.01 \
--beta2 0.01 \
--save-dir outputs/dry/VAR


# MuLTI View=2, L=6, p=1, D_c = 4
CUDA_VISIBLE_DEVICES=1 python train_multi.py --batch-size 2000 \
--space-type unbounded \
--num-view 2 \
--n-shared 4 \
--causal-con-order 1 \
--time-lag 6 \
--length 8 \
--m-p 2 \
--c-p 1 \
--c-param 0.05 \
--p 1 \
--n-steps 800001 \
--lr 0.003 \
--seed 42 \
--beta1 0.01 \
--beta2 0.01 \
--save-dir outputs/dry/VAR


# MuLTI View=3, L=2, p=1, D_c = 4
CUDA_VISIBLE_DEVICES=1 python train_multi.py --batch-size 2000 \
--space-type unbounded \
--num-view 3 \
--n-shared 4 \
--causal-con-order 1 \
--time-lag 2 \
--length 4 \
--m-p 2 \
--c-p 1 \
--c-param 0.05 \
--p 1 \
--n-steps 800001 \
--lr 0.003 \
--seed 42 \
--beta1 0.01 \
--beta2 0.01 \
--save-dir outputs/dry/VAR

# MuLTI View=4, L=2, p=1, D_c = 4
CUDA_VISIBLE_DEVICES=1 python train_multi.py --batch-size 2000 \
--space-type unbounded \
--num-view 4 \
--n-shared 2 \
--causal-con-order 1 \
--time-lag 2 \
--length 4 \
--m-p 2 \
--c-p 1 \
--c-param 0.05 \
--p 1 \
--n-steps 800001 \
--lr 0.003 \
--seed 42 \
--beta1 0.01 \
--beta2 0.01 \
--save-dir outputs/dry/VAR
