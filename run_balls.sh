# MuLTI View=2, L=2, p=1, D_c = 4
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multi_balls.py --batch-size 128 \
--space-type unbounded \
--num-view 2 \
--n 16 \
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
--save-dir outputs/dry/Mass-spring
