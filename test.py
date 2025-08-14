import torch

state_dict = torch.load('/mnt/petrelfs/zhenghao/code/research/HiCroPL/output/imagenet/HiCroPL/vit_b16_c2_ep20_batch4_4ctx_cross_datasets_16shots/seed3/VLPromptLearner/model.pth.tar-5')

keys = state_dict['state_dict'].keys()
print(state_dict['state_dict']['prompt_learner.ctx'].shape)