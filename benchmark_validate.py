"""
Batch benchmark evaluator for Search-R1 checkpoints.

Usage example (same level as `train_grpo.sh`):
python3 benchmark_validate.py \
  actor_rollout_ref.model.path=verl_checkpoints/your_exp/actor/global_step_1000 \
  data.val_files=data/nq_search/test.parquet \
  data.val_batch_size=256 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  do_search=true \
  max_turns=2 \
  retriever.url=http://127.0.0.1:8000/retrieve \
  retriever.topk=3
"""

import json
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    raise NotImplementedError


class RewardManager:
    """Function RM aligned with main_ppo.py."""

    def __init__(self, tokenizer, num_examine=1, format_score=0.0):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score

    def __call__(self, data):
        import torch

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(solution_str=sequences_str,
                                     ground_truth=ground_truth,
                                     format_score=self.format_score)

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


_RAY_WORKER_ENV_VARS = {
    'TOKENIZERS_PARALLELISM': 'true',
    'NCCL_DEBUG': 'WARN',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
}


@hydra.main(config_path='verl/trainer/config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': _RAY_WORKER_ENV_VARS})

    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Benchmark only uses validation; still keep trainer initialization robust.
    with open_dict(config):
        if not config.data.get('train_files'):
            config.data.train_files = config.data.val_files
        if not config.data.get('train_data_num'):
            config.data.train_data_num = config.data.val_data_num
        config.trainer.val_before_train = False
        config.trainer.val_only = True

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)

    trainer.init_workers()
    try:
        metrics = trainer._validate()
        print('Benchmark metrics:')
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    finally:
        trainer.shutdown()


if __name__ == '__main__':
    main()
