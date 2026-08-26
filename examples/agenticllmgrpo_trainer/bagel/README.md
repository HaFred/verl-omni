For the bagel und+gen setup, we want to make the `../agent_llm` RPCO recipe grpo training works, for the flow matching part with:
Training targets:
1. agent llm: bagel understanding (und) path -> GRPO
2. generation tool: bagel generation (gen) path -> FlowGRPO

The noteworthy/challenging part is to make the generation tool trainable, and also the actor models will incoporate both the und path and the gen path, as both the agent llm and the flow matching in bagel are trained together for multi-turns image generations.

The orginal @/home/fq9hpsac/fq9hpsacuser11/fred/verlomni-fredfork/examples/agenticllmgrpo_trainer/agent_llm/run_agenticrpco_grpo_lora.sh will evolve to using verl-omni with flowgrpo:```
unicot_img_train_path=xxx
unicot_img_test_path=xxx
python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$unicot_img_train_path \
    data.val_files=$unicot_img_test_path \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    trainer.project_name=flow_grpo \
```