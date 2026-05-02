.
├── app
│   ├── cli_helpers.py
│   ├── __init__.py
│   ├── live_stream.py
│   ├── main.py
│   ├── runtime.py
│   └── server.py
├── CLOUD_DEPLOYMENT.md
├── configs
│   ├── advanced.yaml
│   ├── agent_auto.yaml
│   ├── attention_multimodal.yaml
│   ├── attention_transfer_advanced.yaml
│   ├── attention_video.yaml
│   ├── auto_student.yaml
│   ├── causal_lm_core_minimal.yaml
│   ├── cifar10_vit.yaml
│   ├── cpu_smoke_test.yaml
│   ├── default.yaml
│   ├── electra_high_compression.yaml
│   ├── feature_distillation.yaml
│   ├── kaggle_dual_t4.yaml
│   ├── kaggle_quick_test.yaml
│   ├── kd_hinton_advanced.yaml
│   ├── m2_test.yaml
│   ├── mac_m2_optimized.yaml
│   ├── mac_m2_test.yaml
│   ├── multimodal_clip.yaml
│   ├── multi_stage_auto.yaml
│   ├── multi_stage.yaml
│   ├── quick_test_minilm.yaml
│   ├── roberta_balanced.yaml
│   ├── similarity_transfer.yaml
│   ├── t4_gpt_feature_lowmem_qat_plan.yaml
│   ├── t4_gpt_feature_teacher_student.yaml
│   ├── test_backend_validation.yaml
│   ├── test_invalid_data.yaml
│   ├── test_invalid.yaml
│   ├── test_live_quick.yaml
│   ├── test_visualization.yaml
│   ├── tinybert_high_compression.yaml
│   ├── vision_cifar10.yaml
│   └── with_teacher_training.yaml
├── core
│   ├── adapters
│   │   ├── adapter_registry.py
│   │   ├── base_adapter.py
│   │   ├── __init__.py
│   │   ├── multimodal_adapter.py
│   │   ├── text_adapter.py
│   │   ├── vision_adapter.py
│   │   └── vlm_adapter.py
│   ├── agents
│   │   ├── __init__.py
│   │   └── teacher_agent.py
│   ├── auto_student
│   │   ├── auto_student_builder.py
│   │   ├── heuristics.py
│   │   ├── __init__.py
│   │   ├── templates
│   │   │   ├── base_transformer.json
│   │   │   └── student_template.yaml
│   │   └── validator.py
│   ├── config
│   │   ├── config_manager.py
│   │   └── __init__.py
│   ├── distillers
│   │   ├── attention_transfer.py
│   │   ├── base_distiller.py
│   │   ├── causal_lm
│   │   │   ├── checkpoint.py
│   │   │   ├── determinism.py
│   │   │   ├── distillation.py
│   │   │   ├── fault_injection.py
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py
│   │   │   ├── regression_gate.py
│   │   │   ├── trainer.py
│   │   │   └── validation.py
│   │   ├── feature_distiller.py
│   │   ├── __init__.py
│   │   ├── kd_hinton.py
│   │   ├── multi_stage_distiller.py
│   │   ├── multi_stage_distiller.py.backup
│   │   ├── presets.py
│   │   ├── similarity_transfer.py
│   │   └── toolkit.py
│   ├── explainability
│   │   ├── __init__.py
│   │   ├── lime_explainer.py
│   │   └── shap_explainer.py
│   ├── inference
│   │   └── predict.py
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   ├── model_saver.py
│   │   ├── model_wrapper.py
│   │   └── projection_heads.py
│   ├── pipelines
│   │   ├── base_pipeline.py
│   │   ├── __init__.py
│   │   ├── multi_stage_pipeline.py
│   │   ├── pipeline_builder.py
│   │   ├── pipeline_registry.py
│   │   ├── README.md
│   │   └── single_distiller_pipeline.py
│   ├── pkg
│   │   ├── exporter.py
│   │   ├── __init__.py
│   │   └── manifest.py
│   ├── preflight
│   │   ├── analyser.py
│   │   ├── data_inspector.py
│   │   ├── __init__.py
│   │   ├── model_inspector.py
│   │   ├── model_validator.py
│   │   └── resource_probe.py
│   ├── preprocessing
│   │   ├── advanced.py
│   │   ├── built_ins.py
│   │   └── registry.py
│   ├── quant
│   │   ├── calibration.py
│   │   ├── __init__.py
│   │   ├── ptq.py
│   │   └── qat.py
│   └── utils
│       ├── checkpoint.py
│       ├── data_validator.py
│       ├── device_manager.py
│       ├── device_utils.py
│       ├── download_monitor.py
│       ├── download_progress.py
│       ├── hf_dataset_loader.py
│       ├── __init__.py
│       ├── json_utils.py
│       ├── logger.py
│       ├── metrics.py
│       └── progress_tracker.py
├── data
│   ├── augmentations.py
│   ├── dataloaders.py
│   ├── generated_students
│   │   ├── student_bert-base-uncased_balanced_50pct_20251103_100155.json
│   │   ├── student_bert-base-uncased_balanced_50pct_20251103_100155.yaml
│   │   ├── student_bert-base-uncased_balanced_50pct_20251103_121615.json
│   │   ├── student_bert-base-uncased_balanced_50pct_20251103_121615.yaml
│   │   ├── student_bert-base-uncased_balanced_50pct_20251109_152842.json
│   │   ├── student_bert-base-uncased_balanced_50pct_20251109_152842.yaml
│   │   ├── student_roberta-base_balanced_50pct_20251103_121743.json
│   │   ├── student_roberta-base_balanced_50pct_20251103_121743.yaml
│   │   ├── student_roberta-base_balanced_50pct_20251103_122703.json
│   │   └── student_roberta-base_balanced_50pct_20251103_122703.yaml
│   ├── image_dataloaders.py
│   ├── imdb_sample.jsonl
│   ├── imdb_train.jsonl
│   ├── imdb_train.jsonl.backup
│   ├── imdb_val.jsonl
│   ├── imdb_val.jsonl.backup
│   ├── __init__.py
│   ├── preprocess.py
│   ├── sample_train.jsonl
│   ├── sample_val.jsonl
│   ├── sst2_test
│   │   ├── test.jsonl
│   │   ├── train.jsonl
│   │   └── validation.jsonl
│   └── test_hf_dataset
│       ├── test.jsonl
│       ├── train.jsonl
│       └── validation.jsonl
├── dev.sh
├── docs
│   ├── ATTENTION_IMPLEMENTATION_SUMMARY.md
│   ├── ATTENTION_QUICKREF.md
│   ├── ATTENTION_TRANSFER_GUIDE.md
│   ├── AUTO_STUDENT_GUIDE.md
│   ├── COMPARISON_COMPLETE.md
│   ├── COMPARISON_FEATURE.md
│   ├── COMPARISON_IMPLEMENTATION_SUMMARY.md
│   ├── COMPARISON_INTEGRATION_SUMMARY.md
│   ├── COMPARISON_QUICKREF.md
│   ├── CONFIG_MANAGER_IMPROVEMENTS.md
│   ├── CONFIG_REVIEW_SUMMARY.md
│   ├── design.md
│   ├── DISTILLATION_WORKFLOW.md
│   ├── FUTURE_FEATURES.md
│   ├── ISSUE_RESOLUTION_OCT23.md
│   ├── MAC_M2_MODEL_PAIRS.md
│   ├── model_comparison_guide.md
│   ├── msme_playbook.md
│   ├── OPTIMIZER_GUIDE.md
│   ├── overview.md
│   ├── PHASE_A_COMPLETE.md
│   ├── PHASE_A_INTEGRATION_COMPLETE.md
│   ├── PREFLIGHT_GUIDE.md
│   ├── PREFLIGHT_INTEGRATION.md
│   ├── quickstart.md
│   ├── SIMILARITY_TRANSFER.md
│   ├── structure_overview.md
│   ├── superpowers
│   │   └── plans
│   │       ├── 2026-04-24-latitude-7490-audit-and-kaggle-validation.md
│   │       ├── 2026-04-24-mypy-fix-plan.md
│   │       └── 2026-04-25-kd-foundation-stabilization.md
│   ├── TEACHER_AGENT.md
│   ├── TEACHER_AGENT_UI_INTEGRATION.md
│   ├── TEACHER_ISSUE_ANALYSIS.md
│   ├── TEACHER_TRAINING_AND_CONFUSION_MATRIX.md
│   ├── TRAINER_FIX_EXPLAINED.md
│   ├── workflow_paper.tex
│   ├── WORKFLOW_TEST_WITH_VISUALIZATION.md
│   ├── ZYNTHE_EVALX_PHASE_A.md
│   ├── ZYNTHE_EVALX_QUICKREF.md
│   └── ZYNTHE_EVALX_SUMMARY.md
├── evaluation
│   ├── benchmark.py
│   ├── diagnostics.py
│   ├── evaluation_report.py
│   ├── evaluator_extended.py
│   ├── evaluator.py
│   ├── __init__.py
│   ├── metrics_extended.py
│   ├── metrics.py
│   ├── model_comparison.py
│   ├── report.py
│   ├── tasks
│   │   ├── gsm8k_subset.py
│   │   ├── mmlu_mini.py
│   │   └── truthfulqa_lite.py
│   └── visualizer.py
├── examples
│   ├── attention_transfer_examples.py
│   ├── causal_lm_validation_smoke.py
│   ├── compare_teacher_student.py
│   ├── minimal_distill.py
│   ├── minimal_eval.py
│   ├── pipeline_trainer_integration.py
│   ├── teacher_agent_demo.py
│   └── Teacher_vs_Student_Comparison.ipynb
├── full_integration_test.log
├── implementation_plan.md
├── model_comparison.png
├── notebooks
│   ├── kaggle_dual_t4_test.ipynb
│   ├── kaggle_multi_model_test.ipynb
│   ├── pipeline_colab_test.ipynb
│   ├── test_distillation_colab.ipynb
│   ├── test_vision_colab.ipynb
│   └── Zynthe_Cloud_Backend.ipynb
├── opencode.json
├── package-lock.json
├── pending_plan.md
├── preflight_reports
│   ├── optimized_config.yaml
│   ├── preflight_report_20251023_020251.json
│   └── preflight_report_20251023_020251.txt
├── PROJECT_STRUCTURE.md
├── README.md
├── requirements.txt
├── scripts
│   ├── check_artifacts.py
│   ├── debug_preprocessing.py
│   ├── setup-venv.sh
│   └── test_visualizations.py
├── start.sh
├── start-ui.sh
├── start-zynthe.sh
├── test_output
│   └── stage_1_checkpoint.pt
├── tests
│   ├── conftest.py
│   ├── test_adapters.py
│   ├── test_cli_banner.py
│   ├── test_cpu_smoke.py
│   ├── test_evaluation_report.py
│   ├── test_fixes.py
│   ├── test_model_loader_modality.py
│   ├── test_model_saver.py
│   ├── test_optimizer_and_overrides.py
│   ├── test_pipeline_refactor.py
│   └── test_pipelines
│       └── __init__.py
├── tools
│   ├── diagnose_teacher.py
│   ├── dryrun_training_loop.py
│   ├── repair_teacher_labels.py
│   └── system_tester.py
├── training
│   ├── __init__.py
│   ├── optimizer.py
│   ├── scheduler.py
│   └── trainer.py
└── ZYNTHE_PLAN.md

38 directories, 259 files
