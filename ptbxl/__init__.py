"""PTB-XL multi-label ECG classification with a Vision Transformer and three baselines."""

from ptbxl.config import CLASS_MAP, LABEL_NAMES, LEAD_NAMES, VIT_CONFIG, PipelineConfig
from ptbxl.data.cache import cache_paths, load_arrays, load_or_build_arrays, save_arrays
from ptbxl.data.dataset import PTBXLDataset, collect_labeled_arrays
from ptbxl.data.utils import (
    aggregate_diagnostic,
    build_targets,
    labels_to_multihot,
    minmax_scale,
    preprocess_signal,
    run_data_pipeline,
    split_by_folds,
    window_signal,
)
from ptbxl.evaluation.metrics import (
    confusion_metrics,
    evaluate_model,
    predict_labels,
    safe_divide,
    threshold_sweep,
)
from ptbxl.model.vit import (
    AsymmetricLoss,
    EncoderBlock,
    MLP,
    StochasticDepth,
    ViT,
    ViTEmbeddings,
    load_model,
)
from ptbxl.model.resnet1d import ResNet1D, BasicBlock1D, load_resnet1d
from ptbxl.model.bilstm import BiLSTM_ECG, load_bilstm
from ptbxl.model.inception import InceptionTime, InceptionModule, load_inception
from ptbxl.model.registry import (
    MODEL_REGISTRY,
    build_model,
    build_vit,
    count_parameters,
    list_models,
    load_checkpoint,
)
from ptbxl.pipeline import ensure_dirs, run_pipeline, save_config
from ptbxl.training.trainer import (
    TrainingResult,
    make_dataloader,
    set_seed,
    train_loop,
    train_model,
)
from ptbxl.visualization.plots import (
    attention_style,
    extract_attention,
    plot_attention_heads,
    plot_attention_heads_grid,
    plot_attention_leads,
    plot_confusion_heatmap,
    plot_losses,
)

__all__ = [
    # config
    "CLASS_MAP", "LABEL_NAMES", "LEAD_NAMES", "VIT_CONFIG", "PipelineConfig",
    # data.utils
    "aggregate_diagnostic", "build_targets", "labels_to_multihot",
    "minmax_scale", "preprocess_signal", "run_data_pipeline",
    "split_by_folds", "window_signal",
    # data.dataset / data.cache
    "PTBXLDataset", "collect_labeled_arrays",
    "cache_paths", "load_arrays", "load_or_build_arrays", "save_arrays",
    # models
    "AsymmetricLoss", "EncoderBlock", "MLP", "StochasticDepth",
    "ViT", "ViTEmbeddings", "load_model",
    "ResNet1D", "BasicBlock1D", "load_resnet1d",
    "BiLSTM_ECG", "load_bilstm",
    "InceptionTime", "InceptionModule", "load_inception",
    "MODEL_REGISTRY", "build_model", "build_vit", "count_parameters",
    "list_models", "load_checkpoint",
    # training
    "TrainingResult", "make_dataloader", "set_seed", "train_loop", "train_model",
    # evaluation
    "confusion_metrics", "evaluate_model", "predict_labels",
    "safe_divide", "threshold_sweep",
    # visualization
    "attention_style", "extract_attention", "plot_attention_heads", "plot_attention_heads_grid", "plot_attention_leads",
    "plot_confusion_heatmap", "plot_losses",
    # pipeline
    "ensure_dirs", "run_pipeline", "save_config",
]
