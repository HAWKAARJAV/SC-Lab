# Figure manifest (illustrative + paper reference)

**Provenance:** A1–A9 → `00_paper_reference/` (published crops + derived A4–A5). B10–E33 → `01_expected_illustrative_scenario/` (synthetic teaching data).

| ID | Name | Path | Status | Data source | Provenance |
|----|------|------|--------|-------------|------------|
| A1 | Paper Fig. 7 eye mRMR | `00_paper_reference/figures/figure_07_eye_mrmr_ranking.png` | generated | eye_mrmr_ranking.csv | published_paper_or_derived_methodology |
| A2 | Paper Fig. 8 Sugeno FIS | `00_paper_reference/figures/figure_08_sugeno_fis_architecture.png` | generated | n/a | published_paper_or_derived_methodology |
| A3 | Paper Fig. 9 GA convergence | `00_paper_reference/figures/figure_09_ga_convergence.png` | generated | n/a | published_paper_or_derived_methodology |
| A4 | Derived triangular MFs | `00_paper_reference/figures/figure_A4_fuzzy_membership_functions.png` | generated | paper_figures_method.py | published_paper_or_derived_methodology |
| A5 | Derived rule space | `00_paper_reference/figures/figure_A5_fuzzy_rule_distribution.png` | generated | paper_figures_method.py | published_paper_or_derived_methodology |
| A6 | Paper Fig. 10 SEED-IV CM | `00_paper_reference/figures/figure_10_seed_iv_confusion_matrices.png` | generated | n/a | published_paper_or_derived_methodology |
| A7 | Paper Fig. 11 SEED-IV ROC | `00_paper_reference/figures/figure_11_seed_iv_roc_curves.png` | generated | n/a | published_paper_or_derived_methodology |
| A8 | Paper Fig. 12 MAHNOB CM | `00_paper_reference/figures/figure_12_mahnob_confusion_matrices.png` | generated | n/a | published_paper_or_derived_methodology |
| A9 | Paper Fig. 13 MAHNOB ROC | `00_paper_reference/figures/figure_13_mahnob_roc_curves.png` | generated | n/a | published_paper_or_derived_methodology |
| B10 | Hard vs fuzzy timeline | `M1_fuzzy_gate/figures/10_hard_vs_fuzzy_timeline.png` | generated | gate_weights.csv | synthetic_teaching |
| B11 | M1 membership functions | `M1_fuzzy_gate/figures/11_fuzzy_membership_functions.png` | generated | n/a (FUZZY params) | synthetic_teaching |
| B12 | Gate weight histogram | `M1_fuzzy_gate/figures/12_gate_weight_distribution.png` | generated | gate_weights.csv | synthetic_teaching |
| B13 | Gate ablation by band | `M1_fuzzy_gate/figures/13_gate_ablation_by_band.png` | generated | gate_ablation_by_band.csv | synthetic_teaching |
| B14 | Gate P/R/F1 | `M1_fuzzy_gate/figures/14_gate_precision_recall_f1.png` | generated | synthetic episode mask | synthetic_teaching |
| C15 | Raw vs normalised pupil | `M2_adaptive_temporal_gate/figures/15_raw_vs_normalised_pupil.png` | generated | temporal_gate_scores.csv | synthetic_teaching |
| C16 | Temporal smoothing | `M2_adaptive_temporal_gate/figures/16_temporal_smoothing.png` | generated | temporal_gate_scores.csv | synthetic_teaching |
| C17 | Adaptive gate components | `M2_adaptive_temporal_gate/figures/17_adaptive_gate_components.png` | generated | temporal_gate_scores.csv | synthetic_teaching |
| C18 | Gate mechanism comparison | `M2_adaptive_temporal_gate/figures/18_hard_fuzzy_adaptive_comparison.png` | generated | temporal_gate_scores.csv | synthetic_teaching |
| C19 | Gate quality | `M2_adaptive_temporal_gate/figures/19_gate_quality_comparison.png` | generated | episode mask in teaching_data | synthetic_teaching |
| C20 | Noise sensitivity | `M2_adaptive_temporal_gate/figures/20_noise_sensitivity.png` | generated | noise_gate_sweep() | synthetic_teaching |
| D21 | Individual band accuracy | `M3_multiband_fusion/figures/21_individual_band_accuracy.png` | generated | band_probabilities.csv | synthetic_teaching |
| D22 | Learned band weights | `M3_multiband_fusion/figures/22_learned_band_weights.png` | generated | learned_band_weights.csv | synthetic_teaching |
| D23 | Alpha vs multiband | `M3_multiband_fusion/figures/23_alpha_vs_multiband.png` | generated | band_probabilities.csv | synthetic_teaching |
| D24 | Band-count fusion | `M3_multiband_fusion/figures/24_fusion_band_count.png` | generated | band_probabilities.csv | synthetic_teaching |
| D25 | Equal vs learned | `M3_multiband_fusion/figures/25_equal_vs_learned_fusion.png` | generated | learned_band_weights.csv | synthetic_teaching |
| D26 | Fused confusion matrix | `M3_multiband_fusion/figures/26_fused_confusion_matrix.png` | generated | fusion_predictions.csv | synthetic_teaching |
| D27 | Fused ROC | `M3_multiband_fusion/figures/27_fused_roc_curves.png` | generated | fusion_predictions.csv | synthetic_teaching |
| E28 | Combined architecture | `M4_combined_model/figures/28_combined_architecture.png` | generated | n/a (diagram) | synthetic_teaching |
| E29 | M4 adaptive gate | `M4_combined_model/figures/29_m4_adaptive_gate.png` | generated | band_probabilities.csv + gates | synthetic_teaching |
| E30 | M4 band contributions | `M4_combined_model/figures/30_m4_band_contributions.png` | generated | learned_band_weights.csv | synthetic_teaching |
| E31 | Final CM | `M4_combined_model/figures/31_final_confusion_matrix.png` | generated | final_predictions.csv | synthetic_teaching |
| E32 | Final ROC | `M4_combined_model/figures/32_final_roc_curves.png` | generated | final_predictions.csv | synthetic_teaching |
| E33 | Class-wise metrics | `M4_combined_model/figures/33_classwise_precision_recall_f1.png` | generated | final_predictions.csv | synthetic_teaching |
