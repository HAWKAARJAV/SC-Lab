# Research notes for the fuzzy-gating manuscript (prepared 4 Nov 2026)

These notes accompany `main.tex` / `main.pdf`. They record what was verified, what is still open, and what the team must decide.

---

## 1. Decision needed first: which dataset?

No public EEG + eye-tracking dataset called **"Zuko"** could be found. The name almost certainly refers to **ZuCo** (Zurich Cognitive Language Processing Corpus):

| | ZuCo 1.0 (Hollenstein et al. 2018, *Sci Data* 5:180291) | SEED-IV (Zheng et al. 2019, *IEEE TCYB* 49(3)) |
|---|---|---|
| Access | Free, CC BY 4.0, https://osf.io/q3zws | Application form, BCMI lab, SJTU |
| Task | Natural reading of sentences | Watching 72 emotional film clips |
| Labels | Sentiment of the text: positive 140 / negative 137 / neutral 123 (from the Stanford Sentiment Treebank) | Induced emotion: happy / sad / fear / neutral |
| Subjects | 12 | 15 × 3 sessions |
| EEG | EGI 128-ch, 500 Hz (105 used) | NeuroScan 62-ch, 1000 Hz |
| Eye tracker | EyeLink 1000 Plus, 500 Hz, single pupil-size value | SMI glasses; pupil X/Y diameters |
| Used by the baseline? | No | Yes (94.42%) |

**Consequence.** ZuCo has **no sadness, happiness or fear labels**, and its labels describe the sentence, not the reader's emotional state. On ZuCo, the four-class emotion task from the project brief cannot be done.

The manuscript is currently written for ZuCo 1.0, as three-class stimulus sentiment, and says so openly in Sect. 5. If the team actually means SEED-IV (or MAHNOB-HCI), the following parts must be rewritten:
- Sect. 5
- the "Consequences" paragraph
- Table 5
- the class names in Tables 9–10
- the 35.0% majority rate
- Limitation (i)

The methodology does not change. Note that a SEED-IV study is directly comparable with the baseline, but the dataset requires an access application.

Further ZuCo-specific risks:
- Sentences are read in a few seconds, so with 4-s windows each trial yields only 1–2 windows and there is little for any gate to act on. Measure the reading-time distribution before fixing L.
- The baseline's eight eye-tracking features become six, because EyeLink gives one pupil value rather than separate X and Y diameters.
- The baseline channels T7, T8 and Fp2 have no exact HydroCel-128 equivalents.
- Published EEG-only sentiment results on ZuCo are weak. Wang & Ji (AAAI 2022) report supervised EEG baselines near chance (~33%) for ternary sentiment. Expect low absolute numbers.

## 2. Critical review of the research direction

Things a reviewer will raise. Several are already addressed in the draft.

1. **Baseline gate is under-specified.** The original paper says the "maximum" of mean pupil Y is the threshold, and that EEG at the moments it is reached is used. It does not say per trial, per subject or global, nor how many windows are kept. The draft formalises this as `w = 1[z ≥ τ]` with τ PENDING. Document exactly what the code does.
2. **The baseline's evaluation likely leaks.** It runs 5-fold CV over pooled 4-s segments, so windows from the same clip and subject appear in both train and test. Feature selection also appears to have been done before CV. The 94.42% figure is probably optimistic. The draft makes subject- and sentence-disjoint CV (P1) primary, and keeps the original protocol (P3) only to measure that optimism.
3. **The eye-tracking gate feature is chosen using class labels.** mRMR over labels is fine only if it runs inside training folds. The draft requires this.
4. **A single-input fuzzy gate is a piecewise-linear soft threshold.** Its novelty therefore rests on (a) graded relevance, (b) interpretability, (c) the integration points. It must be compared against a plain linear or sigmoid soft gate (ablation A4); otherwise a reviewer can say "this is just soft weighting".
5. **The computation argument flips.** The baseline sells gating as reducing computation. M1 processes every window, so report cost honestly.
6. **Pupil confounds.** Pupil size indexes arousal, not valence (Bradley 2008). It is dominated by luminance (Mathôt 2018) and, during reading, by cognitive load and gaze position. Neither model corrects for this; it is listed as a limitation.
7. **Entropy on short signals.** SampEn and PermEn with m = 5 on DWT coefficient vectors of 4-s windows (64 samples at level 3) are unreliable. Compute them on reconstructed sub-band signals, and confirm in the code.
8. **DWT bands ≠ stated bands.** At 128 Hz, the dyadic levels are 32–64, 16–32, 8–16, 4–8 and 0–4 Hz, not the 8–13 / 13–25 / 25–60 Hz the baseline quotes.
9. **Low statistical power.** With 12 subjects, a small M0–M1 difference will probably not be significant. Report effect sizes and CIs, and do not call a null result "equivalence".
10. **S-FIS output.** The baseline mixes Sugeno inference with "centroid defuzzification". The draft uses per-class rule support (Eq. 7) so that windows can be aggregated. Reconcile this with the code.

## 3. Writing-style guide

Derived from Rani & Shokeen, *J Supercomput* 2022 and 2024. The rules below are applied in the draft.

**Section skeleton**
- Abstract, then Keywords separated by "·"
- 1 Introduction, with 1.1 Motivation, 1.2 Contributions (numbered list) and 1.3 Organization of the paper
- 2 Related works, ending with a summary table of related works (Ref / Approach / Dataset / Result / Limitation)
- 3 Background or proposed model; 4 Experimental results; 5 Conclusion and future work
- Declarations at the end: Funding, Conflict of interest, Data availability, Author contributions

**Tables and figures**
- An early "List of abbreviations" table and a "Key notations" table (Notation / Description / Value).
- Captions read "Table n", "Fig. n" in short title case. Each table or figure is introduced in the text before it appears ("Table 2 gives…").
- One architecture figure for the whole proposed model, plus block diagrams of its sub-components.
- Results are given both as tables of per-dataset metrics and as figures, followed by a comparison table against existing work.

**Prose and equations**
- Paragraphs open with a topic sentence. Related work is presented author by author with exact reported numbers ("X et al. [n] proposed … achieving 87.32%").
- Numbered equations are introduced with "using Eq. (n)".
- Statistical testing of the improvement appears in the 2022 paper (H0 rejected).

**References**
- Springer basic numbered style, ordered alphabetically: `Author AB, Author CD (Year) Title. J Abbrev Vol(Issue):pages. https://doi.org/...`

**Deliberate deviations**
- The draft avoids the reference papers' habit of claiming "state-of-the-art" or "best results". Under our integrity rules those claims must wait for the data.

## 4. Literature matrix (verified)

Metadata for every reference in `main.tex` was checked against Crossref, the publisher page, ijcai.org or ACL Anthology. Most papers were read at abstract level only; exceptions are noted. *Read full text before final submission*, especially [Lu 2015], [Zheng 2019], [Fu 2023] and [Farhadi Sedehi 2024b].

| Theme | Key refs | Why it is cited |
|---|---|---|
| Baseline | Farhadi Sedehi 2024a (*IEEE Access* 12:118122–118140; full text read); 2024b (*Heliyon*, also hard pupil gate) | The method being modified |
| EEG features | Duan 2013 (DE); Zheng & Lu 2015; Murugappan 2010 (db4 DWT); Richman 2000; Bandt 2002; Żebrowski 2000; Rohila 2019; Jenke 2014; Peng 2005 (mRMR); Alarcão 2019 | Pipeline components |
| Evaluation validity | Li 2021 (*TPAMI*, block design); Khan 2025 and Kukhilava 2025 (arXiv **preprints**, labelled as such); Zheng 2019 cross-session drop | Leakage argument |
| Eye tracking / pupil | Bradley 2008; Partala 2003; Mathôt 2018; Raiturkar 2016 (luminance); Holmqvist 2012; Kret 2019; Lim 2020 | Uncertainty of the gating signal |
| EEG + ET fusion | Zheng 2014; Soleymani 2012 (×2); Lu 2015 (fuzzy integral, 87.59%); Zheng 2019; Liu 2022; Fu 2023 | Eye tracking is used as features/decisions, not temporal selection |
| Fuzzy / gating | Zadeh 1965; Takagi–Sugeno 1985; Wang–Mendel 1992; Jang 1993; Cho & Kim 1995; Matiko 2014; Kaya 2024; Jacobs 1991; Arevalo 2020; Kittler 1998; Nandakumar 2008; Lin & Wang 2002 | Positions fuzzy soft gating |
| Dataset | Hollenstein 2018 and 2020; Socher 2013; Pedroni 2019; Winkler 2011 | ZuCo |
| Statistics | Demšar 2006; McNemar 1947 | Paired comparison |

**Novelty search result**
- No prior work was found that uses fuzzy memberships of eye-tracking features to weight EEG windows for emotion recognition.
- Closest work:
  - the baseline group's own hard gate (2024a, 2024b);
  - Lu 2015, fuzzy-integral decision fusion;
  - attention-based fusion (Liu 2022, Fu 2023), whose weights are learned and not temporal.
- The search was not a formal systematic review, so the draft says "to the best of our knowledge".
- An arXiv 2025 cross-modal temporal-attention paper on SEED-IV (arXiv:2509.04938) came up but was not verified. Check it and distinguish our method from it before submission.

**Items to double-check**
- Lu 2015 page range: ijcai.org gives 1170–1175, other sources 1170–1176.
- Demšar 2006 has no DOI; JMLR 7:1–30.

## 5. Pending checklist (all marked in red in the PDF)

Search `main.tex` for `\pending` and `\revise`. There are 70 markers in total. The main groups:

**Dataset**
- [ ] Dataset confirmation (ZuCo vs SEED-IV)
- [ ] Usable subjects
- [ ] Reading-time distribution
- [ ] Windows per trial

**Preprocessing and windowing**
- [ ] Raw vs Automagic-preprocessed EEG
- [ ] Zero-phase filtering
- [ ] Notch frequency
- [ ] Window L and step S

**Features and selection**
- [ ] Entropy on reconstructed sub-bands
- [ ] PermEn normalisation
- [ ] mRMR per band or pooled
- [ ] Eye-tracking feature list

**Classifier and GA**
- [ ] t-norm
- [ ] Conflict-resolution score
- [ ] GA encoding, population, crossover/mutation, seeds and repetitions

**Gates**
- [ ] Hard-threshold rule τ
- [ ] Fuzzy MF shape and parameters (fixed or tuned)
- [ ] Consequents r_L, r_M, r_H
- [ ] Handling of low-quality eye-tracking windows
- [ ] One-input vs two-input gate
- [ ] Weights in training, at inference, or both
- [ ] Which ablations are run

**Environment and front matter**
- [ ] Hardware and software environment
- [ ] Repository link
- [ ] Authors, funding, conflict of interest, author contributions

## 6. Integrating results when the runs finish

For each run, save:
- per-fold, per-subject trial predictions as CSV: `subject, sentence_id, fold, y_true, y_pred_M0, y_pred_M1, S_c...`
- the gate weights per window
- timing logs

Every number in the paper can then be regenerated from those files.

Then:
1. Fill Tables 7–10 (main, bands, class-wise, protocols) and Fig. 3 (confusion matrices) from the CSVs. Keep P3 strictly as a leakage diagnostic.
2. Run the Wilcoxon test (per-subject macro-F1, P1) and McNemar (pooled trials). Report the median difference with a bootstrap 95% CI.
3. Rewrite the abstract's last sentence, the contribution 4 sentence, Sect. 7.9 Discussion, the limitations and the conclusion. Claim an improvement only if the effect and CI support it.
4. If the paper exceeds the page limit (currently 19 pp in 11-pt single column), port it to the conference template. Then:
   - shorten Sect. 2.4;
   - merge Tables 1 and 3 into an appendix;
   - drop the P2 table if space is tight.
