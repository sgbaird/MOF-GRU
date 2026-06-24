Question: I am doing Bayesian optimization / active learning over a *predefined candidate space* of metal-organic frameworks (MOFs). Each candidate is featurized with a frozen, pretrained bidirectional-GRU sequence encoder (MOF-GRU): the pooled hidden state is a 400-dimensional embedding (2 x hidden_size=200). Empirically, with a stationary Matern Gaussian process (with ARD length scales) fit on this *raw 400-d* embedding, EI-driven candidate selection barely beats random search, whereas a small deep-ensemble neural network head (the GRU's own Linear->ReLU->Linear regression head, retrained each round) used as the surrogate clearly wins. I want a high-effort, well-cited answer on how to make a *GP* (or a better-calibrated probabilistic surrogate) competitive over such a learned embedding, and more generally how to get a better optimization-ready embedding. Cover concretely: (1) Dimensionality reduction of the learned embedding before GP modeling -- when unsupervised PCA helps vs. not (my intuition is that the 400 GRU hidden units may carry roughly equal importance so PCA wouldn't compress much; is that right, and how to check via the PCA explained-variance spectrum?), and *supervised* alternatives (PLS regression, supervised PCA, linear discriminant / Fisher directions, neighborhood-components / metric learning, or learning a low-rank projection jointly with the property). (2) Why high-dimensional stationary GPs with ARD are hard to fit from tens of labels (curse of dimensionality, non-identifiable length scales) and the standard fixes: deep-kernel learning (DKL, Wilson et al.), GP on a learned feature extractor, additive/low-dimensional-structure GPs (ADD-GP, SAASBO sparse-axis-aligned priors, Eriksson & Jankowiak), random-feature / linear surrogates, and Bayesian neural networks / deep ensembles as drop-in probabilistic surrogates (compare calibration & sample efficiency to GPs). (3) Combining the learned embedding with hand-engineered global descriptors (pore geometry, density, porosity, surface area): concatenation, separate kernels (sum/product of kernels over each feature block), and feature selection. (4) Retraining the encoder with a *smaller* bottleneck embedding dimension (e.g. 8-64) so the surrogate lives in a compact space -- trade-offs vs. the full 400-d representation, and whether a bottleneck regularizes the optimization landscape. (5) Training a VAE or other ML-based embedding (beta-VAE, sequence VAE over the SELFIES token vocabulary, contrastive/SimCLR-style or self-supervised embeddings, or a jointly trained supervised autoencoder) to get a smooth, low-dimensional, optimization-friendly latent space, and how that compares to just reducing the existing embedding. (6) Practical recommendations and a ranked, actionable plan specifically for improving GP-based candidate-space BO over a frozen sequence-model embedding when only tens-to-hundreds of labels are available. Provide citations throughout.

# Making GP-Based Bayesian Optimization Competitive over Learned MOF Embeddings: A Comprehensive Guide

## Overview

The scenario described—a stationary Matérn GP with ARD length scales fit on a raw 400-dimensional bidirectional-GRU embedding barely beating random search, while a small deep-ensemble neural head clearly wins—is a textbook manifestation of the curse of dimensionality for Gaussian process surrogates. This document systematically addresses six interrelated strategies for closing the gap, drawing on the high-dimensional Bayesian optimization, surrogate modeling, and materials informatics literatures.

---

## 1. Dimensionality Reduction of the Learned Embedding Before GP Modeling

### 1.1 When Unsupervised PCA Helps vs. Not

PCA is the most common unsupervised dimensionality reduction method applied before surrogate modeling, and has been widely combined with Gaussian processes and Kriging for high-dimensional problems (hou2022dimensionalityreductionin pages 15-16, hou2022dimensionalityreductionin pages 3-4). The central diagnostic is the **explained-variance spectrum**: plot the cumulative fraction of total variance captured by the top *k* principal components. If the spectrum drops rapidly—say, 90% of variance in the first 20–40 components—PCA can dramatically compress the representation without losing much signal, and fitting a GP on those top components will be far more tractable than on the full 400 dimensions.

Your intuition that a bidirectional-GRU's 400 pooled hidden units may carry roughly equal importance is plausible. GRU/LSTM architectures typically do not impose orthogonality or decorrelation on their hidden states; different units often encode overlapping, distributed features. If the PCA spectrum is flat (no sharp elbow), this confirms that variance is spread across many dimensions and unsupervised PCA will compress poorly—removing any principal component discards a nontrivial fraction of information. **How to check**: compute the eigenvalues of the covariance matrix of your 400-d embeddings across your candidate pool, plot the cumulative explained variance ratio, and look for an elbow. If the first 30 components explain less than ~70% of variance, PCA is unlikely to help much for GP fitting (hou2022dimensionalityreductionin pages 3-4).

### 1.2 Supervised Alternatives

When unsupervised PCA fails, **supervised dimensionality reduction** is often more effective because it identifies the directions that matter for the *property*, not for total variance. Supervised methods produce more suitable topology representations of input-output maps than unsupervised methods in surrogate modeling contexts (hou2022dimensionalityreductionin pages 3-4).

- **Partial Least Squares (PLS)**: PLS finds directions in the input space that have maximum covariance with the response variable, making it directly label-aware. KPLS (Kriging with PLS) has been shown to enable faster convergence than standard high-dimensional Kriging by reducing the number of hyperparameters through PLS preprocessing (hou2022dimensionalityreductionin pages 16-18). PLS is particularly attractive when you have multicollinear predictors (as in learned embeddings) and few labeled observations (hou2022dimensionalityreductionin pages 5-7).

- **Supervised PCA / Active Subspace Methods**: Rather than maximizing variance, these methods learn projection directions that maximize predictive power. The active subspace methodology uses gradient information (or GP-based estimation thereof) to identify the low-dimensional subspace in which the function varies most, and can be combined with GP surrogates (binois2022asurveyon pages 15-18).

- **Metric Learning / Neighborhood Components Analysis**: These methods learn a distance function (Mahalanobis metric or nonlinear mapping) such that points with similar property values are closer in the projected space. Deep kernel learning (discussed below) can be viewed as a neural form of metric learning for GP modeling (wilson2016deepkernellearning pages 14-17).

- **Learning a Low-Rank Projection Jointly with the Property**: The approach of Lataniotis et al. combines dimensionality reduction and surrogate modeling in a joint, data-driven framework (DRSM), enabling supervised kernel PCA coupled with Kriging to handle problems up to O(10⁴) input dimensions (lataniotis2020extendingclassicalsurrogate pages 34-37, lataniotis2020extendingclassicalsurrogate pages 37-39). In the BO context, Moriconi et al. learn a low-dimensional feature space jointly with the GP response surface and a reconstruction mapping, achieving better compression than linear methods (moriconi2020highdimensionalbayesianoptimization pages 16-18).

---

## 2. Why High-Dimensional Stationary GPs with ARD Fail and the Standard Fixes

### 2.1 The Core Problem

Fitting a stationary GP with ARD in 400 dimensions from tens of labels suffers from several compounding issues:

1. **Distance concentration**: In high dimensions, pairwise distances between points concentrate around a narrow band, making it difficult for covariance-based kernels to distinguish nearby from far-away points (binois2022asurveyon pages 8-11).

2. **Non-identifiable length scales**: With *d* = 400 ARD length-scale parameters but only tens of observations, the marginal likelihood landscape is extremely flat with many local optima, making hyperparameter optimization unreliable. The number of hyperparameters grows linearly with *d*, but the information available from *n* ≪ *d* observations is grossly insufficient (binois2022asurveyon pages 8-11).

3. **Gradient vanishing**: For the Squared Exponential (SE) kernel specifically, Xu et al. showed that the gradient factor ρ²/exp(ρ²) falls below machine epsilon as dimensionality increases, preventing effective length-scale learning. For commonly used initializations, this probability exceeds 0.99 at *d* ≥ 205 (xu2402standardgaussianprocess pages 2-4, xu2402standardgaussianprocess pages 4-6). Matérn kernels are more robust because their gradient factor decays more slowly (xu2402standardgaussianprocess pages 4-6).

4. **Acquisition function optimization**: The acquisition function itself becomes difficult to optimize in high dimensions due to multi-modal landscapes and concentration of volume at hypercube boundaries (binois2022asurveyon pages 8-11).

### 2.2 SAASBO: Sparse Axis-Aligned Subspaces

SAASBO (Eriksson & Jankowiak, 2021) is one of the most directly relevant fixes for your scenario. It places a half-Cauchy (horseshoe-like) sparsity-inducing prior on the inverse length-scale parameters, encouraging most dimensions to be "turned off" (kudva2024efficientperformancebasedmpc pages 2-3, eriksson2021highdimensionalbayesianoptimization pages 1-2). The method uses fully Bayesian inference via Hamiltonian Monte Carlo, which rapidly identifies the most relevant low-dimensional subspace from even small datasets. SAASBO balances flexibility against parsimony: by assuming only a sparse subset of coordinates matter, it dramatically reduces the effective parameter count and enables sample-efficient BO in problems with hundreds of input dimensions (eriksson2021highdimensionalbayesianoptimization pages 4-5). For a frozen embedding where only a subset of GRU hidden units may carry property-relevant signal, SAASBO's sparse prior is an excellent structural match.

### 2.3 Deep Kernel Learning (DKL)

Wilson et al.'s DKL transforms inputs through a deep neural network before applying a base GP kernel, jointly training all parameters via the GP marginal likelihood (wilson2016deepkernellearning pages 1-3, wilson2016deepkernellearning pages 6-9). DKL learns an adaptive, non-Euclidean metric that overcomes the limitations of stationary kernels in high dimensions (wilson2016deepkernellearning pages 14-17). In materials design, Kiyohara & Kumagai showed DKL's performance was relatively robust regardless of feature count, unlike standard GPs which heavily depended on feature selection—a critical advantage when optimizing hyperparameters with small datasets (kiyohara2025bayesianoptimizationwith pages 6-6, kiyohara2025bayesianoptimizationwith pages 1-2). However, DKL can overfit with noisy, small datasets: pretraining can be detrimental on experimental data, and larger weight decay values are needed (kiyohara2025bayesianoptimizationwith pages 4-6). In your setup, you could use your existing GRU-to-linear head as the DKL feature map, replacing the final regression with a GP.

### 2.4 Additive and Low-Dimensional-Structure GPs

Additive GPs decompose the objective as a sum of low-dimensional functions, enabling each component to be modeled independently. ADD-GP-UCB optimizes each component separately, avoiding the full curse of dimensionality (malu2021bayesianoptimizationin pages 4-5). Random embedding methods like REMBO project the search space to a lower-dimensional subspace using random matrices, assuming the objective has low effective dimensionality (malu2021bayesianoptimizationin pages 4-5, binois2022asurveyon pages 15-18).

### 2.5 Random-Feature / Linear Surrogates

Snoek et al. proposed DNGO, which uses neural networks to learn adaptive basis functions for Bayesian linear regression, achieving competitive performance with GP-based approaches while scaling linearly with data (snoek2015scalablebayesianoptimization pages 1-2). Random Fourier features provide efficient kernel approximation for handling high-dimensional inputs (malu2021bayesianoptimizationin pages 4-5).

### 2.6 Deep Ensembles and BNNs as Drop-In Surrogates

Li et al. (2024) conducted the most comprehensive comparison of BNN surrogates versus GPs for BO. Key findings include: (i) HMC-based BNNs are the most successful fully stochastic BNN inference procedure; (ii) deep ensembles perform surprisingly poorly for BO despite their success elsewhere—they show plateaued rewards and fail at effective space exploration in small-data BO; (iii) infinite-width BNNs (I-BNNs) are particularly promising in high dimensions, consistently outperforming other surrogates in high-dimensional settings; (iv) a hybrid approach using HMC for mean estimates and GP for uncertainty estimates performs best overall; (v) deep kernel learning is relatively competitive as a partially stochastic model (li2305astudyof pages 8-10, li2305astudyof pages 25-27, li2305astudyof pages 1-2, li2305astudyof pages 5-7). Your observation that a small deep-ensemble head wins over the vanilla GP is consistent with the finding that neural surrogates have advantages for representation learning in high-dimensional settings, though their uncertainty may be less well-calibrated than GPs in the small-data regime (li2305astudyof pages 2-4).

The following table compares the main surrogate strategies:

| Strategy | Dimensionality Handled | Data Efficiency (few labels) | Calibration Quality | Implementation Complexity | Key References |
|---|---|---|---|---|---|
| Standard GP with ARD on raw 400-d embedding | Poor in very high ambient dimension unless the task is truly smooth and labels are ample; ARD length scales become hard to identify and distance concentration hurts kernel learning | Low | High in principle, but often degraded in practice when hyperparameters are weakly identified | Low | (binois2022asurveyon pages 8-11, xu2402standardgaussianprocess pages 2-4, xu2402standardgaussianprocess pages 4-6) |
| PCA + GP | Moderate if most signal lies in a low-variance-rank subspace; weak if predictive signal is not aligned with top unsupervised variance directions | Low-to-Medium | Medium-to-High | Low | (hou2022dimensionalityreductionin pages 15-16, lataniotis2020extendingclassicalsurrogate pages 34-37, hou2022dimensionalityreductionin pages 3-4) |
| Supervised DR (PLS) + GP | Moderate-to-Good when only a few latent directions are property-relevant; better than PCA when label information is precious | Medium-to-High | Medium-to-High | Medium | (hou2022dimensionalityreductionin pages 16-18, hou2022dimensionalityreductionin pages 3-4, lataniotis2020extendingclassicalsurrogate pages 34-37) |
| SAASBO sparse-axis-aligned GP | Good for hundreds of inputs if only a small subset of coordinates matter; especially attractive for frozen embeddings with sparse relevance structure | High | High | Medium-to-High | (kudva2024efficientperformancebasedmpc pages 2-3, eriksson2021highdimensionalbayesianoptimization pages 1-2, eriksson2021highdimensionalbayesianoptimization pages 4-5) |
| Deep Kernel Learning | Good for high-dimensional learned or raw inputs when a neural feature map can reshape geometry before the GP; often stronger than vanilla GP in materials tasks | Medium | Medium-to-High | High | (wilson2016deepkernellearning pages 1-3, wilson2016deepkernellearning pages 14-17, kiyohara2025bayesianoptimizationwith pages 1-2, kiyohara2025bayesianoptimizationwith pages 6-6, kiyohara2025bayesianoptimizationwith pages 4-6) |
| Deep ensemble / BNN surrogate | Good to Excellent in high dimensions and nonstationary settings; often more robust than vanilla GP on learned embeddings, but results are problem dependent | Medium | Medium for deep ensembles; Medium-to-High for stronger BNN inference; often below GP for uncertainty quality on small data | Medium-to-High | (li2305astudyof pages 8-10, li2305astudyof pages 25-27, li2305astudyof pages 1-2, snoek2015scalablebayesianoptimization pages 1-2) |
| GP on retrained bottleneck embedding (8-64d) | Good if the encoder is retrained so task-relevant information is compressed into a compact latent space; poor if bottleneck is too aggressive | Medium-to-High | High | Medium-to-High | (gomezbombarelli2018automaticchemicaldesign pages 7-8, maus2022locallatentspace pages 1-3, maus2022locallatentspace pages 9-11, talibart2025learningachemistryaware pages 18-21) |
| GP on VAE latent space | Good if the latent space is smooth, valid, and aligned with the objective; can fail when BO explores dead regions or off-manifold areas | Medium | Medium | High | (griffiths2020constrainedbayesianoptimization pages 2-3, gomezbombarelli2018automaticchemicaldesign pages 7-8, maus2022locallatentspace pages 1-3, moriconi2020highdimensionalbayesianoptimization pages 16-18) |
| Kernel composition (learned embedding + handcrafted descriptors) | Good when learned sequence features and global physical descriptors capture complementary structure; often safer than choosing one block alone | Medium-to-High | High | Medium | (kiyohara2025bayesianoptimizationwith pages 6-6, han2026machinelearningguideddesignof pages 12-13, gomezgualdron2026machinelearningto pages 8-9, schultz2026graphneuralnetworkbased pages 6-7) |


*Table: This table compares practical surrogate options for Bayesian optimization over high-dimensional learned embeddings, emphasizing their behavior with tens to hundreds of labels. It is useful for prioritizing which GP variants or alternatives are most likely to outperform a vanilla ARD GP on a raw 400-dimensional embedding.*

---

## 3. Combining the Learned Embedding with Hand-Engineered MOF Descriptors

MOF property prediction benefits from both learned representations and physically interpretable descriptors. For gas adsorption and related properties, global geometric descriptors—void fraction, volumetric surface area, largest included/free sphere diameters, pore limiting diameter, and density—are among the most predictive features in classical ML models (gomezgualdron2026machinelearningto pages 8-9, schultz2026graphneuralnetworkbased pages 6-7, park2024fromdatato pages 2-4).

Three strategies for combining these with the 400-d GRU embedding:

**Concatenation**: The simplest approach: z-score both feature blocks and concatenate them into a single input vector for the GP. This works well when the GP can learn differential length scales, but with few labels, the additional dimensions may hurt unless feature selection is applied. Feature selection via filter methods (correlation screening), wrapper methods (recursive feature elimination), or embedded methods (Random Forest importance, SHAP values) can reduce redundancy while preserving key structure-property relationships (han2026machinelearningguideddesignof pages 12-13).

**Additive kernel (sum of kernels)**: Define k = k₁(x_GRU, x'_GRU) + k₂(x_geom, x'_geom), where k₁ operates on the learned embedding and k₂ operates on the hand-engineered descriptors. This is appropriate when the property decomposes additively into contributions from sequence-level features and global geometry. Each kernel block can have its own hyperparameters and dimensionality, and can be independently regularized.

**Product kernel**: k = k₁(x_GRU, x'_GRU) × k₂(x_geom, x'_geom) encodes interactions between the feature blocks. This is more expressive but harder to fit from few labels.

In materials BO, Kiyohara & Kumagai demonstrated that DKL's performance was robust to feature count, while standard GP performance heavily depended on feature selection—making the addition of strongly correlated physical descriptors particularly valuable for standard GPs (kiyohara2025bayesianoptimizationwith pages 6-6).

---

## 4. Retraining the Encoder with a Smaller Bottleneck Embedding

### 4.1 Rationale

If you have access to the GRU encoder weights and sufficient MOF data, retraining with a smaller bottleneck layer (e.g., 8–64 dimensions rather than 400) is one of the most powerful ways to improve the GP surrogate's statistical efficiency. The fundamental tension in latent-space BO is that small latent spaces limit molecule/material representation capacity, while large ones complicate surrogate modeling (maus2022locallatentspace pages 9-11). Maus et al. demonstrated that jointly training the latent space with the surrogate significantly outperforms fixed pretrained spaces across molecular benchmarks (maus2022locallatentspace pages 9-11).

### 4.2 Trade-offs

A smaller bottleneck acts as an **information bottleneck**: it forces the encoder to compress away task-irrelevant variation, retaining only the features most predictive of downstream properties. This regularizes the optimization landscape by making GP distances more meaningful and reducing the number of ARD parameters. However, if the bottleneck is too aggressive, it may discard chemistry/topology information needed for accurate property prediction—particularly problematic if you later want to predict different properties with the same embedding.

### 4.3 Practical Recommendations

- Start with bottleneck sizes of 16, 32, and 64 and monitor both reconstruction quality and downstream GP prediction accuracy.
- Consider jointly training the bottleneck projection with the property loss (semi-supervised or supervised autoencoder), so the compressed representation is explicitly optimized for the property of interest (gomezbombarelli2018automaticchemicaldesign pages 7-8, maus2022locallatentspace pages 1-3).
- Griffiths & Hernández-Lobato showed that "dead regions" in VAE latent spaces arise partly from artificially high latent dimensionality; smaller latent spaces mitigate this (griffiths2020constrainedbayesianoptimization pages 2-3).

---

## 5. Training a VAE or Other ML-Based Embedding for an Optimization-Friendly Latent Space

### 5.1 VAE-Based Latent Space Optimization

The seminal work of Gómez-Bombarelli et al. demonstrated that training a VAE over molecular strings creates a continuous latent space where GP-driven Bayesian optimization can efficiently search for molecules with desired properties (gomezbombarelli2018automaticchemicaldesign pages 7-8). The encoder maps discrete structures to continuous vectors, and the decoder maps back; a property predictor is jointly trained on the latent representation. Griffiths & Hernández-Lobato extended this with constrained BO to handle the problem of dead zones—regions of latent space that decode to no valid molecule—and showed that enforcing latent space validity constraints substantially improves optimization outcomes (griffiths2020constrainedbayesianoptimization pages 2-3).

### 5.2 β-VAE and Disentangled Representations

Increasing the KL-divergence weight (β > 1) in a β-VAE encourages more disentangled, axis-aligned latent representations at the cost of reconstruction quality. For BO, disentangled latent spaces are advantageous because they align with the assumptions of ARD and SAASBO-style sparsity priors, making GP fitting more tractable.

### 5.3 SELFIES-Based Sequence VAEs

SELFIES (Self-Referencing Embedded Strings) provide 100% chemical validity, eliminating the problem of decoding to invalid structures. LOL-BO (Maus et al.) demonstrated that a SELFIES-based transformer VAE combined with local latent-space BO achieved up to 22× improvement over existing latent space BO methods on molecular design benchmarks, and that joint training of the VAE with a sparse GP surrogate was critical (maus2022locallatentspace pages 1-3, maus2022locallatentspace pages 9-11). For MOFs, SELFIES-like text representations (e.g., MOFid combining SMILES of building blocks with topology codes) have been used with transformer-based encoders like MOFormer, which can be pretrained on >400K structures in a self-supervised manner (wang2023selfsupervisedrepresentationlearning pages 122-125).

### 5.4 Contrastive / SimCLR-Style Self-Supervised Embeddings

Contrastive learning has emerged as a powerful paradigm for learning molecular and materials representations without labels. MolCLR (Wang et al.) pre-trained on ~10 million unlabeled molecules using graph augmentations significantly improved GNN performance on both classification and regression tasks (wang2023selfsupervisedrepresentationlearning pages 23-28). Crystal Twins adapted Barlow Twins and SimSiam frameworks for crystalline materials, demonstrating that self-supervised representations better separate materials by their properties in t-SNE projections than supervised baselines (magar2022crystaltwinsselfsupervised pages 1-4, magar2022crystaltwinsselfsupervised pages 11-16). PolyCL applied contrastive learning to polymer representations, achieving competitive performance as a feature extractor for transfer learning (zhou2025polyclcontrastivelearning pages 1-4). For MOFs specifically, MOFormer-style self-supervised pretraining on large unlabeled datasets improves data efficiency and generalization for downstream property prediction (wang2023selfsupervisedrepresentationlearning pages 138-143, wang2023selfsupervisedrepresentationlearning pages 122-125).

### 5.5 Chemistry-Aware Latent Spaces

Talibart & Gilis introduced a SELFIES Transformer VAE with a novel loss explicitly enforcing latent space distances to reflect Tanimoto similarities between molecular fingerprints, producing a smooth, well-structured latent space that enables meaningful exploration and interpolation (talibart2025learningachemistryaware pages 18-21). This "chemistry-aware" approach to latent space construction is especially relevant for BO, where the smoothness and semantic coherence of the representation directly impacts surrogate model quality.

### 5.6 Comparison with Reducing the Existing Embedding

Training a new embedding (VAE, contrastive, or supervised autoencoder) is more powerful but more expensive than simply applying PCA or PLS to the existing GRU embedding. The key advantage is that the new embedding can be explicitly optimized for properties relevant to BO: smoothness, label-relevance, validity, and compactness. The key disadvantage is the engineering overhead and the risk of overfitting the embedding to a small labeled dataset. For a predefined candidate set (where you do not need decoding), a simpler approach—PLS or SAASBO on the frozen embedding with added physical descriptors—may yield most of the gain at much lower cost.

---

## 6. Practical Recommendations: A Ranked, Actionable Plan

The following table provides a prioritized roadmap for improving GP-based candidate-space BO over the frozen 400-d MOF-GRU embedding:

| Priority Rank (1 = do first) | Action | Expected Gain | Effort Level | Notes/Rationale |
|---|---|---|---|---|
| 1 | Diagnose the embedding before changing the BO stack: standardize the 400-d MOF-GRU embedding, inspect the PCA explained-variance spectrum, and benchmark GP on PCA-reduced spaces (e.g., 8, 16, 32, 64 dims) plus supervised PLS projections | Medium | Low | If the cumulative explained variance rises quickly, unsupervised PCA may remove nuisance variance and stabilize GP fitting; if the spectrum is flat, that supports your intuition that variance is spread across many units, so PCA may not help much. Because PCA ignores labels, also test PLS or related supervised DR, which is often better when only a few directions matter for the property rather than for total variance (hou2022dimensionalityreductionin pages 15-16, hou2022dimensionalityreductionin pages 16-18, hou2022dimensionalityreductionin pages 3-4) |
| 2 | Use a Matérn kernel rather than an SE/RBF kernel, with careful length-scale initialization and strong input normalization | Medium | Low | High-dimensional GP failures are often partly kernel/initialization failures. Recent analysis shows SE kernels can suffer gradient-vanishing pathologies in high dimension, while Matérn kernels are more robust; even when you already use Matérn, good scaling and initialization still matter for stable hyperparameter learning from few labels (xu2402standardgaussianprocess pages 2-4, xu2402standardgaussianprocess pages 4-6, binois2022asurveyon pages 8-11) |
| 3 | Replace vanilla ARD-GP with SAASBO / SAAS-GP on the frozen embedding | High | Medium | With tens to hundreds of labels in 400 dimensions, ARD length scales are weakly identified. SAAS imposes sparse priors on inverse length scales, effectively assuming only a small subset of embedding coordinates matter for the property. This is one of the most directly relevant GP fixes for your exact regime: high ambient dimension, few labels, candidate-space BO (kudva2024efficientperformancebasedmpc pages 2-3, eriksson2021highdimensionalbayesianoptimization pages 1-2, eriksson2021highdimensionalbayesianoptimization pages 4-5) |
| 4 | Add hand-engineered MOF descriptors (density, void fraction, surface area, pore diameters, porosity) and compare concatenation versus separate kernel blocks | High | Medium | Learned sequence embeddings and global physical descriptors are often complementary in MOFs. Start with concatenation after z-scoring; then try additive kernels such as k = k_GRU + k_geom and optionally product forms if you believe interactions matter. Physical descriptors can rescue signal that a sequence encoder does not preserve cleanly, and standard GPs benefit when informative descriptors are directly available (gomezgualdron2026machinelearningto pages 8-9, schultz2026graphneuralnetworkbased pages 6-7, kiyohara2025bayesianoptimizationwith pages 6-6, han2026machinelearningguideddesignof pages 12-13) |
| 5 | Try feature selection / relevance pruning on the combined feature set before GP fitting | Medium | Low-to-Medium | If you concatenate 400 learned features with engineered descriptors, do not assume the GP can sort everything out from few labels. Use simple filters, sparsity-inducing linear models, recursive feature elimination, or SHAP/RF-based screening on an auxiliary predictor to shrink the search space before the GP stage (han2026machinelearningguideddesignof pages 12-13, kiyohara2025bayesianoptimizationwith pages 6-6) |
| 6 | Try deep kernel learning: a small MLP or your GRU head as a feature map feeding a GP | High | High | DKL is often the most principled way to make a GP competitive on a learned embedding: instead of forcing stationarity in raw 400-d Euclidean space, it learns a task-shaped representation and places the GP on top. In materials BO, DKL often matches or outperforms standard GPs when strong manual descriptors are not already available, though it can overfit noisy small datasets and needs regularization (wilson2016deepkernellearning pages 1-3, wilson2016deepkernellearning pages 14-17, kiyohara2025bayesianoptimizationwith pages 1-2, kiyohara2025bayesianoptimizationwith pages 6-6, kiyohara2025bayesianoptimizationwith pages 4-6) |
| 7 | Retrain or fine-tune the encoder to output a much smaller bottleneck embedding (e.g., 8-64 dims) jointly with the property head | High | High | If BO lives in a compact latent space, the GP has a much easier statistical problem. A smaller bottleneck can regularize away property-irrelevant variation and make distances more meaningful, but too small a bottleneck can discard critical chemistry/topology information. This is best viewed as learning an optimization-ready representation rather than post hoc compressing a fixed one (gomezbombarelli2018automaticchemicaldesign pages 7-8, maus2022locallatentspace pages 1-3, maus2022locallatentspace pages 9-11, talibart2025learningachemistryaware pages 18-21) |
| 8 | If you have abundant unlabeled MOFs, retrain the representation with self-supervision or contrastive learning, then fit the surrogate on the new embedding | Medium-to-High | High | Self-supervised representation learning has improved data efficiency and transferability in molecules and crystals, and MOF-specific language/sequence models like MOFormer show the value of large-scale pretraining. Contrastive or self-supervised retraining can produce embeddings whose neighborhoods better reflect chemical or structural similarity relevant to downstream prediction and BO (magar2022crystaltwinsselfsupervised pages 1-4, magar2022crystaltwinsselfsupervised pages 11-16, wang2023selfsupervisedrepresentationlearning pages 1-9, wang2023selfsupervisedrepresentationlearning pages 138-143, wang2023selfsupervisedrepresentationlearning pages 122-125) |
| 9 | If inverse design or latent-space smoothness matters, consider a VAE / supervised autoencoder / constrained latent BO route rather than only reducing the existing embedding | Medium | High | A learned latent space can be more optimization-friendly than a frozen pooled GRU state, but only if validity and manifold structure are enforced. Prior work shows naive VAE latents can contain dead regions; constrained BO and joint training of the latent model with the surrogate improve this substantially. For a predefined candidate set, this is less urgent than SAAS/DKL, but it is a strong medium-term representation fix (griffiths2020constrainedbayesianoptimization pages 2-3, gomezbombarelli2018automaticchemicaldesign pages 7-8, maus2022locallatentspace pages 1-3, maus2022locallatentspace pages 9-11) |
| 10 | If GP variants still do not beat your neural surrogate, use a calibrated deep ensemble / stronger BNN as the production surrogate and keep the GP as a baseline | High | Low-to-Medium | In practice, representation-learning surrogates often win on high-dimensional learned embeddings. Comparative BO studies find standard GPs usually provide better uncertainty quality in small data, but BNN-style surrogates and some neural methods can be superior in high-dimensional or nonstationary problems; deep ensembles are often strong practical baselines even if they are not always the best-calibrated option (li2305astudyof pages 8-10, li2305astudyof pages 25-27, li2305astudyof pages 1-2, snoek2015scalablebayesianoptimization pages 1-2) |


*Table: This table gives a prioritized, practical roadmap for making GP-based Bayesian optimization more competitive over a frozen 400-dimensional MOF-GRU embedding with only tens to hundreds of labels. It emphasizes the highest-leverage fixes first: diagnosing the embedding, using more robust GP priors and kernels, adding physical MOF descriptors, and only then moving to representation retraining or neural surrogates.*

### Summary of Key Recommendations

**Immediate (Low-Effort, High-Diagnostic Value)**:
1. Inspect the PCA explained-variance spectrum of the 400-d embedding; if flat, unsupervised PCA will not help and supervised reduction (PLS) is preferred.
2. Ensure you are using a Matérn-5/2 (not SE/RBF) kernel with proper normalization and initialization, as SE kernels suffer from gradient vanishing at d ≥ 200 (xu2402standardgaussianprocess pages 2-4, xu2402standardgaussianprocess pages 4-6).

**Short-Term (Moderate Effort, High Expected Gain)**:
3. Apply SAASBO sparse priors to the GP, which is the single most impactful change for a stationary GP in 400 dimensions with few labels (eriksson2021highdimensionalbayesianoptimization pages 1-2, eriksson2021highdimensionalbayesianoptimization pages 4-5).
4. Concatenate hand-engineered MOF geometric descriptors (void fraction, surface area, pore diameters) with the GRU embedding using additive kernel composition (gomezgualdron2026machinelearningto pages 8-9, kiyohara2025bayesianoptimizationwith pages 6-6).

**Medium-Term (High Effort, High Expected Gain)**:
5. Implement deep kernel learning using your existing GRU head architecture as the feature map (wilson2016deepkernellearning pages 1-3, kiyohara2025bayesianoptimizationwith pages 1-2, kiyohara2025bayesianoptimizationwith pages 6-6).
6. Retrain the encoder with a smaller bottleneck (16–64 dims), ideally jointly with the property prediction loss (maus2022locallatentspace pages 1-3, maus2022locallatentspace pages 9-11).

**Longer-Term (Highest Effort, Potentially Transformative)**:
7. If abundant unlabeled MOFs are available, retrain the representation using contrastive or self-supervised learning for a more optimization-friendly embedding (magar2022crystaltwinsselfsupervised pages 1-4, wang2023selfsupervisedrepresentationlearning pages 138-143, wang2023selfsupervisedrepresentationlearning pages 122-125).
8. If the GP still cannot match the deep ensemble, accept the ensemble as the production surrogate and use the GP primarily for its uncertainty calibration, potentially in a hybrid scheme (li2305astudyof pages 8-10, li2305astudyof pages 25-27).

The most impactful single change for your exact problem—a frozen 400-d embedding with tens-to-hundreds of labels—is likely **SAASBO with added physical MOF descriptors via kernel composition**. This directly addresses the curse of dimensionality in the GP without requiring encoder retraining, and physical descriptors for MOFs (porosity, surface area, etc.) are cheap to compute and often among the strongest univariate predictors for adsorption and related properties (gomezgualdron2026machinelearningto pages 8-9, gomezgualdron2026machinelearningto pages 15-15).

References

1. (hou2022dimensionalityreductionin pages 15-16): Chun Kit Jeffery Hou and Kamran Behdinan. Dimensionality reduction in surrogate modeling: a review of combined methods. Data Science and Engineering, 7:402-427, Aug 2022. URL: https://doi.org/10.1007/s41019-022-00193-5, doi:10.1007/s41019-022-00193-5. This article has 148 citations and is from a peer-reviewed journal.

2. (hou2022dimensionalityreductionin pages 3-4): Chun Kit Jeffery Hou and Kamran Behdinan. Dimensionality reduction in surrogate modeling: a review of combined methods. Data Science and Engineering, 7:402-427, Aug 2022. URL: https://doi.org/10.1007/s41019-022-00193-5, doi:10.1007/s41019-022-00193-5. This article has 148 citations and is from a peer-reviewed journal.

3. (hou2022dimensionalityreductionin pages 16-18): Chun Kit Jeffery Hou and Kamran Behdinan. Dimensionality reduction in surrogate modeling: a review of combined methods. Data Science and Engineering, 7:402-427, Aug 2022. URL: https://doi.org/10.1007/s41019-022-00193-5, doi:10.1007/s41019-022-00193-5. This article has 148 citations and is from a peer-reviewed journal.

4. (hou2022dimensionalityreductionin pages 5-7): Chun Kit Jeffery Hou and Kamran Behdinan. Dimensionality reduction in surrogate modeling: a review of combined methods. Data Science and Engineering, 7:402-427, Aug 2022. URL: https://doi.org/10.1007/s41019-022-00193-5, doi:10.1007/s41019-022-00193-5. This article has 148 citations and is from a peer-reviewed journal.

5. (binois2022asurveyon pages 15-18): Mickaël Binois and Nathan Wycoff. A survey on high-dimensional gaussian process modeling with application to bayesian optimization. ACM Transactions on Evolutionary Learning and Optimization, 2:1-26, Jun 2022. URL: https://doi.org/10.1145/3545611, doi:10.1145/3545611. This article has 300 citations.

6. (wilson2016deepkernellearning pages 14-17): Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P. Xing. Deep kernel learning. Preprint, Jan 2016. URL: https://doi.org/10.48550/arxiv.1511.02222, doi:10.48550/arxiv.1511.02222. This article has 1469 citations.

7. (lataniotis2020extendingclassicalsurrogate pages 34-37): Christos Lataniotis, Stefano Marelli, and Bruno Sudret. Extending classical surrogate modeling to high dimensions through supervised dimensionality reduction: a data-driven approach. Jan 2020. URL: https://doi.org/10.1615/int.j.uncertaintyquantification.2020031935, doi:10.1615/int.j.uncertaintyquantification.2020031935. This article has 116 citations and is from a peer-reviewed journal.

8. (lataniotis2020extendingclassicalsurrogate pages 37-39): Christos Lataniotis, Stefano Marelli, and Bruno Sudret. Extending classical surrogate modeling to high dimensions through supervised dimensionality reduction: a data-driven approach. Jan 2020. URL: https://doi.org/10.1615/int.j.uncertaintyquantification.2020031935, doi:10.1615/int.j.uncertaintyquantification.2020031935. This article has 116 citations and is from a peer-reviewed journal.

9. (moriconi2020highdimensionalbayesianoptimization pages 16-18): Riccardo Moriconi, Marc Peter Deisenroth, and K. S. Sesh Kumar. High-dimensional bayesian optimization using low-dimensional feature spaces. Machine Learning, 109:1925-1943, Sep 2020. URL: https://doi.org/10.1007/s10994-020-05899-z, doi:10.1007/s10994-020-05899-z. This article has 251 citations and is from a highest quality peer-reviewed journal.

10. (binois2022asurveyon pages 8-11): Mickaël Binois and Nathan Wycoff. A survey on high-dimensional gaussian process modeling with application to bayesian optimization. ACM Transactions on Evolutionary Learning and Optimization, 2:1-26, Jun 2022. URL: https://doi.org/10.1145/3545611, doi:10.1145/3545611. This article has 300 citations.

11. (xu2402standardgaussianprocess pages 2-4): Zhitong Xu, Haitao Wang, Jeff M Phillips, and Shandian Zhe. Standard gaussian process is all you need for high-dimensional bayesian optimization. ArXiv, Feb 2025. URL: https://doi.org/10.48550/arxiv.2402.02746, doi:10.48550/arxiv.2402.02746. This article has 51 citations.

12. (xu2402standardgaussianprocess pages 4-6): Zhitong Xu, Haitao Wang, Jeff M Phillips, and Shandian Zhe. Standard gaussian process is all you need for high-dimensional bayesian optimization. ArXiv, Feb 2025. URL: https://doi.org/10.48550/arxiv.2402.02746, doi:10.48550/arxiv.2402.02746. This article has 51 citations.

13. (kudva2024efficientperformancebasedmpc pages 2-3): Akshay Kudva, Melanie T. Huynh, Ali Mesbah, and Joel A. Paulson. Efficient performance-based mpc tuning in high dimensions using bayesian optimization over sparse subspaces. IFAC-PapersOnLine, 58:458-463, Jan 2024. URL: https://doi.org/10.1016/j.ifacol.2024.08.379, doi:10.1016/j.ifacol.2024.08.379. This article has 13 citations and is from a peer-reviewed journal.

14. (eriksson2021highdimensionalbayesianoptimization pages 1-2): David Eriksson and Martin Jankowiak. High-dimensional bayesian optimization with sparse axis-aligned subspaces. Preprint, Jan 2021. URL: https://doi.org/10.48550/arxiv.2103.00349, doi:10.48550/arxiv.2103.00349. This article has 311 citations.

15. (eriksson2021highdimensionalbayesianoptimization pages 4-5): David Eriksson and Martin Jankowiak. High-dimensional bayesian optimization with sparse axis-aligned subspaces. Preprint, Jan 2021. URL: https://doi.org/10.48550/arxiv.2103.00349, doi:10.48550/arxiv.2103.00349. This article has 311 citations.

16. (wilson2016deepkernellearning pages 1-3): Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P. Xing. Deep kernel learning. Preprint, Jan 2016. URL: https://doi.org/10.48550/arxiv.1511.02222, doi:10.48550/arxiv.1511.02222. This article has 1469 citations.

17. (wilson2016deepkernellearning pages 6-9): Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P. Xing. Deep kernel learning. Preprint, Jan 2016. URL: https://doi.org/10.48550/arxiv.1511.02222, doi:10.48550/arxiv.1511.02222. This article has 1469 citations.

18. (kiyohara2025bayesianoptimizationwith pages 6-6): Shin Kiyohara and Yu Kumagai. Bayesian optimization with gaussian processes assisted by deep learning for material designs. The Journal of Physical Chemistry Letters, 16:5244-5251, May 2025. URL: https://doi.org/10.1021/acs.jpclett.5c00592, doi:10.1021/acs.jpclett.5c00592. This article has 8 citations.

19. (kiyohara2025bayesianoptimizationwith pages 1-2): Shin Kiyohara and Yu Kumagai. Bayesian optimization with gaussian processes assisted by deep learning for material designs. The Journal of Physical Chemistry Letters, 16:5244-5251, May 2025. URL: https://doi.org/10.1021/acs.jpclett.5c00592, doi:10.1021/acs.jpclett.5c00592. This article has 8 citations.

20. (kiyohara2025bayesianoptimizationwith pages 4-6): Shin Kiyohara and Yu Kumagai. Bayesian optimization with gaussian processes assisted by deep learning for material designs. The Journal of Physical Chemistry Letters, 16:5244-5251, May 2025. URL: https://doi.org/10.1021/acs.jpclett.5c00592, doi:10.1021/acs.jpclett.5c00592. This article has 8 citations.

21. (malu2021bayesianoptimizationin pages 4-5): Mohit Malu, Gautam Dasarathy, and Andreas Spanias. Bayesian optimization in high-dimensional spaces: a brief survey. 2021 12th International Conference on Information, Intelligence, Systems & Applications (IISA), pages 1-8, Jul 2021. URL: https://doi.org/10.1109/iisa52424.2021.9555522, doi:10.1109/iisa52424.2021.9555522. This article has 118 citations.

22. (snoek2015scalablebayesianoptimization pages 1-2): Jasper Snoek, Oren Rippel, Kevin Swersky, Ryan Kiros, Nadathur Satish, Narayanan Sundaram, Md. Mostofa Ali Patwary, Prabhat, and Ryan P. Adams. Scalable bayesian optimization using deep neural networks. Preprint, Jan 2015. URL: https://doi.org/10.48550/arxiv.1502.05700, doi:10.48550/arxiv.1502.05700. This article has 1439 citations.

23. (li2305astudyof pages 8-10): Yucen Lily Li, Tim G. J. Rudner, and Andrew Gordon Wilson. A study of bayesian neural network surrogates for bayesian optimization. ArXiv, May 2024. URL: https://doi.org/10.48550/arxiv.2305.20028, doi:10.48550/arxiv.2305.20028. This article has 86 citations.

24. (li2305astudyof pages 25-27): Yucen Lily Li, Tim G. J. Rudner, and Andrew Gordon Wilson. A study of bayesian neural network surrogates for bayesian optimization. ArXiv, May 2024. URL: https://doi.org/10.48550/arxiv.2305.20028, doi:10.48550/arxiv.2305.20028. This article has 86 citations.

25. (li2305astudyof pages 1-2): Yucen Lily Li, Tim G. J. Rudner, and Andrew Gordon Wilson. A study of bayesian neural network surrogates for bayesian optimization. ArXiv, May 2024. URL: https://doi.org/10.48550/arxiv.2305.20028, doi:10.48550/arxiv.2305.20028. This article has 86 citations.

26. (li2305astudyof pages 5-7): Yucen Lily Li, Tim G. J. Rudner, and Andrew Gordon Wilson. A study of bayesian neural network surrogates for bayesian optimization. ArXiv, May 2024. URL: https://doi.org/10.48550/arxiv.2305.20028, doi:10.48550/arxiv.2305.20028. This article has 86 citations.

27. (li2305astudyof pages 2-4): Yucen Lily Li, Tim G. J. Rudner, and Andrew Gordon Wilson. A study of bayesian neural network surrogates for bayesian optimization. ArXiv, May 2024. URL: https://doi.org/10.48550/arxiv.2305.20028, doi:10.48550/arxiv.2305.20028. This article has 86 citations.

28. (gomezbombarelli2018automaticchemicaldesign pages 7-8): Rafael Gómez-Bombarelli, Jennifer N. Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D. Hirzel, Ryan P. Adams, and Alán Aspuru-Guzik. Automatic chemical design using a data-driven continuous representation of molecules. Jan 2018. URL: https://doi.org/10.1021/acscentsci.7b00572, doi:10.1021/acscentsci.7b00572. This article has 4888 citations and is from a highest quality peer-reviewed journal.

29. (maus2022locallatentspace pages 1-3): Natalie Maus, Haydn T. Jones, Juston S. Moore, Matt J. Kusner, John Bradshaw, and Jacob R. Gardner. Local latent space bayesian optimization over structured inputs. Preprint, Jan 2022. URL: https://doi.org/10.48550/arxiv.2201.11872, doi:10.48550/arxiv.2201.11872. This article has 147 citations.

30. (maus2022locallatentspace pages 9-11): Natalie Maus, Haydn T. Jones, Juston S. Moore, Matt J. Kusner, John Bradshaw, and Jacob R. Gardner. Local latent space bayesian optimization over structured inputs. Preprint, Jan 2022. URL: https://doi.org/10.48550/arxiv.2201.11872, doi:10.48550/arxiv.2201.11872. This article has 147 citations.

31. (talibart2025learningachemistryaware pages 18-21): Hugo Talibart and Dimitri Gilis. Learning a chemistry-aware latent space for molecular encoding and generation with a large-scale transformer variational autoencoder. bioRxiv, Dec 2025. URL: https://doi.org/10.64898/2025.12.19.695394, doi:10.64898/2025.12.19.695394. This article has 0 citations.

32. (griffiths2020constrainedbayesianoptimization pages 2-3): Ryan-Rhys Griffiths and José Miguel Hernández-Lobato. Constrained bayesian optimization for automatic chemical design using variational autoencoders. JournalArticle, Oct 2020. URL: https://doi.org/10.17863/cam.76888, doi:10.17863/cam.76888. This article has 608 citations.

33. (han2026machinelearningguideddesignof pages 12-13): Bo Han, Marcus de Carvalho, Jie Zhang, Hui Mao, and Qingyu Yan. Machine-learning-guided design of mof-based electrocatalysts for sustainable ammonia production. Chemical communications, Jan 2026. URL: https://doi.org/10.1039/d5cc07118f, doi:10.1039/d5cc07118f. This article has 8 citations and is from a domain leading peer-reviewed journal.

34. (gomezgualdron2026machinelearningto pages 8-9): Diego A. Gómez-Gualdrón, Tatiane Gercina de Vilas, Katherine Ardila, Fernando Fajardo-Rojas, and Alexander J. Pak. Machine learning to design metal–organic frameworks: progress and challenges from a data efficiency perspective. Materials Horizons, 13:1694-1715, Jan 2026. URL: https://doi.org/10.1039/d5mh01467k, doi:10.1039/d5mh01467k. This article has 8 citations and is from a domain leading peer-reviewed journal.

35. (schultz2026graphneuralnetworkbased pages 6-7): Lane E. Schultz, Nickolas Gantzler, N. Scott Bobbitt, Dorina F. Sava Gallis, and Rémi Dingreville. Graph neural network-based multi-objective bayesian optimization for enhanced screening of metal–organic frameworks with optimal separation performance. Journal of Materials Chemistry A, 14:10836-10853, Jan 2026. URL: https://doi.org/10.1039/d5ta09133k, doi:10.1039/d5ta09133k. This article has 2 citations.

36. (park2024fromdatato pages 2-4): Junkil Park, Honghui Kim, Yeonghun Kang, Yunsung Lim, and Jihan Kim. From data to discovery: recent trends of machine learning in metal–organic frameworks. JACS Au, 4:3727-3743, Sep 2024. URL: https://doi.org/10.1021/jacsau.4c00618, doi:10.1021/jacsau.4c00618. This article has 81 citations and is from a peer-reviewed journal.

37. (wang2023selfsupervisedrepresentationlearning pages 122-125): Yuyang Wang. Self-supervised representation learning for molecular property predictions. Text, Jan 2023. URL: https://doi.org/10.1184/r1/23635671, doi:10.1184/r1/23635671. This article has 4 citations and is from a peer-reviewed journal.

38. (wang2023selfsupervisedrepresentationlearning pages 23-28): Yuyang Wang. Self-supervised representation learning for molecular property predictions. Text, Jan 2023. URL: https://doi.org/10.1184/r1/23635671, doi:10.1184/r1/23635671. This article has 4 citations and is from a peer-reviewed journal.

39. (magar2022crystaltwinsselfsupervised pages 1-4): Rishikesh Magar, Yuyang Wang, and Amir Barati Farimani. Crystal twins: self-supervised learning for crystalline material property prediction. npj Computational Materials, 8:1-8, May 2022. URL: https://doi.org/10.48550/arxiv.2205.01893, doi:10.48550/arxiv.2205.01893. This article has 69 citations and is from a peer-reviewed journal.

40. (magar2022crystaltwinsselfsupervised pages 11-16): Rishikesh Magar, Yuyang Wang, and Amir Barati Farimani. Crystal twins: self-supervised learning for crystalline material property prediction. npj Computational Materials, 8:1-8, May 2022. URL: https://doi.org/10.48550/arxiv.2205.01893, doi:10.48550/arxiv.2205.01893. This article has 69 citations and is from a peer-reviewed journal.

41. (zhou2025polyclcontrastivelearning pages 1-4): Jiajun Zhou, Yijie Yang, Austin M. Mroz, and Kim E. Jelfs. Polycl: contrastive learning for polymer representation learning via explicit and implicit augmentations. Digital Discovery, 4:149-160, Aug 2025. URL: https://doi.org/10.48550/arxiv.2408.07556, doi:10.48550/arxiv.2408.07556. This article has 13 citations and is from a peer-reviewed journal.

42. (wang2023selfsupervisedrepresentationlearning pages 138-143): Yuyang Wang. Self-supervised representation learning for molecular property predictions. Text, Jan 2023. URL: https://doi.org/10.1184/r1/23635671, doi:10.1184/r1/23635671. This article has 4 citations and is from a peer-reviewed journal.

43. (wang2023selfsupervisedrepresentationlearning pages 1-9): Yuyang Wang. Self-supervised representation learning for molecular property predictions. Text, Jan 2023. URL: https://doi.org/10.1184/r1/23635671, doi:10.1184/r1/23635671. This article has 4 citations and is from a peer-reviewed journal.

44. (gomezgualdron2026machinelearningto pages 15-15): Diego A. Gómez-Gualdrón, Tatiane Gercina de Vilas, Katherine Ardila, Fernando Fajardo-Rojas, and Alexander J. Pak. Machine learning to design metal–organic frameworks: progress and challenges from a data efficiency perspective. Materials Horizons, 13:1694-1715, Jan 2026. URL: https://doi.org/10.1039/d5mh01467k, doi:10.1039/d5mh01467k. This article has 8 citations and is from a domain leading peer-reviewed journal.