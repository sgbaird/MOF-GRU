Question: In the context of inverse design of metal-organic frameworks (MOFs), how can one optimize over a learned continuous latent space instead of over global/hand-engineered descriptor variables? Provide a high-effort, well-cited review covering: (1) Variational autoencoders (VAEs) over string/SELFIES/SMILES or graph representations that yield a direct, smooth latent space for Bayesian or gradient-based optimization (e.g., ChemVAE/Gomez-Bombarelli, junction-tree VAE, SELFIES-based VAEs, MOF-specific generative models such as SmVAE/Yao MOF generation, GHP-MOFassemble, and recent diffusion/transformer MOF generators); (2) using the hidden state / pooled embedding of a sequence model (e.g., a bidirectional GRU over a MOF 'sentence' of topology+node+linker SELFIES tokens, as in MOF-GRU) as a latent space for optimization, and the pitfalls of optimizing a non-generative encoder's latent space (no decoder, holes/invalid regions, off-manifold extrapolation); (3) graph neural network latent embeddings of crystal/MOF structure and how to make them optimizable/invertible; (4) latent-space optimization methods: Bayesian optimization with Gaussian processes in latent space, latent-space BO with trust regions (e.g., LS-BO, LOL-BO, weighted retraining by Tripp et al.), and gradient ascent on a differentiable property predictor; (5) ensuring decodability/validity and constraining to the data manifold (jointly trained property predictors, semi-supervised VAEs, uncertainty/penalty terms). Compare these latent-space approaches to optimizing over global variables directly, and give concrete, actionable recommendations for the MOF-GRU codebase. (6) Also cover the alternative 'featurization over a predefined candidate space' paradigm (as in the Honegumi/Ax featurization tutorial): rather than optimizing a free continuous latent vector that must be decoded, enumerate a fixed library of real, synthesizable MOF candidates, featurize each candidate (e.g., with a learned encoder embedding such as the MOF-GRU pooled hidden state, or with hand-engineered global descriptors), and run Bayesian optimization / active learning over the discrete candidate set so that every proposed point is guaranteed valid and decode-free. Discuss when discrete-candidate BO with a learned featurization is preferable to generative latent-space optimization, the role of the GP kernel over learned features, batch/active-learning acquisition over candidate pools, and how this maps onto MOF-GRU's existing get_hidden_layer_output embeddings.

## Optimizing Over Learned Continuous Latent Spaces for Inverse Design of Metal-Organic Frameworks: A Comprehensive Review

### 1. Variational Autoencoders Over String/Graph Representations for MOF Inverse Design

#### 1.1 ChemVAE: The Foundational Paradigm

The seminal work by Gómez-Bombarelli et al. established the paradigm of converting discrete molecular representations into a continuous latent space amenable to optimization. Their approach uses a variational autoencoder (VAE) where an encoder converts SMILES strings into fixed-dimensional continuous vectors via 1D convolutional layers followed by fully connected layers, a decoder reconstructs SMILES strings from latent vectors using gated recurrent units, and a jointly trained multilayer perceptron predicts chemical properties directly from the latent representation (gomezbombarelli2018automaticchemicaldesign pages 1-2, gomezbombarelli2018automaticchemicaldesign pages 7-8, gomezbombarelli2018automaticchemicaldesign pages 2-4). The jointly trained property predictor organizes the latent space so that high-value molecules cluster in specific regions, enabling Gaussian process-based Bayesian optimization to navigate the continuous space and consistently outperform random search and genetic algorithm baselines (gomezbombarelli2018automaticchemicaldesign pages 5-7). This architecture demonstrated that continuous representations allow gradient-based optimization to efficiently guide the search for optimized functional compounds (gomezbombarelli2018automaticchemicaldesign pages 1-2).

#### 1.2 Junction Tree VAE (JT-VAE)

The Junction Tree VAE addresses a critical limitation of SMILES-based VAEs: decoded strings often fail to correspond to valid molecules. JT-VAE decomposes molecular graphs into sub-pieces (rings, functional groups, atoms) from a training set vocabulary and encodes both the full graph and its tree structure decomposition into two separate latent spaces (schwalbekoda2020generativemodelsfor pages 11-14). The tree-structured scaffold approach ensures chemical validity at every generation step by only sampling from cluster labels that are chemically compatible with parent nodes (jin2018junctiontreevariational pages 3-5). JT-VAE achieves 100% validity on small molecules and creates a meaningful latent space essential for molecular optimization (schwalbekoda2020generativemodelsfor pages 11-14, jin2018junctiontreevariational pages 8-10). However, it was designed for small molecules and cannot directly handle large compound structures due to their high spatial order (ochiai2023variationalautoencoderbasedchemical pages 1-2).

#### 1.3 SELFIES-Based VAEs

SELFIES (Self-Referencing Embedded Strings) provides a 100% robust molecular string representation where every string maps to a valid molecular graph (krenn2019selfiesarobust pages 1-2). When used in VAEs, SELFIES eliminates "dead areas" in latent space—100% of decoded points yield valid molecules, compared to only a small fraction with SMILES (krenn2019selfiesarobust pages 3-4, talibart2025learningachemistryaware pages 1-3). SELFIES-based VAEs produce over 100 times more diverse valid molecules than SMILES-based counterparts (krenn2019selfiesarobust pages 4-6). Recent work by Talibart and Gilis introduced a large-scale Transformer VAE using SELFIES with a novel loss term that explicitly enforces consistency between embedding distances and Tanimoto chemical similarities, achieving 97% reconstruction and 100% validity rates (talibart2025learningachemistryaware pages 1-3, talibart2025learningachemistryaware pages 3-5).

#### 1.4 MOF-Specific Generative Models

**SmVAE (Supramolecular VAE):** Yao et al. developed the SmVAE for inverse design of nanoporous reticular materials by deconstructing MOFs into their modular components—metal nodes, multiconnected organic nodes, ditopic linkers, and topological nets—encoded via an RFcode representation (yao2021inversedesignof pages 4-6). The model was jointly trained in a semi-supervised fashion using 45k MOFs with property labels and ~2 million MOFs without labels, with a Gaussian process model built on the latent space to guide optimization toward target gas separation properties (yao2021inversedesignof pages 6-8). The SmVAE achieved approximately 61.5% chemical validity for randomly sampled latent vectors (duan2026theriseof pages 3-7).

**GHP-MOFassemble:** Park et al. introduced a diffusion model-based framework that generates novel MOF linkers using DiffLinker, which are then assembled with pre-selected metal nodes into MOFs with primitive cubic topology (park2024agenerativeartificial pages 1-2, park2024agenerativeartificial pages 7-9). The framework uses an E(3)-equivariant graph neural network for denoising and generated over 120,000 MOF candidates, identifying structures with CO₂ capacities exceeding 96.9% of hypothetical MOF datasets (park2023ghpmofassemblediffusionmodeling pages 1-4, park2024agenerativeartificial pages 1-2).

**MOFDiff:** Fu et al. proposed a coarse-grained diffusion model that generates MOF structures through denoising diffusion over building block coordinates and identities, using equivariant graph neural networks to respect crystal symmetries (fu2023mofdiffcoarsegraineddiffusion pages 1-2). This approach avoids pre-defined topology templates and enables greater chemical diversity than template-based methods.

**MOFFUSION:** A latent diffusion model trained on signed distance function (SDF) representations of MOF pore structures, enabling conditional generation based on numeric, categorical, or textual descriptors (duan2026theriseof pages 3-7, han2025egmofefficientgeneration pages 26-31).

**MOFGPT:** A transformer-based language model trained on MOFid sequences with reinforcement learning-based property optimization, integrating MOFormer for property prediction and guiding generation toward targeted properties (badrinarayanan2025mofgptgenerativedesign pages 1-4).

**Deep Dreaming for MOFs:** Cleeton and Sarkisov developed a deep dreaming methodology that integrates property prediction and structure optimization within a single interpretable framework, using a chemical language model augmented with attention mechanisms to optimize MOF linkers by gradient ascent on the predictor (cleeton2025inversedesignof pages 1-2).

### 2. Sequence Model Hidden States as Latent Spaces: MOF-GRU and Pitfalls

#### 2.1 MOF-GRU Architecture

MOF-GRU (Li et al., ACS Applied Materials & Interfaces, 2023) uses a bidirectional GRU over MOFID-based text representations of MOFs to predict gas separation performance. The MOFID representation encodes MOF topology, nodes, and linkers as a structured text string, which can incorporate SELFIES tokens for molecular components. The model's `get_hidden_layer_output` function extracts the pooled hidden state of the bidirectional GRU, providing a learned continuous embedding for each MOF. This embedding captures structure-property relationships learned during supervised training and can, in principle, serve as a feature space for downstream optimization.

#### 2.2 Pitfalls of Non-Generative Encoder Latent Spaces

Using a discriminative encoder's hidden state as a latent space for optimization presents fundamental challenges that have been extensively documented in the molecular design literature:

**No decoder / non-invertibility:** Without a trained decoder, there is no mechanism to convert an optimized latent vector back into a valid MOF structure. The encoder maps discrete inputs to continuous vectors, but the reverse mapping is undefined (griffiths2020constrainedbayesianoptimization pages 1-2, moss2507returnofthe pages 2-3).

**Holes and invalid regions:** Even in VAE latent spaces with decoders, "dead regions" exist where decoded outputs are invalid. In a purely discriminative encoder's latent space, this problem is far worse because no generative training objective constrains the space to be smoothly decodable. Bayesian optimization queries latent space points far from the training data distribution, resulting in invalid structures (griffiths2020constrainedbayesianoptimization pages 1-2, griffiths2020constrainedbayesianoptimization pages 2-3).

**Off-manifold extrapolation:** The Bayesian optimization scheme operates decoupled from the encoder without knowledge of the learned manifold's location, causing exploration to select points in regions that do not correspond to any real MOF (griffiths2020constrainedbayesianoptimization pages 1-2). VAEs trained offline in an unsupervised manner cause misalignment between the objective function and latent space, and neural networks can distort local neighborhoods so that smoothness assumptions do not transfer (moss2507returnofthe pages 2-3).

**Latent space smoothness violations:** Small changes in latent space can cause large changes in reconstructed inputs, violating the GP surrogate assumptions essential for Bayesian optimization (maus2022locallatentspace pages 3-5). For a non-generative encoder, there is no guarantee that nearby points in embedding space correspond to chemically similar structures.

### 3. Graph Neural Network Latent Embeddings for Crystal/MOF Structures

#### 3.1 CDVAE: Graph-Based Crystal Generation

The Crystal Diffusion Variational Autoencoder (CDVAE) by Xie et al. uses SE(3) equivariant periodic graph neural networks (PGNNs) for both encoder and decoder. The encoder learns latent representations of stable materials, while the decoder generates new structures via a score-matching diffusion process that outputs gradients driving atomic coordinates toward energy minima (xie2021crystaldiffusionvariational pages 1-2, xie2021crystaldiffusionvariational pages 2-4). Materials are represented as directed multi-graphs where nodes represent atoms and edges represent bonds across periodic unit cells (xie2021crystaldiffusionvariational pages 2-4). GNN architectures such as DimeNet++, GINE, and GemNetT are employed (pakornchote2024diffusionprobabilisticmodels pages 5-6).

#### 3.2 Mofasa: GNN Autoencoder for MOFs

Mofasa employs a shared GNN backbone adapted from machine-learned interatomic potential architectures, with a hierarchical message passing scheme processing local (atomic neighborhoods) and global (lattice parameters) features separately. The autoencoder maps standardized crystal structures to continuous latent representations with dimensionality D=4, regularized using residual vector quantization and KL-divergence, while a diffusion model generates these latent representations for MOF structure generation (simkus2025mofasaastep pages 19-22, simkus2025mofasaastep pages 22-24).

#### 3.3 Making GNN Embeddings Optimizable/Invertible

Invertible neural networks (e.g., MatDesINNe by Fung et al.) use normalizing flows to establish bijective mappings between design spaces and target properties, enabling both forward prediction and inverse design without separate encoder-decoder training. The FTCP framework combines VAEs with crystallographic representations using real-space and reciprocal-space Fourier-transformed features for conditional crystal generation (li2025materialsgenerationin pages 9-10). The key challenge remains that standard GNN property predictors produce embeddings without guaranteed invertibility; pairing them with generative decoders (as in CDVAE) or using normalizing flows addresses this limitation.

### 4. Latent-Space Optimization Methods

#### 4.1 Bayesian Optimization with Gaussian Processes in Latent Space

The foundational LS-BO approach trains a GP surrogate on the latent vectors produced by a VAE encoder, then optimizes an acquisition function (e.g., expected improvement) over the continuous latent space (gomezbombarelli2018automaticchemicaldesign pages 5-7, gonzalezduque2024asurveyand pages 4-5). The GP provides posterior mean and uncertainty estimates that balance exploration and exploitation. However, standard LS-BO suffers from the curse of dimensionality in high-dimensional latent spaces and from the misalignment between GP smoothness assumptions and the actual structure of the latent space (maus2022locallatentspace pages 3-5, moss2507returnofthe pages 2-3).

#### 4.2 LOL-BO: Trust Regions in Latent Space

LOL-BO (Local Latent Space Bayesian Optimization) by Maus et al. adapts trust region methods to restrict searches to small hyper-rectangular regions centered at the incumbent solution, avoiding over-exploration in high-dimensional latent spaces (maus2022locallatentspace pages 3-5, maus2022locallatentspace pages 1-3). A key innovation is reformulating the encoder to function both as a global denoising autoencoder encoder and as a deep kernel for the GP surrogate within trust regions, better aligning local optimization in latent space with local optimization in input space. LOL-BO achieves up to 22× improvements over prior latent space BO approaches and demonstrates superior sample efficiency (maus2022locallatentspace pages 1-3, maus2022locallatentspace pages 9-11).

#### 4.3 Weighted Retraining (W-LBO)

Tripp et al. proposed weighted retraining schemes that periodically retrain the VAE with emphasis on high-performing molecules discovered during optimization, reshaping the latent space to concentrate representational capacity on promising regions (gonzalezduque2024asurveyand pages 4-5, abeer2024multiobjectivelatentspace pages 2-3). This approach has been combined with BO to improve optimization beyond simple random sampling (abeer2024multiobjectivelatentspace pages 9-11).

#### 4.4 Gradient Ascent on Differentiable Property Predictors

PASITHEA (Shen et al.) demonstrates direct gradient-based molecule optimization using SELFIES representations, applying "inceptionism" techniques by backpropagating errors with respect to input molecular representations while keeping network weights fixed (shen2021deepmoleculardreaming pages 3-5, shen2021deepmoleculardreaming pages 1-3). SELFIES guarantees 100% validity of all intermediate and final outputs during gradient-based optimization (shen2021deepmoleculardreaming pages 3-5). This approach enables direct property-oriented optimization without requiring a VAE, though it requires a differentiable predictor and a surjective molecular representation.

#### 4.5 Additional Methods

**InvBO** addresses the misalignment problem in LBO by searching for latent codes that perfectly reconstruct target data, with potential-aware trust region anchor selection (from the literature on Chu et al., 2024). **LaMBO** (Stanton et al.) jointly trains a denoising autoencoder with a discriminative multi-task GP head, enabling gradient-based optimization of multi-objective acquisition functions in latent space (muthyala2025generativemultiobjectivebayesian pages 11-15). **COWBOYS** (Moss et al.) proposes a decoupled approach that trains the generative model and GP surrogate separately, combining them via a Bayesian update rule—arguing that separation allows each component to focus on its strengths (moss2507returnofthe pages 1-2).

### 5. Ensuring Decodability/Validity and Constraining to the Data Manifold

#### 5.1 Constrained Bayesian Optimization

Griffiths and Hernández-Lobato formalized the constraint that a latent point must decode successfully with high probability: maximizing f(z) subject to Pr(C(z)) ≥ 1 − δ, where δ is a user-specified confidence level (griffiths2020constrainedbayesianoptimization pages 2-3, griffiths2020constrainedbayesianoptimization pages 1-2). This addresses the off-manifold pathology where standard BO queries "dead regions" of latent space far from the training data manifold.

#### 5.2 Semi-Supervised VAEs

Kang and Cho proposed a semi-supervised VAE that simultaneously performs property prediction and molecule generation, conditioning generation on desired property ranges (abeer2024multiobjectivelatentspace pages 1-2). The semi-supervised approach exploits unlabeled molecules to improve property prediction while enabling targeted generation. Similarly, the SeMole method extends JT-VAE with semi-supervised learning to incorporate target properties into the latent representation with limited labeled data (hamidizadeh2208semisupervisedjunctiontree pages 1-2).

#### 5.3 Jointly Trained Property Predictors

Training a property predictor jointly with the VAE organizes the latent space by property values, creating smooth property gradients that facilitate optimization (gomezbombarelli2018automaticchemicaldesign pages 5-7, gomezbombarelli2018automaticchemicaldesign pages 2-4). LOL-BO takes this further by jointly training the VAE and GP surrogate through variational inference, encouraging the latent space to align with GP priors (maus2022locallatentspace pages 1-3, maus2022locallatentspace pages 3-5).

#### 5.4 Representation-Level Validity Guarantees

SELFIES provides the strongest validity guarantee: 100% of decoded points yield valid molecules regardless of where in latent space the optimization explores (krenn2019selfiesarobust pages 3-4, krenn2019selfiesarobust pages 4-6). JT-VAE ensures validity through tree-structured scaffold generation with chemical feasibility checks at each step (jin2018junctiontreevariational pages 3-5, schwalbekoda2020generativemodelsfor pages 11-14). Chemistry-aware latent spaces that align embedding distances with chemical similarities provide additional smoothness guarantees (talibart2025learningachemistryaware pages 1-3, talibart2025learningachemistryaware pages 3-5).

#### 5.5 Latent Space Quality Measures

Key indicators of a well-structured latent space for optimization include: high reconstruction accuracy, absence of dead regions, correlation between latent distances and chemical similarity, and the continuity property where decoded molecules remain structurally similar to reference molecules with gradually decaying similarity as distance increases (talibart2025learningachemistryaware pages 10-12, haddad2025targetedmoleculargeneration pages 7-9).

### 6. The Discrete-Candidate BO / Active Learning Paradigm

#### 6.1 Featurization Over a Predefined Candidate Space

An alternative to generative latent-space optimization is to enumerate a fixed library of real, synthesizable MOF candidates, featurize each candidate, and run Bayesian optimization or active learning over this discrete candidate set. The Honegumi interface by Baird et al. simplifies creating BO workflows on the Ax platform, explicitly supporting discrete candidate evaluation rather than continuous optimization, allowing researchers to configure key parameters and generate ready-to-use optimization scripts (muthyala2025generativemultiobjectivebayesian pages 11-15).

In this paradigm, each candidate is represented by a feature vector—either hand-engineered global descriptors (e.g., pore volume, surface area, metal identity) or learned embeddings (e.g., from a pretrained encoder such as MOF-GRU's pooled hidden state). A GP surrogate with an appropriate kernel (e.g., Matérn or RBF) is fit over these features, and an acquisition function (e.g., expected improvement) is maximized over the finite candidate pool to select the next candidate(s) for evaluation (bishnoi2305materialsinformaticsan pages 28-30, oftelie2018activelearningfor pages 1-2).

#### 6.2 When Discrete-Candidate BO Is Preferable

Discrete-candidate BO with learned featurization is preferable to generative latent-space optimization when:
- **Validity is paramount:** Every proposed point is guaranteed valid and synthesizable because candidates are drawn from a curated library (oftelie2018activelearningfor pages 1-2).
- **The candidate space is manageable:** When the number of feasible MOFs is in the thousands to millions (e.g., from hypothetical MOF databases), exhaustive featurization is tractable.
- **No decoder is available:** When using a discriminative encoder like MOF-GRU that has no decoder, discrete-candidate BO sidesteps the need for decoding entirely.
- **Evaluation is expensive:** Pool-based active learning efficiently selects maximally informative experiments from a finite set, reducing the number of costly evaluations (oftelie2018activelearningfor pages 1-2, bishnoi2305materialsinformaticsan pages 28-30).

#### 6.3 GP Kernels Over Learned Features

Using learned representations (e.g., LLM embeddings, GRU hidden states) as input features to GP kernels has been shown to be effective for BO over molecules. Kristiadi et al. demonstrated that pretrained language model embeddings can serve as fixed feature extractors for GP surrogate models with Matérn kernels, though domain-specific pretraining or finetuning is critical for strong performance (kristiadi2024asoberlook pages 4-5, kristiadi2024asoberlook pages 1-2, kristiadi2024asoberlook pages 2-3). The kernel computes similarity between candidates in the learned feature space, and the GP provides calibrated uncertainty estimates for acquisition function evaluation.

#### 6.4 Batch/Active-Learning Acquisition Over Candidate Pools

Batch BO selects multiple candidates per iteration for parallel evaluation. Approaches such as qPMHI (probability of maximum hypervolume improvement) enable batch-optimal selection from a generated or curated pool of candidates, decoupling generation from selection (muthyala2025generativemultiobjectivebayesian pages 11-15). ROBOT (Rank-Ordered Bayesian Optimization with Trust-regions) aims to find diverse portfolios of high-performing solutions (from Maus et al., 2022). These methods are directly applicable to pool-based MOF optimization.

### 7. Comparison: Latent-Space Optimization vs. Global/Hand-Engineered Descriptors vs. Discrete-Candidate BO

Optimizing over a learned continuous latent space offers several advantages over optimizing global descriptors directly: (i) the latent space captures complex, nonlinear structure-property relationships that hand-engineered descriptors may miss; (ii) gradient-based optimization and GP surrogate modeling are natural in continuous spaces; (iii) novel, previously unseen structures can be generated by decoding optimized latent vectors (gomezbombarelli2018automaticchemicaldesign pages 1-2, gomezbombarelli2018automaticchemicaldesign pages 2-4). However, latent-space optimization faces challenges including invalid decoded structures, off-manifold extrapolation, and the need for carefully structured latent spaces (griffiths2020constrainedbayesianoptimization pages 2-3, griffiths2020constrainedbayesianoptimization pages 1-2, moss2507returnofthe pages 2-3).

Discrete-candidate BO with learned featurization offers a pragmatic middle ground: it uses the representational power of learned embeddings (capturing the same nonlinear relationships) while guaranteeing validity and avoiding the decoder problem entirely. The trade-off is that discovery is limited to the pre-enumerated candidate pool, precluding truly novel structure generation.

### 8. Concrete Recommendations for the MOF-GRU Codebase

1. **Discrete-candidate BO using MOF-GRU embeddings (recommended first approach):** Use `get_hidden_layer_output` to extract pooled hidden states for each MOF in a candidate library. Fit a GP with a Matérn-5/2 kernel over these embeddings and run BO with expected improvement or Thompson sampling to select the next MOFs for simulation/experiment. This is decode-free and guarantees validity (kristiadi2024asoberlook pages 4-5, oftelie2018activelearningfor pages 1-2).

2. **If generative latent-space optimization is desired, add a decoder:** Train a VAE by augmenting MOF-GRU with a decoder (e.g., an autoregressive GRU decoder over MOFID tokens). Use SELFIES for linker tokens to guarantee validity of decoded components (krenn2019selfiesarobust pages 3-4, krenn2019selfiesarobust pages 4-6). Jointly train a property prediction head to organize the latent space by target properties (gomezbombarelli2018automaticchemicaldesign pages 5-7, gomezbombarelli2018automaticchemicaldesign pages 2-4).

3. **Apply trust region methods:** If optimizing in the latent space, use LOL-BO-style trust regions to restrict BO to well-populated regions of the latent manifold, avoiding the off-manifold pathology (maus2022locallatentspace pages 3-5, maus2022locallatentspace pages 1-3).

4. **Add constrained BO:** Implement Griffiths-style probability-of-validity constraints to penalize latent points unlikely to decode to valid MOFs (griffiths2020constrainedbayesianoptimization pages 2-3, griffiths2020constrainedbayesianoptimization pages 1-2).

5. **Chemistry-aware latent space regularization:** Add loss terms aligning latent distances with chemical similarity (e.g., Tanimoto similarity between MOF fingerprints), as demonstrated by Talibart and Gilis, to ensure smooth and meaningful latent neighborhoods (talibart2025learningachemistryaware pages 1-3, talibart2025learningachemistryaware pages 3-5).

6. **Consider PASITHEA-style gradient optimization:** If using SELFIES-based MOFID representations, direct gradient ascent on the property predictor with respect to the input representation can optimize MOF structures without a latent space, leveraging SELFIES' 100% validity guarantee (shen2021deepmoleculardreaming pages 3-5, shen2021deepmoleculardreaming pages 1-3).

7. **Batch active learning over candidate pools:** For practical deployment, enumerate a large candidate library of hypothetical MOFs, embed all candidates using MOF-GRU, and use batch acquisition functions (e.g., q-EI or q-PMHI) to select diverse, high-performing candidates for parallel evaluation, following the Honegumi/Ax paradigm.

In summary, the choice between generative latent-space optimization and discrete-candidate BO with learned featurization depends on whether the goal is to discover entirely novel MOF structures (favoring generative approaches) or to efficiently identify the best candidates from a known design space (favoring pool-based BO). For the MOF-GRU codebase specifically, the most immediately actionable approach is discrete-candidate BO using the existing `get_hidden_layer_output` embeddings as GP features, with generative latent-space optimization requiring architectural extensions (decoder addition, SELFIES adoption, constrained BO) that would constitute a more substantial but potentially more powerful upgrade.

References

1. (gomezbombarelli2018automaticchemicaldesign pages 1-2): Rafael Gómez-Bombarelli, Jennifer N. Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D. Hirzel, Ryan P. Adams, and Alán Aspuru-Guzik. Automatic chemical design using a data-driven continuous representation of molecules. Jan 2018. URL: https://doi.org/10.1021/acscentsci.7b00572, doi:10.1021/acscentsci.7b00572. This article has 4888 citations and is from a highest quality peer-reviewed journal.

2. (gomezbombarelli2018automaticchemicaldesign pages 7-8): Rafael Gómez-Bombarelli, Jennifer N. Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D. Hirzel, Ryan P. Adams, and Alán Aspuru-Guzik. Automatic chemical design using a data-driven continuous representation of molecules. Jan 2018. URL: https://doi.org/10.1021/acscentsci.7b00572, doi:10.1021/acscentsci.7b00572. This article has 4888 citations and is from a highest quality peer-reviewed journal.

3. (gomezbombarelli2018automaticchemicaldesign pages 2-4): Rafael Gómez-Bombarelli, Jennifer N. Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D. Hirzel, Ryan P. Adams, and Alán Aspuru-Guzik. Automatic chemical design using a data-driven continuous representation of molecules. Jan 2018. URL: https://doi.org/10.1021/acscentsci.7b00572, doi:10.1021/acscentsci.7b00572. This article has 4888 citations and is from a highest quality peer-reviewed journal.

4. (gomezbombarelli2018automaticchemicaldesign pages 5-7): Rafael Gómez-Bombarelli, Jennifer N. Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D. Hirzel, Ryan P. Adams, and Alán Aspuru-Guzik. Automatic chemical design using a data-driven continuous representation of molecules. Jan 2018. URL: https://doi.org/10.1021/acscentsci.7b00572, doi:10.1021/acscentsci.7b00572. This article has 4888 citations and is from a highest quality peer-reviewed journal.

5. (schwalbekoda2020generativemodelsfor pages 11-14): Daniel Schwalbe-Koda and Rafael Gómez-Bombarelli. Generative models for automatic chemical design. Lecture Notes in Physics, pages 445-467, Jan 2020. URL: https://doi.org/10.1007/978-3-030-40245-7\_21, doi:10.1007/978-3-030-40245-7\_21. This article has 117 citations and is from a peer-reviewed journal.

6. (jin2018junctiontreevariational pages 3-5): Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. Preprint, Jan 2018. URL: https://doi.org/10.48550/arxiv.1802.04364, doi:10.48550/arxiv.1802.04364. This article has 2357 citations.

7. (jin2018junctiontreevariational pages 8-10): Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. Preprint, Jan 2018. URL: https://doi.org/10.48550/arxiv.1802.04364, doi:10.48550/arxiv.1802.04364. This article has 2357 citations.

8. (ochiai2023variationalautoencoderbasedchemical pages 1-2): Toshiki Ochiai, Tensei Inukai, Manato Akiyama, Kairi Furui, Masahito Ohue, Nobuaki Matsumori, Shinsuke Inuki, Motonari Uesugi, Toshiaki Sunazuka, Kazuya Kikuchi, Hideaki Kakeya, and Yasubumi Sakakibara. Variational autoencoder-based chemical latent space for large molecular structures with 3d complexity. Communications Chemistry, Nov 2023. URL: https://doi.org/10.1038/s42004-023-01054-6, doi:10.1038/s42004-023-01054-6. This article has 113 citations and is from a peer-reviewed journal.

9. (krenn2019selfiesarobust pages 1-2): Mario Krenn, F. Häse, AkshatKumar Nigam, Pascal Friederich, and Alán Aspuru-Guzik. Selfies: a robust representation of semantically constrained graphs with an example application in chemistry. ArXiv, May 2019. URL: https://doi.org/10.48550/arxiv.1905.13741, doi:10.48550/arxiv.1905.13741. This article has 110 citations.

10. (krenn2019selfiesarobust pages 3-4): Mario Krenn, F. Häse, AkshatKumar Nigam, Pascal Friederich, and Alán Aspuru-Guzik. Selfies: a robust representation of semantically constrained graphs with an example application in chemistry. ArXiv, May 2019. URL: https://doi.org/10.48550/arxiv.1905.13741, doi:10.48550/arxiv.1905.13741. This article has 110 citations.

11. (talibart2025learningachemistryaware pages 1-3): Hugo Talibart and Dimitri Gilis. Learning a chemistry-aware latent space for molecular encoding and generation with a large-scale transformer variational autoencoder. bioRxiv, Dec 2025. URL: https://doi.org/10.64898/2025.12.19.695394, doi:10.64898/2025.12.19.695394. This article has 0 citations.

12. (krenn2019selfiesarobust pages 4-6): Mario Krenn, F. Häse, AkshatKumar Nigam, Pascal Friederich, and Alán Aspuru-Guzik. Selfies: a robust representation of semantically constrained graphs with an example application in chemistry. ArXiv, May 2019. URL: https://doi.org/10.48550/arxiv.1905.13741, doi:10.48550/arxiv.1905.13741. This article has 110 citations.

13. (talibart2025learningachemistryaware pages 3-5): Hugo Talibart and Dimitri Gilis. Learning a chemistry-aware latent space for molecular encoding and generation with a large-scale transformer variational autoencoder. bioRxiv, Dec 2025. URL: https://doi.org/10.64898/2025.12.19.695394, doi:10.64898/2025.12.19.695394. This article has 0 citations.

14. (yao2021inversedesignof pages 4-6): Zhenpeng Yao, Benjamín Sánchez-Lengeling, N. Scott Bobbitt, Benjamin J. Bucior, Sai Govind Hari Kumar, Sean P. Collins, Thomas Burns, Tom K. Woo, Omar K. Farha, Randall Q. Snurr, and Alán Aspuru-Guzik. Inverse design of nanoporous crystalline reticular materials with deep generative models. Jan 2021. URL: https://doi.org/10.1038/s42256-020-00271-1, doi:10.1038/s42256-020-00271-1. This article has 470 citations and is from a domain leading peer-reviewed journal.

15. (yao2021inversedesignof pages 6-8): Zhenpeng Yao, Benjamín Sánchez-Lengeling, N. Scott Bobbitt, Benjamin J. Bucior, Sai Govind Hari Kumar, Sean P. Collins, Thomas Burns, Tom K. Woo, Omar K. Farha, Randall Q. Snurr, and Alán Aspuru-Guzik. Inverse design of nanoporous crystalline reticular materials with deep generative models. Jan 2021. URL: https://doi.org/10.1038/s42256-020-00271-1, doi:10.1038/s42256-020-00271-1. This article has 470 citations and is from a domain leading peer-reviewed journal.

16. (duan2026theriseof pages 3-7): Chenru Duan, Aditya Nandy, Shyam Chand Pal, Xin Yang, Wenhao Gao, Yuanqi Du, Hendrik Kraß, Yeonghun Kang, Varinia Bernales, Zuyang Ye, Tristan Pyle, Ray Yang, Zeqi Gu, Philippe Schwaller, Shengqian Ma, Shijing Sun, Alán Aspuru-Guzik, Seyed Mohamad Moosavi, Robert Wexler, and Zhiling Zheng. The rise of generative ai for metal-organic framework design and synthesis. Matter, 9(5):102748, May 2026. URL: https://doi.org/10.1016/j.matt.2026.102748, doi:10.1016/j.matt.2026.102748. This article has 12 citations and is from a peer-reviewed journal.

17. (park2024agenerativeartificial pages 1-2): Hyun Park, Xiaoli Yan, Ruijie Zhu, Eliu A. Huerta, Santanu Chaudhuri, Donny Cooper, Ian T. Foster, and E. Tajkhorshid. A generative artificial intelligence framework based on a molecular diffusion model for the design of metal-organic frameworks for carbon capture. Communications Chemistry, Feb 2024. URL: https://doi.org/10.1038/s42004-023-01090-2, doi:10.1038/s42004-023-01090-2. This article has 112 citations and is from a peer-reviewed journal.

18. (park2024agenerativeartificial pages 7-9): Hyun Park, Xiaoli Yan, Ruijie Zhu, Eliu A. Huerta, Santanu Chaudhuri, Donny Cooper, Ian T. Foster, and E. Tajkhorshid. A generative artificial intelligence framework based on a molecular diffusion model for the design of metal-organic frameworks for carbon capture. Communications Chemistry, Feb 2024. URL: https://doi.org/10.1038/s42004-023-01090-2, doi:10.1038/s42004-023-01090-2. This article has 112 citations and is from a peer-reviewed journal.

19. (park2023ghpmofassemblediffusionmodeling pages 1-4): Hyun Park, Xiaoli Yan, Ruijie Zhu, E. Huerta, Santanu Chaudhuri, Donny Cooper, Ian T. Foster, and Emad Tajkhorshid. Ghp-mofassemble: diffusion modeling, high throughput screening, and molecular dynamics for rational discovery of novel metal-organic frameworks for carbon capture at scale. ArXiv, Jun 2023. URL: https://doi.org/10.21203/rs.3.rs-3084157/v1, doi:10.21203/rs.3.rs-3084157/v1. This article has 4 citations.

20. (fu2023mofdiffcoarsegraineddiffusion pages 1-2): Xiang Fu, Tian Xie, Andrew S. Rosen, Tommi Jaakkola, and Jake Smith. Mofdiff: coarse-grained diffusion for metal-organic framework design. ArXiv, Oct 2024. URL: https://doi.org/10.48550/arxiv.2310.10732, doi:10.48550/arxiv.2310.10732. This article has 52 citations.

21. (han2025egmofefficientgeneration pages 26-31): Seunghee Han, Y. Kang, Taeun Bae, Varinia Bernales, Alán Aspuru-Guzik, and Jihan Kim. Egmof: efficient generation of metal-organic frameworks using a hybrid diffusion-transformer architecture. ArXiv, Nov 2025. URL: https://doi.org/10.48550/arxiv.2511.03122, doi:10.48550/arxiv.2511.03122. This article has 2 citations.

22. (badrinarayanan2025mofgptgenerativedesign pages 1-4): Srivathsan Badrinarayanan, Rishikesh Magar, Akshay Antony, Radheesh Sharma Meda, and Amir Barati Farimani. Mofgpt: generative design of metal–organic frameworks using language models. Journal of Chemical Information and Modeling, 65:9049-9060, Aug 2025. URL: https://doi.org/10.1021/acs.jcim.5c01625, doi:10.1021/acs.jcim.5c01625. This article has 30 citations and is from a peer-reviewed journal.

23. (cleeton2025inversedesignof pages 1-2): Conor Cleeton and Lev Sarkisov. Inverse design of metal-organic frameworks using deep dreaming approaches. Nature Communications, May 2025. URL: https://doi.org/10.1038/s41467-025-59952-3, doi:10.1038/s41467-025-59952-3. This article has 30 citations and is from a highest quality peer-reviewed journal.

24. (griffiths2020constrainedbayesianoptimization pages 1-2): Ryan-Rhys Griffiths and José Miguel Hernández-Lobato. Constrained bayesian optimization for automatic chemical design using variational autoencoders. JournalArticle, Oct 2020. URL: https://doi.org/10.17863/cam.76888, doi:10.17863/cam.76888. This article has 608 citations.

25. (moss2507returnofthe pages 2-3): Henry B. Moss, Sebastian W. Ober, and Tom Diethe. Return of the latent space cowboys: re-thinking the use of vaes for bayesian optimisation of structured spaces. ArXiv, Jul 2507. URL: https://doi.org/10.48550/arxiv.2507.03910, doi:10.48550/arxiv.2507.03910. This article has 12 citations.

26. (griffiths2020constrainedbayesianoptimization pages 2-3): Ryan-Rhys Griffiths and José Miguel Hernández-Lobato. Constrained bayesian optimization for automatic chemical design using variational autoencoders. JournalArticle, Oct 2020. URL: https://doi.org/10.17863/cam.76888, doi:10.17863/cam.76888. This article has 608 citations.

27. (maus2022locallatentspace pages 3-5): Natalie Maus, Haydn T. Jones, Juston S. Moore, Matt J. Kusner, John Bradshaw, and Jacob R. Gardner. Local latent space bayesian optimization over structured inputs. Preprint, Jan 2022. URL: https://doi.org/10.48550/arxiv.2201.11872, doi:10.48550/arxiv.2201.11872. This article has 147 citations.

28. (xie2021crystaldiffusionvariational pages 1-2): Tian Xie, Xiang Fu, Octavian-Eugen Ganea, Regina Barzilay, and Tommi Jaakkola. Crystal diffusion variational autoencoder for periodic material generation. Preprint, Jan 2021. URL: https://doi.org/10.48550/arxiv.2110.06197, doi:10.48550/arxiv.2110.06197. This article has 587 citations.

29. (xie2021crystaldiffusionvariational pages 2-4): Tian Xie, Xiang Fu, Octavian-Eugen Ganea, Regina Barzilay, and Tommi Jaakkola. Crystal diffusion variational autoencoder for periodic material generation. Preprint, Jan 2021. URL: https://doi.org/10.48550/arxiv.2110.06197, doi:10.48550/arxiv.2110.06197. This article has 587 citations.

30. (pakornchote2024diffusionprobabilisticmodels pages 5-6): Teerachote Pakornchote, Natthaphon Choomphon-anomakhun, Sorrjit Arrerut, Chayanon Atthapak, Sakarn Khamkaeo, Thiparat Chotibut, and Thiti Bovornratanaraks. Diffusion probabilistic models enhance variational autoencoder for crystal structure generative modeling. Scientific Reports, Aug 2024. URL: https://doi.org/10.48550/arxiv.2308.02165, doi:10.48550/arxiv.2308.02165. This article has 52 citations and is from a peer-reviewed journal.

31. (simkus2025mofasaastep pages 19-22): Vaidotas Simkus, Anders S. Christensen, Steven Bennett, I. Johnson, Mark Neumann, James Gin, Jonathan Godwin, and Benjamin Rhodes. Mofasa: a step change in metal-organic framework generation. ArXiv, Dec 2025. URL: https://doi.org/10.48550/arxiv.2512.01756, doi:10.48550/arxiv.2512.01756. This article has 3 citations.

32. (simkus2025mofasaastep pages 22-24): Vaidotas Simkus, Anders S. Christensen, Steven Bennett, I. Johnson, Mark Neumann, James Gin, Jonathan Godwin, and Benjamin Rhodes. Mofasa: a step change in metal-organic framework generation. ArXiv, Dec 2025. URL: https://doi.org/10.48550/arxiv.2512.01756, doi:10.48550/arxiv.2512.01756. This article has 3 citations.

33. (li2025materialsgenerationin pages 9-10): Zhixun Li, Bin Cao, Rui Jiao, Liang Wang, Ding Wang, Yang Liu, Dingshuo Chen, Jia Li, Qiang Liu, Yu Rong, Tong-yi Zhang, and Jeffrey Xu Yu. Materials generation in the era of artificial intelligence: a comprehensive survey. ArXiv, May 2025. URL: https://doi.org/10.48550/arxiv.2505.16379, doi:10.48550/arxiv.2505.16379. This article has 16 citations.

34. (gonzalezduque2024asurveyand pages 4-5): Miguel González-Duque, Richard Michael, Simon Bartels, Yevgen Zainchkovskyy, Søren Hauberg, and Wouter Boomsma. A survey and benchmark of high-dimensional bayesian optimization of discrete sequences. Jan 2024. URL: https://doi.org/10.48550/arxiv.2406.04739, doi:10.48550/arxiv.2406.04739. This article has 34 citations.

35. (maus2022locallatentspace pages 1-3): Natalie Maus, Haydn T. Jones, Juston S. Moore, Matt J. Kusner, John Bradshaw, and Jacob R. Gardner. Local latent space bayesian optimization over structured inputs. Preprint, Jan 2022. URL: https://doi.org/10.48550/arxiv.2201.11872, doi:10.48550/arxiv.2201.11872. This article has 147 citations.

36. (maus2022locallatentspace pages 9-11): Natalie Maus, Haydn T. Jones, Juston S. Moore, Matt J. Kusner, John Bradshaw, and Jacob R. Gardner. Local latent space bayesian optimization over structured inputs. Preprint, Jan 2022. URL: https://doi.org/10.48550/arxiv.2201.11872, doi:10.48550/arxiv.2201.11872. This article has 147 citations.

37. (abeer2024multiobjectivelatentspace pages 2-3): A N M Nafiz Abeer, Nathan Urban, M Ryan Weil, Francis J. Alexander, and Byung-Jun Yoon. Multi-objective latent space optimization of generative molecular design models. Patterns, Mar 2024. URL: https://doi.org/10.48550/arxiv.2203.00526, doi:10.48550/arxiv.2203.00526. This article has 50 citations and is from a peer-reviewed journal.

38. (abeer2024multiobjectivelatentspace pages 9-11): A N M Nafiz Abeer, Nathan Urban, M Ryan Weil, Francis J. Alexander, and Byung-Jun Yoon. Multi-objective latent space optimization of generative molecular design models. Patterns, Mar 2024. URL: https://doi.org/10.48550/arxiv.2203.00526, doi:10.48550/arxiv.2203.00526. This article has 50 citations and is from a peer-reviewed journal.

39. (shen2021deepmoleculardreaming pages 3-5): Cynthia Shen, Mario Krenn, Sagi Eppel, and Alán Aspuru-Guzik. Deep molecular dreaming: inverse machine learning for de-novo molecular design and interpretability with surjective representations. Machine Learning: Science and Technology, 2:03LT02, Jul 2021. URL: https://doi.org/10.1088/2632-2153/ac09d6, doi:10.1088/2632-2153/ac09d6. This article has 74 citations and is from a peer-reviewed journal.

40. (shen2021deepmoleculardreaming pages 1-3): Cynthia Shen, Mario Krenn, Sagi Eppel, and Alán Aspuru-Guzik. Deep molecular dreaming: inverse machine learning for de-novo molecular design and interpretability with surjective representations. Machine Learning: Science and Technology, 2:03LT02, Jul 2021. URL: https://doi.org/10.1088/2632-2153/ac09d6, doi:10.1088/2632-2153/ac09d6. This article has 74 citations and is from a peer-reviewed journal.

41. (muthyala2025generativemultiobjectivebayesian pages 11-15): Madhav R. Muthyala, Farshud Sorourifar, Tianhong Tan, You Peng, and Joel A. Paulson. Generative multiobjective bayesian optimization with scalable batch evaluations for sample-efficient de novo molecular design. Industrial &amp; Engineering Chemistry Research, 65:628-642, Dec 2025. URL: https://doi.org/10.1021/acs.iecr.5c03166, doi:10.1021/acs.iecr.5c03166. This article has 2 citations and is from a peer-reviewed journal.

42. (moss2507returnofthe pages 1-2): Henry B. Moss, Sebastian W. Ober, and Tom Diethe. Return of the latent space cowboys: re-thinking the use of vaes for bayesian optimisation of structured spaces. ArXiv, Jul 2507. URL: https://doi.org/10.48550/arxiv.2507.03910, doi:10.48550/arxiv.2507.03910. This article has 12 citations.

43. (abeer2024multiobjectivelatentspace pages 1-2): A N M Nafiz Abeer, Nathan Urban, M Ryan Weil, Francis J. Alexander, and Byung-Jun Yoon. Multi-objective latent space optimization of generative molecular design models. Patterns, Mar 2024. URL: https://doi.org/10.48550/arxiv.2203.00526, doi:10.48550/arxiv.2203.00526. This article has 50 citations and is from a peer-reviewed journal.

44. (hamidizadeh2208semisupervisedjunctiontree pages 1-2): Atia Hamidizadeh, Tony Shen, and Martin Ester. Semi-supervised junction tree variational autoencoder for molecular property prediction. ArXiv, Jan 2208. URL: https://doi.org/10.48550/arxiv.2208.05119, doi:10.48550/arxiv.2208.05119. This article has 1 citations.

45. (talibart2025learningachemistryaware pages 10-12): Hugo Talibart and Dimitri Gilis. Learning a chemistry-aware latent space for molecular encoding and generation with a large-scale transformer variational autoencoder. bioRxiv, Dec 2025. URL: https://doi.org/10.64898/2025.12.19.695394, doi:10.64898/2025.12.19.695394. This article has 0 citations.

46. (haddad2025targetedmoleculargeneration pages 7-9): Ragy Haddad, Eleni Litsa, Zhen Liu, Xin Yu, Daniel Burkhardt, Danny Reidenbach, Tyler Shimko, and Govinda Bhisetti. Targeted molecular generation with latent reinforcement learning. Scientific Reports, Apr 2025. URL: https://doi.org/10.1038/s41598-025-99785-0, doi:10.1038/s41598-025-99785-0. This article has 18 citations and is from a peer-reviewed journal.

47. (bishnoi2305materialsinformaticsan pages 28-30): Bhupesh Bishnoi. Materials informatics: an algorithmic design rule. ArXiv, May 2305. URL: https://doi.org/10.48550/arxiv.2305.03797, doi:10.48550/arxiv.2305.03797. This article has 4 citations.

48. (oftelie2018activelearningfor pages 1-2): Lindsay Bassman Oftelie, Pankaj Rajak, Rajiv K. Kalia, Aiichiro Nakano, Fei Sha, Jifeng Sun, David J. Singh, Muratahan Aykol, Patrick Huck, Kristin Persson, and Priya Vashishta. Active learning for accelerated design of layered materials. npj Computational Materials, 4:1-9, Dec 2018. URL: https://doi.org/10.1038/s41524-018-0129-0, doi:10.1038/s41524-018-0129-0. This article has 208 citations and is from a peer-reviewed journal.

49. (kristiadi2024asoberlook pages 4-5): Agustinus Kristiadi, Felix Strieth-Kalthoff, Marta Skreta, Pascal Poupart, Alán Aspuru-Guzik, and Geoff Pleiss. A sober look at llms for material discovery: are they actually good for bayesian optimization over molecules? ArXiv, Feb 2024. URL: https://doi.org/10.48550/arxiv.2402.05015, doi:10.48550/arxiv.2402.05015. This article has 69 citations.

50. (kristiadi2024asoberlook pages 1-2): Agustinus Kristiadi, Felix Strieth-Kalthoff, Marta Skreta, Pascal Poupart, Alán Aspuru-Guzik, and Geoff Pleiss. A sober look at llms for material discovery: are they actually good for bayesian optimization over molecules? ArXiv, Feb 2024. URL: https://doi.org/10.48550/arxiv.2402.05015, doi:10.48550/arxiv.2402.05015. This article has 69 citations.

51. (kristiadi2024asoberlook pages 2-3): Agustinus Kristiadi, Felix Strieth-Kalthoff, Marta Skreta, Pascal Poupart, Alán Aspuru-Guzik, and Geoff Pleiss. A sober look at llms for material discovery: are they actually good for bayesian optimization over molecules? ArXiv, Feb 2024. URL: https://doi.org/10.48550/arxiv.2402.05015, doi:10.48550/arxiv.2402.05015. This article has 69 citations.