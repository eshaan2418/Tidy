# Tidy: Prior Work

**Verification date:** 2026-10-24.
**How entries were checked:** I confirmed each arXiv entry's title, first author, and first-submission year against the arXiv export API (`export.arxiv.org/api/query?id_list=...`). Where I mention a paper's method, that comes from its arXiv abstract. Venues are listed only when the arXiv comment or journal-ref field states them, or when the ICLR proceedings page or project page confirms them in a web search. Everything else is marked `arXiv <year>`, so a missing venue means I did not confirm one, not that the paper is unpublished. I opened each GitHub URL on the verification date and it returned HTTP 200. Anything I could not verify is listed in the last section and is not cited.

Entry format: **Title**, first author et al. (venue or arXiv, year), ID or URL, then what it means for Tidy.

---

## 1. Learning manipulation from human videos

- **DexMV: Imitation Learning for Dexterous Manipulation from Human Videos**, Qin et al. (arXiv 2021). arXiv:2108.05877. Builds a pipeline from human video to 3D hand and object poses to robot demonstrations. It targets dexterous hands, while Tidy targets a parallel gripper.
- **R3M: A Universal Visual Representation for Robot Manipulation**, Nair et al. (CoRL 2022). arXiv:2203.12601. Pretrains a visual representation on human video (Ego4D). Tidy could use it as a frozen encoder baseline.
- **VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training**, Ma et al. (ICLR 2023). arXiv:2210.00030. Learns a reward and representation from human video. It is relevant to Tidy's learned success and progress signals.
- **Human-to-Robot Imitation in the Wild (WHIRL)**, Bahl et al. (RSS 2022). arXiv:2207.09450. Does one-shot imitation from a human video and then improves through robot self-practice. This is a close analogue to Tidy's "video plus brief self-practice" design.
- **VideoDex: Learning Dexterity from Internet Videos**, Shaw et al. (CoRL 2022). arXiv:2212.04498. Uses internet video as a prior for dexterous policies.
- **MimicPlay: Long-Horizon Imitation Learning by Watching Human Play**, Wang et al. (CoRL 2023). arXiv:2302.12422. Learns high-level plans from human play video and low-level control from robot teleop data. It still needs teleoperation.
- **AR2-D2: Training a Robot Without a Robot**, Duan et al. (arXiv 2023). arXiv:2306.13818. Collects demonstrations with a phone and AR and needs no physical robot. It is close to Tidy's phone-only data collection.
- **Any-point Trajectory Modeling for Policy Learning (ATM)**, Wen et al. (arXiv 2023). arXiv:2401.00025. Predicts point tracks from video and uses them to guide a policy, and it can use human or out-of-domain video.
- **DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation**, Wang et al. (arXiv 2024). arXiv:2403.07788. Captures human hand data with a mocap glove and cameras. The specialized hardware contrasts with Tidy's phone-only capture.
- **Vid2Robot: End-to-end Video-conditioned Policy Learning with Cross-Attention Transformers**, Jain et al. (RSS 2024). arXiv:2403.12943. Trains a policy conditioned on a human video at large scale using robot data.
- **Track2Act: Predicting Point Tracks from Internet Videos enables Generalizable Robot Manipulation**, Bharadhwaj et al. (ECCV 2024). arXiv:2405.01527. Predicts point tracks from web video and adds a residual policy.
- **HumanPlus: Humanoid Shadowing and Imitation from Humans**, Fu et al. (arXiv 2024). arXiv:2406.10454. Humanoid shadowing and imitation from human pose estimates.
- **Dreamitate: Real-World Visuomotor Policy Learning via Video Generation**, Liang et al. (arXiv 2024). arXiv:2406.16862. Fine-tunes a video generator on human demonstrations and tracks a tool in the generated video.
- **Flow as the Cross-Domain Manipulation Interface (Im2Flow2Act)**, Xu et al. (CoRL 2024). arXiv:2407.15208. Uses object flow from human video as the interface for transfer to the robot.
- **Hand-Object Interaction Pretraining from Videos**, Singh et al. (arXiv 2024). arXiv:2409.08273. Pretrains on hand-object interaction video.
- **Gen2Act: Human Video Generation in Novel Scenarios enables Generalizable Robot Manipulation**, Bharadhwaj et al. (arXiv 2024). arXiv:2409.16283. Generates a human video and then runs a video-conditioned policy.
- **OKAMI: Teaching Humanoid Robots Manipulation Skills through Single Video Imitation**, Li et al. (CoRL 2024). arXiv:2410.11792. Imitates from a single RGB-D human video using an object-aware retargeting stage.
- **EgoMimic: Scaling Imitation Learning via Egocentric Video**, Kareer et al. (arXiv 2024). arXiv:2410.24221. Co-trains on egocentric human video and robot data.
- **Motion Tracks: A Unified Representation for Human-Robot Transfer in Few-Shot Imitation Learning**, Ren et al. (arXiv 2025). arXiv:2501.06994. Uses a shared action space of motion tracks for few-shot transfer from human to robot.
- **Point Policy: Unifying Observations and Actions with Key Points for Robot Manipulation**, Haldar et al. (arXiv 2025). arXiv:2502.20391. Learns from human video through keypoints.
- **Phantom: Training Robots Without Robots Using Only Human Videos**, Lepert et al. (CoRL 2025). arXiv:2503.00779. Turns human video into robot data with no robot data at all. This is very close to Tidy's "no teleop" premise.
- **Humanoid Policy ~ Human Policy**, Qiu et al. (arXiv 2025). arXiv:2503.13441. Trains a policy on human and humanoid data together.
- **ZeroMimic: Distilling Robotic Manipulation Skills from Web Videos**, Shi et al. (ICRA 2025). arXiv:2503.23877. Learns skills zero-shot from egocentric web video.
- **Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration**, Lum et al. (arXiv 2025). arXiv:2504.12609. Goes from one human RGB-D video through simulation RL to a real robot.
- **X-Sim: Cross-Embodiment Learning via Real-to-Sim-to-Real**, Dan et al. (arXiv 2025). arXiv:2505.07096. Uses object motion from human video as the learning signal, with no teleoperation.
- **EgoZero: Robot Learning from Smart Glasses**, Liu et al. (arXiv 2025). arXiv:2505.20290. Robot learning from egocentric smart-glasses video.
- **Masquerade: Learning from In-the-wild Human Videos using Data-Editing**, Lepert et al. (ICRA 2026). arXiv:2508.09976. Edits human video to look robotized and co-trains with 50 robot demonstrations per task.
- **EasyMimic: A Low-Cost Framework for Robot Imitation Learning from Human Videos**, Zhang et al. (ICRA 2026). arXiv:2602.11464. Extracts 3D hand trajectories from RGB video, maps them to the gripper of a low-cost LeRobot-platform arm, adds hand visual augmentation, and co-trains with a small amount of robot data. **This is the closest prior work to Tidy's data pipeline** (see Section 8).
- **What Matters When Cotraining Robot Manipulation Policies on Everyday Human Videos?**, Li et al. (arXiv 2026). arXiv:2606.06627. Finds that hand-pose quality and the human-robot motion gap both limit transfer. This directly bears on Tidy's retargeting risk.
- **HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos**, Wang et al. (arXiv 2026). arXiv:2605.24934. Learns from a few minutes of egocentric video with no robot data.
- **Robotizing Human Videos with Physically Consistent Interactions**, Cheng et al. (arXiv 2026). arXiv:2610.06137. Converts reconstructed hand contacts into parallel-jaw grasps and composites a robot into the video.
- **Robot Learning from Human Videos: A Survey**, Ma et al. (arXiv 2026). arXiv:2604.27621. A recent survey to check for coverage gaps.

## 2. Hand-to-robot retargeting and 3D hand pose

- **Embodied Hands: Modeling and Capturing Hands and Bodies Together (MANO)**, Romero et al. (ACM TOG / SIGGRAPH Asia 2017). arXiv:2201.02610. The MANO hand model that HaMeR and WiLoR output.
- **DexPilot: Vision Based Teleoperation of Dexterous Robotic Hand-Arm System**, Handa et al. (arXiv 2019). arXiv:1910.03135. An early vision-based hand retargeting system.
- **MediaPipe Hands: On-device Real-time Hand Tracking**, Zhang et al. (CVPR Workshops 2020). arXiv:2006.10214. A lightweight 2D/2.5D hand-keypoint baseline that can run on a phone.
- **FrankMocap: Fast Monocular 3D Hand and Body Motion Capture by Regression and Integration**, Rong et al. (arXiv 2020). arXiv:2008.08324. Monocular 3D hand capture. It is an older baseline.
- **AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System**, Qin et al. (RSS 2023). arXiv:2307.04577. Its retargeting optimizers are released as `dex-retargeting`.
- **dex-retargeting (software)**, dexsuite. https://github.com/dexsuite/dex-retargeting. Optimizers that map human hand motion to robot hands. Tidy would reduce this to a wrist pose plus a gripper width.
- **Reconstructing Hands in 3D with Transformers (HaMeR)**, Pavlakos et al. (arXiv 2023). arXiv:2312.05251. A candidate monocular MANO estimator for Tidy.
- **WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild**, Potamias et al. (CVPR 2025). arXiv:2409.12259. A candidate detector and reconstructor for Tidy, and an alternative to HaMeR.
- **Robot See Robot Do: Imitating Articulated Object Manipulation with Monocular 4D Reconstruction**, Kerr et al. (CoRL 2024). arXiv:2409.18121. An object-centric alternative to hand retargeting.
- **Hand-to-parallel-gripper mappings** already exist in Phantom (2503.00779), EasyMimic (2602.11464), Masquerade (2508.09976), and Robotizing Human Videos (2610.06137). Tidy's thumb-and-index-finger-to-gripper mapping should be framed as an engineering instance of these, not as a contribution.

## 3. Perception foundation models

- **Segment Anything**, Kirillov et al. (arXiv 2023). arXiv:2304.02643.
- **SAM 2: Segment Anything in Images and Videos**, Ravi et al. (arXiv 2024). arXiv:2408.00714. Tidy uses it to segment and track objects and hands in video.
- **CoTracker: It is Better to Track Together**, Karaev et al. (arXiv 2023). arXiv:2307.07635.
- **CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos**, Karaev et al. (arXiv 2024). arXiv:2410.11831. Tidy uses it for dense point tracks of objects.
- **FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects**, Wen et al. (arXiv 2023). arXiv:2312.08344. Provides 6D object pose for placing and stacking.
- **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data**, Yang et al. (CVPR 2024). arXiv:2401.10891.
- **Depth Anything V2**, Yang et al. (arXiv 2024). arXiv:2406.09414. Gives monocular depth from phone RGB.
- **Video Depth Anything: Consistent Depth Estimation for Super-Long Videos**, Chen et al. (arXiv 2025). arXiv:2501.12375. Gives temporally consistent depth for the videos.
- **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**, Liu et al. (arXiv 2023). arXiv:2303.05499. Can supply text prompts as boxes to seed SAM 2.
- **DINOv2: Learning Robust Visual Features without Supervision**, Oquab et al. (arXiv 2023). arXiv:2304.07193. A frozen visual backbone for the policies and the success classifier.

## 4. Low-cost arms and LeRobot

- **SO-ARM100 / SO-101 (hardware)**, TheRobotStudio. https://github.com/TheRobotStudio/SO-ARM100. Tidy's target arm. The repo marks SO-100 as superseded by SO-101.
- **low_cost_robot (Koch v1.0)**, A. Koch. https://github.com/AlexanderKoch-Koch/low_cost_robot. **koch-v1-1**, J. Moss. https://github.com/jess-moss/koch-v1-1. An alternative low-cost arm that LeRobot supports.
- **LeRobot (software)**, Hugging Face. https://github.com/huggingface/lerobot.
- **LeRobot: An Open-Source Library for End-to-End Robot Learning**, Cadene et al. (ICLR 2026). arXiv:2602.22818. Provides the dataset format, ACT, Diffusion Policy, SmolVLA, and the SO-101 drivers.
- **Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ALOHA / ACT)**, Zhao et al. (arXiv 2023). arXiv:2304.13705. Introduces the ALOHA hardware and the ACT policy.
- **Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation**, Fu et al. (arXiv 2024). arXiv:2401.02117.
- **GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators**, Wu et al. (arXiv 2023). arXiv:2309.13037. Low-cost leader-arm teleoperation, which is the kind of baseline Tidy is meant to replace.
- **Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots (UMI)**, Chi et al. (arXiv 2024). arXiv:2402.10329. Collects data without a robot using a handheld gripper. It is the closest non-video way to avoid teleoperation.
- **SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics**, Shukor et al. (arXiv 2025). arXiv:2506.01844. A VLA trained on community SO-100-class data.
- **Benchmarking Vision-Language-Action Models on SO-101: Failure and Recovery Analysis**, Yu et al. (arXiv 2026). arXiv:2606.08881. Benchmarks π0.5, SmolVLA, Wall-X, and ACT on SO-101 from teleop data, with a failure taxonomy. It is relevant to both Tidy's evaluation and its failure taxonomy.

## 5. Policy classes

- **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**, Chi et al. (RSS 2023; arXiv v5 is the extended journal version). arXiv:2303.04137. The source of Tidy's "diffusion action entropy" uncertainty signal.
- **ACT**, see ALOHA (arXiv:2304.13705) in Section 4.
- **RT-1: Robotics Transformer for Real-World Control at Scale**, Brohan et al. (arXiv 2022). arXiv:2212.06817.
- **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control**, Brohan et al. (arXiv 2023). arXiv:2307.15818.
- **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**, Open X-Embodiment Collaboration (arXiv 2023). arXiv:2310.08864.
- **Octo: An Open-Source Generalist Robot Policy**, Octo Model Team (arXiv 2024). arXiv:2405.12213.
- **OpenVLA: An Open-Source Vision-Language-Action Model**, Kim et al. (arXiv 2024). arXiv:2406.09246.
- **π0: A Vision-Language-Action Flow Model for General Robot Control**, Black et al. (RSS 2025). arXiv:2410.24164.
- **π0.5: a Vision-Language-Action Model with Open-World Generalization**, Physical Intelligence et al. (arXiv 2025). arXiv:2504.16054. Targets household tidying, the same task family as Tidy, but trains on large amounts of robot data.
- **GR00T N1: An Open Foundation Model for Generalist Humanoid Robots**, NVIDIA et al. (arXiv 2025). arXiv:2503.14734.
- **Gemini Robotics: Bringing AI into the Physical World**, Gemini Robotics Team (arXiv 2025). arXiv:2503.20020.

## 6. Active and interactive imitation learning

- **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger)**, Ross et al. (AISTATS 2011). arXiv:1011.0686.
- **EnsembleDAgger: A Bayesian Approach to Safe Imitation Learning**, Menda et al. (IROS 2019). arXiv:1807.08364. Uses ensemble disagreement to gate when the expert takes over. This is the direct precedent for Tidy's ensemble signal.
- **HG-DAgger: Interactive Imitation Learning with Human Experts**, Kelly et al. (arXiv 2018). arXiv:1810.02890.
- **Uncertainty-Aware Data Aggregation for Deep Imitation Learning**, Cui et al. (ICRA 2019). arXiv:1905.02780. Uses ensemble uncertainty to choose which states to query.
- **LazyDAgger: Reducing Context Switching in Interactive Imitation Learning**, Hoque et al. (CASE 2021). arXiv:2104.00053.
- **ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive Imitation Learning**, Hoque et al. (CoRL 2021). arXiv:2109.08273. Gates queries on novelty and risk under a budget. This is the closest precedent for budgeted human queries.
- **Eliciting Compatible Demonstrations for Multi-Human Imitation Learning**, Gandhi et al. (CoRL 2022). arXiv:2210.08073. Actively shapes which demonstrations humans provide.
- **Interactive Imitation Learning in Robotics: A Survey**, Celemin et al. (arXiv 2022). arXiv:2211.00600.
- **Robot Learning on the Job: Human-in-the-Loop Autonomy and Learning During Deployment (Sirius)**, Liu et al. (RSS 2023). arXiv:2211.08416. Learns from human interventions during deployment.
- **Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners (KnowNo)**, Ren et al. (CoRL 2023). arXiv:2307.01928. Uses conformal prediction to decide when to ask for help.
- **Data Quality in Imitation Learning**, Belkhale et al. (arXiv 2023). arXiv:2306.02437.
- **MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations**, Mandlekar et al. (CoRL 2023). arXiv:2310.17596. Its subtask-segmented demonstrations relate to Tidy's phase decomposition.
- **IntervenGen: Interventional Data Generation for Robust and Data-Efficient Robot Imitation Learning**, Hoque et al. (arXiv 2024). arXiv:2405.01472.
- **Active Fine-Tuning of Multi-Task Policies (AMF)**, Bagatella et al. (arXiv 2024). arXiv:2410.05026. Lets the agent adaptively choose **which task to have demonstrated next**. **This is the closest prior work to Tidy's targeted demo requests.**
- **Conformalized Interactive Imitation Learning (ConformalDAgger)**, Zhao et al. (arXiv 2024). arXiv:2410.08852.
- **Diff-DAgger: Uncertainty Estimation with Diffusion Policy for Robotic Manipulation**, Lee et al. (ICRA 2025, per arXiv comment). arXiv:2410.14868. Uses the diffusion loss as an uncertainty signal to trigger queries. **This is a direct precedent for Tidy's diffusion-uncertainty signal.**
- **Robot Data Curation with Mutual Information Estimators (DemInf)**, Hejna et al. (RSS 2025). arXiv:2502.08623. Scores the quality of individual demonstrations.
- **Curating Demonstrations using Online Experience (Demo-SCORE)**, Chen et al. (arXiv 2025). arXiv:2503.03707. Trains a success classifier on rollouts to curate demonstrations. It is a precedent for Tidy's "learned success classifier" signal.
- **ASkDAgger: Active Skill-level Data Aggregation for Interactive Imitation Learning**, Luijkx et al. (TMLR 2025). arXiv:2508.05310.

## 7. Failure detection and recovery

- **Vision-Language Models as Success Detectors**, Du et al. (arXiv 2023). arXiv:2303.07280. A baseline for Tidy's per-phase success checks.
- **REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction**, Liu et al. (CoRL 2023). arXiv:2306.15724.
- **RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning**, Dai et al. (arXiv 2024). arXiv:2409.14674.
- **AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation**, Duan et al. (ICLR 2025). arXiv:2410.00371. Its **FailGen** pipeline, which procedurally perturbs successful RLBench demos, is part of this paper and is not a separate publication. Code: https://github.com/NVlabs/AHA.
- **Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress (Sentinel)**, Agia et al. (CoRL 2024). arXiv:2410.04640. Detects failures from the temporal consistency of diffusion policies.
- **Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure Detection**, Zhou et al. (CVPR 2025). arXiv:2412.04455.
- **Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies (FAIL-Detect)**, Xu et al. (RSS 2025). arXiv:2503.08558.
- **RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction**, Ye et al. (arXiv 2025). arXiv:2505.12224.
- **SAFE: Multitask Failure Detection for Vision-Language-Action Models**, Gu et al. (NeurIPS 2025). arXiv:2506.09937.
- **RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and Correction**, Hu et al. (arXiv 2025). arXiv:2509.07953. Collects human rewind-and-correct data per subtask.
- **Guardian: Detecting Robotic Planning and Execution Errors with Vision-Language Models**, Pacaud et al. (arXiv 2025). arXiv:2512.01946. Verifies each step with a VLM and retries.
- **Rewind-IL: Online Failure Detection and State Respawning for Imitation Learning**, Zheng et al. (IEEE RA-L 2026). arXiv:2604.16683. Uses a conformal-calibrated, training-free detector and returns the robot to a verified intermediate state. **This is the closest prior work to Tidy's "retry only the failed phase".**
- **Phase-Conditioned Imitation Learning with Autonomous Failure Recovery for Robust Deformable Object Manipulation**, Chen et al. (IEEE/ASME T-Mech, accepted). arXiv:2605.29407. Combines a phase predictor, phase-conditioned ACT, and triggered recovery. It is another close precedent for per-phase recovery.
- **Beyond Monotonic Progress: Retry-Supervised Value Learning for Robot Imitation**, Qin et al. (arXiv 2026). arXiv:2606.24633.
- **FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement**, Hao et al. (CoRL 2026). arXiv:2607.01111.
- **FLARE: A Failure-Aware Framework for Autonomous Correction and Recovery in Visual-Language Robotic Manipulation**, Zhao et al. (CVPR 2026). arXiv:2608.26645. Adds subtask-level retry and a reset adapter, and is evaluated in simulation.
- **FARM: Reading Failure Signals from the Internal Predictive States of a Frozen Robotic World Model**, Pei et al. (arXiv 2026). arXiv:2609.11445. Its real-robot evaluation includes SO-101.

---

## 8. Novelty position (conservative)

**We do not claim to be "first" for any individual component.** Every building block below already exists:

| Tidy component | Already exists in |
|---|---|
| Robot policies from human video with no teleop | Phantom, Masquerade, HumanEgo, X-Sim, ZeroMimic, AR2-D2, WHIRL |
| Phone or RGB human video on a **low-cost LeRobot arm** | **EasyMimic** (2602.11464), which also co-trains with a small amount of robot data |
| Hand-to-parallel-gripper retargeting | Phantom, EasyMimic, Masquerade, Robotizing Human Videos |
| Off-the-shelf perception stack (HaMeR or WiLoR, SAM 2, CoTracker, FoundationPose, Depth Anything) | Used throughout Section 1 |
| Video plus robot self-practice | WHIRL; Demo-SCORE (curates using rollouts) |
| Ensemble-disagreement queries | EnsembleDAgger, Cui et al. 2019, ThriftyDAgger |
| Diffusion-policy uncertainty for queries | Diff-DAgger; FAIL-Detect and Sentinel (for detection) |
| Learned success classifier from rollouts | Demo-SCORE; VLM success detectors |
| Choosing *which task or demo* to request next | AMF (Active Fine-Tuning of Multi-Task Policies), Gandhi et al. 2022 |
| Per-phase or subtask failure detection plus local retry | Rewind-IL, Phase-Conditioned IL, FLARE, Guardian, RaC, FAR |
| Failure analysis on SO-101 | Benchmarking VLAs on SO-101 (2606.08881), FARM |

**What appears new, as far as this review can tell.** I found no work that combines all of the following:
1. Only ~10 **phone** videos of human demonstrations, with **zero teleoperated robot demos** (EasyMimic still co-trains with robot data). This targets a ~$250 SO-101-class arm on household tabletop chores.
2. A **closed loop in which the learner chooses which human-video demo to request next** using uncertainty, where the request is for another *phone video* rather than a teleop correction or an intervention. It compares three uncertainty sources (ensemble disagreement, diffusion action entropy, learned success classifier) against random and round-robin requests under the same budget.
3. **Per-phase failure detection with reset-and-retry of only the failed phase**, on the same low-cost platform, applied to a policy that was learned from video.
4. An **equal-human-time comparison** against teleoperated data collection on the same arm. Minutes of the person's time is the budget, not the number of demonstrations.

The claim should be stated as an empirical systems contribution: *we show that this combination works, at this cost point, and measure how human time trades off against success rate.* It should not be stated as a new algorithm.

**Closest prior works and how Tidy differs:**
- **EasyMimic (ICRA 2026):** uses the same low-cost LeRobot arm and the same video-to-gripper idea, but it uses robot co-training data and has no active requests or phase recovery. Tidy must cite it prominently and compare against it, or replicate its recipe, as a baseline.
- **Phantom / Masquerade:** no robot data, or only light co-training, with video-to-robot conversion. They use more expensive arms and have no active requests or phase recovery.
- **AMF / Diff-DAgger / EnsembleDAgger / ThriftyDAgger:** use uncertainty-driven queries over **teleop or expert action labels**, not new human videos.
- **Rewind-IL / Phase-Conditioned IL / FLARE:** do phase or checkpoint recovery for policies trained on teleop data, on higher-end arms or in simulation.
- **WHIRL:** combines human video with self-practice, but has no active requests and no per-phase recovery, and runs on a more expensive arm.

**What would weaken novelty (check before submission):**
- A paper that combines human video with *active* selection of the next video to record. My arXiv title and abstract searches found none, but the 2026 literature moves fast. Re-run the search before submission.
- An EasyMimic follow-up, or any SO-100/101 paper, that removes robot data entirely or adds recovery.
- Results showing that the three uncertainty signals perform no better than round-robin. That would be a valid negative result, but it would reframe the contribution.
- A gain from "per-phase retry" that comes from simply allowing more attempts. Retry-all-phases and same-attempt-budget baselines are needed.
- An equal-time comparison that favours teleop, for example because GELLO-style leader arms are cheap for SO-101. That would undercut the premise and should be reported honestly.
- Self-practice data that silently amounts to robot demonstrations. Be explicit that it is autonomous and unlabelled apart from success.

---

## Unverified (not cited)

- **"FailGen" as a standalone paper.** I found no separate publication. It is the data-generation pipeline inside AHA (arXiv:2410.00371) and is cited only through AHA.
- **Venues for these entries.** These are marked `arXiv <year>` because neither the arXiv metadata nor a quick search confirmed a venue (the DBLP lookup failed due to rate limiting): ACT/ALOHA, HaMeR, SAM 2, CoTracker, FoundationPose, Octo, OpenVLA, RT-1, RT-2, DexMV, DexCap, EgoMimic, HumanPlus, ATM, HG-DAgger, and others. Several of these are widely reported at major venues, but that was not confirmed here.
- **A Koch-arm publication.** I found only GitHub repositories, so those are what is cited.
- **"A New Framework for Query Efficient Active Imitation Learning"** (arXiv:1912.13037). The arXiv-listed author did not match what search snippets implied, so I omitted it.
