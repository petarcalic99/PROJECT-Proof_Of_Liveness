# Bibliography

This bibliography covers the research foundations of the Proof of Liveness project, a system for decentralized, privacy-preserving human verification. The referenced works span zero-knowledge proofs, blockchain-based authentication, bot detection and fraud prevention, adversarial machine learning, biometric liveness detection, and core machine learning techniques. Together, they form the theoretical and practical basis for building a robust, on-device liveness attestation system that resists both automated bots and adversarial attacks.

## Zero-Knowledge Proofs & Blockchain

### ZKSENSE: A Friction-less Privacy-Preserving Human Attestation Mechanism for Mobile Devices
**Inigo Querejeta-Azurmendi, Panagiotis Papadopoulos, Matteo Varvello, Antonio Nappa, Jiexin Zhang, Benjamin Livshits** | 2021

Presents zkSENSE, a zero-knowledge proof-based human attestation system for mobile devices that classifies motion sensor outputs (accelerometer and gyroscope) to distinguish humans from bots. The system performs on-device classification using an SVM model and encloses the result in a zero-knowledge proof (zkSVM) for integrity verification, achieving 92% accuracy with attestation taking about 3 seconds. Unlike traditional CAPTCHAs, it requires no user interaction and preserves privacy by keeping sensor data on-device.

*Relevance: This is among the most directly relevant papers, as it combines lightweight ML classification with zero-knowledge proofs for decentralized, privacy-preserving human verification on mobile devices, closely mirroring the Proof of Liveness project's architecture.*

---

### Cairo -- a Turing-complete STARK-friendly CPU architecture
**Lior Goldberg, Shahar Papini, Michael Riabzev** | 2021

Describes Cairo, a practically-efficient Turing-complete CPU architecture designed for generating STARK proofs of computational integrity. Instead of requiring developers to write polynomial equations for each statement to prove, Cairo allows writing programs in a high-level language that compiles to a single set of polynomial constraints, enabling general-purpose zero-knowledge proof generation. The architecture supports von Neumann design principles, nondeterminism, and builtins for common cryptographic operations.

*Relevance: Cairo provides a key infrastructure layer for implementing ZK proof systems in the Proof of Liveness project, as it enables writing verifiable computation programs that can attest to the correctness of neural network inference or liveness checks.*

---

### DAuth: A Decentralized Web Authentication System using Ethereum based Blockchain
**Shibasis Patel, Anisha Sahoo, Bhabendu Kumar Mohanta, Soumyashree S Panda, Debasish Jena** | 2019

Proposes DAuth, a decentralized web authentication protocol built on the Ethereum blockchain as an alternative to OAuth 2.0. The system uses Ethereum smart contracts and Web3 to enable user authentication without relying on third-party identity providers, thereby enhancing transparency and user control over identity data. A working prototype demonstrates the feasibility of blockchain-based authentication that eliminates single points of failure and central data control.

*Relevance: DAuth provides a blueprint for decentralized identity and authentication on blockchain that directly informs the Proof of Liveness project's need for a trustless, decentralized human verification system.*

---

### Robust KYC via Distributed Ledger Technology
**Matus Drgon, Lamprini Georgiou, Aggelos Kiayias** | 2020

Addresses the costly Know-Your-Customer (KYC) process by proposing a novel risk-limiting smart contract mechanism on distributed ledger technology that enables financial institutions to share KYC verification costs. The mechanism fine-tunes the trade-off between security and cost-efficiency, repairing the brittleness of previous DLT-based KYC solutions. The authors provide a theoretical security analysis and an implementation as a Solidity smart contract.

*Relevance: This paper demonstrates how blockchain smart contracts can coordinate identity verification processes across institutions, providing relevant patterns for the Proof of Liveness project's decentralized verification framework.*

## Biometrics & Liveness Detection

### Face Recognition With Radial Basis Function (RBF) Neural Networks
**Meng Joo Er, Shiqian Wu, Juwei Lu, Hock Lye Toh** | 2002

Presents a face recognition system using Radial Basis Function (RBF) neural networks designed for small training sets of high dimension. The approach first extracts features using PCA, then applies Fisher's Linear Discriminant (FLD) for dimensionality reduction, and finally uses an RBF neural classifier with a novel homogeneous clustering paradigm and hybrid learning algorithm. Experiments on the ORL database achieve excellent classification performance with high learning efficiency.

*Relevance: This paper provides techniques for efficient face recognition with small datasets using lightweight neural networks, relevant to the Proof of Liveness project's need for computationally efficient biometric models suitable for on-device deployment.*

## Bot Detection & Fraud

### BeCAPTCHA-Mouse: Synthetic Mouse Trajectories and Improved Bot Detection
**Alejandro Acien, Aythami Morales, Julian Fierrez, Ruben Vera-Rodriguez** | 2021

Presents BeCAPTCHA-Mouse, a bot detector based on a neuromotor model of mouse dynamics combined with synthetic mouse trajectory generation using both heuristic functions and Generative Adversarial Networks (GANs). The system achieves 93% accuracy in detecting high-realism bot trajectories from a single mouse trajectory, and when fused with state-of-the-art features, detection accuracy improves by over 36%. A public benchmark of 15,000 mouse trajectories (from 58 real users and various bots) is introduced.

*Relevance: This work demonstrates behavioral biometric-based bot detection using neural networks and synthetic data augmentation, directly relevant to Proof of Liveness's goal of distinguishing human from automated interactions.*

---

### Clustering Web Users By Mouse Movement to Detect Bots and Botnet Attacks
**Justin Morgan** | 2021

A master's thesis proposing a novel unsupervised learning approach to detect sophisticated web bots by clustering users based on mouse movement behavioral biometrics. The approach uses feature profiles from mouse dynamics (speed, acceleration, trajectory patterns) and applies clustering to differentiate users without requiring labeled data, functioning as a Human Observational Proof (HOP) that avoids user friction. The method aims to detect bots classified as sophisticated or advanced persistent that can mimic human behavior.

*Relevance: This thesis provides a direct application of behavioral biometrics for passive human verification that aligns with the Proof of Liveness project's goal of non-intrusive, continuous liveness detection.*

---

### Large-scale Bot Detection for Search Engines
**Hongwen Kang, Kuansan Wang, David Soukal, Fritz Behr, Zijian Zheng** | 2010

Proposes a semi-supervised learning approach for classifying bot-generated web search traffic from genuine human users at scale. The method uses CAPTCHA responses and simple heuristics to generate initial training labels at essentially zero cost, then applies a semi-supervised learning algorithm with unlabeled data to improve classification, achieving a 2:1 improvement over traditional supervised approaches. The work addresses the sampling bias issue inherent in CAPTCHA-based label generation.

*Relevance: This paper provides foundational methods for large-scale human-vs-bot classification using behavioral signals and semi-supervised learning, relevant to the scalability challenges of the Proof of Liveness verification system.*

---

### Real Time Click Fraud Prevention using Multi-level Data Fusion
**Chamila Walgampaya, Mehmed Kantardzic, Roman Yampolskiy** | 2010

Details a Collaborative Click Fraud Detection and Prevention (CCFDP) system using Dempster-Shafer evidence theory for multi-level data fusion to detect and prevent click fraud in real time. The system combines evidence from multiple data sources (IP, referrer, city, country, ISP) and maintains an online database of suspicious parameters to block fraudulent traffic. Testing with real-world ad campaign data shows that multi-level fusion improves click fraud analysis quality.

*Relevance: The multi-sensor fusion approach for real-time fraud detection provides useful architectural patterns for combining multiple liveness signals in the Proof of Liveness project.*

---

### Is image-based CAPTCHA secure against attacks based on machine learning? An experimental study
**Fatmah H. Alqahtani, Fawaz A. Alsulaiman** | 2020

Examines the security of image-based CAPTCHAs (specifically Google reCAPTCHA) against machine learning attacks using deep learning and classical ML algorithms including random forest, CART, bagging, and Naive Bayes. The proposed attack mechanism achieved 85.32% average accuracy and successfully solved 56.29% of reCAPTCHA challenges, demonstrating that current image-based CAPTCHAs provide a false sense of security. The study highlights the vulnerability of traditional human verification methods to ML-based automated solvers.

*Relevance: This paper motivates the Proof of Liveness project by demonstrating that conventional CAPTCHA-based human verification is insufficient against modern ML attacks, necessitating alternative approaches like biometric liveness detection.*

---

### SoK: Machine vs. Machine -- A Systematic Classification of Automated Machine Learning-Based CAPTCHA Solvers
**Antreas Dionysiou, Elias Athanasopoulos** | 2020

A systematization of knowledge surveying 51 papers on automated ML-based CAPTCHA breaking, focusing on text-based CAPTCHAs. The study classifies attack techniques and demonstrates that ML significantly increases accuracy, speed, and abstraction (generalizability across CAPTCHA schemes) of automated solvers. The authors also build their own ML-only classifiers and conclude that fundamentally different approaches to reverse Turing tests are needed for Internet services security.

*Relevance: This comprehensive survey reinforces the need for the Proof of Liveness project by showing that ML-based attacks have rendered traditional CAPTCHAs obsolete, motivating the shift toward biometric and behavioral liveness verification.*

---

### Capture the Bot: Using Adversarial Examples to Improve CAPTCHA Robustness to Bot Attacks
**Dorjan Hitaj, Briland Hitaj, Sushil Jajodia, Luigi V. Mancini** | 2020

Introduces CAPTURE, a novel CAPTCHA scheme that leverages adversarial examples to improve robustness against ML-based bot solvers. The approach generates CAPTCHA images that are easy for humans to solve but incorporate adversarial perturbations that fool deep neural network classifiers, exploiting the transferability of adversarial examples across different models. Empirical evaluations demonstrate that CAPTURE effectively thwarts sophisticated ML-based bot solvers while maintaining human usability.

*Relevance: This work demonstrates the use of adversarial ML techniques to strengthen human verification systems, a concept that could enhance the robustness of the Proof of Liveness system's neural network classifiers against adversarial attacks.*

## Adversarial ML & DNN Robustness

### Explaining and Harnessing Adversarial Examples
**Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy** | 2015

A seminal paper arguing that the vulnerability of neural networks to adversarial examples is primarily due to their linear nature, not nonlinearity or overfitting as previously hypothesized. Introduces the Fast Gradient Sign Method (FGSM) for efficiently generating adversarial examples and demonstrates that adversarial training can serve as a regularization technique. The paper reveals a fundamental tension between model linearity (ease of training) and robustness to adversarial perturbation.

*Relevance: Understanding adversarial vulnerabilities is critical for the Proof of Liveness project, as the neural network classifiers used for liveness detection must be robust against adversarial attacks that attempt to fool the system.*

---

### Deep Convolutional Neural Networks and Noisy Images
**Tiago S. Nazare, Gabriel B. Paranhos da Costa, Welinton A. Contato, Moacir Ponti** | 2017

Evaluates the generalization of deep convolutional neural networks when dealing with different types and levels of image noise (Gaussian and salt-and-pepper), testing on MNIST, CIFAR-10, and SVHN datasets. The study shows that noise makes classification harder, but models trained on noisy data can be more resilient to quality variations in deployment. Denoising methods (Non-Local Means, median filter) help mitigate performance degradation but cannot fully recover noise-free performance.

*Relevance: This work informs the Proof of Liveness project on how to build noise-robust neural networks for biometric classification, important when sensor data quality varies across different mobile devices.*

---

### An empirical study on the effects of different types of noise in image classification tasks
**Gabriel B. Paranhos da Costa, Welinton A. Contato, Tiago S. Nazare, Joao E. S. Batista Neto, Moacir Ponti** | 2017

Analyzes the impact of Gaussian, Poisson, and salt-and-pepper noise on image classification using LBP and HOG feature descriptors with linear SVM classifiers on Corel and Caltech101-600 datasets. The study demonstrates that noise significantly hinders classification performance by making classes harder to separate, and that denoising methods (median filter and Non-Local Means) can partially mitigate the problem but not fully recover noise-free performance levels.

*Relevance: Provides empirical evidence on noise impacts on feature-based classification, relevant to ensuring the Proof of Liveness project's robustness when processing real-world biometric sensor data of varying quality.*

---

### Generalized Out-of-Distribution Detection: A Survey
**Jingkang Yang, Kaiyang Zhou, Yixuan Li, Ziwei Liu** | 2022

A comprehensive survey unifying five closely related detection problems -- anomaly detection, novelty detection, open set recognition, out-of-distribution (OOD) detection, and outlier detection -- under a generalized OOD detection framework. The paper reviews methods ranging from classification-based to density-based to distance-based approaches, clarifying distinctions based on covariate shift versus semantic shift. It covers technical developments in each sub-area and identifies open challenges for trustworthy ML systems.

*Relevance: OOD detection is fundamental to the Proof of Liveness project's ability to reject synthetic or adversarial inputs that fall outside the distribution of genuine human biometric data.*

## Machine Learning Foundations

### Learning Deep Features for One-Class Classification
**Pramuditha Perera, Vishal M. Patel** | 2018

Proposes Deep One-Class (DOC) classification, a deep learning approach for one-class transfer learning that uses labeled data from an unrelated task for feature learning. The method introduces compactness loss and descriptiveness loss with a parallel CNN architecture to produce features with low intra-class variance, combined with template matching for testing. Extensive experiments on anomaly detection, novelty detection, and mobile active authentication datasets show significant improvements over the state of the art.

*Relevance: One-class classification is highly relevant to Proof of Liveness, as the system needs to distinguish genuine human behavior (the known class) from potentially unbounded forms of synthetic or bot behavior (alien classes).*

---

### Zero-Shot Text-to-Image Generation (DALL-E)
**Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever** | 2021

Describes DALL-E, a transformer-based approach for zero-shot text-to-image generation that autoregressively models text and image tokens as a single stream of data using a two-stage training procedure: a discrete variational autoencoder (dVAE) for image tokenization, followed by an autoregressive transformer over concatenated text and image tokens. The system achieves competitive results on MS-COCO without training on it, and demonstrates emergent capabilities like image-to-image translation and combining novel concepts.

*Relevance: DALL-E demonstrates the power of generative models to create realistic synthetic content, highlighting the adversarial threat landscape that the Proof of Liveness project must defend against when verifying the authenticity of visual or biometric data.*

---

### Apprentissage supervise: Apprentissage profond pour la classification d'images (Supervised Learning: Deep Learning for Image Classification)
**Nicolas Perrin-Gilbert** | 2020

A set of lecture slides (in French) covering supervised deep learning for image classification, focusing on the MNIST dataset as a reference benchmark. The slides cover increasingly complex architectures for image classification and include practical exercises (TD) and an introduction to Generative Adversarial Networks (GANs). This is a pedagogical resource rather than a research paper.

*Relevance: Provides foundational educational material on deep learning and image classification techniques that underpin the neural network components used in the Proof of Liveness project.*

## General References

### Symbiotic Interaction: 5th International Workshop, Symbiotic 2016, Revised Selected Papers
**Luciano Gamberini, Anna Spagnolli, Giulio Jacucci, Benjamin Blankertz, Jonathan Freeman (Eds.)** | 2017

Proceedings of the 5th International Workshop on Symbiotic Interaction (Symbiotic 2016), held in Padua, Italy. The volume covers topics at the intersection of human-computer interaction, brain-computer interfaces, physiological computing, and adaptive systems. Papers explore how implicit user signals (e.g., eye gaze, neural signals, physiological data) can be used to create symbiotic human-computer systems that adapt to user states.

*Relevance: The symbiotic interaction paradigm, particularly the use of physiological and behavioral signals for implicit human-computer interaction, provides conceptual background for the Proof of Liveness project's approach to passive liveness verification.*

---

### Advances in Cybernetics, Cognition, and Machine Learning for Communication Technologies
**Vinit Kumar Gunjan, Sabrina Senatore, Amit Kumar, Xiao-Zhi Gao, Suresh Merugu (Eds.)** | 2020

An edited volume in the Lecture Notes in Electrical Engineering series covering advances in cybernetics, cognition, and machine learning applied to communication technologies. The book spans topics including communication engineering, signal and image processing, wireless and mobile communication, IoT, cybersecurity, and control systems. It compiles contributions from researchers working on modern ML techniques applied to various engineering domains.

*Relevance: Provides general reference material on machine learning techniques and cybersecurity applications that form the broader technical context for the Proof of Liveness project's integration of ML and security.*

---

### Information Security: 7th International Conference, ISC 2004, Proceedings
**Kan Zhang, Yuliang Zheng (Eds.)** | 2004

Proceedings of the 7th International Conference on Information Security (ISC 2004), held in Palo Alto, CA. The volume covers a broad range of topics in information security including key management, digital signatures, cryptanalysis, authentication protocols, and privacy-enhancing technologies. Papers present both theoretical advances and practical cryptographic systems for securing digital communications and identity.

*Relevance: Provides foundational security and cryptographic concepts (key management, digital signatures, authentication protocols) that underpin the blockchain and zero-knowledge proof components of the Proof of Liveness project.*
