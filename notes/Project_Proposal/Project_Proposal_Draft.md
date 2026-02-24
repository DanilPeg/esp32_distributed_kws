# Project Proposal (Draft v0.2)

NATIONAL RESEARCH UNIVERSITY HIGHER SCHOOL OF ECONOMICS  
Moscow Institute of Electronics and Mathematics (MIEM)

Program: Informatics and Computer Engineering (BIV223)  
Student: Danil Pegov  
Supervisor: S.N. Polesskiy (per technical assignment)  
Moscow, 2026

Title: Development of a Distributed ESP32-Based Edge AI System for Voice Command Recognition and Image Classification

## Abstract
This project proposes the development of a distributed system of ESP32 microcontroller nodes that perform local inference for voice command recognition and image classification. Each node captures audio via I2S MEMS microphones and/or images via camera modules, runs on-device inference with quantized models, and exchanges commands and status messages over a dedicated communication protocol. Model training will be carried out in an external environment (PC or Google Colab), followed by optimization and conversion for deployment on ESP32 devices. A lightweight web server will visualize recognition results and system decisions. The project will deliver a reproducible pipeline from data preparation to deployment, firmware implementing distributed coordination, and an evaluation of accuracy, latency, memory footprint, and energy efficiency on the target hardware. Documentation will be prepared in accordance with GOST 34.003-90 and the program’s requirements.

## Keywords
ESP32, TinyML, keyword spotting, image classification, distributed sensing, edge inference, ESP-NOW

## Introduction
The bachelor’s project focuses on the design and implementation of a distributed microcontroller network that processes audio and image data locally. The technical assignment explicitly requires that inference be executed directly on ESP32 nodes without transmitting raw audio or video to a server, and that nodes coordinate their actions through a dedicated data exchange protocol. The system must also provide a web-based visualization of recognition results and decisions [1]. These requirements define the system boundaries and motivate the architectural choice of local inference with lightweight communication.

The project is situated within the TinyML paradigm, which targets the deployment of machine learning models on extremely resource-constrained devices such as microcontrollers. TinyML emphasizes compact models, quantization, and careful control of latency and memory [7]. These constraints are critical for ESP32-class hardware and shape the selection of datasets, model architectures, and inference runtime.

The primary goal is to build an end-to-end pipeline that connects dataset preparation, model training, model compression, and on-device inference with embedded networking and visualization. The project will produce firmware for multiple ESP32 nodes, integrate I2S audio and camera inputs, deploy quantized models on microcontrollers, and implement a communication protocol for command and status exchange. The resulting system will be evaluated on-device in terms of recognition accuracy, inference latency, memory usage, and energy efficiency. The produced artifacts will be organized in the repository structure required by the project (firmware, training notebooks, protocol description, and analysis artifacts) [1].

Distributed processing provides two key advantages for the target scenario. First, it enables spatially separated sensing for audio and vision, which increases coverage and makes the system more robust to local noise or occlusions. Second, it allows decisions to be formed collaboratively, where each node contributes a local inference result rather than streaming raw data. This design aligns with the technical requirement to avoid raw data transmission and keeps bandwidth and privacy risks low while preserving responsiveness at the edge [1]. The project therefore emphasizes local inference and explicit coordination between nodes instead of centralized processing.

The scope is constrained by the hardware and software requirements stated in the technical assignment: ESP32-family microcontrollers, I2S MEMS microphones, camera modules, optional LCD/OLED indicators; Arduino IDE for firmware development; Python and Google Colab for training and conversion; and TensorFlow Lite for Microcontrollers for inference. Documentation must comply with GOST 34.003-90. The official milestones are dated 22 February 2026 (project submission), 24 April 2026 (first draft), and 15 May 2026 (final version), with the first date already in the past as of 23 February 2026 [1].

## Related Work
Research on embedded keyword spotting often relies on compact convolutional architectures. The Speech Commands dataset introduced a standard benchmark for limited-vocabulary speech recognition and is widely used for keyword spotting experiments [2]. Early work on small-footprint CNNs demonstrated that convolutional models can achieve acceptable accuracy under tight computational budgets, making them suitable for microcontroller deployment [3]. These results provide a baseline for model selection and training strategies for voice command recognition in constrained environments.

For embedded image classification, efficient convolutional architectures are essential. MobileNets provide depthwise separable convolutional designs that reduce computation while retaining accuracy, and MobileNetV2 further improves efficiency through inverted residual blocks and linear bottlenecks [4], [5]. Recent work such as MicroNets explores architecture search and design choices specifically targeted at TinyML systems, highlighting trade-offs between accuracy and resource usage [8]. For the image modality, a small labeled dataset such as CIFAR-10 provides a compact, well-studied benchmark of natural images that can be adapted to low-resolution inference pipelines [6].

Model compression and quantization are central to TinyML practice because microcontrollers have limited RAM and storage. The TinyML literature emphasizes the need to reduce model size and computational cost while preserving acceptable accuracy [7]. These constraints motivate the planned use of quantized models and careful profiling of inference latency and memory footprint in the project’s evaluation.

On-device inference for microcontrollers is commonly implemented with TensorFlow Lite for Microcontrollers, a microcontroller-focused runtime derived from TensorFlow Lite [9]. For low-latency coordination between ESP32 nodes, ESP-NOW provides a lightweight, connectionless communication mechanism in the Espressif ecosystem and is a plausible candidate for the project’s protocol layer [10]. These sources inform the implementation choices for embedded inference and inter-node communication.

## Methods
The project will be carried out in a sequence of interlinked stages that map directly to the technical assignment.

1. System architecture and hardware definition. The system will be specified as a network of ESP32 nodes with distinct sensing roles (audio, vision, or hybrid) and an optional coordinator node. Each node will perform local inference and broadcast recognition outputs and status updates. The overall architecture and message flow will be defined to satisfy the requirement for distributed data capture and local inference without raw data transfer [1].

2. Data pipeline definition. For voice commands, the Speech Commands dataset will be used as the primary benchmark [2]. Audio will be converted into compact feature representations (e.g., log-mel spectrograms or MFCCs) suitable for microcontroller inference. For images, CIFAR-10 will be used as the initial benchmark and adapted to the target resolution through resizing and normalization [6]. If the camera module output differs substantially from the benchmark distribution, a small supplementary dataset will be collected to reduce domain shift. The final dataset selection and preprocessing will be documented and validated in notebooks.

3. Model design and training. The baseline keyword-spotting model will be a compact CNN informed by prior work on small-footprint architectures [3]. The image classifier will start from a lightweight family such as MobileNet or MobileNetV2, with structural simplifications as required by ESP32 constraints [4], [5]. Additional compact design principles from MicroNets will be considered during model refinement [8]. Training will be performed in Python on a PC or Google Colab, with explicit validation splits and repeatable training scripts.

4. Model optimization and deployment. Trained models will be quantized and converted to TensorFlow Lite for Microcontrollers format for deployment on ESP32 hardware [9]. This stage will include measurement of model size, RAM usage, and inference latency, as well as validation that quantization does not degrade accuracy beyond acceptable limits. The conversion and deployment pipeline will be scripted for reproducibility.

5. Firmware implementation. ESP32 firmware will be written in the Arduino IDE, integrating sensor drivers, inference runtime, and communication components. Audio capture will use I2S MEMS microphone modules, and image capture will use a compatible camera module. The inference pipeline will be embedded within the main loop with explicit timing measurements. Communication will be implemented using ESP-NOW or UDP, with a defined message schema for commands, recognition outputs, and status reports [10]. Reliability mechanisms (e.g., acknowledgments or sequence numbers) will be added if required by the protocol experiments.

6. Web visualization. A lightweight web server will be deployed on a designated node (or gateway) to display live recognition results, node statuses, and decisions. The visualization will serve as evidence that inference is occurring locally and that the network coordination is functional.

7. Evaluation and reporting. The evaluation will measure classification accuracy on held-out datasets, on-device inference latency, memory footprint, and energy consumption. All experiments will include configuration logs and raw measurements to enable replication. The results will be summarized in plots and tables and aligned with the documentation requirements of the project.

In addition to end-to-end evaluation, targeted diagnostic experiments will be conducted to understand system behavior under varying conditions. These include measuring accuracy under different noise levels for audio, testing image inference under varying lighting conditions, and monitoring communication latency as the number of nodes increases. The diagnostics will help identify performance bottlenecks and guide design choices such as model size, sampling rates, and protocol parameters. All such experiments will be documented with explicit hardware configurations and firmware versions to ensure reproducibility.

Where possible, baseline measurements will be collected using a reference keyword-spotting example from the TensorFlow Lite for Microcontrollers ecosystem to establish initial expectations for latency and memory on ESP32-class hardware [9]. These baselines will be treated as guidance for optimization rather than final results and will be complemented by measurements on the project’s trained models.

## Results Anticipated and/or Achieved
At the time of writing, no experimental results have been produced. The anticipated outcomes are:

1. A working multi-node ESP32 network that captures audio and image data and performs local inference without transmitting raw signals [1].
2. Quantized models deployed via TensorFlow Lite for Microcontrollers with documented memory and latency characteristics [9].
3. A functioning communication protocol for inter-node coordination with a described message format and reproducible test logs [10].
4. A web-based visualization of recognition results and system decisions, demonstrating end-to-end system behavior [1].
5. A reproducible repository containing firmware, training notebooks, protocol specifications, and analysis artifacts that support verification of the results.

## Conclusion
This project will deliver a distributed ESP32-based Edge AI system that performs on-device recognition of voice commands and images and coordinates decisions over a lightweight network protocol. The proposal aligns with the technical assignment by emphasizing local inference, external model training with quantization, and web-based visualization [1]. By grounding model selection in established TinyML literature and by documenting the full pipeline from datasets to firmware, the project aims to provide a defensible and reproducible diploma outcome.

Word Count: 1530

## References
[1] “Technical Assignment for the Bachelor’s Thesis: Development of a Distributed Information Processing System Based on Microcontrollers Using Neural Networks for Voice Commands and Images,” internal document, Moscow, 2026.
[2] P. Warden, “Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition,” arXiv:1804.03209 [cs.CL], Apr. 2018, doi: 10.48550/arXiv.1804.03209.
[3] T. N. Sainath and C. Parada, “Convolutional Neural Networks for Small-footprint Keyword Spotting,” in Proc. Interspeech 2015, pp. 1478–1482, 2015, doi: 10.21437/Interspeech.2015-352.
[4] A. G. Howard et al., “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,” arXiv:1704.04861 [cs.CV], Apr. 2017, doi: 10.48550/arXiv.1704.04861.
[5] M. Sandler et al., “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” in Proc. IEEE/CVF CVPR 2018, pp. 4510–4520; arXiv:1801.04381 [cs.CV].
[6] A. Krizhevsky, “Learning Multiple Layers of Features from Tiny Images,” Tech. Rep., University of Toronto, Apr. 2009. Available: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf (accessed 2026-02-23).
[7] P. Warden and D. Situnayake, TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers. O’Reilly Media, Dec. 2019.
[8] C. Banbury et al., “MicroNets: Neural Network Architectures for Deploying TinyML Applications on Commodity Microcontrollers,” Proc. MLSys 2021.
[9] TensorFlow Lite for Microcontrollers, GitHub repository. Available: https://github.com/tensorflow/tflite-micro (accessed 2026-02-23).
[10] Espressif Systems, “ESP-NOW,” ESP-IDF Programming Guide (v5.2.4). Available: https://docs.espressif.com/projects/esp-idf/en/v5.2.4/esp32/api-reference/network/esp_now.html (accessed 2026-02-23).

## Appendices
A. Hardware and software requirements (from technical assignment): ESP32-family MCUs; I2S MEMS microphones; camera modules; optional LCD/OLED indicators; Arduino IDE; Python and Google Colab; TensorFlow (Keras), TensorFlow Lite Converter, TensorFlow Lite for Microcontrollers; communication via ESP-NOW or UDP [1].
B. Milestones (from technical assignment): project submission by 22 February 2026; first draft by 24 April 2026; final version by 15 May 2026 [1].
