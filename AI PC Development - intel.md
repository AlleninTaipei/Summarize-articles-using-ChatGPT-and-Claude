# AI PC Development - intel

* Advances in AI-focused hardware and software enable AI on the PC. Seamlessly transition projects from early AI development on the PC to cloud-based training to edge deployment. Learn what is required of AI workloads and what is available to get started today.

## AI PC Development

### DirectML & ONNX OpenVINO Execution Provider Example

* DirectML is Microsoft's low-level API for machine learning. [ONNX (Open Neural Network Exchange)](https://onnxruntime.ai/) is an open format for representing machine learning models. 
* OpenVINO is Intel's toolkit for optimizing deep learning models. An execution provider in this context would be a backend that can run ONNX models using OpenVINO on DirectML-compatible hardware.
* [OpenVINO™ Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

```python
import onnxruntime as ort
import numpy as np

# Set up the execution provider
providers = [
    ('OpenVINOExecutionProvider', {
        'device_type': 'GPU',
        'enable_vpu_fast_compile': 'true',
        'device_id': '0'
    }),
    ('DirectMLExecutionProvider', {
        'device_id': 0,
    })
]

# Load the ONNX model
session = ort.InferenceSession("path/to/model.onnx", providers=providers)

# Prepare input data
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
output = session.run(None, {input_name: input_data})

print(output)

```

* Business Example: An e-commerce company uses computer vision models to automatically categorize and tag product images. By implementing this execution provider setup, they can leverage both DirectML and OpenVINO to run their ONNX models efficiently on various hardware configurations, including Intel GPUs and other DirectML-compatible devices. This allows them to process a large volume of product images quickly and accurately, improving their catalog management and search functionality.

### [RAG(OpenVINO) - Phi3](https://www.intel.com/content/www/us/en/content-details/825153/demo-beyond-the-llm-rag-powered-by-phi-3-intel.html)

* RAG stands for Retrieval-Augmented Generation, a technique that enhances language models with external knowledge. In this context, it seems to be implemented using OpenVINO, possibly with the Phi3 language model (developed by Microsoft). This combination would allow for efficient RAG implementations using Intel's optimization toolkit.

* Business Example: A legal tech startup uses RAG with Phi3 optimized by OpenVINO to enhance their contract analysis tool. The system retrieves relevant clauses and precedents from a vast database of legal documents, then uses Phi3 to generate comprehensive summaries and recommendations for lawyers. OpenVINO optimization allows them to run this complex system efficiently on standard office hardware, making it accessible to small and medium-sized law firms.

### Meeting summary: Whisper + LLM

* This appears to be a summary of a meeting discussing the integration or use of two AI models: Whisper (OpenAI's automatic speech recognition system) and LLaMA 2 (Meta AI's large language model). The meeting likely covered how these models could be used together or compared.

* Business Example: A multinational corporation implements a meeting transcription and analysis system using Whisper for speech-to-text and LLaMA 2 for summarization and action item extraction. This system automatically transcribes all company meetings, generates concise summaries, and identifies key action items and decisions. It significantly improves information sharing and follow-up across different time zones and departments.

* [Automatic speech recognition using Distil-Whisper and OpenVINO](https://docs.openvino.ai/2024/notebooks/distil-whisper-asr-with-output.html)

### C++ production reference: OV Gen-AI (LCM / Whisper)

* This suggests there's a C++ implementation or reference for using OpenVINO (OV) with generative AI models. It specifically mentions LCM (likely Latent Consistency Model, a text-to-image model) and Whisper (for speech recognition). This reference probably demonstrates how to deploy these models efficiently using C++ and OpenVINO.

* Business Example: A smart home device manufacturer integrates a voice-controlled AI assistant into their products. They use a C++ implementation with OpenVINO-optimized Whisper for speech recognition and LCM for generating appropriate responses or actions. This allows them to run sophisticated AI models directly on their resource-constrained IoT devices, providing users with a responsive and intelligent home automation experience without relying on cloud processing.

* [OpenVINO Latent Consistency Model C++ pipeline with LoRA model suppor](https://blog.openvino.ai/blog-posts/lcm-cpp-pipeline-with-lora)

### AI Profiling tool (VTune)

* [VTune](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html#gs.bl5bwd) is Intel's performance profiler for optimizing software. In this AI context, it's being used as a profiling tool for AI applications. VTune can help developers analyze and improve the performance of machine learning models or AI-related code, identifying bottlenecks and optimization opportunities.

* Business Example: A financial services company uses complex AI models for real-time fraud detection in transaction processing. They use Intel's VTune to profile and optimize their AI pipeline, identifying bottlenecks in data preprocessing, model inference, and post-processing stages. By optimizing these bottlenecks, they reduce the average transaction processing time by 40%, allowing them to handle a higher volume of transactions and improve fraud detection rates without upgrading their hardware infrastructure.

### WebNN example

* [WebNN (Web Neural Network API)](https://www.intel.com/content/www/us/en/developer/topic-technology/ai-pc/webnn.html#gs.bl7mry) is a low-level API for performing hardware-accelerated neural network operations in web browsers.
* An example in this context would demonstrate how to use this API for machine learning tasks in a web environment, possibly showing how to run inference on pre-trained models directly in a browser.


```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WebNN Image Classification</title>
</head>
<body>
    <h1>WebNN Image Classification</h1>
    <input type="file" id="imageInput" accept="image/*">
    <div id="result"></div>

    <script>
        async function classifyImage(imageData) {
            const model = await navigator.ml.getNeuralNetworkContext();
            const graph = await model.createGraph();
            
            // Assume we have a pre-trained MobileNet model
            const mobilenet = await graph.loadModel('mobilenet.onnx');
            
            // Preprocess image
            const preprocessed = graph.input(imageData)
                                    .resizeBilinear([224, 224])
                                    .sub(123.68).div(58.82);
            
            // Run inference
            const output = mobilenet(preprocessed);
            
            // Get top prediction
            const topClass = await output.argmax().data();
            
            return topClass[0];
        }

        document.getElementById('imageInput').addEventListener('change', async (event) => {
            const file = event.target.files[0];
            const imageData = await createImageBitmap(file);
            const result = await classifyImage(imageData);
            document.getElementById('result').textContent = `Predicted class: ${result}`;
        });
    </script>
</body>
</html>

```

* Business Example: An online fashion retailer implements a "virtual try-on" feature using WebNN. Customers can upload photos of themselves, and the web application uses a machine learning model to overlay selected clothing items realistically. By running the AI model directly in the browser using WebNN, the retailer provides a smooth, responsive experience without the need for server-side processing, reducing latency and server costs while improving customer engagement and reducing return rates.
