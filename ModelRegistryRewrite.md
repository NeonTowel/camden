# Model Registry Rewrite

Support for multiple models in the registry. Registry is used to store the models and their configurations.
Interfaces to the registry are updated to support multiple models for either moderation or tagging, or both.

Models are downloaded to the .vendor/models directory.

Models come from Hugging Face and are explicitly in ONNX format.

Models are carefully examined and their configuration for auxiliary files such as labels, tags, classifications
should be taken into consideration. Auxiliary files may be used as is or logic embedded directly to code (in
classification tiers, for example). When there are hundreds of tags, we should favor using auxiliary files
instead placed next to the model onnx files.

## Image classification models

**Phase 1:**

n4xtan/nsfw-classification

https://huggingface.co/n4xtan/nsfw-classification/blob/main/nsfw_model.onnx
https://huggingface.co/n4xtan/nsfw-classification/blob/main/global_config.json
https://huggingface.co/n4xtan/nsfw-classification/blob/main/model_config.json

spiele/nsfw_image_detector-ONNX

https://huggingface.co/spiele/nsfw_image_detector-ONNX/blob/main/config.json
https://huggingface.co/spiele/nsfw_image_detector-ONNX/blob/main/preprocessor_config.json
https://huggingface.co/spiele/nsfw_image_detector-ONNX/blob/main/quantize_config.json
https://huggingface.co/spiele/nsfw_image_detector-ONNX/blob/main/onnx/model.onnx
https://huggingface.co/spiele/nsfw_image_detector-ONNX/blob/main/onnx/model_bnb4.onnx
https://huggingface.co/spiele/nsfw_image_detector-ONNX/blob/main/onnx/model_fp16.onnx
https://huggingface.co/spiele/nsfw_image_detector-ONNX/blob/main/onnx/model_int8.onnx
https://huggingface.co/spiele/nsfw_image_detector-ONNX/blob/main/onnx/model_q4.onnx
https://huggingface.co/spiele/nsfw_image_detector-ONNX/blob/main/onnx/model_q4f16.onnx
https://huggingface.co/spiele/nsfw_image_detector-ONNX/blob/main/onnx/model_quantized.onnx
https://huggingface.co/spiele/nsfw_image_detector-ONNX/blob/main/onnx/model_uint8.onnx

vladmandic/nudenet

https://huggingface.co/vladmandic/nudenet/blob/main/nudenet.onnx

deepghs/nudenet_onnx

https://huggingface.co/deepghs/nudenet_onnx/blob/main/nms-yolov8.onnx
https://huggingface.co/deepghs/nudenet_onnx/blob/main/320n.onnx

taufiqdp/mobilenetv4_conv_small.e2400_r224_in1k_nsfw_classifier

https://huggingface.co/taufiqdp/mobilenetv4_conv_small.e2400_r224_in1k_nsfw_classifier/blob/main/mobilenetv4_conv_small.e2400_r224_in1k_nsfw_classifier.onnx
https://huggingface.co/taufiqdp/mobilenetv4_conv_small.e2400_r224_in1k_nsfw_classifier/blob/main/config.json
https://huggingface.co/taufiqdp/mobilenetv4_conv_small.e2400_r224_in1k_nsfw_classifier/blob/main/model.safetensors
https://huggingface.co/taufiqdp/mobilenetv4_conv_small.e2400_r224_in1k_nsfw_classifier/blob/main/pytorch_model.bin

**Phase 2:**

AdamCodd/vit-base-nsfw-detector
onnx-community/nsfw-classifier-ONNX
onnx-community/nsfw_image_detection-ONNX
UnfilteredAI/NSFW-gen

** Image tagging: **

**Phase 1:**

SmilingWolf/wd-v1-4-convnextv2-tagger-v2

https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2/blob/main/keras_metadata.pb
https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2/blob/main/model.onnx
https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2/blob/main/saved_model.pb
https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2/blob/main/selected_tags.csv
https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2/blob/main/variables/variables.data-00000-of-00001
https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2/blob/main/variables/variables.index

fancyfeast/joytag

https://huggingface.co/fancyfeast/joytag/blob/main/config.json
https://huggingface.co/fancyfeast/joytag/blob/main/model.onnx
https://huggingface.co/fancyfeast/joytag/blob/main/model.safetensors
https://huggingface.co/fancyfeast/joytag/blob/main/top_tags.txt

deepghs/nudenet_onnx

https://huggingface.co/deepghs/nudenet_onnx/blob/main/320n.onnx
https://huggingface.co/deepghs/nudenet_onnx/blob/main/nms-yolov8.onnx

cella110n/cl_tagger
https://huggingface.co/cella110n/cl_tagger/blob/main/cl_tagger_1_02/model.onnx
https://huggingface.co/cella110n/cl_tagger/blob/main/cl_tagger_1_02/model_optimized.onnx
https://huggingface.co/cella110n/cl_tagger/blob/main/cl_tagger_1_02/tag_mapping.json

**Phase 2:**

trpakov/vit-face-expression
SmilingWolf/wd-vit-tagger-v3
SmilingWolf/wd-vit-large-tagger-v3

## Ensemble

Models are used together and their output is considered equally. Moderation and tagging pipelines may use one or more models running in parallel.

Favor more extensive and accurate models. Design a default set to be used in both pipelines using three of the best models in each.
