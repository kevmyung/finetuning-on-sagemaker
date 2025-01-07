# Finetuning on SageMaker

This project demonstrates fine-tuning Llama 3B model to generate JSON-formatted responses using the Databricks Dolly dataset. The project is structured into three main notebooks that cover data preparation, model fine-tuning, and deployment. All code has been validated in SageMaker Notebook environment with a cloud mode.

## Scenario

1. **Model Access**
   - Request access to [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) on HuggingFace
   - You must be granted access to the model before running the fine-tuning process
   - This requires accepting Meta's license agreement and waiting for approval

2. **Data Preparation**
   - Use Databricks Dolly dataset for training
   - Format responses into structured JSON format
   - Example format:
   ```json
   {
     "response": "answer text here",
     "metadata": {
       "has_context": true,
       "input_type": "text"
     }
   }
   ```

3. **Model Fine-tuning**
    - Fine-tune Meta's Llama 3B model using QLoRA technique
    - Train model to generate responses in JSON format
    - Validate response format during training
    - Use ml.g5.4xlarge instance for training


4. **Deployement & Testing**
    - Deploy fine-tuned model to SageMaker endpoint
    - Test model responses for:
        - JSON format compliance
        - Answer quality


## Initial Setup

1. Create `env.local` file in project root:
    ```bash
    # env.local
    HF_TOKEN=your_huggingface_token_here
    ```

2. Create and activate conda environment:
This script will create conda environment with Python 3.10.14
    ```bash
    chmod +x init_env.sh
    ./init_env.sh llama3-env
    ```


## Project Structure

```code
finetuning-on-sagemaker/
├── notebooks/
│   ├── 1-prep-data.ipynb     # Data preparation and processing
│   ├── 2-finetune.ipynb      # Model fine-tuning on SageMaker
│   └── 3-deploy.ipynb        # Model deployment and inference
├── scripts/
│   ├── requirements.txt      # Training environment dependencies
│   └── train.py             # Training script
├── data/                     # Dataset storage
├── env.local                 # Environment variables (HF token)
├── init_env.sh              # Environment setup script
└── README.md
```

## Citation
```Code
@misc{aws-genai-workshop-kr,
  title={AWS Gen AI Workshop - Llama 3 Fine-tuning},
  author={AWS Samples},
  year={2024},
  publisher={GitHub},
  url={https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/30_fine_tune/05-fine-tune-llama3/llama3-2}
}

@misc{databricks-dolly,
  title={Databricks Dolly 15k Dataset},
  author={Databricks},
  year={2023},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/databricks/databricks-dolly-15k}
}

@misc{llama3,
  title={Llama-3.2-3B},
  author={Meta AI},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/meta-llama/Llama-3.2-3B}
}
```