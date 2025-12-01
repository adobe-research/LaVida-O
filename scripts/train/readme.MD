## Data Preparation

The training data are configured with a yaml file in the `config` folder. Each data is defined has the following key fields:

`json_path`: the path to actual data record files.  Note that we typically use csv or parquet for efficiency. The field name is kept as `json_path` for compatibility reasons

`sampling_strategy`: How dataset is sampled. By default, we use `full`, which means all data are used. Other options are `dup:2` (duplicate data 2x), `random:200000`, random sample 200k, etc.

`preprocess_fn`: This is the most important element in the dataset configuration. It takes columns of csv/parquet file and convert them to LLaVa-Style conversation used in training.  They are defined in [llava/train/data/process_functions.py](/llava/train/data/process_functions.py)

## An Example

On our infrastructure, we host all images on AWS s3, and use parquet files to document them. A typical data pipeline for text to image generation would be 


1. prepare a csv with columns: `s3_path`, `caption`,`fltLaionAesthScore` (score used for filtering),`intHeight`,`intWidth`
2. add the following entry to yaml
```
  - name: our-dataset
    json_path: /path/to/parquet/data.parquet
    sampling_strategy: all
    preprocess_fn: preproces_text_to_image_generation_s3
    columns: ['s3_path', 'caption', 'fltLaionAesthScore','intHeight','intWidth']
    aes_cutoff: 5.67
    min_size: 512
```
3. During the training, `preproces_text_to_image_generation_s3` is called to convert the data to the following LLaVa format

```
    payload = {
        "id": "000951660",
        "image_gen": img_path,
        "conversations": [
        {
            "from": "human",
            "value": f"Generate an image with the caption: {caption}"
        },
        {
            "from": "gpt",
            "value": "Sure <image_gen>"
        }
        ]
    }
```


### List of datasets used

We provide links to all the datasets that are used. We are unable to provide the exact parquet files because the they include s3 urls on our private bucket. Please process each data following the above guidelines. Feel free to email us if you encounter any questions. We note that LaViDa-O is trained **entirely on public data**.

#### Text-to-Image Generation, Image Editing

[LAION-2B](https://huggingface.co/datasets/laion/relaion2B-en-research-safe)

[COYO-700M](https://huggingface.co/datasets/kakaobrain/coyo-700m)

[BLIP-3o](https://huggingface.co/BLIP3o/datasets)

[Reflect-DiT](jacklishufan/reflect-DiT)

[ShareGPT-4o](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image)

[Echo-4o-Image](https://huggingface.co/datasets/Yejy53/Echo-4o-Image)

[GPT-Image-Edit-1.5M](https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M)

#### Understanding and Grounding

[MAmmoTH-VL](https://huggingface.co/datasets/MAmmoTH-VL/MAmmoTH-VL-Instruct-12M)

[GranD](https://huggingface.co/datasets/MBZUAI/GranD-f)

[VisualWebInstruct](https://huggingface.co/datasets/TIGER-Lab/VisualWebInstruct)



## Training

The following scripts are used to train the model

```
scripts/train/s1-gnd.sh
scripts/train/s2-256.sh
scripts/train/s2-1024.sh
scripts/train/s3-unified.sh
```

Before launch training scripte, please make sure to 

1. Add your huggingface token in environment variable
2. Change the batch size, number of gpus, number of nodes,max steps etc according to your infrastructure
