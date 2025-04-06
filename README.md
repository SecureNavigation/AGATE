# AGATE

### Install Benchmarks

`pip install benchmarks/CLIP_benchmark`

### Sample running code for zero-shot evaluation with AGATE:

### 

```bash
#zero-shot retrieval

clip_benchmark eval --model 'YOURMODEL' \
                    --pretrained 'YOURMODELVERSION' \
                    --dataset='YOURDATASET' \
                    --output=result.json \
                    --batch_size=32  \
                    --language=en \
                    --trigger_num=16 \
                    --watermark_dim=512 \
                    --dataset_root "/root" \
                    --watermark_dir "/root/watermark"                   

#zero-shot classification 

clip_benchmark eval --dataset='YOURDATASET' \
                    --pretrained='YOURMODELVERSION' \
                    --model='YOURMODEL' \
                    --output=result.json \
                    --batch_size=32 \
                    --trigger_num=16 \
                    --watermark_dim=512 \
                    --watermark_dir "/root/watermark"  \
                    --dataset_root "/root" \
```

   
