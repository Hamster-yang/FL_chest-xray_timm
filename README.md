# FL_chest-xray_timm



### How to use this job
set this job in NVFlare workspace's admin/transfer folder <br>
path = {workspace}/poc/admin/transfer/chest-xary_timm

### summit job
```
# open admin mode
summit chest-xray_timm
```

### download job
```
# use admin mode
download {job_id}
```

### get accuracy
```
cat {workspace}/poc/admin/transfer/{job_id}/workspace/cross_site_val/cross_val_results.json
```

### get final_global-model_weight
```
{workspace}/poc/admin/transfer/{job_id}/workspace/app_server/FL_global_model.pt
```
