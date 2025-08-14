from torchvision import models

from pretrained_download import get_pretrained_names, compare_models

model_names = get_pretrained_names()

for n_model in range(len(model_names)):
    model_name = model_names[int(n_model)]
    model = models.__dict__[model_name](pretrained=True)

    for s in dir(models):
        if model_name in s.lower() and 'weights' in s.lower():
            model_weights = models.__dict__[s]
            for dataset_name in dir(model_weights):
                if 'imagenet' in dataset_name.lower():
                    model_v = models.__dict__[model_name](weights=model_weights[dataset_name])
                    if compare_models(model, model_v):
                        print(f"{model_name} pretrained version is trained on {dataset_name}")
                    #else:
                    #    print(f"{model_name} pretrained version is trained on {dataset_name}")

            break
