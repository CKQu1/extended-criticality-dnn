import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset 
from torchvision import datasets
import torchvision.transforms as transforms

def transform_cifar10(image):   # flattening cifar10
    return (torch.Tensor(image.getdata()).T.reshape(-1)/255)*2 - 1

def set_data(name, rshape: bool, **kwargs):
    data_path = "data"
    if rshape:        
        if name.lower() == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,)),
                                            transforms.Lambda(lambda x: torch.flatten(x))]
                                            )
            #transform = transform_cifar10
            train_ds = datasets.MNIST(root=data_path, download=True, transform=transform)
            valid_ds = datasets.MNIST(root=data_path, download=True, transform=transform, train=False)
        elif name.lower() == "gaussian":
            from UTILS.generate_gaussian_data import delayed_mixed_gaussian
            
            # same setting as MNIST
            num_train, num_test = kwargs.get("num_train"), kwargs.get("num_test")
            X_dim = kwargs.get("X_dim")
            Y_classes, X_clusters = kwargs.get("Y_classes"), kwargs.get("X_clusters")
            n_hold, final_time_point = kwargs.get("n_hold"), kwargs.get("final_time_point")    # not needed for feedforwards nets, always set to 0 for MLPs
            noise_sigma = kwargs.get("noise_sigma")
            cluster_seed, assignment_and_noise_seed = kwargs.get("cluster_seed"), kwargs.get("assignment_and_noise_seed")
            avg_magn = kwargs.get("avg_magn")
            min_sep = kwargs.get("min_sep")
            freeze_input = kwargs.get("freeze_input")

            class_datasets, centers, cluster_class_label = delayed_mixed_gaussian(num_train, num_test, X_dim, Y_classes, X_clusters, n_hold,
                                                                                  final_time_point, noise_sigma,
                                                                                  cluster_seed, assignment_and_noise_seed, avg_magn,                                        
                                                                                  min_sep, freeze_input)
            train_ds = class_datasets['train']
            valid_ds = class_datasets['val']        

        elif name.lower() == "fashionmnist":
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)), 
                                            transforms.Lambda(lambda x: torch.flatten(x))]
                                            )
            train_ds = datasets.FashionMNIST(root=data_path, download=True, train=True, transform=transform)
            valid_ds = datasets.FashionMNIST(root=data_path, download=True, train=False, transform=transform)
        elif name.lower() == 'cifar10':
            transform=transform_cifar10
            train_ds = datasets.CIFAR10(root=data_path, download=True, transform=transform)
            valid_ds = datasets.CIFAR10(root=data_path, download=True, transform=transform, train=False)
        else:
            raise NameError("name is not defined in function set_data")

    else:
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        """
        transform_train = transforms.Compose([transforms.Resize((70, 70)),
                                               transforms.RandomCrop((64, 64)),
                                               transforms.ToTensor(),
                                               normalize,])

        transform_test = transforms.Compose([transforms.Resize((70, 70)),
                                              transforms.CenterCrop((64, 64)),
                                              transforms.ToTensor(),
                                              normalize,])
        """

        if name == 'mnist':
            """
            train_ds = torchvision.datasets.MNIST('mnist', download=True, transform=transform_train)
            valid_ds = torchvision.datasets.MNIST('mnist', download=True, transform=transform_test, train=False)
            """
            normalize = transforms.Normalize((0.1307,), (0.3081,))
            train_ds = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize,]))
            valid_ds = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize,]))

        elif name == 'cifar100':
            #mean and std of cifar100 dataset
            CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

            #CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
            #CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

            transform_train = transforms.Compose([
                #transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
            ])
            #cifar100_training = CIFAR100Train(path, transform=transform_train)
            train_ds = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
            ])
            #cifar100_test = CIFAR100Test(path, transform=transform_test)
            valid_ds = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)


        elif name == 'cifar10':
            if "cifar_upsize" in kwargs and kwargs.get("cifar_upsize") == True:     # for AlexNet (torch version)
                upsize_cifar10 = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                train_ds = torchvision.datasets.CIFAR10(root=join(data_path, "cifar10_upsize"), download=True, transform=upsize_cifar10)
                valid_ds = torchvision.datasets.CIFAR10(root=join(data_path, "cifar10_upsize"), download=True, transform=upsize_cifar10, train=False)
            else:
                train_ds = torchvision.datasets.CIFAR10(root=data_path, download=True, transform=transform_train)
                valid_ds = torchvision.datasets.CIFAR10(root=data_path, download=True, transform=transform_test, train=False)

        elif name == 'cifar10circ':
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.2023, 0.1994, 0.2010])
            
            train_ds = torchvision.datasets.CIFAR10(root=join(data_path,"cifar10_circ"), train=True, download=True,
                                                    transform=transforms.Compose([
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    normalize,
                                                    ]))
            
            valid_ds = torchvision.datasets.CIFAR10(root=join(data_path,"cifar10_circ"), train=False, download=True,
                                                    transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    normalize,
                                                    ]))      

        else:
            raise NameError("name is not defined in function set_data")
    
    if name.lower() == "gaussian":
        return train_ds, valid_ds, centers, cluster_class_label
    else:
        return train_ds, valid_ds

def get_data(train_ds, valid_ds, bs, **kwargs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs * 2, **kwargs),
    )    