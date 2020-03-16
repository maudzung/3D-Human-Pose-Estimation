from torch.utils.data import DataLoader
from data_process import h36m_dataset, transformations

def H36m_dataloader(configs):
    h36m_train_dataset = h36m_dataset.H36M_Dataset(
        dataset_dir=configs.datasets_dir,
        subjects=configs.train.subjects,
        cameras=configs.train.cameras,
        act_names=configs.train.act_names,
        act_ids=configs.train.act_ids,
        protocol_name=configs.protocol_name,
        n_images=configs.n_images,
        inp_res=configs.inp_res,
        scale_range=configs.train.scale_range,
        img_transformations=transformations.get_img_transformations(configs, is_train_mode=True),
        is_train_mode=True,
        is_debug_mode=False
    )

    h36m_val_dataset = h36m_dataset.H36M_Dataset(
        dataset_dir=configs.datasets_dir,
        subjects=configs.val.subjects,
        cameras=configs.val.cameras,
        act_names=configs.val.act_names,
        act_ids=configs.val.act_ids,
        protocol_name=configs.protocol_name,
        n_images=configs.n_images,
        inp_res=configs.inp_res,
        scale_range=configs.val.scale_range,
        img_transformations=transformations.get_img_transformations(configs, is_train_mode=False),
        is_train_mode=False,
        is_debug_mode=False
    )
    h36m_train_dataloader = DataLoader(
        h36m_train_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        pin_memory=configs.pin_memory,
        num_workers=configs.n_workers,
        drop_last=configs.drop_last
    )

    h36m_val_dataloader = DataLoader(
        h36m_val_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        pin_memory=configs.pin_memory,
        num_workers=configs.n_workers,
        drop_last=configs.drop_last
    )
    return h36m_train_dataloader, h36m_val_dataloader